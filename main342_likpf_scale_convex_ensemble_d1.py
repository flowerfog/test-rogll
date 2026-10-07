"""M342-D1: deterministic cross-scale likPF convex ensemble gateway.

Unlike ``likpf_mean`` (an unweighted mean over PF seeds), this combines the
four likelihood-temperature endpoints. Convex weights are selected on one
complete-well fold and evaluated on the other for Folds 4/5. Only when both
directions improve is one Fold4+5 weight vector evaluated on Fold3. Fold1 and
Fold2 are never loaded.
"""

from pathlib import Path
import itertools
import json
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
CACHE_DIR = OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1"
FEATURE_DIR = CACHE_DIR / "features"
DONOR_FEATURE_DIR = (
    OOF_ROOT / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75" / "features"
)
V168_PATH = OOF_ROOT / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
OUT_DIR = OOF_ROOT / "main342_likpf_scale_convex_ensemble_d1"

SCALES = (
    "likpf_scale_3", "likpf_scale_5", "likpf_scale_8", "likpf_scale_12",
)
DISCOVERY_FOLDS = (4, 5)
CONFIRMATION_FOLD = 3
WEIGHT_STEP = 0.05


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _sample_path(well):
    path = FEATURE_DIR / f"{well}__official__sample.pkl"
    if path.is_file():
        return path
    donor = DONOR_FEATURE_DIR / f"{well}__official__sample.pkl"
    return donor if donor.is_file() else path


def _weights_grid():
    units = int(round(1.0 / WEIGHT_STEP))
    rows = []
    for values in itertools.product(range(units + 1), repeat=len(SCALES) - 1):
        used = sum(values)
        if used > units:
            continue
        rows.append((*values, units - used))
    return np.asarray(rows, np.float64) / units


def _load_fold(wells, fold):
    parts = []
    for number, well in enumerate(wells, 1):
        path = _sample_path(well)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_pickle(path)
        missing = [column for column in SCALES if column not in frame]
        if missing:
            raise RuntimeError(f"{well}: missing {missing}")
        last = frame["last_known_tvt"].to_numpy(np.float64)
        truth = last + frame["target"].to_numpy(np.float64)
        prediction = frame[list(SCALES)].to_numpy(np.float64)
        prediction = np.where(np.isfinite(prediction), prediction, last[:, None])
        parts.append(pd.DataFrame({
            "well": str(well),
            "fold": int(fold),
            "truth": truth,
            **{name: prediction[:, index] for index, name in enumerate(SCALES)},
        }))
        if number % 50 == 0 or number == len(wells):
            print(f"M342-D1 load fold{fold} {number}/{len(wells)}", flush=True)
    return pd.concat(parts, ignore_index=True)


def _fit_weights(frame, grid):
    prediction = frame[list(SCALES)].to_numpy(np.float64)
    truth = frame["truth"].to_numpy(np.float64)
    # Evaluate every simplex point from sufficient statistics, avoiding a
    # large rows x configurations temporary matrix.
    gram = prediction.T @ prediction
    cross = prediction.T @ truth
    truth_ss = float(np.dot(truth, truth))
    sse = np.einsum("bi,ij,bj->b", grid, gram, grid) - 2.0 * (grid @ cross) + truth_ss
    best = int(np.argmin(sse))
    return grid[best].copy(), float(np.sqrt(sse[best] / len(frame)))


def _single_scan(frame):
    truth = frame["truth"].to_numpy(np.float64)
    rows = [
        {"candidate": name, "rmse": float(np.sqrt(np.mean((frame[name] - truth) ** 2)))}
        for name in SCALES
    ]
    return pd.DataFrame(rows).sort_values(["rmse", "candidate"]).reset_index(drop=True)


def _evaluate(frame, weights, donor_single):
    truth = frame["truth"].to_numpy(np.float64)
    matrix = frame[list(SCALES)].to_numpy(np.float64)
    blend = matrix @ weights
    equal = matrix.mean(axis=1)
    fixed = frame[donor_single].to_numpy(np.float64)
    local = pd.DataFrame({
        "well": frame["well"].astype(str),
        "blend_e2": np.square(blend - truth),
        "fixed_e2": np.square(fixed - truth),
        "equal_e2": np.square(equal - truth),
    }).groupby("well", sort=False).mean()
    blend_rmse = float(np.sqrt(np.mean(np.square(blend - truth))))
    fixed_rmse = float(np.sqrt(np.mean(np.square(fixed - truth))))
    equal_rmse = float(np.sqrt(np.mean(np.square(equal - truth))))
    return {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "donor_single": donor_single,
        "fixed_rmse": fixed_rmse,
        "equal_rmse": equal_rmse,
        "blend_rmse": blend_rmse,
        "gain_vs_fixed": fixed_rmse - blend_rmse,
        "gain_vs_equal": equal_rmse - blend_rmse,
        "positive_well_fraction_vs_fixed": float(
            np.mean(local["blend_e2"] < local["fixed_e2"])
        ),
        "p95_change_vs_fixed": float(
            np.sqrt(local["blend_e2"]).quantile(0.95)
            - np.sqrt(local["fixed_e2"]).quantile(0.95)
        ),
    }


started = time.time()
for required in (FEATURE_DIR, V168_PATH):
    if not required.exists():
        raise FileNotFoundError(required)
OUT_DIR.mkdir(parents=True, exist_ok=True)

base = pd.read_parquet(V168_PATH, columns=["well", "fold"])
base["well"] = base["well"].astype(str)
fold_map = base.groupby("well")["fold"].first().astype(int)
coverage = fold_map.value_counts().sort_index().astype(int).to_dict()
if coverage.get(3, 0) < 145 or coverage.get(4, 0) < 145 or coverage.get(5, 0) < 145:
    raise RuntimeError(f"Insufficient D1 coverage: {coverage}")

frames = {}
for fold in (3, 4, 5):
    wells = sorted(fold_map[fold_map.eq(fold)].index)
    frames[fold] = _load_fold(wells, fold)
grid = _weights_grid()

discovery = []
discovery_weights = {}
for held_fold in DISCOVERY_FOLDS:
    donor_fold = 5 if held_fold == 4 else 4
    weights, donor_blend_rmse = _fit_weights(frames[donor_fold], grid)
    donor_single_scan = _single_scan(frames[donor_fold])
    donor_single = str(donor_single_scan.iloc[0]["candidate"])
    report = _evaluate(frames[held_fold], weights, donor_single)
    discovery_weights[held_fold] = weights
    discovery.append({
        "held_fold": held_fold,
        "donor_fold": donor_fold,
        "donor_blend_rmse": donor_blend_rmse,
        **{f"weight_{name}": float(weight) for name, weight in zip(SCALES, weights)},
        **report,
    })

discovery_frame = pd.DataFrame(discovery)
weight_l1 = float(np.sum(np.abs(discovery_weights[4] - discovery_weights[5])))
discovery_pass = bool(
    discovery_frame["gain_vs_fixed"].min() >= 0.03
    and discovery_frame["gain_vs_equal"].min() >= 0.0
    and discovery_frame["p95_change_vs_fixed"].max() <= 0.50
    and weight_l1 <= 0.80
)

confirmation = None
combined_weights = None
if discovery_pass:
    donor = pd.concat([frames[4], frames[5]], ignore_index=True)
    combined_weights, combined_fit_rmse = _fit_weights(donor, grid)
    donor_single = str(_single_scan(donor).iloc[0]["candidate"])
    confirmation = {
        "fold": CONFIRMATION_FOLD,
        "donor_folds": list(DISCOVERY_FOLDS),
        "donor_blend_rmse": combined_fit_rmse,
        **{f"weight_{name}": float(weight) for name, weight in zip(SCALES, combined_weights)},
        **_evaluate(frames[CONFIRMATION_FOLD], combined_weights, donor_single),
    }

confirmation_pass = bool(
    confirmation is not None
    and confirmation["gain_vs_fixed"] >= 0.03
    and confirmation["gain_vs_equal"] >= 0.0
    and confirmation["p95_change_vs_fixed"] <= 0.50
)
strict_pass = bool(discovery_pass and confirmation_pass)
verdict = {
    "experiment": "main342_likpf_scale_convex_ensemble_d1",
    "stage": "crossfold_sampled_deterministic_likpf_scale_ensemble",
    "candidate_scales": list(SCALES),
    "weight_step": WEIGHT_STEP,
    "configurations": int(len(grid)),
    "coverage": coverage,
    "fold1_touched": False,
    "fold2_touched": False,
    "discovery": discovery,
    "discovery_weight_l1_distance": weight_l1,
    "discovery_pass": discovery_pass,
    "combined_weights": combined_weights,
    "confirmation": confirmation,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> reconstruct the frozen scale ensemble on full Fold3 rows and test V168 blend"
        if strict_pass else
        "FAIL -> fixed scale ensembling is not stable; retain the existing likPF endpoint"
    ),
}
discovery_frame.to_csv(OUT_DIR / "discovery.csv", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("M342-D1 DISCOVERY")
print(discovery_frame.to_string(index=False))
print("\nM342-D1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR)

