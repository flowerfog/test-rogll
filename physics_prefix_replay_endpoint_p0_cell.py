"""Zero-ML physical endpoint selected by visible-prefix walk-forward replay.

The endpoint predicts the structural surface S = TVT + Z from trajectory
geometry and the visible TVT_input prefix.  It deliberately consumes no GR,
neighbour, CP8, or historical model prediction.  The exact 6.377 lineage is
loaded only after the physical endpoint is frozen, for correlation and blend
audits.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "physics_prefix_replay_endpoint_p0"
SEED = 20260727
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DISCOVERY_WELLS = 24

WINDOWS_FT = (256.0, 512.0, 1024.0, 2048.0, math.inf)
SLOPE_DOSES = (0.25, 0.5, 1.0)
REPLAY_BACKOFFS_FT = (128.0, 256.0, 384.0, 512.0, 768.0, 1024.0)
MIN_TRAIN_ROWS = 64
MIN_REPLAY_ROWS = 48
SURFACE_SLOPE_CAP = 0.08
BASE_BLEND_GRID = np.arange(0.0, 0.5001, 0.025)


def find_data_root() -> Path:
    candidates = [
        Path.cwd(),
        Path("/root/autodl-tmp"),
        Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    ]
    for candidate in candidates:
        train = candidate / "train"
        if train.is_dir() and next(train.glob("*__horizontal_well.csv"), None):
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate original competition train directory")


DATA_ROOT = find_data_root()
ROOT = Path("/root/autodl-tmp") if os.name != "nt" else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT if os.name != "nt" else Path.cwd() / "output" / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
BASE_PATH = (
    OOF_ROOT
    / "frontier_6p377_cp8_neighbor_dose_replay_v33"
    / "cp8_neighbor_replay_oof.parquet"
)
started = time.time()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def stable_folds(wells):
    scored = []
    for well in wells:
        digest = hashlib.sha256(f"{SEED}:{well}".encode("ascii")).digest()
        scored.append((int.from_bytes(digest[:8], "little"), well))
    ordered = [well for _, well in sorted(scored)]
    output = {}
    for fold, indices in enumerate(np.array_split(np.arange(len(ordered)), 5), 1):
        for index in indices:
            output[ordered[int(index)]] = fold
    return output


def load_fold_map(wells):
    if FOLD_PATH.is_file():
        metadata = pd.read_csv(FOLD_PATH)
        well_column = next((c for c in ("well", "well_id", "pair_id") if c in metadata), None)
        fold_column = next((c for c in ("fold", "outer_fold") if c in metadata), None)
        if well_column and fold_column:
            mapping = dict(
                zip(metadata[well_column].astype(str), metadata[fold_column].astype(int))
            )
            missing = sorted(set(wells).difference(mapping))
            if not missing:
                return {well: int(mapping[well]) for well in wells}, str(FOLD_PATH)
    return stable_folds(wells), "stable_sha256_fallback"


def fixed_anchor_slope(x, y, anchor_x, anchor_y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid] - float(anchor_x)
    y = y[valid] - float(anchor_y)
    if len(x) < 8 or np.dot(x, x) <= 1e-9:
        return 0.0
    weights = np.ones(len(x), np.float64)
    slope = 0.0
    for _ in range(6):
        denominator = float(np.dot(weights * x, x))
        if denominator <= 1e-9:
            break
        slope = float(np.dot(weights * x, y) / denominator)
        residual = y - slope * x
        centre = float(np.median(residual))
        scale = max(1.4826 * float(np.median(np.abs(residual - centre))), 0.25)
        standardized = np.abs(residual - centre) / (1.5 * scale)
        weights = np.ones(len(standardized), np.float64)
        np.divide(
            1.0,
            standardized,
            out=weights,
            where=standardized > 1.0,
        )
    return float(np.clip(slope, -SURFACE_SLOPE_CAP, SURFACE_SLOPE_CAP))


def candidate_names():
    names = ["tvt_flat", "surface_flat"]
    for window in WINDOWS_FT:
        label = "all" if not np.isfinite(window) else str(int(window))
        for dose in SLOPE_DOSES:
            names.append(f"surface_w{label}_d{dose:g}")
    return names


CANDIDATE_NAMES = candidate_names()


def candidate_paths(md, z, tvt_visible, cut, query):
    anchor_md = float(md[cut])
    anchor_tvt = float(tvt_visible[cut])
    anchor_surface = anchor_tvt + float(z[cut])
    distance = md[query] - anchor_md
    result = {
        "tvt_flat": np.full(len(query), anchor_tvt, np.float64),
        "surface_flat": anchor_surface - z[query],
    }
    known = np.flatnonzero(np.isfinite(tvt_visible) & (np.arange(len(md)) <= cut))
    surface = tvt_visible + z
    for window in WINDOWS_FT:
        label = "all" if not np.isfinite(window) else str(int(window))
        selected = known if not np.isfinite(window) else known[md[known] >= anchor_md - window]
        if len(selected) < MIN_TRAIN_ROWS:
            selected = known[-MIN_TRAIN_ROWS:]
        slope = fixed_anchor_slope(
            md[selected], surface[selected], anchor_md, anchor_surface
        )
        for dose in SLOPE_DOSES:
            result[f"surface_w{label}_d{dose:g}"] = (
                anchor_surface + float(dose) * slope * distance - z[query]
            )
    return result


def replay_losses(md, z, tvt_visible, anchor):
    anchor_md = float(md[anchor])
    per_cut = {name: [] for name in CANDIDATE_NAMES}
    payloads = 0
    for backoff in REPLAY_BACKOFFS_FT:
        eligible = np.flatnonzero(
            np.isfinite(tvt_visible)
            & np.isfinite(md)
            & np.isfinite(z)
            & (md <= anchor_md - backoff)
        )
        if len(eligible) < MIN_TRAIN_ROWS:
            continue
        cut = int(eligible[-1])
        query = np.flatnonzero(
            np.isfinite(tvt_visible)
            & np.isfinite(md)
            & np.isfinite(z)
            & (np.arange(len(md)) > cut)
            & (np.arange(len(md)) <= anchor)
        )
        if len(query) < MIN_REPLAY_ROWS:
            continue
        paths = candidate_paths(md, z, tvt_visible, cut, query)
        for name in CANDIDATE_NAMES:
            error = paths[name] - tvt_visible[query]
            per_cut[name].append(float(np.sqrt(np.mean(error * error))))
        payloads += 1
    summary = {}
    for name, values in per_cut.items():
        if values:
            array = np.asarray(values, np.float64)
            summary[name] = {
                "mean": float(np.mean(array)),
                "max": float(np.max(array)),
                "p75": float(np.quantile(array, 0.75)),
            }
    return summary, payloads


SELECTOR_CONFIGS = []
for penalty in (0.0, 0.25, 0.5):
    SELECTOR_CONFIGS.append({"id": f"hard_p{penalty:g}", "penalty": penalty, "tau": 0.0})
    for tau in (0.5, 1.0, 2.0):
        SELECTOR_CONFIGS.append(
            {"id": f"soft_p{penalty:g}_t{tau:g}", "penalty": penalty, "tau": tau}
        )


def selector_weights(losses, config):
    names = [name for name in CANDIDATE_NAMES if name in losses]
    if not names:
        names = ["surface_flat"]
        return names, np.array([1.0], np.float64)
    score = np.array(
        [
            losses[name]["mean"]
            + float(config["penalty"]) * (losses[name]["max"] - losses[name]["mean"])
            for name in names
        ],
        np.float64,
    )
    if float(config["tau"]) <= 0:
        weights = np.zeros(len(names), np.float64)
        weights[int(np.argmin(score))] = 1.0
    else:
        weights = np.exp(-(score - score.min()) / float(config["tau"]))
        weights /= weights.sum()
    return names, weights


def metrics(frame, prediction, label, mask=None):
    local = frame if mask is None else frame.loc[np.asarray(mask, bool)]
    pred = np.asarray(prediction if mask is None else prediction[np.asarray(mask, bool)], np.float64)
    truth = local["tvt_true"].to_numpy(np.float64)
    error = pred - truth
    well_sse = pd.DataFrame({"well": local["well"].to_numpy(), "e2": error * error}).groupby("well")["e2"].mean()
    distance = local["md_since"].to_numpy(np.float64)
    far = distance > 2000.0
    return {
        "endpoint": label,
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(well_sse))),
        "p95_well_rmse": float(np.quantile(np.sqrt(well_sse), 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else None,
    }


files = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
wells = [path.name.split("__")[0] for path in files]
fold_by_well, fold_source = load_fold_map(wells)
print(
    "PHYSICS-P0 CONFIG",
    {
        "data_root": str(DATA_ROOT),
        "wells": len(wells),
        "fold_source": fold_source,
        "candidates": len(CANDIDATE_NAMES),
        "selectors": len(SELECTOR_CONFIGS),
    },
    flush=True,
)

rows = []
well_payload = {}
failures = []
for number, path in enumerate(files, 1):
    well = path.name.split("__")[0]
    try:
        frame = pd.read_csv(path, usecols=["MD", "Z", "TVT", "TVT_input"])
        md = pd.to_numeric(frame["MD"], errors="coerce").to_numpy(np.float64)
        z = pd.to_numeric(frame["Z"], errors="coerce").to_numpy(np.float64)
        truth = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(np.float64)
        tvt_visible = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(np.float64)
        known = np.flatnonzero(np.isfinite(tvt_visible) & np.isfinite(md) & np.isfinite(z))
        evaluation = np.flatnonzero(~np.isfinite(tvt_visible) & np.isfinite(truth) & np.isfinite(md) & np.isfinite(z))
        if len(known) < MIN_TRAIN_ROWS + MIN_REPLAY_ROWS or len(evaluation) < 100:
            raise RuntimeError(f"known/evaluation={len(known)}/{len(evaluation)}")
        anchor = int(known[-1])
        paths = candidate_paths(md, z, tvt_visible, anchor, evaluation)
        losses, replay_payloads = replay_losses(md, z, tvt_visible, anchor)
        config_predictions = {}
        config_weights = {}
        for config in SELECTOR_CONFIGS:
            names, weights = selector_weights(losses, config)
            config_predictions[config["id"]] = np.column_stack([paths[name] for name in names]) @ weights
            config_weights[config["id"]] = dict(zip(names, weights.tolist()))
        md_since = md[evaluation] - md[anchor]
        local = pd.DataFrame(
            {
                "id": [f"{well}_{int(index)}" for index in evaluation],
                "well": well,
                "row_idx": evaluation.astype(np.int32),
                "fold": int(fold_by_well[well]),
                "tvt_true": truth[evaluation].astype(np.float32),
                "md_since": md_since.astype(np.float32),
            }
        )
        for name, prediction in paths.items():
            local[f"candidate__{name}"] = prediction.astype(np.float32)
        for name, prediction in config_predictions.items():
            local[f"selector__{name}"] = prediction.astype(np.float32)
        rows.append(local)
        well_payload[well] = {
            "fold": int(fold_by_well[well]),
            "replay_payloads": int(replay_payloads),
            "losses": losses,
            "weights": config_weights,
        }
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(files):
        print(f"PHYSICS-P0 wells {number}/{len(files)} success={len(rows)} failures={len(failures)}", flush=True)

if len(rows) < 760:
    raise RuntimeError(f"Only {len(rows)} usable wells: {failures[:10]}")
oof = pd.concat(rows, ignore_index=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)


def subset_mask(fold, discovery=False):
    well_set = sorted(oof.loc[oof["fold"] == fold, "well"].unique())
    if fold == SELECTION_FOLD:
        scored = sorted(
            (
                int.from_bytes(hashlib.sha256(f"discovery:{well}".encode()).digest()[:8], "little"),
                well,
            )
            for well in well_set
        )
        discovery_set = {well for _, well in scored[:DISCOVERY_WELLS]}
        return (oof["fold"].eq(fold) & oof["well"].isin(discovery_set if discovery else set(well_set) - discovery_set)).to_numpy()
    return oof["fold"].eq(fold).to_numpy()


discovery_mask = subset_mask(SELECTION_FOLD, True)
confirmation_mask = subset_mask(SELECTION_FOLD, False)
application_mask = subset_mask(APPLICATION_FOLD)

selector_scan = []
for config in SELECTOR_CONFIGS:
    column = f"selector__{config['id']}"
    prediction = oof[column].to_numpy(np.float64)
    result = metrics(oof, prediction, config["id"], discovery_mask)
    selector_scan.append({**config, **result})
selector_scan = pd.DataFrame(selector_scan).sort_values("rmse", kind="stable")
selected_config = selector_scan.iloc[0].to_dict()
selected_id = str(selected_config["id"])
selected_column = f"selector__{selected_id}"
physical = oof[selected_column].to_numpy(np.float64)

candidate_columns = [f"candidate__{name}" for name in CANDIDATE_NAMES]
candidate_matrix = oof[candidate_columns].to_numpy(np.float64)
truth_all = oof["tvt_true"].to_numpy(np.float64)
well_indices = oof.groupby("well", sort=False).indices
oracle = np.empty(len(oof), np.float64)
oracle_choice = {}
for well, positions in well_indices.items():
    positions = np.asarray(positions, np.int64)
    errors = candidate_matrix[positions] - truth_all[positions, None]
    choice = int(np.argmin(np.mean(errors * errors, axis=0)))
    oracle[positions] = candidate_matrix[positions, choice]
    oracle_choice[well] = CANDIDATE_NAMES[choice]

results = {
    "discovery": metrics(oof, physical, "physical", discovery_mask),
    "confirmation": metrics(oof, physical, "physical", confirmation_mask),
    "application": metrics(oof, physical, "physical", application_mask),
    "confirmation_oracle": metrics(oof, oracle, "physical_family_oracle", confirmation_mask),
    "application_oracle": metrics(oof, oracle, "physical_family_oracle", application_mask),
}
print("\nPHYSICS-P0 SELECTOR SCAN")
print(selector_scan[["id", "rmse", "far_gt2000_rmse"]].to_string(index=False))
print("PHYSICS-P0 SELECTED", selected_id, flush=True)

# Freeze the physical endpoint before loading any historical prediction.
audit = None
if BASE_PATH.is_file():
    base_frame = pd.read_parquet(BASE_PATH)
    required = {"id", "well", "fold", "tvt_true", "md_since", "current_dose_0p20"}
    missing = sorted(required.difference(base_frame.columns))
    if missing:
        raise RuntimeError(f"6.377 OOF lacks columns: {missing}")
    base_frame = base_frame[list(required)].copy()
    base_frame["id"] = base_frame["id"].astype(str)
    aligned = base_frame.merge(
        oof[["id", selected_column]], on="id", how="inner", validate="one_to_one"
    )
    if len(aligned) < 2_200_000:
        raise RuntimeError(f"Only {len(aligned)} rows align with 6.377 OOF")
    base = aligned["current_dose_0p20"].to_numpy(np.float64)
    phys = aligned[selected_column].to_numpy(np.float64)
    truth = aligned["tvt_true"].to_numpy(np.float64)
    error_base = base - truth
    error_phys = phys - truth
    raw_rho = float(np.corrcoef(error_base, error_phys)[0, 1])
    per_well_base = error_base.copy()
    per_well_phys = error_phys.copy()
    for positions in aligned.groupby("well", sort=False).indices.values():
        positions = np.asarray(positions, np.int64)
        per_well_base[positions] -= np.mean(per_well_base[positions])
        per_well_phys[positions] -= np.mean(per_well_phys[positions])
    centred_rho = float(np.corrcoef(per_well_base, per_well_phys)[0, 1])
    distance_bins = [0.0, 500.0, 1000.0, 2000.0, np.inf]
    bin_rho = {}
    distance = aligned["md_since"].to_numpy(np.float64)
    for low, high in zip(distance_bins[:-1], distance_bins[1:]):
        mask = (distance > low) & (distance <= high)
        label = f"{int(low)}_{'inf' if not np.isfinite(high) else int(high)}"
        bin_rho[label] = float(np.corrcoef(error_base[mask], error_phys[mask])[0, 1])

    selection = aligned["fold"].eq(SELECTION_FOLD).to_numpy()
    application = aligned["fold"].eq(APPLICATION_FOLD).to_numpy()
    blend_scan = []
    for weight in BASE_BLEND_GRID:
        prediction = (1.0 - weight) * base + weight * phys
        selection_rmse = float(np.sqrt(np.mean((prediction[selection] - truth[selection]) ** 2)))
        blend_scan.append({"physics_weight": float(weight), "selection_rmse": selection_rmse})
    blend_scan = pd.DataFrame(blend_scan).sort_values("selection_rmse", kind="stable")
    selected_weight = float(blend_scan.iloc[0]["physics_weight"])
    blend = (1.0 - selected_weight) * base + selected_weight * phys
    base_result = metrics(aligned, base, "base_6p377")
    physics_result = metrics(aligned, phys, "physical")
    blend_result = metrics(aligned, blend, "fold3_weight_blend")
    base_app = metrics(aligned, base, "base_6p377", application)
    blend_app = metrics(aligned, blend, "fold3_weight_blend", application)
    audit = {
        "aligned_rows": int(len(aligned)),
        "aligned_wells": int(aligned["well"].nunique()),
        "raw_signed_error_rho": raw_rho,
        "within_well_centred_error_rho": centred_rho,
        "distance_bin_error_rho": bin_rho,
        "selected_physics_weight_on_fold3": selected_weight,
        "base": base_result,
        "physics": physics_result,
        "blend": blend_result,
        "fold2_base": base_app,
        "fold2_blend": blend_app,
        "fold2_blend_gain": base_app["rmse"] - blend_app["rmse"],
    }
    blend_scan.to_csv(OUT_DIR / "fold3_blend_scan.csv", index=False)
else:
    print(f"WARNING: exact 6.377 OOF not found at {BASE_PATH}; skipping blend audit", flush=True)

output = oof[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["physics_prefix_replay"] = physical.astype(np.float32)
output["physics_family_oracle"] = oracle.astype(np.float32)
try:
    output.to_parquet(
        OUT_DIR / "physics_prefix_replay_oof.parquet", index=False, compression="zstd"
    )
except ImportError:
    # Local Windows smoke environments may not include pyarrow.  AutoDL does.
    output.to_pickle(OUT_DIR / "physics_prefix_replay_oof.pkl")
selector_scan.to_csv(OUT_DIR / "selector_discovery_scan.csv", index=False)
pd.DataFrame(
    [{"well": well, "oracle_choice": choice} for well, choice in oracle_choice.items()]
).to_csv(OUT_DIR / "physical_oracle_choices.csv", index=False)
write_json(OUT_DIR / "well_replay_diagnostics.json", well_payload)

checks = {
    "confirmation_physics_rmse_le_12": bool(results["confirmation"]["rmse"] <= 12.0),
    "application_physics_rmse_le_12": bool(results["application"]["rmse"] <= 12.0),
    "confirmation_family_oracle_rmse_le_7": bool(results["confirmation_oracle"]["rmse"] <= 7.0),
    "application_family_oracle_rmse_le_7": bool(results["application_oracle"]["rmse"] <= 7.0),
}
if audit is not None:
    checks.update(
        {
            "raw_error_rho_le_0p75": bool(audit["raw_signed_error_rho"] <= 0.75),
            "fold2_fixed_blend_gain_ge_0p05": bool(audit["fold2_blend_gain"] >= 0.05),
            "nonzero_fold3_blend_weight": bool(audit["selected_physics_weight_on_fold3"] >= 0.025),
        }
    )
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "raw_physics_visible_prefix_multicut_replay_endpoint_gateway",
    "sample_unit": "one complete physical well",
    "endpoint_inputs": ["MD", "Z", "TVT_input_visible_prefix"],
    "uses_gr": False,
    "uses_neighbor_spatial_cp8_gold": False,
    "historical_prediction_used_for_endpoint": False,
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "selection_discovery_wells": DISCOVERY_WELLS,
        "selection_confirmation_wells": results["confirmation"]["wells"],
        "application_fold": APPLICATION_FOLD,
        "fold1_touched": False,
        "fold_source": fold_source,
    },
    "candidate_family": {
        "candidates": CANDIDATE_NAMES,
        "windows_ft": WINDOWS_FT,
        "slope_doses": SLOPE_DOSES,
        "replay_backoffs_ft": REPLAY_BACKOFFS_FT,
    },
    "selected_selector": selected_id,
    "results": results,
    "base_6p377_audit": audit,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> retain as a genuinely different physics endpoint and test one frozen LB blend"
        if strict_pass
        else "FAIL -> visible-prefix geometry does not supply a useful independent endpoint; stop this physical family"
    ),
}
write_json(OUT_DIR / "result.json", verdict)
print("\nPHYSICS-P0 FINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
