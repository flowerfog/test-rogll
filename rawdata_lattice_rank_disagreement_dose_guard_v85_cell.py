"""V85: use ranker disagreement only as a tail-risk dose guard.

V84 showed that replacing the seed-1 path by a multi-seed aggregate does not
transfer.  This experiment keeps the V76 seed-1 coarse path and its adaptive
well dose unchanged.  On wells where independently seeded rankers disagree,
it only shrinks that correction after 1000 ft.  The first 1000 ft is exactly
frozen.  Fold-3 selects one rule and Fold-2 confirms it once; Fold-1 is not
touched in this cell.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_rank_disagreement_dose_guard_v85"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V83_SOURCE = ROOT / "rawdata_lattice_multiseed_rank_ensemble_v83_cell.py"
V83_DIR = OOF_ROOT / "rawdata_lattice_multiseed_rank_ensemble_v83b"
V84_DIR = OOF_ROOT / "rawdata_lattice_multiseed_rank_confirmation_v84"
V70_PATH = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70" / "all773_proxy_crossfit_oof.parquet"
V76_WEIGHTS = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
START_FT = 1000.0
TRANSITION_FT = 250.0
MEMBER_COLUMNS = (
    "coarse_single_seed1", "coarse_member2", "coarse_member3",
    "coarse_member4", "coarse_member5",
)
RISK_FEATURES = (
    "member_path_std_rms",
    "member_far_std_rms",
    "consensus_delta_rms",
    "consensus_far_delta_rms",
    "member_endpoint_std",
)
QUANTILES = (0.50, 0.67, 0.75, 0.80, 0.90)
RETAIN_FRACTIONS = (0.0, 0.25, 0.50, 0.75)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _bootstrap_v83():
    if not V83_SOURCE.is_file():
        raise FileNotFoundError(V83_SOURCE)
    source = V83_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V83 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"Bootstrap marker missing from {V83_SOURCE}")
    module = types.ModuleType("raw_lattice_v83_v85_definitions")
    module.__file__ = str(V83_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V83_SOURCE), "exec"), module.__dict__)
    return module


def _load_models(label):
    paths = [V83_DIR / "models" / f"{label}_seed{number}.txt" for number in range(1, 6)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen rankers: {missing}")
    return [lgb.Booster(model_file=str(path)) for path in paths]


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    error = prediction - frame[TRUTH].to_numpy(np.float64)
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid prediction: {endpoint}")
    error2 = error * error
    per_well = pd.DataFrame({"well": frame["well"], "e2": error2})
    per_well = np.sqrt(per_well.groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(np.median(per_well)),
        "p95_well_rmse": float(np.quantile(per_well, 0.95)),
    }
    masks = {
        "near_0_500": distance <= 500.0,
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for name, mask in masks.items():
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result


def _risk_features(frame):
    missing = sorted(set(MEMBER_COLUMNS + ("coarse_rank_mean3",)).difference(frame.columns))
    if missing:
        raise RuntimeError(f"Multi-seed frame missing columns: {missing}")
    rows = []
    for well, local in frame.groupby("well", sort=False):
        local = local.sort_values("row_idx", kind="stable")
        members = local.loc[:, MEMBER_COLUMNS].to_numpy(np.float64)
        distance = local["md_since"].to_numpy(np.float64)
        row_std = np.std(members, axis=1)
        consensus_delta = local["coarse_rank_mean3"].to_numpy(np.float64) - members[:, 0]
        far = distance > 2000.0
        rows.append({
            "fold": int(local["fold"].iloc[0]),
            "well": str(well),
            "rows": int(len(local)),
            "member_path_std_rms": float(np.sqrt(np.mean(row_std ** 2))),
            "member_far_std_rms": float(np.sqrt(np.mean(row_std[far] ** 2))) if far.any() else 0.0,
            "consensus_delta_rms": float(np.sqrt(np.mean(consensus_delta ** 2))),
            "consensus_far_delta_rms": float(np.sqrt(np.mean(consensus_delta[far] ** 2))) if far.any() else 0.0,
            "member_endpoint_std": float(np.std(members[-1])),
        })
    output = pd.DataFrame(rows).sort_values(["fold", "well"], kind="stable").reset_index(drop=True)
    if not np.isfinite(output.loc[:, RISK_FEATURES].to_numpy(np.float64)).all():
        raise RuntimeError("Non-finite disagreement features")
    return output


def _base_frame(fold, multiseed):
    base = pd.read_parquet(
        V70_PATH,
        columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
        filters=[("fold", "=", int(fold))],
    ).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    weights = pd.read_parquet(V76_WEIGHTS)
    weights = weights[weights["fold"].eq(fold)][["well", "ensemble_weight"]]
    weight_map = dict(zip(weights["well"].astype(str), weights["ensemble_weight"]))
    dose = base["well"].astype(str).map(weight_map)
    if dose.isna().any():
        raise RuntimeError(f"Missing V76 weights on Fold{fold}")
    base["v76"] = base[PARENT].to_numpy(np.float64) + dose.to_numpy(np.float64) * (
        base[COARSE].to_numpy(np.float64) - base[PARENT].to_numpy(np.float64)
    )
    multi = multiseed.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    if not base[["id", "well", "row_idx"]].equals(multi[["id", "well", "row_idx"]]):
        raise RuntimeError(f"Fold{fold} V70/multiseed key alignment mismatch")
    truth_gap = float(np.max(np.abs(base[TRUTH].to_numpy(np.float64) - multi[TRUTH].to_numpy(np.float64))))
    seed_delta = base[COARSE].to_numpy(np.float64) - multi["coarse_single_seed1"].to_numpy(np.float64)
    lineage_audit = {
        "fold": int(fold),
        "truth_max_abs_gap": truth_gap,
        "v70_vs_v83_seed1_rmse": float(np.sqrt(np.mean(seed_delta ** 2))),
        "v70_vs_v83_seed1_max_abs_gap": float(np.max(np.abs(seed_delta))),
    }
    if truth_gap > 0.01:
        raise RuntimeError(f"Fold{fold} truth parity failed: {truth_gap}")
    # V70 and V83 use different ranker-training lineages.  Their selected
    # paths are not expected to match; V83 contributes disagreement only.
    print("V85 LINEAGE AUDIT", lineage_audit, flush=True)
    for column in MEMBER_COLUMNS + ("coarse_rank_mean3",):
        base[column] = multi[column].to_numpy(np.float32)
    return base, lineage_audit


def _guard_prediction(frame, feature_by_well, feature, threshold, retain, direction="high"):
    risk = frame["well"].astype(str).map(feature_by_well)
    if risk.isna().any():
        raise RuntimeError("Missing risk feature for an evaluation row")
    selected = risk.to_numpy(np.float64) >= float(threshold)
    if direction == "low":
        selected = ~selected
    distance = frame["md_since"].to_numpy(np.float64)
    taper = _smoothstep((distance - START_FT) / TRANSITION_FT)
    v76 = frame["v76"].to_numpy(np.float64)
    parent = frame[PARENT].to_numpy(np.float64)
    shrink = selected.astype(np.float64) * taper * (1.0 - float(retain))
    return v76 + shrink * (parent - v76)


def _scan(frame, risk_table):
    rows = []
    baseline = _metric(frame, frame["v76"], "v76")
    rows.append({
        "feature": "none", "quantile": 1.0, "threshold": np.inf,
        "retain": 1.0, "selected_wells": 0, **baseline,
    })
    for feature in RISK_FEATURES:
        values = risk_table[feature].to_numpy(np.float64)
        mapping = dict(zip(risk_table["well"].astype(str), values))
        for quantile in QUANTILES:
            threshold = float(np.quantile(values, quantile))
            selected_wells = int(np.sum(values >= threshold))
            for retain in RETAIN_FRACTIONS:
                prediction = _guard_prediction(frame, mapping, feature, threshold, retain)
                rows.append({
                    "feature": feature, "quantile": quantile, "threshold": threshold,
                    "retain": retain, "selected_wells": selected_wells,
                    **_metric(frame, prediction, "guarded"),
                })
    return pd.DataFrame(rows), baseline


for required in (V83_SOURCE, V70_PATH, V76_WEIGHTS, V84_DIR / "fold2_matched_oof.parquet"):
    if not required.is_file():
        raise FileNotFoundError(required)

T = _bootstrap_v83()
fold_by_well = T.fold_by_well
fold3_wells = sorted(well for well, fold in fold_by_well.items() if fold == 3)
models3 = _load_models("fold45")
multi3, reports3 = T._evaluate(fold3_wells, models3, "v85_fold3_disagreement")
frame3, lineage3 = _base_frame(3, multi3)
risk3 = _risk_features(frame3)
risk3.to_csv(OUT_DIR / "fold3_disagreement_features.csv", index=False)

scan3, baseline3 = _scan(frame3, risk3)
scan3["gain"] = baseline3["rmse"] - scan3["rmse"]
scan3["far_gain"] = baseline3["far_gt2000_rmse"] - scan3["far_gt2000_rmse"]
scan3["near_degradation"] = scan3["near_0_500_rmse"] - baseline3["near_0_500_rmse"]
scan3["p95_gain"] = baseline3["p95_well_rmse"] - scan3["p95_well_rmse"]
eligible3 = scan3[
    scan3["feature"].ne("none")
    & scan3["gain"].ge(0.03)
    & scan3["far_gain"].ge(0.03)
    & scan3["near_degradation"].le(1e-10)
].copy()
scan3.sort_values(["rmse", "feature", "quantile", "retain"], kind="stable").to_csv(
    OUT_DIR / "fold3_scan.csv", index=False
)

selection_pass = not eligible3.empty
selected = None
result2 = None
confirmation_pass = False
control = None
summary_rows = [{"stage": "fold3_selection", **baseline3}]

if selection_pass:
    selected = eligible3.sort_values(["rmse", "feature", "quantile", "retain"], kind="stable").iloc[0].to_dict()
    summary_rows.append({
        "stage": "fold3_selection",
        **_metric(
            frame3,
            _guard_prediction(
                frame3,
                dict(zip(risk3["well"].astype(str), risk3[selected["feature"]])),
                selected["feature"], selected["threshold"], selected["retain"],
            ),
            "selected_guard",
        ),
    })

    multi2 = pd.read_parquet(V84_DIR / "fold2_matched_oof.parquet")
    frame2, lineage2 = _base_frame(2, multi2)
    risk2 = _risk_features(frame2)
    risk2.to_csv(OUT_DIR / "fold2_disagreement_features.csv", index=False)
    mapping2 = dict(zip(risk2["well"].astype(str), risk2[selected["feature"]]))
    prediction2 = _guard_prediction(
        frame2, mapping2, selected["feature"], selected["threshold"], selected["retain"]
    )
    base2 = _metric(frame2, frame2["v76"], "v76")
    candidate2 = _metric(frame2, prediction2, "selected_guard")

    rng = np.random.default_rng(20260730)
    shuffled_values = risk2[selected["feature"]].to_numpy(np.float64).copy()
    rng.shuffle(shuffled_values)
    shuffled_map = dict(zip(risk2["well"].astype(str), shuffled_values))
    shuffled2 = _metric(
        frame2,
        _guard_prediction(
            frame2, shuffled_map, selected["feature"], selected["threshold"], selected["retain"]
        ),
        "matched_risk_shuffle",
    )
    opposite2 = _metric(
        frame2,
        _guard_prediction(
            frame2, mapping2, selected["feature"], selected["threshold"], selected["retain"], direction="low"
        ),
        "opposite_gate",
    )
    result2 = {
        "gain": base2["rmse"] - candidate2["rmse"],
        "far_gain": base2["far_gt2000_rmse"] - candidate2["far_gt2000_rmse"],
        "near_degradation": candidate2["near_0_500_rmse"] - base2["near_0_500_rmse"],
        "p95_gain": base2["p95_well_rmse"] - candidate2["p95_well_rmse"],
        "shuffle_gain": base2["rmse"] - shuffled2["rmse"],
        "opposite_gain": base2["rmse"] - opposite2["rmse"],
    }
    confirmation_pass = bool(
        result2["gain"] >= 0.03
        and result2["far_gain"] >= 0.03
        and result2["near_degradation"] <= 1e-10
        and result2["p95_gain"] >= 0.0
        and result2["shuffle_gain"] <= 0.30 * result2["gain"]
    )
    control = {"baseline": base2, "candidate": candidate2, "shuffle": shuffled2, "opposite": opposite2}
    summary_rows.extend([
        {"stage": "fold2_confirmation", **base2},
        {"stage": "fold2_confirmation", **candidate2},
        {"stage": "fold2_confirmation", **shuffled2},
        {"stage": "fold2_confirmation", **opposite2},
    ])
    pd.DataFrame({
        "id": frame2["id"], "well": frame2["well"], "row_idx": frame2["row_idx"],
        "fold": frame2["fold"], TRUTH: frame2[TRUTH], "md_since": frame2["md_since"],
        PARENT: frame2[PARENT], "v76": frame2["v76"], "v85_guarded": prediction2,
    }).to_parquet(OUT_DIR / "fold2_matched_oof.parquet", index=False, compression="zstd")

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
pd.DataFrame(reports3).to_json(OUT_DIR / "fold3_evaluation_report.jsonl", orient="records", lines=True)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_ranker_disagreement_guard_selection_Fold2_single_confirmation",
    "sample_unit": "one complete physical well",
    "candidate_and_path": "frozen V70 crossfit coarse endpoint",
    "base_dose": "frozen V76 fixed Hist/Ridge ensemble",
    "risk_signal": "five independently seeded V82/V4 LambdaRank selectors; disagreement only, never their path",
    "lineage_audit": {
        "fold3": lineage3,
        "fold2": lineage2 if selection_pass else None,
        "interpretation": "V70 and V83 paths are different training lineages; only keys/truth must match",
    },
    "guard": {
        "prefix_frozen_ft": START_FT,
        "transition_ft": TRANSITION_FT,
        "action": "shrink V76 move toward parent on high-disagreement wells only",
    },
    "selection_search_count": int(len(scan3) - 1),
    "selected_recipe": selected,
    "fold3_selection_pass": selection_pass,
    "fold2_confirmation": result2,
    "fold2_controls": control,
    "fold1_touched": False,
    "strict_pass": bool(selection_pass and confirmation_pass),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> freeze this rule and run a separate untouched Fold1 final cell"
        if selection_pass and confirmation_pass
        else "FAIL -> ranker disagreement does not identify where V76 should be shrunk; retain V76"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV85 FOLD3 TOP 20")
print(scan3.sort_values("rmse", kind="stable").head(20).to_string(index=False), flush=True)
print("\nV85 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV85 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
