"""Final matched OOF audit for CP8 on the deployable 6.39 tail.

Run after v9, v11, v14, and the full v15 visible-prefix replay.  This audit
does not represent hidden-ID, guarded same-well, or Q0522 edits because those
have no honest five-fold OOF analogue.
"""

from pathlib import Path
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "frontier_6p39_full_oof_cp_increment_v18"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V9_PATH = (
    OOF_ROOT / "frontier_6p39_cp8_nocp12_matched_v9"
    / "cp8_nocp12_oof.parquet"
)
V14_PATH = (
    OOF_ROOT / "frontier_6p39_cp8_modelpkg_replay_v14"
    / "modelpkg_replay_oof.parquet"
)
V15_PATH = (
    OOF_ROOT / "frontier_6p39_balanced_visible_prefix_oof_v15"
    / "full_visible_prefix_oof.parquet"
)
TAIL_MODE = os.environ.get("ROGII_FULLTAIL_MODE", "auto").strip().lower()
if TAIL_MODE not in {"auto", "modelpkg", "visible_prefix"}:
    raise ValueError("ROGII_FULLTAIL_MODE must be auto, modelpkg, or visible_prefix")

DOSES = np.round(np.arange(0.0, 1.0001, 0.05), 2)
REPORT_DOSES = (0.0, 0.25, 0.50, 0.75, 1.0)
DISTANCE_EDGES = (-np.inf, 500.0, 1000.0, 2000.0, np.inf)
DISTANCE_NAMES = (
    "near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000"
)
started = time.time()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    prediction = np.asarray(prediction, np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    if len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")
    error = prediction - truth
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }
    well_mse = pd.Series(error ** 2).groupby(frame["well"], sort=False).mean()
    well_rmse = np.sqrt(well_mse.to_numpy(np.float64))
    result.update({
        "mean_well_rmse": float(np.mean(well_rmse)),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    })
    distance = frame["md_since"].to_numpy(np.float64)
    for lower, upper, name in zip(
        DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:], DISTANCE_NAMES
    ):
        mask = (distance > lower) & (distance <= upper)
        result[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return result


def _worst_twenty(
    frame: pd.DataFrame,
    parent_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    endpoint: str,
) -> dict:
    truth = frame["tvt_true"].to_numpy(np.float64)
    parent_error2 = (np.asarray(parent_prediction) - truth) ** 2
    table = pd.DataFrame({"well": frame["well"], "error2": parent_error2})
    ranking = table.groupby("well", sort=False)["error2"].mean().sort_values(ascending=False)
    count = max(1, int(np.ceil(0.20 * len(ranking))))
    selected = set(ranking.head(count).index.astype(str))
    mask = frame["well"].astype(str).isin(selected).to_numpy()
    error = np.asarray(candidate_prediction)[mask] - truth[mask]
    return {
        "endpoint": endpoint,
        "wells": int(count),
        "rows": int(mask.sum()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }


for path in (V9_PATH, V14_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run v9, v11, v14, then v15 with "
            "ROGII_VP_STAGE=full before this cell."
        )
if TAIL_MODE == "visible_prefix" and not V15_PATH.exists():
    raise FileNotFoundError(
        f"TAIL_MODE=visible_prefix requires {V15_PATH}. Rerun patched v15 with "
        "ROGII_VP_STAGE=full."
    )

v9 = pd.read_parquet(V9_PATH)
v14 = pd.read_parquet(V14_PATH)
use_visible_prefix = TAIL_MODE == "visible_prefix" or (
    TAIL_MODE == "auto" and V15_PATH.exists()
)
if use_visible_prefix:
    v15 = pd.read_parquet(V15_PATH)
else:
    # The 24-well discovery gate rejected visible-prefix. Under the registered
    # workflow the promoted full tail therefore ends at the model-package gate.
    v15 = v14[[
        "id", "well", "row_idx", "fold", "tvt_true", "md_since",
        "parent_midpoint_modelpkg", "cp8_midpoint_modelpkg",
    ]].copy()
    v15["parent_balanced_visible_prefix"] = v15["parent_midpoint_modelpkg"]
    v15["balanced_visible_prefix"] = v15["cp8_midpoint_modelpkg"]

required_v9 = {
    "id", "well", "row_idx", "fold", "tvt_true",
    "frontier_6p39_core", "cp8_nocp12",
}
required_v14 = {
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    "parent_midpoint_modelpkg", "cp8_midpoint_modelpkg",
}
required_v15 = {
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    "cp8_midpoint_modelpkg", "balanced_visible_prefix",
}
for label, frame, required in (
    ("v9", v9, required_v9), ("v14", v14, required_v14), ("v15", v15, required_v15)
):
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")
    if frame["id"].duplicated().any():
        raise RuntimeError(f"{label} contains duplicate ids")

base = v15[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
v15_prediction_columns = ["id", "cp8_midpoint_modelpkg", "balanced_visible_prefix"]
if "parent_balanced_visible_prefix" in v15.columns:
    v15_prediction_columns.append("parent_balanced_visible_prefix")
base = base.merge(
    v9[["id", "frontier_6p39_core", "cp8_nocp12"]],
    on="id", how="left", validate="one_to_one",
).merge(
    v14[["id", "parent_midpoint_modelpkg", "cp8_midpoint_modelpkg"]],
    on="id", how="left", validate="one_to_one",
).merge(
    v15[v15_prediction_columns].rename(
        columns={"cp8_midpoint_modelpkg": "cp8_midpoint_modelpkg_v15"}
    ),
    on="id", how="left", validate="one_to_one",
)
if base.isna().any().any():
    raise RuntimeError("Incomplete v9/v14/v15 alignment")

truth_gap_v14 = np.abs(
    v14.set_index("id").loc[base["id"], "tvt_true"].to_numpy(np.float64)
    - base["tvt_true"].to_numpy(np.float64)
)
cp8_gap = np.abs(
    base["cp8_midpoint_modelpkg"].to_numpy(np.float64)
    - base["cp8_midpoint_modelpkg_v15"].to_numpy(np.float64)
)
if float(truth_gap_v14.max()) > 0.01 or float(cp8_gap.max()) > 1e-4:
    raise RuntimeError(
        f"Alignment mismatch: truth={truth_gap_v14.max()} cp8={cp8_gap.max()}"
    )

core = base["frontier_6p39_core"].to_numpy(np.float64)
cp8 = base["cp8_nocp12"].to_numpy(np.float64)
parent_modelpkg = base["parent_midpoint_modelpkg"].to_numpy(np.float64)
cp8_modelpkg = base["cp8_midpoint_modelpkg"].to_numpy(np.float64)
cp8_full = base["balanced_visible_prefix"].to_numpy(np.float64)

visible_prefix_move = cp8_full - cp8_modelpkg
if "parent_balanced_visible_prefix" in v15.columns:
    parent_full_replay = (
        v15.set_index("id").loc[base["id"], "parent_balanced_visible_prefix"]
        .to_numpy(np.float64)
    )
    parent_profile_mode = "exact_same_candidates_nonlinear_profile_replay"
else:
    # Backward-compatible fallback for old v15 output. Rerunning the patched v15
    # is preferred because profile clipping can depend on the parent endpoint.
    parent_full_replay = parent_modelpkg + visible_prefix_move
    parent_profile_mode = "fixed_cp8_visible_prefix_move_fallback"
raw_cp_move = cp8 - core
ordered_cp_move = cp8_full - parent_full_replay

predictions = {
    "core": core,
    "cp8_before_tail": cp8,
    "parent_midpoint_modelpkg": parent_modelpkg,
    "cp8_midpoint_modelpkg": cp8_modelpkg,
    "full_parent_matched_vp_replay": parent_full_replay,
    "full_cp8_ordered": cp8_full,
}
for dose in REPORT_DOSES:
    predictions[f"full_parent_plus_raw_cp_dose_{dose:.2f}"] = (
        parent_full_replay + dose * raw_cp_move
    )

# Cross-fit only the scalar CP-last dose.  Every held-out fold receives the dose
# minimizing exact SSE on the other four folds; no held-out labels select it.
crossfit_prediction = np.empty(len(base), np.float64)
crossfit_rows = []
for fold in sorted(base["fold"].astype(int).unique()):
    valid = base["fold"].astype(int).eq(fold).to_numpy()
    train = ~valid
    truth_train = base.loc[train, "tvt_true"].to_numpy(np.float64)
    scores = []
    for dose in DOSES:
        candidate = parent_full_replay[train] + dose * raw_cp_move[train]
        scores.append(float(np.dot(candidate - truth_train, candidate - truth_train)))
    selected_dose = float(DOSES[int(np.argmin(scores))])
    crossfit_prediction[valid] = (
        parent_full_replay[valid] + selected_dose * raw_cp_move[valid]
    )
    crossfit_rows.append({
        "heldout_fold": int(fold),
        "train_rows": int(train.sum()),
        "valid_rows": int(valid.sum()),
        "selected_dose": selected_dose,
    })
predictions["full_parent_plus_raw_cp_crossfit"] = crossfit_prediction

# Controls retain move scale but destroy direction or well correspondence.
rng = np.random.default_rng(20260725)
distance_band = pd.cut(
    base["md_since"], DISTANCE_EDGES, labels=False, include_lowest=True
).astype(int)
shuffled_move = raw_cp_move.copy()
keys = pd.DataFrame({"fold": base["fold"].astype(int), "band": distance_band})
for _, indices in keys.groupby(["fold", "band"], sort=False).groups.items():
    idx = np.asarray(list(indices), np.int64)
    shuffled_move[idx] = rng.permutation(shuffled_move[idx])
controls = {
    "full_parent_minus_raw_cp_dose_0.50": parent_full_replay - 0.50 * raw_cp_move,
    "full_parent_shuffled_raw_cp_dose_0.50": parent_full_replay + 0.50 * shuffled_move,
}

summary_rows = []
for fold in sorted(base["fold"].astype(int).unique()):
    mask = base["fold"].astype(int).eq(fold).to_numpy()
    fold_frame = base.loc[mask].reset_index(drop=True)
    for name, prediction in {**predictions, **controls}.items():
        summary_rows.append({"fold": int(fold), **_metric(fold_frame, prediction[mask], name)})
for name, prediction in {**predictions, **controls}.items():
    summary_rows.append({"fold": 0, **_metric(base, prediction, name)})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "full_oof_cp_comparison.csv", index=False)
pd.DataFrame(crossfit_rows).to_csv(OUT_DIR / "crossfit_doses.csv", index=False)

overall = summary[summary["fold"].eq(0)].set_index("endpoint")
fold_table = summary[summary["fold"].ne(0)].pivot(
    index="fold", columns="endpoint", values="rmse"
)
parent_name = "full_parent_matched_vp_replay"
ordered_name = "full_cp8_ordered"
crossfit_name = "full_parent_plus_raw_cp_crossfit"
ordered_gain = float(overall.loc[parent_name, "rmse"] - overall.loc[ordered_name, "rmse"])
crossfit_gain = float(overall.loc[parent_name, "rmse"] - overall.loc[crossfit_name, "rmse"])
ordered_fold_gain = fold_table[parent_name] - fold_table[ordered_name]
crossfit_fold_gain = fold_table[parent_name] - fold_table[crossfit_name]

denominator = float(np.dot(raw_cp_move, raw_cp_move))
projection = float(np.dot(ordered_cp_move, raw_cp_move) / denominator) if denominator else np.nan
active = np.abs(raw_cp_move) >= 0.05
cancelled = active & (
    (ordered_cp_move * raw_cp_move <= 0.0)
    | (np.abs(ordered_cp_move) < 0.10 * np.abs(raw_cp_move))
)
cancelled_fraction = float(cancelled.sum() / max(1, active.sum()))

worst = pd.DataFrame([
    _worst_twenty(base, parent_full_replay, prediction, name)
    for name, prediction in predictions.items()
])
worst.to_csv(OUT_DIR / "worst20_parent_wells.csv", index=False)

gates = {
    "ordered_gain_ge_0p10": bool(ordered_gain >= 0.10),
    "ordered_positive_folds_ge_4": bool((ordered_fold_gain > 0.0).sum() >= 4),
    "crossfit_gain_ge_0p10": bool(crossfit_gain >= 0.10),
    "crossfit_positive_folds_ge_4": bool((crossfit_fold_gain > 0.0).sum() >= 4),
    "crossfit_fold1_positive": bool(crossfit_fold_gain.loc[1] > 0.0),
    "crossfit_far_nonworse": bool(
        overall.loc[crossfit_name, "far_gt2000_rmse"]
        <= overall.loc[parent_name, "far_gt2000_rmse"]
    ),
    "crossfit_p95_degradation_le_0p10": bool(
        overall.loc[crossfit_name, "p95_well_rmse"]
        <= overall.loc[parent_name, "p95_well_rmse"] + 0.10
    ),
    "ordered_move_projection_positive": bool(projection > 0.0),
}

row_output = base[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
row_output["raw_cp_move"] = raw_cp_move.astype(np.float32)
row_output["visible_prefix_move"] = visible_prefix_move.astype(np.float32)
for name, prediction in predictions.items():
    row_output[name] = np.asarray(prediction, np.float32)
row_output.to_parquet(OUT_DIR / "full_oof_cp_endpoints.parquet", index=False, compression="zstd")

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_full_generalizable_6p39_tail_plus_cp8_increment_audit",
    "inputs": {
        "v9": str(V9_PATH), "v9_sha256": _sha256(V9_PATH),
        "v14": str(V14_PATH), "v14_sha256": _sha256(V14_PATH),
        "v15": str(V15_PATH) if use_visible_prefix else None,
        "v15_sha256": _sha256(V15_PATH) if use_visible_prefix else None,
    },
    "scope": {
        "included": [
            "fold-clean 6.39 core", "CP8-noCP12", "PF midpoint",
            "persisted model-package OOF gate",
            *(["balanced visible-prefix pseudo-mask replay"] if use_visible_prefix else []),
        ],
        "excluded_no_honest_oof": [
            "Q0522 hidden-well edit", "hidden-ID correction", "guarded hidden same-well override",
        ],
        "parent_visible_prefix_note": (
            parent_profile_mode
        ),
        "strict_outer_fold_model_package_claim": False,
        "tail_mode": "visible_prefix" if use_visible_prefix else "modelpkg_after_vp_discovery_fail",
    },
    "alignment": {
        "rows": int(len(base)),
        "wells": int(base["well"].nunique()),
        "maximum_truth_gap": float(truth_gap_v14.max()),
        "maximum_v14_v15_cp8_gap": float(cp8_gap.max()),
    },
    "parent_profile_mode": parent_profile_mode,
    "overall": overall.reset_index().to_dict(orient="records"),
    "ordered_cp8_gain": ordered_gain,
    "ordered_fold_gain": {str(int(k)): float(v) for k, v in ordered_fold_gain.items()},
    "crossfit_cp_last_gain": crossfit_gain,
    "crossfit_fold_gain": {str(int(k)): float(v) for k, v in crossfit_fold_gain.items()},
    "crossfit_doses": crossfit_rows,
    "move_survival": {
        "raw_cp_active_rows_ge_0p05": int(active.sum()),
        "ordered_projection_on_raw_cp_move": projection,
        "ordered_cancelled_active_fraction": cancelled_fraction,
        "raw_cp_mean_abs_ft": float(np.mean(np.abs(raw_cp_move))),
        "ordered_cp_mean_abs_ft": float(np.mean(np.abs(ordered_cp_move))),
    },
    "worst_twenty_percent_parent_wells": worst.to_dict(orient="records"),
    "condition_number": "not_applicable; scalar dose selected by finite grid",
    "solve_failures": 0,
    "test_well_optimizer_or_search_count": 0,
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "free_gb": float(shutil.disk_usage(ROOT).free / (1024 ** 3)),
    "next": (
        "PASS -> submit the frozen cross-fit median dose as CP-last; keep CP checkpoint as final writer"
        if all(gates.values())
        else "FAIL -> do not replace the published full chain; inspect ordered and CP-last fold tables"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

display_columns = [
    "endpoint", "rows", "wells", "exact_sse", "rmse", "mae",
    "median_well_rmse", "p95_well_rmse", "near_0_500_rmse",
    "mid_500_1000_rmse", "mid_1000_2000_rmse", "far_gt2000_rmse",
]
print("FULL 6P39+CP OOF SUMMARY")
print(overall.reset_index()[display_columns].to_string(index=False), flush=True)
print("\nCROSSFIT DOSES")
print(pd.DataFrame(crossfit_rows).to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
