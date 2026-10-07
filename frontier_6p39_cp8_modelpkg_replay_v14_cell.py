"""Matched OOF replay of the frozen 6.39 saved-model correction.

This consumes only the model package's persisted five-fold OOF predictions.
It applies the exact selected notebook gate to the parent+midpoint and
CP8+midpoint endpoints. No model fitting or parameter selection occurs here.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


EXPERIMENT = "frontier_6p39_cp8_modelpkg_replay_v14"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
POST_PATH = (
    OOF_ROOT
    / "frontier_6p39_cp8_nocp12_postprocess_v11"
    / "postprocess_oof.parquet"
)
MANIFEST_PATH = ROOT / "metadata" / "model_package_manifest.json"
OUT_DIR = OOF_ROOT / EXPERIMENT

# Frozen selected profile: vp_balanced_modelpkg_005.
GATED_MAX_WEIGHT = 0.00425
GATED_SCALE_FT = 6.0
DIFF_P95_DISABLE_FT = 25.0
SHUFFLE_SEED = 20260724


def _metric(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")
    work = pd.DataFrame(
        {"well": frame["well"].to_numpy(), "se": error * error}
    )
    well_rmse = np.sqrt(work.groupby("well", sort=False)["se"].mean().to_numpy())
    row = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "max_well_rmse": float(np.max(well_rmse)),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    for name, low, high in (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    ):
        mask = (distance > low) & (distance <= high)
        row[f"{name}_rmse"] = (
            float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
        )
    return row


def _gate(base, package):
    difference = np.abs(package - base)
    p95 = float(np.quantile(difference, 0.95))
    disabled = bool(p95 > DIFF_P95_DISABLE_FT)
    gate = GATED_MAX_WEIGHT / (1.0 + (difference / GATED_SCALE_FT) ** 2)
    if disabled:
        gate[:] = 0.0
    return gate, difference, p95, disabled


started = time.time()
OUT_DIR.mkdir(parents=True, exist_ok=True)
for path in (POST_PATH, MANIFEST_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
package_root = MANIFEST_PATH.parents[1]
oof_config = manifest.get("oof", {})
train_path = package_root / oof_config.get("train_gt", "oof/train_gt.parquet")
postprocessed_path = package_root / oof_config.get(
    "blend_oof_postprocessed", "oof/blend_oof_postprocessed.npy"
)
for path in (train_path, postprocessed_path):
    if not path.exists():
        raise FileNotFoundError(path)

frame = pd.read_parquet(POST_PATH)
required = {
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    "frontier_6p39_core", "cp8_nocp12", "midpoint_shift_ft",
}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"Postprocess OOF missing columns: {missing}")
if frame["id"].duplicated().any():
    raise RuntimeError("Duplicate ids in CP8 postprocess OOF")

package_gt = pd.read_parquet(train_path)
package_required = {"id", "well_id", "row_index", "last_known_TVT", "target_tvt"}
missing = sorted(package_required.difference(package_gt.columns))
if missing:
    raise RuntimeError(f"Package train_gt missing columns: {missing}")
package_delta = np.load(postprocessed_path, mmap_mode="r", allow_pickle=False)
if package_delta.shape != (len(package_gt),):
    raise RuntimeError(
        f"Package prediction shape {package_delta.shape} != train_gt rows {len(package_gt)}"
    )
if not np.isfinite(package_delta).all():
    raise RuntimeError("Non-finite package OOF predictions")

package = package_gt[["id", "well_id", "row_index", "last_known_TVT", "target_tvt"]].copy()
package["model_package_tvt"] = (
    package["last_known_TVT"].to_numpy(np.float64)
    + np.asarray(package_delta, dtype=np.float64)
)
package = package.rename(
    columns={"well_id": "package_well", "row_index": "package_row"}
)
frame["id"] = frame["id"].astype(str)
package["id"] = package["id"].astype(str)
frame = frame.merge(package, on="id", how="left", validate="one_to_one")
if frame["model_package_tvt"].isna().any():
    raise RuntimeError("Package OOF does not cover every CP8 row")

truth_gap = np.abs(
    frame["tvt_true"].to_numpy(np.float64)
    - frame["target_tvt"].to_numpy(np.float64)
)
well_match = frame["well"].astype(str).eq(frame["package_well"].astype(str))
row_match = frame["row_idx"].astype(np.int64).eq(frame["package_row"].astype(np.int64))
if float(truth_gap.max()) > 0.01 or not well_match.all() or not row_match.all():
    raise RuntimeError(
        "Package/CP8 alignment failed: "
        f"truth_gap={truth_gap.max()}, well_match={well_match.mean()}, "
        f"row_match={row_match.mean()}"
    )

shift = frame["midpoint_shift_ft"].to_numpy(np.float64)
package_tvt = frame["model_package_tvt"].to_numpy(np.float64)
bases = {
    "parent_midpoint": frame["frontier_6p39_core"].to_numpy(np.float64) + shift,
    "cp8_midpoint": frame["cp8_nocp12"].to_numpy(np.float64) + shift,
}

predictions = {"model_package_only": package_tvt}
gate_reports = []
for name, base in bases.items():
    gate, difference, p95, disabled = _gate(base, package_tvt)
    predictions[name] = base
    predictions[f"{name}_modelpkg"] = base + gate * (package_tvt - base)
    predictions[f"{name}_modelpkg_reverse"] = base - gate * (package_tvt - base)
    gate_reports.append(
        {
            "base": name,
            "p95_abs_package_difference": p95,
            "disabled_by_diff_guard": disabled,
            "gate_mean": float(gate.mean()),
            "gate_p95": float(np.quantile(gate, 0.95)),
            "gate_max": float(gate.max()),
            "mean_abs_final_move": float(np.mean(gate * difference)),
            "p95_abs_final_move": float(np.quantile(gate * difference, 0.95)),
            "max_abs_final_move": float(np.max(gate * difference)),
        }
    )

# Matched negative control: shuffle package residuals within fold and distance
# stratum, retaining their broad scale while breaking well/row correspondence.
rng = np.random.default_rng(SHUFFLE_SEED)
distance_bin = pd.cut(
    frame["md_since"], [-np.inf, 500, 1000, 2000, np.inf], labels=False
).astype(int)
shuffle_key = pd.DataFrame({"fold": frame["fold"].astype(int), "band": distance_bin})
for name, base in bases.items():
    residual = package_tvt - base
    shuffled = residual.copy()
    for _, indices in shuffle_key.groupby(["fold", "band"], sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=np.int64)
        shuffled[idx] = rng.permutation(shuffled[idx])
    pseudo_package = base + shuffled
    shuffle_gate, _, _, shuffle_disabled = _gate(base, pseudo_package)
    if shuffle_disabled:
        shuffle_gate[:] = 0.0
    predictions[f"{name}_modelpkg_shuffled"] = base + shuffle_gate * shuffled

summary_rows = []
for fold in sorted(frame["fold"].astype(int).unique()):
    mask = frame["fold"].astype(int).eq(fold).to_numpy()
    fold_frame = frame.loc[mask].reset_index(drop=True)
    for name, prediction in predictions.items():
        summary_rows.append(
            {"fold": int(fold), **_metric(fold_frame, prediction[mask], name)}
        )
for name, prediction in predictions.items():
    summary_rows.append({"fold": 0, **_metric(frame, prediction, name)})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_modelpkg_summary.csv", index=False)
pd.DataFrame(gate_reports).to_csv(OUT_DIR / "modelpkg_gate_report.csv", index=False)

overall = summary[summary["fold"].eq(0)].set_index("endpoint")
fold_table = summary[summary["fold"].ne(0)].pivot(
    index="fold", columns="endpoint", values="rmse"
)
incremental = {}
fold_incremental = {}
for name in bases:
    candidate = f"{name}_modelpkg"
    incremental[name] = float(
        overall.loc[name, "rmse"] - overall.loc[candidate, "rmse"]
    )
    fold_incremental[name] = {
        str(int(fold)): float(fold_table.loc[fold, name] - fold_table.loc[fold, candidate])
        for fold in fold_table.index
    }

cp8_before = float(overall.loc["cp8_midpoint", "rmse"])
cp8_after = float(overall.loc["cp8_midpoint_modelpkg", "rmse"])
parent_after = float(overall.loc["parent_midpoint_modelpkg", "rmse"])
cp8_gain_after = parent_after - cp8_after
cp8_fold_gain_after = (
    fold_table["parent_midpoint_modelpkg"] - fold_table["cp8_midpoint_modelpkg"]
)

gates = {
    "package_guard_not_disabled_for_cp8": bool(
        not next(row for row in gate_reports if row["base"] == "cp8_midpoint")
        ["disabled_by_diff_guard"]
    ),
    "modelpkg_incremental_gain_on_cp8_positive": bool(
        incremental["cp8_midpoint"] > 0.0
    ),
    "modelpkg_gain_on_cp8_positive_folds_ge_3": bool(
        sum(value > 0.0 for value in fold_incremental["cp8_midpoint"].values()) >= 3
    ),
    "correct_direction_beats_reverse": bool(
        cp8_after < float(overall.loc["cp8_midpoint_modelpkg_reverse", "rmse"])
    ),
    "correct_direction_beats_shuffled": bool(
        cp8_after < float(overall.loc["cp8_midpoint_modelpkg_shuffled", "rmse"])
    ),
    "cp8_gain_vs_parent_after_modelpkg_ge_0p10": bool(cp8_gain_after >= 0.10),
}

verdict = {
    "experiment": EXPERIMENT,
    "stage": "fivefold_matched_saved_model_correction_replay_after_midpoint",
    "inputs": {
        "postprocess_oof": str(POST_PATH),
        "manifest": str(MANIFEST_PATH),
        "train_gt": str(train_path),
        "blend_oof_postprocessed": str(postprocessed_path),
    },
    "alignment": {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "maximum_truth_gap": float(truth_gap.max()),
        "well_match_rate": float(well_match.mean()),
        "row_match_rate": float(row_match.mean()),
    },
    "frozen_recipe": {
        "gated_max_weight": GATED_MAX_WEIGHT,
        "scale_ft": GATED_SCALE_FT,
        "diff_p95_disable_ft": DIFF_P95_DISABLE_FT,
        "optimizer_or_search_count": 0,
    },
    "manifest_oof_caveat": manifest.get("oof_imputer_note"),
    "strict_outer_fold_claim": False,
    "gate_reports": gate_reports,
    "overall": overall.reset_index().to_dict(orient="records"),
    "incremental_gain": incremental,
    "fold_incremental_gain": fold_incremental,
    "cp8_gain_vs_parent_after_modelpkg": float(cp8_gain_after),
    "fold_cp8_gain_vs_parent_after_modelpkg": {
        str(int(fold)): float(value) for fold, value in cp8_fold_gain_after.items()
    },
    "gates": gates,
    "modelpkg_scope_pass": bool(all(gates.values())),
    "full_tail_replay_complete": False,
    "strict_pass": False,
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "next": (
        "MODEL PACKAGE PASS -> reconstruct balanced visible-prefix pseudo-mask OOF"
        if all(gates.values())
        else "MODEL PACKAGE FAIL -> omit this tiny correction from the CP8 branch; reconstruct visible-prefix OOF next"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

output = frame[[
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    "frontier_6p39_core", "cp8_nocp12", "midpoint_shift_ft",
    "model_package_tvt",
]].copy()
for name, prediction in predictions.items():
    output[name] = prediction.astype(np.float32)
output.to_parquet(OUT_DIR / "modelpkg_replay_oof.parquet", index=False, compression="zstd")

print("MODEL-PACKAGE GATE REPORT")
print(pd.DataFrame(gate_reports).to_string(index=False), flush=True)
print("\nMATCHED MODEL-PACKAGE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
