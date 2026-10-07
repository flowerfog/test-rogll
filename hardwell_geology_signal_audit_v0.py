"""Fast geology-oriented audit of hard wells against the current best OOF.

The predictive feature block uses only values available at inference time.
Truth-derived structural-event columns are reported separately as diagnostics
and are never passed to the hardness classifier.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
PREDICTION_PATH = ROOT / "full_oof_cp_crossfit_viewer.csv"
OUT_DIR = ROOT / "output" / "hardwell_geology_signal_audit_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SURFACES = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
TAIL_WINDOWS_FT = (128.0, 256.0, 512.0)
RANDOM_SEED = 20260725


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return np.nan
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def slope(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) <= 0:
        return np.nan
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def stable_fold(well: str) -> int:
    digest = hashlib.sha256(well.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % 5 + 1


def typewell_digest(path: Path) -> str:
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    values = frame.apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float)
    if not len(values):
        return "empty"
    values = np.round(values, 4)
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def load_predictions() -> dict[str, np.ndarray]:
    frame = pd.read_csv(PREDICTION_PATH, dtype={"id": str})
    if not {"id", "tvt"}.issubset(frame.columns):
        raise RuntimeError(f"Unexpected prediction columns: {frame.columns.tolist()}")
    split = frame["id"].str.rsplit("_", n=1, expand=True)
    frame["well"] = split[0].str.lower()
    frame["row_idx"] = pd.to_numeric(split[1], errors="raise").astype(np.int32)
    if frame.duplicated(["well", "row_idx"]).any():
        raise RuntimeError("Prediction file contains duplicate well/row_idx keys")
    return {
        well: group.sort_values("row_idx")[["row_idx", "tvt"]].to_numpy(float)
        for well, group in frame.groupby("well", sort=False)
    }


def well_features(path: Path, prediction: np.ndarray) -> dict[str, float | str]:
    well = path.name.split("__", 1)[0].lower()
    frame = pd.read_csv(path)
    required = {"MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{well}: missing {sorted(required - set(frame.columns))}")
    numeric_columns = sorted(required | (set(SURFACES) & set(frame.columns)))
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    md = numeric["MD"].to_numpy(float)
    x = numeric["X"].to_numpy(float)
    y = numeric["Y"].to_numpy(float)
    z = numeric["Z"].to_numpy(float)
    gr = numeric["GR"].to_numpy(float)
    truth = numeric["TVT"].to_numpy(float)
    tvt_input = numeric["TVT_input"].to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known):
        raise RuntimeError(f"{well}: no visible TVT prefix")
    anchor = int(known[-1])

    pred_rows = prediction[:, 0].astype(np.int64)
    valid_rows = pred_rows[(pred_rows >= 0) & (pred_rows < len(frame))]
    if len(valid_rows) != len(pred_rows):
        raise RuntimeError(f"{well}: prediction row outside source CSV")
    pred = prediction[:, 1]
    target = truth[valid_rows]
    finite_score = np.isfinite(pred) & np.isfinite(target)
    if finite_score.sum() < 32:
        raise RuntimeError(f"{well}: too few scored rows")
    pred = pred[finite_score]
    target = target[finite_score]
    valid_rows = valid_rows[finite_score]
    error = pred - target
    md_since = md[valid_rows] - md[anchor]

    dx = np.gradient(x, md, edge_order=1)
    dy = np.gradient(y, md, edge_order=1)
    dz = np.gradient(z, md, edge_order=1)
    ddx = np.gradient(dx, md, edge_order=1)
    ddy = np.gradient(dy, md, edge_order=1)
    ddz = np.gradient(dz, md, edge_order=1)
    dogleg = np.sqrt(ddx**2 + ddy**2 + ddz**2) * 1000.0
    s_visible = tvt_input + z

    output: dict[str, float | str] = {
        "well": well,
        "rows": int(len(valid_rows)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "far_rmse": float(
            np.sqrt(np.mean(error[md_since > 2000.0] ** 2))
            if np.any(md_since > 2000.0)
            else np.nan
        ),
        "eval_length_ft": float(np.max(md_since)),
        "eval_gr_missing": float(np.mean(~np.isfinite(gr[valid_rows]))),
        "full_gr_missing": float(np.mean(~np.isfinite(gr))),
        "gr_scale": robust_scale(gr),
        "anchor_md": float(md[anchor]),
        "anchor_tvt": float(tvt_input[anchor]),
        "anchor_z": float(z[anchor]),
        "eval_z_slope": slope(md[valid_rows], z[valid_rows]),
        "eval_z_curvature_rms": float(np.sqrt(np.nanmean(ddz[valid_rows] ** 2)) * 1000.0),
        "eval_dogleg_p95": float(np.nanquantile(dogleg[valid_rows], 0.95)),
        "eval_dogleg_max": float(np.nanmax(dogleg[valid_rows])),
        "fold": stable_fold(well),
    }
    for window in TAIL_WINDOWS_FT:
        mask = known[md[known] >= md[anchor] - window]
        output[f"prefix_surface_slope_{int(window)}"] = slope(md[mask], s_visible[mask])
        output[f"prefix_z_slope_{int(window)}"] = slope(md[mask], z[mask])
        output[f"prefix_gr_missing_{int(window)}"] = float(np.mean(~np.isfinite(gr[mask])))
        output[f"prefix_gr_scale_{int(window)}"] = robust_scale(gr[mask])

    # Diagnostic-only truth columns below this point.
    surface = truth + z
    eval_md = md[valid_rows]
    eval_surface = surface[valid_rows]
    control_md = np.linspace(eval_md[0], eval_md[-1], 65)
    control_surface = np.interp(control_md, eval_md, eval_surface)
    rates = np.diff(control_surface) / np.maximum(np.diff(control_md), 1e-6)
    rate_change = np.diff(rates)
    output.update(
        {
            "truth_surface_rate_tv": float(np.sum(np.abs(rate_change))),
            "truth_surface_rate_change_max": float(np.max(np.abs(rate_change))),
            "truth_event_count_0015": int(np.sum(np.abs(rate_change) > 0.015)),
            "truth_event_count_0030": int(np.sum(np.abs(rate_change) > 0.030)),
            "truth_surface_linear_rmse": float(
                np.sqrt(
                    np.mean(
                        (
                            eval_surface
                            - np.polyval(np.polyfit(eval_md, eval_surface, 1), eval_md)
                        )
                        ** 2
                    )
                )
            ),
        }
    )
    return output


predictions = load_predictions()
paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
rows = []
failures = []
digests = {}
for number, path in enumerate(paths, 1):
    well = path.name.split("__", 1)[0].lower()
    if well not in predictions:
        failures.append({"well": well, "reason": "missing prediction"})
        continue
    try:
        row = well_features(path, predictions[well])
        digest = typewell_digest(TRAIN_DIR / f"{well}__typewell.csv")
        row["typewell_group"] = digest
        digests[well] = digest
        rows.append(row)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 100 == 0 or number == len(paths):
        print(f"geology audit {number}/{len(paths)} failures={len(failures)}", flush=True)

frame = pd.DataFrame(rows)
if len(frame) < 760:
    raise RuntimeError(f"Only {len(frame)} wells were audited: {failures[:10]}")
group_sizes = frame["typewell_group"].value_counts()
frame["typewell_group_size"] = frame["typewell_group"].map(group_sizes).astype(int)
hard_cut = float(frame["rmse"].quantile(0.80))
frame["hard_worst20"] = (frame["rmse"] >= hard_cut).astype(np.int8)

diagnostic_columns = [column for column in frame if column.startswith("truth_")]
excluded = {
    "well", "rmse", "mae", "bias", "far_rmse", "fold", "typewell_group",
    "hard_worst20", *diagnostic_columns,
}
legal_features = [
    column for column in frame.columns
    if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
]
feature_frame = frame[legal_features].replace([np.inf, -np.inf], np.nan)
feature_frame = feature_frame.fillna(feature_frame.median()).fillna(0.0)

frame["hard_probability_oof"] = np.nan
fold_rows = []
for fold in range(1, 6):
    train_mask = frame["fold"].ne(fold).to_numpy()
    valid_mask = ~train_mask
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=5.0,
        random_state=RANDOM_SEED + fold,
    )
    model.fit(feature_frame.loc[train_mask], frame.loc[train_mask, "hard_worst20"])
    probability = model.predict_proba(feature_frame.loc[valid_mask])[:, 1]
    frame.loc[valid_mask, "hard_probability_oof"] = probability
    truth_fold = frame.loc[valid_mask, "hard_worst20"].to_numpy()
    fold_rows.append(
        {
            "fold": fold,
            "wells": int(valid_mask.sum()),
            "hard_wells": int(truth_fold.sum()),
            "roc_auc": float(roc_auc_score(truth_fold, probability)),
            "average_precision": float(average_precision_score(truth_fold, probability)),
        }
    )

associations = []
for column in [*legal_features, *diagnostic_columns]:
    valid = np.isfinite(frame[column]) & np.isfinite(frame["rmse"])
    rho, pvalue = spearmanr(frame.loc[valid, column], frame.loc[valid, "rmse"])
    associations.append(
        {
            "feature": column,
            "kind": "truth_diagnostic" if column in diagnostic_columns else "legal",
            "spearman_rmse": float(rho),
            "pvalue": float(pvalue),
        }
    )
association_frame = pd.DataFrame(associations).sort_values(
    "spearman_rmse", key=lambda series: series.abs(), ascending=False
)

hard = frame[frame["hard_worst20"].eq(1)]
easy = frame[frame["hard_worst20"].eq(0)]
contrast_rows = []
for column in [*legal_features, *diagnostic_columns]:
    contrast_rows.append(
        {
            "feature": column,
            "kind": "truth_diagnostic" if column in diagnostic_columns else "legal",
            "easy_median": float(easy[column].median()),
            "hard_median": float(hard[column].median()),
        }
    )
contrast = pd.DataFrame(contrast_rows)

overall_auc = float(roc_auc_score(frame["hard_worst20"], frame["hard_probability_oof"]))
overall_ap = float(
    average_precision_score(frame["hard_worst20"], frame["hard_probability_oof"])
)
event_rho = float(
    association_frame.set_index("feature").loc[
        "truth_surface_rate_change_max", "spearman_rmse"
    ]
)
missing_rho = float(
    association_frame.set_index("feature").loc["eval_gr_missing", "spearman_rmse"]
)
verdict = {
    "experiment": "hardwell_geology_signal_audit_v0",
    "wells": int(len(frame)),
    "rows": int(frame["rows"].sum()),
    "oof_rmse": float(np.sqrt(np.average(frame["rmse"] ** 2, weights=frame["rows"]))),
    "hard_cut_rmse": hard_cut,
    "worst20_sse_fraction": float(
        np.sum(hard["rows"] * hard["rmse"] ** 2)
        / np.sum(frame["rows"] * frame["rmse"] ** 2)
    ),
    "legal_hardness_classifier": {
        "features": legal_features,
        "crossfit_roc_auc": overall_auc,
        "crossfit_average_precision": overall_ap,
    },
    "diagnostic": {
        "truth_event_max_change_spearman_rmse": event_rho,
        "eval_gr_missing_spearman_rmse": missing_rho,
    },
    "gates": {
        "hardness_auc_ge_0p70": overall_auc >= 0.70,
        "event_association_abs_ge_0p20": abs(event_rho) >= 0.20,
        "missingness_association_abs_ge_0p20": abs(missing_rho) >= 0.20,
    },
}
if verdict["gates"]["event_association_abs_ge_0p20"]:
    verdict["next"] = "Build an event-state decoder; structural changes are concentrated in hard wells."
elif verdict["gates"]["hardness_auc_ge_0p70"]:
    verdict["next"] = "Build a legal risk gate and route hard wells to a conservative structural prior."
else:
    verdict["next"] = "Do not build a state machine; these legal geology summaries do not identify hard wells."

frame.sort_values("rmse", ascending=False).to_csv(OUT_DIR / "wells.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUT_DIR / "classifier_folds.csv", index=False)
association_frame.to_csv(OUT_DIR / "associations.csv", index=False)
contrast.to_csv(OUT_DIR / "hard_easy_contrast.csv", index=False)
pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nTOP ASSOCIATIONS")
print(association_frame.head(20).to_string(index=False))
print("\nHARD/EASY CONTRAST")
print(
    contrast.sort_values(
        "feature", key=lambda series: series.map(
            association_frame.set_index("feature")["spearman_rmse"].abs()
        ), ascending=False,
    ).head(20).to_string(index=False)
)
print("\nCLASSIFIER FOLDS")
print(pd.DataFrame(fold_rows).to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
