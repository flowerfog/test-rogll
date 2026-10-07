"""Local OOF tail diagnostic for rogii-det-mha160sep4.ipynb.

Run this after the notebook has created ``train_df`` and
``ridge_oof_preds``.  The primary metric reproduces the notebook's OOF
``apply_pp`` calculation.  A second final-SG variant mirrors the subsequent
per-well Savitzky-Golay step used for test predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


PRIMARY_VARIANT = "ridge_pp_pre_sg"
OUTPUT_DIR = Path(
    globals().get("OOF_TAIL_OUTPUT_DIR", Path.cwd() / "oof_tail_diagnostic")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _as_vector(value, name, expected_rows):
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(vector) != expected_rows:
        raise RuntimeError(
            f"{name} has {len(vector)} rows, expected {expected_rows}"
        )
    if not np.isfinite(vector).all():
        bad = int((~np.isfinite(vector)).sum())
        raise RuntimeError(f"{name} contains {bad} non-finite values")
    return vector


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _apply_pp_local(frame, model_delta, pf_delta, parameters):
    required = {"alpha", "tau", "w_pf"}
    missing = required.difference(parameters)
    if missing:
        raise RuntimeError(f"pp_params misses {sorted(missing)}")
    delta = model_delta * (1.0 - parameters["w_pf"]) + pf_delta * parameters["w_pf"]
    tau = float(parameters["tau"])
    if tau > 0:
        warmup = 1.0 - np.exp(
            -np.maximum(frame["md_since"].to_numpy(np.float64), 0.0) / tau
        )
        delta = delta * warmup
    return delta * float(parameters["alpha"])


def _smooth_by_well(frame, prediction, window=17, polynomial=3):
    """Mirror sg_smooth, while making within-well MD ordering explicit."""

    result = np.asarray(prediction, np.float64).copy()
    work = frame[["well", "md_since"]].reset_index(drop=True)
    for _, positions in work.groupby("well", sort=False).groups.items():
        positions = np.asarray(list(positions), dtype=np.int64)
        order = np.argsort(work.loc[positions, "md_since"].to_numpy(np.float64))
        sorted_positions = positions[order]
        values = result[sorted_positions]
        length = len(values)
        selected_window = min(int(window), length)
        if selected_window % 2 == 0:
            selected_window -= 1
        if selected_window >= int(polynomial) + 2:
            values = savgol_filter(values, selected_window, int(polynomial))
        result[sorted_positions] = values
    return result


def _distance_label(values):
    values = np.asarray(values, np.float64)
    labels = np.full(len(values), "far_gt2000", dtype=object)
    labels[values <= 2000.0] = "mid_1000_2000"
    labels[values <= 1000.0] = "mid_500_1000"
    labels[values <= 500.0] = "near_0_500"
    return labels


def _variant_summary(frame, truth, predictions):
    rows = []
    well_tables = {}
    for name, prediction in predictions.items():
        error = prediction - truth
        squared_error = error * error
        work = pd.DataFrame(
            {
                "well": frame["well"].to_numpy(str),
                "error": error,
                "err2": squared_error,
            }
        )
        grouped = work.groupby("well", sort=False)
        per_well = grouped.agg(
            n=("err2", "size"),
            sse=("err2", "sum"),
            mae=("error", lambda values: float(np.mean(np.abs(values)))),
            bias=("error", "mean"),
        )
        per_well["rmse"] = np.sqrt(per_well["sse"] / per_well["n"])
        per_well = per_well.sort_values("rmse", ascending=False)
        well_tables[name] = per_well

        worst_count = max(1, int(np.ceil(0.20 * len(per_well))))
        worst_wells = set(per_well.head(worst_count).index.astype(str))
        worst_mask = work["well"].isin(worst_wells).to_numpy()
        total_sse = float(squared_error.sum())
        worst_sse = float(squared_error[worst_mask].sum())
        rows.append(
            {
                "variant": name,
                "rows": int(len(error)),
                "wells": int(len(per_well)),
                "exact_sse": total_sse,
                "rmse": float(np.sqrt(total_sse / len(error))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
                "median_well_rmse": float(per_well["rmse"].median()),
                "p75_well_rmse": float(per_well["rmse"].quantile(0.75)),
                "p95_well_rmse": float(per_well["rmse"].quantile(0.95)),
                "max_well_rmse": float(per_well["rmse"].max()),
                "worst20_wells": worst_count,
                "worst20_rows": int(worst_mask.sum()),
                "worst20_pooled_rmse": float(
                    np.sqrt(worst_sse / max(int(worst_mask.sum()), 1))
                ),
                "worst20_sse_fraction": float(worst_sse / max(total_sse, 1e-12)),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse"), well_tables


required_globals = ("train_df", "ridge_oof_preds")
missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Run this cell after the notebook OOF calculation. Missing globals: "
        + repr(missing_globals)
    )

required_columns = {"well", "last_known_tvt", "target", "pf_ancc", "md_since"}
missing_columns = required_columns.difference(train_df.columns)
if missing_columns:
    raise RuntimeError(f"train_df misses {sorted(missing_columns)}")

diagnostic = train_df.reset_index(drop=True).copy()
diagnostic["well"] = diagnostic["well"].astype(str)
for column in ("last_known_tvt", "target", "pf_ancc", "md_since"):
    diagnostic[column] = pd.to_numeric(diagnostic[column], errors="coerce")
if diagnostic[list(required_columns - {"well"})].isna().any().any():
    counts = diagnostic[list(required_columns - {"well"})].isna().sum()
    raise RuntimeError(f"non-finite required train_df values: {counts.to_dict()}")

n_rows = len(diagnostic)
base = diagnostic["last_known_tvt"].to_numpy(np.float64)
target_delta = diagnostic["target"].to_numpy(np.float64)
truth = base + target_delta
ridge_delta = _as_vector(ridge_oof_preds, "ridge_oof_preds", n_rows)
pf_delta = diagnostic["pf_ancc"].to_numpy(np.float64) - base

parameters = dict(globals().get("pp_params", {"alpha": 1.0, "tau": 85, "w_pf": 0.09}))
pp_delta = _apply_pp_local(diagnostic, ridge_delta, pf_delta, parameters)
pre_sg_prediction = base + pp_delta
final_sg_prediction = _smooth_by_well(diagnostic, pre_sg_prediction)

predictions = {
    "last_known_anchor": base,
    "pf_ancc": base + pf_delta,
    "ridge_raw": base + ridge_delta,
    "ridge_pp_pre_sg": pre_sg_prediction,
    "ridge_pp_final_sg": final_sg_prediction,
}
for name, prediction in predictions.items():
    predictions[name] = _as_vector(prediction, name, n_rows)

summary, well_tables = _variant_summary(diagnostic, truth, predictions)
summary.to_csv(OUTPUT_DIR / "oof_variant_summary.csv", index=False)

distance = diagnostic["md_since"].to_numpy(np.float64)
distance_labels = _distance_label(distance)
distance_rows = []
for name, prediction in predictions.items():
    error = prediction - truth
    for subset in ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000"):
        mask = distance_labels == subset
        if not mask.any():
            continue
        squared_error = error[mask] ** 2
        distance_rows.append(
            {
                "variant": name,
                "subset": subset,
                "rows": int(mask.sum()),
                "wells": int(diagnostic.loc[mask, "well"].nunique()),
                "exact_sse": float(squared_error.sum()),
                "rmse": float(np.sqrt(np.mean(squared_error))),
                "mae": float(np.mean(np.abs(error[mask]))),
                "bias": float(np.mean(error[mask])),
            }
        )
distance_frame = pd.DataFrame(distance_rows)
distance_frame.to_csv(OUTPUT_DIR / "oof_distance_strata.csv", index=False)

primary = predictions[PRIMARY_VARIANT]
primary_error = primary - truth
primary_frame = pd.DataFrame(
    {
        "well": diagnostic["well"].to_numpy(str),
        "md_since": distance,
        "err": primary_error,
        "err2": primary_error**2,
    }
)
primary_grouped = primary_frame.groupby("well", sort=False)
per_well = primary_grouped.agg(
    n=("err2", "size"),
    sse=("err2", "sum"),
    mae=("err", lambda values: float(np.mean(np.abs(values)))),
    bias=("err", "mean"),
    max_abs_error=("err", lambda values: float(np.max(np.abs(values)))),
)
per_well["rmse"] = np.sqrt(per_well["sse"] / per_well["n"])

far = primary_frame[primary_frame["md_since"] > 2000.0]
if len(far):
    far_grouped = far.groupby("well", sort=False)
    far_table = far_grouped.agg(far_n=("err2", "size"), far_sse=("err2", "sum"))
    far_table["far_rmse"] = np.sqrt(far_table["far_sse"] / far_table["far_n"])
    per_well = per_well.join(far_table, how="left")
else:
    per_well["far_n"] = 0
    per_well["far_sse"] = np.nan
    per_well["far_rmse"] = np.nan

for name, table in well_tables.items():
    per_well[f"rmse__{name}"] = table["rmse"]
per_well = per_well.sort_values("rmse", ascending=False)
per_well.to_csv(OUTPUT_DIR / "oof_per_well.csv")
per_well.head(40).to_csv(OUTPUT_DIR / "oof_worst40.csv")

primary_summary = summary[summary["variant"].eq(PRIMARY_VARIANT)].iloc[0].to_dict()
final_summary = summary[summary["variant"].eq("ridge_pp_final_sg")].iloc[0].to_dict()
verdict = {
    "experiment": "rogii_det_oof_tail_diagnostic_local_v1",
    "rows": n_rows,
    "wells": int(diagnostic["well"].nunique()),
    "primary_variant": PRIMARY_VARIANT,
    "pp_params": {key: float(value) for key, value in parameters.items()},
    "primary": primary_summary,
    "final_sg": final_summary,
    "sg_delta_rmse": float(primary_summary["rmse"] - final_summary["rmse"]),
    "outputs": {
        "summary": str(OUTPUT_DIR / "oof_variant_summary.csv"),
        "distance": str(OUTPUT_DIR / "oof_distance_strata.csv"),
        "per_well": str(OUTPUT_DIR / "oof_per_well.csv"),
        "worst40": str(OUTPUT_DIR / "oof_worst40.csv"),
    },
}
verdict = _jsonable(verdict)
(OUTPUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nOOF VARIANT SUMMARY")
print(summary.to_string(index=False))
print(
    f"\nOOF pooled ({PRIMARY_VARIANT}) = "
    f"{verdict['primary']['rmse']:.9f}"
)
print("\nPRIMARY DISTANCE STRATA")
print(
    distance_frame[distance_frame["variant"].eq(PRIMARY_VARIANT)].to_string(
        index=False
    )
)
print("\nWORST 40 WELLS")
print(per_well.head(40).to_string())
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUTPUT_DIR)
