# Direct Kaggle cell 1: calibrate the PSG blend weight on exactly aligned OOF.
# Run after the mainline notebook has created train_df and its OOF prediction.

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


FUSE_CLIP = 60.0
MAX_BLEND_W = 0.80
MIN_ALIGNMENT_RATE = 0.995
EXPECTED_OMITTED_WELLS = 3
EXPECTED_OMITTED_ROWS = 14151
TRUTH_ATOL_FT = 0.01


def _find_unique_psg_oof():
    override = globals().get("PSG_OOF_PATH")
    if override is not None:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    hits = sorted(
        {
            Path(path)
            for pattern in ("psg_oof.parquet", "psg_oof.csv")
            for path in glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
        }
    )
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one PSG OOF file, found {len(hits)}: {hits}. "
            "Set PSG_OOF_PATH explicitly when multiple datasets are mounted."
        )
    return hits[0]


def _read_table(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _row_index(frame):
    if "row_idx" in frame.columns:
        values = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    elif "id" in frame.columns:
        extracted = frame["id"].astype(str).str.extract(r"_(\d+)$", expand=False)
        if extracted.isna().any():
            bad = frame.loc[extracted.isna(), "id"].head(5).tolist()
            raise RuntimeError(f"Cannot parse absolute row_idx from id suffix: {bad}")
        values = extracted.astype(np.int64)
    else:
        raise RuntimeError("train_df needs row_idx or an id ending in _<absolute-row-index>")
    return values.to_numpy()


def _mainline_oof_prediction(frame):
    # Preferred: explicitly provide the exact final mainline OOF vector.
    for name in ("MAINLINE_OOF_PRED", "mainline_oof_pred", "final_oof_pred"):
        if name in globals():
            values = np.asarray(globals()[name], dtype=np.float64)
            if len(values) != len(frame):
                raise RuntimeError(f"{name} length {len(values)} != train_df {len(frame)}")
            print(f"mainline OOF source: {name}")
            return values

    required = ("ridge_oof_preds", "apply_pp", "pp_params")
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(
            "Set MAINLINE_OOF_PRED to the exact final OOF prediction, or run the "
            f"mainline OOF cells first. Missing globals: {missing}"
        )
    needed_columns = {"last_known_tvt", "pf_ancc"}
    absent = sorted(needed_columns.difference(frame.columns))
    if absent:
        raise RuntimeError(f"train_df lacks mainline columns: {absent}")
    base = frame["last_known_tvt"].to_numpy(np.float64)
    pf_delta = frame["pf_ancc"].to_numpy(np.float64) - base
    delta = apply_pp(frame, ridge_oof_preds, pf_delta, **pp_params)
    print("mainline OOF source: apply_pp(train_df, ridge_oof_preds, pf_delta, **pp_params)")
    return base + np.asarray(delta, dtype=np.float64)


if "train_df" not in globals():
    raise RuntimeError("Run this cell after the notebook OOF section has created train_df")

PSG_OOF_PATH = _find_unique_psg_oof()
psg = _read_table(PSG_OOF_PATH).copy()
required_psg = {"well", "row_idx", "psg_tvt"}
missing_psg = sorted(required_psg.difference(psg.columns))
if missing_psg:
    raise RuntimeError(f"{PSG_OOF_PATH} lacks columns: {missing_psg}")
psg["well"] = psg["well"].astype(str)
psg["row_idx"] = pd.to_numeric(psg["row_idx"], errors="raise").astype(np.int64)
if psg.duplicated(["well", "row_idx"]).any():
    duplicate = psg.loc[psg.duplicated(["well", "row_idx"], keep=False)].head(10)
    raise RuntimeError(f"PSG OOF has duplicate keys:\n{duplicate}")

main = pd.DataFrame(
    {
        "well": train_df["well"].astype(str).to_numpy(),
        "row_idx": _row_index(train_df),
        "mainline_tvt": _mainline_oof_prediction(train_df),
    }
)
if main.duplicated(["well", "row_idx"]).any():
    duplicate = main.loc[main.duplicated(["well", "row_idx"], keep=False)].head(10)
    raise RuntimeError(f"mainline OOF has duplicate keys:\n{duplicate}")

if "true_tvt" in train_df.columns:
    main["true_tvt"] = train_df["true_tvt"].to_numpy(np.float64)
elif {"target", "last_known_tvt"}.issubset(train_df.columns):
    main["true_tvt"] = (
        train_df["target"].to_numpy(np.float64)
        + train_df["last_known_tvt"].to_numpy(np.float64)
    )
else:
    raise RuntimeError("Cannot construct true TVT: need true_tvt or target + last_known_tvt")

if "md_since" in train_df.columns:
    main["md_since"] = train_df["md_since"].to_numpy(np.float64)

keep_columns = ["well", "row_idx", "psg_tvt"]
for optional in ("true_tvt", "md_since", "fold"):
    if optional in psg.columns:
        keep_columns.append(optional)
aligned = main.merge(
    psg[keep_columns],
    on=["well", "row_idx"],
    how="inner",
    validate="one_to_one",
    suffixes=("_main", "_psg"),
)
alignment_rate = len(aligned) / len(main)
print(
    f"PSG OOF: {PSG_OOF_PATH} | rows={len(psg):,} wells={psg.well.nunique()}\n"
    f"alignment: {len(aligned):,}/{len(main):,} = {alignment_rate:.6%}"
)
missing = main.merge(
    psg[["well", "row_idx"]], on=["well", "row_idx"], how="left", indicator=True
)
missing = missing[missing["_merge"].eq("left_only")].copy()
missing_wells = sorted(missing["well"].unique())
partial_missing_wells = sorted(set(missing_wells).intersection(set(psg["well"].unique())))
expected_visible_test_omission = (
    len(missing_wells) == EXPECTED_OMITTED_WELLS
    and len(missing) == EXPECTED_OMITTED_ROWS
    and not partial_missing_wells
)
if alignment_rate < MIN_ALIGNMENT_RATE or not expected_visible_test_omission:
    raise RuntimeError(
        f"Alignment rate {alignment_rate:.6%} < {MIN_ALIGNMENT_RATE:.3%}. "
        "Expected only the three complete train wells deliberately omitted because "
        "their IDs occur in the visible test set. "
        f"missing_rows={len(missing)}, missing_wells={missing_wells}, "
        f"partial_missing_wells={partial_missing_wells}.\n{missing.head(20)}"
    )
print(
    "expected whole-well OOF omission:",
    {"wells": missing_wells, "rows": len(missing), "partial_wells": partial_missing_wells},
)

truth_gap_max = 0.0
truth_gap_p99 = 0.0
if "true_tvt_psg" in aligned.columns:
    truth_gap = np.abs(aligned["true_tvt_main"] - aligned["true_tvt_psg"])
    truth_gap_max = float(truth_gap.max())
    truth_gap_p99 = float(truth_gap.quantile(0.99))
    if not np.isfinite(truth_gap).all() or truth_gap_max > TRUTH_ATOL_FT:
        raise RuntimeError(
            f"PSG/mainline truth mismatch: max_abs={truth_gap.max():.9f}. "
            "These OOF files do not share the same row/target contract."
        )
    aligned["true_tvt"] = aligned["true_tvt_main"]
elif "true_tvt_main" in aligned.columns:
    aligned["true_tvt"] = aligned["true_tvt_main"]
else:
    aligned["true_tvt"] = aligned["true_tvt"]

y = aligned["true_tvt"].to_numpy(np.float64)
a = aligned["mainline_tvt"].to_numpy(np.float64)
b = aligned["psg_tvt"].to_numpy(np.float64)
finite = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
if finite.mean() < 0.999:
    raise RuntimeError(f"Finite aligned prediction rate is only {finite.mean():.6%}")
aligned = aligned.loc[finite].reset_index(drop=True)
y, a, b = y[finite], a[finite], b[finite]

eligible = np.abs(b - a) < FUSE_CLIP
direction = b - a
denominator = float(np.dot(direction[eligible], direction[eligible]))
if denominator <= 0:
    raise RuntimeError("PSG and mainline predictions are identical on eligible rows")
error_a = a - y
continuous_w = float(
    np.clip(-np.dot(error_a[eligible], direction[eligible]) / denominator, 0.0, MAX_BLEND_W)
)
grid = np.unique(np.r_[np.arange(0.0, MAX_BLEND_W + 0.0001, 0.01), continuous_w])


def _blend(weight):
    prediction = a.copy()
    prediction[eligible] = (1.0 - weight) * a[eligible] + weight * b[eligible]
    return prediction


def _metrics(prediction):
    error = prediction - y
    return {
        "rows": int(len(error)),
        "exact_sse": float(np.sum(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


scan_rows = []
for weight in grid:
    scan_rows.append({"weight": float(weight), **_metrics(_blend(float(weight)))})
scan = pd.DataFrame(scan_rows).sort_values("rmse")
best = scan.iloc[0]
BLEND_W = float(best["weight"])
prediction = _blend(BLEND_W)

aligned["mainline_error"] = a - y
aligned["psg_error"] = b - y
aligned["blend_tvt"] = prediction
aligned["blend_error"] = prediction - y
aligned["eligible"] = eligible
per_well = aligned.groupby("well", sort=False).agg(
    rows=("row_idx", "size"),
    eligible_rows=("eligible", "sum"),
    mainline_sse=("mainline_error", lambda value: float(np.sum(value**2))),
    psg_sse=("psg_error", lambda value: float(np.sum(value**2))),
    blend_sse=("blend_error", lambda value: float(np.sum(value**2))),
).reset_index()
per_well["mainline_rmse"] = np.sqrt(per_well["mainline_sse"] / per_well["rows"])
per_well["psg_rmse"] = np.sqrt(per_well["psg_sse"] / per_well["rows"])
per_well["blend_rmse"] = np.sqrt(per_well["blend_sse"] / per_well["rows"])
per_well["gain"] = per_well["mainline_rmse"] - per_well["blend_rmse"]

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
scan.to_csv(WORK / "psg_blend_weight_scan.csv", index=False)
per_well.sort_values("mainline_sse", ascending=False).to_csv(
    WORK / "psg_blend_per_well.csv", index=False
)

base_metrics = _metrics(a)
psg_metrics = _metrics(b)
blend_metrics = _metrics(prediction)
report = {
    "psg_oof": str(PSG_OOF_PATH),
    "rows": len(aligned),
    "wells": int(aligned["well"].nunique()),
    "alignment_rate": alignment_rate,
    "omitted_complete_wells": missing_wells,
    "omitted_rows": int(len(missing)),
    "truth_gap_max_ft": truth_gap_max,
    "truth_gap_p99_ft": truth_gap_p99,
    "fuse_clip": FUSE_CLIP,
    "eligible_rows": int(eligible.sum()),
    "eligible_fraction": float(eligible.mean()),
    "error_correlation": float(np.corrcoef(a - y, b - y)[0, 1]),
    "continuous_optimum": continuous_w,
    "BLEND_W": BLEND_W,
    "mainline": base_metrics,
    "psg": psg_metrics,
    "blend": blend_metrics,
    "gain": base_metrics["rmse"] - blend_metrics["rmse"],
}
(WORK / "psg_blend_calibration.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

print("\nPSG BLEND CALIBRATION")
print(json.dumps(report, indent=2))
print(f"\n>>> Put BLEND_W = {BLEND_W:.6f} in the inference cell")
print("saved:", WORK / "psg_blend_calibration.json")
