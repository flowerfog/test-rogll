"""V183: T1 anti-shrink and hard near-anchor reanchoring on frozen V168.

The deployed parent is immutable V168.  Anti-shrink amplifies only the
prediction residual around a legal continuation fitted to the visible prefix;
it never scales absolute TVT.  Reanchor length is selected on the other four
folds and applied once to each held-out fold.
"""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
DATA_ROOT = next((p for p in (ROOT / "rogii-wellbore-geology-prediction", ROOT) if (p / "train").is_dir()), ROOT)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").is_dir() else ROOT / "output"
SOURCE = OOF_ROOT / "historical_analog_metadata_fullrow_v168/fullrow_oof.parquet"
OUT_DIR = OOF_ROOT / "v168_antishrink_reanchor_v183"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARENT = "metadata_analog_w0p05"
GAIN = 1.17
LENGTHS = (150.0, 225.0, 300.0)
PREFIX_WINDOW_FT = 512.0
EXPECTED_ROWS, EXPECTED_WELLS = 3_783_989, 773
EXPECTED_PARENT_SHA256 = "43f23cb1f53212e8da77b38e873b24ae53ddd092a8f2fa16fa761e7a1536e8c6"


def _clean(value):
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, (np.integer, int)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)): return bool(value)
    return value


def _metric(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    by_well = pd.DataFrame({"well": frame["well"].to_numpy(), "e2": error * error}).groupby("well", sort=False)["e2"].mean()
    result = {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "exact_sse": float(error @ error),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(by_well))),
        "p95_well_rmse": float(np.quantile(np.sqrt(by_well), 0.95)),
    }
    bands = (("near_0_500_rmse", 0, 500), ("mid_500_1000_rmse", 500, 1000),
             ("mid_1000_2000_rmse", 1000, 2000), ("far_gt2000_rmse", 2000, np.inf))
    for name, low, high in bands:
        mask = (distance >= low) & (distance < high)
        result[name] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result


def _prefix_tangent(well, first_hidden_row):
    path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    horizontal = pd.read_csv(path, usecols=lambda c: c in {"MD", "TVT"})
    if not {"MD", "TVT"}.issubset(horizontal.columns):
        raise RuntimeError(f"{well}: missing MD/TVT")
    anchor = int(first_hidden_row) - 1
    if anchor < 1 or anchor >= len(horizontal):
        raise RuntimeError(f"{well}: invalid anchor {anchor}")
    md = horizontal["MD"].to_numpy(np.float64)
    tvt = horizontal["TVT"].to_numpy(np.float64)
    md_anchor, tvt_anchor = float(md[anchor]), float(tvt[anchor])
    visible = np.arange(anchor + 1)
    visible = visible[(md[visible] >= md_anchor - PREFIX_WINDOW_FT) & np.isfinite(md[visible]) & np.isfinite(tvt[visible])]
    if len(visible) < 32:
        visible = np.arange(max(0, anchor - 127), anchor + 1)
    x = md[visible] - md_anchor
    y = tvt[visible] - tvt_anchor
    denominator = float(x @ x)
    slope = float((x @ y) / denominator) if denominator > 1e-9 else 0.0
    return md, md_anchor, tvt_anchor, slope, anchor


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

frame = pd.read_parquet(SOURCE)
required = {"id", "well", "row_idx", "fold", "tvt_true", "md_since", PARENT}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V168 missing columns: {missing}")
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
if len(frame) != EXPECTED_ROWS or frame["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(f"Incomplete V168: rows={len(frame)} wells={frame['well'].nunique()}")

base = frame[PARENT].to_numpy(np.float64)
parent_sha256 = hashlib.sha256(base.astype(np.float32).tobytes()).hexdigest()
if parent_sha256 != EXPECTED_PARENT_SHA256:
    raise RuntimeError(f"V168 parent hash mismatch: {parent_sha256}")
prior = np.full(len(frame), np.nan, np.float64)
distance0 = np.full(len(frame), np.nan, np.float64)
last_known = np.full(len(frame), np.nan, np.float64)
prefix_rows = []

for number, (well, positions) in enumerate(frame.groupby("well", sort=False).indices.items(), 1):
    positions = np.asarray(positions, np.int64)
    local = frame.iloc[positions]
    rows = local["row_idx"].to_numpy(np.int64)
    md, md_anchor, tvt_anchor, slope, anchor = _prefix_tangent(str(well), int(rows.min()))
    query_md = md[rows]
    local_distance = np.maximum(0.0, query_md - md_anchor)
    prior[positions] = tvt_anchor + slope * local_distance
    distance0[positions] = local_distance - local_distance.min()
    last_known[positions] = tvt_anchor
    prefix_rows.append({"well": str(well), "fold": int(local["fold"].iloc[0]), "anchor_row": anchor,
                        "last_known_tvt": tvt_anchor, "prefix_slope": slope, "first_distance": float(local_distance.min())})
    if number % 100 == 0 or number == EXPECTED_WELLS:
        print(f"V183 prefix {number}/{EXPECTED_WELLS}", flush=True)

if not all(np.isfinite(values).all() for values in (base, prior, distance0, last_known)):
    raise RuntimeError("Non-finite V183 inputs")

amplified = prior + GAIN * (base - prior)
candidates = {}
controls = {}
for length in LENGTHS:
    decay = np.exp(-distance0 / length)
    # Exact first-hidden-row continuity, with correction vanishing in the tail.
    first_by_well = pd.Series(amplified).groupby(frame["well"], sort=False).transform("first").to_numpy(np.float64)
    candidates[length] = amplified + (last_known - first_by_well) * decay
    base_first = pd.Series(base).groupby(frame["well"], sort=False).transform("first").to_numpy(np.float64)
    controls[length] = base + (last_known - base_first) * decay

rows = []
for fold in (0, 1, 2, 3, 4, 5):
    mask = np.ones(len(frame), bool) if fold == 0 else frame["fold"].to_numpy(int) == fold
    local = frame.loc[mask].reset_index(drop=True)
    rows.append({"fold": fold, "length": 0.0, "kind": "base", **_metric(local, base[mask], "v168")})
    rows.append({"fold": fold, "length": 0.0, "kind": "amplify_only", **_metric(local, amplified[mask], "g1p17")})
    for length in LENGTHS:
        rows.append({"fold": fold, "length": length, "kind": "combined", **_metric(local, candidates[length][mask], f"g1p17_L{int(length)}")})
        rows.append({"fold": fold, "length": length, "kind": "reanchor_control", **_metric(local, controls[length][mask], f"reanchor_L{int(length)}")})

scan = pd.DataFrame(rows)
base_by_fold = scan[(scan["kind"].eq("base")) & (scan["fold"].gt(0))].set_index("fold")
selected = []
nested_prediction = np.full(len(frame), np.nan, np.float64)
for held_fold in (1, 2, 3, 4, 5):
    train_folds = [fold for fold in (1, 2, 3, 4, 5) if fold != held_fold]
    choices = []
    for length in LENGTHS:
        local = scan[(scan["kind"].eq("combined")) & (scan["length"].eq(length)) & (scan["fold"].isin(train_folds))]
        base_local = base_by_fold.loc[train_folds]
        gain = float(np.sqrt(base_local["exact_sse"].sum() / base_local["rows"].sum()) - np.sqrt(local["exact_sse"].sum() / local["rows"].sum()))
        choices.append((gain, -length, length))
    inner_gain, _, chosen_length = max(choices)
    mask = frame["fold"].to_numpy(int) == held_fold
    nested_prediction[mask] = candidates[chosen_length][mask]
    selected.append({"fold": held_fold, "selected_length": chosen_length, "inner_gain": inner_gain})

if not np.isfinite(nested_prediction).all():
    raise RuntimeError("Incomplete nested prediction")

overall_base = _metric(frame, base, "v168")
overall_nested = _metric(frame, nested_prediction, "nested_g1p17_reanchor")
fold_results = []
for fold in (1, 2, 3, 4, 5):
    mask = frame["fold"].to_numpy(int) == fold
    base_m = _metric(frame.loc[mask].reset_index(drop=True), base[mask], "v168")
    nested_m = _metric(frame.loc[mask].reset_index(drop=True), nested_prediction[mask], "nested")
    fold_results.append({"fold": fold, "gain": base_m["rmse"] - nested_m["rmse"],
                         "far_gain": base_m["far_gt2000_rmse"] - nested_m["far_gt2000_rmse"],
                         "near_gain": base_m["near_0_500_rmse"] - nested_m["near_0_500_rmse"],
                         "p95_gain": base_m["p95_well_rmse"] - nested_m["p95_well_rmse"]})

gain = overall_base["rmse"] - overall_nested["rmse"]
far_gain = overall_base["far_gt2000_rmse"] - overall_nested["far_gt2000_rmse"]
near_gain = overall_base["near_0_500_rmse"] - overall_nested["near_0_500_rmse"]
p95_gain = overall_base["p95_well_rmse"] - overall_nested["p95_well_rmse"]
checks = {
    "gain_ge_0p10": gain >= 0.10,
    "positive_folds_ge_4": sum(row["gain"] > 0 for row in fold_results) >= 4,
    "minimum_fold_gain_ge_minus0p03": min(row["gain"] for row in fold_results) >= -0.03,
    "far_gain_ge_0p10": far_gain >= 0.10,
    "near_not_worse_by_gt_0p02": near_gain >= -0.02,
    "p95_not_worse_by_gt_0p10": p95_gain >= -0.10,
}
strict_pass = bool(all(checks.values()))

prefix_table = pd.DataFrame(prefix_rows)
prefix_table.to_csv(OUT_DIR / "prefix_tangent.csv", index=False)
scan.to_csv(OUT_DIR / "scan.csv", index=False)
pd.DataFrame(selected).to_csv(OUT_DIR / "nested_selection.csv", index=False)
pd.DataFrame(fold_results).to_csv(OUT_DIR / "folds.csv", index=False)
if strict_pass:
    output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
    output["v168"] = base.astype(np.float32)
    output["nested_g1p17_reanchor"] = nested_prediction.astype(np.float32)
    output.to_parquet(OUT_DIR / "fullrow_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": "v168_antishrink_reanchor_v183",
    "stage": "fivefold_nested_T1_fullrow_validation",
    "parent": {"path": SOURCE, "column": PARENT, "prediction_sha256": parent_sha256},
    "recipe": {"gain": GAIN, "reference": "visible_prefix_512ft_origin_constrained_linear_tangent",
               "reanchor_lengths": LENGTHS, "override_rows_in_oof": 0},
    "selection": selected,
    "base": overall_base,
    "candidate": overall_nested,
    "values": {"gain": gain, "far_gain": far_gain, "near_gain": near_gain, "p95_gain": p95_gain},
    "folds": fold_results,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> package T1 after override eligibility mask" if strict_pass else "FAIL -> retain V168 and do not deploy T1",
}
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nV183 SCAN")
print(scan[(scan["fold"].eq(0))].to_string(index=False), flush=True)
print("\nV183 FOLDS")
print(pd.DataFrame(fold_results).to_string(index=False), flush=True)
print("\nV183 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
