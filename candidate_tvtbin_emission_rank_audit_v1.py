"""Direct audit of candidate-conditioned TVT-bin GR aggregation.

For every complete well, construct anchored datum-shift candidates around the
current OOF path. Horizontal GR is grouped by each hypothetical TVT path and
compared with the prefix-calibrated typewell curve. No learned selector is used.
Score variants are selected on four well folds and applied to the held fold.
"""

from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = Path(os.environ.get("TVTBIN_TRAIN_DIR", ROOT / "train"))
OOF_PATH = Path(os.environ.get("TVTBIN_OOF_PATH", ROOT / "full_oof_cp_crossfit_viewer.csv"))
OUTPUT_ROOT = Path(os.environ.get("TVTBIN_OUTPUT_ROOT", ROOT / "output"))
OUT_DIR = OUTPUT_ROOT / "candidate_tvtbin_emission_rank_audit_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OFFSETS = np.arange(-48.0, 48.1, 4.0, dtype=np.float64)
BIN_WIDTHS = (0.5, 1.0, 2.0, 4.0)
RAMP_FT = 1000.0
MIN_BIN_ROWS = 3
TOPK = (1, 3, 5, 8, 16, 25)
SEED = 20260725
SOURCE_FOLD_BY_WELL = {}


def stable_fold(well: str) -> int:
    digest = hashlib.sha256(well.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % 5 + 1


def corr(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4:
        return 0.0
    x = x[valid] - np.mean(x[valid])
    y = y[valid] - np.mean(y[valid])
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator > 1e-9 else 0.0


def prepare_typewell(path: Path):
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    return frame["TVT"].to_numpy(float), frame["GR"].to_numpy(float)


def calibrate(tw_tvt, tw_gr, tvt_input, horizontal_gr):
    valid = np.isfinite(tvt_input) & np.isfinite(horizontal_gr)
    expected = np.interp(tvt_input[valid], tw_tvt, tw_gr, left=np.nan, right=np.nan)
    observed = horizontal_gr[valid]
    finite = np.isfinite(expected) & np.isfinite(observed)
    x, y = expected[finite], observed[finite]
    if len(x) < 32 or np.std(x) < 1e-6:
        return tw_gr.copy(), 12.0
    keep = np.ones(len(x), bool)
    gain, intercept = 1.0, float(np.median(y - x))
    for _ in range(4):
        design = np.column_stack([x[keep], np.ones(keep.sum())])
        gain, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        gain = float(np.clip(gain, 0.5, 1.5))
        residual = y - (gain * x + intercept)
        center = np.median(residual)
        scale = max(1.4826 * np.median(np.abs(residual - center)), 1.0)
        new_keep = np.abs(residual - center) <= 3.5 * scale
        if new_keep.sum() < 24 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    residual = y - (gain * x + intercept)
    gamma = float(max(np.median(np.abs(residual - np.median(residual))), 2.0))
    return gain * tw_gr + intercept, gamma


def projection_scores(path, gr, tw_tvt, tw_gr, gamma, width):
    valid = (
        np.isfinite(path) & np.isfinite(gr)
        & (path >= tw_tvt[0]) & (path <= tw_tvt[-1])
    )
    if valid.sum() < 32:
        return np.inf, np.inf, np.inf, 0.0, 0
    values = path[valid]
    observed = gr[valid]
    bins = np.floor(values / width).astype(np.int64)
    unique, inverse = np.unique(bins, return_inverse=True)
    count = np.bincount(inverse).astype(float)
    total = np.bincount(inverse, weights=observed)
    mean = total / np.maximum(count, 1.0)
    usable = count >= MIN_BIN_ROWS
    if usable.sum() < 8:
        usable = count >= 1
    centers = (unique.astype(float) + 0.5) * width
    reference = np.interp(centers, tw_tvt, tw_gr, left=np.nan, right=np.nan)
    usable &= np.isfinite(reference) & np.isfinite(mean)
    if usable.sum() < 8:
        return np.inf, np.inf, np.inf, float(valid.mean()), int(usable.sum())
    residual = mean[usable] - reference[usable]
    mse = float(np.mean(residual**2))
    cauchy = float(np.mean(np.log1p((residual / gamma) ** 2)))
    ncc = -corr(mean[usable], reference[usable])
    return mse, cauchy, ncc, float(valid.mean()), int(usable.sum())


def prediction_groups(path: Path):
    global SOURCE_FOLD_BY_WELL
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        prediction_column = os.environ.get("TVTBIN_PREDICTION_COLUMN")
        if prediction_column:
            if prediction_column not in frame.columns:
                raise RuntimeError(
                    f"TVTBIN_PREDICTION_COLUMN={prediction_column!r} is absent. "
                    f"Available columns: {frame.columns.tolist()}"
                )
        else:
            preferred = (
                "tvt", "prediction", "pred", "oof_pred", "crossfit_tvt",
                "full_oof_cp", "cp_crossfit", "selected_tvt",
            )
            prediction_column = next((name for name in preferred if name in frame.columns), None)
            if prediction_column is None:
                excluded = {
                    "id", "well", "row_idx", "row", "fold", "tvt_true",
                    "truth", "target", "TVT", "TVT_input",
                }
                numeric = [
                    column for column in frame.columns
                    if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
                ]
                if len(numeric) == 1:
                    prediction_column = numeric[0]
                else:
                    raise RuntimeError(
                        "Cannot safely choose the OOF prediction column. Set "
                        "os.environ['TVTBIN_PREDICTION_COLUMN'] to one of: "
                        + repr(numeric)
                        + ". All columns: "
                        + repr(frame.columns.tolist())
                    )

        if "well" in frame.columns:
            wells = frame["well"].astype(str).str.lower()
        elif "id" in frame.columns:
            wells = frame["id"].astype(str).str[:8].str.lower()
        else:
            raise RuntimeError(f"Parquet has no well/id key: {frame.columns.tolist()}")
        if "row_idx" in frame.columns:
            row_index = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
        elif "row" in frame.columns:
            row_index = pd.to_numeric(frame["row"], errors="raise").astype(np.int64)
        elif "id" in frame.columns:
            identifiers = frame["id"].astype(str)
            extracted = identifiers.str.extract(r"_(\d+)$", expand=False)
            if extracted.isna().any():
                raise RuntimeError("Could not parse absolute row suffix from parquet id")
            row_index = extracted.astype(np.int64)
        else:
            raise RuntimeError(f"Parquet has no row_idx/row/id key: {frame.columns.tolist()}")
        compact = pd.DataFrame(
            {
                "well": wells,
                "row_idx": row_index,
                "prediction": pd.to_numeric(frame[prediction_column], errors="coerce"),
            }
        ).sort_values(["well", "row_idx"])
        if "fold" in frame.columns:
            fold_frame = pd.DataFrame(
                {
                    "well": wells,
                    "fold": pd.to_numeric(frame["fold"], errors="raise").astype(int),
                }
            )
            if fold_frame.groupby("well")["fold"].nunique().max() != 1:
                raise RuntimeError("Parquet assigns more than one fold to a physical well")
            SOURCE_FOLD_BY_WELL = (
                fold_frame.drop_duplicates("well").set_index("well")["fold"].to_dict()
            )
        print(
            "TVTBIN PARQUET SOURCE",
            {
                "path": str(path),
                "prediction_column": prediction_column,
                "rows": len(compact),
                "wells": compact["well"].nunique(),
            },
            flush=True,
        )
        for well, group in compact.groupby("well", sort=False):
            yield (
                str(well),
                group["row_idx"].to_numpy(np.int32),
                group["prediction"].to_numpy(np.float32),
            )
        return

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["id", "tvt"]:
            raise RuntimeError(f"Unexpected OOF header: {header}")
        current = None
        rows, values = [], []
        for identifier, value in reader:
            well = identifier[:8].lower()
            if current is not None and well != current:
                yield current, np.asarray(rows, np.int32), np.asarray(values, np.float32)
                rows, values = [], []
            current = well
            rows.append(int(identifier[9:]))
            values.append(float(value))
        if current is not None:
            yield current, np.asarray(rows, np.int32), np.asarray(values, np.float32)


started = time.time()
well_names = sorted(path.name.split("__", 1)[0].lower() for path in TRAIN_DIR.glob("*__horizontal_well.csv"))
if not well_names:
    raise FileNotFoundError(
        f"No *__horizontal_well.csv files found under TVTBIN_TRAIN_DIR={TRAIN_DIR}. "
        "Point TVTBIN_TRAIN_DIR at the directory containing the 773 raw train well pairs."
    )
typewells = {well: prepare_typewell(TRAIN_DIR / f"{well}__typewell.csv") for well in well_names}
rng = np.random.default_rng(SEED)
shuffled_names = np.asarray(well_names, object).copy()
rng.shuffle(shuffled_names)
shuffle_map = dict(zip(well_names, shuffled_names.tolist()))
for well in well_names:
    if shuffle_map[well] == well:
        index = well_names.index(well)
        shuffle_map[well] = well_names[(index + 1) % len(well_names)]

records = []
failures = []
for number, (well, rows, base) in enumerate(prediction_groups(OOF_PATH), 1):
    try:
        horizontal = pd.read_csv(
            TRAIN_DIR / f"{well}__horizontal_well.csv",
            usecols=["MD", "GR", "TVT", "TVT_input"],
        ).apply(pd.to_numeric, errors="coerce")
        if np.any(rows < 0) or np.any(rows >= len(horizontal)):
            raise RuntimeError("OOF row outside source file")
        md = horizontal["MD"].to_numpy(float)
        gr = horizontal["GR"].to_numpy(float)
        truth = horizontal["TVT"].to_numpy(float)[rows]
        tvt_input = horizontal["TVT_input"].to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        anchor = int(known[-1])
        distance = md[rows] - md[anchor]
        ramp = np.clip(distance / RAMP_FT, 0.0, 1.0)
        eval_gr = gr[rows]
        finite = np.isfinite(base) & np.isfinite(truth)
        rows, base, truth, ramp, eval_gr = (
            values[finite] for values in (rows, base.astype(float), truth, ramp, eval_gr)
        )
        tw_tvt, tw_gr = typewells[well]
        calibrated, gamma = calibrate(tw_tvt, tw_gr, tvt_input, gr)
        shuffled_well = shuffle_map[well]
        st_tvt, st_gr = typewells[shuffled_well]
        shuffled_calibrated, shuffled_gamma = calibrate(st_tvt, st_gr, tvt_input, gr)

        for offset in OFFSETS:
            candidate = base + offset * ramp
            error = candidate - truth
            row = {
                "well": well,
                "fold": int(SOURCE_FOLD_BY_WELL.get(well, stable_fold(well))),
                "offset": float(offset),
                "rows": int(len(error)),
                "sse": float(np.dot(error, error)),
                "mse": float(np.mean(error**2)),
            }
            for band, lower, upper in (
                ("near_0_500", 0.0, 500.0),
                ("mid_500_1000", 500.0, 1000.0),
                ("mid_1000_2000", 1000.0, 2000.0),
                ("far_gt2000", 2000.0, np.inf),
            ):
                mask = (distance[finite] >= lower) & (distance[finite] < upper)
                row[f"{band}_rows"] = int(mask.sum())
                row[f"{band}_sse"] = float(np.dot(error[mask], error[mask]))
            raw_reference = np.interp(candidate, tw_tvt, calibrated, left=np.nan, right=np.nan)
            raw_valid = np.isfinite(raw_reference) & np.isfinite(eval_gr)
            raw_residual = eval_gr[raw_valid] - raw_reference[raw_valid]
            row["raw_cauchy"] = (
                float(np.mean(np.log1p((raw_residual / gamma) ** 2)))
                if len(raw_residual) >= 16 else np.inf
            )
            for width in BIN_WIDTHS:
                mse, cauchy, ncc, coverage, bins = projection_scores(
                    candidate, eval_gr, tw_tvt, calibrated, gamma, width
                )
                row[f"bin{width:g}_mse"] = mse
                row[f"bin{width:g}_cauchy"] = cauchy
                row[f"bin{width:g}_ncc"] = ncc
                row[f"bin{width:g}_coverage"] = coverage
                row[f"bin{width:g}_bins"] = bins
                _, shuffle_cauchy, _, _, _ = projection_scores(
                    candidate, eval_gr, st_tvt, shuffled_calibrated, shuffled_gamma, width
                )
                row[f"shuffle_bin{width:g}_cauchy"] = shuffle_cauchy
            records.append(row)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(well_names):
        print(f"TVT-bin audit {number}/{len(well_names)} failures={len(failures)}", flush=True)

candidates = pd.DataFrame(records)
pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)
if candidates["well"].nunique() < 760:
    raise RuntimeError(f"Only {candidates['well'].nunique()} wells: {failures[:5]}")

score_columns = ["raw_cauchy"] + [
    f"bin{width:g}_{metric}" for width in BIN_WIDTHS for metric in ("mse", "cauchy", "ncc")
]
shuffle_columns = [f"shuffle_bin{width:g}_cauchy" for width in BIN_WIDTHS]


def evaluate_score(column: str) -> dict:
    selected = candidates.loc[candidates.groupby("well")[column].idxmin()]
    return {
        "score": column,
        "rows": int(selected["rows"].sum()),
        "wells": int(len(selected)),
        "rmse": float(np.sqrt(selected["sse"].sum() / selected["rows"].sum())),
        "mean_well_rmse": float(np.sqrt(selected["mse"]).mean()),
        "median_well_rmse": float(np.sqrt(selected["mse"]).median()),
    }


score_summary = pd.DataFrame([evaluate_score(column) for column in score_columns + shuffle_columns])
base_rows = candidates[np.isclose(candidates["offset"], 0.0)]
base_rmse = float(np.sqrt(base_rows["sse"].sum() / base_rows["rows"].sum()))
oracle_rows = candidates.loc[candidates.groupby("well")["mse"].idxmin()]
oracle_rmse = float(np.sqrt(oracle_rows["sse"].sum() / oracle_rows["rows"].sum()))

# Fold-nested selection chooses the score formula, never an offset or dose.
nested_parts = []
fold_choices = []
for fold in range(1, 6):
    train = candidates["fold"].ne(fold)
    valid_wells = set(candidates.loc[~train, "well"])
    best = None
    for column in score_columns:
        train_frame = candidates[train]
        chosen = train_frame.loc[train_frame.groupby("well")[column].idxmin()]
        value = float(np.sqrt(chosen["sse"].sum() / chosen["rows"].sum()))
        key = (value, column)
        if best is None or key < best:
            best = key
    column = best[1]
    valid_frame = candidates[candidates["well"].isin(valid_wells)]
    chosen = valid_frame.loc[valid_frame.groupby("well")[column].idxmin()].copy()
    nested_parts.append(chosen)
    fold_choices.append({"fold": fold, "score": column, "train_rmse": best[0]})
nested = pd.concat(nested_parts, ignore_index=True)
nested_rmse = float(np.sqrt(nested["sse"].sum() / nested["rows"].sum()))

# Rank/oracle curves answer whether correct paths move toward the front even if
# top-1 remains imperfect.
rank_rows = []
for column in score_columns:
    truth_ranks = []
    top_sse = {k: 0.0 for k in TOPK}
    total_rows = 0
    for _, group in candidates.groupby("well", sort=False):
        ordered = group.sort_values(column, kind="mergesort")
        oracle_index = int(np.argmin(group["mse"].to_numpy(float)))
        oracle_offset = float(group.iloc[oracle_index]["offset"])
        rank = int(np.flatnonzero(np.isclose(ordered["offset"], oracle_offset))[0]) + 1
        truth_ranks.append(rank)
        total_rows += int(group["rows"].iloc[0])
        for k in TOPK:
            selected = ordered.head(k).iloc[int(np.argmin(ordered.head(k)["mse"].to_numpy(float)))]
            top_sse[k] += float(selected["sse"])
    row = {
        "score": column,
        "median_truth_rank": float(np.median(truth_ranks)),
        "p90_truth_rank": float(np.quantile(truth_ranks, 0.90)),
        "top1_exact_oracle_fraction": float(np.mean(np.asarray(truth_ranks) == 1)),
    }
    for k in TOPK:
        row[f"top{k}_oracle_rmse"] = float(np.sqrt(top_sse[k] / total_rows))
    rank_rows.append(row)
rank_summary = pd.DataFrame(rank_rows).sort_values(["top8_oracle_rmse", "median_truth_rank"])

example = candidates[candidates["well"].eq("000d7d20")].copy()
example.to_csv(OUT_DIR / "000d7d20_candidates.csv", index=False)
score_summary.to_csv(OUT_DIR / "score_summary.csv", index=False)
rank_summary.to_csv(OUT_DIR / "rank_summary.csv", index=False)
pd.DataFrame(fold_choices).to_csv(OUT_DIR / "nested_score_choices.csv", index=False)

best_rank = rank_summary.iloc[0].to_dict()
best_direct = score_summary[score_summary["score"].isin(score_columns)].sort_values("rmse").iloc[0].to_dict()
best_shuffle = score_summary[score_summary["score"].isin(shuffle_columns)].sort_values("rmse").iloc[0].to_dict()
verdict = {
    "experiment": "candidate_tvtbin_emission_rank_audit_v1",
    "stage": "fivefold_candidate_conditioned_TVT_bin_direct_emission_audit",
    "sample_unit": "complete physical well",
    "candidate_bank": {"offsets_ft": OFFSETS.tolist(), "ramp_ft": RAMP_FT},
    "wells": int(candidates["well"].nunique()),
    "base_rmse": base_rmse,
    "bank_oracle_rmse": oracle_rmse,
    "nested_score_selection_rmse": nested_rmse,
    "best_direct_score": best_direct,
    "best_rank_curve": best_rank,
    "best_shuffled_typewell": best_shuffle,
    "fold_choices": fold_choices,
    "checks": {
        "nested_gain_ge_0p10": bool(base_rmse - nested_rmse >= 0.10),
        "top8_oracle_le_5p5": bool(best_rank["top8_oracle_rmse"] <= 5.5),
        "median_truth_rank_le_5": bool(best_rank["median_truth_rank"] <= 5),
        "shuffle_worse_ge_0p05": bool(best_shuffle["rmse"] - best_direct["rmse"] >= 0.05),
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["checks"].values()))
verdict["next"] = (
    "PASS -> add TVT-bin projection to the candidate selector"
    if verdict["strict_pass"]
    else "FAIL -> true-TVT overlap is descriptive but does not solve candidate inversion"
)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\nSCORE SUMMARY")
print(score_summary.sort_values("rmse").to_string(index=False))
print("\nRANK SUMMARY")
print(rank_summary.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR)
