from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V4_SOURCE = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
if not V4_SOURCE.is_file():
    V4_SOURCE = Path("rawdata_lattice_coarse_ranker_v4_cell.py")
HN3_DIR = OOF_ROOT / "realistic_pairwise_crossfit_hn3"
MAINLINE_PATH = (
    OOF_ROOT
    / "frontier_6p377_cp8_neighbor_dose_replay_v33"
    / "cp8_neighbor_replay_oof.parquet"
)
OUT_DIR = OOF_ROOT / "realistic_pairwise_mainline_fusion_hn4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROW_OOF_PATH = OUT_DIR / "hn3_row_oof.parquet"
WEIGHTS = np.round(np.arange(0.0, 0.301, 0.025), 3)
DISTANCE_BINS = np.asarray([0.0, 500.0, 1000.0, 2000.0, np.inf])


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def _metrics(frame, prediction):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    well_frame = pd.DataFrame(
        {"well": frame["well"].astype(str), "error2": error * error}
    )
    well_rmse = well_frame.groupby("well", sort=False)["error2"].mean().pow(0.5)
    far = distance > 2000.0
    return {
        "rows": int(len(frame)),
        "wells": int(well_rmse.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def _correlation(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3 or np.std(left[finite]) <= 1e-12 or np.std(right[finite]) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def _distance_detrend(error, distance):
    error = np.asarray(error, np.float64).copy()
    distance = np.asarray(distance, np.float64)
    result = np.empty_like(error)
    labels = np.digitize(distance, DISTANCE_BINS[1:-1], right=True)
    for label in range(len(DISTANCE_BINS) - 1):
        mask = labels == label
        result[mask] = error[mask] - np.mean(error[mask]) if mask.any() else error[mask]
    return result


def _bootstrap_v4():
    source = V4_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V4 execution marker missing from {V4_SOURCE}")
    namespace = {"__name__": "raw_lattice_v4_definitions", "__file__": str(V4_SOURCE)}
    exec(compile(source.split(marker, 1)[0], str(V4_SOURCE), "exec"), namespace)
    return namespace


def _reconstruct_hn3_rows():
    if ROW_OOF_PATH.is_file():
        cached = pd.read_parquet(ROW_OOF_PATH)
        print(
            "HN4 ROW CACHE",
            {"rows": len(cached), "wells": int(cached["well"].nunique())},
            flush=True,
        )
        return cached

    choices = pd.read_parquet(HN3_DIR / "crossfit_bywell.parquet")
    choices = choices.loc[choices["control"].eq("aligned")].copy()
    if choices["well"].duplicated().any():
        raise RuntimeError("HN3 aligned choices contain duplicate wells")

    G = _bootstrap_v4()
    np.random.seed(G["RANDOM_SEED"])
    data_root = G["DATA_ROOT"]
    all_wells = [
        path.name.split("__", 1)[0]
        for path in sorted((data_root / "train").glob("*__horizontal_well.csv"))
    ]
    stable_folds = G["V"]["_stable_folds"](all_wells)
    stable_fold_by_well = dict(
        zip(stable_folds["well"].astype(str), stable_folds["fold"].astype(int))
    )

    prior_records = []
    prior_failures = []
    prior_wells = [well for well in all_wells if stable_fold_by_well[well] in (4, 5)]
    for number, well in enumerate(prior_wells, 1):
        try:
            prior_records.append(G["V"]["_load_record"](well))
        except Exception as exc:
            prior_failures.append({"well": well, "reason": repr(exc)})
        if number % 50 == 0 or number == len(prior_wells):
            print(
                f"HN4 prior records {number}/{len(prior_wells)} failures={len(prior_failures)}",
                flush=True,
            )
    if len(prior_records) < 307:
        raise RuntimeError(f"Only {len(prior_records)} Fold45 prior records")
    prior = G["V"]["_fit_prior"](prior_records, G["SEGMENT_LENGTH_FT"])

    blocks = []
    failures = []
    for number, choice in enumerate(choices.itertuples(index=False), 1):
        well = str(choice.well)
        try:
            record = G["V"]["_load_record"](well)
            paths, _, _, _, segments, _ = G["_candidate_union"](record, prior)
            panel = G["_load_panel"](well)
            pairwise_index = int(choice.pairwise_index)
            legal_index = int(choice.legal_index)
            if len(paths) != len(panel["mse"]):
                raise RuntimeError(
                    f"candidate count changed: regenerated={len(paths)} panel={len(panel['mse'])}"
                )
            eval_rows = np.asarray(record["eval_rows"], np.int64)
            truth = np.asarray(record["truth"], np.float64)[eval_rows]
            pairwise = G["B"]["_path_prediction"](
                record, prior, paths[pairwise_index], segments
            )[eval_rows]
            legal = G["B"]["_path_prediction"](
                record, prior, paths[legal_index], segments
            )[eval_rows]
            pairwise_sse = float(np.sum((pairwise - truth) ** 2))
            expected_sse = float(panel["sse"][pairwise_index])
            tolerance = max(0.05, 2e-6 * max(expected_sse, 1.0))
            if abs(pairwise_sse - expected_sse) > tolerance:
                raise RuntimeError(
                    f"selected path mismatch: regenerated={pairwise_sse} panel={expected_sse}"
                )
            md = np.asarray(record["md"], np.float64)[eval_rows]
            block = pd.DataFrame(
                {
                    "id": [f"{well}_{int(row)}" for row in eval_rows],
                    "well": well,
                    "row_idx": eval_rows,
                    "hn_fold": int(choice.fold),
                    "tvt_true": truth.astype(np.float32),
                    "md_since": (md - float(record["anchor_md"])).astype(np.float32),
                    "hn3_pairwise": np.asarray(pairwise, np.float32),
                    "hn3_legal": np.asarray(legal, np.float32),
                }
            )
            blocks.append(block)
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 20 == 0 or number == len(choices):
            print(
                f"HN4 reconstruct {number}/{len(choices)} success={len(blocks)} failures={len(failures)}",
                flush=True,
            )
    pd.DataFrame(failures).to_csv(OUT_DIR / "reconstruction_failures.csv", index=False)
    if len(blocks) < 600:
        raise RuntimeError(f"Only {len(blocks)} HN3 wells reconstructed: {failures[:5]}")
    output = pd.concat(blocks, ignore_index=True)
    output.to_parquet(ROW_OOF_PATH, index=False, compression="zstd")
    return output


started = time.time()
if not (HN3_DIR / "crossfit_bywell.parquet").is_file():
    raise FileNotFoundError("Run realistic_pairwise_crossfit_hn3_cell.py first")
if not MAINLINE_PATH.is_file():
    raise FileNotFoundError(MAINLINE_PATH)

hn3 = _reconstruct_hn3_rows()
mainline = pd.read_parquet(MAINLINE_PATH)
required_mainline = {
    "id",
    "well",
    "row_idx",
    "tvt_true",
    "md_since",
    "selected_neighbor_dose",
}
missing = sorted(required_mainline.difference(mainline.columns))
if missing:
    raise RuntimeError(f"Missing mainline columns: {missing}; got {mainline.columns.tolist()}")
mainline = mainline[list(required_mainline)].rename(
    columns={
        "tvt_true": "mainline_truth",
        "md_since": "mainline_md_since",
        "selected_neighbor_dose": "base_6p377",
    }
)
aligned = mainline.merge(
    hn3,
    on=["id", "well", "row_idx"],
    how="inner",
    validate="one_to_one",
)
truth_gap = np.abs(
    aligned["mainline_truth"].to_numpy(np.float64)
    - aligned["tvt_true"].to_numpy(np.float64)
)
if not len(aligned) or float(np.max(truth_gap)) > 0.01:
    raise RuntimeError(f"HN3/mainline truth alignment failed; max gap={np.max(truth_gap)}")
aligned["md_since"] = aligned["mainline_md_since"].astype(np.float32)
aligned = aligned.drop(columns=["mainline_truth", "mainline_md_since"])
print(
    "HN4 ALIGNMENT",
    {
        "rows": len(aligned),
        "wells": int(aligned["well"].nunique()),
        "hn3_wells": int(hn3["well"].nunique()),
        "mainline_wells": int(mainline["well"].nunique()),
        "truth_gap_max": float(np.max(truth_gap)),
    },
    flush=True,
)

truth = aligned["tvt_true"].to_numpy(np.float64)
base = aligned["base_6p377"].to_numpy(np.float64)
candidate = aligned["hn3_pairwise"].to_numpy(np.float64)
distance = aligned["md_since"].to_numpy(np.float64)
base_error = base - truth
candidate_error = candidate - truth
raw_rho = _correlation(base_error, candidate_error)
detrended_rho = _correlation(
    _distance_detrend(base_error, distance),
    _distance_detrend(candidate_error, distance),
)
distance_rho = {}
for index, (lower, upper) in enumerate(zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:])):
    mask = (distance > lower) & (distance <= upper)
    distance_rho[f"{lower:g}_{upper:g}"] = _correlation(
        base_error[mask], candidate_error[mask]
    )

fold_rows = []
predictions = np.empty(len(aligned), np.float64)
for held_fold in sorted(aligned["hn_fold"].unique()):
    held = aligned["hn_fold"].to_numpy(int) == int(held_fold)
    train = ~held
    scan = []
    for weight in WEIGHTS:
        pred = (1.0 - weight) * base[train] + weight * candidate[train]
        scan.append((float(np.sqrt(np.mean((pred - truth[train]) ** 2))), float(weight)))
    _, selected_weight = min(scan, key=lambda item: (item[0], item[1]))
    held_prediction = (1.0 - selected_weight) * base[held] + selected_weight * candidate[held]
    predictions[held] = held_prediction
    held_frame = aligned.loc[held].reset_index(drop=True)
    base_metrics = _metrics(held_frame, base[held])
    blend_metrics = _metrics(held_frame, held_prediction)
    fold_rows.append(
        {
            "fold": int(held_fold),
            "selected_weight": selected_weight,
            "rows": int(held.sum()),
            "wells": int(held_frame["well"].nunique()),
            "base_rmse": base_metrics["rmse"],
            "rmse": blend_metrics["rmse"],
            "gain": base_metrics["rmse"] - blend_metrics["rmse"],
            "base_far_rmse": base_metrics["far_gt2000_rmse"],
            "far_rmse": blend_metrics["far_gt2000_rmse"],
            "far_gain": base_metrics["far_gt2000_rmse"]
            - blend_metrics["far_gt2000_rmse"],
            "p95_change": blend_metrics["p95_well_rmse"]
            - base_metrics["p95_well_rmse"],
        }
    )

base_metrics = _metrics(aligned, base)
candidate_metrics = _metrics(aligned, candidate)
blend_metrics = _metrics(aligned, predictions)
fold_frame = pd.DataFrame(fold_rows)
fold_frame.to_csv(OUT_DIR / "folds.csv", index=False)
aligned["nested_blend"] = predictions.astype(np.float32)
aligned.to_parquet(OUT_DIR / "aligned_oof.parquet", index=False, compression="zstd")

gain = base_metrics["rmse"] - blend_metrics["rmse"]
far_gain = base_metrics["far_gt2000_rmse"] - blend_metrics["far_gt2000_rmse"]
checks = {
    "aligned_wells_ge_300": aligned["well"].nunique() >= 300,
    "raw_error_rho_le_0p80": raw_rho <= 0.80,
    "detrended_error_rho_le_0p75": detrended_rho <= 0.75,
    "nested_gain_ge_0p03": gain >= 0.03,
    "nested_far_gain_ge_0p05": far_gain >= 0.05,
    "positive_folds_ge_3": int((fold_frame["gain"] > 0).sum()) >= 3,
    "minimum_fold_gain_ge_minus0p05": float(fold_frame["gain"].min()) >= -0.05,
    "maximum_p95_change_le_0p20": float(fold_frame["p95_change"].max()) <= 0.20,
}
verdict = {
    "experiment": "realistic_pairwise_mainline_fusion_hn4",
    "stage": "hn3_crossfit_endpoint_vs_exact_6p377_mainline_nested_fusion",
    "sample_unit": "row-aligned OOF predictions from complete-well crossfit",
    "sources": {
        "hn3": HN3_DIR,
        "mainline": MAINLINE_PATH,
    },
    "alignment": {
        "rows": int(len(aligned)),
        "wells": int(aligned["well"].nunique()),
        "hn3_wells": int(hn3["well"].nunique()),
        "mainline_wells": int(mainline["well"].nunique()),
        "truth_gap_max": float(np.max(truth_gap)),
    },
    "error_correlation": {
        "raw": raw_rho,
        "distance_detrended": detrended_rho,
        "distance_strata": distance_rho,
    },
    "base": base_metrics,
    "candidate": candidate_metrics,
    "nested_blend": blend_metrics,
    "gain": gain,
    "far_gain": far_gain,
    "folds": fold_rows,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> fit one conservative fixed weight and rebuild a full-data HN endpoint for Kaggle"
        if all(checks.values())
        else "FAIL -> HN3 is causally real but cannot improve the exact mainline robustly"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nHN4 FOLDS")
print(fold_frame.to_string(index=False), flush=True)
print("\nHN4 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
