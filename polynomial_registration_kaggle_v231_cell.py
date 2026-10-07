"""Kaggle inference for the frozen V231b polynomial registration addon.

Run immediately after the V116 consensus-center cell.  The current
submission.csv must byte-match submission_v116_consensus_center.csv.  V232
validated the frozen correction on three distinct OOF parents, so this cell
transfers the same cap-12/dose-0.25 recipe without parent-specific retuning.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED_V231K = time.time()
WORK_V231K = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
PARENT_V231K = WORK_V231K / "submission.csv"
PARENT_NAMED_V231K = WORK_V231K / "submission_v116_consensus_center.csv"
PARENT_COPY_V231K = WORK_V231K / "submission_v116_parent_before_v231.csv"
OUTPUT_V231K = WORK_V231K / "submission.csv"
NAMED_V231K = WORK_V231K / "submission_v231_polynomial_registration.csv"
SUMMARY_V231K = WORK_V231K / "v231_polynomial_registration_summary.json"

SEED_V231K = 231
DEGREE_V231K = 4
CANDIDATES_V231K = 4096
EMPIRICAL_V231K = 2048
COST_POINTS_V231K = 256
BATCH_V231K = 256
PREFIX_WINDOW_V231K = 128.0
PREFIX_SCALE_V231K = 4.0
LAMBDA_V231K = 0.3
TOPK_V231K = 32
TEMPERATURE_V231K = 0.02
CAP_V231K = 12.0
DOSE_V231K = 0.25

required_live_v231k = ("sample", "sample_value_column", "data_root", "well_rows")
missing_live_v231k = [name for name in required_live_v231k if name not in globals()]
if missing_live_v231k:
    raise RuntimeError("Run the mainline and V116 first. Missing: " + repr(missing_live_v231k))


def _sha256_v231k(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _well_hash_v231k(well):
    return int(hashlib.sha256(str(well).encode()).hexdigest()[:12], 16)


def _design_v231k(progress):
    progress = np.asarray(progress, np.float64)
    return np.column_stack([progress ** power for power in range(1, DEGREE_V231K + 1)])


def _fit_coefficient_v231k(progress, surface_delta):
    matrix = _design_v231k(progress)
    coefficient = np.linalg.lstsq(matrix, surface_delta, rcond=None)[0]
    for _ in range(4):
        residual = surface_delta - matrix @ coefficient
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        weight = 1.0 / (1.0 + (residual / (2.5 * scale)) ** 2)
        coefficient = np.linalg.lstsq(
            matrix * weight[:, None], surface_delta * weight, rcond=None
        )[0]
    return coefficient


def _robust_affine_v231k(reference, observed):
    x = np.asarray(reference, np.float64)
    y = np.asarray(observed, np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 24:
        return 1.0, 0.0
    for _ in range(4):
        design = np.column_stack((x[keep], np.ones(keep.sum())))
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        centre = np.nanmedian(residual[keep])
        scale = 1.4826 * np.nanmedian(np.abs(residual[keep] - centre)) + 1e-6
        updated = np.isfinite(x) & np.isfinite(y) & (np.abs(residual - centre) <= 3.5 * scale)
        if updated.sum() < 24 or np.array_equal(updated, keep):
            break
        keep = updated
    return float(slope), float(intercept)


def _candidate_bank_v231k(values, parent_coefficient, well):
    median = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - median), axis=0) + 1e-3
    rng = np.random.default_rng(SEED_V231K + (_well_hash_v231k(well) % 2_000_000_000))
    empirical_index = rng.integers(0, len(values), size=EMPIRICAL_V231K)
    empirical = values[empirical_index].copy()
    empirical_jitter = np.array((0.02, 0.05, 0.10, 0.20))[np.arange(len(empirical)) % 4, None]
    empirical += rng.normal(size=empirical.shape) * scale[None, :] * empirical_jitter
    local_count = CANDIDATES_V231K - EMPIRICAL_V231K - 1
    local = np.repeat(parent_coefficient[None, :], local_count, axis=0)
    local_jitter = np.array((0.01, 0.03, 0.06, 0.12, 0.25))[np.arange(local_count) % 5, None]
    local += rng.normal(size=local.shape) * scale[None, :] * local_jitter
    return np.vstack((empirical, local, parent_coefficient[None, :]))


if not PARENT_V231K.is_file() or not PARENT_NAMED_V231K.is_file():
    raise FileNotFoundError(
        "Run V116 immediately before V231; submission.csv and "
        "submission_v116_consensus_center.csv are both required"
    )
if _sha256_v231k(PARENT_V231K) != _sha256_v231k(PARENT_NAMED_V231K):
    raise RuntimeError(
        "submission.csv does not match the frozen V116 named copy; "
        "a later cell rewrote the parent"
    )

parent_v231k = pd.read_csv(PARENT_V231K)
if not parent_v231k["id"].astype(str).equals(sample["id"].astype(str)):
    raise RuntimeError("V116 parent IDs do not match sample_submission")
parent_v231k.to_csv(PARENT_COPY_V231K, index=False)
parent_lookup_v231k = parent_v231k.set_index(parent_v231k["id"].astype(str))[sample_value_column]

# Rebuild the legal all-train coefficient library used at deployment.
train_dir_v231k = Path(data_root) / "train"
train_files_v231k = sorted(train_dir_v231k.glob("*__horizontal_well.csv"))
coefficient_rows_v231k = []
for number_v231k, path_v231k in enumerate(train_files_v231k, 1):
    well_v231k = path_v231k.name.split("__", 1)[0]
    frame_v231k = pd.read_csv(path_v231k, usecols=["MD", "Z", "TVT", "TVT_input"])
    md_v231k = frame_v231k["MD"].to_numpy(np.float64)
    z_v231k = frame_v231k["Z"].to_numpy(np.float64)
    truth_v231k = frame_v231k["TVT"].to_numpy(np.float64)
    input_v231k = frame_v231k["TVT_input"].to_numpy(np.float64)
    hidden_v231k = ~np.isfinite(input_v231k) & np.isfinite(truth_v231k)
    rows_v231k = np.flatnonzero(hidden_v231k)
    if len(rows_v231k) < 16:
        continue
    anchor_v231k = int(np.flatnonzero(np.isfinite(input_v231k))[-1])
    length_v231k = max(float(md_v231k[rows_v231k[-1]] - md_v231k[anchor_v231k]), 1.0)
    progress_v231k = (md_v231k[rows_v231k] - md_v231k[anchor_v231k]) / length_v231k
    anchor_surface_v231k = float(input_v231k[anchor_v231k] + z_v231k[anchor_v231k])
    surface_delta_v231k = truth_v231k[rows_v231k] + z_v231k[rows_v231k] - anchor_surface_v231k
    coefficient_rows_v231k.append(
        _fit_coefficient_v231k(progress_v231k, surface_delta_v231k)
    )
    if number_v231k % 200 == 0 or number_v231k == len(train_files_v231k):
        print(f"V231 coefficient library {number_v231k}/{len(train_files_v231k)}", flush=True)

coefficient_bank_v231k = np.asarray(coefficient_rows_v231k, np.float64)
if len(coefficient_bank_v231k) < 750 or coefficient_bank_v231k.shape[1] != DEGREE_V231K:
    raise RuntimeError(f"Incomplete coefficient library: {coefficient_bank_v231k.shape}")

final_v231k = parent_v231k[sample_value_column].to_numpy(np.float64).copy()
sample_position_v231k = pd.Series(np.arange(len(sample)), index=sample["id"].astype(str))
reports_v231k = []

for number_v231k, well_v231k in enumerate(sorted(well_rows), 1):
    try:
        horizontal_v231k = pd.read_csv(Path(data_root) / "test" / f"{well_v231k}__horizontal_well.csv")
        horizontal_v231k["_source_row"] = np.arange(len(horizontal_v231k), dtype=np.int64)
        horizontal_v231k = horizontal_v231k.sort_values("MD", kind="stable").reset_index(drop=True)
        typewell_v231k = pd.read_csv(
            Path(data_root) / "test" / f"{well_v231k}__typewell.csv", usecols=["TVT", "GR"]
        ).apply(pd.to_numeric, errors="coerce").dropna().sort_values("TVT")
        md_v231k = horizontal_v231k["MD"].to_numpy(np.float64)
        z_v231k = horizontal_v231k["Z"].to_numpy(np.float64)
        gr_v231k = horizontal_v231k["GR"].to_numpy(np.float64)
        input_v231k = horizontal_v231k["TVT_input"].to_numpy(np.float64)
        visible_v231k = np.isfinite(input_v231k) & np.isfinite(gr_v231k)
        hidden_rows_v231k = np.flatnonzero(~np.isfinite(input_v231k))
        anchor_v231k = int(np.flatnonzero(np.isfinite(input_v231k))[-1])
        length_v231k = max(float(md_v231k[hidden_rows_v231k[-1]] - md_v231k[anchor_v231k]), 1.0)
        progress_full_v231k = (md_v231k[hidden_rows_v231k] - md_v231k[anchor_v231k]) / length_v231k
        design_full_v231k = _design_v231k(progress_full_v231k)
        source_rows_v231k = horizontal_v231k["_source_row"].to_numpy(np.int64)[hidden_rows_v231k]
        ids_v231k = [f"{well_v231k}_{int(row)}" for row in source_rows_v231k]
        positions_v231k = sample_position_v231k.reindex(ids_v231k).to_numpy()
        if not np.isfinite(positions_v231k).all():
            raise RuntimeError("hidden row IDs do not align to sample_submission")
        positions_v231k = positions_v231k.astype(np.int64)
        parent_values_v231k = parent_lookup_v231k.reindex(ids_v231k).to_numpy(np.float64)
        if not np.isfinite(parent_values_v231k).all():
            raise RuntimeError("incomplete V116 parent alignment")

        anchor_surface_v231k = float(input_v231k[anchor_v231k] + z_v231k[anchor_v231k])
        parent_surface_delta_v231k = (
            parent_values_v231k + z_v231k[hidden_rows_v231k] - anchor_surface_v231k
        )
        parent_coefficient_v231k = _fit_coefficient_v231k(
            progress_full_v231k, parent_surface_delta_v231k
        )
        candidates_v231k = _candidate_bank_v231k(
            coefficient_bank_v231k, parent_coefficient_v231k, well_v231k
        )

        cost_positions_v231k = np.flatnonzero(np.isfinite(gr_v231k[hidden_rows_v231k]))
        sampled_v231k = np.unique(np.round(np.linspace(
            0, len(cost_positions_v231k) - 1,
            min(COST_POINTS_V231K, len(cost_positions_v231k)),
        )).astype(int))
        local_v231k = cost_positions_v231k[sampled_v231k]
        sample_rows_v231k = hidden_rows_v231k[local_v231k]
        design_v231k = _design_v231k(progress_full_v231k[local_v231k])
        tw_tvt_v231k = typewell_v231k["TVT"].to_numpy(np.float64)
        tw_gr_v231k = typewell_v231k["GR"].to_numpy(np.float64)
        tw_prefix_v231k = np.interp(input_v231k[visible_v231k], tw_tvt_v231k, tw_gr_v231k)
        slope_v231k, intercept_v231k = _robust_affine_v231k(tw_prefix_v231k, gr_v231k[visible_v231k])
        gr_scale_v231k = max(
            1.4826 * np.median(np.abs(gr_v231k[visible_v231k] - np.median(gr_v231k[visible_v231k]))),
            5.0,
        )
        gr_cost_v231k = np.empty(len(candidates_v231k), np.float64)
        for start_v231k in range(0, len(candidates_v231k), BATCH_V231K):
            stop_v231k = min(start_v231k + BATCH_V231K, len(candidates_v231k))
            surface_v231k = anchor_surface_v231k + candidates_v231k[start_v231k:stop_v231k] @ design_v231k.T
            candidate_tvt_v231k = surface_v231k - z_v231k[sample_rows_v231k][None, :]
            inside_v231k = (candidate_tvt_v231k >= tw_tvt_v231k[0]) & (candidate_tvt_v231k <= tw_tvt_v231k[-1])
            aligned_v231k = np.interp(
                candidate_tvt_v231k.reshape(-1), tw_tvt_v231k, tw_gr_v231k
            ).reshape(candidate_tvt_v231k.shape)
            residual_v231k = (
                gr_v231k[sample_rows_v231k][None, :]
                - (slope_v231k * aligned_v231k + intercept_v231k)
            ) / gr_scale_v231k
            gr_cost_v231k[start_v231k:stop_v231k] = (
                np.mean(np.log1p((residual_v231k / 1.5) ** 2), axis=1)
                + 4.0 * (1.0 - inside_v231k.mean(axis=1))
            )

        prefix_rows_v231k = np.flatnonzero(
            visible_v231k
            & (md_v231k >= md_v231k[anchor_v231k] - PREFIX_WINDOW_V231K)
            & (md_v231k <= md_v231k[anchor_v231k])
        )
        if len(prefix_rows_v231k) < 16:
            prefix_rows_v231k = np.flatnonzero(visible_v231k)[-min(64, visible_v231k.sum()):]
        prefix_design_v231k = _design_v231k(
            (md_v231k[prefix_rows_v231k] - md_v231k[anchor_v231k]) / length_v231k
        )
        observed_prefix_v231k = (
            input_v231k[prefix_rows_v231k] + z_v231k[prefix_rows_v231k] - anchor_surface_v231k
        )
        prefix_distance_v231k = np.mean(
            (candidates_v231k @ prefix_design_v231k.T - observed_prefix_v231k[None, :]) ** 2,
            axis=1,
        ) / (PREFIX_SCALE_V231K ** 2)
        score_v231k = gr_cost_v231k + LAMBDA_V231K * prefix_distance_v231k
        selected_v231k = np.argsort(score_v231k, kind="stable")[:TOPK_V231K]
        local_score_v231k = score_v231k[selected_v231k]
        weight_v231k = np.exp(-(local_score_v231k - local_score_v231k.min()) / TEMPERATURE_V231K)
        weight_v231k /= weight_v231k.sum()
        coefficient_v231k = weight_v231k @ candidates_v231k[selected_v231k]
        move_v231k = design_full_v231k @ (coefficient_v231k - parent_coefficient_v231k)
        shaped_move_v231k = CAP_V231K * np.tanh(move_v231k / CAP_V231K)
        final_v231k[positions_v231k] = parent_values_v231k + DOSE_V231K * shaped_move_v231k
        reports_v231k.append({
            "well": well_v231k, "status": "ok", "rows": len(ids_v231k),
            "mean_abs_move": float(np.mean(np.abs(DOSE_V231K * shaped_move_v231k))),
            "max_abs_move": float(np.max(np.abs(DOSE_V231K * shaped_move_v231k))),
        })
    except Exception as exc_v231k:
        reports_v231k.append({"well": well_v231k, "status": "failed", "reason": repr(exc_v231k)})
    if number_v231k % 10 == 0 or number_v231k == len(well_rows):
        print(f"V231 inference {number_v231k}/{len(well_rows)}", flush=True)

report_v231k = pd.DataFrame(reports_v231k)
report_v231k.to_csv(WORK_V231K / "v231_inference_report.csv", index=False)
failed_v231k = report_v231k.loc[report_v231k["status"].ne("ok")]
if len(failed_v231k):
    raise RuntimeError("V231 inference failures; parent submission was preserved: " + repr(failed_v231k.head().to_dict("records")))

output_v231k = sample.copy()
output_v231k[sample_value_column] = final_v231k
output_v231k.to_csv(OUTPUT_V231K, index=False)
output_v231k.to_csv(NAMED_V231K, index=False)
summary_v231k = {
    "experiment": "polynomial_registration_kaggle_v231",
    "parent": "V116 consensus center on the exact live mainline",
    "parent_portability_evidence": {
        "experiment": "polynomial_registration_parent_portability_v232",
        "v144_gain": 0.19826589196857647,
        "v169_gain": 0.2062888913354186,
        "v66_gain": 0.25761339173829434,
        "all_three_positive_folds": 5,
    },
    "recipe": {
        "window": PREFIX_WINDOW_V231K, "lambda": LAMBDA_V231K,
        "mode": f"soft{TOPK_V231K}", "cap": CAP_V231K, "dose": DOSE_V231K,
    },
    "training_coefficients": len(coefficient_bank_v231k),
    "rows": len(output_v231k),
    "test_wells": len(well_rows),
    "parent_sha256": _sha256_v231k(PARENT_COPY_V231K),
    "submission_sha256": _sha256_v231k(OUTPUT_V231K),
    "elapsed_minutes": (time.time() - STARTED_V231K) / 60.0,
    "instruction": "Submit /kaggle/working/submission.csv; no later cell may rewrite it.",
}
SUMMARY_V231K.write_text(json.dumps(summary_v231k, indent=2), encoding="utf-8")
print("\nV231 KAGGLE SUMMARY")
print(json.dumps(summary_v231k, indent=2), flush=True)
