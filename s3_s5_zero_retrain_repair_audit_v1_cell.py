"""Frozen S3/S5 repair audit: S-space decode, modes, and reachability.

Run this in the same AutoDL kernel after the original S3 and S5 definition
cells have been executed.  Existing checkpoints are loaded from disk.  No
optimizer is created and no model is trained.

The audit has two independent parts:

1. S5 is decoded in structure-S space with the visible heel endpoint included
   in interpolation.  It scans frozen posterior temperature, candidate-mode
   multiplicity correction, pruned/full menus, and a truth-target posterior
   ceiling that is explicitly marked oracle-only.
2. S3 exports iterations 0..8 and measures whether truth remains reachable by
   a hypothetical local update limited to +/-20, +/-40, or +/-64 ft.
"""

from pathlib import Path
import gc
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


REQUIRED_GLOBALS = (
    "fold",
    "train_store_raft",
    "valid_store_raft",
    "device_raft",
    "PREFIX_COLUMNS_RAFT",
    "RAFT_ITERATIONS",
    "AxialRAFTTVT",
    "RAFTPanelDataset",
    "_model_forward",
    "MODEL_DIR_RAFT",
    "SEED_RAFT",
    "S5ReferenceStore",
    "S5TemplateBank",
    "S5PanelBuilder",
    "S5WholePathScorer",
    "S5_MODEL_DIR",
    "S5_SEED",
)
missing_globals = [name for name in REQUIRED_GLOBALS if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Run the S3 and S5 definition/training cells in this kernel first; "
        "missing globals: " + repr(missing_globals)
    )


OOF_ROOT_REPAIR = Path(MODEL_DIR_RAFT).parents[1]
OUT_DIR_REPAIR = OOF_ROOT_REPAIR / "s3_s5_zero_retrain_repair_audit_v1"
OUT_DIR_REPAIR.mkdir(parents=True, exist_ok=True)

S5_TEMPERATURES = (0.50, 0.75, 1.00, 1.50, 2.00, 3.00)
S5_TRUTH_TEMPERATURES_FT = (0.50, 1.00, 2.00, 4.00)
S5_MODE_THRESHOLDS_FT = (1.0, 2.0, 4.0)
S3_REACHABILITY_FT = (20.0, 40.0, 64.0)


def _repair_rmse(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _softmax(values):
    values = np.asarray(values, np.float64)
    values = values - np.max(values)
    probability = np.exp(np.clip(values, -80.0, 0.0))
    return probability / max(float(probability.sum()), 1e-12)


def _summary_from_bywell(frame, rmse_column):
    frame = frame.copy()
    frame["sse"] = frame["rows"] * frame[rmse_column] ** 2
    pooled = float(np.sqrt(frame["sse"].sum() / max(frame["rows"].sum(), 1)))
    ordered = frame.sort_values("sse", ascending=False).reset_index(drop=True)
    total_sse = float(ordered["sse"].sum())
    total_rows = float(ordered["rows"].sum())
    output = {
        "rmse": pooled,
        "median_well_rmse": float(frame[rmse_column].median()),
        "p75_well_rmse": float(frame[rmse_column].quantile(0.75)),
        "p95_well_rmse": float(frame[rmse_column].quantile(0.95)),
    }
    for fraction, label in ((0.05, "top5pct"), (0.10, "top10pct")):
        count = max(1, int(math.ceil(len(ordered) * fraction)))
        removed_sse = float(ordered.iloc[:count]["sse"].sum())
        removed_rows = float(ordered.iloc[:count]["rows"].sum())
        output[f"{label}_well_sse_share"] = removed_sse / max(total_sse, 1e-12)
        output[f"rmse_without_{label}_wells"] = float(
            np.sqrt(
                max(total_sse - removed_sse, 0.0)
                / max(total_rows - removed_rows, 1.0)
            )
        )
    return output


def _candidate_rows_tvt_legacy(record, candidate_tvt):
    prefix = int(PREFIX_COLUMNS_RAFT)
    column_md = np.asarray(record["column_md"], np.float64)[prefix:]
    return np.stack(
        [
            np.interp(record["eval_md"], column_md, path)
            for path in np.asarray(candidate_tvt, np.float64)
        ]
    )


def _candidate_rows_s_anchor(record, candidate_tvt):
    """Interpolate candidate structure S, including the known heel endpoint."""
    prefix = int(PREFIX_COLUMNS_RAFT)
    anchor_md = float(record["column_md"][prefix - 1])
    anchor_s = float(
        record["center_tvt"][prefix - 1] + record["column_z"][prefix - 1]
    )
    eval_column_md = np.asarray(record["column_md"], np.float64)[prefix:]
    eval_column_z = np.asarray(record["column_z"], np.float64)[prefix:]
    candidate_s = np.asarray(candidate_tvt, np.float64) + eval_column_z[None, :]
    interpolation_md = np.r_[anchor_md, eval_column_md]
    output = []
    for path_s in candidate_s:
        anchored_s = np.r_[anchor_s, path_s]
        row_s = np.interp(record["eval_md"], interpolation_md, anchored_s)
        output.append(row_s - np.asarray(record["eval_z"], np.float64))
    return np.stack(output)


def _mode_labels(candidate_rows, threshold_ft):
    """Single-link clusters of near-duplicate complete-well candidates."""
    sampled = np.asarray(candidate_rows, np.float64)[:, :: max(1, candidate_rows.shape[1] // 128)]
    squared_norm = np.sum(sampled * sampled, axis=1)
    distance_squared = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * sampled @ sampled.T
    ) / max(sampled.shape[1], 1)
    distance = np.sqrt(np.maximum(distance_squared, 0.0))
    parent = np.arange(len(sampled), dtype=np.int32)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pairs = np.argwhere(np.triu(distance <= float(threshold_ft), 1))
    for left, right in pairs:
        union(int(left), int(right))
    roots = np.asarray([find(index) for index in range(len(parent))])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32)


def _multiplicity_probability(scores, labels, temperature):
    scores = np.asarray(scores, np.float64) / float(temperature)
    labels = np.asarray(labels, np.int32)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    adjusted = scores - np.log(np.maximum(counts[labels], 1))
    return _softmax(adjusted)


def _s5_variant_prediction(candidate_rows, scores, temperature=1.0, labels=None):
    if labels is None:
        probability = _softmax(np.asarray(scores) / float(temperature))
    else:
        probability = _multiplicity_probability(scores, labels, temperature)
    return probability @ np.asarray(candidate_rows, np.float64), probability


def _load_frozen_s5():
    fold_map = fold.groupby("well")["fold"].first().astype(int).to_dict()
    train_records = list(train_store_raft.records)
    valid_records = list(valid_store_raft.records)
    references = S5ReferenceStore(train_records + valid_records)
    banks = {}
    for held_out_fold in (1, 2, 3, 4, 5):
        source = [
            record
            for record in train_records
            if int(fold_map[str(record["well"])]) != held_out_fold
        ]
        if held_out_fold == 1:
            source = train_records
        banks[held_out_fold] = S5TemplateBank(
            source,
            label=f"heldout_{held_out_fold}",
            seed=S5_SEED + held_out_fold * 101,
        )
    builder = S5PanelBuilder(valid_records, banks, fold_map, references)
    checkpoint_path = Path(S5_MODEL_DIR) / "best_fold1.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location=device_raft, weights_only=False
    )
    model = S5WholePathScorer(
        int(checkpoint["sequence_channels"]),
        int(checkpoint["summary_channels"]),
    ).to(device_raft)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, builder, checkpoint, references


@torch.no_grad()
def _run_s5_audit():
    model, builder, checkpoint, references = _load_frozen_s5()
    variant_errors = {}
    bywell_rows = []
    mode_rows = []

    for index, record in enumerate(
        tqdm(builder.records, desc="frozen S5 S-space repair audit")
    ):
        truth = np.asarray(record["eval_truth"], np.float64)
        for menu_name, full_menu in (("pruned", False), ("full", True)):
            data = builder.build(index, mode="full", full_menu=full_menu)
            sequence = torch.from_numpy(data["sequence"])[None].to(device_raft)
            summary = torch.from_numpy(data["summary"])[None].to(device_raft)
            with torch.autocast(
                device_type=device_raft.type,
                dtype=torch.float16,
                enabled=(device_raft.type == "cuda"),
            ):
                scores = model(sequence, summary)[0].float().cpu().numpy()
            legacy_rows = _candidate_rows_tvt_legacy(record, data["candidate_tvt"])
            corrected_rows = _candidate_rows_s_anchor(record, data["candidate_tvt"])
            candidate_rmse = np.sqrt(
                np.mean((corrected_rows - truth[None, :]) ** 2, axis=1)
            )

            def add_variant(name, prediction, oracle_only=False):
                error = np.asarray(prediction, np.float64) - truth
                variant_errors.setdefault(name, []).append(error)
                bywell_rows.append(
                    {
                        "well": record["well"],
                        "rows": len(truth),
                        "menu": menu_name,
                        "variant": name,
                        "oracle_only": bool(oracle_only),
                        "rmse": float(np.sqrt(np.mean(error**2))),
                    }
                )

            if menu_name == "pruned":
                legacy_prediction, _ = _s5_variant_prediction(
                    legacy_rows, scores, temperature=1.0
                )
                add_variant("legacy_tvt_posterior_t1_pruned", legacy_prediction)

            map_index = int(np.argmax(scores))
            add_variant(
                f"s_anchor_map_{menu_name}", corrected_rows[map_index]
            )
            for temperature in S5_TEMPERATURES:
                prediction, _ = _s5_variant_prediction(
                    corrected_rows, scores, temperature=temperature
                )
                add_variant(
                    f"s_anchor_posterior_t{temperature:.2f}_{menu_name}", prediction
                )

            for threshold in S5_MODE_THRESHOLDS_FT:
                labels = _mode_labels(corrected_rows, threshold)
                counts = np.bincount(labels)
                mode_rows.append(
                    {
                        "well": record["well"],
                        "menu": menu_name,
                        "threshold_ft": threshold,
                        "candidates": len(labels),
                        "modes": len(counts),
                        "largest_multiplicity": int(counts.max()),
                        "duplicate_fraction": float(1.0 - len(counts) / len(labels)),
                    }
                )
                for temperature in (0.75, 1.00, 1.50):
                    prediction, _ = _s5_variant_prediction(
                        corrected_rows,
                        scores,
                        temperature=temperature,
                        labels=labels,
                    )
                    add_variant(
                        "mode_nll_"
                        f"r{threshold:.0f}_t{temperature:.2f}_{menu_name}",
                        prediction,
                    )

            for truth_temperature in S5_TRUTH_TEMPERATURES_FT:
                truth_probability = _softmax(
                    -(candidate_rmse - candidate_rmse.min()) / truth_temperature
                )
                add_variant(
                    f"truth_target_t{truth_temperature:.2f}_{menu_name}",
                    truth_probability @ corrected_rows,
                    oracle_only=True,
                )
            add_variant(
                f"candidate_oracle_{menu_name}",
                corrected_rows[int(np.argmin(candidate_rmse))],
                oracle_only=True,
            )

    bywell = pd.DataFrame(bywell_rows)
    modes = pd.DataFrame(mode_rows)
    summaries = []
    for variant, frame in bywell.groupby("variant", sort=False):
        summary = _summary_from_bywell(frame, "rmse")
        summaries.append(
            {
                "variant": variant,
                "oracle_only": bool(frame["oracle_only"].iloc[0]),
                **summary,
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["oracle_only", "rmse"], ascending=[True, True]
    )
    summary.to_csv(OUT_DIR_REPAIR / "s5_decoder_summary.csv", index=False)
    bywell.to_csv(OUT_DIR_REPAIR / "s5_decoder_bywell.csv", index=False)
    modes.to_csv(OUT_DIR_REPAIR / "s5_mode_multiplicity.csv", index=False)
    mode_summary = modes.groupby(
        ["menu", "threshold_ft"], as_index=False
    ).agg(
        mean_candidates=("candidates", "mean"),
        mean_modes=("modes", "mean"),
        median_largest_multiplicity=("largest_multiplicity", "median"),
        mean_duplicate_fraction=("duplicate_fraction", "mean"),
    )
    mode_summary.to_csv(OUT_DIR_REPAIR / "s5_mode_summary.csv", index=False)
    print("\nS5 DEPLOYABLE DECODERS")
    print(summary[~summary["oracle_only"]].head(30).to_string(index=False))
    print("\nS5 ORACLE-ONLY POSTERIORS")
    print(summary[summary["oracle_only"]].head(20).to_string(index=False))
    print("\nS5 MODE MULTIPLICITY")
    print(mode_summary.to_string(index=False))
    result = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_deployable_variant_same_fold": str(
            summary.loc[~summary["oracle_only"]].iloc[0]["variant"]
        ),
        "best_deployable_rmse_same_fold": float(
            summary.loc[~summary["oracle_only"]].iloc[0]["rmse"]
        ),
        "best_truth_target_rmse": float(
            summary.loc[summary["oracle_only"]]["rmse"].min()
        ),
    }
    del model, builder, references
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _load_frozen_s3():
    checkpoint_path = Path(MODEL_DIR_RAFT) / f"seed_{SEED_RAFT}.pt"
    if not checkpoint_path.exists():
        candidates = sorted(Path(MODEL_DIR_RAFT).glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(checkpoint_path)
        checkpoint_path = candidates[0]
    checkpoint = torch.load(
        checkpoint_path, map_location=device_raft, weights_only=False
    )
    geometry_channels = int(
        checkpoint.get(
            "geometry_channels",
            valid_store_raft.records[0]["geometry"].shape[0],
        )
    )
    model = AxialRAFTTVT(geometry_channels).to(device_raft)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, checkpoint_path


def _s3_row_prediction(record, correction):
    correction = np.asarray(correction, np.float64).copy()
    correction[: int(PREFIX_COLUMNS_RAFT)] = 0.0
    column_s = (
        np.asarray(record["center_tvt"], np.float64)
        + correction
        + np.asarray(record["column_z"], np.float64)
    )
    return (
        np.interp(record["eval_md"], record["column_md"], column_s)
        - np.asarray(record["eval_z"], np.float64)
    )


@torch.no_grad()
def _run_s3_audit():
    model, checkpoint, checkpoint_path = _load_frozen_s3()
    dataset = RAFTPanelDataset(valid_store_raft, mode="full")
    iteration_errors = {iteration: [] for iteration in range(RAFT_ITERATIONS + 1)}
    iteration_bywell = []
    reach_errors = {
        (iteration, radius): []
        for iteration in range(RAFT_ITERATIONS + 1)
        for radius in S3_REACHABILITY_FT
    }
    reach_rows = []

    for index in tqdm(range(len(dataset)), desc="frozen S3 iteration/reachability audit"):
        item = dataset[index]
        batch = {
            key: value[None]
            for key, value in item.items()
            if key != "panel_weight"
        }
        with torch.autocast(
            device_type=device_raft.type,
            dtype=torch.float16,
            enabled=(device_raft.type == "cuda"),
        ):
            model_predictions = _model_forward(model, batch)
        predictions = [
            np.zeros_like(valid_store_raft.records[index]["center_tvt"], np.float32)
        ] + [prediction[0].float().cpu().numpy() for prediction in model_predictions]
        record = valid_store_raft.records[index]
        truth = np.asarray(record["eval_truth"], np.float64)
        truth_residual = np.asarray(record["truth_residual"], np.float64)
        eval_mask = np.asarray(record["eval_mask"], np.float64) > 0.5

        for iteration, correction in enumerate(predictions):
            prediction = _s3_row_prediction(record, correction)
            error = prediction - truth
            iteration_errors[iteration].append(error)
            iteration_bywell.append(
                {
                    "well": record["well"],
                    "rows": len(truth),
                    "iteration": iteration,
                    "rmse": float(np.sqrt(np.mean(error**2))),
                }
            )
            residual_gap = truth_residual - np.asarray(correction, np.float64)
            for radius in S3_REACHABILITY_FT:
                reachable_correction = np.asarray(correction, np.float64) + np.clip(
                    residual_gap, -float(radius), float(radius)
                )
                reachable_correction[~eval_mask] = 0.0
                reachable_prediction = _s3_row_prediction(
                    record, reachable_correction
                )
                reachable_error = reachable_prediction - truth
                reach_errors[(iteration, radius)].append(reachable_error)
                reach_rows.append(
                    {
                        "well": record["well"],
                        "rows": len(truth),
                        "iteration": iteration,
                        "radius_ft": radius,
                        "column_coverage": float(
                            np.mean(np.abs(residual_gap[eval_mask]) <= radius)
                        ),
                        "reachable_rmse": float(
                            np.sqrt(np.mean(reachable_error**2))
                        ),
                    }
                )

    bywell = pd.DataFrame(iteration_bywell)
    reach_bywell = pd.DataFrame(reach_rows)
    iteration_summary = []
    for iteration, errors in iteration_errors.items():
        error = np.concatenate(errors)
        frame = bywell[bywell["iteration"] == iteration]
        iteration_summary.append(
            {
                "iteration": iteration,
                "rmse": float(np.sqrt(np.mean(error**2))),
                "median_well_rmse": float(frame["rmse"].median()),
                "p75_well_rmse": float(frame["rmse"].quantile(0.75)),
                "p95_well_rmse": float(frame["rmse"].quantile(0.95)),
            }
        )
    iteration_summary = pd.DataFrame(iteration_summary)

    reach_summary = []
    for (iteration, radius), errors in reach_errors.items():
        error = np.concatenate(errors)
        frame = reach_bywell[
            (reach_bywell["iteration"] == iteration)
            & (reach_bywell["radius_ft"] == radius)
        ]
        reach_summary.append(
            {
                "iteration": iteration,
                "radius_ft": radius,
                "pooled_column_coverage": float(
                    np.average(frame["column_coverage"], weights=frame["rows"])
                ),
                "median_well_coverage": float(frame["column_coverage"].median()),
                "p10_well_coverage": float(frame["column_coverage"].quantile(0.10)),
                "reachable_oracle_rmse": float(np.sqrt(np.mean(error**2))),
            }
        )
    reach_summary = pd.DataFrame(reach_summary)
    iteration_summary.to_csv(
        OUT_DIR_REPAIR / "s3_iteration_summary.csv", index=False
    )
    bywell.to_csv(OUT_DIR_REPAIR / "s3_iteration_bywell.csv", index=False)
    reach_summary.to_csv(
        OUT_DIR_REPAIR / "s3_reachability_summary.csv", index=False
    )
    reach_bywell.to_csv(
        OUT_DIR_REPAIR / "s3_reachability_bywell.csv", index=False
    )
    print("\nS3 ITERATIONS")
    print(iteration_summary.to_string(index=False))
    print("\nS3 REACHABILITY")
    print(reach_summary.to_string(index=False))
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "best_iteration": int(
            iteration_summary.loc[iteration_summary["rmse"].idxmin(), "iteration"]
        ),
        "best_iteration_rmse": float(iteration_summary["rmse"].min()),
        "final_radius64_coverage": float(
            reach_summary[
                (reach_summary["iteration"] == RAFT_ITERATIONS)
                & (reach_summary["radius_ft"] == 64.0)
            ]["pooled_column_coverage"].iloc[0]
        ),
        "final_radius64_oracle_rmse": float(
            reach_summary[
                (reach_summary["iteration"] == RAFT_ITERATIONS)
                & (reach_summary["radius_ft"] == 64.0)
            ]["reachable_oracle_rmse"].iloc[0]
        ),
    }
    del model, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


started = time.time()
print(
    {
        "experiment": "s3_s5_zero_retrain_repair_audit_v1",
        "s3_checkpoint_dir": str(MODEL_DIR_RAFT),
        "s5_checkpoint_dir": str(S5_MODEL_DIR),
        "valid_wells": len(valid_store_raft.records),
        "device": str(device_raft),
        "training": False,
    },
    flush=True,
)

s5_result = _run_s5_audit()
s3_result = _run_s3_audit()

verdict = {
    "experiment": "s3_s5_zero_retrain_repair_audit_v1",
    "training": False,
    "validation": "existing fold 1; diagnostic temperature/menu scans are not confirmatory",
    "s5": s5_result,
    "s3": s3_result,
    "strict_pass": False,
    "next": (
        "freeze the best repair and confirm it on a new split/nested OOF; "
        "use S3 multi-start only if the reachability audit shows a global-mode miss"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with open(OUT_DIR_REPAIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR_REPAIR)
