"""Read-only support/rank audit for V202 Fold2 posterior logits.

Run in the same live kernel after V202 (or its Fold2 resume) completed. This
does not train, select a deployable recipe, or touch Fold1.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


started_v202b = time.time()
required_v202b = (
    "full_items", "shuffled_items", "STATE_OFFSETS", "_row_prediction",
    "_metrics", "OUT",
)
missing_v202b = [name for name in required_v202b if name not in globals()]
if missing_v202b:
    raise RuntimeError(
        "Run V202 Fold2 inference in this kernel first. Missing: "
        + repr(missing_v202b)
    )

OUT_V202B = Path(OUT).parent / "block_posterior_support_audit_v202b"
OUT_V202B.mkdir(parents=True, exist_ok=True)
TOPK_V202B = (1, 2, 4, 8, 16, 32, 64, 128, 257)


def _oracle_parts_v202b(items, topk):
    parts = []
    for item in items:
        record = item["record"]
        logits = item["logits"]
        k = min(int(topk), logits.shape[0])
        if k == logits.shape[0]:
            top = np.broadcast_to(
                np.arange(logits.shape[0], dtype=np.int64)[:, None], logits.shape
            )
        else:
            top = np.argpartition(logits, -k, axis=0)[-k:]
        candidates = STATE_OFFSETS[top]
        nearest = np.argmin(np.abs(candidates - item["label"][None, :]), axis=0)
        offsets = candidates[nearest, np.arange(candidates.shape[1])]
        prediction = _row_prediction(record, item["rows"], offsets)
        eval_rows = record.eval_index
        parts.append({
            "well": record.well,
            "row_idx": eval_rows.astype(np.int64),
            "truth": record.truth[eval_rows].astype(np.float64),
            "prediction": prediction,
            "distance": (
                record.md[eval_rows].astype(np.float64)
                - float(record.md[record.anchor_index])
            ),
        })
    return parts


def _rank_rows_v202b(aligned, shuffled):
    rows = []
    for item, shuffled_item in zip(aligned, shuffled):
        if item["record"].well != shuffled_item["record"].well:
            raise RuntimeError("V202 aligned/shuffled item order mismatch")
        logits = item["logits"].astype(np.float64)
        shuffled_logits = shuffled_item["logits"].astype(np.float64)
        label = item["label"].astype(np.float64)
        nearest = np.argmin(np.abs(STATE_OFFSETS[:, None] - label[None, :]), axis=0)
        column = np.arange(logits.shape[1])
        truth_score = logits[nearest, column]
        shuffled_truth_score = shuffled_logits[nearest, column]
        rank = 1 + np.sum(logits > truth_score[None, :], axis=0)
        probability_logits = logits - np.max(logits, axis=0, keepdims=True)
        probability = np.exp(probability_logits)
        probability /= np.sum(probability, axis=0, keepdims=True)
        entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=0)
        posterior_mean = STATE_OFFSETS.astype(np.float64) @ probability
        for index in range(len(label)):
            rows.append({
                "well": item["record"].well,
                "block": index,
                "md_since": float(item["md"][index]),
                "label": float(label[index]),
                "in_axis": bool(STATE_OFFSETS[0] <= label[index] <= STATE_OFFSETS[-1]),
                "truth_state_rank": int(rank[index]),
                "truth_score": float(truth_score[index]),
                "shuffled_truth_score": float(shuffled_truth_score[index]),
                "aligned_truth_score_increment": float(
                    truth_score[index] - shuffled_truth_score[index]
                ),
                "posterior_entropy": float(entropy[index]),
                "posterior_mean_error": float(posterior_mean[index] - label[index]),
            })
    return pd.DataFrame(rows)


oracle_rows_v202b = []
for topk_v202b in TOPK_V202B:
    metrics_v202b = _metrics(
        _oracle_parts_v202b(full_items, topk_v202b),
        f"posterior_oracle_top{topk_v202b}",
    )
    oracle_rows_v202b.append({"topk": topk_v202b, **metrics_v202b})
    print(
        "V202b oracle",
        {"topk": topk_v202b, "rmse": metrics_v202b["rmse"]},
        flush=True,
    )

oracle_v202b = pd.DataFrame(oracle_rows_v202b)
rank_v202b = _rank_rows_v202b(full_items, shuffled_items)
rank_v202b["distance_band"] = pd.cut(
    rank_v202b["md_since"],
    [-np.inf, 500.0, 1000.0, 2000.0, np.inf],
    labels=["0_500", "500_1000", "1000_2000", "gt2000"],
)

rank_summary_v202b = rank_v202b.groupby("distance_band", observed=True).agg(
    blocks=("truth_state_rank", "size"),
    in_axis_fraction=("in_axis", "mean"),
    median_rank=("truth_state_rank", "median"),
    p75_rank=("truth_state_rank", lambda values: np.quantile(values, 0.75)),
    p90_rank=("truth_state_rank", lambda values: np.quantile(values, 0.90)),
    top8_recall=("truth_state_rank", lambda values: np.mean(values <= 8)),
    top32_recall=("truth_state_rank", lambda values: np.mean(values <= 32)),
    top64_recall=("truth_state_rank", lambda values: np.mean(values <= 64)),
    mean_truth_score_increment=("aligned_truth_score_increment", "mean"),
    median_entropy=("posterior_entropy", "median"),
    posterior_mean_bias=("posterior_mean_error", "mean"),
).reset_index()

overall_v202b = {
    "blocks": len(rank_v202b),
    "in_axis_fraction": float(rank_v202b["in_axis"].mean()),
    "median_truth_rank": float(rank_v202b["truth_state_rank"].median()),
    "top8_recall": float((rank_v202b["truth_state_rank"] <= 8).mean()),
    "top32_recall": float((rank_v202b["truth_state_rank"] <= 32).mean()),
    "top64_recall": float((rank_v202b["truth_state_rank"] <= 64).mean()),
    "mean_aligned_truth_score_increment": float(
        rank_v202b["aligned_truth_score_increment"].mean()
    ),
    "positive_aligned_truth_score_increment_fraction": float(
        (rank_v202b["aligned_truth_score_increment"] > 0).mean()
    ),
}
oracle_lookup_v202b = oracle_v202b.set_index("topk")["rmse"].to_dict()
checks_v202b = {
    "top32_oracle_rmse_le_7": bool(oracle_lookup_v202b[32] <= 7.0),
    "top64_oracle_rmse_le_6p5": bool(oracle_lookup_v202b[64] <= 6.5),
    "top64_recall_ge_0p70": bool(overall_v202b["top64_recall"] >= 0.70),
    "aligned_truth_score_increment_positive": bool(
        overall_v202b["mean_aligned_truth_score_increment"] > 0.0
    ),
}
retain_cost_map_v202b = bool(all(checks_v202b.values()))
verdict_v202b = {
    "experiment": "block_posterior_support_audit_v202b",
    "stage": "read_only_Fold2_posterior_support_and_rank_audit",
    "selection_or_training_count": 0,
    "fold1_touched": 0,
    "overall": overall_v202b,
    "oracle_rmse": {str(key): float(value) for key, value in oracle_lookup_v202b.items()},
    "checks": checks_v202b,
    "retain_cost_map": retain_cost_map_v202b,
    "elapsed_minutes": (time.time() - started_v202b) / 60.0,
    "next": (
        "PASS -> retain V202 evidence and replace posterior calibration/ranking"
        if retain_cost_map_v202b
        else "FAIL -> posterior support is inadequate; replace candidate evidence construction"
    ),
}

oracle_v202b.to_csv(OUT_V202B / "oracle_by_topk.csv", index=False)
rank_v202b.to_parquet(OUT_V202B / "block_rank_detail.parquet", index=False)
rank_summary_v202b.to_csv(OUT_V202B / "rank_by_distance.csv", index=False)
(OUT_V202B / "verdict.json").write_text(
    json.dumps(verdict_v202b, indent=2, allow_nan=False), encoding="utf-8"
)

print("\nV202b ORACLE SUPPORT")
print(oracle_v202b[["topk", "rmse", "median_well_rmse", "p95_well_rmse", "far_gt2000_rmse"]].to_string(index=False))
print("\nV202b RANK BY DISTANCE")
print(rank_summary_v202b.to_string(index=False))
print("\nV202b FINAL VERDICT")
print(json.dumps(verdict_v202b, indent=2, allow_nan=False), flush=True)
print("saved:", OUT_V202B, flush=True)
