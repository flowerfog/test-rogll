"""V172b: tie-safe read-only audit of V172 sequence scores.

Run immediately after V172 in the same kernel. V172 inherited candidate rows
that were originally appended in truth-SSE order. Stable sorting on a tied
model score therefore leaked oracle order into top-k selection. This audit
uses reference_well as the deterministic truth-independent tie breaker and
does not refit any model.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


STARTED_V172B = time.time()
ROOT_V172B = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OUT_V172B = ROOT_V172B / "oof_exports" / "historical_analog_forced_sequence_audit_v172b"
OUT_V172B.mkdir(parents=True, exist_ok=True)

required_v172b = (
    "panel_v172", "sample_v167", "records", "_transfer", "_metric",
    "model_rows_v172",
)
missing_v172b = [name for name in required_v172b if name not in globals()]
if missing_v172b:
    raise RuntimeError("Run V172 first. Missing: " + repr(missing_v172b))


def _clean_v172b(value):
    if isinstance(value, dict):
        return {str(key): _clean_v172b(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_v172b(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _lookup_v172b(score_column, oracle_k=None):
    ranked = panel_v172.sort_values(
        ["well", score_column, "reference_well"],
        ascending=[True, False, True],
        kind="stable",
    )
    if oracle_k is None:
        selected = ranked.groupby("well", sort=False).head(1)
    else:
        shortlist = ranked.groupby("well", sort=False).head(int(oracle_k))
        indices = shortlist.groupby("well", sort=False)["sse"].idxmin()
        selected = shortlist.loc[indices]
    return selected.set_index("well")["reference_well"].astype(str).to_dict()


def _expand_v172b(lookup):
    frames, predictions = [], []
    for well, group in sample_v167.groupby("well", sort=False):
        well = str(well)
        target = records[well]
        reference = records[str(lookup[well])]
        row_idx = group["row_idx"].to_numpy(np.int64)
        query = target["md"][row_idx] - target["md"][target["anchor"]]
        prediction, _ = _transfer(reference, target, query, target["z"][row_idx])
        frames.append(group)
        predictions.append(np.asarray(prediction, np.float64))
    return pd.concat(frames, ignore_index=True), np.concatenate(predictions)


frame_v172b, top1_v172b = _expand_v172b(_lookup_v172b("sequence_score"))
_, rolled_v172b = _expand_v172b(_lookup_v172b("rolled_sequence_score"))
_, oracle4_v172b = _expand_v172b(_lookup_v172b("sequence_score", 4))
_, oracle8_v172b = _expand_v172b(_lookup_v172b("sequence_score", 8))
_, oracle16_v172b = _expand_v172b(_lookup_v172b("sequence_score", 16))
base_v172b = frame_v172b["base"].to_numpy(np.float64)

summary_v172b = pd.DataFrame([
    _metric(frame_v172b, base_v172b, "base_current"),
    _metric(frame_v172b, top1_v172b, "tie_safe_sequence_top1"),
    _metric(frame_v172b, rolled_v172b, "tie_safe_rolled_top1"),
    _metric(frame_v172b, oracle4_v172b, "tie_safe_oracle_top4"),
    _metric(frame_v172b, oracle8_v172b, "tie_safe_oracle_top8"),
    _metric(frame_v172b, oracle16_v172b, "tie_safe_oracle_top16"),
])

score_stats_v172b = panel_v172.groupby("well", sort=False).agg(
    aligned_std=("sequence_score", "std"),
    rolled_std=("rolled_sequence_score", "std"),
    aligned_unique=("sequence_score", "nunique"),
    rolled_unique=("rolled_sequence_score", "nunique"),
).reset_index()
aligned_lookup_v172b = _lookup_v172b("sequence_score")
rolled_lookup_v172b = _lookup_v172b("rolled_sequence_score")
choice_change_v172b = float(np.mean([
    aligned_lookup_v172b[well] != rolled_lookup_v172b[well]
    for well in aligned_lookup_v172b
]))
indexed_v172b = summary_v172b.set_index("endpoint")
used_features_v172b = [int(row["used_features"]) for row in model_rows_v172]
checks_v172b = {
    "median_score_unique_ge_8": bool(score_stats_v172b["aligned_unique"].median() >= 8),
    "all_models_use_sequence_features": bool(min(used_features_v172b) > 0),
    "tie_safe_oracle_top8_rmse_le_7": bool(indexed_v172b.loc["tie_safe_oracle_top8", "rmse"] <= 7.0),
    "aligned_top1_beats_rolled_ge_0p20": bool(
        indexed_v172b.loc["tie_safe_rolled_top1", "rmse"]
        - indexed_v172b.loc["tie_safe_sequence_top1", "rmse"] >= 0.20
    ),
    "choice_change_fraction_ge_0p10": bool(choice_change_v172b >= 0.10),
}
verdict_v172b = {
    "experiment": "historical_analog_forced_sequence_audit_v172b",
    "stage": "truth_independent_tie_break_audit_of_frozen_V172_scores",
    "selection_or_training_count": 0,
    "median_aligned_score_std": float(score_stats_v172b["aligned_std"].median()),
    "median_aligned_unique_scores": float(score_stats_v172b["aligned_unique"].median()),
    "used_features_by_fold": used_features_v172b,
    "rolled_choice_change_fraction": choice_change_v172b,
    "checks": checks_v172b,
    "strict_pass": bool(all(checks_v172b.values())),
    "elapsed_minutes": (time.time() - STARTED_V172B) / 60.0,
    "next": (
        "PASS -> sequence observability is real; build a segmental decoder"
        if all(checks_v172b.values())
        else "FAIL -> V172 4.945 was ordering leakage; close analog retrieval"
    ),
}

summary_v172b.to_csv(OUT_V172B / "summary.csv", index=False)
score_stats_v172b.to_csv(OUT_V172B / "score_stats.csv", index=False)
(OUT_V172B / "verdict.json").write_text(
    json.dumps(_clean_v172b(verdict_v172b), indent=2, allow_nan=False), encoding="utf-8"
)
print("\nV172b TIE-SAFE SUMMARY")
print(summary_v172b.to_string(index=False), flush=True)
print("\nV172b FINAL VERDICT")
print(json.dumps(_clean_v172b(verdict_v172b), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_V172B, flush=True)
