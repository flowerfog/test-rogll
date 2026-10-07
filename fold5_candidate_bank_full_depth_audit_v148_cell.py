"""V148: Fold5 deep/full candidate-bank ceiling audit.

This is a read-only oracle audit built on the V145 engine.  It isolates the
worst canonical fold and extends support beyond V147's top1024.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = ROOT / "full5_blendaware_candidate_support_oracle_v145_cell.py"
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

source = SOURCE.read_text(encoding="utf-8")
source = source.replace(
    'EXPERIMENT = "full5_blendaware_candidate_support_oracle_v145"',
    'EXPERIMENT = "fold5_candidate_bank_full_depth_audit_v148"',
    1,
)
source = source.replace("FOLDS = (1, 2, 3, 4, 5)", "FOLDS = (5,)", 1)
source = source.replace(
    "TOPK_VALUES = (1, 4, 8, 16, 32, 64)",
    "TOPK_VALUES = (1, 512, 1024, 1536, 2048)",
    1,
)
source = source.replace("V145 MATCHED SUMMARY", "V148 MATCHED SUMMARY")
source = source.replace("V145 FINAL VERDICT", "V148 FINAL VERDICT")
source = source.replace("V145 candidate oracle", "V148 candidate oracle")
source = source.replace(
    'frame["well"] = frame["well"].astype(str)\nrequired_columns = {',
    'frame["well"] = frame["well"].astype(str)\n'
    'frame = frame[frame["fold"].isin(FOLDS)].copy()\n'
    'required_columns = {',
    1,
)
source = source.replace(
    'if fold_counts != {1: 155, 2: 155, 3: 154, 4: 155, 5: 154}:',
    'if fold_counts != {5: 154}:',
    1,
)
source = source.replace(
    'if len(frame) != 3_783_989 or frame["well"].nunique() != 773 or frame["id"].duplicated().any():',
    'if len(frame) != 756_245 or frame["well"].nunique() != 154 or frame["id"].duplicated().any():',
    1,
)
source = source.replace(
    '"coverage_exact": len(matched) == 3_783_989 and matched["well"].nunique() == 773,',
    '"coverage_exact": len(matched) == 756_245 and matched["well"].nunique() == 154,',
    1,
)
source = source.replace(
    '"top64_oracle_rmse_lt_6": oracle_values["64"]["rmse"] < 6.0,',
    '"top2048_oracle_rmse_lt_6": oracle_values["2048"]["rmse"] < 6.0,',
)
source = source.replace(
    '"top64_oracle_each_fold_lt_6p5": all(value < 6.5 for value in top64_fold_rmse.values()),',
    '"top2048_oracle_rmse_lt_6p5": oracle_values["2048"]["rmse"] < 6.5,',
)
source = source.replace(
    '"top32_oracle_gain_vs_v144_ge_0p50": oracle_values["32"]["gain_vs_v144"] >= 0.50,',
    '"top1024_to_top2048_gain_ge_0p10": (\n'
    '        oracle_values["1024"]["rmse"] - oracle_values["2048"]["rmse"] >= 0.10\n'
    '    ),',
)
source = source.replace("top64_fold_rmse = {", "top2048_fold_rmse = {")
source = source.replace(
    '"blendaware_oracle_top64"),\n            "rmse",\n        ].iloc[0]',
    '"blendaware_oracle_top2048"),\n            "rmse",\n        ].iloc[0]',
)
source = source.replace('"top64_fold_rmse": top64_fold_rmse,', '"top2048_fold_rmse": top2048_fold_rmse,')
source = source.replace(
    '"best_rank_top64": best64 + 1,\n            "best_top64_rmse": float(np.sqrt(cumulative_best[-1] / len(local))),',
    '"best_rank_top2048": best64 + 1,\n            "best_top2048_rmse": float(np.sqrt(cumulative_best[-1] / len(local))),',
)
source = source.replace(
    '"fold123": "prior45 plus frozen Fold45 ranker",\n        "fold45": "prior123 plus leave-one-fold-out rankers",',
    '"fold45": "prior123 plus leave-one-fold-out rankers",',
)
source = source.replace(
    'if len(order) < MAX_TOPK:\n            raise RuntimeError(f"only {len(order)} unique candidates")',
    'if len(order) < 1:\n            raise RuntimeError("no unique candidates")',
)
source = source.replace(
    'candidate_sse = np.empty(MAX_TOPK, np.float64)\n        candidate_predictions = []\n        for rank, index in enumerate(order[:MAX_TOPK]):',
    'available_max = min(MAX_TOPK, len(order))\n'
    '        candidate_sse = np.empty(available_max, np.float64)\n'
    '        candidate_predictions = []\n'
    '        for rank, index in enumerate(order[:available_max]):',
)
source = source.replace(
    'best_index = int(np.argmin(candidate_sse[:topk]))\n            predictions[topk] = candidate_predictions[best_index]\n            report[f"best_rank_top{topk}"] = best_index + 1',
    'effective_topk = min(topk, available_max)\n'
    '            best_index = int(np.argmin(candidate_sse[:effective_topk]))\n'
    '            predictions[topk] = candidate_predictions[best_index]\n'
    '            report[f"best_rank_top{topk}"] = best_index + 1\n'
    '            report[f"available_top{topk}"] = effective_topk',
)
source = source.replace(
    '"unique_candidates": int(len(order)),\n            "best_rank_top64": best64 + 1,',
    '"unique_candidates": int(len(order)),\n            "available_candidates_evaluated": int(available_max),\n            "best_rank_top64": best64 + 1,',
)
code = compile(source, str(SOURCE), "exec")
exec(code, {"__name__": "__main__", "__file__": str(SOURCE)})
