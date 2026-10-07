"""V147: Fold4/5 deep candidate-bank support audit.

This reuses the audited V145 engine and changes only the held-out folds and
support depths.  The result is an oracle ceiling audit, never a deployable
prediction.
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
    'EXPERIMENT = "fold45_candidate_bank_depth_audit_v147"',
    1,
)
source = source.replace("FOLDS = (1, 2, 3, 4, 5)", "FOLDS = (4, 5)", 1)
source = source.replace(
    "TOPK_VALUES = (1, 4, 8, 16, 32, 64)",
    "TOPK_VALUES = (1, 64, 128, 256, 512, 1024)",
    1,
)
source = source.replace("V145 MATCHED SUMMARY", "V147 MATCHED SUMMARY")
source = source.replace("V145 FINAL VERDICT", "V147 FINAL VERDICT")
source = source.replace("V145 candidate oracle", "V147 candidate oracle")
source = source.replace(
    'if fold_counts != {1: 155, 2: 155, 3: 154, 4: 155, 5: 154}:',
    'if fold_counts != {4: 155, 5: 154}:',
    1,
)
source = source.replace(
    'frame["well"] = frame["well"].astype(str)\nrequired_columns = {',
    'frame["well"] = frame["well"].astype(str)\n'
    'frame = frame[frame["fold"].isin(FOLDS)].copy()\n'
    'required_columns = {',
    1,
)
source = source.replace(
    'if len(frame) != 3_783_989 or frame["well"].nunique() != 773 or frame["id"].duplicated().any():',
    'if len(frame) != 1_513_346 or frame["well"].nunique() != 309 or frame["id"].duplicated().any():',
    1,
)
source = source.replace(
    '"coverage_exact": len(matched) == 3_783_989 and matched["well"].nunique() == 773,',
    '"coverage_exact": len(matched) == 1_513_346 and matched["well"].nunique() == 309,',
    1,
)
source = source.replace(
    '"top64_oracle_rmse_lt_6": oracle_values["64"]["rmse"] < 6.0,',
    '"top1024_oracle_rmse_lt_6": oracle_values["1024"]["rmse"] < 6.0,',
)
source = source.replace(
    '"top64_oracle_each_fold_lt_6p5": all(value < 6.5 for value in top64_fold_rmse.values()),',
    '"top1024_oracle_each_fold_lt_6p5": all(value < 6.5 for value in top1024_fold_rmse.values()),',
)
source = source.replace(
    '"top32_oracle_gain_vs_v144_ge_0p50": oracle_values["32"]["gain_vs_v144"] >= 0.50,',
    '"top256_oracle_gain_vs_v144_ge_0p50": oracle_values["256"]["gain_vs_v144"] >= 0.50,',
)
source = source.replace("top64_fold_rmse = {", "top1024_fold_rmse = {")
source = source.replace(
    '"blendaware_oracle_top64"),\n            "rmse",\n        ].iloc[0]',
    '"blendaware_oracle_top1024"),\n            "rmse",\n        ].iloc[0]',
)
source = source.replace('"top64_fold_rmse": top64_fold_rmse,', '"top1024_fold_rmse": top1024_fold_rmse,')
source = source.replace(
    '"best_rank_top64": best64 + 1,\n            "best_top64_rmse": float(np.sqrt(cumulative_best[-1] / len(local))),',
    '"best_rank_top1024": best64 + 1,\n            "best_top1024_rmse": float(np.sqrt(cumulative_best[-1] / len(local))),',
)
source = source.replace(
    '"fold123": "prior45 plus frozen Fold45 ranker",\n        "fold45": "prior123 plus leave-one-fold-out rankers",',
    '"fold45": "prior123 plus leave-one-fold-out rankers",',
)
code = compile(source, str(SOURCE), "exec")
exec(code, {"__name__": "__main__", "__file__": str(SOURCE)})
