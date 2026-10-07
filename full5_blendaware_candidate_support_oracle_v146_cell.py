"""V146: extend the V145 support ceiling audit to top128/top256.

The V145 implementation is kept byte-for-byte as the audited engine.  This
wrapper changes only the cumulative support sizes and output experiment name;
it does not alter candidate generation, rankers, dose, taper, cap, or metric
logic.
"""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = ROOT / "full5_blendaware_candidate_support_oracle_v145_cell.py"
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

source = SOURCE.read_text(encoding="utf-8")
source = source.replace(
    'EXPERIMENT = "full5_blendaware_candidate_support_oracle_v145"',
    'EXPERIMENT = "full5_blendaware_candidate_support_oracle_v146"',
    1,
)
source = source.replace(
    "TOPK_VALUES = (1, 4, 8, 16, 32, 64)",
    "TOPK_VALUES = (1, 4, 8, 16, 32, 64, 128, 256)",
    1,
)
source = source.replace("V145 MATCHED SUMMARY", "V146 MATCHED SUMMARY")
source = source.replace("V145 FINAL VERDICT", "V146 FINAL VERDICT")
source = source.replace("V145 candidate oracle", "V146 candidate oracle")
source = source.replace(
    '"top64_oracle_rmse_lt_6": oracle_values["64"]["rmse"] < 6.0,',
    '"top256_oracle_rmse_lt_6": oracle_values["256"]["rmse"] < 6.0,',
)
source = source.replace(
    '"top64_oracle_each_fold_lt_6p5": all(value < 6.5 for value in top64_fold_rmse.values()),',
    '"top256_oracle_each_fold_lt_6p5": all(value < 6.5 for value in top256_fold_rmse.values()),',
)
source = source.replace(
    'top64_fold_rmse = {',
    'top256_fold_rmse = {',
)
source = source.replace(
    '"blendaware_oracle_top64"),\n            "rmse",\n        ].iloc[0]',
    '"blendaware_oracle_top256"),\n            "rmse",\n        ].iloc[0]',
)
source = source.replace(
    '"top64_fold_rmse": top64_fold_rmse,',
    '"top256_fold_rmse": top256_fold_rmse,',
)
source = source.replace(
    '"best_rank_top64": best64 + 1,\n            "best_top64_rmse": float(np.sqrt(cumulative_best[-1] / len(local))),',
    '"best_rank_top256": best64 + 1,\n            "best_top256_rmse": float(np.sqrt(cumulative_best[-1] / len(local))),',
)
code = compile(source, str(SOURCE), "exec")
exec(code, {"__name__": "__main__", "__file__": str(SOURCE)})
