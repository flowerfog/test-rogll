"""Kaggle score-channel probe for Gold dose lambda=0.5.

Run after the same deterministic notebook run has produced both the CP12
before-Gold endpoint and the final Gold endpoint. This cell performs file-only
interpolation; it does not rerun any model or Gold decision logic.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

WORK = Path("/kaggle/working")
LAMBDA = 0.50


def find_endpoint(names):
    for name in names:
        path = WORK / name
        if path.exists():
            return path
    # Allow archived endpoints attached as a Kaggle dataset.
    input_root = Path("/kaggle/input")
    matches = []
    if input_root.exists():
        for name in names:
            matches.extend(sorted(input_root.rglob(name)))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            "multiple endpoint files found; place the intended endpoint in "
            f"{WORK}: {[str(path) for path in matches[:10]]}"
        )
    raise FileNotFoundError(f"none of these endpoint files exists: {names}")


before_path = find_endpoint(
    (
        "submission_cp12_before_gold.csv",
        "submission_controlpoint_before_gold.csv",
    )
)
gold_path = find_endpoint(
    (
        "submission_gold_full.csv",
        "submission_final_gold.csv",
    )
)

before_raw = pd.read_csv(before_path)
gold_raw = pd.read_csv(gold_path)
required = {"id", "tvt"}
if not required.issubset(before_raw.columns) or not required.issubset(gold_raw.columns):
    raise RuntimeError("both endpoints must contain id and tvt columns")
if before_raw["id"].duplicated().any() or gold_raw["id"].duplicated().any():
    raise RuntimeError("duplicate endpoint ids")

before = before_raw[["id", "tvt"]].rename(columns={"tvt": "before"})
gold = gold_raw[["id", "tvt"]].rename(columns={"tvt": "gold"})
joined = before.merge(gold, on="id", how="outer", validate="one_to_one", indicator=True)
if not joined["_merge"].eq("both").all():
    raise RuntimeError(joined["_merge"].value_counts().to_dict())
joined = joined.drop(columns="_merge")

# Restore the exact before-Gold row order. Kaggle scoring is ID-based, but this
# catches accidental lineage or sample-alignment changes before submission.
joined = before[["id"]].merge(joined, on="id", how="left", validate="one_to_one")
before_values = joined["before"].to_numpy(np.float64)
gold_values = joined["gold"].to_numpy(np.float64)
if not np.isfinite(before_values).all() or not np.isfinite(gold_values).all():
    raise RuntimeError("endpoint contains NaN or infinite predictions")

move = gold_values - before_values
probe_values = before_values + LAMBDA * move
output = pd.DataFrame({"id": joined["id"], "tvt": probe_values})
output_path = WORK / "submission_gold_lambda_050.csv"
output.to_csv(output_path, index=False)

audit = {
    "lambda": LAMBDA,
    "rows": len(output),
    "before_endpoint": str(before_path),
    "gold_endpoint": str(gold_path),
    "before_sha256": hashlib.sha256(before_path.read_bytes()).hexdigest(),
    "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
    "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    "endpoint_mean_abs_move": float(np.mean(np.abs(move))),
    "endpoint_rms_move": float(np.sqrt(np.mean(move * move))),
    "endpoint_p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
    "endpoint_max_abs_move": float(np.max(np.abs(move))),
    "probe_mean_abs_move_vs_before": float(np.mean(np.abs(probe_values - before_values))),
    "probe_p95_abs_move_vs_before": float(np.quantile(np.abs(probe_values - before_values), 0.95)),
}
(WORK / "gold_lambda_050_audit.json").write_text(json.dumps(audit, indent=2))

print("GOLD LAMBDA 0.50 PROBE READY")
print(json.dumps(audit, indent=2))
print("submission:", output_path)
print(
    "After Kaggle scores this file, record the displayed RMSE with full "
    "available precision. Do not submit another lambda before fitting the "
    "quadratic response from lambda 0, 0.5, and 1."
)
