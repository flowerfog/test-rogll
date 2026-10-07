from pathlib import Path
import gc
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch


# Locate the uploaded source folder. No fixed Kaggle dataset slug is assumed.
project_candidates = []
for marker in Path("/kaggle/input").rglob("first_place_repro/submit_single.py"):
    candidate = marker.parent.parent
    if (candidate / "pyproject.toml").is_file():
        project_candidates.append(candidate)
project_candidates = list(dict.fromkeys(project_candidates))
if len(project_candidates) != 1:
    raise RuntimeError({
        "message": "expected exactly one single-model source folder",
        "found": [str(path) for path in project_candidates],
    })

source_project = project_candidates[0]
runtime = Path("/kaggle/working/first_place_single_runtime")
if runtime.exists():
    shutil.rmtree(runtime)
shutil.copytree(source_project, runtime)
project = runtime
assert (project / "first_place_repro" / "submit_single.py").is_file()


# Locate one full-fold 0719 v11 checkpoint. The hidden test size is irrelevant
# to this model audit; only train-time checkpoint metadata is checked.
expected_contract = "0719_contract_v11_pageable_activation_offload"
valid_checkpoints = []
checkpoint_audit = []
for path in Path("/kaggle/input").rglob("*.pt"):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        checkpoint_audit.append({"path": str(path), "error": repr(error)})
        continue
    if not isinstance(checkpoint, dict):
        del checkpoint
        continue
    metrics = checkpoint.get("metrics", {})
    report = {
        "path": str(path),
        "version": checkpoint.get("version"),
        "contract_version": checkpoint.get("contract_version"),
        "epoch": checkpoint.get("epoch"),
        "selected_model": checkpoint.get("selected_model"),
        "row_rmse": metrics.get("row_rmse"),
        "validation_rows": metrics.get("rows"),
    }
    checkpoint_audit.append(report)
    if (
        checkpoint.get("version") == "0719_V1"
        and checkpoint.get("contract_version") == expected_contract
        and "model" in checkpoint
        and int(metrics.get("rows", 0)) >= 500_000
        and np.isfinite(float(metrics.get("row_rmse", np.nan)))
    ):
        valid_checkpoints.append(report)
    del checkpoint
    gc.collect()
if len(valid_checkpoints) != 1:
    raise RuntimeError({
        "message": "expected exactly one full-fold 0719 v11 best.pt",
        "valid": valid_checkpoints,
        "audit": checkpoint_audit,
    })
checkpoint_report = valid_checkpoints[0]
checkpoint_path = Path(checkpoint_report["path"])


# Locate the competition root from its data contract, not its mounted slug.
data_matches = []
for sample_path in Path("/kaggle/input").rglob("sample_submission.csv"):
    candidate = sample_path.parent
    if (candidate / "test").is_dir():
        data_matches.append(candidate)
data_matches = list(dict.fromkeys(data_matches))
if len(data_matches) != 1:
    raise RuntimeError({
        "message": "expected exactly one competition data root",
        "found": [str(path) for path in data_matches],
    })
data_root = data_matches[0]
sample_path = data_root / "sample_submission.csv"
official = pd.read_csv(sample_path, dtype={"id": str})
if list(official.columns) != ["id", "tvt"] or official["id"].duplicated().any():
    raise RuntimeError("invalid hidden sample_submission contract")


# Run directly from the copied source. Avoid pip/network activity during the
# competition rerun.
output = Path("/kaggle/working/submission.csv")
command = [
    sys.executable,
    "-u",
    "-m",
    "first_place_repro.submit_single",
    "--data-root",
    str(data_root),
    "--checkpoint",
    str(checkpoint_path),
    "--sample-submission",
    str(sample_path),
    "--output",
    str(output),
]
environment = os.environ.copy()
environment["PYTHONPATH"] = str(project) + os.pathsep + environment.get("PYTHONPATH", "")
print("RERUN CONTRACT", json.dumps({
    "source_project": str(source_project),
    "checkpoint": checkpoint_report,
    "data_root": str(data_root),
    "sample_rows": int(len(official)),
    "sample_wells": int(official["id"].str.rsplit("_", n=1).str[0].nunique()),
}), flush=True)
print("RUN:", " ".join(command), flush=True)
subprocess.run(command, check=True, cwd=project, env=environment)


# Hidden-rerun-safe validation: compare to the mounted sample dynamically.
submission = pd.read_csv(output, dtype={"id": str})
if list(submission.columns) != ["id", "tvt"]:
    raise RuntimeError(f"wrong submission columns: {list(submission.columns)}")
if len(submission) != len(official):
    raise RuntimeError({"submission_rows": len(submission), "sample_rows": len(official)})
if submission["id"].tolist() != official["id"].tolist():
    raise RuntimeError("submission IDs/order differ from hidden sample_submission")
values = submission["tvt"].to_numpy(dtype=np.float64)
if not np.isfinite(values).all():
    raise RuntimeError("submission contains non-finite predictions")

audit_path = output.with_suffix(".audit.json")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("members") != 1 or audit.get("version") != "0719_V1":
    raise RuntimeError(f"unexpected inference audit: {audit}")
if int(audit.get("submission_rows", -1)) != len(official):
    raise RuntimeError("audit row count differs from hidden sample")

print("SUBMISSION READY", json.dumps({
    "path": str(output),
    "rows": int(len(submission)),
    "wells": int(submission["id"].str.rsplit("_", n=1).str[0].nunique()),
    "tvt_min": float(values.min()),
    "tvt_mean": float(values.mean()),
    "tvt_max": float(values.max()),
    "sha256": audit["submission_sha256"],
}), flush=True)
