"""Recover Fold-1 feature frames retained by the preceding Jupyter traceback."""

import gc
import json
import sys

import numpy as np
import pandas as pd


required = {
    "_bootstrap_notebook",
    "NOTEBOOK_PATH",
    "FLE_ROOT",
    "_fit_fle_fold",
    "make_prediction",
}
missing = sorted(required.difference(globals()))
if missing:
    raise RuntimeError(
        f"Missing live globals {missing}. Run this immediately after the NameError, "
        "without restarting the kernel."
    )

if "_device" not in globals() or "lgb_configs" not in globals():
    source = "".join(_bootstrap_notebook["cells"][35].get("source", []))
    source = source.replace("cache=True", "cache=False")
    exec(compile(source, f"{NOTEBOOK_PATH}::cell_35", "exec"), globals())

traceback = getattr(sys, "last_traceback", None)
fit_frame = None
while traceback is not None:
    if traceback.tb_frame.f_code.co_name == "_fit_fle_fold":
        fit_frame = traceback.tb_frame
        break
    traceback = traceback.tb_next

if fit_frame is None:
    raise RuntimeError(
        "The previous _fit_fle_fold traceback is no longer available. "
        "Use the corrected main cell; Fold-1 features will have to be rebuilt."
    )

local = fit_frame.f_locals
outer_fold = int(local["outer_fold"])
train = local["train"]
valid = local["valid"]
columns = list(local["columns"])
if outer_fold != 1:
    raise RuntimeError(f"Expected retained Fold-1 frames, found Fold-{outer_fold}")
if train.empty or valid.empty or len(columns) < 100:
    raise RuntimeError(
        f"Invalid retained frames: train={len(train)}, valid={len(valid)}, "
        f"features={len(columns)}"
    )

schema_path = FLE_ROOT / "fold1_features.json"
schema_path.write_text(json.dumps(columns, indent=2), encoding="utf-8")
print(
    "RECOVERED FOLD-1 FEATURES",
    {
        "train_rows": len(train),
        "valid_rows": len(valid),
        "train_wells": int(train["well"].nunique()),
        "valid_wells": int(valid["well"].nunique()),
        "features": len(columns),
    },
    flush=True,
)

learned_delta, model_reports = _fit_fle_fold(1, train, valid, columns)
prediction = make_prediction(valid, learned_delta, None)
output = pd.DataFrame(
    {
        "id": valid["id"].astype(str).to_numpy(),
        "well": valid["well"].astype(str).to_numpy(),
        "row_idx": valid["row_idx"].to_numpy(np.int32),
        "fold": np.int8(1),
        "tvt_true": (
            valid["last_known_tvt"].to_numpy(np.float64)
            + valid["target"].to_numpy(np.float64)
        ),
        "fle_gs13": prediction,
        "learned_delta": learned_delta,
        "likpf_scale_5": valid["likpf_scale_5"].to_numpy(np.float32),
    }
)
output_path = FLE_ROOT / "fold1_oof.pkl"
output.to_pickle(output_path)
(FLE_ROOT / "fold1_report.json").write_text(
    json.dumps(
        {
            "outer_fold": 1,
            "recovered_from_traceback": True,
            "train_wells": int(train["well"].nunique()),
            "valid_wells": int(valid["well"].nunique()),
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "features": len(columns),
            "models": model_reports,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(
    "FOLD-1 RECOVERY COMPLETE",
    {"output": str(output_path), "rows": len(output), "models": len(model_reports)},
    flush=True,
)

del train, valid, learned_delta, prediction, output
sys.last_traceback = None
gc.collect()
