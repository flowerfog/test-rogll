"""Resume V202 at its once-only Fold2 gateway in the current live kernel.

Run this only after the original V202 cell reached Fold2 inference and failed
on the old truth-filtered block-node guard. It reuses the trained model and
frozen Fold3 decoder; no training or model selection is repeated.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch


required = (
    "model", "gate_wells", "_load_record", "_inputs", "device",
    "selected_decoder", "STATE_OFFSETS", "BLOCK_FT",
)
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(
        "The trained V202 kernel state is unavailable. Rerun the updated full "
        f"V202 script instead. Missing: {missing}"
    )


def _block_nodes(record):
    """Create a complete, truth-independent MD block chain."""
    eval_rows = record.eval_index.astype(np.int64)
    distance = record.md[eval_rows].astype(np.float64) - float(record.md[record.anchor_index])
    block_id = np.floor(np.maximum(distance, 0.0) / BLOCK_FT).astype(np.int64)
    rows = []
    for _, positions in pd.Series(np.arange(len(eval_rows))).groupby(block_id, sort=True):
        local = eval_rows[positions.to_numpy(np.int64)]
        rows.append(int(local[len(local) // 2]))
    rows = np.asarray(rows, np.int64)
    if len(rows) < 2:
        raise RuntimeError(f"{record.well}: only {len(rows)} MD-defined block nodes")
    label = record.truth[rows].astype(np.float32) - float(record.anchor_tvt)
    md = record.md[rows].astype(np.float32) - float(record.md[record.anchor_index])
    z = record.condition[4, rows].astype(np.float32) * 512.0
    return rows, label, md, z


@torch.no_grad()
def _logits_for_records(model, records, device, mode="full"):
    model.eval()
    output = []
    for index, record in enumerate(records):
        rows, label, md, z = _block_nodes(record)
        donor = (
            records[(index + max(1, len(records) // 2)) % len(records)]
            if mode == "shuffled_typewell" else None
        )
        logits = model(
            *_inputs(record, rows, device, mode=mode, donor=donor)
        )[0].float().cpu().numpy()
        output.append({
            "record": record,
            "rows": rows,
            "label": label,
            "md": md,
            "z": z,
            "logits": logits,
        })
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"V202 inference {mode} {index + 1}/{len(records)}", flush=True)
    return output


source_candidates = (
    Path("/root/autodl-tmp/block_correlation_jump_posterior_v202_cell.py"),
    Path.cwd() / "block_correlation_jump_posterior_v202_cell.py",
)
source_path = next((path for path in source_candidates if path.is_file()), None)
if source_path is None:
    raise FileNotFoundError("Upload the updated block_correlation_jump_posterior_v202_cell.py")

source = source_path.read_text(encoding="utf-8")
marker = "gate_records = [_load_record(well) for well in gate_wells]"
if marker not in source:
    raise RuntimeError(f"Fold2 resume marker not found in {source_path}")

print("V202 RESUME: reusing trained model and frozen decoder; Fold2 inference only", flush=True)
tail = marker + source.split(marker, 1)[1]
exec(compile(tail, str(source_path) + "<fold2-resume>", "exec"), globals())
