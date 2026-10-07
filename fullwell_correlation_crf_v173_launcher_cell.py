"""Launch V173 Correlation-CRF from a fresh AutoDL notebook kernel.

Upload this file together with:
  * native_resolution_dualref_capacity_v0_cell.py
  * fullwell_correlation_crf_nested_gateway_v1_cell.py

Run this launcher as one notebook cell. It performs no argparse handling and
keeps the original nested Fold345 -> Fold2 -> untouched Fold1 protocol.
"""

from pathlib import Path
import shutil

import torch


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
NATIVE = ROOT / "native_resolution_dualref_capacity_v0_cell.py"
SOURCE = ROOT / "fullwell_correlation_crf_nested_gateway_v1_cell.py"

for required in (NATIVE, SOURCE):
    if not required.is_file():
        raise FileNotFoundError(required)
if not torch.cuda.is_available():
    raise RuntimeError("V173 requires a CUDA GPU")

free_gb = shutil.disk_usage(ROOT).free / 1024**3
if free_gb < 1.5:
    raise RuntimeError(
        f"V173 requires at least 1.5 GB free for checkpoints/results; found {free_gb:.2f} GB"
    )
ephemeral_cache = free_gb < 8.0

device = torch.cuda.get_device_properties(0)
print(
    "V173 LAUNCH",
    {
        "gpu": device.name,
        "gpu_memory_gb": round(device.total_memory / 1024**3, 2),
        "free_disk_gb": round(free_gb, 2),
        "ephemeral_record_cache": ephemeral_cache,
        "native_source": str(NATIVE),
        "experiment_source": str(SOURCE),
    },
    flush=True,
)

source = SOURCE.read_text(encoding="utf-8")
old_experiment = 'EXPERIMENT = "fullwell_correlation_crf_nested_gateway_v1"'
new_experiment = 'EXPERIMENT = "fullwell_correlation_crf_nested_gateway_v173"'
if old_experiment not in source:
    raise RuntimeError("Cannot locate the frozen experiment name in the V1 source")
source = source.replace(old_experiment, new_experiment, 1)

if ephemeral_cache:
    # The experiment ultimately keeps every Record in RAM for training. Avoid
    # first duplicating newly built records onto a nearly full filesystem.
    experiment_marker = new_experiment + "\n"
    source = source.replace(
        experiment_marker,
        experiment_marker
        + "V173_EPHEMERAL_RECORD_CACHE = True\n"
        + "_V173_MEMORY_RECORDS = {}\n",
        1,
    )
    save_call = "                _save_record(record)\n"
    built_label = '                source = "built"'
    if save_call not in source or built_label not in source:
        raise RuntimeError("Cannot patch V173 record-cache write path")
    source = source.replace(
        save_call,
        "                _V173_MEMORY_RECORDS[str(well)] = record\n",
        1,
    )
    source = source.replace(
        built_label, '                source = "built_memory"', 1
    )
    load_marker = 'records = {well: _load_record(well) for well in wells}'
    load_replacement = '''records = {
    well: (
        _V173_MEMORY_RECORDS[str(well)]
        if str(well) in _V173_MEMORY_RECORDS
        else _load_record(well)
    )
    for well in wells
}'''
    if load_marker not in source:
        raise RuntimeError("Cannot patch V173 in-memory record handoff")
    source = source.replace(load_marker, load_replacement, 1)

namespace = {
    "__name__": "__main__",
    "__file__": str(SOURCE),
}
exec(compile(source, str(SOURCE), "exec"), namespace)
