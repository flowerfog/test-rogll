"""Build all legal native dual-reference records without training a model.

Upload this file and ``native_resolution_dualref_capacity_v0_cell.py`` to
``/root/autodl-tmp`` and run this cell once.  The capacity audit intentionally
persists only its 24 selected wells; downstream fold experiments need all 773.
"""

from pathlib import Path
import json
import sys
import time
import types

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
SOURCE_CANDIDATES = (
    ROOT / "native_resolution_dualref_capacity_v0_cell.py",
    Path.cwd() / "native_resolution_dualref_capacity_v0_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError(
        "Upload native_resolution_dualref_capacity_v0_cell.py to /root/autodl-tmp"
    )

source = SOURCE_PATH.read_text(encoding="utf-8")
marker = "# Execute"
if marker not in source:
    raise RuntimeError(f"Cannot locate the execution marker in {SOURCE_PATH}")

# Keep the capacity module's globals isolated so later notebook cells cannot
# replace its Record class or path configuration.  Python 3.12 dataclasses
# require the defining module to be registered in sys.modules.
module_name = "_native_dualref_record_builder_runtime"
capacity_module = types.ModuleType(module_name)
capacity_module.__file__ = str(SOURCE_PATH)
sys.modules[module_name] = capacity_module
capacity = capacity_module.__dict__
exec(compile(source.split(marker, 1)[0], str(SOURCE_PATH), "exec"), capacity)

cache_dir = Path(capacity["CACHE_DIR"])
cache_dir.mkdir(parents=True, exist_ok=True)
paths = capacity["_horizontal_paths"]()
if len(paths) != 773:
    raise RuntimeError(f"Expected 773 horizontal wells; found {len(paths)}")

started = time.time()
rows = []
for number, horizontal_path in enumerate(paths, start=1):
    well = horizontal_path.name.split("__", 1)[0]
    output = cache_dir / f"{well}.npz"
    status = "cached"
    try:
        rebuild = True
        if output.exists():
            with np.load(output, allow_pickle=False) as data:
                rebuild = (
                    "cache_version" not in data.files
                    or int(data["cache_version"][0]) != 1
                )
        if rebuild:
            record = capacity["_build_record"](well)
            capacity["_save_record"](record)
            status = "built"
        else:
            record = capacity["_load_record"](well)
        rows.append(
            {
                "well": well,
                "status": status,
                "rows": len(record.md),
                "eval_rows": record.eval_rows,
                "teacher_rmse": record.teacher_rmse,
                "axis_coverage": record.axis_coverage,
            }
        )
    except Exception as error:
        rows.append({"well": well, "status": "failed", "reason": repr(error)})
    if number % 50 == 0 or number == len(paths):
        print(
            f"build all native records {number}/{len(paths)} "
            f"elapsed={(time.time() - started) / 60.0:.1f}m",
            flush=True,
        )

report = pd.DataFrame(rows)
report_path = Path(capacity["OUT_DIR"]) / "all_record_build_report.csv"
report.to_csv(report_path, index=False)
failed = report[report["status"].eq("failed")]
summary = {
    "experiment": "build_native_dualref_all_records_v0",
    "source": str(SOURCE_PATH),
    "cache_dir": str(cache_dir),
    "wells": int(len(report)),
    "built": int(report["status"].eq("built").sum()),
    "cached": int(report["status"].eq("cached").sum()),
    "failed": int(len(failed)),
    "files_after": int(sum(1 for _ in cache_dir.glob("*.npz"))),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
print("ALL RECORDS SUMMARY")
print(json.dumps(summary, indent=2), flush=True)
if len(failed):
    print(failed.head(20).to_string(index=False), flush=True)
    raise RuntimeError(f"Failed to build {len(failed)} records; see {report_path}")
if summary["files_after"] != 773:
    raise RuntimeError(f"Expected 773 cache files; found {summary['files_after']}")
print("saved:", report_path)
