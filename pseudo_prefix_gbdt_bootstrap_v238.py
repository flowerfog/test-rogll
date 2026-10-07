"""Bootstrap public-baseline definitions, then launch V238."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path


ROOT = Path("/root/autodl-tmp")
explicit = os.environ.get("V238_NOTEBOOK")
candidates = [
    Path(explicit) if explicit else None,
    ROOT / "public_baseline.ipynb",
    ROOT / "fork-of-rogii-wellbore-geol-ecf6db (1).ipynb",
    ROOT / "fork-of-rogii-wellbore-geology-prediction.ipynb",
]
notebook = next((path for path in candidates if path is not None and path.is_file()), None)
if notebook is None:
    raise FileNotFoundError({"searched": [str(path) for path in candidates if path is not None], "hint": "set V238_NOTEBOOK"})

payload = json.loads(notebook.read_text(encoding="utf-8"))
namespace = globals()
executed = 0
for cell in payload.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    if "def build_well" not in source and "def lik_pf" not in source and executed == 0:
        # Keep imports/constants/helpers before the definitions.
        pass
    cleaned = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )
    if not cleaned.strip():
        continue
    exec(compile(cleaned, f"{notebook}:cell_{executed}", "exec"), namespace, namespace)
    executed += 1
    if "build_well" in namespace and "lik_pf" in namespace:
        break

if "build_well" not in namespace or "lik_pf" not in namespace:
    raise RuntimeError(f"Definitions not found after executing {executed} cells from {notebook}")
print("V238 BOOTSTRAP", {"notebook": str(notebook), "cells": executed, "definitions": ["build_well", "lik_pf"]}, flush=True)

runpy.run_path(
    "/root/autodl-tmp/pseudo_prefix_gbdt_gateway_v238.py",
    init_globals=namespace,
    run_name="__main__",
)
