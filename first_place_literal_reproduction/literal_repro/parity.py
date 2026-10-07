from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    ref = pd.read_csv(args.reference).sort_values("id").reset_index(drop=True)
    cand = pd.read_csv(args.candidate).sort_values("id").reset_index(drop=True)
    if ref["id"].tolist() != cand["id"].tolist():
        raise RuntimeError("submission ids differ")
    delta = cand["tvt"].to_numpy(float) - ref["tvt"].to_numpy(float)
    report = {
        "rows": len(ref),
        "max_abs_difference": float(np.max(np.abs(delta))),
        "rmse_difference": float(np.sqrt(np.mean(np.square(delta)))),
        "atol": args.atol,
    }
    report["ok"] = report["max_abs_difference"] <= args.atol
    print("0719 PARITY", json.dumps(report, indent=2), flush=True)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
