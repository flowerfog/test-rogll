"""Run the first-place ConvNeXt alignment reproduction over all folds/seeds.

This launcher deliberately reuses first_place_convnext_alignment_v243.py so the
single-fold implementation and the full run cannot silently diverge. It does
not invent a new model or alter the recipe between folds.

Autodl:
  python run_first_place_v243_full.py

Useful overrides:
  V243_SEEDS=243,244,245
  V243_FOLDS=1,2,3,4,5
  V243_EPOCHS=18
  V243_ENCODER_CHECKPOINT=/root/autodl-tmp/model.safetensors
  V243_ENCODER_WEIGHTS=none
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "first_place_convnext_alignment_v243.py"


def ints(name: str, default: str) -> list[int]:
    raw = os.environ.get(name, default)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    if not SCRIPT.is_file():
        raise FileNotFoundError(SCRIPT)
    folds = ints("V243_FOLDS", "1,2,3,4,5")
    seeds = ints("V243_SEEDS", "243,244,245")
    mode = os.environ.get("V243_MODE", "full")
    if mode not in {"smoke", "full"}:
        raise ValueError("V243_MODE must be smoke or full")

    root = os.environ.get("ROGII_ROOT", "/root/autodl-tmp")
    reports: list[dict] = []
    for fold in folds:
        for seed in seeds:
            env = os.environ.copy()
            env.update(
                {
                    "ROGII_ROOT": root,
                    "V243_MODE": mode,
                    "V243_VALID_FOLD": str(fold),
                    "V243_SEED": str(seed),
                    "V243_RUN_TAG": f"fold{fold}_seed{seed}",
                }
            )
            print(f"V243 RUN fold={fold} seed={seed} mode={mode}", flush=True)
            completed = subprocess.run([sys.executable, str(SCRIPT)], env=env, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"V243 failed: fold={fold}, seed={seed}, exit={completed.returncode}")
            out = Path(root) / "oof_exports" / f"first_place_convnext_alignment_v243_fold{fold}_seed{seed}"
            verdicts = sorted(out.glob("*_verdict.json"))
            if not verdicts:
                raise RuntimeError(f"V243 completed without verdict: {out}")
            verdict = json.loads(verdicts[-1].read_text(encoding="utf-8"))
            reports.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "best_rmse": verdict.get("best", {}).get("rmse"),
                    "checkpoint_dir": str(out),
                }
            )
            print("V243 DONE", reports[-1], flush=True)

    summary = {
        "experiment": "first_place_convnext_alignment_v243_full",
        "folds": folds,
        "seeds": seeds,
        "runs": reports,
        "status": "complete",
    }
    output = Path(root) / "oof_exports" / "first_place_convnext_alignment_v243_full"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("V243 FULL SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
