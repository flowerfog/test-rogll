"""Run the disclosed 3-seed x 5-fold whole-well members.

This is the full member sweep for the two publicly described U-Net variants;
the later combiner reads the exported fold OOF files.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(os.environ.get("V239_ROOT", "/root/autodl-tmp"))
SCRIPT = ROOT / "ninth_place_wholewell_unet_v239.py"
OUT_ROOT = ROOT / "oof_exports" / "ninth_place_members_v241"
variants = [v.strip() for v in os.environ.get("V241_VARIANTS", "stock,relief").split(",") if v.strip()]
seeds = [int(v) for v in os.environ.get("V241_SEEDS", "239,240,241").split(",")]
folds = [int(v) for v in os.environ.get("V241_FOLDS", "1,2,3,4,5").split(",")]
epochs = os.environ.get("V241_EPOCHS", "18")

if not SCRIPT.is_file():
    raise FileNotFoundError(SCRIPT)
started = time.time()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
manifest = []
for variant in variants:
    for seed in seeds:
        for fold in folds:
            env = os.environ.copy()
            env.update(
                {
                    "V239_MODE": "full",
                    "V239_GRID_MODE": "fixed16",
                    "V239_MODEL_VARIANT": variant,
                    "V239_SEED": str(seed),
                    "V239_VALID_FOLD": str(fold),
                    "V239_EPOCHS": epochs,
                    "V239_OUT_ROOT": str(OUT_ROOT),
                }
            )
            label = f"{variant}/seed{seed}/fold{fold}"
            print("V241 MEMBER START", label, flush=True)
            result = subprocess.run([sys.executable, str(SCRIPT)], env=env)
            manifest.append({"variant": variant, "seed": seed, "fold": fold, "returncode": result.returncode})
            if result.returncode != 0:
                raise RuntimeError(f"member failed: {label}")
            print("V241 MEMBER DONE", label, flush=True)

(OUT_ROOT / "member_manifest.csv").write_text(
    "variant,seed,fold,returncode\n"
    + "\n".join(f"{row['variant']},{row['seed']},{row['fold']},{row['returncode']}" for row in manifest)
    + "\n",
    encoding="utf-8",
)
print("V241 MEMBERS COMPLETE", {"members": len(manifest), "elapsed_minutes": (time.time() - started) / 60.0}, flush=True)

