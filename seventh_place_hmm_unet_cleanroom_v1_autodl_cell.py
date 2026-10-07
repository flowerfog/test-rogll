"""Single Jupyter cell launcher for AutoDL."""

from pathlib import Path
import runpy
import sys


SOURCE = Path("/root/autodl-tmp/seventh_place_hmm_unet_cleanroom_v1.py")
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OUTPUT = Path("/root/autodl-tmp/oof_exports/seventh_place_hmm_unet_cleanroom_v1")
DEVICE = "cuda"

if not SOURCE.is_file():
    raise FileNotFoundError(
        f"Missing {SOURCE}. Upload seventh_place_hmm_unet_cleanroom_v1.py "
        "to /root/autodl-tmp first."
    )
if not (DATA_ROOT / "train").is_dir() or not (DATA_ROOT / "test").is_dir():
    raise FileNotFoundError(f"Invalid competition data root: {DATA_ROOT}")

sys.argv = [
    str(SOURCE),
    "--data-root", str(DATA_ROOT),
    "--output", str(OUTPUT),
    "--device", DEVICE,
]
runpy.run_path(str(SOURCE), run_name="__main__")
