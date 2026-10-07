"""Kaggle Jupyter deployment cell for the clean-room HMM + UNet pipeline."""

from pathlib import Path
import runpy
import shutil
import sys


def _find_file(name: str) -> Path:
    roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    candidates = []
    for root in roots:
        if root.is_dir():
            candidates.extend(root.rglob(name))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot locate {name}. Upload it as a Kaggle Dataset or place it in /kaggle/working."
        )
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


def _find_data_root() -> Path:
    roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    for root in roots:
        if not root.is_dir():
            continue
        for sample in root.rglob("sample_submission.csv"):
            candidate = sample.parent
            if (candidate / "train").is_dir() and (candidate / "test").is_dir():
                return candidate
    raise FileNotFoundError(
        "Cannot locate competition data root with train/, test/, and sample_submission.csv"
    )


SOURCE = _find_file("seventh_place_hmm_unet_cleanroom_v1.py")
DATA_ROOT = _find_data_root()
OUTPUT = Path("/kaggle/working/seventh_place_hmm_unet_cleanroom_v2")

print(
    {
        "source": str(SOURCE),
        "data_root": str(DATA_ROOT),
        "output": str(OUTPUT),
        "device": "cuda",
    },
    flush=True,
)

sys.argv = [
    str(SOURCE),
    "--data-root",
    str(DATA_ROOT),
    "--output",
    str(OUTPUT),
    "--device",
    "cuda",
]
runpy.run_path(str(SOURCE), run_name="__main__")

generated = OUTPUT / "submission.csv"
if not generated.is_file():
    raise FileNotFoundError(f"Pipeline finished without {generated}")
shutil.copy2(generated, Path("/kaggle/working/submission.csv"))
print(
    {
        "submission": "/kaggle/working/submission.csv",
        "bytes": Path("/kaggle/working/submission.csv").stat().st_size,
    },
    flush=True,
)
