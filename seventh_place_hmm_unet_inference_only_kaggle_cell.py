"""Kaggle cell: load AutoDL-trained emission and Refiner checkpoints only."""

from pathlib import Path
import runpy
import shutil
import sys


def _find(name: str) -> Path:
    hits = []
    for root in (Path("/kaggle/working"), Path("/kaggle/input")):
        if root.is_dir():
            hits.extend(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"Missing {name}; upload the trained model bundle")
    return sorted(hits, key=lambda p: (len(p.parts), str(p)))[0]


def _data_root() -> Path:
    for root in (Path("/kaggle/working"), Path("/kaggle/input")):
        if root.is_dir():
            for sample in root.rglob("sample_submission.csv"):
                candidate = sample.parent
                if (candidate / "train").is_dir() and (candidate / "test").is_dir():
                    return candidate
    raise FileNotFoundError("Cannot locate competition data root")


source = _find("seventh_place_hmm_unet_cleanroom_v1.py")
emission = _find("emission_model.json")
model_dir = emission.parent
for seed in (1103, 2207):
    checkpoint = model_dir / f"refiner_seed{seed}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

output = Path("/kaggle/working/seventh_place_hmm_unet_inference_only")
data_root = _data_root()
print({"source": str(source), "model_dir": str(model_dir), "data_root": str(data_root)})

sys.argv = [
    str(source),
    "--data-root", str(data_root),
    "--output", str(output),
    "--device", "cuda",
    "--inference-only",
]
runpy.run_path(str(source), run_name="__main__")
shutil.copy2(output / "submission.csv", "/kaggle/working/submission.csv")
print("READY /kaggle/working/submission.csv")
