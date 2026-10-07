from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

from .constants import AUTHOR_SRC, PACKAGE_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one source-derived branch with the exact author runtime.")
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_cfg = args.output_dir / "cfg.pkl"
    target_cfg.write_bytes(args.cfg.read_bytes())
    # The author CLI creates a fresh default cfg for training.  A tiny launcher
    # loads the branch cfg and calls run_common directly; all training, CV,
    # augmentation, PF/XY construction, model and postprocessing remain author code.
    launcher = (
        "import pickle,sys; from pathlib import Path; "
        "sys.path.insert(0,sys.argv[1]); "
        "exec(\"from literal_repro.offline_timm import install; install(sys.argv[5])\") if sys.argv[5] else None; "
        "import seq_NN_main as m; cfg=pickle.load(open(sys.argv[2],'rb')); "
        "cfg.data_dir=Path(sys.argv[3]); cfg.train_path=cfg.data_dir/'train'; cfg.test_path=cfg.data_dir/'test'; "
        "cfg.output_dir=Path(sys.argv[4]); cfg.f=1; cfg.submit_mode=False; cfg.refresh(); "
        "m.run_common(cfg)"
    )
    command = [
        sys.executable, "-u", "-c", launcher, str(AUTHOR_SRC), str(target_cfg),
        str(args.data_root), str(args.output_dir), str(args.encoder_checkpoint or ""),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(PACKAGE_ROOT), str(AUTHOR_SRC), environment.get("PYTHONPATH", "")]
    )
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
