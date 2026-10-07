from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

from .config import FOLDS
from .data import list_wells, load_well
from .pf import build_pf_features
from .xy import build_xy_features


def _pf_job(arguments: tuple[str, str, str, str]) -> str:
    data_root, output_root, split, well_name = arguments
    record = load_well(Path(data_root), well_name, split)
    result = build_pf_features(record, Path(output_root) / "pf" / split)
    assert result["pf_prob"].shape == (400, 345)
    return well_name


def build_folds(data_root: Path, output_root: Path) -> pd.DataFrame:
    wells = np.array(list_wells(data_root, "train"))
    centers = []
    for well in wells:
        hz = pd.read_csv(data_root / "train" / f"{well}__horizontal_well.csv",
                         usecols=["X", "Y"])
        centers.append((float(hz.X.mean()), float(hz.Y.mean())))
    centers = np.asarray(centers, dtype=np.float64)
    scale = np.maximum(centers.std(axis=0), 1e-9)
    clusters = KMeans(n_clusters=20, random_state=0, n_init=10).fit_predict(
        (centers - centers.mean(axis=0)) / scale
    )
    fold = np.zeros(len(wells), int)
    splitter = StratifiedKFold(5, shuffle=True, random_state=7)
    for number, (_, valid) in enumerate(splitter.split(wells, clusters), 1):
        fold[valid] = number
    frame = pd.DataFrame({"well": wells, "fold": fold, "geo_cluster": clusters,
                          "center_x": centers[:, 0], "center_y": centers[:, 1]})
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "folds.csv", index=False)
    return frame


def save_aux(path: Path, channels: dict[str, np.ndarray], stats: dict[str, float] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(channels)
    if stats:
        payload.update({f"stat__{key}": np.asarray(value) for key, value in stats.items()})
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--component", choices=("folds", "pf", "xy", "all"), default="all")
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    folds = build_folds(args.data_root, args.output_root)
    if args.component == "folds":
        return
    splits = ("train", "test") if args.split == "both" else (args.split,)
    report: dict[str, object] = {"component": args.component, "splits": splits, "folds": FOLDS}
    train_records = {well: load_well(args.data_root, well, "train") for well in folds.well}
    for split in splits:
        wells = list_wells(args.data_root, split)
        if args.limit:
            wells = wells[: args.limit]
        pf_parallel = args.component in ("pf", "all") and args.workers > 1
        if pf_parallel:
            jobs = [(str(args.data_root), str(args.output_root), split, well) for well in wells]
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for number, _ in enumerate(executor.map(_pf_job, jobs), 1):
                    if number % 10 == 0 or number == len(wells):
                        print("PREPARE PF", {"split": split, "complete": number, "total": len(wells)}, flush=True)
            if args.component == "pf":
                continue
        if args.component in ("xy", "all") and split == "train":
            for outer_fold in FOLDS:
                support_names = folds.loc[folds.fold.ne(outer_fold), "well"].tolist()
                support = [train_records[name] for name in support_names]
                for number, well_name in enumerate(wells, 1):
                    # A training query can itself belong to the outer-fold support bank.
                    # Remove it so its future truth never predicts itself.
                    query_support = support if well_name not in support_names else [
                        record for record in support if record.well != well_name
                    ]
                    result = build_xy_features(train_records[well_name], query_support)
                    save_aux(args.output_root / "xy" / split / f"fold{outer_fold}" /
                             f"{well_name}.npz", result.channels, result.stats)
                    if number % 10 == 0 or number == len(wells):
                        print("PREPARE XY", {"split": split, "outer_fold": outer_fold,
                                             "complete": number, "total": len(wells)}, flush=True)
            continue
        for number, well_name in enumerate(wells, 1):
            record = load_well(args.data_root, well_name, split)
            if args.component in ("pf", "all") and not pf_parallel:
                features = build_pf_features(record, args.output_root / "pf" / split)
                # build_pf_features writes the same cache itself.
                assert features["pf_prob"].shape == (400, 345)
            if args.component in ("xy", "all"):
                support_names = folds.well.tolist()
                result = build_xy_features(record, [train_records[name] for name in support_names])
                save_aux(args.output_root / "xy" / split / f"{well_name}.npz", result.channels, result.stats)
            if number % 10 == 0 or number == len(wells):
                print("PREPARE", {"split": split, "complete": number, "total": len(wells)}, flush=True)
    (args.output_root / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
