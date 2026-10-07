"""Audit deployment-legal raw information before admitting a new model route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"
RESAMPLE_POINTS = 256


def _well_id(path: Path, suffix: str) -> str:
    return path.name.removesuffix(suffix)


def _schema(paths: list[Path]) -> dict[str, list[str]]:
    schemas: dict[str, list[str]] = {}
    for path in paths:
        columns = tuple(pd.read_csv(path, nrows=0).columns.astype(str))
        schemas.setdefault("|".join(columns), []).append(_well_id(path, HORIZONTAL_SUFFIX))
    return schemas


def _finite_interp(values: np.ndarray, points: int = RESAMPLE_POINTS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.full(points, np.nan, dtype=np.float64)
    source = np.linspace(0.0, 1.0, len(values))[finite]
    target = np.linspace(0.0, 1.0, points)
    return np.interp(target, source, values[finite])


def _typewell_signature(path: Path) -> tuple[np.ndarray, str]:
    frame = pd.read_csv(path, usecols=lambda column: column in {"TVT", "GR"})
    gr = _finite_interp(frame["GR"].to_numpy(np.float64))
    median = float(np.nanmedian(gr))
    scale = float(np.nanmedian(np.abs(gr - median))) * 1.4826
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanstd(gr))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    normalized = np.nan_to_num((gr - median) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    quantized = np.round(normalized, 3).astype(np.float32)
    digest = hashlib.sha256(quantized.tobytes()).hexdigest()
    return normalized.astype(np.float32), digest


def _typewell_panel(paths: list[Path], split: str) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    for path in paths:
        vector, digest = _typewell_signature(path)
        rows.append(
            {
                "split": split,
                "well": _well_id(path, TYPEWELL_SUFFIX),
                "typewell_shape_hash": digest,
            }
        )
        vectors.append(vector)
    return pd.DataFrame(rows), np.stack(vectors) if vectors else np.empty((0, RESAMPLE_POINTS))


def _nearest_train_distance(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(train) or not len(query):
        return np.full(len(query), np.nan), np.full(len(query), -1, dtype=np.int64)
    train_norm = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    query_norm = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    similarity = query_norm @ train_norm.T
    indices = np.argmax(similarity, axis=1)
    distances = 1.0 - similarity[np.arange(len(query)), indices]
    return distances.astype(np.float64), indices.astype(np.int64)


def _prefix_report(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        frame = pd.read_csv(path, usecols=lambda column: column in {"TVT_input", "MD"})
        known = frame["TVT_input"].notna().to_numpy()
        first_hidden = int(np.flatnonzero(~known)[0]) if (~known).any() else len(frame)
        prefix_is_contiguous = bool(known[:first_hidden].all() and not known[first_hidden:].any())
        rows.append(
            {
                "well": _well_id(path, HORIZONTAL_SUFFIX),
                "rows": len(frame),
                "known_rows": int(known.sum()),
                "hidden_rows": int((~known).sum()),
                "known_fraction": float(known.mean()),
                "prefix_is_contiguous": prefix_is_contiguous,
            }
        )
    return pd.DataFrame(rows)


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=None)
    args, _ = parser.parse_known_args()

    root = args.root.resolve()
    train_dir = root / "train"
    test_dir = root / "test"
    out_dir = (args.out or root / "output" / "raw_information_novelty_audit").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_horizontal = sorted(train_dir.glob(f"*{HORIZONTAL_SUFFIX}"))
    test_horizontal = sorted(test_dir.glob(f"*{HORIZONTAL_SUFFIX}"))
    train_typewell = sorted(train_dir.glob(f"*{TYPEWELL_SUFFIX}"))
    test_typewell = sorted(test_dir.glob(f"*{TYPEWELL_SUFFIX}"))
    if not train_horizontal or not train_typewell:
        raise RuntimeError(f"Training CSVs were not found under {train_dir}")

    train_ids = {_well_id(path, HORIZONTAL_SUFFIX) for path in train_horizontal}
    test_ids = {_well_id(path, HORIZONTAL_SUFFIX) for path in test_horizontal}
    overlap = sorted(train_ids & test_ids)

    train_schemas = _schema(train_horizontal)
    test_schemas = _schema(test_horizontal)
    train_columns = set.intersection(*(set(key.split("|")) for key in train_schemas))
    test_columns = set.intersection(*(set(key.split("|")) for key in test_schemas)) if test_schemas else set()

    train_panel, train_vectors = _typewell_panel(train_typewell, "train")
    test_panel, test_vectors = _typewell_panel(test_typewell, "test")
    if len(test_panel):
        distances, indices = _nearest_train_distance(train_vectors, test_vectors)
        test_panel["nearest_train_well"] = train_panel.iloc[indices]["well"].to_numpy()
        test_panel["nearest_train_cosine_distance"] = distances
        test_panel["exact_shape_match"] = test_panel["typewell_shape_hash"].isin(
            set(train_panel["typewell_shape_hash"])
        )

    combined = pd.concat([train_panel, test_panel], ignore_index=True)
    group_sizes = combined.groupby("typewell_shape_hash")["well"].transform("size")
    combined["shape_group_size"] = group_sizes.astype(int)
    combined.to_csv(out_dir / "typewell_shape_registry.csv", index=False)

    prefix = _prefix_report(test_horizontal)
    prefix.to_csv(out_dir / "test_prefix_registry.csv", index=False)

    train_duplicate_fraction = float((train_panel["typewell_shape_hash"].duplicated(False)).mean())
    report = {
        "experiment": "raw_information_novelty_audit",
        "model_training_count": 0,
        "train_wells": len(train_ids),
        "test_wells_visible_locally": len(test_ids),
        "train_test_well_overlap": len(overlap),
        "overlap_wells": overlap,
        "train_horizontal_schemas": {
            schema: len(wells) for schema, wells in train_schemas.items()
        },
        "test_horizontal_schemas": {
            schema: len(wells) for schema, wells in test_schemas.items()
        },
        "deployment_legal_horizontal_columns": sorted(train_columns & test_columns),
        "train_only_horizontal_columns": sorted(train_columns - test_columns),
        "train_typewell_duplicate_shape_fraction": train_duplicate_fraction,
        "train_typewell_exact_shape_groups": int(
            train_panel.loc[
                train_panel["typewell_shape_hash"].duplicated(False), "typewell_shape_hash"
            ].nunique()
        ),
        "test_exact_typewell_match_fraction": (
            float(test_panel["exact_shape_match"].mean()) if len(test_panel) else None
        ),
        "test_nearest_typewell_cosine_distance": (
            test_panel["nearest_train_cosine_distance"].describe().to_dict()
            if len(test_panel)
            else None
        ),
        "test_prefix_contiguous": bool(prefix["prefix_is_contiguous"].all()) if len(prefix) else None,
        "interpretation": {
            "formation_columns": "train-only labels; forbidden as direct validation features",
            "overlap": "local public examples only; cannot support a hidden-well CV claim",
            "typewell_duplicates": "admit a new retrieval route only if exact groups are material and split-clean",
        },
    }
    report = _json_value(report)
    (out_dir / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
