"""V194: train-only Geology supervision for legal GR-to-facies registration.

Geology is used only as a training label. Two independent classifiers map local
horizontal and typewell GR context into a shared facies probability simplex.
Fold4/5 train both classifiers; Fold2 measures classification and pointwise TVT
retrieval. No historical trajectory endpoint or held-out truth enters inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


EXPERIMENT = "facies_probability_registration_gateway_v194"
TRAIN_FOLDS = (4, 5)
HELD_FOLD = 2
SEED = 20260806
PROFILE_WINDOWS = (5, 17, 65, 257)
MAX_HORIZONTAL_TRAIN_ROWS = 512
MAX_TYPEWELL_TRAIN_ROWS = 384
HELD_ROWS_PER_WELL = 192
SEARCH_RADIUS_FT = 128.0


@dataclass
class Record:
    well: str
    fold: int
    md: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    truth: np.ndarray
    tvt_input: np.ndarray
    hidden: np.ndarray
    tw_tvt: np.ndarray
    tw_gr: np.ndarray
    tw_geology: np.ndarray


def _find_data_root(root: Path) -> Path:
    for candidate in (root / "rogii-wellbore-geology-prediction", root, Path.cwd()):
        if (candidate / "train").is_dir():
            return candidate
    raise FileNotFoundError("Competition train directory not found")


def _find_metadata(root: Path) -> Path:
    for path in (
        root / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
        root / "oof_exports" / "multicut_k5_full_spatial_lgb_bundle_v1" / "well_metadata.csv",
        Path.cwd() / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError("Frozen well metadata not found")


def _fill(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        raise RuntimeError("GR has fewer than two finite values")
    positions = np.arange(len(values))
    return np.interp(positions, positions[finite], values[finite])


def _profile_features(gr: np.ndarray) -> np.ndarray:
    gr = _fill(gr)
    median = float(np.median(gr))
    scale = float(np.quantile(np.abs(gr - median), 0.75))
    scale = max(scale, 1.0)
    normalised = np.clip((gr - median) / scale, -8.0, 8.0)
    series = pd.Series(normalised)
    parts = [
        normalised,
        np.clip(gr / 150.0, -2.0, 4.0),
        np.clip(np.gradient(normalised), -4.0, 4.0),
        np.clip(np.gradient(np.gradient(normalised)), -4.0, 4.0),
    ]
    for window in PROFILE_WINDOWS:
        rolling = series.rolling(window, center=True, min_periods=1)
        mean = rolling.mean().to_numpy()
        std = rolling.std(ddof=0).fillna(0.0).to_numpy()
        minimum = rolling.min().to_numpy()
        maximum = rolling.max().to_numpy()
        parts.extend((mean, std, maximum - minimum, normalised - mean))
    matrix = np.column_stack(parts).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite GR profile features")
    return matrix


def _load_record(data_root: Path, well: str, fold: int) -> Record:
    horizontal = pd.read_csv(
        data_root / "train" / f"{well}__horizontal_well.csv",
        usecols=["MD", "Z", "GR", "TVT", "TVT_input"],
    )
    typewell = pd.read_csv(data_root / "train" / f"{well}__typewell.csv")
    for column in horizontal.columns:
        horizontal[column] = pd.to_numeric(horizontal[column], errors="coerce")
    for column in ("TVT", "GR"):
        typewell[column] = pd.to_numeric(typewell[column], errors="coerce")
    horizontal_values = horizontal[["MD", "Z", "TVT"]].to_numpy(np.float64)
    if not np.isfinite(horizontal_values).all():
        raise RuntimeError(f"{well}: non-finite horizontal inputs or truth")
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    hidden = ~np.isfinite(tvt_input)
    first_hidden = int(np.flatnonzero(hidden)[0])
    if first_hidden < 3 or hidden[:first_hidden].any() or not hidden[first_hidden:].all():
        raise RuntimeError(f"{well}: invalid visible prefix")
    typewell = typewell.dropna(subset=["TVT"]).sort_values("TVT")
    typewell = typewell.groupby("TVT", as_index=False).agg(
        GR=("GR", "mean"), Geology=("Geology", "first")
    )
    geology = typewell["Geology"].astype("string").fillna("").str.strip().to_numpy(str)
    return Record(
        well=well,
        fold=fold,
        md=horizontal["MD"].to_numpy(np.float64),
        z=horizontal["Z"].to_numpy(np.float64),
        gr=_fill(horizontal["GR"].to_numpy(np.float64)),
        truth=horizontal["TVT"].to_numpy(np.float64),
        tvt_input=tvt_input,
        hidden=hidden,
        tw_tvt=typewell["TVT"].to_numpy(np.float64),
        tw_gr=_fill(typewell["GR"].to_numpy(np.float64)),
        tw_geology=geology,
    )


def _nearest_label(record: Record, tvt: np.ndarray) -> np.ndarray:
    labelled = record.tw_geology != ""
    axis = record.tw_tvt[labelled]
    labels = record.tw_geology[labelled]
    output = np.full(len(tvt), "", dtype=object)
    if len(axis) == 0:
        return output
    positions = np.searchsorted(axis, tvt)
    left = np.clip(positions - 1, 0, len(axis) - 1)
    right = np.clip(positions, 0, len(axis) - 1)
    nearest = np.where(np.abs(tvt - axis[left]) <= np.abs(tvt - axis[right]), left, right)
    inside = (tvt >= axis[0]) & (tvt <= axis[-1])
    output[inside] = labels[nearest[inside]]
    return output.astype(str)


def _sample_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    indices = np.asarray(indices, np.int64)
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum).round().astype(np.int64)
    return indices[np.unique(positions)]


def _training_tables(records: list[Record]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizontal_x, horizontal_y, typewell_x, typewell_y = [], [], [], []
    for number, record in enumerate(records, 1):
        h_features = _profile_features(record.gr)
        h_labels = _nearest_label(record, record.truth)
        h_valid = np.flatnonzero(h_labels != "")
        h_rows = _sample_indices(h_valid, MAX_HORIZONTAL_TRAIN_ROWS)
        horizontal_x.append(h_features[h_rows])
        horizontal_y.append(h_labels[h_rows])

        tw_features = _profile_features(record.tw_gr)
        tw_valid = np.flatnonzero(record.tw_geology != "")
        tw_rows = _sample_indices(tw_valid, MAX_TYPEWELL_TRAIN_ROWS)
        typewell_x.append(tw_features[tw_rows])
        typewell_y.append(record.tw_geology[tw_rows])
        if number % 100 == 0:
            print(f"V194 training features {number}/{len(records)}", flush=True)
    return (
        np.concatenate(horizontal_x),
        np.concatenate(horizontal_y),
        np.concatenate(typewell_x),
        np.concatenate(typewell_y),
    )


def _fit_classifier(matrix: np.ndarray, labels: np.ndarray) -> lgb.LGBMClassifier:
    classes, encoded = np.unique(labels, return_inverse=True)
    counts = np.bincount(encoded)
    weights = len(encoded) / (len(counts) * counts[encoded])
    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(matrix, encoded, sample_weight=weights)
    model.facies_classes_ = classes
    return model


def _probabilities(model: lgb.LGBMClassifier, matrix: np.ndarray, classes: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(matrix)
    output = np.full((len(matrix), len(classes)), 1e-6, np.float64)
    lookup = {label: index for index, label in enumerate(classes)}
    for source, label in enumerate(model.facies_classes_):
        if label in lookup:
            output[:, lookup[label]] = raw[:, source]
    output /= output.sum(axis=1, keepdims=True)
    return output


def _classification_report(
    records: list[Record], horizontal_model, typewell_model, classes: np.ndarray
) -> dict[str, object]:
    true_h, pred_h, true_tw, pred_tw = [], [], [], []
    known_classes = set(classes.tolist())
    for record in records:
        h_labels = _nearest_label(record, record.truth)
        h_rows = _sample_indices(np.flatnonzero(np.isin(h_labels, classes)), 384)
        h_prob = _probabilities(horizontal_model, _profile_features(record.gr)[h_rows], classes)
        true_h.extend(h_labels[h_rows]); pred_h.extend(classes[np.argmax(h_prob, axis=1)])

        tw_rows = _sample_indices(
            np.flatnonzero(np.array([label in known_classes for label in record.tw_geology])), 384
        )
        tw_prob = _probabilities(typewell_model, _profile_features(record.tw_gr)[tw_rows], classes)
        true_tw.extend(record.tw_geology[tw_rows]); pred_tw.extend(classes[np.argmax(tw_prob, axis=1)])

    def report(truth, prediction):
        return {
            "rows": len(truth),
            "accuracy": float(accuracy_score(truth, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, average="macro")),
        }
    return {"horizontal": report(true_h, pred_h), "typewell": report(true_tw, pred_tw)}


def _retrieve_one(
    record: Record,
    horizontal_model,
    typewell_model,
    classes: np.ndarray,
    donor: Record | None = None,
) -> pd.DataFrame:
    hidden = np.flatnonzero(record.hidden)
    rows = _sample_indices(hidden, HELD_ROWS_PER_WELL)
    h_prob = _probabilities(horizontal_model, _profile_features(record.gr)[rows], classes)
    reference = record if donor is None else donor
    tw_prob = _probabilities(typewell_model, _profile_features(reference.tw_gr), classes)
    anchor = int(hidden[0] - 1)
    baseline = record.tvt_input[anchor] - (record.z[rows] - record.z[anchor])
    outputs = []
    for local_index, (row, centre) in enumerate(zip(rows, baseline, strict=True)):
        candidates = np.flatnonzero(np.abs(reference.tw_tvt - centre) <= SEARCH_RADIUS_FT)
        if len(candidates) < 32:
            candidates = np.argsort(np.abs(reference.tw_tvt - centre))[: max(32, min(256, len(reference.tw_tvt)))]
        similarity = tw_prob[candidates] @ h_prob[local_index]
        prior = -0.5 * np.square((reference.tw_tvt[candidates] - centre) / 96.0)
        score = np.log(np.maximum(similarity, 1e-8)) + prior
        order = candidates[np.argsort(score)[::-1]]
        truth = float(record.truth[row])
        result = {
            "well": record.well,
            "fold": record.fold,
            "row_idx": int(row),
            "truth": truth,
            "baseline": float(centre),
            "top1": float(reference.tw_tvt[order[0]]),
            "distance": float(record.md[row] - record.md[anchor]),
        }
        for topk in (4, 8, 16, 32):
            support = reference.tw_tvt[order[:topk]]
            result[f"oracle_top{topk}"] = float(support[np.argmin(np.abs(support - truth))])
        outputs.append(result)
    return pd.DataFrame(outputs)


def _rmse(frame: pd.DataFrame, column: str) -> float:
    return float(np.sqrt(np.mean(np.square(frame[column] - frame["truth"]))))


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp"))
    args, _ = parser.parse_known_args()
    started = time.time()
    root = args.root.resolve()
    data_root = _find_data_root(root)
    out_dir = root / "oof_exports" / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(_find_metadata(root), dtype={"well": str})[["well", "fold"]]
    metadata = metadata.drop_duplicates("well")
    metadata["well"] = metadata["well"].str.strip().str.lower()
    metadata["fold"] = metadata["fold"].astype(int)
    metadata = metadata[metadata["fold"].isin((*TRAIN_FOLDS, HELD_FOLD))].sort_values(["fold", "well"])
    records = []
    for number, row in enumerate(metadata.itertuples(index=False), 1):
        records.append(_load_record(data_root, row.well, int(row.fold)))
        if number % 100 == 0:
            print(f"V194 records {number}/{len(metadata)}", flush=True)
    coverage = pd.Series([record.fold for record in records]).value_counts().sort_index().to_dict()
    if coverage != {2: 155, 4: 155, 5: 154}:
        raise RuntimeError(f"Unexpected V194 coverage: {coverage}")
    train = [record for record in records if record.fold in TRAIN_FOLDS]
    held = [record for record in records if record.fold == HELD_FOLD]

    horizontal_x, horizontal_y, typewell_x, typewell_y = _training_tables(train)
    classes = np.intersect1d(np.unique(horizontal_y), np.unique(typewell_y))
    horizontal_keep = np.isin(horizontal_y, classes)
    typewell_keep = np.isin(typewell_y, classes)
    horizontal_model = _fit_classifier(horizontal_x[horizontal_keep], horizontal_y[horizontal_keep])
    typewell_model = _fit_classifier(typewell_x[typewell_keep], typewell_y[typewell_keep])
    joblib.dump(horizontal_model, out_dir / "horizontal_facies_lgb.joblib")
    joblib.dump(typewell_model, out_dir / "typewell_facies_lgb.joblib")

    classification = _classification_report(held, horizontal_model, typewell_model, classes)
    retrieval_parts, control_parts = [], []
    donors = held[1:] + held[:1]
    for number, (record, donor) in enumerate(zip(held, donors, strict=True), 1):
        retrieval_parts.append(_retrieve_one(record, horizontal_model, typewell_model, classes))
        control_parts.append(_retrieve_one(record, horizontal_model, typewell_model, classes, donor=donor))
        if number % 25 == 0:
            print(f"V194 retrieval {number}/{len(held)}", flush=True)
    retrieval = pd.concat(retrieval_parts, ignore_index=True)
    control = pd.concat(control_parts, ignore_index=True)
    retrieval.to_parquet(out_dir / "fold2_sample192_retrieval.parquet", index=False)

    values = {
        "baseline_rmse": _rmse(retrieval, "baseline"),
        "top1_rmse": _rmse(retrieval, "top1"),
        "oracle_top4_rmse": _rmse(retrieval, "oracle_top4"),
        "oracle_top8_rmse": _rmse(retrieval, "oracle_top8"),
        "oracle_top16_rmse": _rmse(retrieval, "oracle_top16"),
        "oracle_top32_rmse": _rmse(retrieval, "oracle_top32"),
        "rolled_typewell_top1_rmse": _rmse(control, "top1"),
    }
    checks = {
        "horizontal_macro_f1_ge_0p40": classification["horizontal"]["macro_f1"] >= 0.40,
        "typewell_macro_f1_ge_0p50": classification["typewell"]["macro_f1"] >= 0.50,
        "oracle_top32_rmse_le_6": values["oracle_top32_rmse"] <= 6.0,
        "top1_rmse_le_10": values["top1_rmse"] <= 10.0,
        "rolled_typewell_degradation_ge_0p50": (
            values["rolled_typewell_top1_rmse"] - values["top1_rmse"] >= 0.50
        ),
    }
    strict_pass = bool(all(checks.values()))
    verdict = _clean(
        {
            "experiment": EXPERIMENT,
            "stage": "Fold45_Geology_supervision_Fold2_facies_registration_gateway",
            "historical_prediction_inputs": [],
            "training_only_labels": ["typewell Geology", "horizontal label via true TVT during training"],
            "inference_inputs": ["horizontal GR", "typewell GR", "Z", "TVT_input visible prefix"],
            "coverage": coverage,
            "classes": classes.tolist(),
            "features": horizontal_x.shape[1],
            "classification": classification,
            "retrieval": values,
            "checks": checks,
            "strict_pass": strict_pass,
            "retrieval_sha256": hashlib.sha256(
                retrieval["top1"].to_numpy(np.float32).tobytes()
            ).hexdigest(),
            "elapsed_minutes": (time.time() - started) / 60.0,
            "next": (
                "PASS -> build one monotonic facies-probability DP on untouched Fold3"
                if strict_pass
                else "FAIL -> facies probabilities do not localize TVT; close without decoder"
            ),
        }
    )
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("\nV194 FINAL VERDICT")
    print(json.dumps(verdict, indent=2))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
