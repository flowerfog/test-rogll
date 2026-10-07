"""Matched-real-negative pairwise tournament for the frozen V4 lattice bank.

This is deliberately narrower than V60.  It trains only on V55 pairs whose
positive and negative paths came from a shared beam generator and had similar
legal generator rank.  A symmetric pair dataset teaches P(path A > path B),
then a label-free round-robin tournament selects one path per held-out well.

Protocol: Fold 4/5 train -> Fold 3 choose feature arm -> Fold 3/4/5 refit ->
Fold 2 confirmation.  Fold 1 remains untouched.
"""

from __future__ import annotations

import json
import hashlib
import math
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/oof_exports")
V4_DIR = ROOT / "rawdata_lattice_coarse_ranker_v4"
PANEL_DIR = V4_DIR / "panels"
V55_DIR = ROOT / "realistic_hard_negative_audit_v55"
OUT_DIR = ROOT / "realistic_pairwise_tournament_hn2b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = OUT_DIR.name
TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
MENU_SIZE = 128
PAIR_BATCH = 65536
SEED = 20260728


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _load_panel(well):
    path = PANEL_DIR / f"{well}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _find_data_root():
    candidates = (
        Path.cwd(),
        Path("/root/autodl-tmp"),
        Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
    )
    for candidate in candidates:
        train = candidate / "train"
        if train.is_dir() and next(train.glob("*__horizontal_well.csv"), None):
            return candidate
    raise FileNotFoundError("Cannot locate original train directory for canonical folds")


def _fold_table():
    train_dir = _find_data_root() / "train"
    all_wells = [
        path.name.split("__", 1)[0]
        for path in sorted(train_dir.glob("*__horizontal_well.csv"))
    ]
    if len(all_wells) != 773 or len(set(all_wells)) != 773:
        raise RuntimeError(f"Expected 773 original wells, found {len(all_wells)}/{len(set(all_wells))}")
    scored = []
    for well in all_wells:
        digest = hashlib.sha256(f"20260721:{well}".encode("ascii")).digest()
        scored.append((int.from_bytes(digest[:8], "little"), well))
    ordered = [well for _, well in sorted(scored)]
    fold_values = np.empty(len(ordered), dtype=np.int64)
    for fold, indices in enumerate(np.array_split(np.arange(len(ordered)), 5), start=1):
        fold_values[indices] = fold
    frame = pd.DataFrame({"well": ordered, "fold": fold_values})
    available = {path.stem for path in PANEL_DIR.glob("*.npz")}
    return frame[frame["well"].isin(available)].reset_index(drop=True)


def _finite_features(panel, columns):
    values = np.asarray(panel["features"][:, columns], np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=12.0, neginf=-12.0)


def _pair_matrix(pairs, columns, label):
    blocks = []
    labels = []
    weights = []
    for number, (well, local) in enumerate(pairs.groupby("well", sort=False), 1):
        panel = _load_panel(str(well))
        features = _finite_features(panel, columns)
        positive = local["positive_index"].to_numpy(np.int64)
        negative = local["negative_index"].to_numpy(np.int64)
        if positive.max(initial=-1) >= len(features) or negative.max(initial=-1) >= len(features):
            raise RuntimeError(f"{well}: V55 pair index exceeds panel")
        delta = features[positive] - features[negative]
        blocks.extend((delta, -delta))
        labels.extend((np.ones(len(delta), np.int8), np.zeros(len(delta), np.int8)))
        per_pair = np.full(len(delta), 0.5 / max(len(delta), 1), np.float32)
        weights.extend((per_pair, per_pair))
        if number % 100 == 0:
            print(f"HN2 pairs {label} {number}/{pairs['well'].nunique()}", flush=True)
    if not blocks:
        raise RuntimeError(f"No pair rows for {label}")
    return np.vstack(blocks), np.concatenate(labels), np.concatenate(weights)


def _fit(pairs, columns, label):
    x, y, weight = _pair_matrix(pairs, columns, label)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=7,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.25,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x, y, sample_weight=weight)
    print(f"HN2 fit {label}: rows={len(x):,} features={x.shape[1]}", flush=True)
    return model


def _menu(panel):
    rank = np.min(np.asarray(panel["arm_rank"], np.float64), axis=1)
    order = np.argsort(rank, kind="stable")
    return order[: min(MENU_SIZE, len(order))], rank


def _tournament(model, values):
    count = len(values)
    if count == 1:
        return np.ones(1, np.float64)
    sums = np.zeros(count, np.float64)
    seen = np.zeros(count, np.int64)
    left_all, right_all = np.triu_indices(count, 1)
    for start in range(0, len(left_all), PAIR_BATCH):
        left = left_all[start:start + PAIR_BATCH]
        right = right_all[start:start + PAIR_BATCH]
        probability = model.predict_proba(values[left] - values[right])[:, 1]
        sums[left] += probability
        sums[right] += 1.0 - probability
        seen[left] += 1
        seen[right] += 1
    return sums / np.maximum(seen, 1)


def _evaluate(model, wells, columns, control="aligned"):
    rows = []
    rng = np.random.default_rng(SEED + (0 if control == "aligned" else 1000))
    for number, well in enumerate(wells, 1):
        panel = _load_panel(well)
        indices, legal_rank = _menu(panel)
        values = _finite_features(panel, columns)[indices]
        if control == "candidate_alignment_shuffle" and len(values) > 1:
            values = values[rng.permutation(len(values))].copy()
        score = _tournament(model, values)
        local_choice = int(np.argmax(score))
        if control == "candidate_alignment_shuffle":
            # Scores belong to shuffled features while paths retain original order.
            chosen = int(indices[local_choice])
        else:
            chosen = int(indices[local_choice])
        legal = int(indices[int(np.argmin(legal_rank[indices]))])
        oracle = int(indices[int(np.argmin(panel["mse"][indices]))])
        row = {
            "well": well,
            "rows": int(panel["rows"][0]),
            "far_rows": int(panel["far_rows"][0]),
        }
        for prefix, index in (("pairwise", chosen), ("legal", legal), ("oracle", oracle)):
            row[f"{prefix}_sse"] = float(panel["sse"][index])
            row[f"{prefix}_sae"] = float(panel["sae"][index])
            row[f"{prefix}_far_sse"] = float(panel["far_sse"][index])
        rows.append(row)
        if number % 50 == 0 or number == len(wells):
            print(f"HN2 evaluate {control} {number}/{len(wells)}", flush=True)
    frame = pd.DataFrame(rows)
    metrics = {}
    for prefix in ("pairwise", "legal", "oracle"):
        total_rows = int(frame["rows"].sum())
        far_rows = int(frame["far_rows"].sum())
        sse = float(frame[f"{prefix}_sse"].sum())
        well_rmse = np.sqrt(frame[f"{prefix}_sse"] / frame["rows"])
        metrics[prefix] = {
            "rows": total_rows,
            "wells": int(len(frame)),
            "exact_sse": sse,
            "rmse": math.sqrt(sse / total_rows),
            "mae": float(frame[f"{prefix}_sae"].sum() / total_rows),
            "median_well_rmse": float(np.median(well_rmse)),
            "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
            "far_gt2000_rmse": math.sqrt(float(frame[f"{prefix}_far_sse"].sum()) / max(far_rows, 1)),
        }
    return metrics, frame


started = time.time()
feature_path = V4_DIR / "feature_names.json"
pair_path = V55_DIR / "hard_negative_pairs.parquet"
if not feature_path.is_file() or not pair_path.is_file():
    raise FileNotFoundError(
        "Run realistic_hard_negative_audit_v55_cell.py first; V4 feature names and V55 pairs are required"
    )

feature_names = json.loads(feature_path.read_text(encoding="utf-8"))
pairs = pd.read_parquet(pair_path)
pairs["well"] = pairs["well"].astype(str)
folds = _fold_table()
fold_by_well = dict(zip(folds["well"], folds["fold"].astype(int)))
pairs["fold"] = pairs["well"].map(fold_by_well)
if pairs["fold"].isna().any():
    raise RuntimeError("V55 pair wells do not align to fold metadata")

coverage = {
    "panel_wells": int(len(folds)),
    "train_folds45_wells": int(folds["fold"].isin(TRAIN_FOLDS).sum()),
    "selection_fold3_wells": int(folds["fold"].eq(SELECTION_FOLD).sum()),
    "application_fold2_wells": int(folds["fold"].eq(APPLICATION_FOLD).sum()),
}
print("HN2b CANONICAL COVERAGE", coverage, flush=True)
if (
    coverage["train_folds45_wells"] < 307
    or coverage["selection_fold3_wells"] < 154
    or coverage["application_fold2_wells"] < 154
):
    raise RuntimeError(f"Non-canonical fold coverage: {coverage}")

arms = {
    "evidence_only": np.asarray(
        [index for index, name in enumerate(feature_names) if not name.startswith("gen_")],
        np.int64,
    ),
    "full": np.arange(len(feature_names), dtype=np.int64),
}
train_pairs = pairs[pairs["fold"].isin(TRAIN_FOLDS)].copy()
selection_wells = folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].tolist()
application_wells = folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].tolist()

selection_rows = []
inner_models = {}
for arm, columns in arms.items():
    model = _fit(train_pairs, columns, f"inner_{arm}")
    metrics, _ = _evaluate(model, selection_wells, columns)
    inner_models[arm] = model
    selection_rows.append(
        {
            "arm": arm,
            "rmse": metrics["pairwise"]["rmse"],
            "legal_rmse": metrics["legal"]["rmse"],
            "oracle_rmse": metrics["oracle"]["rmse"],
            "gain": metrics["legal"]["rmse"] - metrics["pairwise"]["rmse"],
            "far_gain": metrics["legal"]["far_gt2000_rmse"] - metrics["pairwise"]["far_gt2000_rmse"],
        }
    )

selection = pd.DataFrame(selection_rows).sort_values(["rmse", "arm"], kind="stable")
selection.to_csv(OUT_DIR / "fold3_arm_selection.csv", index=False)
selected_arm = str(selection.iloc[0]["arm"])
selected_columns = arms[selected_arm]
print("\nHN2 FOLD3 ARM SELECTION")
print(selection.to_string(index=False), flush=True)
print("HN2 SELECTED", selected_arm, flush=True)

outer_pairs = pairs[pairs["fold"].isin((3, 4, 5))].copy()
outer_model = _fit(outer_pairs, selected_columns, f"outer_{selected_arm}")
outer_model.booster_.save_model(str(OUT_DIR / "pairwise_model.txt"))
(OUT_DIR / "selected_features.json").write_text(
    json.dumps([feature_names[index] for index in selected_columns], indent=2), encoding="utf-8"
)

fold2, fold2_frame = _evaluate(outer_model, application_wells, selected_columns, "aligned")
shuffle, _ = _evaluate(
    outer_model, application_wells, selected_columns, "candidate_alignment_shuffle"
)
fold2_frame.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)

gain = float(fold2["legal"]["rmse"] - fold2["pairwise"]["rmse"])
far_gain = float(
    fold2["legal"]["far_gt2000_rmse"] - fold2["pairwise"]["far_gt2000_rmse"]
)
shuffle_loss = float(shuffle["pairwise"]["rmse"] - fold2["pairwise"]["rmse"])
gates = {
    "menu_oracle_rmse_le_5p5": fold2["oracle"]["rmse"] <= 5.5,
    "gain_vs_legal_ge_0p30": gain >= 0.30,
    "far_gain_ge_0p30": far_gain >= 0.30,
    "shuffle_loses_ge_0p20": shuffle_loss >= 0.20,
    "pairwise_rmse_le_9p8": fold2["pairwise"]["rmse"] <= 9.8,
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "matched_same_generator_pairwise_tournament_fold2_confirmation",
    "sample_unit": "complete physical well; symmetric real-path pair supervision",
    "legal_inputs_only": True,
    "protocol": {
        "inner_train_folds": list(TRAIN_FOLDS),
        "feature_arm_selection_fold": SELECTION_FOLD,
        "outer_refit_folds": [3, 4, 5],
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
    },
    "canonical_coverage": coverage,
    "training_pairs": {
        "inner": int(len(train_pairs)),
        "outer": int(len(outer_pairs)),
        "source": pair_path,
        "definition": "V55 same-generator score-matched real hard negatives",
    },
    "menu_size": MENU_SIZE,
    "selected_arm": selected_arm,
    "selected_feature_count": int(len(selected_columns)),
    "fold3_selection": selection.to_dict("records"),
    "fold2": fold2,
    "fold2_candidate_alignment_shuffle": shuffle,
    "gain_vs_legal": gain,
    "far_gain_vs_legal": far_gain,
    "shuffle_loss": shuffle_loss,
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> compare pairwise endpoint error correlation with the 6.381 lineage"
        if all(gates.values())
        else "FAIL -> realistic matched negatives are still not learnably separable; close the V3 bank"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nHN2 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
