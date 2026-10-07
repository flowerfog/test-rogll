from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/oof_exports")
HN2_SOURCE = Path("/root/autodl-tmp/realistic_pairwise_tournament_hn2_cell.py")
if not HN2_SOURCE.is_file():
    HN2_SOURCE = Path("realistic_pairwise_tournament_hn2_cell.py")
OUT_DIR = ROOT / "realistic_pairwise_crossfit_hn3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = OUT_DIR.name
EVALUATION_FOLDS = (2, 3, 4, 5)
SEED = 20260728


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


source = HN2_SOURCE.read_text(encoding="utf-8")
marker = "started = time.time()"
if marker not in source:
    raise RuntimeError(f"Cannot find definition marker in {HN2_SOURCE}")
H = {"__name__": "realistic_pairwise_hn2_definitions", "__file__": str(HN2_SOURCE)}
exec(compile(source.split(marker, 1)[0], str(HN2_SOURCE), "exec"), H)

V4_DIR = H["V4_DIR"]
V55_DIR = H["V55_DIR"]
feature_names = json.loads((V4_DIR / "feature_names.json").read_text(encoding="utf-8"))
pairs = pd.read_parquet(V55_DIR / "hard_negative_pairs.parquet")
pairs["well"] = pairs["well"].astype(str)

folds = H["_fold_table"]()
fold_by_well = dict(zip(folds["well"], folds["fold"].astype(int)))
pairs["fold"] = pairs["well"].map(fold_by_well)
if pairs["fold"].isna().any():
    raise RuntimeError("V55 pair wells do not align to canonical V4 folds")

columns = np.asarray(
    [index for index, name in enumerate(feature_names) if not name.startswith("gen_")],
    np.int64,
)
selected_feature_names = [feature_names[index] for index in columns]
(OUT_DIR / "selected_features.json").write_text(
    json.dumps(selected_feature_names, indent=2), encoding="utf-8"
)


def _summarize(frame):
    metrics = {}
    total_rows = int(frame["rows"].sum())
    far_rows = int(frame["far_rows"].sum())
    for prefix in ("pairwise", "legal", "oracle"):
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
            "far_gt2000_rmse": math.sqrt(
                float(frame[f"{prefix}_far_sse"].sum()) / max(far_rows, 1)
            ),
        }
    return metrics


def _evaluate(model, wells, control, held_fold):
    output = []
    rng = np.random.default_rng(SEED + 100 * held_fold)
    for number, well in enumerate(wells, 1):
        panel = H["_load_panel"](well)
        indices, legal_rank = H["_menu"](panel)
        values = H["_finite_features"](panel, columns)[indices]
        if control == "candidate_alignment_shuffle" and len(values) > 1:
            values = values[rng.permutation(len(values))].copy()
        score = H["_tournament"](model, values)
        chosen = int(indices[int(np.argmax(score))])
        legal = int(indices[int(np.argmin(legal_rank[indices]))])
        oracle = int(indices[int(np.argmin(panel["mse"][indices]))])
        sorted_score = np.sort(np.asarray(score, np.float64))
        margin = float(sorted_score[-1] - sorted_score[-2]) if len(sorted_score) > 1 else 0.0
        row = {
            "well": well,
            "fold": held_fold,
            "control": control,
            "rows": int(panel["rows"][0]),
            "far_rows": int(panel["far_rows"][0]),
            "pairwise_index": chosen,
            "legal_index": legal,
            "oracle_index": oracle,
            "tournament_margin": margin,
        }
        for prefix, index in (("pairwise", chosen), ("legal", legal), ("oracle", oracle)):
            row[f"{prefix}_sse"] = float(panel["sse"][index])
            row[f"{prefix}_sae"] = float(panel["sae"][index])
            row[f"{prefix}_far_sse"] = float(panel["far_sse"][index])
        output.append(row)
        if number % 50 == 0 or number == len(wells):
            print(f"HN3 fold={held_fold} {control} {number}/{len(wells)}", flush=True)
    frame = pd.DataFrame(output)
    return _summarize(frame), frame


started = time.time()
coverage = folds.groupby("fold")["well"].nunique().astype(int).to_dict()
# A handful of V4 panels can be absent because the raw record was ineligible.
# Keep the crossfit canonical at the fold level without requiring impossible
# 155/155 panel coverage in every fold.
required = {2: 150, 3: 150, 4: 150, 5: 150}
if any(coverage.get(fold, 0) < minimum for fold, minimum in required.items()):
    raise RuntimeError(f"Non-canonical HN3 coverage: {coverage}")
coverage_fraction = {
    fold: coverage.get(fold, 0) / expected
    for fold, expected in {2: 155, 3: 155, 4: 155, 5: 154}.items()
}
print(
    "HN3 CONFIG",
    {
        "fold_wells": coverage,
        "fold_coverage_fraction": coverage_fraction,
        "features": len(columns),
    },
    flush=True,
)

fold_reports = []
aligned_frames = []
shuffle_frames = []
for held_fold in EVALUATION_FOLDS:
    train_folds = tuple(fold for fold in EVALUATION_FOLDS if fold != held_fold)
    train_pairs = pairs[pairs["fold"].isin(train_folds)].copy()
    held_wells = folds.loc[folds["fold"].eq(held_fold), "well"].tolist()
    model = H["_fit"](train_pairs, columns, f"crossfit_hold{held_fold}")
    model.booster_.save_model(str(OUT_DIR / f"pairwise_hold{held_fold}.txt"))
    aligned, aligned_frame = _evaluate(model, held_wells, "aligned", held_fold)
    shuffled, shuffle_frame = _evaluate(
        model, held_wells, "candidate_alignment_shuffle", held_fold
    )
    aligned_frames.append(aligned_frame)
    shuffle_frames.append(shuffle_frame)
    report = {
        "fold": held_fold,
        "train_folds": list(train_folds),
        "training_pairs": int(len(train_pairs)),
        "wells": int(len(held_wells)),
        "pairwise_rmse": aligned["pairwise"]["rmse"],
        "legal_rmse": aligned["legal"]["rmse"],
        "oracle_rmse": aligned["oracle"]["rmse"],
        "gain": aligned["legal"]["rmse"] - aligned["pairwise"]["rmse"],
        "far_gain": aligned["legal"]["far_gt2000_rmse"]
        - aligned["pairwise"]["far_gt2000_rmse"],
        "shuffle_rmse": shuffled["pairwise"]["rmse"],
        "shuffle_loss": shuffled["pairwise"]["rmse"] - aligned["pairwise"]["rmse"],
    }
    fold_reports.append(report)
    print("HN3 FOLD", report, flush=True)

aligned_frame = pd.concat(aligned_frames, ignore_index=True)
shuffle_frame = pd.concat(shuffle_frames, ignore_index=True)
aligned_frame.to_parquet(OUT_DIR / "crossfit_bywell.parquet", index=False)
shuffle_frame.to_parquet(OUT_DIR / "crossfit_shuffle_bywell.parquet", index=False)
pd.DataFrame(fold_reports).to_csv(OUT_DIR / "folds.csv", index=False)

overall = _summarize(aligned_frame)
overall_shuffle = _summarize(shuffle_frame)
gain = overall["legal"]["rmse"] - overall["pairwise"]["rmse"]
far_gain = (
    overall["legal"]["far_gt2000_rmse"]
    - overall["pairwise"]["far_gt2000_rmse"]
)
shuffle_loss = overall_shuffle["pairwise"]["rmse"] - overall["pairwise"]["rmse"]
fold_gains = [row["gain"] for row in fold_reports]
fold_far_gains = [row["far_gain"] for row in fold_reports]
checks = {
    "overall_gain_ge_0p30": gain >= 0.30,
    "overall_far_gain_ge_0p30": far_gain >= 0.30,
    "positive_folds_ge_3": sum(value > 0 for value in fold_gains) >= 3,
    "minimum_fold_gain_ge_minus0p10": min(fold_gains) >= -0.10,
    "minimum_fold_far_gain_ge_minus0p10": min(fold_far_gains) >= -0.10,
    "shuffle_loses_ge_0p20": shuffle_loss >= 0.20,
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "symmetric_fourfold_same_generator_pairwise_crossfit",
    "sample_unit": "complete physical well",
    "legal_inputs_only": True,
    "protocol": {
        "evaluation_folds": list(EVALUATION_FOLDS),
        "three_folds_train_each_model": True,
        "feature_arm": "evidence_only_frozen_from_hn2b",
        "fold1_touched": False,
    },
    "coverage": coverage,
    "coverage_fraction": coverage_fraction,
    "selected_feature_count": int(len(columns)),
    "folds": fold_reports,
    "overall": overall,
    "overall_shuffle": overall_shuffle,
    "gain_vs_legal": gain,
    "far_gain_vs_legal": far_gain,
    "shuffle_loss": shuffle_loss,
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> reconstruct row-level crossfit endpoint and test fixed-dose fusion against exact mainline lineage"
        if all(checks.values())
        else "FAIL -> pairwise selector is fold-unstable; retain only the hard-negative audit assets"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nHN3 FOLDS")
print(pd.DataFrame(fold_reports).to_string(index=False), flush=True)
print("\nHN3 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
