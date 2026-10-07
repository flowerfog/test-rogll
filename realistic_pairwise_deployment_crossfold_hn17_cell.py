"""HN17: clean Fold2/Fold3 deployment-equivalent HN selector gateway.

For each held fold, fit four pairwise scorers.  Every scorer excludes the held
fold and one different auxiliary fold, matching the four-model deployment
shape while keeping every evaluation well unseen.  Candidate-wise median score
is frozen from HN13.  Fold4/5 are never evaluated because the frozen candidate
bank prior was fitted on those folds.
"""

from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
HN2_SOURCE = ROOT / "realistic_pairwise_tournament_hn2_cell.py"
if not HN2_SOURCE.is_file():
    HN2_SOURCE = Path("realistic_pairwise_tournament_hn2_cell.py")
HN3_DIR = OOF_ROOT / "realistic_pairwise_crossfit_hn3"
V55_DIR = OOF_ROOT / "realistic_hard_negative_audit_v55"
HN12_DIR = OOF_ROOT / "realistic_pairwise_deployment_aggregation_hn12"
OUT_DIR = OOF_ROOT / "realistic_pairwise_deployment_crossfold_hn17"
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = OUT_DIR.name
EVALUATION_FOLDS = (2, 3)
ALL_FOLDS = (1, 2, 3, 4, 5)
SEED = 20260730
MIN_TRAIN_PAIR_WELLS = 200


def _clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _summarize(frame):
    rows = int(frame["rows"].sum())
    far_rows = int(frame["far_rows"].sum())
    output = {}
    for prefix in ("deployment", "legal", "oracle"):
        sse = float(frame[f"{prefix}_sse"].sum())
        well_rmse = np.sqrt(frame[f"{prefix}_sse"] / frame["rows"])
        output[prefix] = {
            "rows": rows,
            "wells": int(len(frame)),
            "exact_sse": sse,
            "rmse": math.sqrt(sse / rows),
            "mae": float(frame[f"{prefix}_sae"].sum() / rows),
            "median_well_rmse": float(np.median(well_rmse)),
            "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
            "far_gt2000_rmse": math.sqrt(
                float(frame[f"{prefix}_far_sse"].sum()) / max(far_rows, 1)
            ),
        }
    return output


def _model_scores(models, values, tournament):
    return np.vstack([tournament(model, values) for model in models])


def _evaluate(models, wells, held_fold, columns, H, control):
    output = []
    rng = np.random.default_rng(SEED + 1000 * held_fold)
    for number, well in enumerate(wells, 1):
        panel = H["_load_panel"](well)
        indices, legal_rank = H["_menu"](panel)
        values = H["_finite_features"](panel, columns)[indices]
        if control == "candidate_alignment_shuffle" and len(values) > 1:
            values = values[rng.permutation(len(values))].copy()
        scores = _model_scores(models, values, H["_tournament"])
        aggregate = np.median(scores, axis=0)
        deployment = int(indices[int(np.argmax(aggregate))])
        legal = int(indices[int(np.argmin(legal_rank[indices]))])
        oracle = int(indices[int(np.argmin(panel["mse"][indices]))])
        sorted_score = np.sort(aggregate)
        row = {
            "well": str(well),
            "fold": int(held_fold),
            "control": control,
            "rows": int(panel["rows"][0]),
            "far_rows": int(panel["far_rows"][0]),
            "deployment_index": deployment,
            "legal_index": legal,
            "oracle_index": oracle,
            "score_margin": float(sorted_score[-1] - sorted_score[-2]),
            "unique_model_winners": int(len(np.unique(np.argmax(scores, axis=1)))),
        }
        for prefix, index in (
            ("deployment", deployment), ("legal", legal), ("oracle", oracle)
        ):
            row[f"{prefix}_sse"] = float(panel["sse"][index])
            row[f"{prefix}_sae"] = float(panel["sae"][index])
            row[f"{prefix}_far_sse"] = float(panel["far_sse"][index])
        output.append(row)
        if number % 25 == 0 or number == len(wells):
            print(
                f"HN17 fold={held_fold} {control} {number}/{len(wells)}",
                flush=True,
            )
    frame = pd.DataFrame(output)
    return _summarize(frame), frame


started = time.time()
if not HN2_SOURCE.is_file():
    raise FileNotFoundError(HN2_SOURCE)
source = HN2_SOURCE.read_text(encoding="utf-8")
marker = "started = time.time()"
if marker not in source:
    raise RuntimeError(f"Cannot find definition marker in {HN2_SOURCE}")
H = {"__name__": "hn17_hn2_definitions", "__file__": str(HN2_SOURCE)}
exec(compile(source.split(marker, 1)[0], str(HN2_SOURCE), "exec"), H)

def _recover_feature_names():
    direct = H["V4_DIR"] / "feature_names.json"
    if direct.is_file():
        names = json.loads(direct.read_text(encoding="utf-8"))
        source_path = direct
    else:
        metadata_paths = sorted((H["V4_DIR"] / "panels").glob("*.json"))
        names = None
        source_path = None
        for metadata_path in metadata_paths:
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            candidate = payload.get("feature_names")
            if isinstance(candidate, list) and candidate:
                names = candidate
                source_path = metadata_path
                break
        if names is None:
            raise FileNotFoundError(
                f"Neither {direct} nor panel metadata with feature_names exists"
            )
    first_panel_path = next(iter(sorted((H["V4_DIR"] / "panels").glob("*.npz"))), None)
    if first_panel_path is None:
        raise FileNotFoundError(H["V4_DIR"] / "panels")
    with np.load(first_panel_path, allow_pickle=False) as panel:
        feature_count = int(panel["features"].shape[1])
    if len(names) != feature_count or len(set(names)) != len(names):
        raise RuntimeError(
            f"Recovered feature schema mismatch: names/panel={len(names)}/{feature_count}"
        )
    print(
        "HN17 FEATURE SCHEMA",
        {"source": str(source_path), "features": len(names)},
        flush=True,
    )
    return names


feature_names = _recover_feature_names()


def _recover_selected_features(feature_names):
    direct = HN3_DIR / "selected_features.json"
    candidates = [direct]
    candidates.extend(OOF_ROOT.rglob("selected_features.json"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            names = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            isinstance(names, list)
            and len(names) == 122
            and len(set(names)) == 122
            and set(names).issubset(feature_names)
            and all(not name.startswith("gen_") for name in names)
        ):
            print(
                "HN17 SELECTED SCHEMA",
                {"source": str(path), "features": len(names)},
                flush=True,
            )
            return names
    # HN3's source rule was deterministic, not learned: retain every V4
    # evidence feature and remove only the nine generator-provenance columns.
    names = [name for name in feature_names if not name.startswith("gen_")]
    if len(names) != 122:
        raise RuntimeError(
            f"Deterministic HN3 feature recovery produced {len(names)}, expected 122"
        )
    print(
        "HN17 SELECTED SCHEMA",
        {"source": "deterministic_not_gen_rule", "features": len(names)},
        flush=True,
    )
    return names


selected_features = _recover_selected_features(feature_names)
feature_index = {name: index for index, name in enumerate(feature_names)}
missing = [name for name in selected_features if name not in feature_index]
if missing:
    raise RuntimeError(f"Missing selected features: {missing[:10]}")
columns = np.asarray([feature_index[name] for name in selected_features], np.int64)


def _ensure_fold3_panels():
    v4_source = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
    if not v4_source.is_file():
        v4_source = Path("rawdata_lattice_coarse_ranker_v4_cell.py")
    if not v4_source.is_file():
        raise FileNotFoundError(v4_source)
    text = v4_source.read_text(encoding="utf-8")
    execution_marker = "# --- V4 EXECUTION ---"
    if execution_marker not in text:
        raise RuntimeError(f"V4 execution marker missing from {v4_source}")
    G = {"__name__": "hn17_v4_definitions", "__file__": str(v4_source)}
    exec(compile(text.split(execution_marker, 1)[0], str(v4_source), "exec"), G)
    all_wells = [
        path.name.split("__", 1)[0]
        for path in sorted((G["DATA_ROOT"] / "train").glob("*__horizontal_well.csv"))
    ]
    stable = G["V"]["_stable_folds"](all_wells)
    stable_fold = dict(zip(stable["well"].astype(str), stable["fold"].astype(int)))
    fold3_wells = [well for well in all_wells if stable_fold[well] == 3]
    existing = [well for well in fold3_wells if G["_panel_path"](well).is_file()]
    if len(existing) >= 145:
        print(
            "HN17 PANEL RECOVERY",
            {"fold3_existing": len(existing), "rebuilt": 0},
            flush=True,
        )
        return

    prior_wells = [well for well in all_wells if stable_fold[well] in (4, 5)]
    records = {}
    failures = []
    needed = prior_wells + [well for well in fold3_wells if well not in set(existing)]
    for number, well in enumerate(needed, 1):
        try:
            records[well] = G["V"]["_load_record"](well)
        except Exception as exc:
            failures.append({"well": well, "fold": stable_fold[well], "reason": repr(exc)})
        if number % 50 == 0 or number == len(needed):
            print(
                f"HN17 panel records {number}/{len(needed)} failures={len(failures)}",
                flush=True,
            )
    usable_prior = [records[well] for well in prior_wells if well in records]
    if len(usable_prior) < 307:
        raise RuntimeError(f"Only {len(usable_prior)} usable Fold4/5 prior records")
    prior = G["V"]["_fit_prior"](usable_prior, G["SEGMENT_LENGTH_FT"])
    rebuild = [well for well in fold3_wells if well in records and well not in set(existing)]
    build_failures = []
    for number, well in enumerate(rebuild, 1):
        try:
            G["_build_panel"](records[well], prior, force=False)
        except Exception as exc:
            build_failures.append({"well": well, "reason": repr(exc)})
        if number % 20 == 0 or number == len(rebuild):
            print(
                f"HN17 rebuild Fold3 panels {number}/{len(rebuild)} "
                f"failures={len(build_failures)}",
                flush=True,
            )
    pd.DataFrame(failures + build_failures).to_csv(
        OUT_DIR / "panel_recovery_failures.csv", index=False
    )
    achieved = sum(G["_panel_path"](well).is_file() for well in fold3_wells)
    if achieved < 145:
        raise RuntimeError(
            f"Only {achieved} Fold3 panels after recovery: "
            f"{(failures + build_failures)[:5]}"
        )
    print(
        "HN17 PANEL RECOVERY",
        {"fold3_existing": len(existing), "rebuilt": achieved - len(existing)},
        flush=True,
    )


_ensure_fold3_panels()


def _complete_fold_table():
    train_dir = H["_find_data_root"]() / "train"
    all_wells = [
        path.name.split("__", 1)[0]
        for path in sorted(train_dir.glob("*__horizontal_well.csv"))
    ]
    if len(all_wells) != 773 or len(set(all_wells)) != 773:
        raise RuntimeError(f"Expected 773 original wells, got {len(all_wells)}")
    scored = []
    for well in all_wells:
        digest = hashlib.sha256(f"20260721:{well}".encode("ascii")).digest()
        scored.append((int.from_bytes(digest[:8], "little"), well))
    ordered = [well for _, well in sorted(scored)]
    fold_values = np.empty(len(ordered), dtype=np.int64)
    for fold, indices in enumerate(np.array_split(np.arange(len(ordered)), 5), 1):
        fold_values[indices] = fold
    return pd.DataFrame({"well": ordered, "fold": fold_values})


full_folds = _complete_fold_table()
fold_by_well = dict(zip(full_folds["well"], full_folds["fold"].astype(int)))
available_panels = {path.stem for path in (H["V4_DIR"] / "panels").glob("*.npz")}
folds = full_folds[full_folds["well"].isin(available_panels)].reset_index(drop=True)
pairs = pd.read_parquet(V55_DIR / "hard_negative_pairs.parquet")
pairs["well"] = pairs["well"].astype(str)
pairs["fold"] = pairs["well"].map(fold_by_well)
pairs = pairs.dropna(subset=["fold"]).copy()
pairs["fold"] = pairs["fold"].astype(int)


def _ensure_pair_panels(required_wells):
    missing_wells = sorted(
        well for well in set(required_wells)
        if not (H["V4_DIR"] / "panels" / f"{well}.npz").is_file()
    )
    if not missing_wells:
        print("HN17 PAIR PANEL RECOVERY", {"missing": 0, "rebuilt": 0}, flush=True)
        return
    v4_source = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
    if not v4_source.is_file():
        v4_source = Path("rawdata_lattice_coarse_ranker_v4_cell.py")
    text = v4_source.read_text(encoding="utf-8")
    execution_marker = "# --- V4 EXECUTION ---"
    if execution_marker not in text:
        raise RuntimeError(f"V4 execution marker missing from {v4_source}")
    G = {"__name__": "hn17_pair_panel_v4_definitions", "__file__": str(v4_source)}
    exec(compile(text.split(execution_marker, 1)[0], str(v4_source), "exec"), G)
    all_wells = [
        path.name.split("__", 1)[0]
        for path in sorted((G["DATA_ROOT"] / "train").glob("*__horizontal_well.csv"))
    ]
    stable = G["V"]["_stable_folds"](all_wells)
    stable_fold = dict(zip(stable["well"].astype(str), stable["fold"].astype(int)))
    prior_wells = [well for well in all_wells if stable_fold[well] in (4, 5)]
    needed = sorted(set(prior_wells).union(missing_wells))
    records = {}
    failures = []
    for number, well in enumerate(needed, 1):
        try:
            records[well] = G["V"]["_load_record"](well)
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 50 == 0 or number == len(needed):
            print(
                f"HN17 pair panel records {number}/{len(needed)} failures={len(failures)}",
                flush=True,
            )
    usable_prior = [records[well] for well in prior_wells if well in records]
    if len(usable_prior) < 307:
        raise RuntimeError(f"Only {len(usable_prior)} usable Fold4/5 prior records")
    prior = G["V"]["_fit_prior"](usable_prior, G["SEGMENT_LENGTH_FT"])
    build_failures = []
    for number, well in enumerate(missing_wells, 1):
        try:
            if well not in records:
                raise RuntimeError("record unavailable")
            G["_build_panel"](records[well], prior, force=False)
        except Exception as exc:
            build_failures.append({"well": well, "reason": repr(exc)})
        if number % 20 == 0 or number == len(missing_wells):
            print(
                f"HN17 rebuild pair panels {number}/{len(missing_wells)} "
                f"failures={len(build_failures)}",
                flush=True,
            )
    pd.DataFrame(failures + build_failures).to_csv(
        OUT_DIR / "pair_panel_recovery_failures.csv", index=False
    )
    unresolved = [
        well for well in missing_wells
        if not (H["V4_DIR"] / "panels" / f"{well}.npz").is_file()
    ]
    if unresolved:
        raise RuntimeError(
            f"{len(unresolved)} required pair panels remain missing: {unresolved[:10]}"
        )
    print(
        "HN17 PAIR PANEL RECOVERY",
        {"missing": len(missing_wells), "rebuilt": len(missing_wells)},
        flush=True,
    )


_ensure_pair_panels(pairs["well"].astype(str).unique())

coverage = folds.groupby("fold")["well"].nunique().astype(int).to_dict()
if any(coverage.get(fold, 0) < 145 for fold in EVALUATION_FOLDS):
    raise RuntimeError(f"Insufficient clean application coverage: {coverage}")
print(
    "HN17 CONFIG",
    {
        "panel_fold_wells": coverage,
        "pair_wells": int(pairs["well"].nunique()),
        "features": len(columns),
        "evaluation_folds": EVALUATION_FOLDS,
    },
    flush=True,
)

fold_reports = []
choice_frames = []
shuffle_frames = []
for held_fold in EVALUATION_FOLDS:
    auxiliary_folds = tuple(fold for fold in ALL_FOLDS if fold != held_fold)
    models = []
    training_sets = []
    for auxiliary_fold in auxiliary_folds:
        train_folds = tuple(
            fold for fold in ALL_FOLDS if fold not in (held_fold, auxiliary_fold)
        )
        train_pairs = pairs[pairs["fold"].isin(train_folds)].copy()
        if train_pairs["well"].nunique() < MIN_TRAIN_PAIR_WELLS:
            raise RuntimeError(
                f"held={held_fold} aux={auxiliary_fold}: only "
                f"{train_pairs['well'].nunique()} pair wells; "
                f"minimum={MIN_TRAIN_PAIR_WELLS}"
            )
        model = H["_fit"](
            train_pairs,
            columns,
            f"hn17_hold{held_fold}_aux{auxiliary_fold}",
        )
        model.booster_.save_model(
            str(MODEL_DIR / f"pairwise_hold{held_fold}_aux{auxiliary_fold}.txt")
        )
        models.append(model)
        training_sets.append(
            {
                "auxiliary_fold": auxiliary_fold,
                "train_folds": list(train_folds),
                "pairs": int(len(train_pairs)),
                "wells": int(train_pairs["well"].nunique()),
            }
        )
    held_wells = folds.loc[folds["fold"].eq(held_fold), "well"].tolist()
    aligned, choices = _evaluate(
        models, held_wells, held_fold, columns, H, "aligned"
    )
    shuffled, shuffled_frame = _evaluate(
        models, held_wells, held_fold, columns, H, "candidate_alignment_shuffle"
    )
    choice_frames.append(choices)
    shuffle_frames.append(shuffled_frame)
    report = {
        "fold": held_fold,
        "wells": len(held_wells),
        "training_sets": training_sets,
        "deployment_rmse": aligned["deployment"]["rmse"],
        "legal_rmse": aligned["legal"]["rmse"],
        "oracle_rmse": aligned["oracle"]["rmse"],
        "gain_vs_legal": aligned["legal"]["rmse"] - aligned["deployment"]["rmse"],
        "far_gain_vs_legal": aligned["legal"]["far_gt2000_rmse"]
        - aligned["deployment"]["far_gt2000_rmse"],
        "shuffle_rmse": shuffled["deployment"]["rmse"],
        "shuffle_loss": shuffled["deployment"]["rmse"]
        - aligned["deployment"]["rmse"],
        "mean_unique_model_winners": float(choices["unique_model_winners"].mean()),
    }
    fold_reports.append(report)
    print("HN17 FOLD", report, flush=True)

choices = pd.concat(choice_frames, ignore_index=True)
shuffles = pd.concat(shuffle_frames, ignore_index=True)
choices.to_parquet(OUT_DIR / "deployment_choices_fold23.parquet", index=False)
shuffles.to_parquet(OUT_DIR / "deployment_shuffle_fold23.parquet", index=False)
pd.DataFrame(
    [{key: value for key, value in row.items() if key != "training_sets"} for row in fold_reports]
).to_csv(OUT_DIR / "folds.csv", index=False)

checks = {
    "fold2_wells_ge_145": int(coverage.get(2, 0)) >= 145,
    "fold3_wells_ge_145": int(coverage.get(3, 0)) >= 145,
    "both_oracles_le_6p5": max(row["oracle_rmse"] for row in fold_reports) <= 6.5,
    "both_shuffle_losses_ge_0p20": min(row["shuffle_loss"] for row in fold_reports) >= 0.20,
    "both_deployment_rmse_le_12p5": max(row["deployment_rmse"] for row in fold_reports) <= 12.5,
}
strict_pass = all(checks.values())
verdict = {
    "experiment": EXPERIMENT,
    "stage": "clean_fold2_fold3_four_model_score_median_selector_gateway",
    "selection_or_tuning_count": 0,
    "protocol": {
        "evaluation_folds": list(EVALUATION_FOLDS),
        "four_models_per_fold": True,
        "every_model_excludes_evaluation_fold": True,
        "each_model_additionally_excludes_one_auxiliary_fold": True,
        "aggregation": "candidate_wise_median_tournament_score",
        "minimum_training_pair_wells_per_model": MIN_TRAIN_PAIR_WELLS,
        "fold4_fold5_not_evaluated_due_to_frozen_prior_lineage": True,
    },
    "coverage": coverage,
    "selected_feature_count": len(columns),
    "folds": fold_reports,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> reconstruct selected paths row-wise and run outer residual confirmation"
        if strict_pass
        else "FAIL -> deployment selector is not cross-fold stable; stop HN16 local promotion"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2), encoding="utf-8"
)
print("\nHN17 FOLDS")
print(pd.read_csv(OUT_DIR / "folds.csv").to_string(index=False), flush=True)
print("\nHN17 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
