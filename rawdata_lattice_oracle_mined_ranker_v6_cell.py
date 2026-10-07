"""Oracle-mined ranker on the exact frozen V4 deployment menu.

V5 trained on a generator-rank/random menu but inferred on the V4 coarse
top-384 menu.  This cell removes that train/inference mismatch.  Candidate
truth is used only to construct supervised labels on outer-train wells.
Validation and application candidate menus are entirely label-free.
"""

from pathlib import Path
import hashlib
import json
import math
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/oof_exports")
V4_DIR = ROOT / "rawdata_lattice_coarse_ranker_v4"
PANEL_DIR = V4_DIR / "panels"
OUT_DIR = ROOT / "rawdata_lattice_oracle_mined_ranker_v6"
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = OUT_DIR.name
TOPK = 384
TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
SEED = 20260725
MAX_ROUNDS = 700
EARLY_STOPPING = 80

INNER_COARSE_PATH = V4_DIR / "coarse_ranker.txt"
OUTER_COARSE_PATH = V4_DIR / "coarse_ranker_outer345.txt"


def _ensure_v4_artifacts():
    ready = (
        INNER_COARSE_PATH.exists()
        and OUTER_COARSE_PATH.exists()
        and (V4_DIR / "feature_names.json").exists()
        and len(list(PANEL_DIR.glob("*.npz"))) >= 617
    )
    if ready:
        print("V6 bootstrap: reusing complete V4 artifacts", flush=True)
        return

    candidates = [
        Path.cwd() / "rawdata_lattice_coarse_ranker_v4_cell.py",
        Path("/root/autodl-tmp/rawdata_lattice_coarse_ranker_v4_cell.py"),
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "V4 artifacts were deleted. Upload rawdata_segment_lattice_nested_v1_cell.py, "
            "rawdata_lattice_beam_survival_v2_cell.py, rawdata_lattice_mode_union_v3_cell.py, "
            "and rawdata_lattice_coarse_ranker_v4_cell.py to /root/autodl-tmp."
        )
    print("V6 bootstrap: rebuilding frozen V4 panels and coarse models", flush=True)
    namespace = {"__name__": "raw_lattice_v4_rebuild"}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    missing = [
        str(path)
        for path in (INNER_COARSE_PATH, OUTER_COARSE_PATH, V4_DIR / "feature_names.json")
        if not path.exists()
    ]
    panel_count = len(list(PANEL_DIR.glob("*.npz")))
    if missing or panel_count < 617:
        raise RuntimeError(
            f"V4 rebuild incomplete: missing={missing}, panels={panel_count}/617"
        )


_ensure_v4_artifacts()
FEATURE_NAMES = json.loads((V4_DIR / "feature_names.json").read_text(encoding="utf-8"))


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _load_panel(well):
    with np.load(PANEL_DIR / f"{well}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _fold_table():
    candidates = [
        ROOT / "rawdata_lattice_mode_union_v3" / "folds.csv",
        ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError("Cannot locate fold metadata")
    table = pd.read_csv(source)
    well_column = next(column for column in ("well", "Well_ID", "well_id") if column in table)
    fold_column = next(column for column in ("fold", "Fold") if column in table)
    table = table[[well_column, fold_column]].rename(columns={well_column: "well", fold_column: "fold"})
    table["well"] = table["well"].astype(str)
    eligible = {path.stem for path in PANEL_DIR.glob("*.npz")}
    return table[table["well"].isin(eligible)].drop_duplicates("well").reset_index(drop=True)


FOLDS = _fold_table()
FOLD_BY_WELL = dict(zip(FOLDS["well"], FOLDS["fold"]))
TRAIN_WELLS = [well for well, fold in FOLD_BY_WELL.items() if fold in TRAIN_FOLDS]
SELECTION_WELLS = [well for well, fold in FOLD_BY_WELL.items() if fold == SELECTION_FOLD]
APPLICATION_WELLS = [well for well, fold in FOLD_BY_WELL.items() if fold == APPLICATION_FOLD]


def _seed(text):
    return int.from_bytes(hashlib.sha256(text.encode("ascii")).digest()[:8], "little")


def _coarse_menu(panel, booster):
    score = booster.predict(panel["features"])
    indices = np.argsort(-score)[: min(TOPK, len(score))].astype(np.int64)
    return indices, np.asarray(score[indices], np.float32)


GR_COLUMNS = np.asarray([name.startswith("gr_") for name in FEATURE_NAMES], dtype=bool)


def _rank_fraction(values):
    order = np.argsort(values, axis=0, kind="stable")
    rank = np.empty_like(order, dtype=np.float32)
    columns = np.arange(values.shape[1])[None, :]
    rank[order, columns] = np.arange(len(values), dtype=np.float32)[:, None]
    return rank / max(len(values) - 1, 1)


def _augment(panel, indices, coarse_score, control="full", seed=0):
    raw = panel["features"][indices].astype(np.float32, copy=True)
    if control == "no_gr":
        raw[:, GR_COLUMNS] = np.median(raw[:, GR_COLUMNS], axis=0, keepdims=True)
        coarse_score = np.full_like(coarse_score, np.median(coarse_score))
    elif control == "shuffled_alignment":
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(raw))
        raw[:, GR_COLUMNS] = raw[permutation][:, GR_COLUMNS]
        coarse_score = coarse_score[permutation]
    elif control != "full":
        raise ValueError(control)

    median = np.median(raw, axis=0)
    q25 = np.quantile(raw, 0.25, axis=0)
    q75 = np.quantile(raw, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-3)
    relative = np.clip((raw - median) / scale, -12.0, 12.0).astype(np.float32)

    # Percentile ranks are particularly useful for emission channels whose
    # absolute scale varies strongly from well to well.
    ranks = _rank_fraction(raw).astype(np.float32)
    score_median = float(np.median(coarse_score))
    score_scale = max(float(np.quantile(coarse_score, 0.75) - np.quantile(coarse_score, 0.25)), 1e-3)
    score_z = np.clip((coarse_score - score_median) / score_scale, -12.0, 12.0)
    score_rank = _rank_fraction(coarse_score[:, None])[:, 0]
    extras = np.stack([score_z, score_rank], axis=1).astype(np.float32)
    return np.concatenate([raw, relative, ranks, extras], axis=1)


def _labels(mse, scheme):
    rmse = np.sqrt(np.maximum(np.asarray(mse, np.float64), 0.0))
    regret = rmse - float(rmse.min())
    if scheme == "fine":
        edges = np.asarray([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0])
        gains = np.asarray([30, 28, 25, 21, 17, 12, 7, 3, 1, 0], np.int32)
    elif scheme == "wide":
        edges = np.asarray([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
        gains = np.asarray([30, 27, 22, 15, 8, 3, 0], np.int32)
    else:
        raise ValueError(scheme)
    return gains[np.searchsorted(edges, regret, side="right")]


def _matrix(wells, booster, scheme):
    xs, ys, weights, groups = [], [], [], []
    coverage = []
    median_rows = float(np.median([int(_load_panel(well)["rows"][0]) for well in wells]))
    for number, well in enumerate(wells, start=1):
        panel = _load_panel(well)
        indices, coarse_score = _coarse_menu(panel, booster)
        label = _labels(panel["mse"][indices], scheme)
        xs.append(_augment(panel, indices, coarse_score))
        ys.append(label)
        groups.append(len(indices))
        query_weight = float(np.clip(int(panel["rows"][0]) / median_rows, 0.5, 2.0))
        weights.append(np.full(len(indices), query_weight, np.float32))
        oracle_global = int(np.argmin(panel["mse"]))
        coverage.append(bool(oracle_global in set(indices.tolist())))
        if number % 50 == 0 or number == len(wells):
            print(f"matrix {scheme} {number}/{len(wells)}", flush=True)
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(weights),
        groups,
        float(np.mean(coverage)),
    )


def _model(iterations=MAX_ROUNDS):
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[1, 8, 32],
        label_gain=list(range(31)),
        n_estimators=int(iterations),
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def _fit(train_wells, valid_wells, train_booster, valid_booster, scheme, iterations=None):
    x_train, y_train, w_train, g_train, train_coverage = _matrix(train_wells, train_booster, scheme)
    model = _model(MAX_ROUNDS if iterations is None else iterations)
    kwargs = {}
    valid_coverage = None
    if valid_wells:
        x_valid, y_valid, w_valid, g_valid, valid_coverage = _matrix(valid_wells, valid_booster, scheme)
        kwargs = {
            "eval_set": [(x_valid, y_valid)],
            "eval_group": [g_valid],
            "eval_sample_weight": [w_valid],
            "callbacks": [lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(25)],
        }
    model.fit(x_train, y_train, sample_weight=w_train, group=g_train, **kwargs)
    return model, train_coverage, valid_coverage


def _metrics(frame, prefix):
    rows = int(frame["rows"].sum())
    far_rows = int(frame["far_rows"].sum())
    sse = float(frame[f"{prefix}_sse"].sum())
    well_rmse = np.sqrt(frame[f"{prefix}_sse"] / frame["rows"])
    return {
        "rows": rows,
        "wells": int(frame["well"].nunique()),
        "exact_sse": sse,
        "rmse": math.sqrt(sse / rows),
        "mae": float(frame[f"{prefix}_sae"].sum() / rows),
        "median_well_rmse": float(np.median(well_rmse)),
        "p75_well_rmse": float(np.quantile(well_rmse, 0.75)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": math.sqrt(float(frame[f"{prefix}_far_sse"].sum()) / far_rows),
    }


def _evaluate(model, wells, booster, control="full"):
    rows = []
    for well in wells:
        panel = _load_panel(well)
        indices, coarse_score = _coarse_menu(panel, booster)
        values = _augment(panel, indices, coarse_score, control, _seed(f"{control}:{well}"))
        iteration = getattr(model, "best_iteration_", None)
        if iteration is not None and int(iteration) <= 0:
            iteration = None
        score = model.predict(values, num_iteration=iteration)
        chosen = int(indices[int(np.argmax(score))])
        oracle = int(indices[int(np.argmin(panel["mse"][indices]))])
        coarse = int(indices[0])
        rows.append({
            "well": well,
            "rows": int(panel["rows"][0]),
            "far_rows": int(panel["far_rows"][0]),
            "map_sse": float(panel["sse"][chosen]),
            "map_sae": float(panel["sae"][chosen]),
            "map_far_sse": float(panel["far_sse"][chosen]),
            "coarse_sse": float(panel["sse"][coarse]),
            "coarse_sae": float(panel["sae"][coarse]),
            "coarse_far_sse": float(panel["far_sse"][coarse]),
            "oracle_sse": float(panel["sse"][oracle]),
            "oracle_sae": float(panel["sae"][oracle]),
            "oracle_far_sse": float(panel["far_sse"][oracle]),
            "chosen_regret_rmse": float(np.sqrt(panel["mse"][chosen]) - np.sqrt(panel["mse"][oracle])),
        })
    frame = pd.DataFrame(rows)
    return {key: _metrics(frame, key) for key in ("map", "coarse", "oracle")}, frame


started = time.time()
inner_booster = lgb.Booster(model_file=str(INNER_COARSE_PATH))
outer_booster = lgb.Booster(model_file=str(OUTER_COARSE_PATH))

selection_rows = []
inner_models = {}
for scheme in ("fine", "wide"):
    print(f"\nV6 INNER {scheme}", flush=True)
    model, train_coverage, valid_coverage = _fit(
        TRAIN_WELLS, SELECTION_WELLS, inner_booster, inner_booster, scheme
    )
    result, frame = _evaluate(model, SELECTION_WELLS, inner_booster)
    frame.to_csv(OUT_DIR / f"fold3_{scheme}_bywell.csv", index=False)
    inner_models[scheme] = model
    selection_rows.append({
        "scheme": scheme,
        "best_iteration": int(model.best_iteration_),
        "train_menu_global_oracle_coverage": train_coverage,
        "valid_menu_global_oracle_coverage": valid_coverage,
        **{f"map_{key}": value for key, value in result["map"].items()},
        "coarse_rmse": result["coarse"]["rmse"],
        "oracle_rmse": result["oracle"]["rmse"],
    })

selection = pd.DataFrame(selection_rows).sort_values("map_rmse").reset_index(drop=True)
selection.to_csv(OUT_DIR / "fold3_scheme_selection.csv", index=False)
print("\nV6 SCHEME SELECTION")
print(selection.to_string(index=False), flush=True)
selected_scheme = str(selection.iloc[0]["scheme"])
selected_iterations = int(selection.iloc[0]["best_iteration"])
inner_models[selected_scheme].booster_.save_model(str(MODEL_DIR / "best_inner.txt"))

outer_wells = TRAIN_WELLS + SELECTION_WELLS
print(f"\nV6 OUTER {selected_scheme} iterations={selected_iterations}", flush=True)
outer_model, outer_coverage, _ = _fit(
    outer_wells, [], outer_booster, outer_booster, selected_scheme, selected_iterations
)
outer_model.booster_.save_model(str(MODEL_DIR / "outer345.txt"))

fold2_full, fold2_frame = _evaluate(outer_model, APPLICATION_WELLS, outer_booster, "full")
fold2_no_gr, _ = _evaluate(outer_model, APPLICATION_WELLS, outer_booster, "no_gr")
fold2_shuffle, _ = _evaluate(outer_model, APPLICATION_WELLS, outer_booster, "shuffled_alignment")
fold2_frame.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)

controls = {
    "no_gr_degradation": fold2_no_gr["map"]["rmse"] - fold2_full["map"]["rmse"],
    "shuffled_alignment_degradation": fold2_shuffle["map"]["rmse"] - fold2_full["map"]["rmse"],
}
gates = {
    "top384_oracle_rmse_le_4p9": fold2_full["oracle"]["rmse"] <= 4.9,
    "gain_vs_v4_coarse_ge_0p5": fold2_full["coarse"]["rmse"] - fold2_full["map"]["rmse"] >= 0.5,
    "map_rmse_le_9p5": fold2_full["map"]["rmse"] <= 9.5,
    "each_control_degradation_ge_0p5": min(controls.values()) >= 0.5,
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "exact_deployment_menu_oracle_mined_path_ranking",
    "sample_unit": "one complete physical well with 384 candidate paths",
    "legal_inputs_only": True,
    "candidate_truth_usage": "outer-train supervised relevance labels only",
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "inner_train_folds": list(TRAIN_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "outer_refit_folds": [3, 4, 5],
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
    },
    "topk": TOPK,
    "raw_features": len(FEATURE_NAMES),
    "augmented_features": len(FEATURE_NAMES) * 3 + 2,
    "selected_scheme": selected_scheme,
    "selected_iterations": selected_iterations,
    "outer_menu_global_oracle_coverage": outer_coverage,
    "selection": selection.to_dict(orient="records"),
    "fold2": fold2_full,
    "fold2_no_gr": fold2_no_gr,
    "fold2_shuffled_alignment": fold2_shuffle,
    "controls": controls,
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> candidate selection was train-menu limited; confirm on untouched Fold1"
        if all(gates.values())
        else "FAIL -> exact-menu supervised ranking still cannot read the V3 bank; stop candidate reranking"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV6 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR)
