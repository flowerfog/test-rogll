"""M342-W1: in-domain self-supervised GR denoising observability gateway.

PCA is fitted only to standardised GR patches from competition Fold 4/5 wells;
no TVT target is used. Candidate states follow HN23's legal constant-offset
protocol. Fold 3 selects the retained PCA dimension/window, Fold 2 confirms it,
and a matched-range typewell shuffle attributes any recall gain.
"""

from pathlib import Path
import json
import sys
import time
import types
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


warnings.filterwarnings("ignore")
STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
HN23_SOURCE = ROOT / "multiscale_phase_event_observability_hn23_cell.py"
OUT = ROOT / "oof_exports" / "main342_indomain_gr_denoising_observability_w1"
OUT.mkdir(parents=True, exist_ok=True)

REPRESENTATION_FOLDS = (4, 5)
DISCOVERY_FOLD = 3
DISCOVERY_WELLS = 24
CONFIRMATION_FOLD = 2
UNTOUCHED_FOLD = 1
ROWS_PER_WELL = 32
HALF_WINDOWS = (128, 256)
PATCH_POINTS = 128
PCA_COMPONENTS = 32
COMPONENT_MENU = (4, 8, 16, 24, 32)
PATCHES_PER_SOURCE = 12
MIN_WINDOW_COVERAGE = 0.35
SEED = 20260804


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _load_hn23_definitions():
    source = HN23_SOURCE.read_text(encoding="utf-8")
    marker = "v66 = pd.read_parquet(V66_PATH)"
    if marker not in source:
        raise RuntimeError("HN23 execution marker not found")
    name = "m342_w1_hn23_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(HN23_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(HN23_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


H = _load_hn23_definitions()
DATA_ROOT = H["DATA_ROOT"]
V66_PATH = H["V66_PATH"]
OFFSETS = H["OFFSETS"].astype(np.float64)


def _resample(values, points=PATCH_POINTS):
    values = H["_fill"](values)
    if values is None or len(values) < 8:
        return None
    x = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    return np.interp(target, x, values)


def _standardise_rows(values):
    values = np.asarray(values, np.float64)
    median = np.median(values, axis=-1, keepdims=True)
    scale = np.median(np.abs(values - median), axis=-1, keepdims=True) * 1.4826
    fallback = np.std(values, axis=-1, keepdims=True)
    scale = np.where(scale > 1e-5, scale, fallback)
    return (values - median) / np.maximum(scale, 1e-5)


def _source_patches(values, count):
    values = np.asarray(values, np.float64)
    if len(values) < 64:
        return []
    width = min(max(64, len(values) // 8), 512)
    centres = np.unique(np.round(np.linspace(width // 2, len(values) - width // 2 - 1, count)).astype(int))
    output = []
    for centre in centres:
        left = max(0, centre - width // 2)
        right = min(len(values), centre + width // 2 + 1)
        local = values[left:right]
        if np.isfinite(local).mean() < MIN_WINDOW_COVERAGE:
            continue
        patch = _resample(local)
        if patch is not None:
            output.append(patch)
    return output


v66 = pd.read_parquet(V66_PATH)
prediction_column = H["_prediction_column"](v66)
if "well" not in v66:
    v66["well"] = v66["id"].astype(str).str.rsplit("_", n=1).str[0]
if "row_idx" not in v66:
    v66["row_idx"] = v66["id"].astype(str).str.rsplit("_", n=1).str[1].astype(int)
v66 = v66[["well", "row_idx", "fold", "tvt_true", prediction_column]].rename(
    columns={prediction_column: "base"}
)
v66["well"] = v66["well"].astype(str).str.lower()
fold_by_well = v66.groupby("well")["fold"].first().astype(int).to_dict()

representation_wells = sorted(
    well for well, fold in fold_by_well.items() if fold in REPRESENTATION_FOLDS
)
patches = []
patch_failures = []
for number, well in enumerate(representation_wells, 1):
    try:
        horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
        typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
        tw_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64)
        patches.extend(_source_patches(tw_gr, PATCHES_PER_SOURCE))
        visible = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
        h_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
        if visible.sum() >= 64:
            patches.extend(_source_patches(h_gr[np.flatnonzero(visible)], PATCHES_PER_SOURCE))
    except Exception as exc:
        patch_failures.append({"well": well, "reason": repr(exc)})
    if number % 75 == 0 or number == len(representation_wells):
        print(f"M342-W1 patches {number}/{len(representation_wells)} count={len(patches)}", flush=True)

patch_matrix = _standardise_rows(np.asarray(patches, np.float64))
if len(patch_matrix) < 5000 or patch_matrix.shape[1] != PATCH_POINTS:
    raise RuntimeError(f"Insufficient representation patches: {patch_matrix.shape}")
pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=SEED)
pca.fit(patch_matrix)
print("M342-W1 PCA", {
    "patches": len(patch_matrix),
    "components": PCA_COMPONENTS,
    "explained_variance": float(pca.explained_variance_ratio_.sum()),
}, flush=True)


def _scores(horizontal, candidates):
    horizontal = _standardise_rows(np.asarray(horizontal, np.float64)[None, :])[0]
    candidates = _standardise_rows(candidates)
    raw = H["_cosine_rows"](horizontal, candidates)
    h_score = pca.transform(horizontal[None, :])[0]
    q_score = pca.transform(candidates)
    output = {"raw_amplitude": raw}
    for components in COMPONENT_MENU:
        h_reconstruction = pca.mean_ + h_score[:components] @ pca.components_[:components]
        q_reconstruction = pca.mean_[None, :] + q_score[:, :components] @ pca.components_[:components]
        output[f"pca_{components}"] = H["_cosine_rows"](h_reconstruction, q_reconstruction)
    return output


def _evaluate(wells, control, shuffle_map=None, half_windows=HALF_WINDOWS):
    records = []
    failures = []
    for number, well in enumerate(wells, 1):
        local = v66.loc[v66["well"].eq(well)].sort_values("row_idx")
        before = len(records)
        try:
            override = shuffle_map.get(well) if shuffle_map else None
            base, horizontal_gr, truth, tw_tvt, tw_gr = H["_load_well"](well, local, override)
            positions = H["_sample_positions"](local["row_idx"].to_numpy(np.int64), ROWS_PER_WELL)
            for centre in positions:
                if not np.isfinite(truth[centre]):
                    continue
                truth_offset = float(truth[centre] - base[centre])
                if truth_offset < OFFSETS[0] or truth_offset > OFFSETS[-1]:
                    continue
                for half_window in half_windows:
                    left = max(0, int(centre) - int(half_window))
                    right = min(len(base), int(centre) + int(half_window) + 1)
                    horizontal_window = horizontal_gr[left:right]
                    if len(horizontal_window) < int(half_window) or np.isfinite(horizontal_window).mean() < MIN_WINDOW_COVERAGE:
                        continue
                    horizontal_patch = _resample(horizontal_window)
                    if horizontal_patch is None:
                        continue
                    candidate_tvt = base[left:right][None, :] + OFFSETS[:, None]
                    candidate_gr = np.interp(
                        candidate_tvt.ravel(), tw_tvt, tw_gr, left=np.nan, right=np.nan
                    ).reshape(len(OFFSETS), -1)
                    valid = np.isfinite(candidate_gr).mean(axis=1) >= 0.95
                    if valid.sum() < 64:
                        continue
                    candidate_gr = np.where(
                        np.isfinite(candidate_gr), candidate_gr,
                        np.nanmedian(candidate_gr, axis=1, keepdims=True),
                    )
                    candidate_patch = np.asarray([_resample(row) for row in candidate_gr], np.float64)
                    for config, score in _scores(horizontal_patch, candidate_patch).items():
                        score = np.asarray(score, np.float64)
                        score[~valid] = -np.inf
                        rank, winner, _ = H["_rank"](score, truth_offset)
                        records.append({
                            "control": control, "well": well,
                            "fold": int(local["fold"].iloc[0]), "row_idx": int(centre),
                            "half_window": int(half_window), "config": config,
                            "truth_offset": truth_offset,
                            "winner_offset": float(OFFSETS[winner]), "rank": rank,
                        })
        except Exception as exc:
            failures.append({"well": well, "control": control, "reason": repr(exc)})
        if len(records) == before and not any(item["well"] == well for item in failures):
            failures.append({"well": well, "control": control, "reason": "no_accepted_windows"})
        if number % 25 == 0 or number == len(wells):
            print(f"M342-W1 {control} {number}/{len(wells)} rows={len(records)} failures={len(failures)}", flush=True)
    return pd.DataFrame(records), failures


def _summary(records):
    rows = []
    for keys, local in records.groupby(["control", "half_window", "config"], sort=False):
        error = local["winner_offset"].to_numpy(np.float64) - local["truth_offset"].to_numpy(np.float64)
        rows.append({
            "control": keys[0], "half_window": int(keys[1]), "config": keys[2],
            "rows": len(local), "wells": local["well"].nunique(),
            "median_rank": float(local["rank"].median()),
            "top8_recall": float(local["rank"].lt(8).mean()),
            "top32_recall": float(local["rank"].lt(32).mean()),
            "argmax_rmse": float(np.sqrt(np.mean(error * error))),
        })
    return pd.DataFrame(rows)


fold3_wells = sorted(well for well, fold in fold_by_well.items() if fold == DISCOVERY_FOLD)
discovery_wells = fold3_wells[:DISCOVERY_WELLS]
confirmation_wells = sorted(well for well, fold in fold_by_well.items() if fold == CONFIRMATION_FOLD)
discovery, failures_a = _evaluate(discovery_wells, "aligned")
discovery_summary = _summary(discovery)
pca_discovery = discovery_summary[discovery_summary["config"].str.startswith("pca_")].sort_values(
    ["top32_recall", "top8_recall", "argmax_rmse"], ascending=[False, False, True]
)
if pca_discovery.empty:
    raise RuntimeError("No W1 discovery rows")
selected = pca_discovery.iloc[0].to_dict()
selected_config = str(selected["config"])
selected_window = int(selected["half_window"])

aligned_records, failures_b = _evaluate(
    confirmation_wells, "aligned", half_windows=(selected_window,)
)
centres = {}
for well in confirmation_wells:
    table = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv", usecols=["TVT"])
    centres[well] = float(pd.to_numeric(table["TVT"], errors="coerce").median())
ordered = sorted(confirmation_wells, key=lambda well: (centres[well], well))
shuffle_map = dict(zip(ordered, ordered[1:] + ordered[:1]))
shuffled_records, failures_c = _evaluate(
    confirmation_wells, "typewell_shuffle", shuffle_map, half_windows=(selected_window,)
)
confirmation_summary = _summary(pd.concat([aligned_records, shuffled_records], ignore_index=True))


def _row(control, config):
    local = confirmation_summary[
        confirmation_summary["control"].eq(control)
        & confirmation_summary["config"].eq(config)
    ]
    if len(local) != 1:
        raise RuntimeError(f"Missing W1 summary row: {control}/{config}")
    return local.iloc[0]


aligned = _row("aligned", selected_config)
raw = _row("aligned", "raw_amplitude")
shuffle = _row("typewell_shuffle", selected_config)
shuffle_raw = _row("typewell_shuffle", "raw_amplitude")
aligned_gain32 = float(aligned["top32_recall"] - raw["top32_recall"])
aligned_gain8 = float(aligned["top8_recall"] - raw["top8_recall"])
shuffle_gain32 = float(shuffle["top32_recall"] - shuffle_raw["top32_recall"])
checks = {
    "representation_patches_ge_5000": len(patch_matrix) >= 5000,
    "confirmation_wells_ge_145": int(aligned["wells"]) >= 145,
    "top32_gain_ge_0p03": aligned_gain32 >= 0.03,
    "top8_gain_ge_0p015": aligned_gain8 >= 0.015,
    "aligned_increment_exceeds_shuffle_by_0p015": aligned_gain32 - shuffle_gain32 >= 0.015,
    "argmax_rmse_improves_ge_2ft": float(raw["argmax_rmse"] - aligned["argmax_rmse"]) >= 2.0,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_indomain_gr_denoising_observability_w1",
    "stage": "fold45_selfsupervised_pca_fold3_selection_fold2_observability",
    "target_usage_in_representation": False,
    "protocol": {
        "representation_folds": REPRESENTATION_FOLDS,
        "discovery_fold": DISCOVERY_FOLD, "discovery_wells": len(discovery_wells),
        "confirmation_fold": CONFIRMATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
    },
    "representation": {
        "patches": len(patch_matrix), "points": PATCH_POINTS,
        "components": PCA_COMPONENTS,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    },
    "selected": {"config": selected_config, "half_window": selected_window},
    "confirmation": {
        "raw": raw.to_dict(), "denoised": aligned.to_dict(),
        "shuffle_raw": shuffle_raw.to_dict(), "shuffle_denoised": shuffle.to_dict(),
        "top32_gain": aligned_gain32, "top8_gain": aligned_gain8,
        "shuffle_top32_gain": shuffle_gain32,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "failures": {
        "patch": len(patch_failures), "discovery": len(failures_a),
        "aligned": len(failures_b), "shuffle": len(failures_c),
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> add the frozen denoised score to a candidate-bank ranker"
        if strict_pass else
        "FAIL -> in-domain denoising does not improve alias observability; close W1"
    ),
}

discovery_summary.to_csv(OUT / "discovery_summary.csv", index=False)
confirmation_summary.to_csv(OUT / "confirmation_summary.csv", index=False)
pd.concat([pd.DataFrame(failures_a), pd.DataFrame(failures_b), pd.DataFrame(failures_c)], ignore_index=True).to_csv(
    OUT / "failures.csv", index=False
)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-W1 DISCOVERY")
print(discovery_summary.to_string(index=False), flush=True)
print("\nM342-W1 CONFIRMATION")
print(confirmation_summary.to_string(index=False), flush=True)
print("\nM342-W1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
