"""Multi-pseudo-PS candidate ranker gateway.

Run this as one AutoDL notebook cell after a complete multi-cut feature bank has
been built. It does not rerun PF/beam/fleongg feature generation.

The experiment answers one narrow question:

    Does training a candidate selector on multiple pseudo prediction starts
    improve selection on untouched official prediction starts?

The A/B is deliberately strict:

* single_ps ranker: official state from training wells only;
* multi_ps ranker: official + all available pseudo states from training wells;
* validation: official state of held-out wells only;
* grouping: every state from a held-out well stays outside training;
* application: fixed top-2 selector delta, shrunk by 0.25 and capped at 6 ft;
* no truth-derived value is present in the ranker inputs.

If candidate_pool_full_merged_oof_honest_fle.csv is available, the selected
candidate delta is also re-centered around CP12 and strong_bias. This is the
primary downstream test. The code writes only compact diagnostics and models;
it does not create another multi-million-row OOF file.
"""

from pathlib import Path
from collections import Counter
import gc
import json
import re
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from tqdm.auto import tqdm


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Fixed configuration.
# ---------------------------------------------------------------------------
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
CACHE_OVERRIDE = None
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
OUT_DIR = OOF_ROOT / "multips_candidate_ranker_v1"
MODEL_DIR = OUT_DIR / "models"

N_FOLDS = 5
SEED = 20260715
MIN_COMPLETE_WELLS = 700
MAX_STATES_PER_WELL = 5

# Pre-registered safe application. The raw selected-candidate delta is capped
# before shrinkage, so the final row movement cannot exceed 6 ft.
TOP_K = 2
SAFE_ALPHA = 0.25
RAW_MOVE_CAP = 24.0
APPLIED_MOVE_CAP = SAFE_ALPHA * RAW_MOVE_CAP

RANKER_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=650,
    learning_rate=0.035,
    num_leaves=31,
    min_child_samples=45,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.25,
    reg_lambda=8.0,
    max_bin=255,
    n_jobs=-1,
    verbosity=-1,
)

ABSOLUTE_CANDIDATES = (
    "pf_ancc",
    "pf_z",
    "likpf_scale_3",
    "likpf_scale_5",
    "likpf_scale_8",
    "likpf_scale_12",
    "likpf_mean",
)

DELTA_CANDIDATES = (
    "beam_cons_d",
    "beam_loose_d",
    "beam_vcons_d",
    "beam_sm5_d",
    "beam_vloose_d",
    "beam_mid_d",
    "beam_stiff_d",
    "beam_mean_d",
    "beam_med_d",
    "sc8_d",
    "sc15_d",
    "sc25_d",
    "sc_cons_d",
    "sc_ens_d",
    "hyb_d",
    "slp_b_d_all",
    "slp_b_d_50",
    "sig_mean_d",
)

STATE_SCALARS = (
    "known_len",
    "eval_len",
    "pfx_rmse",
    "cal_a",
    "cal_b",
    "slp_all",
    "slp_50",
    "slp_z",
    "ktvt_range",
    "ktvt_std",
    "tw_range",
    "tw_gr_mean",
    "sc_trust",
    "dense_rmse",
    "dense_bias",
    "dense_nb_std",
)

STATE_SEQUENCES = (
    "md_since",
    "frac",
    "dxy",
    "dz",
    "dzdmd",
    "beam_std_d",
    "sig_std",
    "pf_ancc_std",
    "sc8_sc",
    "sc15_sc",
    "sc25_sc",
    "spatial_knn_dist",
    "dense_dist",
)

QUALITY_COLUMN = {
    "pf_ancc": "pf_ancc_std",
    "sc8_d": "sc8_sc",
    "sc15_d": "sc15_sc",
    "sc25_d": "sc25_sc",
}

for directory in (OUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _rmse_from_sse(sse, rows):
    return float(np.sqrt(float(sse) / max(int(rows), 1)))


def _safe_array(values, fallback=0.0):
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = np.nan
    if np.isfinite(values).any():
        values = (
            pd.Series(values)
            .interpolate(limit_direction="both")
            .fillna(float(fallback))
            .to_numpy(float)
        )
    else:
        values = np.full(len(values), float(fallback), dtype=float)
    return values


def _summary(values, prefix):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        finite = np.array([0.0], dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_p10": float(np.quantile(finite, 0.10)),
        f"{prefix}_p50": float(np.quantile(finite, 0.50)),
        f"{prefix}_p90": float(np.quantile(finite, 0.90)),
        f"{prefix}_last": float(finite[-1]),
        f"{prefix}_span": float(np.ptp(finite)),
    }


def _candidate_path(cache_dir, wid, tag, full=False):
    feature_dir = cache_dir / "features"
    if full:
        return feature_dir / f"{wid}__official_full.pkl"
    return feature_dir / f"{wid}__{tag}__sample.pkl"


def _tag_sort_key(tag):
    if tag == "official":
        return (0, 0, tag)
    match = re.search(r"(\d+)$", tag)
    return (1, int(match.group(1)) if match else 999, tag)


def _inspect_cache(cache_dir):
    feature_dir = cache_dir / "features"
    meta_path = cache_dir / "well_metadata.csv"
    if not feature_dir.is_dir() or not meta_path.exists():
        return None
    try:
        meta = pd.read_csv(meta_path, dtype={"well": str})
    except Exception:
        return None
    if "well" not in meta or "fold" not in meta:
        return None
    wells = meta["well"].astype(str).unique().tolist()
    if len(wells) < MIN_COMPLETE_WELLS:
        return None

    counts = Counter()
    pattern = re.compile(r"^(.+?)__(.+?)__sample\.pkl$")
    for path in feature_dir.glob("*__*__sample.pkl"):
        match = pattern.match(path.name)
        if match:
            counts[match.group(2)] += 1
    full_count = sum(1 for _ in feature_dir.glob("*__official_full.pkl"))
    tags = [
        tag
        for tag, count in counts.items()
        if count >= MIN_COMPLETE_WELLS
    ]
    tags = sorted(tags, key=_tag_sort_key)
    if "official" not in tags or full_count < MIN_COMPLETE_WELLS:
        return None
    tags = ["official"] + [tag for tag in tags if tag != "official"]
    tags = tags[:MAX_STATES_PER_WELL]

    complete = []
    for wid in wells:
        required = [_candidate_path(cache_dir, wid, tag) for tag in tags]
        required.append(_candidate_path(cache_dir, wid, "official", full=True))
        if all(path.exists() for path in required):
            complete.append(wid)
    if len(complete) < MIN_COMPLETE_WELLS:
        return None

    # Matched empirical pseudo starts are preferred over fixed 35/50/65/75.
    preference = 100 if "matched_k3_gateway" in cache_dir.name else 0
    preference += 10 * len(tags) + len(complete) / 1000.0
    return {
        "cache": cache_dir,
        "meta": meta[meta["well"].isin(complete)].copy(),
        "tags": tags,
        "counts": dict(counts),
        "full_count": full_count,
        "complete_wells": complete,
        "score": preference,
    }


def _discover_cache():
    if CACHE_OVERRIDE is not None:
        result = _inspect_cache(Path(CACHE_OVERRIDE))
        if result is None:
            raise FileNotFoundError(f"Incomplete CACHE_OVERRIDE: {CACHE_OVERRIDE}")
        return result

    preferred = [
        OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1",
        OOF_ROOT / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75",
        OOF_ROOT / "multicut_fleongg_v0_k3_nospatial_r768_lgb1",
    ]
    candidates = []
    seen = set()
    for cache_dir in preferred + list(OOF_ROOT.glob("multicut_fleongg*")):
        key = str(cache_dir.resolve()) if cache_dir.exists() else str(cache_dir)
        if key in seen:
            continue
        seen.add(key)
        result = _inspect_cache(cache_dir)
        if result is not None:
            candidates.append(result)
    if not candidates:
        nearby = [str(path) for path in OOF_ROOT.glob("multicut_fleongg*")]
        raise FileNotFoundError(
            "No complete multi-cut feature bank was found. Need >=700 wells, "
            "official_full, and at least official+pseudo sample frames. Nearby:\n"
            + "\n".join(nearby[:50])
        )
    return max(candidates, key=lambda item: item["score"])


source = _discover_cache()
CACHE_DIR = source["cache"]
TAGS = tuple(source["tags"])
META = source["meta"].copy()
META["well"] = META["well"].astype(str)
META["fold"] = META["fold"].astype(int)
META = META[META["fold"].between(1, N_FOLDS)].reset_index(drop=True)
WELLS = META["well"].tolist()
FOLD_MAP = META.set_index("well")["fold"].to_dict()
N_EVAL_MAP = (
    META.set_index("well")["n_eval"].to_dict()
    if "n_eval" in META
    else {wid: 1.0 for wid in WELLS}
)

print(
    {
        "experiment": "multips_candidate_ranker_v1",
        "source_cache": str(CACHE_DIR),
        "wells": len(WELLS),
        "tags": TAGS,
        "cache_counts": source["counts"],
        "safe_apply": {
            "top_k": TOP_K,
            "alpha": SAFE_ALPHA,
            "raw_cap": RAW_MOVE_CAP,
            "applied_cap": APPLIED_MOVE_CAP,
        },
        "external_pool": str(POOL_PATH) if POOL_PATH.exists() else None,
    }
)


def _prepare_frame(frame):
    frame = frame.copy()
    if "_row_idx" in frame:
        frame = frame.sort_values("_row_idx")
    frame = frame.reset_index(drop=True)
    missing = {"last_known_tvt", "target"} - set(frame.columns)
    if missing:
        raise RuntimeError(f"feature frame missing {sorted(missing)}")
    return frame


def _available_candidates(frame):
    available = []
    available.extend(column for column in ABSOLUTE_CANDIDATES if column in frame)
    available.extend(column for column in DELTA_CANDIDATES if column in frame)
    return available


def _candidate_curves(frame, menu):
    frame = _prepare_frame(frame)
    last = _safe_array(frame["last_known_tvt"], 0.0)
    last_fallback = float(np.nanmedian(last)) if len(last) else 0.0
    curves = {}
    for name in menu:
        if name in ABSOLUTE_CANDIDATES:
            curves[name] = _safe_array(frame[name], last_fallback)
        else:
            delta = _safe_array(frame[name], 0.0)
            curves[name] = last + delta
    return last, curves


# Freeze one candidate menu available in every probed state and official-full
# frame. Exact duplicates are removed without looking at truth.
probe_wells = WELLS[: min(20, len(WELLS))]
probe_frames = []
for wid in probe_wells:
    for tag in TAGS:
        probe_frames.append(pd.read_pickle(_candidate_path(CACHE_DIR, wid, tag)))
    probe_frames.append(
        pd.read_pickle(_candidate_path(CACHE_DIR, wid, "official", full=True))
    )

menu_set = set(ABSOLUTE_CANDIDATES + DELTA_CANDIDATES)
for frame in probe_frames:
    menu_set &= set(_available_candidates(frame))
MENU = [
    name
    for name in ABSOLUTE_CANDIDATES + DELTA_CANDIDATES
    if name in menu_set
]

probe_curve_parts = {name: [] for name in MENU}
for frame in probe_frames[: min(20, len(probe_frames))]:
    _, curves = _candidate_curves(frame, MENU)
    for name in MENU:
        take = np.linspace(0, len(frame) - 1, min(64, len(frame))).round().astype(int)
        probe_curve_parts[name].append(curves[name][take])

deduped = []
for name in MENU:
    values = np.concatenate(probe_curve_parts[name])
    duplicate = False
    for previous in deduped:
        previous_values = np.concatenate(probe_curve_parts[previous])
        if float(np.sqrt(np.mean((values - previous_values) ** 2))) < 1e-5:
            duplicate = True
            break
    if not duplicate:
        deduped.append(name)
MENU = tuple(deduped)
del probe_frames, probe_curve_parts
gc.collect()

if len(MENU) < 8:
    raise RuntimeError(f"Only {len(MENU)} non-duplicate candidates: {MENU}")
print("candidate menu:", len(MENU), MENU)


def _manifest_fraction_map():
    path = CACHE_DIR / "cut_manifest.csv"
    if not path.exists():
        return {}
    manifest = pd.read_csv(path, dtype={"well": str})
    tag_column = next(
        (column for column in ("tag", "cut_tag", "_cut_tag") if column in manifest),
        None,
    )
    frac_column = next(
        (
            column
            for column in ("actual_frac", "known_frac", "cut_frac", "target_frac")
            if column in manifest
        ),
        None,
    )
    if tag_column is None or frac_column is None:
        return {}
    return {
        (str(row.well), str(getattr(row, tag_column))): float(
            getattr(row, frac_column)
        )
        for row in manifest.itertuples(index=False)
        if np.isfinite(float(getattr(row, frac_column)))
    }


FRAC_MAP = _manifest_fraction_map()


def _state_feature_block(frame, wid, tag, curves):
    frame = _prepare_frame(frame)
    last = _safe_array(frame["last_known_tvt"], 0.0)
    stack = np.column_stack([curves[name] for name in MENU])
    consensus = np.median(stack, axis=1)
    spread = np.max(stack, axis=1) - np.min(stack, axis=1)

    known_len = (
        float(np.nanmedian(frame["known_len"]))
        if "known_len" in frame
        else np.nan
    )
    eval_len = (
        float(np.nanmedian(frame["eval_len"]))
        if "eval_len" in frame
        else len(frame)
    )
    fallback_frac = known_len / max(known_len + eval_len, 1.0)
    cut_frac = FRAC_MAP.get((wid, tag), fallback_frac)

    block = {
        "state_rows": float(len(frame)),
        "cut_frac": float(cut_frac),
        "candidate_count": float(len(MENU)),
    }
    block.update(_summary(spread, "menu_spread"))
    block.update(_summary(consensus - last, "consensus_move"))

    for column in STATE_SCALARS:
        if column in frame:
            values = np.asarray(frame[column], dtype=float)
            finite = values[np.isfinite(values)]
            block[f"state_{column}"] = (
                float(np.median(finite)) if len(finite) else 0.0
            )
        else:
            block[f"state_{column}"] = 0.0

    for column in STATE_SEQUENCES:
        if column in frame:
            block.update(_summary(frame[column], f"state_{column}"))
        else:
            block.update(_summary(np.array([0.0]), f"state_{column}"))
    return block, last, consensus, stack


def _build_state_rows(frame, wid, tag):
    frame = _prepare_frame(frame)
    last, curves = _candidate_curves(frame, MENU)
    truth = last + np.asarray(frame["target"], dtype=float)
    block, last, consensus, stack = _state_feature_block(
        frame, wid, tag, curves
    )
    state_id = f"{wid}|{tag}"
    means = np.mean(stack - last[:, None], axis=0)
    mean_order = np.argsort(np.argsort(means)) / max(len(MENU) - 1, 1)
    rows = []

    for index, name in enumerate(MENU):
        curve = curves[name]
        move = curve - last
        versus_consensus = curve - consensus
        versus_pf = curve - curves.get("pf_ancc", consensus)
        versus_lik = curve - curves.get("likpf_scale_5", consensus)
        error = curve - truth
        first_difference = np.diff(curve)
        second_difference = np.diff(curve, n=2)

        row = {
            "well": wid,
            "tag": tag,
            "state_id": state_id,
            "fold": int(FOLD_MAP[wid]),
            "candidate": name,
            "candidate_order": index,
            "n_rows": len(frame),
            "official_n_eval": float(N_EVAL_MAP.get(wid, len(frame))),
            "label_rmse": float(np.sqrt(np.mean(error**2))),
            "label_mae": float(np.mean(np.abs(error))),
            "label_bias": float(np.mean(error)),
            "candidate_mean_rank": float(mean_order[index]),
            "candidate_smooth_d1_std": float(np.std(first_difference))
            if len(first_difference)
            else 0.0,
            "candidate_smooth_d1_p95": float(
                np.quantile(np.abs(first_difference), 0.95)
            )
            if len(first_difference)
            else 0.0,
            "candidate_smooth_d2_std": float(np.std(second_difference))
            if len(second_difference)
            else 0.0,
        }
        row.update(block)
        row.update(_summary(move, "candidate_move"))
        row.update(_summary(versus_consensus, "candidate_vs_consensus"))
        row.update(_summary(versus_pf, "candidate_vs_pf"))
        row.update(_summary(versus_lik, "candidate_vs_likpf5"))

        quality = QUALITY_COLUMN.get(name)
        if quality and quality in frame:
            row.update(_summary(frame[quality], "candidate_native_quality"))
        else:
            row.update(
                _summary(np.array([0.0]), "candidate_native_quality")
            )
        for menu_name in MENU:
            row[f"candidate_is_{menu_name}"] = float(name == menu_name)
        rows.append(row)

    output = pd.DataFrame(rows)
    rank = output["label_rmse"].rank(method="first", ascending=True).astype(int)
    output["label_relevance"] = len(MENU) - rank
    output["label_is_best"] = rank.eq(1).astype(int)
    return output


# ---------------------------------------------------------------------------
# Build compact candidate-state table from all sample frames.
# ---------------------------------------------------------------------------
state_parts = []
failures = []
build_started = time.time()
for wid in tqdm(WELLS, desc="candidate states"):
    for tag in TAGS:
        try:
            frame = pd.read_pickle(_candidate_path(CACHE_DIR, wid, tag))
            state_parts.append(_build_state_rows(frame, wid, tag))
        except Exception as exc:
            failures.append({"well": wid, "tag": tag, "reason": repr(exc)})

failure_frame = pd.DataFrame(failures)
failure_frame.to_csv(OUT_DIR / "state_failures.csv", index=False)
if failures:
    display(failure_frame.head(30))
    raise RuntimeError(f"Failed to build {len(failures)} candidate states")

states = pd.concat(state_parts, ignore_index=True, sort=False)
del state_parts
gc.collect()

metadata_columns = {
    "well",
    "tag",
    "state_id",
    "fold",
    "candidate",
    "candidate_order",
    "n_rows",
    "official_n_eval",
    "label_rmse",
    "label_mae",
    "label_bias",
    "label_relevance",
    "label_is_best",
}
FEATURE_COLUMNS = [
    column
    for column in states.columns
    if column not in metadata_columns
    and pd.api.types.is_numeric_dtype(states[column])
]

print(
    {
        "candidate_rows": len(states),
        "states": states["state_id"].nunique(),
        "wells": states["well"].nunique(),
        "features": len(FEATURE_COLUMNS),
        "build_minutes": (time.time() - build_started) / 60.0,
    }
)
print("state counts by tag:", states.groupby("tag")["state_id"].nunique().to_dict())


def _matrix(frame):
    return (
        frame.reindex(columns=FEATURE_COLUMNS)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _fit_ranker(train, fold, arm, n_tags):
    train = train.sort_values(
        ["state_id", "candidate_order"], kind="mergesort"
    ).reset_index(drop=True)
    group_sizes = train.groupby("state_id", sort=False).size().to_numpy(int)
    if not np.all(group_sizes == len(MENU)):
        raise RuntimeError(f"{arm}/fold{fold}: inconsistent rank groups")
    weights = (
        train["official_n_eval"].to_numpy(float)
        / max(n_tags, 1)
        / len(MENU)
    )
    params = dict(RANKER_PARAMS)
    params["random_state"] = SEED + fold + (100 if arm == "multi_ps" else 0)
    params["label_gain"] = list(range(len(MENU) + 1))
    model = LGBMRanker(**params)
    model.fit(
        _matrix(train),
        train["label_relevance"].to_numpy(int),
        group=group_sizes,
        sample_weight=weights,
    )
    joblib.dump(model, MODEL_DIR / f"{arm}_fold{fold}.joblib")
    return model


def _nested_fixed_candidate(train_official):
    report = (
        train_official.assign(
            weighted_sse=lambda frame: frame["label_rmse"] ** 2
            * frame["n_rows"]
        )
        .groupby("candidate", as_index=False)
        .agg(sse=("weighted_sse", "sum"), rows=("n_rows", "sum"))
    )
    report["rmse"] = np.sqrt(report["sse"] / report["rows"])
    return str(report.sort_values(["rmse", "candidate"]).iloc[0]["candidate"])


def _select_candidates(model, valid_states):
    valid_states = valid_states.copy()
    valid_states["score"] = model.predict(_matrix(valid_states))
    selections = {}
    for wid, group in valid_states.groupby("well", sort=False):
        ranked = group.sort_values(
            ["score", "candidate"], ascending=[False, True]
        )
        selections[str(wid)] = ranked["candidate"].head(TOP_K).tolist()
    return selections


# Optional downstream bases. Indexing by id is memory-heavy but avoids writing
# another 3.8M-row merge artifact and makes per-well validation deterministic.
pool_index = None
external_columns = []
if POOL_PATH.exists():
    header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
    external_columns = [
        column for column in ("cp12", "strong_bias", "strong") if column in header
    ]
    if external_columns:
        usecols = ["id"] + external_columns
        pool_index = pd.read_csv(POOL_PATH, usecols=usecols)
        pool_index["id"] = pool_index["id"].astype(str)
        for column in external_columns:
            pool_index[column] = pool_index[column].astype(np.float32)
        pool_index = pool_index.set_index("id", verify_integrity=True)
print("external bases:", external_columns)


metric_rows = []
selection_rows = []
fold_model_rows = []
cv_started = time.time()

for fold in range(1, N_FOLDS + 1):
    train_wells = set(META.loc[META["fold"] != fold, "well"])
    valid_wells = META.loc[META["fold"] == fold, "well"].tolist()
    train_common = states[states["well"].isin(train_wells)]
    train_official = train_common[train_common["tag"].eq("official")]
    valid_official = states[
        states["well"].isin(valid_wells) & states["tag"].eq("official")
    ]
    fixed_candidate = _nested_fixed_candidate(train_official)

    single_model = _fit_ranker(train_official, fold, "single_ps", 1)
    multi_model = _fit_ranker(train_common, fold, "multi_ps", len(TAGS))
    single_selection = _select_candidates(single_model, valid_official)
    multi_selection = _select_candidates(multi_model, valid_official)
    fold_model_rows.extend(
        [
            {
                "fold": fold,
                "arm": "single_ps",
                "train_wells": len(train_wells),
                "train_states": train_official["state_id"].nunique(),
                "fixed_candidate": fixed_candidate,
            },
            {
                "fold": fold,
                "arm": "multi_ps",
                "train_wells": len(train_wells),
                "train_states": train_common["state_id"].nunique(),
                "fixed_candidate": fixed_candidate,
            },
        ]
    )

    for wid in tqdm(valid_wells, desc=f"official selector fold {fold}"):
        frame = _prepare_frame(
            pd.read_pickle(_candidate_path(CACHE_DIR, wid, "official", full=True))
        )
        last, curves = _candidate_curves(frame, MENU)
        truth = last + np.asarray(frame["target"], dtype=float)
        fixed = curves[fixed_candidate]
        single_names = single_selection[wid]
        multi_names = multi_selection[wid]
        single_selected = np.mean(
            np.column_stack([curves[name] for name in single_names]), axis=1
        )
        multi_selected = np.mean(
            np.column_stack([curves[name] for name in multi_names]), axis=1
        )
        single_delta = np.clip(
            single_selected - fixed, -RAW_MOVE_CAP, RAW_MOVE_CAP
        )
        multi_delta = np.clip(
            multi_selected - fixed, -RAW_MOVE_CAP, RAW_MOVE_CAP
        )

        predictions = {
            "internal_fixed": fixed,
            "internal_single_safe": fixed + SAFE_ALPHA * single_delta,
            "internal_multi_safe": fixed + SAFE_ALPHA * multi_delta,
            "internal_single_hard": curves[single_names[0]],
            "internal_multi_hard": curves[multi_names[0]],
        }

        ids = (
            frame["id"].astype(str).to_numpy()
            if "id" in frame
            else np.array([f"{wid}_{i}" for i in range(len(frame))])
        )
        if pool_index is not None:
            external = pool_index.reindex(ids)
            if external[external_columns].isna().any().any():
                raise RuntimeError(f"Pool id mismatch for well {wid}")
            for base_name in external_columns:
                base = external[base_name].to_numpy(float)
                predictions[base_name] = base
                predictions[f"{base_name}_single_recentered"] = (
                    base + SAFE_ALPHA * single_delta
                )
                predictions[f"{base_name}_multi_recentered"] = (
                    base + SAFE_ALPHA * multi_delta
                )

        candidate_rmse = {
            name: float(np.sqrt(np.mean((curve - truth) ** 2)))
            for name, curve in curves.items()
        }
        oracle_name = min(candidate_rmse, key=candidate_rmse.get)
        selection_rows.append(
            {
                "well": wid,
                "fold": fold,
                "rows": len(frame),
                "fixed_candidate": fixed_candidate,
                "single_top1": single_names[0],
                "single_top2": "|".join(single_names),
                "multi_top1": multi_names[0],
                "multi_top2": "|".join(multi_names),
                "oracle_candidate": oracle_name,
                "oracle_rmse": candidate_rmse[oracle_name],
                "fixed_rmse": candidate_rmse[fixed_candidate],
                "single_top1_rmse": candidate_rmse[single_names[0]],
                "multi_top1_rmse": candidate_rmse[multi_names[0]],
                "single_top1_hit": int(single_names[0] == oracle_name),
                "multi_top1_hit": int(multi_names[0] == oracle_name),
                "single_top2_hit": int(oracle_name in single_names),
                "multi_top2_hit": int(oracle_name in multi_names),
            }
        )

        for variant, prediction in predictions.items():
            error = np.asarray(prediction, float) - truth
            reference = None
            if variant.startswith("internal_") and variant != "internal_fixed":
                reference = fixed
            else:
                for base_name in external_columns:
                    if variant.startswith(f"{base_name}_"):
                        reference = predictions[base_name]
                        break
            move = (
                np.asarray(prediction, float) - np.asarray(reference, float)
                if reference is not None
                else np.zeros(len(prediction), dtype=float)
            )
            metric_rows.append(
                {
                    "well": wid,
                    "fold": fold,
                    "variant": variant,
                    "rows": len(frame),
                    "sse": float(np.sum(error**2)),
                    "mean_abs_move": float(np.mean(np.abs(move))),
                    "max_abs_move": float(np.max(np.abs(move))),
                }
            )

    del single_model, multi_model
    gc.collect()

metrics = pd.DataFrame(metric_rows)
selections = pd.DataFrame(selection_rows)
model_report = pd.DataFrame(fold_model_rows)


def _aggregate_metrics(frame, grouping=None):
    group_columns = ["variant"] if grouping is None else [grouping, "variant"]
    report = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            rows=("rows", "sum"),
            sse=("sse", "sum"),
            mean_abs_move=("mean_abs_move", "mean"),
            max_abs_move=("max_abs_move", "max"),
        )
    )
    report["rmse"] = np.sqrt(report["sse"] / report["rows"])
    return report


overall = _aggregate_metrics(metrics)
fold_report = _aggregate_metrics(metrics, "fold")

if "cp12" in external_columns:
    PRIMARY_BASE = "cp12"
elif "strong_bias" in external_columns:
    PRIMARY_BASE = "strong_bias"
else:
    PRIMARY_BASE = "internal_fixed"
PRIMARY_SINGLE = f"{PRIMARY_BASE}_single_recentered" if PRIMARY_BASE != "internal_fixed" else "internal_single_safe"
PRIMARY_MULTI = f"{PRIMARY_BASE}_multi_recentered" if PRIMARY_BASE != "internal_fixed" else "internal_multi_safe"


def _score(report, variant):
    match = report[report["variant"].eq(variant)]
    if len(match) != 1:
        raise RuntimeError(f"Missing metric variant {variant}")
    return float(match.iloc[0]["rmse"])


base_rmse = _score(overall, PRIMARY_BASE)
single_rmse = _score(overall, PRIMARY_SINGLE)
multi_rmse = _score(overall, PRIMARY_MULTI)

fold_pivot = fold_report.pivot(index="fold", columns="variant", values="rmse")
fold_comparison = pd.DataFrame(
    {
        "fold": fold_pivot.index,
        "base_rmse": fold_pivot[PRIMARY_BASE],
        "single_rmse": fold_pivot[PRIMARY_SINGLE],
        "multi_rmse": fold_pivot[PRIMARY_MULTI],
    }
).reset_index(drop=True)
fold_comparison["single_gain_vs_base"] = (
    fold_comparison["base_rmse"] - fold_comparison["single_rmse"]
)
fold_comparison["multi_gain_vs_base"] = (
    fold_comparison["base_rmse"] - fold_comparison["multi_rmse"]
)
fold_comparison["multi_gain_vs_single"] = (
    fold_comparison["single_rmse"] - fold_comparison["multi_rmse"]
)

# Top40 is defined only from the untouched primary base, then frozen.
base_well = metrics[metrics["variant"].eq(PRIMARY_BASE)].copy()
top40 = set(base_well.nlargest(40, "sse")["well"])
top40_report = _aggregate_metrics(metrics[metrics["well"].isin(top40)])
top40_base = _score(top40_report, PRIMARY_BASE)
top40_single = _score(top40_report, PRIMARY_SINGLE)
top40_multi = _score(top40_report, PRIMARY_MULTI)

selection_summary = {
    "single_top1_accuracy": float(selections["single_top1_hit"].mean()),
    "multi_top1_accuracy": float(selections["multi_top1_hit"].mean()),
    "single_top2_recall": float(selections["single_top2_hit"].mean()),
    "multi_top2_recall": float(selections["multi_top2_hit"].mean()),
    "single_mean_regret": float(
        (selections["single_top1_rmse"] - selections["oracle_rmse"]).mean()
    ),
    "multi_mean_regret": float(
        (selections["multi_top1_rmse"] - selections["oracle_rmse"]).mean()
    ),
}

strict_pass = bool(
    base_rmse - multi_rmse >= 0.10
    and single_rmse - multi_rmse >= 0.03
    and (fold_comparison["multi_gain_vs_base"] > 0).all()
    and (fold_comparison["multi_gain_vs_single"] > 0).sum() >= 4
    and top40_base - top40_multi > 0
)

result = {
    "source_cache": str(CACHE_DIR),
    "tags": list(TAGS),
    "wells": len(WELLS),
    "candidate_menu": list(MENU),
    "candidate_features": len(FEATURE_COLUMNS),
    "primary_base": PRIMARY_BASE,
    "primary_single": PRIMARY_SINGLE,
    "primary_multi": PRIMARY_MULTI,
    "base_rmse": base_rmse,
    "single_ps_rmse": single_rmse,
    "multi_ps_rmse": multi_rmse,
    "single_gain_vs_base": base_rmse - single_rmse,
    "multi_gain_vs_base": base_rmse - multi_rmse,
    "multi_gain_vs_single": single_rmse - multi_rmse,
    "multi_min_fold_gain_vs_base": float(
        fold_comparison["multi_gain_vs_base"].min()
    ),
    "multi_min_fold_gain_vs_single": float(
        fold_comparison["multi_gain_vs_single"].min()
    ),
    "multi_positive_folds_vs_base": int(
        (fold_comparison["multi_gain_vs_base"] > 0).sum()
    ),
    "multi_positive_folds_vs_single": int(
        (fold_comparison["multi_gain_vs_single"] > 0).sum()
    ),
    "top40_gain_single_vs_base": top40_base - top40_single,
    "top40_gain_multi_vs_base": top40_base - top40_multi,
    "selection": selection_summary,
    "runtime_minutes": (time.time() - cv_started) / 60.0,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> build deployment candidate ranker and test through final chain"
        if strict_pass
        else "FAIL -> multi-pseudo-PS does not rescue candidate selection"
    ),
}

print("\nMODEL / BASELINE FOLDS")
display(model_report)
print("\nPRIMARY FOLDS")
display(fold_comparison)
print("\nALL VARIANTS")
display(overall.sort_values("rmse"))
print("\nSELECTION")
print(selection_summary)
print("\nRESULT")
print(result)
print("\nVERDICT:", result["next"])

model_report.to_csv(OUT_DIR / "model_report.csv", index=False)
fold_comparison.to_csv(OUT_DIR / "primary_fold_report.csv", index=False)
overall.to_csv(OUT_DIR / "overall_variants.csv", index=False)
top40_report.to_csv(OUT_DIR / "top40_variants.csv", index=False)
selections.to_csv(OUT_DIR / "selection_bywell.csv", index=False)
pd.DataFrame({"feature": FEATURE_COLUMNS}).to_csv(
    OUT_DIR / "feature_columns.csv", index=False
)
(OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
print("saved:", OUT_DIR)
