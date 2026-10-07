"""Test-distribution-matched K3 pseudo-PS gateway for fleongg.

Run this as one notebook cell after the original fleongg notebook has defined
``build_well`` and ``lik_pf``. Do not run the original model-training cell.

The experiment compares two otherwise identical LightGBM models:

* single_cut: trained only on each training well's official prediction start;
* matched_k3: official plus two deterministic pseudo starts sampled from the
  empirical test-well known-fraction distribution.

Both models are evaluated only on held-out wells at their official prediction
start. Grouping is always by the original well. Spatial FI/DI features are
neutralised during feature construction and removed before fitting, because
their historical OOF lineage is fold-contaminated.
"""

from pathlib import Path
import gc
import hashlib
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Fixed, pre-registered v0 configuration.
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
CACHE_VERSION = "matched_k3_gateway_v1"
CACHE_DIR = OOF_DIR / f"multicut_fleongg_{CACHE_VERSION}"
FEATURE_DIR = CACHE_DIR / "features"
TEMP_DIR = CACHE_DIR / "masked_tmp"
MODEL_DIR = CACHE_DIR / "models"

N_FOLDS = 5
DONOR_CACHE = OOF_DIR / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75"
DONOR_FEATURE_DIR = DONOR_CACHE / "features"
PSEUDO_TAGS = ("pseudo1", "pseudo2")
K3_TAGS = ("official",) + PSEUDO_TAGS
TRAIN_ROWS_PER_CUT = 768
N_JOBS_FEATURES = 4
RANDOM_STATE = 20260711

# One deliberately conservative model. Full stack is allowed only after v0 passes.
LGB_PARAMS = dict(
    objective="regression",
    boosting_type="gbdt",
    n_estimators=2400,
    learning_rate=0.025,
    num_leaves=64,
    min_child_samples=80,
    subsample=0.80,
    subsample_freq=1,
    colsample_bytree=0.70,
    reg_alpha=0.10,
    reg_lambda=10.0,
    max_bin=255,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=-1,
)

# Fixed fleongg post-processing, identical for the A/B comparison.
PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_MODEL = 0.60
PP_W_PF_ANCC = 0.0
PP_LIKPF_COL = "likpf_scale_5"
PP_SG_WIN = 61
PP_SG_POLY = 3

PASS_CP12_NESTED = 0.05
ALPHA_GRID = np.round(np.arange(0.0, 0.5001, 0.025), 3)

required_globals = ["build_well", "lik_pf"]
missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Run the original fleongg function-definition cells first; missing: "
        + repr(missing_globals)
    )

for directory in (OOF_DIR, CACHE_DIR, FEATURE_DIR, TEMP_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print(
    {
        "experiment": "multicut_fleongg_matched_k3_gateway_v1",
        "cache": str(CACHE_DIR),
        "cuts_per_well": len(K3_TAGS),
        "cut_distribution": "empirical official-mask known_frac, two deterministic strata",
        "k3_tags": K3_TAGS,
        "k3_state_mass": {"official": 0.40, "pseudo1": 0.30, "pseudo2": 0.30},
        "official_donor_cache": str(DONOR_CACHE),
        "train_rows_per_cut": TRAIN_ROWS_PER_CUT,
        "feature_jobs": N_JOBS_FEATURES,
        "validation": "official cut only",
        "spatial_features": "neutralised and dropped",
    }
)


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _evenly_spaced_positions(n_rows, n_take):
    if n_rows <= n_take:
        return np.arange(n_rows, dtype=int)
    return np.unique(
        np.round(np.linspace(0, n_rows - 1, int(n_take))).astype(int)
    )


def _balanced_folds(meta):
    ordered = meta.sort_values(["n_eval", "well"], ascending=[False, True])
    loads = np.zeros(N_FOLDS, dtype=np.int64)
    assignment = {}
    for row in ordered.itertuples(index=False):
        fold = int(np.argmin(loads)) + 1
        assignment[str(row.well)] = fold
        loads[fold - 1] += int(row.n_eval)
    return assignment, loads


class _NeutralFormationImputer:
    """Finite placeholders for build_well; every derived column is dropped."""

    def impute(self, xy_q, self_wid=None):
        n = len(np.atleast_2d(xy_q))
        n_formations = len(globals().get("FORMATIONS", [0, 1, 2, 3, 4, 5]))
        offsets = np.arange(n_formations, dtype=np.float32) * 100.0
        values = np.broadcast_to(offsets, (n, n_formations)).copy()
        return values, np.full(n, 1e6, dtype=np.float32)


class _NeutralDenseImputer:
    def impute(self, xy_q, self_wid=None, k=None, nfetch=None):
        n = len(np.atleast_2d(xy_q))
        return (
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.full(n, 1e6, dtype=np.float32),
        )


SPATIAL_PREFIXES = (
    "tvtF_",
    "tvtFw_",
    "tvtF50_",
    "bw_",
    "bww_",
    "bw50_",
    "bw_early_",
    "bw_mid_",
    "frm_rmse_",
    "dense_",
    "tvt_dense",
)

SPATIAL_EXACT = {
    "form_mean_d",
    "form_std_d",
    "form_rng_d",
    "spatial_ancc_d",
    "spatial_knn_dist",
    "pf_vs_spatial",
    "pf_vs_dense",
    "spatial_vs_dense",
    "beam_vs_spatial",
    "sig_std",
    "sig_mean_d",
}


def _is_spatial_feature(column):
    return column in SPATIAL_EXACT or column.startswith(SPATIAL_PREFIXES)


def _add_likpf_to_frame(frame, masked_hw, typewell, pseudo_wid):
    """Build the four lik-PF columns with ids matching build_well output."""
    defaults = [
        "likpf_scale_3",
        "likpf_scale_5",
        "likpf_scale_8",
        "likpf_mean",
    ]
    last = frame["last_known_tvt"].to_numpy(np.float32)
    for column in defaults:
        frame[column] = last

    try:
        output, row_index, _ = lik_pf(masked_hw.copy(), typewell.copy())
        if not len(row_index):
            return frame
        lp = pd.DataFrame({"_row_idx": np.asarray(row_index, dtype=int)})
        generated = []
        for key, value in output.items():
            column = "likpf_" + key.replace("pf_scale_", "scale_").replace(
                "pf_mean", "mean"
            )
            lp[column] = np.asarray(value, dtype=np.float32)
            generated.append(column)
        frame = frame.drop(columns=[c for c in generated if c in frame.columns])
        frame = frame.merge(lp, on="_row_idx", how="left", validate="one_to_one")
        for column in defaults:
            if column not in frame:
                frame[column] = frame["last_known_tvt"]
            frame[column] = frame[column].fillna(frame["last_known_tvt"]).astype(
                np.float32
            )
        return frame
    except Exception as exc:
        warnings.warn(f"lik_pf fallback for {pseudo_wid}: {exc!r}")
        return frame


def _build_masked_source(source, cut_position, official):
    keep = ["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"]
    masked = source[keep].copy().reset_index(drop=True)
    if not official:
        truth = masked["TVT"].to_numpy(float)
        simulated = np.full(len(masked), np.nan, dtype=float)
        simulated[:cut_position] = truth[:cut_position]
        masked["TVT_input"] = simulated
    return masked


def _feature_paths(wid, tag):
    sample_path = FEATURE_DIR / f"{wid}__{tag}__sample.pkl"
    full_path = FEATURE_DIR / f"{wid}__official_full.pkl" if tag == "official" else None
    if tag == "official" and not (sample_path.exists() and full_path.exists()):
        donor_sample = DONOR_FEATURE_DIR / f"{wid}__official__sample.pkl"
        donor_full = DONOR_FEATURE_DIR / f"{wid}__official_full.pkl"
        if donor_sample.exists() and donor_full.exists():
            return donor_sample, donor_full
    return sample_path, full_path


def _build_one_task(task):
    wid = str(task["well"])
    tag = str(task["tag"])
    cut_position = int(task["cut_position"])
    official = bool(task["official"])
    official_n_eval = int(task["official_n_eval"])
    sample_path, full_path = _feature_paths(wid, tag)

    if sample_path.exists() and (not official or full_path.exists()):
        return {"well": wid, "tag": tag, "status": "cached"}

    horizontal_path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{wid}__typewell.csv"
    pseudo_wid = f"{wid}_mc_{tag}"
    temp_path = TEMP_DIR / f"{pseudo_wid}__horizontal_well.csv"

    try:
        source = pd.read_csv(horizontal_path)
        typewell = pd.read_csv(typewell_path).sort_values("TVT")
        masked = _build_masked_source(source, cut_position, official)
        masked.to_csv(temp_path, index=False)

        frame = build_well(str(temp_path), str(typewell_path), is_train=True)
        if frame is None or len(frame) < 64:
            raise RuntimeError("build_well returned too few rows")

        frame = frame.reset_index(drop=True)
        frame["_row_idx"] = (
            frame["id"].astype(str).str.rsplit("_", n=1).str[1].astype(int)
        )
        frame = _add_likpf_to_frame(frame, masked, typewell, pseudo_wid)
        frame["well"] = wid
        frame["id"] = wid + "_" + frame["_row_idx"].astype(str)
        frame["_cut_tag"] = tag
        frame["_official_n_eval"] = official_n_eval
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)

        if official:
            frame.to_pickle(full_path)

        positions = _evenly_spaced_positions(len(frame), TRAIN_ROWS_PER_CUT)
        sample = frame.iloc[positions].copy()
        state_mass = {"official": 0.40, "pseudo1": 0.30, "pseudo2": 0.30}[tag]
        sample["_sample_weight_k3"] = (
            state_mass * official_n_eval / max(1, len(sample))
        )
        sample["_sample_weight_single"] = official_n_eval / max(1, len(sample))
        sample.to_pickle(sample_path)
        return {"well": wid, "tag": tag, "status": "built", "rows": len(frame)}
    except Exception as exc:
        return {"well": wid, "tag": tag, "status": "failed", "reason": repr(exc)}
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Scan train/test starts and build two deterministic test-matched pseudo starts.
# ---------------------------------------------------------------------------
test_fracs = []
for path in sorted((DATA_ROOT / "test").glob("*__horizontal_well.csv")):
    try:
        values = pd.read_csv(path, usecols=["TVT_input"])["TVT_input"].to_numpy(float)
        known = np.isfinite(values)
        if known.any() and (~known).any():
            test_fracs.append(float(known.sum() / len(values)))
    except Exception:
        continue
test_fracs = np.asarray(test_fracs, dtype=float)
if len(test_fracs):
    print(
        "available demo-test known_frac (audit only):",
        pd.Series(test_fracs).describe(
            percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        ),
    )


def _stable_unit_interval(wid, slot):
    digest = hashlib.blake2b(f"{wid}:{slot}".encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64)


def _distinct_cut(base_cut, used, n_rows, unit):
    min_gap = max(32, int(round(0.02 * n_rows)))
    signs = (1, -1) if unit >= 0.5 else (-1, 1)
    offsets = [0]
    for multiplier in range(1, 7):
        offsets.extend(
            [signs[0] * multiplier * min_gap, signs[1] * multiplier * min_gap]
        )
    for offset in offsets:
        candidate = int(np.clip(base_cut + offset, 64, n_rows - 128))
        if all(abs(candidate - previous) >= min_gap for previous in used):
            return candidate
    # Extremely short/duplicated wells: choose the feasible point farthest
    # from already used cuts rather than silently duplicating an anchor state.
    grid = np.arange(64, max(65, n_rows - 127), max(1, min_gap // 2), dtype=int)
    distances = np.asarray([min(abs(point - previous) for previous in used) for point in grid])
    return int(grid[int(np.argmax(distances))])


meta_rows = []
train_files = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
for path in tqdm(train_files, desc="scan multi-cut metadata"):
    wid = path.name.replace("__horizontal_well.csv", "")
    try:
        tvt_input = pd.read_csv(path, usecols=["TVT_input"])["TVT_input"].to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        hidden = np.flatnonzero(~np.isfinite(tvt_input))
        if len(known) < 64 or len(hidden) < 128:
            continue
        official_cut = int(hidden[0])
        if not np.all(np.isfinite(tvt_input[:official_cut])):
            continue
        if not np.all(~np.isfinite(tvt_input[official_cut:])):
            continue
        meta_rows.append(
            {
                "well": wid,
                "n_rows": len(tvt_input),
                "official_cut": official_cut,
                "visible_frac": official_cut / len(tvt_input),
                "n_eval": len(hidden),
            }
        )
    except Exception:
        continue

meta = pd.DataFrame(meta_rows).sort_values("well").reset_index(drop=True)
if len(meta) < 700:
    raise RuntimeError(f"Only {len(meta)} eligible wells; expected roughly 773")

# AutoDL has only three public/demo test wells; hidden test inputs are injected
# in the Kaggle scoring environment. Three points cannot estimate a deployment
# distribution, so use the 773 official train masks as the honest local proxy.
if len(test_fracs) >= 100:
    reference_fracs = test_fracs
    cut_distribution_source = "test_empirical"
else:
    reference_fracs = meta["visible_frac"].to_numpy(float)
    cut_distribution_source = "train_official_empirical"

print(
    "pseudo-PS reference distribution:",
    cut_distribution_source,
    pd.Series(reference_fracs).describe(
        percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    ),
)

fold_map, fold_loads = _balanced_folds(meta)
meta["fold"] = meta["well"].map(fold_map).astype(int)
meta.to_csv(CACHE_DIR / "well_metadata.csv", index=False)

print("eligible wells:", len(meta))
print("fold row loads:", fold_loads.tolist())

tasks = []
for row in meta.itertuples(index=False):
    cuts = [("official", int(row.official_cut), True, np.nan, float(row.visible_frac))]
    used = {int(row.official_cut)}
    for slot, tag in enumerate(PSEUDO_TAGS):
        unit = _stable_unit_interval(str(row.well), slot)
        # One draw from each half of the empirical test CDF. Across 773 wells
        # this is continuous, reproducible, and exactly test-distribution matched.
        quantile = (slot + unit) / len(PSEUDO_TAGS)
        requested_frac = float(np.quantile(reference_fracs, quantile))
        base_cut = int(
            np.clip(round(requested_frac * row.n_rows), 64, row.n_rows - 128)
        )
        cut = _distinct_cut(base_cut, used, int(row.n_rows), unit)
        used.add(cut)
        cuts.append((tag, cut, False, quantile, requested_frac))
    for tag, cut, official, quantile, requested_frac in cuts:
        tasks.append(
            {
                "well": str(row.well),
                "tag": tag,
                "cut_position": int(cut),
                "official": bool(official),
                "official_n_eval": int(row.n_eval),
                "sample_quantile": quantile,
                "requested_frac": requested_frac,
                "actual_frac": float(cut / row.n_rows),
            }
        )

cut_manifest = pd.DataFrame(tasks)
cut_manifest.to_csv(CACHE_DIR / "cut_manifest.csv", index=False)
print(
    "matched cut fractions:",
    cut_manifest.groupby("tag")["actual_frac"].describe()[
        ["count", "mean", "min", "25%", "50%", "75%", "max"]
    ],
)


# ---------------------------------------------------------------------------
# Feature generation. Neutral globals make this independent of all-well FI/DI.
# ---------------------------------------------------------------------------
original_fi = globals().get("_FI", None)
original_di = globals().get("_DI", None)
globals()["_FI"] = _NeutralFormationImputer()
globals()["_DI"] = _NeutralDenseImputer()

feature_started = time.time()
try:
    from joblib import Parallel, delayed

    smoke_well = str(meta.iloc[0]["well"])
    smoke_tasks = [task for task in tasks if task["well"] == smoke_well]
    smoke_results = [_build_one_task(task) for task in smoke_tasks]
    smoke_failed = [row for row in smoke_results if row["status"] == "failed"]
    print("feature smoke test:", smoke_results, flush=True)
    if smoke_failed:
        raise RuntimeError("Feature smoke test failed: " + repr(smoke_failed))

    results = Parallel(n_jobs=N_JOBS_FEATURES, prefer="threads")(
        delayed(_build_one_task)(task)
        for task in tqdm(tasks, desc="multi-cut feature tasks")
    )
finally:
    globals()["_FI"] = original_fi
    globals()["_DI"] = original_di

feature_status = pd.DataFrame(results)
feature_status.to_csv(CACHE_DIR / "feature_status.csv", index=False)
print("feature status:", feature_status["status"].value_counts().to_dict())
print("feature elapsed hours:", (time.time() - feature_started) / 3600.0)

failed = feature_status[feature_status["status"] == "failed"]
if len(failed):
    display(failed.head(30))
if len(failed) > max(10, 0.02 * len(tasks)):
    raise RuntimeError(f"Too many feature failures: {len(failed)} / {len(tasks)}")

valid_wells = []
for wid in meta["well"].astype(str):
    required_paths = [_feature_paths(wid, tag)[0] for tag in K3_TAGS]
    official_full = _feature_paths(wid, "official")[1]
    if all(path.exists() for path in required_paths) and official_full.exists():
        valid_wells.append(wid)

meta = meta[meta["well"].isin(valid_wells)].reset_index(drop=True)
print("complete wells for training:", len(meta))
if len(meta) < 700:
    raise RuntimeError("Too few complete wells after feature generation")


def _load_samples(wells, tags, keep_columns=None):
    frames = []
    for wid in wells:
        for tag in tags:
            path, _ = _feature_paths(str(wid), tag)
            if path.exists():
                frame = pd.read_pickle(path)
                official_n_eval = float(frame["_official_n_eval"].iloc[0])
                state_mass = {"official": 0.40, "pseudo1": 0.30, "pseudo2": 0.30}[tag]
                frame["_sample_weight_k3"] = (
                    state_mass * official_n_eval / max(1, len(frame))
                )
                frame["_sample_weight_single"] = official_n_eval / max(1, len(frame))
                if keep_columns is not None:
                    for column in keep_columns:
                        if column not in frame.columns:
                            frame[column] = np.nan
                    frame = frame[list(keep_columns)].copy()
                frames.append(frame)
    if not frames:
        raise RuntimeError("No training samples loaded")
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_official_full(wells):
    frames = []
    for wid in wells:
        _, path = _feature_paths(str(wid), "official")
        frames.append(pd.read_pickle(path))
    return pd.concat(frames, ignore_index=True, sort=False)


def _feature_columns(frame):
    excluded = {
        "well",
        "id",
        "target",
        "_row_idx",
        "_cut_tag",
        "_official_n_eval",
        "_sample_weight_k3",
        "_sample_weight_single",
    }
    columns = []
    for column in frame.columns:
        if column in excluded or _is_spatial_feature(column):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def _matrix(frame, columns):
    return frame.reindex(columns=columns).to_numpy(np.float32)


def _postprocess(frame, model_delta):
    last = frame["last_known_tvt"].to_numpy(float)
    md_since = frame["md_since"].to_numpy(float)
    warm = 1.0 - np.exp(-np.maximum(md_since, 0.0) / PP_TAU)
    pf_delta = frame["pf_ancc"].to_numpy(float) - last
    if PP_LIKPF_COL in frame:
        lik_delta = frame[PP_LIKPF_COL].to_numpy(float) - last
    else:
        lik_delta = pf_delta
    learned = PP_ALPHA * warm * (
        model_delta * (1.0 - PP_W_PF_ANCC) + pf_delta * PP_W_PF_ANCC
    )
    prediction = last + PP_W_MODEL * learned + (1.0 - PP_W_MODEL) * lik_delta

    output = prediction.copy()
    offset = 0
    for _, group in frame.groupby("well", sort=False):
        n = len(group)
        values = prediction[offset : offset + n]
        window = min(PP_SG_WIN, n)
        if window % 2 == 0:
            window -= 1
        if window >= PP_SG_POLY + 2:
            output[offset : offset + n] = savgol_filter(values, window, PP_SG_POLY)
        offset += n
    return output


def _fit_lgb(train, valid, feature_columns, weight_column, model_name, fold):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    params = dict(LGB_PARAMS)
    params["device_type"] = "gpu"
    model = LGBMRegressor(**params)
    fit_kwargs = dict(
        sample_weight=train[weight_column].to_numpy(float),
        eval_set=[(_matrix(valid, feature_columns), valid["target"].to_numpy(float))],
        eval_metric="rmse",
        callbacks=[early_stopping(200, verbose=False), log_evaluation(100)],
    )
    try:
        model.fit(
            _matrix(train, feature_columns),
            train["target"].to_numpy(float),
            **fit_kwargs,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("GPU LightGBM unavailable; falling back to CPU:", repr(exc))
        params.pop("device_type", None)
        model = LGBMRegressor(**params)
        model.fit(
            _matrix(train, feature_columns),
            train["target"].to_numpy(float),
            **fit_kwargs,
        )
    joblib.dump(model, MODEL_DIR / f"{model_name}_fold{fold}.joblib")
    return model


# Freeze the feature list from all training sample schemas before CV.
schema_probe = _load_samples(meta["well"].astype(str).head(10), list(K3_TAGS))
feature_columns = _feature_columns(schema_probe)
del schema_probe
print("model features:", len(feature_columns))
print(feature_columns)


def _postprocess_sparse(frame, model_delta):
    """Final-chain algebra without Savitzky-Golay across sparse sampled rows."""
    last = frame["last_known_tvt"].to_numpy(float)
    md_since = frame["md_since"].to_numpy(float)
    warm = 1.0 - np.exp(-np.maximum(md_since, 0.0) / PP_TAU)
    pf_delta = frame["pf_ancc"].to_numpy(float) - last
    if PP_LIKPF_COL in frame:
        lik_values = frame[PP_LIKPF_COL].to_numpy(float)
        lik_values = np.where(np.isfinite(lik_values), lik_values, last)
        lik_delta = lik_values - last
    else:
        lik_delta = pf_delta
    learned = PP_ALPHA * warm * (
        model_delta * (1.0 - PP_W_PF_ANCC) + pf_delta * PP_W_PF_ANCC
    )
    return last + PP_W_MODEL * learned + (1.0 - PP_W_MODEL) * lik_delta


def _error_cosine(y_true, pred_a, pred_b):
    error_a = np.asarray(pred_a, float) - np.asarray(y_true, float)
    error_b = np.asarray(pred_b, float) - np.asarray(y_true, float)
    denominator = np.sqrt(np.dot(error_a, error_a) * np.dot(error_b, error_b))
    return float(np.dot(error_a, error_b) / denominator) if denominator > 0 else np.nan


# ---------------------------------------------------------------------------
# Five-fold K1/K3 gateway. Official full rows are primary; pseudo sampled rows
# are a secondary wide-PS robustness check.
# ---------------------------------------------------------------------------
oof_parts = []
fold_records = []
training_started = time.time()

for fold in range(1, N_FOLDS + 1):
    train_wells = meta.loc[meta["fold"] != fold, "well"].astype(str).tolist()
    valid_fold_wells = meta.loc[meta["fold"] == fold, "well"].astype(str).tolist()
    print(
        f"\nfold {fold}/{N_FOLDS}: train_wells={len(train_wells)} "
        f"valid_wells={len(valid_fold_wells)}",
        flush=True,
    )

    valid = _load_official_full(valid_fold_wells)
    single_columns = list(
        dict.fromkeys(feature_columns + ["target", "_sample_weight_single"])
    )
    train_single = _load_samples(train_wells, ["official"], single_columns)
    model_single = _fit_lgb(
        train_single, valid, feature_columns, "_sample_weight_single", "single", fold
    )
    del train_single
    gc.collect()

    k3_columns = list(dict.fromkeys(feature_columns + ["target", "_sample_weight_k3"]))
    train_k3 = _load_samples(train_wells, list(K3_TAGS), k3_columns)
    model_k3 = _fit_lgb(
        train_k3, valid, feature_columns, "_sample_weight_k3", "matched_k3", fold
    )
    del train_k3
    gc.collect()

    matrix_valid = _matrix(valid, feature_columns)
    delta_single = model_single.predict(
        matrix_valid, num_iteration=model_single.best_iteration_
    )
    delta_k3 = model_k3.predict(matrix_valid, num_iteration=model_k3.best_iteration_)
    pred_single = _postprocess(valid, delta_single)
    pred_k3 = _postprocess(valid, delta_k3)
    truth = valid["target"].to_numpy(float) + valid["last_known_tvt"].to_numpy(float)

    # Secondary validation on held-out wells at their two pseudo starts.
    wide_columns = list(
        dict.fromkeys(
            feature_columns
            + ["target", "last_known_tvt", "md_since", "pf_ancc", PP_LIKPF_COL]
        )
    )
    wide = _load_samples(valid_fold_wells, list(PSEUDO_TAGS), wide_columns)
    matrix_wide = _matrix(wide, feature_columns)
    wide_delta_single = model_single.predict(
        matrix_wide, num_iteration=model_single.best_iteration_
    )
    wide_delta_k3 = model_k3.predict(
        matrix_wide, num_iteration=model_k3.best_iteration_
    )
    wide_truth = (
        wide["target"].to_numpy(float) + wide["last_known_tvt"].to_numpy(float)
    )
    wide_pred_single = _postprocess_sparse(wide, wide_delta_single)
    wide_pred_k3 = _postprocess_sparse(wide, wide_delta_k3)

    single_rmse = _rmse(truth, pred_single)
    k3_rmse = _rmse(truth, pred_k3)
    wide_single_rmse = _rmse(wide_truth, wide_pred_single)
    wide_k3_rmse = _rmse(wide_truth, wide_pred_k3)
    fold_records.append(
        {
            "fold": fold,
            "rows": len(valid),
            "wells": len(valid_fold_wells),
            "single_rmse": single_rmse,
            "matched_k3_rmse": k3_rmse,
            "official_gain": single_rmse - k3_rmse,
            "wide_sample_rows": len(wide),
            "wide_single_rmse": wide_single_rmse,
            "wide_matched_k3_rmse": wide_k3_rmse,
            "wide_gain": wide_single_rmse - wide_k3_rmse,
            "single_best_iteration": int(model_single.best_iteration_),
            "matched_k3_best_iteration": int(model_k3.best_iteration_),
        }
    )
    print("fold result:", fold_records[-1], flush=True)

    oof_parts.append(
        pd.DataFrame(
            {
                "id": valid["id"].astype(str).to_numpy(),
                "well": valid["well"].astype(str).to_numpy(),
                "row_idx": valid["_row_idx"].to_numpy(int),
                "fold": fold,
                "tvt_true": truth,
                "pred_single": pred_single,
                "pred_matched_k3": pred_k3,
            }
        )
    )

    del valid, wide, model_single, model_k3, matrix_valid, matrix_wide
    gc.collect()

oof = pd.concat(oof_parts, ignore_index=True)
fold_summary = pd.DataFrame(fold_records)
single_rmse = _rmse(oof["tvt_true"], oof["pred_single"])
k3_rmse = _rmse(oof["tvt_true"], oof["pred_matched_k3"])


# ---------------------------------------------------------------------------
# CP12 gateway: four folds choose alpha, the fifth fold measures it. Gold is
# intentionally excluded because local Gold is not deployment-equivalent.
# ---------------------------------------------------------------------------
pool_path = OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv"
if not pool_path.exists():
    raise FileNotFoundError(pool_path)
cp12 = pd.read_csv(pool_path, usecols=["id", "cp12"])
cp12["id"] = cp12["id"].astype(str)
if cp12["id"].duplicated().any():
    raise RuntimeError("candidate pool has duplicate ids")
oof = oof.merge(cp12, on="id", how="left", validate="one_to_one")
if oof["cp12"].isna().any():
    raise RuntimeError(f"Missing CP12 for {int(oof['cp12'].isna().sum())} rows")

nested_records = []
nested_prediction = np.full(len(oof), np.nan, dtype=float)
for fold in range(1, N_FOLDS + 1):
    train_mask = oof["fold"].to_numpy(int) != fold
    valid_mask = ~train_mask
    train_y = oof.loc[train_mask, "tvt_true"].to_numpy(float)
    train_cp = oof.loc[train_mask, "cp12"].to_numpy(float)
    train_k3 = oof.loc[train_mask, "pred_matched_k3"].to_numpy(float)
    train_scores = [
        _rmse(train_y, (1.0 - alpha) * train_cp + alpha * train_k3)
        for alpha in ALPHA_GRID
    ]
    alpha = float(ALPHA_GRID[int(np.argmin(train_scores))])

    valid_y = oof.loc[valid_mask, "tvt_true"].to_numpy(float)
    valid_cp = oof.loc[valid_mask, "cp12"].to_numpy(float)
    valid_k3 = oof.loc[valid_mask, "pred_matched_k3"].to_numpy(float)
    valid_blend = (1.0 - alpha) * valid_cp + alpha * valid_k3
    nested_prediction[valid_mask] = valid_blend
    cp_score = _rmse(valid_y, valid_cp)
    blend_score = _rmse(valid_y, valid_blend)
    nested_records.append(
        {
            "fold": fold,
            "alpha": alpha,
            "cp12_rmse": cp_score,
            "blend_rmse": blend_score,
            "gain": cp_score - blend_score,
            "error_cosine_cp12_k3": _error_cosine(valid_y, valid_cp, valid_k3),
        }
    )

oof["pred_cp12_k3_nested"] = nested_prediction
nested_folds = pd.DataFrame(nested_records)
cp12_rmse = _rmse(oof["tvt_true"], oof["cp12"])
nested_rmse = _rmse(oof["tvt_true"], oof["pred_cp12_k3_nested"])
nested_gain = cp12_rmse - nested_rmse
overall_cosine = _error_cosine(
    oof["tvt_true"], oof["cp12"], oof["pred_matched_k3"]
)

bywell = []
for wid, group in oof.groupby("well", sort=False):
    truth = group["tvt_true"].to_numpy(float)
    single = group["pred_single"].to_numpy(float)
    k3 = group["pred_matched_k3"].to_numpy(float)
    cp = group["cp12"].to_numpy(float)
    nested = group["pred_cp12_k3_nested"].to_numpy(float)
    bywell.append(
        {
            "well": wid,
            "rows": len(group),
            "single_sse": float(np.sum((single - truth) ** 2)),
            "k3_sse": float(np.sum((k3 - truth) ** 2)),
            "cp12_sse": float(np.sum((cp - truth) ** 2)),
            "nested_sse": float(np.sum((nested - truth) ** 2)),
        }
    )
bywell = pd.DataFrame(bywell)
top40 = bywell.nlargest(40, "cp12_sse")
top40_cp12 = float(np.sqrt(top40["cp12_sse"].sum() / top40["rows"].sum()))
top40_nested = float(np.sqrt(top40["nested_sse"].sum() / top40["rows"].sum()))

nested_positive = int((nested_folds["gain"] > 0).sum())
alpha_positive = int((nested_folds["alpha"] > 0).sum())
official_gain = single_rmse - k3_rmse
official_positive = int((fold_summary["official_gain"] > 0).sum())
wide_mean_gain = float(fold_summary["wide_gain"].mean())
wide_positive = int((fold_summary["wide_gain"] > 0).sum())

# Architecture replacement and old-chain compatibility are independent paths.
# CP12 is diagnostic here; it cannot veto a base that improves honestly on its
# own and remains robust under held-out pseudo prediction starts.
architecture_pass = bool(
    (official_gain >= 0.10 and official_positive >= 4 and wide_mean_gain > 0 and wide_positive >= 3)
    or k3_rmse <= 8.0
)
incremental_pass = bool(
    nested_gain >= PASS_CP12_NESTED
    and nested_positive == N_FOLDS
    and alpha_positive == N_FOLDS
    and nested_folds["gain"].min() > 0
    and top40_nested <= top40_cp12
)
strict_pass = bool(architecture_pass or incremental_pass)

if architecture_pass and incremental_pass:
    next_step = "PASS BOTH -> extend K5/K8 base-first; retain CP12 as optional diagnostic"
elif architecture_pass:
    next_step = "PASS BASE -> extend K5/K8 base-first; CP12 does not gate the new architecture"
elif incremental_pass:
    next_step = "PASS INCREMENTAL ONLY -> useful for old chain, not yet a replacement base"
else:
    next_step = "FAIL BOTH -> stop pseudo-PS route; keep original 6.991 fallback"

result = {
    "rows": int(len(oof)),
    "wells": int(oof["well"].nunique()),
    "single_rmse": single_rmse,
    "matched_k3_rmse": k3_rmse,
    "matched_k3_gain_vs_single": official_gain,
    "official_mean_fold_gain": float(fold_summary["official_gain"].mean()),
    "official_min_fold_gain": float(fold_summary["official_gain"].min()),
    "official_positive_folds": official_positive,
    "wide_mean_fold_gain": wide_mean_gain,
    "wide_min_fold_gain": float(fold_summary["wide_gain"].min()),
    "wide_positive_folds": wide_positive,
    "cp12_rmse": cp12_rmse,
    "cp12_k3_nested_rmse": nested_rmse,
    "cp12_k3_nested_gain": nested_gain,
    "cp12_nested_min_fold": float(nested_folds["gain"].min()),
    "cp12_nested_positive_folds": nested_positive,
    "nested_alpha_positive_folds": alpha_positive,
    "mean_nested_alpha": float(nested_folds["alpha"].mean()),
    "error_cosine_cp12_k3": overall_cosine,
    "top40_cp12_rmse": top40_cp12,
    "top40_nested_rmse": top40_nested,
    "top40_nested_gain": top40_cp12 - top40_nested,
    "feature_hours": (training_started - feature_started) / 3600.0,
    "training_hours": (time.time() - training_started) / 3600.0,
    "architecture_pass": architecture_pass,
    "incremental_pass": incremental_pass,
    "strict_pass": strict_pass,
    "next": next_step,
}

print("\nK1 / MATCHED-K3 FOLDS")
display(fold_summary)
print("\nCP12 NESTED GATEWAY FOLDS")
display(nested_folds)
print("\nOVERALL")
print(result)
print("\nVERDICT:", result["next"])

oof.to_pickle(CACHE_DIR / "matched_k3_gateway_oof.pkl")
fold_summary.to_csv(CACHE_DIR / "matched_k3_gateway_folds.csv", index=False)
nested_folds.to_csv(CACHE_DIR / "matched_k3_cp12_nested_folds.csv", index=False)
bywell.to_csv(CACHE_DIR / "matched_k3_gateway_bywell.csv", index=False)
pd.DataFrame([result]).to_csv(CACHE_DIR / "matched_k3_gateway_summary.csv", index=False)
pd.Series(feature_columns, name="feature").to_csv(
    CACHE_DIR / "matched_k3_gateway_features.csv", index=False
)
print("saved:", CACHE_DIR)
