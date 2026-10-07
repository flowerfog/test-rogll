"""K5 pseudo-prediction-start extension for the existing multi-cut fleongg run.

Notebook order
--------------
1. Run the original fleongg import/function-definition cells through the cells
   that define ``build_well`` and ``lik_pf``. Do not run its training/submission
   cell.
2. Run this entire file as one notebook cell on AutoDL.

This experiment reuses the exact official/synth1/synth2 feature cache, folds,
and OOF predictions from ``multicut_fleongg_v0_k3_nospatial_r768_lgb1``. It
only builds two additional pseudo prediction starts at 50% and 75% of each
well, then trains one K5 LightGBM per fold. Validation is always the untouched
official prediction start of held-out wells.

The old K3 cache is intentionally required. If it was moved to Trash, this
cell tries to find it. It will not silently rebuild the old 7+ hour cache.
"""

from pathlib import Path
import gc
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
# Fixed experiment configuration. Do not tune these after seeing the result.
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
OLD_CACHE_NAME = "multicut_fleongg_v0_k3_nospatial_r768_lgb1"

# Set this only when automatic cache discovery prints that the old cache is
# missing. Point it at the directory containing well_metadata.csv, features/,
# and multicut_v0_oof.pkl.
OLD_CACHE_OVERRIDE = None

# The old features/ directory may have been moved to .Trash-0 independently
# of the small metadata/OOF files. Normally leave this as None; set it only if
# the automatic split-cache search still cannot locate the feature files.
OLD_FEATURE_DIR_OVERRIDE = None

RUN_DIR = OOF_DIR / "multicut_fleongg_k5_extension_v1"
NEW_FEATURE_DIR = RUN_DIR / "features_new"
TEMP_DIR = RUN_DIR / "masked_tmp"
MODEL_DIR = RUN_DIR / "models"

N_FOLDS = 5
OLD_TAGS = ("official", "synth1", "synth2")
NEW_CUTS = (("synth3", 0.50), ("synth4", 0.75))
K5_TAGS = OLD_TAGS + tuple(tag for tag, _ in NEW_CUTS)
TRAIN_ROWS_PER_CUT = 768
N_JOBS_FEATURES = 4
RANDOM_STATE = 20260714

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

# Exact post-processing from the old K3 experiment.
PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_MODEL = 0.60
PP_W_PF_ANCC = 0.0
PP_LIKPF_COL = "likpf_scale_5"
PP_SG_WIN = 61
PP_SG_POLY = 3

# Pre-registered gates. K5 must improve both the input experiment and the
# strong CP12 chain; a better standalone weak base is not enough.
PASS_K5_VS_K3 = 0.10
PASS_K5_VS_SINGLE = 0.25
PASS_CP12_NESTED = 0.05
ALPHA_GRID = np.round(np.arange(0.0, 0.5001, 0.025), 3)


def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _old_cache_valid(path):
    path = Path(path)
    required = (
        path / "well_metadata.csv",
        path / "cut_manifest.csv",
        path / "multicut_v0_oof.pkl",
        path / "multicut_v0_features.csv",
    )
    return all(item.exists() for item in required)


def _find_old_cache():
    candidates = []
    if OLD_CACHE_OVERRIDE:
        candidates.append(Path(OLD_CACHE_OVERRIDE))
    candidates.append(OOF_DIR / OLD_CACHE_NAME)
    candidates.append(Path("/root/autodl-tmp/.Trash-0/files") / OLD_CACHE_NAME)

    for candidate in candidates:
        if _old_cache_valid(candidate):
            return candidate

    # Search by the small summary file, not by all feature files.
    for root in (
        Path("/root/autodl-tmp/.Trash-0/files"),
        Path("/root/autodl-tmp/.Trash-0"),
    ):
        if not root.exists():
            continue
        try:
            for marker in root.rglob("multicut_v0_oof.pkl"):
                candidate = marker.parent
                if _old_cache_valid(candidate):
                    return candidate
        except Exception:
            pass

    raise FileNotFoundError(
        "Cannot find the old K3 metadata/OOF cache. Expected a directory containing "
        "well_metadata.csv, cut_manifest.csv, multicut_v0_oof.pkl, "
        "and multicut_v0_features.csv. Restore those files from Trash "
        "or set OLD_CACHE_OVERRIDE at the top of this cell. The cell deliberately "
        "does not rebuild the old 7+ hour cache."
    )


def _feature_dir_score(path):
    """Number of wells for which all four required old artifacts exist."""
    path = Path(path)
    if not path.is_dir():
        return 0
    tag_sets = []
    patterns = (
        "*__official__sample.pkl",
        "*__synth1__sample.pkl",
        "*__synth2__sample.pkl",
        "*__official_full.pkl",
    )
    suffixes = (
        "__official__sample.pkl",
        "__synth1__sample.pkl",
        "__synth2__sample.pkl",
        "__official_full.pkl",
    )
    for pattern, suffix in zip(patterns, suffixes):
        wells = {item.name[: -len(suffix)] for item in path.glob(pattern)}
        tag_sets.append(wells)
    return len(set.intersection(*tag_sets)) if tag_sets else 0


def _find_old_feature_dir(old_cache):
    candidates = []
    if OLD_FEATURE_DIR_OVERRIDE:
        candidates.append(Path(OLD_FEATURE_DIR_OVERRIDE))
    candidates.extend(
        [
            Path(old_cache) / "features",
            Path("/root/autodl-tmp/.Trash-0/files") / OLD_CACHE_NAME / "features",
            Path("/root/autodl-tmp/.Trash-0/files/features"),
        ]
    )

    checked = set()
    scored = []

    def add_candidate(path):
        try:
            key = str(Path(path).resolve())
        except Exception:
            key = str(path)
        if key in checked:
            return
        checked.add(key)
        score = _feature_dir_score(path)
        if score:
            scored.append((score, Path(path)))

    for candidate in candidates:
        add_candidate(candidate)

    if not scored or max(score for score, _ in scored) < 700:
        # Locate directories by a single distinctive old-K3 filename, then
        # score each parent once. This also handles Trash flattening/renaming.
        search_roots = (
            OOF_DIR,
            Path("/root/autodl-tmp/.Trash-0/files"),
            Path("/root/autodl-tmp/.Trash-0"),
        )
        for root in search_roots:
            if not root.exists():
                continue
            try:
                for marker in root.rglob("*__synth2__sample.pkl"):
                    add_candidate(marker.parent)
            except Exception:
                pass

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        print(
            "old feature-directory candidates:",
            [(score, str(path)) for score, path in scored[:5]],
        )
        if scored[0][0] >= 700:
            return scored[0][1]

    raise FileNotFoundError(
        "Found the K3 metadata/OOF files but not a feature directory with at "
        "least 700 complete wells (official, synth1, synth2, official_full). "
        "The features may have been permanently deleted or split across Trash "
        "directories. Set OLD_FEATURE_DIR_OVERRIDE if you know their location."
    )


required_globals = ["build_well", "lik_pf"]
missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Run the original fleongg function-definition cells first; missing: "
        + repr(missing_globals)
    )

OLD_CACHE = _find_old_cache()
OLD_FEATURE_DIR = _find_old_feature_dir(OLD_CACHE)
for directory in (OOF_DIR, RUN_DIR, NEW_FEATURE_DIR, TEMP_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print(
    {
        "experiment": "multicut_fleongg_k5_extension_v1",
        "old_cache": str(OLD_CACHE),
        "run_dir": str(RUN_DIR),
        "old_tags": OLD_TAGS,
        "new_cuts": NEW_CUTS,
        "validation": "untouched official PS only",
        "old_models": "reused through exact old OOF predictions",
    }
)


# ---------------------------------------------------------------------------
# Reuse old fold assignment, feature schema, cuts, and exact OOF predictions.
# ---------------------------------------------------------------------------
meta = pd.read_csv(OLD_CACHE / "well_metadata.csv", dtype={"well": str})
old_manifest = pd.read_csv(OLD_CACHE / "cut_manifest.csv", dtype={"well": str})
old_oof = pd.read_pickle(OLD_CACHE / "multicut_v0_oof.pkl")
old_oof["id"] = old_oof["id"].astype(str)
old_oof["well"] = old_oof["well"].astype(str)

feature_columns = (
    pd.read_csv(OLD_CACHE / "multicut_v0_features.csv")["feature"]
    .astype(str)
    .tolist()
)

required_oof = {
    "id",
    "well",
    "fold",
    "tvt_true",
    "pred_single_cut",
    "pred_multi_cut",
}
missing_oof = required_oof - set(old_oof.columns)
if missing_oof:
    raise RuntimeError(f"Old OOF is missing columns: {sorted(missing_oof)}")
if len(meta) < 700 or old_oof["well"].nunique() < 700:
    raise RuntimeError("Old K3 cache is incomplete (<700 wells)")

print(
    "old lineage:",
    {
        "metadata_wells": int(meta["well"].nunique()),
        "oof_rows": int(len(old_oof)),
        "oof_wells": int(old_oof["well"].nunique()),
        "features": len(feature_columns),
    },
)

# Fail before the expensive new feature build when the old cache is only a
# partial directory or its files were moved independently to Trash.
old_complete_wells = []
for wid in meta["well"].astype(str):
    old_paths = [OLD_FEATURE_DIR / f"{wid}__{tag}__sample.pkl" for tag in OLD_TAGS]
    old_paths.append(OLD_FEATURE_DIR / f"{wid}__official_full.pkl")
    if all(path.exists() for path in old_paths):
        old_complete_wells.append(wid)
print("old feature-cache complete wells:", len(old_complete_wells))
if len(old_complete_wells) < 700:
    raise FileNotFoundError(
        f"Old K3 feature cache has only {len(old_complete_wells)} complete wells. "
        "Restore its features/ directory or set OLD_CACHE_OVERRIDE to the complete "
        "copy before running K5; no new features were built."
    )
meta = meta[meta["well"].astype(str).isin(old_complete_wells)].copy().reset_index(drop=True)
old_oof = old_oof[old_oof["well"].isin(old_complete_wells)].copy().reset_index(drop=True)


class _NeutralFormationImputer:
    """Finite placeholders for build_well; all spatial columns stay excluded."""

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


def _evenly_spaced_positions(n_rows, n_take):
    if n_rows <= n_take:
        return np.arange(n_rows, dtype=int)
    return np.unique(np.round(np.linspace(0, n_rows - 1, n_take)).astype(int))


def _add_likpf_to_frame(frame, masked_hw, typewell, pseudo_wid):
    defaults = (
        "likpf_scale_3",
        "likpf_scale_5",
        "likpf_scale_8",
        "likpf_mean",
    )
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
    except Exception as exc:
        warnings.warn(f"lik_pf fallback for {pseudo_wid}: {exc!r}")
    return frame


def _new_sample_path(wid, tag):
    return NEW_FEATURE_DIR / f"{wid}__{tag}__sample.pkl"


def _old_sample_path(wid, tag):
    return OLD_FEATURE_DIR / f"{wid}__{tag}__sample.pkl"


def _old_official_full_path(wid):
    return OLD_FEATURE_DIR / f"{wid}__official_full.pkl"


def _trim_sample(frame):
    required = feature_columns + [
        "target",
        "well",
        "id",
        "_row_idx",
        "_cut_tag",
        "_official_n_eval",
    ]
    required = list(dict.fromkeys(required))
    for column in required:
        if column not in frame.columns:
            if column in {"well", "id", "_cut_tag"}:
                frame[column] = ""
            else:
                frame[column] = np.nan
    return frame[required].copy()


def _build_new_task(task):
    wid = str(task["well"])
    tag = str(task["tag"])
    cut_position = int(task["cut_position"])
    official_n_eval = int(task["official_n_eval"])
    output_path = _new_sample_path(wid, tag)
    if output_path.exists():
        return {"well": wid, "tag": tag, "status": "cached"}

    horizontal_path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{wid}__typewell.csv"
    pseudo_wid = f"{wid}_k5_{tag}"
    temp_path = TEMP_DIR / f"{pseudo_wid}__horizontal_well.csv"

    try:
        source = pd.read_csv(horizontal_path)
        keep = ["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"]
        masked = source[keep].copy().reset_index(drop=True)
        truth = masked["TVT"].to_numpy(float)
        pseudo_input = np.full(len(masked), np.nan, dtype=float)
        pseudo_input[:cut_position] = truth[:cut_position]
        masked["TVT_input"] = pseudo_input
        masked.to_csv(temp_path, index=False)

        typewell = pd.read_csv(typewell_path).sort_values("TVT")
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

        positions = _evenly_spaced_positions(len(frame), TRAIN_ROWS_PER_CUT)
        sample = _trim_sample(frame.iloc[positions].copy())
        sample.to_pickle(output_path)
        return {
            "well": wid,
            "tag": tag,
            "status": "built",
            "rows": len(frame),
            "sample_rows": len(sample),
        }
    except Exception as exc:
        return {"well": wid, "tag": tag, "status": "failed", "reason": repr(exc)}
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Add fixed 50% and 75% pseudo prediction starts without touching old cuts.
# ---------------------------------------------------------------------------
old_used = (
    old_manifest.groupby("well")["cut_position"]
    .apply(lambda values: set(map(int, values)))
    .to_dict()
)
new_tasks = []
for row in meta.itertuples(index=False):
    wid = str(row.well)
    used = set(old_used.get(wid, {int(row.official_cut)}))
    for tag, fraction in NEW_CUTS:
        cut = int(np.clip(round(fraction * int(row.n_rows)), 64, int(row.n_rows) - 128))
        minimum_gap = max(32, int(row.n_rows) // 100)
        while any(abs(cut - previous) < minimum_gap for previous in used):
            direction = -1 if fraction <= 0.5 else 1
            candidate = int(
                np.clip(cut + direction * minimum_gap, 64, int(row.n_rows) - 128)
            )
            if candidate == cut:
                break
            cut = candidate
        used.add(cut)
        new_tasks.append(
            {
                "well": wid,
                "tag": tag,
                "fraction_requested": fraction,
                "cut_position": cut,
                "actual_fraction": cut / int(row.n_rows),
                "official_n_eval": int(row.n_eval),
            }
        )

new_manifest = pd.DataFrame(new_tasks)
new_manifest.to_csv(RUN_DIR / "k5_new_cut_manifest.csv", index=False)
print(
    "new cut fractions:",
    new_manifest.groupby("tag")["actual_fraction"].describe()[
        ["count", "mean", "min", "max"]
    ],
)

original_fi = globals().get("_FI", None)
original_di = globals().get("_DI", None)
globals()["_FI"] = _NeutralFormationImputer()
globals()["_DI"] = _NeutralDenseImputer()

feature_started = time.time()
try:
    from joblib import Parallel, delayed

    smoke_results = [_build_new_task(task) for task in new_tasks[:2]]
    print("feature smoke test:", smoke_results, flush=True)
    if any(row["status"] == "failed" for row in smoke_results):
        raise RuntimeError("New-cut feature smoke test failed: " + repr(smoke_results))

    feature_results = Parallel(n_jobs=N_JOBS_FEATURES, prefer="threads")(
        delayed(_build_new_task)(task)
        for task in tqdm(new_tasks, desc="K5 new pseudo-PS features")
    )
finally:
    globals()["_FI"] = original_fi
    globals()["_DI"] = original_di

feature_status = pd.DataFrame(feature_results)
feature_status.to_csv(RUN_DIR / "feature_status.csv", index=False)
print("feature status:", feature_status["status"].value_counts().to_dict())
print("feature elapsed hours:", (time.time() - feature_started) / 3600.0)
failed = feature_status[feature_status["status"] == "failed"]
if len(failed):
    display(failed.head(30))
if len(failed) > max(10, 0.02 * len(new_tasks)):
    raise RuntimeError(f"Too many new feature failures: {len(failed)} / {len(new_tasks)}")


def _all_paths_exist(wid):
    return (
        all(_old_sample_path(wid, tag).exists() for tag in OLD_TAGS)
        and _old_official_full_path(wid).exists()
        and all(_new_sample_path(wid, tag).exists() for tag, _ in NEW_CUTS)
    )


valid_wells = [wid for wid in meta["well"].astype(str) if _all_paths_exist(wid)]
meta = meta[meta["well"].astype(str).isin(valid_wells)].copy().reset_index(drop=True)
old_oof = old_oof[old_oof["well"].isin(valid_wells)].copy().reset_index(drop=True)
print("complete K5 wells:", len(meta))
if len(meta) < 700:
    raise RuntimeError("Too few complete wells after adding the K5 cuts")


def _sample_path(wid, tag):
    return _old_sample_path(wid, tag) if tag in OLD_TAGS else _new_sample_path(wid, tag)


def _load_k5_samples(wells):
    frames = []
    n_cuts = len(K5_TAGS)
    eval_by_well = meta.set_index("well")["n_eval"].to_dict()
    keep = list(dict.fromkeys(feature_columns + ["target"]))
    for wid in wells:
        for tag in K5_TAGS:
            frame = pd.read_pickle(_sample_path(str(wid), tag))
            for column in keep:
                if column not in frame:
                    frame[column] = np.nan
            frame = frame[keep].copy()
            frame["_sample_weight_k5"] = float(eval_by_well[str(wid)]) / (
                n_cuts * max(1, len(frame))
            )
            frames.append(frame)
    if not frames:
        raise RuntimeError("No K5 training samples loaded")
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_official_full(wells):
    frames = []
    keep = list(
        dict.fromkeys(
            feature_columns
            + [
                "target",
                "well",
                "id",
                "_row_idx",
                "last_known_tvt",
                "md_since",
                "pf_ancc",
                PP_LIKPF_COL,
            ]
        )
    )
    for wid in wells:
        frame = pd.read_pickle(_old_official_full_path(str(wid)))
        for column in keep:
            if column not in frame:
                if column in {"well", "id"}:
                    frame[column] = ""
                else:
                    frame[column] = np.nan
        frames.append(frame[keep].copy())
    return pd.concat(frames, ignore_index=True, sort=False)


def _matrix(frame):
    matrix = frame.reindex(columns=feature_columns).to_numpy(np.float32)
    matrix[np.isinf(matrix)] = np.nan
    return matrix


def _postprocess(frame, model_delta):
    last = frame["last_known_tvt"].to_numpy(float)
    md_since = frame["md_since"].to_numpy(float)
    warm = 1.0 - np.exp(-np.maximum(md_since, 0.0) / PP_TAU)
    pf_delta = frame["pf_ancc"].to_numpy(float) - last
    lik_values = frame[PP_LIKPF_COL].to_numpy(float)
    lik_values = np.where(np.isfinite(lik_values), lik_values, last)
    lik_delta = lik_values - last
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


def _fit_k5(train, valid, fold):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    params = dict(LGB_PARAMS)
    params["device_type"] = "gpu"
    model = LGBMRegressor(**params)
    fit_kwargs = dict(
        sample_weight=train["_sample_weight_k5"].to_numpy(float),
        eval_set=[(_matrix(valid), valid["target"].to_numpy(float))],
        eval_metric="rmse",
        callbacks=[early_stopping(200, verbose=False), log_evaluation(100)],
    )
    try:
        model.fit(_matrix(train), train["target"].to_numpy(float), **fit_kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("GPU LightGBM unavailable; falling back to CPU:", repr(exc))
        params.pop("device_type", None)
        model = LGBMRegressor(**params)
        model.fit(_matrix(train), train["target"].to_numpy(float), **fit_kwargs)
    joblib.dump(model, MODEL_DIR / f"k5_fold{fold}.joblib")
    return model


# ---------------------------------------------------------------------------
# Train only five new K5 models. Single/K3 predictions are exact old OOF data.
# ---------------------------------------------------------------------------
fold_parts = []
fold_records = []
training_started = time.time()

for fold in range(1, N_FOLDS + 1):
    train_wells = meta.loc[meta["fold"] != fold, "well"].astype(str).tolist()
    valid_wells_fold = meta.loc[meta["fold"] == fold, "well"].astype(str).tolist()
    print(
        f"\nK5 fold {fold}/{N_FOLDS}: train={len(train_wells)} "
        f"valid={len(valid_wells_fold)}",
        flush=True,
    )

    train = _load_k5_samples(train_wells)
    valid = _load_official_full(valid_wells_fold)
    model = _fit_k5(train, valid, fold)
    delta = model.predict(_matrix(valid), num_iteration=model.best_iteration_)
    pred_k5 = _postprocess(valid, delta)
    truth = valid["target"].to_numpy(float) + valid["last_known_tvt"].to_numpy(float)

    current = pd.DataFrame(
        {
            "id": valid["id"].astype(str).to_numpy(),
            "well": valid["well"].astype(str).to_numpy(),
            "fold": fold,
            "tvt_true_k5": truth,
            "pred_k5": pred_k5,
        }
    )
    old_fold = old_oof.loc[old_oof["fold"] == fold, list(required_oof)].copy()
    merged = old_fold.merge(current, on=["id", "well", "fold"], validate="one_to_one")
    if len(merged) != len(old_fold) or len(merged) != len(current):
        raise RuntimeError(f"Fold {fold}: old/new validation row mismatch")
    truth_diff = np.max(np.abs(merged["tvt_true"] - merged["tvt_true_k5"]))
    if truth_diff > 1e-3:
        raise RuntimeError(f"Fold {fold}: truth lineage mismatch max={truth_diff}")

    y = merged["tvt_true"].to_numpy(float)
    p_single = merged["pred_single_cut"].to_numpy(float)
    p_k3 = merged["pred_multi_cut"].to_numpy(float)
    p_k5 = merged["pred_k5"].to_numpy(float)
    record = {
        "fold": fold,
        "rows": len(merged),
        "wells": len(valid_wells_fold),
        "single_rmse": _rmse(y, p_single),
        "k3_rmse": _rmse(y, p_k3),
        "k5_rmse": _rmse(y, p_k5),
        "k3_gain_vs_single": _rmse(y, p_single) - _rmse(y, p_k3),
        "k5_gain_vs_single": _rmse(y, p_single) - _rmse(y, p_k5),
        "k5_gain_vs_k3": _rmse(y, p_k3) - _rmse(y, p_k5),
        "k5_best_iteration": int(model.best_iteration_),
    }
    fold_records.append(record)
    print("fold result:", record, flush=True)
    fold_parts.append(merged.drop(columns=["tvt_true_k5"]))

    del train, valid, model, current, old_fold, merged
    gc.collect()

oof = pd.concat(fold_parts, ignore_index=True)
fold_summary = pd.DataFrame(fold_records)
y = oof["tvt_true"].to_numpy(float)
p_single = oof["pred_single_cut"].to_numpy(float)
p_k3 = oof["pred_multi_cut"].to_numpy(float)
p_k5 = oof["pred_k5"].to_numpy(float)

single_rmse = _rmse(y, p_single)
k3_rmse = _rmse(y, p_k3)
k5_rmse = _rmse(y, p_k5)

bywell = []
for wid, group in oof.groupby("well", sort=False):
    wy = group["tvt_true"].to_numpy(float)
    ws = group["pred_single_cut"].to_numpy(float)
    w3 = group["pred_multi_cut"].to_numpy(float)
    w5 = group["pred_k5"].to_numpy(float)
    bywell.append(
        {
            "well": wid,
            "rows": len(group),
            "single_sse": float(np.sum((ws - wy) ** 2)),
            "k3_sse": float(np.sum((w3 - wy) ** 2)),
            "k5_sse": float(np.sum((w5 - wy) ** 2)),
        }
    )
bywell = pd.DataFrame(bywell)
top40 = bywell.nlargest(40, "single_sse")
top40_single = float(np.sqrt(top40["single_sse"].sum() / top40["rows"].sum()))
top40_k3 = float(np.sqrt(top40["k3_sse"].sum() / top40["rows"].sum()))
top40_k5 = float(np.sqrt(top40["k5_sse"].sum() / top40["rows"].sum()))


# ---------------------------------------------------------------------------
# Nested CP12 blend: select alpha on four OOF folds and evaluate the fifth.
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
    raise RuntimeError(f"Missing CP12 predictions for {int(oof['cp12'].isna().sum())} rows")

nested_records = []
nested_predictions = np.full(len(oof), np.nan, dtype=float)
for fold in range(1, N_FOLDS + 1):
    train_mask = oof["fold"].to_numpy(int) != fold
    valid_mask = ~train_mask
    train_y = oof.loc[train_mask, "tvt_true"].to_numpy(float)
    train_cp = oof.loc[train_mask, "cp12"].to_numpy(float)
    train_k5 = oof.loc[train_mask, "pred_k5"].to_numpy(float)

    train_scores = []
    for alpha in ALPHA_GRID:
        prediction = (1.0 - alpha) * train_cp + alpha * train_k5
        train_scores.append(_rmse(train_y, prediction))
    best_index = int(np.argmin(train_scores))
    alpha = float(ALPHA_GRID[best_index])

    valid_y = oof.loc[valid_mask, "tvt_true"].to_numpy(float)
    valid_cp = oof.loc[valid_mask, "cp12"].to_numpy(float)
    valid_k5 = oof.loc[valid_mask, "pred_k5"].to_numpy(float)
    valid_blend = (1.0 - alpha) * valid_cp + alpha * valid_k5
    nested_predictions[valid_mask] = valid_blend
    base_score = _rmse(valid_y, valid_cp)
    blend_score = _rmse(valid_y, valid_blend)
    nested_records.append(
        {
            "fold": fold,
            "alpha": alpha,
            "cp12_rmse": base_score,
            "blend_rmse": blend_score,
            "gain": base_score - blend_score,
        }
    )

oof["pred_cp12_k5_nested"] = nested_predictions
nested_folds = pd.DataFrame(nested_records)
cp12_rmse = _rmse(oof["tvt_true"], oof["cp12"])
nested_rmse = _rmse(oof["tvt_true"], oof["pred_cp12_k5_nested"])
nested_gain = cp12_rmse - nested_rmse

k5_vs_k3 = k3_rmse - k5_rmse
k5_vs_single = single_rmse - k5_rmse
k5_k3_positive = int((fold_summary["k5_gain_vs_k3"] > 0).sum())
k5_single_positive = int((fold_summary["k5_gain_vs_single"] > 0).sum())
nested_positive = int((nested_folds["gain"] > 0).sum())

strict_pass = bool(
    k5_vs_k3 >= PASS_K5_VS_K3
    and k5_vs_single >= PASS_K5_VS_SINGLE
    and k5_k3_positive == N_FOLDS
    and k5_single_positive == N_FOLDS
    and top40_k5 < top40_k3
    and nested_gain >= PASS_CP12_NESTED
    and nested_positive == N_FOLDS
    and nested_folds["gain"].min() > 0
)

result = {
    "wells": int(oof["well"].nunique()),
    "rows": int(len(oof)),
    "single_rmse": single_rmse,
    "k3_rmse": k3_rmse,
    "k5_rmse": k5_rmse,
    "k3_gain_vs_single": single_rmse - k3_rmse,
    "k5_gain_vs_single": k5_vs_single,
    "k5_gain_vs_k3": k5_vs_k3,
    "k5_vs_k3_min_fold": float(fold_summary["k5_gain_vs_k3"].min()),
    "k5_vs_k3_positive_folds": k5_k3_positive,
    "top40_single_rmse": top40_single,
    "top40_k3_rmse": top40_k3,
    "top40_k5_rmse": top40_k5,
    "top40_k5_gain_vs_k3": top40_k3 - top40_k5,
    "cp12_rmse": cp12_rmse,
    "cp12_k5_nested_rmse": nested_rmse,
    "cp12_k5_nested_gain": nested_gain,
    "cp12_nested_min_fold": float(nested_folds["gain"].min()),
    "cp12_nested_positive_folds": nested_positive,
    "mean_nested_alpha": float(nested_folds["alpha"].mean()),
    "feature_hours": (training_started - feature_started) / 3600.0,
    "training_hours": (time.time() - training_started) / 3600.0,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> train CatBoost/XGBoost K5 ensemble and deploy through CP12+Gold"
        if strict_pass
        else "FAIL -> stop pseudo-PS expansion; keep original 6.991 chain"
    ),
}

print("\nK5 FOLDS")
display(fold_summary)
print("\nCP12 NESTED BLEND FOLDS")
display(nested_folds)
print("\nOVERALL")
print(result)
print("\nVERDICT:", result["next"])

oof.to_pickle(RUN_DIR / "k5_oof.pkl")
fold_summary.to_csv(RUN_DIR / "k5_folds.csv", index=False)
nested_folds.to_csv(RUN_DIR / "cp12_k5_nested_folds.csv", index=False)
bywell.to_csv(RUN_DIR / "k5_bywell.csv", index=False)
pd.DataFrame([result]).to_csv(RUN_DIR / "k5_summary.csv", index=False)
print("saved:", RUN_DIR)
