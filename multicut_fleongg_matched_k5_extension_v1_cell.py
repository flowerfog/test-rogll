"""Extend the matched K3 cut bank to K5 and measure saturation.

Run after the original fleongg function-definition cells have defined
``build_well`` and ``lik_pf``. The completed matched-K3 cache must exist.
Only pseudo3/pseudo4 are generated; official/pseudo1/pseudo2 are reused.
"""

from pathlib import Path
import gc
import hashlib
import os
import shutil
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
BANK_DIR = OOF_DIR / "multicut_fleongg_matched_k3_gateway_v1"
FEATURE_DIR = BANK_DIR / "features"
DONOR_FEATURE_DIR = (
    OOF_DIR / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75" / "features"
)
RUN_DIR = OOF_DIR / "multicut_fleongg_matched_k5_extension_v1"
TEMP_DIR = RUN_DIR / "masked_tmp"
MODEL_DIR = RUN_DIR / "models"

OLD_TAGS = ("official", "pseudo1", "pseudo2")
NEW_TAGS = ("pseudo3", "pseudo4")
K3_TAGS = OLD_TAGS
K5_TAGS = OLD_TAGS + NEW_TAGS
TRAIN_ROWS_PER_CUT = 768
N_FOLDS = 5
N_JOBS_FEATURES = 4
RANDOM_STATE = 20260715

LGB_PARAMS = dict(
    objective="regression",
    boosting_type="gbdt",
    n_estimators=4000,
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

PP_ALPHA = 1.0
PP_TAU = 85.0
PP_W_MODEL = 0.60
PP_W_PF_ANCC = 0.0
PP_LIKPF_COL = "likpf_scale_5"
PP_SG_WIN = 61
PP_SG_POLY = 3
ALPHA_GRID = np.round(np.arange(0.0, 0.5001, 0.025), 3)

required_globals = ["build_well", "lik_pf"]
missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Run the original fleongg function-definition cells first; missing: "
        + repr(missing_globals)
    )

required_cache = (
    BANK_DIR / "well_metadata.csv",
    BANK_DIR / "cut_manifest.csv",
    BANK_DIR / "matched_k3_gateway_oof.pkl",
    BANK_DIR / "matched_k3_gateway_features.csv",
    FEATURE_DIR,
)
missing_cache = [str(path) for path in required_cache if not path.exists()]
if missing_cache:
    raise FileNotFoundError("Matched-K3 cache is incomplete: " + repr(missing_cache))

for directory in (RUN_DIR, TEMP_DIR, MODEL_DIR, FEATURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(BANK_DIR / "well_metadata.csv", dtype={"well": str})
old_manifest = pd.read_csv(BANK_DIR / "cut_manifest.csv", dtype={"well": str})
old_oof = pd.read_pickle(BANK_DIR / "matched_k3_gateway_oof.pkl")
old_oof["id"] = old_oof["id"].astype(str)
old_oof["well"] = old_oof["well"].astype(str)
feature_columns = (
    pd.read_csv(BANK_DIR / "matched_k3_gateway_features.csv")["feature"]
    .astype(str)
    .tolist()
)

print(
    {
        "experiment": "multicut_fleongg_matched_k5_extension_v1",
        "bank": str(BANK_DIR),
        "new_tags": NEW_TAGS,
        "models": "matched K3 vs matched K5, both retrained with 4000 rounds",
        "features": len(feature_columns),
        "wells": int(meta["well"].nunique()),
    }
)


def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _error_cosine(y_true, pred_a, pred_b):
    error_a = np.asarray(pred_a, float) - np.asarray(y_true, float)
    error_b = np.asarray(pred_b, float) - np.asarray(y_true, float)
    denominator = np.sqrt(np.dot(error_a, error_a) * np.dot(error_b, error_b))
    return float(np.dot(error_a, error_b) / denominator) if denominator > 0 else np.nan


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
    grid = np.arange(64, max(65, n_rows - 127), max(1, min_gap // 2), dtype=int)
    distances = np.asarray(
        [min(abs(point - previous) for previous in used) for point in grid]
    )
    return int(grid[int(np.argmax(distances))])


class _NeutralFormationImputer:
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


def _add_likpf(frame, masked_hw, typewell, pseudo_wid):
    defaults = ("likpf_scale_3", "likpf_scale_5", "likpf_scale_8", "likpf_mean")
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


# Consolidate donor official files as hard links so the partial fixed-K5 cache
# can be deleted after this point without breaking the matched cut bank.
link_counts = {"linked": 0, "copied": 0, "already_local": 0}
for wid in meta["well"].astype(str):
    pairs = (
        (
            DONOR_FEATURE_DIR / f"{wid}__official__sample.pkl",
            FEATURE_DIR / f"{wid}__official__sample.pkl",
        ),
        (
            DONOR_FEATURE_DIR / f"{wid}__official_full.pkl",
            FEATURE_DIR / f"{wid}__official_full.pkl",
        ),
    )
    for source, destination in pairs:
        if destination.exists():
            link_counts["already_local"] += 1
        elif source.exists():
            try:
                os.link(source, destination)
                link_counts["linked"] += 1
            except OSError:
                shutil.copy2(source, destination)
                link_counts["copied"] += 1
print("official cache consolidation:", link_counts)


def _sample_path(wid, tag):
    return FEATURE_DIR / f"{wid}__{tag}__sample.pkl"


def _official_full_path(wid):
    return FEATURE_DIR / f"{wid}__official_full.pkl"


reference_fracs = meta["visible_frac"].to_numpy(float)
used_by_well = (
    old_manifest.groupby("well")["cut_position"]
    .apply(lambda values: set(map(int, values)))
    .to_dict()
)
new_tasks = []
for row in meta.itertuples(index=False):
    wid = str(row.well)
    used = set(used_by_well.get(wid, {int(row.official_cut)}))
    for offset, tag in enumerate(NEW_TAGS, start=2):
        unit = _stable_unit_interval(wid, offset)
        # A second independent lower/upper-half draw preserves the empirical
        # distribution while making K3 a strict subset of K5.
        half = 0 if tag == "pseudo3" else 1
        quantile = (half + unit) / 2.0
        requested_frac = float(np.quantile(reference_fracs, quantile))
        base_cut = int(
            np.clip(round(requested_frac * row.n_rows), 64, row.n_rows - 128)
        )
        cut = _distinct_cut(base_cut, used, int(row.n_rows), unit)
        used.add(cut)
        new_tasks.append(
            {
                "well": wid,
                "tag": tag,
                "cut_position": cut,
                "official_n_eval": int(row.n_eval),
                "sample_quantile": quantile,
                "requested_frac": requested_frac,
                "actual_frac": cut / int(row.n_rows),
            }
        )

new_manifest = pd.DataFrame(new_tasks)
new_manifest.to_csv(RUN_DIR / "new_cut_manifest.csv", index=False)
print(
    "new cut fractions:",
    new_manifest.groupby("tag")["actual_frac"].describe()[
        ["count", "mean", "min", "25%", "50%", "75%", "max"]
    ],
)


def _build_task(task):
    wid = str(task["well"])
    tag = str(task["tag"])
    cut = int(task["cut_position"])
    output_path = _sample_path(wid, tag)
    if output_path.exists():
        return {"well": wid, "tag": tag, "status": "cached"}

    horizontal_path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{wid}__typewell.csv"
    pseudo_wid = f"{wid}_matched_{tag}"
    temp_path = TEMP_DIR / f"{pseudo_wid}__horizontal_well.csv"
    try:
        source = pd.read_csv(horizontal_path)
        keep = ["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"]
        masked = source[keep].copy().reset_index(drop=True)
        truth = masked["TVT"].to_numpy(float)
        pseudo_input = np.full(len(masked), np.nan, dtype=float)
        pseudo_input[:cut] = truth[:cut]
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
        frame = _add_likpf(frame, masked, typewell, pseudo_wid)
        frame["well"] = wid
        frame["id"] = wid + "_" + frame["_row_idx"].astype(str)
        frame["_cut_tag"] = tag
        frame["_official_n_eval"] = int(task["official_n_eval"])
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        positions = _evenly_spaced_positions(len(frame), TRAIN_ROWS_PER_CUT)
        required = list(
            dict.fromkeys(
                feature_columns
                + ["target", "well", "id", "_row_idx", "_cut_tag", "_official_n_eval"]
            )
        )
        for column in required:
            if column not in frame:
                frame[column] = np.nan
        sample = frame.iloc[positions][required].copy()
        sample.to_pickle(output_path)
        return {"well": wid, "tag": tag, "status": "built", "rows": len(frame)}
    except Exception as exc:
        return {"well": wid, "tag": tag, "status": "failed", "reason": repr(exc)}
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


original_fi = globals().get("_FI", None)
original_di = globals().get("_DI", None)
globals()["_FI"] = _NeutralFormationImputer()
globals()["_DI"] = _NeutralDenseImputer()
feature_started = time.time()
try:
    from joblib import Parallel, delayed

    smoke = [_build_task(task) for task in new_tasks[:2]]
    print("feature smoke:", smoke)
    if any(row["status"] == "failed" for row in smoke):
        raise RuntimeError("Feature smoke failed: " + repr(smoke))
    statuses = Parallel(n_jobs=N_JOBS_FEATURES, prefer="threads")(
        delayed(_build_task)(task)
        for task in tqdm(new_tasks, desc="matched K5 new pseudo states")
    )
finally:
    globals()["_FI"] = original_fi
    globals()["_DI"] = original_di

status = pd.DataFrame(statuses)
status.to_csv(RUN_DIR / "feature_status.csv", index=False)
print("feature status:", status["status"].value_counts().to_dict())
failed = status[status["status"] == "failed"]
if len(failed):
    display(failed.head(30))
if len(failed) > max(10, 0.02 * len(new_tasks)):
    raise RuntimeError(f"Too many feature failures: {len(failed)} / {len(new_tasks)}")

complete_wells = []
for wid in meta["well"].astype(str):
    paths = [_sample_path(wid, tag) for tag in K5_TAGS] + [_official_full_path(wid)]
    if all(path.exists() for path in paths):
        complete_wells.append(wid)
meta = meta[meta["well"].isin(complete_wells)].copy().reset_index(drop=True)
old_oof = old_oof[old_oof["well"].isin(complete_wells)].copy().reset_index(drop=True)
print("complete K5 wells:", len(meta))
if len(meta) < 700:
    raise RuntimeError("Too few complete wells")


def _load_samples(wells, tags, arm):
    state_mass = 0.60 / (len(tags) - 1)
    keep = list(dict.fromkeys(feature_columns + ["target", "_sample_weight"]))
    frames = []
    for wid in wells:
        for tag in tags:
            frame = pd.read_pickle(_sample_path(str(wid), tag))
            n_eval = float(frame["_official_n_eval"].iloc[0])
            mass = 0.40 if tag == "official" else state_mass
            frame["_sample_weight"] = mass * n_eval / max(1, len(frame))
            for column in keep:
                if column not in frame:
                    frame[column] = np.nan
            frames.append(frame[keep].copy())
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_official_full(wells):
    keep = list(
        dict.fromkeys(
            feature_columns
            + ["target", "well", "id", "_row_idx", "last_known_tvt", "md_since", "pf_ancc", PP_LIKPF_COL]
        )
    )
    frames = []
    for wid in wells:
        frame = pd.read_pickle(_official_full_path(str(wid)))
        for column in keep:
            if column not in frame:
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
    lik = frame[PP_LIKPF_COL].to_numpy(float)
    lik = np.where(np.isfinite(lik), lik, last)
    learned = PP_ALPHA * warm * (
        model_delta * (1.0 - PP_W_PF_ANCC) + pf_delta * PP_W_PF_ANCC
    )
    prediction = last + PP_W_MODEL * learned + (1.0 - PP_W_MODEL) * (lik - last)
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


def _fit(train, valid, arm, fold):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    params = dict(LGB_PARAMS)
    params["device_type"] = "gpu"
    model = LGBMRegressor(**params)
    kwargs = dict(
        sample_weight=train["_sample_weight"].to_numpy(float),
        eval_set=[(_matrix(valid), valid["target"].to_numpy(float))],
        eval_metric="rmse",
        callbacks=[early_stopping(300, verbose=False), log_evaluation(100)],
    )
    try:
        model.fit(_matrix(train), train["target"].to_numpy(float), **kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if "gpu" not in message and "opencl" not in message:
            raise
        print("GPU unavailable; using CPU:", repr(exc))
        params.pop("device_type", None)
        model = LGBMRegressor(**params)
        model.fit(_matrix(train), train["target"].to_numpy(float), **kwargs)
    joblib.dump(model, MODEL_DIR / f"{arm}_fold{fold}.joblib")
    return model


oof_parts = []
fold_rows = []
training_started = time.time()
for fold in range(1, N_FOLDS + 1):
    train_wells = meta.loc[meta["fold"] != fold, "well"].astype(str).tolist()
    valid_wells = meta.loc[meta["fold"] == fold, "well"].astype(str).tolist()
    valid = _load_official_full(valid_wells)

    train_k3 = _load_samples(train_wells, K3_TAGS, "k3")
    model_k3 = _fit(train_k3, valid, "k3", fold)
    del train_k3
    gc.collect()

    train_k5 = _load_samples(train_wells, K5_TAGS, "k5")
    model_k5 = _fit(train_k5, valid, "k5", fold)
    del train_k5
    gc.collect()

    valid_matrix = _matrix(valid)
    pred_k3 = _postprocess(
        valid,
        model_k3.predict(valid_matrix, num_iteration=model_k3.best_iteration_),
    )
    pred_k5 = _postprocess(
        valid,
        model_k5.predict(valid_matrix, num_iteration=model_k5.best_iteration_),
    )
    truth = valid["target"].to_numpy(float) + valid["last_known_tvt"].to_numpy(float)
    k3_score = _rmse(truth, pred_k3)
    k5_score = _rmse(truth, pred_k5)
    fold_rows.append(
        {
            "fold": fold,
            "rows": len(valid),
            "wells": len(valid_wells),
            "k3_rmse": k3_score,
            "k5_rmse": k5_score,
            "k5_gain_vs_k3": k3_score - k5_score,
            "k3_best_iteration": int(model_k3.best_iteration_),
            "k5_best_iteration": int(model_k5.best_iteration_),
        }
    )
    print("fold result:", fold_rows[-1])
    oof_parts.append(
        pd.DataFrame(
            {
                "id": valid["id"].astype(str).to_numpy(),
                "well": valid["well"].astype(str).to_numpy(),
                "row_idx": valid["_row_idx"].to_numpy(int),
                "fold": fold,
                "tvt_true": truth,
                "pred_k3_4000": pred_k3,
                "pred_k5_4000": pred_k5,
            }
        )
    )
    del valid, model_k3, model_k5, valid_matrix
    gc.collect()

oof = pd.concat(oof_parts, ignore_index=True)
folds = pd.DataFrame(fold_rows)
k3_rmse = _rmse(oof["tvt_true"], oof["pred_k3_4000"])
k5_rmse = _rmse(oof["tvt_true"], oof["pred_k5_4000"])

old_compare = old_oof[["id", "pred_single"]].copy()
oof = oof.merge(old_compare, on="id", how="left", validate="one_to_one")
single_rmse = _rmse(oof["tvt_true"], oof["pred_single"])

bywell = []
for wid, group in oof.groupby("well", sort=False):
    truth = group["tvt_true"].to_numpy(float)
    p3 = group["pred_k3_4000"].to_numpy(float)
    p5 = group["pred_k5_4000"].to_numpy(float)
    bywell.append(
        {
            "well": wid,
            "rows": len(group),
            "k3_sse": float(np.sum((p3 - truth) ** 2)),
            "k5_sse": float(np.sum((p5 - truth) ** 2)),
        }
    )
bywell = pd.DataFrame(bywell)
top40 = bywell.nlargest(40, "k3_sse")
top40_k3 = float(np.sqrt(top40["k3_sse"].sum() / top40["rows"].sum()))
top40_k5 = float(np.sqrt(top40["k5_sse"].sum() / top40["rows"].sum()))

# CP12 remains a diagnostic only; it does not gate base development.
pool_path = OOF_DIR / "candidate_pool_full_merged_oof_honest_fle.csv"
cp12 = pd.read_csv(pool_path, usecols=["id", "cp12"])
cp12["id"] = cp12["id"].astype(str)
oof = oof.merge(cp12, on="id", how="left", validate="one_to_one")
if oof["cp12"].isna().any():
    raise RuntimeError(f"Missing CP12 for {int(oof['cp12'].isna().sum())} rows")
cp12_cosine = _error_cosine(oof["tvt_true"], oof["cp12"], oof["pred_k5_4000"])

increment = k3_rmse - k5_rmse
positive_folds = int((folds["k5_gain_vs_k3"] > 0).sum())
extend_k8 = bool(increment >= 0.05 and positive_folds >= 4 and top40_k5 <= top40_k3)
result = {
    "rows": int(len(oof)),
    "wells": int(oof["well"].nunique()),
    "old_single_rmse": single_rmse,
    "k3_4000_rmse": k3_rmse,
    "k5_4000_rmse": k5_rmse,
    "k5_gain_vs_k3": increment,
    "k5_gain_vs_old_single": single_rmse - k5_rmse,
    "k5_vs_k3_min_fold": float(folds["k5_gain_vs_k3"].min()),
    "k5_vs_k3_positive_folds": positive_folds,
    "top40_k3_rmse": top40_k3,
    "top40_k5_rmse": top40_k5,
    "top40_gain": top40_k3 - top40_k5,
    "error_cosine_cp12_k5": cp12_cosine,
    "feature_hours": (training_started - feature_started) / 3600.0,
    "training_hours": (time.time() - training_started) / 3600.0,
    "extend_k8": extend_k8,
    "next": (
        "K5 still rising -> append K8"
        if extend_k8
        else "K curve saturated -> use best of K3/K5 for full clean 3-model base"
    ),
}

print("\nK3 / K5 FOLDS")
display(folds)
print("\nOVERALL")
print(result)
print("\nNEXT:", result["next"])

oof.to_pickle(RUN_DIR / "matched_k5_oof.pkl")
folds.to_csv(RUN_DIR / "matched_k5_folds.csv", index=False)
bywell.to_csv(RUN_DIR / "matched_k5_bywell.csv", index=False)
pd.DataFrame([result]).to_csv(RUN_DIR / "matched_k5_summary.csv", index=False)
print("saved:", RUN_DIR)
