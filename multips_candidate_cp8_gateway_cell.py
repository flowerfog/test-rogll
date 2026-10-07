"""Multi-pseudo-PS candidate CP8 gateway.

Prerequisite:
    multicut_fleongg_v0_train_cell.py has already built the masked feature
    cache for official, synth1, and synth2 starts.

This is a strict A/B:
    single_ps: train CP8 only on official-start samples;
    multi_ps:  train the identical CP8 on official+synth1+synth2 samples.

Validation always uses held-out wells at their untouched official start.
Folds are grouped by original well. Raw GR features are deliberately excluded.
The candidate curves are reconstructed from deployable PF, beam, selector-scale,
hybrid, and likelihood-PF columns in each masked feature frame.
"""

from pathlib import Path
import gc
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from lightgbm import LGBMRegressor
from tqdm.auto import tqdm


warnings.filterwarnings("ignore")

OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
PREFERRED_SOURCE_DIR = (
    OOF_ROOT / "multicut_fleongg_v0_k3_nospatial_r768_lgb1"
)
OUT_DIR = OOF_ROOT / "multips_candidate_cp8_gateway_v0"

N_CP = 8
N_FOLDS = 5
SEED = 20260713
WINDOW_HALF_FRACTION = 0.04
CP_ALPHA = 0.50
CP_CAP = 6.0

TAGS_SINGLE = ("official",)
TAGS_MULTI = ("official", "synth1", "synth2")

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
)
CONTEXT_COLUMNS = (
    "known_len",
    "eval_len",
    "md_since",
    "frac",
    "frac2",
    "sqrt_frac",
    "z",
    "dx",
    "dy",
    "dz",
    "dxy",
    "dzdmd",
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


def _discover_cache():
    search_roots = [
        OOF_ROOT,
        Path("/root/autodl-tmp/.Trash-0"),
    ]
    feature_dirs = []
    preferred_features = PREFERRED_SOURCE_DIR / "features"
    if preferred_features.exists():
        feature_dirs.append(preferred_features)

    for root in search_roots:
        if not root.exists():
            continue
        for directory in root.rglob("features"):
            if not directory.is_dir() or directory in feature_dirs:
                continue
            if next(directory.glob("*__official__sample.pkl"), None) is not None:
                feature_dirs.append(directory)

    metadata_paths = []
    preferred_meta = PREFERRED_SOURCE_DIR / "well_metadata.csv"
    if preferred_meta.exists():
        metadata_paths.append(preferred_meta)
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("well_metadata.csv"):
            if path not in metadata_paths:
                metadata_paths.append(path)

    for feature_dir in feature_dirs:
        local_meta = feature_dir.parent / "well_metadata.csv"
        candidates = [local_meta, *metadata_paths]
        for meta_path in candidates:
            if not meta_path.exists():
                continue
            try:
                metadata = pd.read_csv(meta_path, usecols=["well", "fold"])
            except Exception:
                continue
            n_official = sum(1 for _ in feature_dir.glob("*__official__sample.pkl"))
            n_synth1 = sum(1 for _ in feature_dir.glob("*__synth1__sample.pkl"))
            n_synth2 = sum(1 for _ in feature_dir.glob("*__synth2__sample.pkl"))
            n_full = sum(1 for _ in feature_dir.glob("*__official_full.pkl"))
            if (
                metadata["well"].nunique() >= 700
                and min(n_official, n_synth1, n_synth2, n_full) >= 700
            ):
                return feature_dir.parent, feature_dir, meta_path, {
                    "official": n_official,
                    "synth1": n_synth1,
                    "synth2": n_synth2,
                    "full": n_full,
                }

    nearby = []
    if OOF_ROOT.exists():
        nearby.extend(str(path) for path in OOF_ROOT.glob("*multicut*"))
    trash = Path("/root/autodl-tmp/.Trash-0")
    if trash.exists():
        nearby.extend(str(path) for path in trash.rglob("*multicut*"))
    raise FileNotFoundError(
        "The 2316 masked feature frames are not available. "
        "Found multicut-related paths:\n" + "\n".join(nearby[:50])
    )


SOURCE_DIR, FEATURE_DIR, META_PATH, CACHE_COUNTS = _discover_cache()
print(
    {
        "discovered_source": str(SOURCE_DIR),
        "feature_dir": str(FEATURE_DIR),
        "metadata": str(META_PATH),
        "cache_counts": CACHE_COUNTS,
    }
)


def _rmse(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _stats(values, prefix):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        finite = np.array([0.0])
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_p10": float(np.quantile(finite, 0.10)),
        f"{prefix}_p50": float(np.quantile(finite, 0.50)),
        f"{prefix}_p90": float(np.quantile(finite, 0.90)),
        f"{prefix}_span": float(np.ptp(finite)),
    }


def _finite_curve(values, fallback):
    values = np.asarray(values, float).copy()
    fallback = np.asarray(fallback, float)
    values[~np.isfinite(values)] = np.nan
    if np.isfinite(values).any():
        values = (
            pd.Series(values)
            .interpolate(limit_direction="both")
            .to_numpy(float)
        )
    else:
        values = fallback.copy()
    bad = ~np.isfinite(values)
    values[bad] = fallback[bad]
    return values


def _feature_path(wid, tag, full=False):
    if full:
        if tag != "official":
            raise ValueError("only official has a full-frame cache")
        return FEATURE_DIR / f"{wid}__official_full.pkl"
    return FEATURE_DIR / f"{wid}__{tag}__sample.pkl"


meta = pd.read_csv(META_PATH)
meta["well"] = meta["well"].astype(str)
if "fold" not in meta.columns:
    raise RuntimeError("well_metadata.csv has no fold column")

complete = []
for wid in meta["well"]:
    paths = [_feature_path(wid, tag) for tag in TAGS_MULTI]
    paths.append(_feature_path(wid, "official", full=True))
    if all(path.exists() for path in paths):
        complete.append(wid)
meta = meta[meta["well"].isin(complete)].copy().reset_index(drop=True)
if len(meta) < 700:
    raise RuntimeError(f"only {len(meta)} complete wells")
fold_map = meta.set_index("well")["fold"].astype(int).to_dict()


def _prepare_frame(frame):
    frame = frame.copy()
    if "_row_idx" in frame.columns:
        frame = frame.sort_values("_row_idx")
    frame = frame.reset_index(drop=True)
    required = {"last_known_tvt", "target"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"frame missing {sorted(missing)}")
    return frame


def _candidate_curves(frame):
    last = _finite_curve(
        frame["last_known_tvt"].to_numpy(float),
        np.zeros(len(frame), dtype=float),
    )
    curves = {}
    for column in ABSOLUTE_CANDIDATES:
        if column in frame.columns:
            curves[column] = _finite_curve(
                frame[column].to_numpy(float), last
            )
    for column in DELTA_CANDIDATES:
        if column in frame.columns:
            delta = _finite_curve(
                frame[column].to_numpy(float),
                np.zeros(len(frame), dtype=float),
            )
            curves[column] = last + delta

    required = {"pf_ancc", "likpf_mean"}
    if not required <= set(curves):
        raise RuntimeError(
            f"candidate frame missing base curves: {sorted(required-set(curves))}"
        )
    # Fixed, unoptimised base shared by the single- and multi-PS arms.
    base = 0.50 * curves["pf_ancc"] + 0.50 * curves["likpf_mean"]
    return last, base, curves


def _build_states(frame, wid, tag):
    frame = _prepare_frame(frame)
    last, base, curves = _candidate_curves(frame)
    truth = frame["target"].to_numpy(float) + last
    n_rows = len(frame)
    coordinate = (
        frame["_row_idx"].to_numpy(float)
        if "_row_idx" in frame.columns
        else np.arange(n_rows, dtype=float)
    )
    if n_rows < 64:
        raise RuntimeError(f"{wid}/{tag}: only {n_rows} rows")

    centers = np.unique(
        np.round(np.linspace(0, n_rows - 1, N_CP)).astype(int)
    )
    if len(centers) != N_CP:
        raise RuntimeError(f"{wid}/{tag}: expected {N_CP} controls")
    half_window = max(
        8, int(round(WINDOW_HALF_FRACTION * max(n_rows - 1, 1)))
    )
    stack = np.column_stack([curves[name] for name in sorted(curves)])
    rows = []

    for order, center in enumerate(centers):
        left = max(0, center - half_window)
        right = min(n_rows, center + half_window + 1)
        local = np.arange(left, right)
        feature = {
            "well": wid,
            "tag": tag,
            "fold": int(fold_map[wid]),
            "center_local": int(center),
            "n_rows": n_rows,
            "cp_order": order,
            "cp_order_norm": order / max(N_CP - 1, 1),
            "row_norm": center / max(n_rows - 1, 1),
            "target_residual": float(np.mean(truth[local] - base[local])),
            "base_center": float(base[center]),
            "base_slope": float(
                (base[right - 1] - base[left])
                / max(coordinate[right - 1] - coordinate[left], 1.0)
            ),
            "candidate_std_mean": float(
                np.mean(np.std(stack[local], axis=1))
            ),
            "candidate_range_mean": float(
                np.mean(
                    np.max(stack[local], axis=1)
                    - np.min(stack[local], axis=1)
                )
            ),
        }
        feature.update(_stats(base[local], "base"))
        for column in CONTEXT_COLUMNS:
            if column in frame.columns:
                feature.update(
                    _stats(frame[column].to_numpy(float)[local], f"ctx_{column}")
                )
        for name in sorted(curves):
            difference = curves[name][local] - base[local]
            prefix = f"candidate_{name}"
            feature.update(_stats(difference, prefix))
            feature[f"{prefix}_center"] = float(
                curves[name][center] - base[center]
            )
            feature[f"{prefix}_slope"] = float(
                (difference[-1] - difference[0])
                / max(coordinate[right - 1] - coordinate[left], 1.0)
            )
        rows.append(feature)
    return pd.DataFrame(rows), base, truth


# ---------------------------------------------------------------------------
# Build the training-state table from cached sampled frames.
# ---------------------------------------------------------------------------
state_parts = []
failures = []
started = time.time()
for wid in tqdm(meta["well"], desc="multi-PS CP8 training states"):
    for tag in TAGS_MULTI:
        try:
            frame = pd.read_pickle(_feature_path(wid, tag))
            states, _, _ = _build_states(frame, wid, tag)
            state_parts.append(states)
        except Exception as exc:
            failures.append({"well": wid, "tag": tag, "reason": repr(exc)})

pd.DataFrame(failures).to_csv(OUT_DIR / "state_failures.csv", index=False)
if failures:
    display(pd.DataFrame(failures).head(30))
    raise RuntimeError(f"failed to build {len(failures)} sampled states")

states = pd.concat(state_parts, ignore_index=True, sort=False)
metadata_columns = {
    "well",
    "tag",
    "fold",
    "center_local",
    "target_residual",
}
feature_columns = [
    column
    for column in states.columns
    if column not in metadata_columns
    and pd.api.types.is_numeric_dtype(states[column])
]
print(
    {
        "experiment": "multips_candidate_cp8_gateway_v0",
        "wells": states["well"].nunique(),
        "states": states.shape,
        "features": len(feature_columns),
        "single_states": int(states["tag"].eq("official").sum()),
        "multi_states": len(states),
        "elapsed_sec": time.time() - started,
    }
)


def _matrix(frame):
    return (
        frame.reindex(columns=feature_columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _new_model(fold, arm):
    return LGBMRegressor(
        objective="huber",
        alpha=0.90,
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.25,
        reg_lambda=6.0,
        random_state=SEED + fold + (0 if arm == "single_ps" else 100),
        n_jobs=-1,
        verbosity=-1,
    )


def _expand_controls(control_values, positions, n_rows):
    order = np.argsort(positions)
    positions = np.asarray(positions, float)[order]
    control_values = np.asarray(control_values, float)[order]
    keep = np.r_[True, np.diff(positions) > 0]
    positions = positions[keep]
    control_values = control_values[keep]
    if len(positions) == 1:
        move = np.full(n_rows, control_values[0], dtype=float)
    else:
        move = np.interp(
            np.arange(n_rows, dtype=float), positions, control_values
        )
    sigma = max(4.0, n_rows / N_CP / 8.0)
    move = gaussian_filter1d(move, sigma=sigma, mode="nearest")
    return np.clip(CP_ALPHA * move, -CP_CAP, CP_CAP)


# ---------------------------------------------------------------------------
# Five-fold A/B. Model fitting never reads held-out labels. Official full
# frames are loaded only after both models for the fold have been fitted.
# ---------------------------------------------------------------------------
oof_parts = []
fold_rows = []
model_rows = []
cv_started = time.time()
for fold in range(1, N_FOLDS + 1):
    train_wells = set(meta.loc[meta["fold"] != fold, "well"].astype(str))
    valid_wells = meta.loc[meta["fold"] == fold, "well"].astype(str).tolist()
    train_common = states[states["well"].isin(train_wells)]
    train_single = train_common[train_common["tag"].eq("official")]
    train_multi = train_common

    models = {}
    for arm, train_frame, n_tags in (
        ("single_ps", train_single, 1),
        ("multi_ps", train_multi, len(TAGS_MULTI)),
    ):
        model = _new_model(fold, arm)
        well_weight = train_frame["well"].map(
            meta.set_index("well")["n_eval"].to_dict()
        ).to_numpy(float)
        sample_weight = well_weight / (n_tags * N_CP)
        model.fit(
            _matrix(train_frame),
            train_frame["target_residual"].to_numpy(np.float32),
            sample_weight=sample_weight,
        )
        models[arm] = model
        model_rows.append(
            {
                "arm": arm,
                "fold": fold,
                "train_wells": len(train_wells),
                "train_states": len(train_frame),
                "n_estimators": model.n_estimators,
            }
        )

    fold_output = []
    for wid in tqdm(valid_wells, desc=f"official validation fold {fold}"):
        frame = pd.read_pickle(_feature_path(wid, "official", full=True))
        valid_states, base_curve, truth_curve = _build_states(
            frame, wid, "official_full"
        )
        item = {
            "id": (
                _prepare_frame(frame)["id"].astype(str).to_numpy()
                if "id" in frame.columns
                else np.array(
                    [f"{wid}_{i}" for i in range(len(frame))], dtype=object
                )
            ),
            "well": np.repeat(wid, len(frame)),
            "fold": np.repeat(fold, len(frame)),
            "tvt_true": truth_curve,
            "base": base_curve,
        }
        for arm, model in models.items():
            controls = model.predict(_matrix(valid_states))
            move = _expand_controls(
                controls,
                valid_states["center_local"].to_numpy(float),
                len(frame),
            )
            item[arm] = base_curve + move
        fold_output.append(pd.DataFrame(item))

    fold_output = pd.concat(fold_output, ignore_index=True)
    base_rmse = _rmse(fold_output["tvt_true"], fold_output["base"])
    single_rmse = _rmse(fold_output["tvt_true"], fold_output["single_ps"])
    multi_rmse = _rmse(fold_output["tvt_true"], fold_output["multi_ps"])
    fold_rows.append(
        {
            "fold": fold,
            "rows": len(fold_output),
            "wells": len(valid_wells),
            "base_rmse": base_rmse,
            "single_rmse": single_rmse,
            "multi_rmse": multi_rmse,
            "single_gain_vs_base": base_rmse - single_rmse,
            "multi_gain_vs_base": base_rmse - multi_rmse,
            "multi_gain_vs_single": single_rmse - multi_rmse,
        }
    )
    oof_parts.append(fold_output)
    print("fold result:", fold_rows[-1], flush=True)
    del fold_output, models
    gc.collect()

oof = pd.concat(oof_parts, ignore_index=True)
fold_report = pd.DataFrame(fold_rows)
base_rmse = _rmse(oof["tvt_true"], oof["base"])
single_rmse = _rmse(oof["tvt_true"], oof["single_ps"])
multi_rmse = _rmse(oof["tvt_true"], oof["multi_ps"])

bywell = []
for wid, group in oof.groupby("well", sort=False):
    y = group["tvt_true"].to_numpy(float)
    single = group["single_ps"].to_numpy(float)
    multi = group["multi_ps"].to_numpy(float)
    base_well = group["base"].to_numpy(float)
    bywell.append(
        {
            "well": wid,
            "rows": len(group),
            "base_sse": float(np.sum((base_well - y) ** 2)),
            "single_rmse": _rmse(y, single),
            "multi_rmse": _rmse(y, multi),
            "gain": _rmse(y, single) - _rmse(y, multi),
        }
    )
bywell = pd.DataFrame(bywell)
top40 = set(bywell.nlargest(40, "base_sse")["well"])
top_mask = oof["well"].isin(top40).to_numpy()
top40_gain = (
    _rmse(oof.loc[top_mask, "tvt_true"], oof.loc[top_mask, "single_ps"])
    - _rmse(oof.loc[top_mask, "tvt_true"], oof.loc[top_mask, "multi_ps"])
)

strict_pass = bool(
    single_rmse - multi_rmse >= 0.10
    and (fold_report["multi_gain_vs_single"] >= 0.02).all()
    and (fold_report["multi_gain_vs_base"] > 0).all()
)
result = {
    "base_rmse": base_rmse,
    "single_ps_rmse": single_rmse,
    "multi_ps_rmse": multi_rmse,
    "single_gain_vs_base": base_rmse - single_rmse,
    "multi_gain_vs_base": base_rmse - multi_rmse,
    "multi_gain_vs_single": single_rmse - multi_rmse,
    "mean_fold_gain_vs_single": float(
        fold_report["multi_gain_vs_single"].mean()
    ),
    "min_fold_gain_vs_single": float(
        fold_report["multi_gain_vs_single"].min()
    ),
    "positive_folds_vs_single": int(
        (fold_report["multi_gain_vs_single"] > 0).sum()
    ),
    "top40_gain_vs_single": top40_gain,
    "runtime_minutes": (time.time() - cv_started) / 60.0,
    "strict_pass": strict_pass,
    "next": (
        "rebuild full CP12/strong candidate pool at multiple pseudo starts"
        if strict_pass
        else "stop multi-pseudo-PS candidate augmentation"
    ),
}

print("\nMODEL FOLDS")
display(pd.DataFrame(model_rows))
print("\nA/B FOLDS")
display(fold_report)
print("\nOVERALL")
print(result)
print(
    "\nVERDICT:",
    "PASS -> full multi-PS CP8" if strict_pass else "FAIL / STOP multi-PS",
)

oof.to_pickle(OUT_DIR / "multips_candidate_cp8_oof.pkl")
fold_report.to_csv(OUT_DIR / "fold_report.csv", index=False)
bywell.to_csv(OUT_DIR / "bywell_report.csv", index=False)
pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_report.csv", index=False)
states.to_pickle(OUT_DIR / "sampled_state_table.pkl")
(OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
print("saved:", OUT_DIR)
