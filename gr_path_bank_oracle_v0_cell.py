"""Fold-clean whole-trajectory candidate-bank oracle for GR path matching.

This is the first gate for the post-RankSDF GR route.  It does not train a GR
matcher.  It asks whether a deployable, fold-clean menu of anchored residual
shapes contains sufficiently good trajectories for a later path-level GR
ranker to select.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
FOLD_SOURCE = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
OUT_DIR = OOF_ROOT / "gr_path_bank_oracle_v0"
BANK_OUT = OUT_DIR / "banks"

PROGRESS_POINTS = 96
PROGRESS_GRID = np.linspace(0.0, 1.0, PROGRESS_POINTS, dtype=np.float32)
KMEANS_CLUSTERS = 32
KMEANS_SCALES = (0.75, 1.0, 1.25)
PCA_COMPONENTS = 6
PCA_LEVELS = (-2.0, -1.0, 1.0, 2.0)
LINE_QUANTILES = np.linspace(0.02, 0.98, 17)
LENGTH_NORMALISATION_ALPHA = 0.5
SMOOTH_WINDOW = 11
SMOOTH_POLYORDER = 2

PASS_ORACLE_GAIN_VS_STRONG = 1.50
PASS_MIN_FOLD_GAIN = 1.00
PASS_TOP40_GAIN = 3.00


def _rmse(truth, prediction):
    truth = np.asarray(truth, float)
    prediction = np.asarray(prediction, float)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def _safe_qcut(values, bins):
    try:
        return pd.qcut(values, bins, duplicates="drop").astype(str)
    except Exception:
        return pd.Series("all", index=values.index, dtype=str)


def _smooth_shape(progress, residual):
    progress = np.asarray(progress, float)
    residual = np.asarray(residual, float)
    valid = np.isfinite(progress) & np.isfinite(residual)
    progress = progress[valid]
    residual = residual[valid]
    if len(progress) < 8:
        raise ValueError("insufficient residual rows")
    frame = (
        pd.DataFrame({"progress": progress, "residual": residual})
        .groupby("progress", as_index=False)["residual"].median()
        .sort_values("progress")
    )
    x = np.concatenate([[0.0], frame["progress"].to_numpy(float)])
    y = np.concatenate([[0.0], frame["residual"].to_numpy(float)])
    unique_x, unique_index = np.unique(x, return_index=True)
    y = y[unique_index]
    shape = np.interp(PROGRESS_GRID, unique_x, y)
    window = min(SMOOTH_WINDOW, PROGRESS_POINTS if PROGRESS_POINTS % 2 else PROGRESS_POINTS - 1)
    if window >= 5:
        shape = savgol_filter(shape, window_length=window, polyorder=SMOOTH_POLYORDER, mode="interp")
    shape = shape - shape[0]
    return shape.astype(np.float32)


def _deduplicate_bank(templates, alphas, names):
    kept_templates = []
    kept_alphas = []
    kept_names = []
    seen = set()
    for template, alpha, name in zip(templates, alphas, names):
        template = np.asarray(template, np.float32)
        template = template - template[0]
        key = (round(float(alpha), 3), np.round(template, 2).tobytes())
        if key in seen:
            continue
        seen.add(key)
        kept_templates.append(template)
        kept_alphas.append(float(alpha))
        kept_names.append(str(name))
    return (
        np.stack(kept_templates).astype(np.float32),
        np.asarray(kept_alphas, np.float32),
        np.asarray(kept_names, dtype=str),
    )


def _build_bank(train_wells, fold, label):
    shape_matrix = np.stack([WELL_DATA[well]["shape"] for well in train_wells]).astype(np.float32)
    lengths = np.asarray([WELL_DATA[well]["eval_span"] for well in train_wells], float)
    median_length = float(np.median(lengths))
    length_ratio = np.maximum(lengths / max(median_length, 1.0), 0.1)
    normalised_shapes = shape_matrix / length_ratio[:, None] ** LENGTH_NORMALISATION_ALPHA
    pointwise_cap = float(max(64.0, min(192.0, np.quantile(np.abs(shape_matrix), 0.997))))

    templates = [np.zeros(PROGRESS_POINTS, np.float32)]
    alphas = [0.0]
    names = ["zero"]

    endpoints = shape_matrix[:, -1]
    for index, amplitude in enumerate(np.quantile(endpoints, LINE_QUANTILES)):
        templates.append((float(amplitude) * PROGRESS_GRID).astype(np.float32))
        alphas.append(0.0)
        names.append(f"line_q{index:02d}")

    for family, values, alpha in (
        ("raw", shape_matrix, 0.0),
        ("lenhalf", normalised_shapes, LENGTH_NORMALISATION_ALPHA),
    ):
        clusters = min(KMEANS_CLUSTERS, max(2, len(values) // 8))
        model = KMeans(
            n_clusters=clusters,
            random_state=20260716 + int(fold),
            n_init=10,
        ).fit(values)
        for cluster, centre in enumerate(model.cluster_centers_):
            for scale in KMEANS_SCALES:
                templates.append((centre * scale).astype(np.float32))
                alphas.append(alpha)
                names.append(f"{family}_k{cluster:02d}_s{scale:.2f}")

    pca_components = min(PCA_COMPONENTS, len(normalised_shapes) - 1, PROGRESS_POINTS)
    pca = PCA(n_components=pca_components, random_state=20260716 + int(fold)).fit(normalised_shapes)
    templates.append(pca.mean_.astype(np.float32))
    alphas.append(LENGTH_NORMALISATION_ALPHA)
    names.append("pca_mean")
    for component in range(pca_components):
        standard_deviation = float(np.sqrt(max(pca.explained_variance_[component], 0.0)))
        for level in PCA_LEVELS:
            templates.append(
                (pca.mean_ + level * standard_deviation * pca.components_[component]).astype(np.float32)
            )
            alphas.append(LENGTH_NORMALISATION_ALPHA)
            names.append(f"pca_c{component:02d}_l{level:+.0f}")

    templates, alphas, names = _deduplicate_bank(templates, alphas, names)
    output_path = BANK_OUT / f"fold{fold}_{label}.npz"
    np.savez_compressed(
        output_path,
        templates=templates,
        length_alphas=alphas,
        names=names,
        progress_grid=PROGRESS_GRID,
        median_length=np.asarray([median_length], np.float32),
        pointwise_cap=np.asarray([pointwise_cap], np.float32),
        train_wells=np.asarray(train_wells, dtype=str),
    )
    return {
        "templates": templates,
        "alphas": alphas,
        "names": names,
        "median_length": median_length,
        "pointwise_cap": pointwise_cap,
        "path": str(output_path),
    }


def _oracle_for_well(well, bank):
    data = WELL_DATA[well]
    progress = data["progress"]
    residual = data["residual"]
    target_scale = max(data["eval_span"] / max(bank["median_length"], 1.0), 0.1)
    template_scale = target_scale ** bank["alphas"]
    corrections = np.stack(
        [np.interp(progress, PROGRESS_GRID, template) for template in bank["templates"]]
    )
    corrections *= template_scale[:, None]
    corrections = np.clip(corrections, -bank["pointwise_cap"], bank["pointwise_cap"])
    errors = corrections - residual[None, :]
    mse = np.mean(errors * errors, axis=1)
    best = int(np.argmin(mse))
    prediction = data["strong"] + corrections[best]
    return best, prediction.astype(np.float32), float(np.sqrt(mse[best]))


for path in (DATA_ROOT / "train", FOLD_SOURCE):
    if not path.exists():
        raise FileNotFoundError(path)
pool_path = next((path for path in POOL_CANDIDATES if path.exists()), None)
if pool_path is None:
    raise FileNotFoundError("candidate pool not found")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BANK_OUT.mkdir(parents=True, exist_ok=True)

header = pd.read_csv(pool_path, nrows=0).columns.tolist()
required = {"id", "tvt_true", "strong"}
if not required.issubset(header):
    raise RuntimeError(f"pool missing {sorted(required - set(header))}")
usecols = [column for column in ("id", "well", "row_idx", "tvt_true", "strong", "cp12") if column in header]
pool = pd.read_csv(pool_path, usecols=usecols, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = pool["row_idx"].astype(np.int32)

fold_frame = pd.read_csv(FOLD_SOURCE, dtype={"well": str})
fold_map = fold_frame.set_index("well")["fold"].astype(int).to_dict()
pool["fold"] = pool["well"].map(fold_map)
if pool["fold"].isna().any():
    raise RuntimeError("pool contains wells missing from fold metadata")
pool["fold"] = pool["fold"].astype(int)

started = time.time()
WELL_DATA = {}
metadata_rows = []
failures = []
displacements = []

for well, frame in tqdm(pool.groupby("well", sort=False), desc="build residual shapes"):
    try:
        frame = frame.sort_values("row_idx").copy()
        horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        horizontal = pd.read_csv(horizontal_path, usecols=["MD", "X", "Y", "GR", "TVT_input"])
        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
        x = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(float)
        gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
        rows = frame["row_idx"].to_numpy(int)
        if rows.min() < 1 or rows.max() >= len(horizontal):
            raise ValueError("pool row index outside horizontal well")
        anchor = int(rows.min() - 1)
        anchor_md = float(md[anchor])
        eval_md = md[rows]
        eval_span = float(max(eval_md[-1] - anchor_md, 1.0))
        progress = np.clip((eval_md - anchor_md) / eval_span, 0.0, 1.0)
        truth = frame["tvt_true"].to_numpy(float)
        strong = frame["strong"].to_numpy(float)
        residual = truth - strong
        shape = _smooth_shape(progress, residual)
        dx = float(x[-1] - x[0])
        dy = float(y[-1] - y[0])
        norm = float(np.hypot(dx, dy))
        unit = np.asarray([dx, dy], float) / max(norm, 1e-9)
        displacements.append(unit)
        WELL_DATA[str(well)] = {
            "progress": progress.astype(np.float32),
            "residual": residual.astype(np.float32),
            "truth": truth.astype(np.float32),
            "strong": strong.astype(np.float32),
            "ids": frame["id"].to_numpy(str),
            "row_idx": rows.astype(np.int32),
            "fold": int(frame["fold"].iloc[0]),
            "shape": shape,
            "eval_span": eval_span,
            "unit": unit,
        }
        metadata_rows.append(
            {
                "well": str(well),
                "fold": int(frame["fold"].iloc[0]),
                "rows": len(frame),
                "base_sse": float(np.sum((strong - truth) ** 2)),
                "strong_rmse": _rmse(truth, strong),
                "eval_span": eval_span,
                "known_frac": float(np.isfinite(tvt_input).mean()),
                "gr_missing_frac": float(np.mean(~np.isfinite(gr))),
                "dx": dx,
                "dy": dy,
                "azimuth_deg": float(np.degrees(np.arctan2(dy, dx))),
            }
        )
    except Exception as exc:
        failures.append({"well": str(well), "reason": repr(exc)})

metadata = pd.DataFrame(metadata_rows)
failure_report = pd.DataFrame(failures)
failure_report.to_csv(OUT_DIR / "failures.csv", index=False)
if failures:
    raise RuntimeError(f"failed wells: {len(failures)}; see {OUT_DIR / 'failures.csv'}")
if len(metadata) != pool["well"].nunique():
    raise RuntimeError(f"well mismatch: built {len(metadata)} / {pool['well'].nunique()}")

# Infer the signless dominant drilling axis, then assign the two opposite
# directions by projection onto that axis.
units = np.stack([WELL_DATA[well]["unit"] for well in metadata["well"]])
eigenvalues, eigenvectors = np.linalg.eigh(units.T @ units)
dominant_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
if dominant_axis[0] < 0:
    dominant_axis = -dominant_axis
direction_projection = units @ dominant_axis
metadata["direction"] = np.where(direction_projection >= 0.0, "positive", "negative")
metadata["direction_projection"] = direction_projection
metadata["known_frac_bin"] = _safe_qcut(metadata["known_frac"], 4)
metadata["gr_missing_bin"] = _safe_qcut(metadata["gr_missing_frac"], 4)
direction_map = metadata.set_index("well")["direction"].to_dict()
for well in WELL_DATA:
    WELL_DATA[well]["direction"] = direction_map[well]
metadata.to_csv(OUT_DIR / "well_metadata.csv", index=False)

detail_rows = []
oof_parts = []
bank_report_rows = []
all_wells = metadata["well"].tolist()

for fold in sorted(metadata["fold"].unique()):
    train_wells = metadata.loc[metadata["fold"] != fold, "well"].tolist()
    valid_wells = metadata.loc[metadata["fold"] == fold, "well"].tolist()
    global_bank = _build_bank(train_wells, int(fold), "global")
    banks = {"global": global_bank}
    bank_report_rows.append(
        {
            "fold": fold,
            "arm": "global",
            "train_wells": len(train_wells),
            "candidates": len(global_bank["names"]),
            "median_length": global_bank["median_length"],
            "pointwise_cap": global_bank["pointwise_cap"],
        }
    )
    for direction in ("positive", "negative"):
        direction_train = [well for well in train_wells if WELL_DATA[well]["direction"] == direction]
        bank = _build_bank(direction_train, int(fold), f"direction_{direction}")
        banks[f"direction_{direction}"] = bank
        bank_report_rows.append(
            {
                "fold": fold,
                "arm": f"direction_{direction}",
                "train_wells": len(direction_train),
                "candidates": len(bank["names"]),
                "median_length": bank["median_length"],
                "pointwise_cap": bank["pointwise_cap"],
            }
        )

    for well in tqdm(valid_wells, desc=f"fold {fold} path oracle", leave=False):
        data = WELL_DATA[well]
        predictions = {}
        for arm in ("global", "direction"):
            bank_key = "global" if arm == "global" else f"direction_{data['direction']}"
            bank = banks[bank_key]
            best, prediction, oracle_rmse = _oracle_for_well(well, bank)
            predictions[arm] = prediction
            detail_rows.append(
                {
                    "well": well,
                    "fold": fold,
                    "arm": arm,
                    "direction": data["direction"],
                    "rows": len(data["truth"]),
                    "strong_rmse": _rmse(data["truth"], data["strong"]),
                    "oracle_rmse": oracle_rmse,
                    "gain": _rmse(data["truth"], data["strong"]) - oracle_rmse,
                    "candidate_index": best,
                    "candidate_name": str(bank["names"][best]),
                    "candidate_count": len(bank["names"]),
                }
            )
        oof_parts.append(
            pd.DataFrame(
                {
                    "id": data["ids"],
                    "well": well,
                    "row_idx": data["row_idx"],
                    "fold": fold,
                    "tvt_true": data["truth"],
                    "strong": data["strong"],
                    "oracle_global": predictions["global"],
                    "oracle_direction": predictions["direction"],
                }
            )
        )

detail = pd.DataFrame(detail_rows)
oof = pd.concat(oof_parts, ignore_index=True)
bank_report = pd.DataFrame(bank_report_rows)
detail.to_csv(OUT_DIR / "oracle_bywell.csv", index=False)
oof.to_pickle(OUT_DIR / "oracle_oof.pkl")
bank_report.to_csv(OUT_DIR / "bank_report.csv", index=False)

summary_rows = []
fold_rows = []
top40 = (
    metadata.sort_values("base_sse", ascending=False).head(40)["well"].astype(str).tolist()
)
for arm in ("global", "direction"):
    prediction_column = f"oracle_{arm}"
    truth = oof["tvt_true"].to_numpy(float)
    strong = oof["strong"].to_numpy(float)
    prediction = oof[prediction_column].to_numpy(float)
    arm_detail = detail[detail["arm"] == arm].copy()
    top_mask = oof["well"].isin(top40).to_numpy()
    summary_rows.append(
        {
            "arm": arm,
            "rows": len(oof),
            "wells": oof["well"].nunique(),
            "strong_pooled_rmse": _rmse(truth, strong),
            "oracle_pooled_rmse": _rmse(truth, prediction),
            "pooled_gain": _rmse(truth, strong) - _rmse(truth, prediction),
            "strong_mean_well_rmse": float(arm_detail["strong_rmse"].mean()),
            "oracle_mean_well_rmse": float(arm_detail["oracle_rmse"].mean()),
            "mean_well_gain": float(arm_detail["gain"].mean()),
            "median_well_gain": float(arm_detail["gain"].median()),
            "positive_well_fraction": float((arm_detail["gain"] > 0).mean()),
            "top40_strong_rmse": _rmse(truth[top_mask], strong[top_mask]),
            "top40_oracle_rmse": _rmse(truth[top_mask], prediction[top_mask]),
            "top40_gain": _rmse(truth[top_mask], strong[top_mask]) - _rmse(truth[top_mask], prediction[top_mask]),
        }
    )
    for fold in sorted(oof["fold"].unique()):
        mask = oof["fold"].eq(fold).to_numpy()
        fold_rows.append(
            {
                "arm": arm,
                "fold": int(fold),
                "rows": int(mask.sum()),
                "strong_rmse": _rmse(truth[mask], strong[mask]),
                "oracle_rmse": _rmse(truth[mask], prediction[mask]),
                "gain": _rmse(truth[mask], strong[mask]) - _rmse(truth[mask], prediction[mask]),
            }
        )

summary = pd.DataFrame(summary_rows).sort_values("oracle_pooled_rmse")
fold_scores = pd.DataFrame(fold_rows)
summary.to_csv(OUT_DIR / "oracle_summary.csv", index=False)
fold_scores.to_csv(OUT_DIR / "oracle_folds.csv", index=False)

strata_rows = []
best_arm = str(summary.iloc[0]["arm"])
best_detail = detail[detail["arm"] == best_arm].merge(
    metadata[["well", "known_frac_bin", "gr_missing_bin", "direction"]],
    on=["well", "direction"],
    how="left",
    validate="one_to_one",
)
for grouping in ("direction", "known_frac_bin", "gr_missing_bin"):
    for value, group in best_detail.groupby(grouping, dropna=False):
        strata_rows.append(
            {
                "grouping": grouping,
                "value": str(value),
                "wells": len(group),
                "strong_mean_well_rmse": float(group["strong_rmse"].mean()),
                "oracle_mean_well_rmse": float(group["oracle_rmse"].mean()),
                "mean_well_gain": float(group["gain"].mean()),
                "median_well_gain": float(group["gain"].median()),
                "positive_well_fraction": float((group["gain"] > 0).mean()),
            }
        )
strata = pd.DataFrame(strata_rows)
strata.to_csv(OUT_DIR / "oracle_strata.csv", index=False)

best_summary = summary.iloc[0].to_dict()
best_folds = fold_scores[fold_scores["arm"] == best_arm]
strict_pass = bool(
    best_summary["pooled_gain"] >= PASS_ORACLE_GAIN_VS_STRONG
    and best_folds["gain"].min() >= PASS_MIN_FOLD_GAIN
    and best_summary["top40_gain"] >= PASS_TOP40_GAIN
)
verdict = {
    "experiment": "gr_path_bank_oracle_v0",
    "pool": str(pool_path),
    "fold_source": str(FOLD_SOURCE),
    "dominant_direction_axis_xy": dominant_axis.tolist(),
    "direction_counts": metadata["direction"].value_counts().to_dict(),
    "best_arm": best_arm,
    "best": best_summary,
    "min_fold_gain": float(best_folds["gain"].min()),
    "gates": {
        "oracle_gain_vs_strong_min": PASS_ORACLE_GAIN_VS_STRONG,
        "min_fold_gain": PASS_MIN_FOLD_GAIN,
        "top40_gain_min": PASS_TOP40_GAIN,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> build path-level multi-scale GR features and listwise ranker"
        if strict_pass
        else "FAIL -> residual-shape menu lacks enough trajectory ceiling; stop this GR path-bank route"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("\nBANKS")
print(bank_report.to_string(index=False))
print("\nORACLE SUMMARY")
print(summary.to_string(index=False))
print("\nFOLDS")
print(fold_scores.to_string(index=False))
print("\nSTRATA")
print(strata.to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
