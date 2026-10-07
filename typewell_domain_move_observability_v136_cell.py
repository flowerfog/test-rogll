"""V136: typewell-domain observability audit for the failed V135 move.

The V135 confirmation move is held fixed. Typewell GR curves are converted to
label-free fingerprints and clustered without outcome information. For each
well, a cluster dose is estimated from all *other* wells in that cluster. This
uses confirmation labels and is therefore an oracle investment audit, not a
deployable cross-fit result. Random cluster assignments are matched controls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
SOURCE = OOF_ROOT / "anchored_v127_costvolume_residual_v135" / "fold3_confirmation_oof.parquet"
OUT_DIR = OOF_ROOT / "typewell_domain_move_observability_v136"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASE = "nested_sequence_coefficient"
TRUTH = "tvt_true"
GRID_POINTS = 192
CLUSTERS = (4, 8, 16, 24, 32)
SHUFFLE_DRAWS = 32
SEED = 20260802


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _find_train_root():
    candidates = (
        ROOT / "rogii-wellbore-geology-prediction" / "train",
        ROOT / "train",
        Path.cwd() / "rogii-wellbore-geology-prediction" / "train",
        Path.cwd() / "train",
    )
    for path in candidates:
        if path.is_dir() and next(path.glob("*__typewell.csv"), None) is not None:
            return path
    raise FileNotFoundError(f"Cannot find training CSV directory: {[str(path) for path in candidates]}")


def _robust(values):
    values = np.asarray(values, np.float64)
    centre = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(np.std(values))
    return (values - centre) / max(scale, 1e-3)


def _fingerprint(path):
    data = pd.read_csv(path, usecols=["TVT", "GR"])
    data = data[np.isfinite(data["TVT"]) & np.isfinite(data["GR"])].copy()
    data = data.groupby("TVT", as_index=False, sort=True)["GR"].mean()
    if len(data) < 32:
        raise RuntimeError(f"{path.name}: only {len(data)} finite typewell rows")
    tvt = data["TVT"].to_numpy(np.float64)
    gr = data["GR"].to_numpy(np.float64)
    low, high = np.quantile(tvt, [0.01, 0.99])
    if high <= low:
        raise RuntimeError(f"{path.name}: invalid TVT range")
    coordinate = (tvt - low) / (high - low)
    grid = np.linspace(0.0, 1.0, GRID_POINTS)
    curve = np.interp(grid, coordinate, _robust(gr))
    derivative = np.gradient(curve)
    smooth = np.convolve(np.pad(curve, (4, 4), mode="edge"), np.ones(9) / 9.0, mode="valid")
    statistics = np.asarray([
        low / 15000.0,
        high / 15000.0,
        (high - low) / 1000.0,
        np.mean(gr) / 100.0,
        np.std(gr) / 50.0,
        *[value / 100.0 for value in np.quantile(gr, [0.05, 0.25, 0.50, 0.75, 0.95])],
        np.mean(np.abs(derivative)),
        np.std(derivative),
    ], np.float64)
    fingerprint = np.concatenate([curve, smooth, derivative, statistics])
    canonical = np.round(np.concatenate([curve[::3], statistics]), 3).astype(np.float32)
    digest = hashlib.sha1(canonical.tobytes()).hexdigest()[:16]
    return fingerprint.astype(np.float32), digest


def _metric(frame, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - frame[TRUTH].to_numpy(np.float64)
    distance = frame["md_since"].to_numpy(np.float64)
    bywell = pd.DataFrame({"well": frame["well"].astype(str), "error2": error * error})
    well_rmse = np.sqrt(bywell.groupby("well", sort=False)["error2"].mean())
    far = distance > 2000.0
    return {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(np.square(error[far])))),
    }


def _loo_alpha(panel, labels):
    labels = np.asarray(labels)
    numerator = panel["numerator"].to_numpy(np.float64)
    denominator = panel["denominator"].to_numpy(np.float64)
    total_num = float(numerator.sum())
    total_den = float(denominator.sum())
    output = np.zeros(len(panel), np.float64)
    group_size = np.zeros(len(panel), np.int32)
    for index, label in enumerate(labels):
        members = labels == label
        group_size[index] = int(members.sum())
        num = float(numerator[members].sum() - numerator[index])
        den = float(denominator[members].sum() - denominator[index])
        if den <= 1e-8 or group_size[index] < 2:
            num = total_num - numerator[index]
            den = total_den - denominator[index]
        output[index] = np.clip(num / max(den, 1e-8), -0.50, 0.50)
    return output, group_size


def _prediction(frame, alpha_by_well):
    alpha = frame["well"].astype(str).map(alpha_by_well)
    if alpha.isna().any():
        raise RuntimeError("Incomplete V136 alpha mapping")
    return (
        frame[BASE].to_numpy(np.float64)
        + alpha.to_numpy(np.float64) * frame["move"].to_numpy(np.float64)
    )


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
frame = pd.read_parquet(SOURCE)
frame["well"] = frame["well"].astype(str)
required = {"id", "well", "row_idx", TRUTH, "md_since", BASE, "move"}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"V135 confirmation OOF missing columns: {missing}")
if frame["id"].duplicated().any() or frame[list(required)].isna().any().any():
    raise RuntimeError("Invalid V135 confirmation OOF")

train_root = _find_train_root()
wells = sorted(frame["well"].unique())
fingerprints = []
hashes = []
for number, well in enumerate(wells, 1):
    fingerprint, digest = _fingerprint(train_root / f"{well}__typewell.csv")
    fingerprints.append(fingerprint)
    hashes.append(digest)
    if number % 25 == 0 or number == len(wells):
        print(f"V136 typewell fingerprints {number}/{len(wells)}", flush=True)
fingerprints = np.stack(fingerprints)
scaled = StandardScaler().fit_transform(fingerprints)
pca_components = min(24, len(wells) - 1, scaled.shape[1])
embedding = PCA(n_components=pca_components, whiten=True, random_state=SEED).fit_transform(scaled)

well_rows = []
for well in wells:
    local = frame[frame["well"].eq(well)]
    error = local[BASE].to_numpy(np.float64) - local[TRUTH].to_numpy(np.float64)
    move = local["move"].to_numpy(np.float64)
    well_rows.append({
        "well": well,
        "numerator": float(-np.dot(error, move)),
        "denominator": float(np.dot(move, move)),
        "oracle_alpha": float(-np.dot(error, move) / max(np.dot(move, move), 1e-8)),
    })
panel = pd.DataFrame(well_rows)
panel["typewell_hash"] = hashes
base = frame[BASE].to_numpy(np.float64)
base_metric = _metric(frame, base, BASE)

summary_rows = [dict(base_metric, family="base", clusters=0)]
assignment_rows = []
global_labels = np.zeros(len(wells), np.int32)
global_alpha, global_size = _loo_alpha(panel, global_labels)
global_map = dict(zip(wells, global_alpha))
global_metric = _metric(frame, _prediction(frame, global_map), "global_loo_alpha")
summary_rows.append(dict(global_metric, family="global_loo", clusters=1))

exact_labels, exact_uniques = pd.factorize(panel["typewell_hash"], sort=True)
exact_alpha, exact_size = _loo_alpha(panel, exact_labels)
exact_map = dict(zip(wells, exact_alpha))
exact_metric = _metric(frame, _prediction(frame, exact_map), "exact_fingerprint_loo")
summary_rows.append(dict(exact_metric, family="exact_hash", clusters=len(exact_uniques)))

cluster_payload = {}
for k in CLUSTERS:
    labels = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(embedding)
    alpha, size = _loo_alpha(panel, labels)
    prediction = _prediction(frame, dict(zip(wells, alpha)))
    metric = _metric(frame, prediction, f"typewell_cluster{k}_loo")
    summary_rows.append(dict(metric, family="typewell_cluster", clusters=k))
    cluster_payload[k] = {"labels": labels, "alpha": alpha, "size": size}
    for well, label, value, count in zip(wells, labels, alpha, size):
        assignment_rows.append({
            "well": well, "clusters": k, "cluster": int(label),
            "loo_alpha": float(value), "cluster_size": int(count),
        })

summary = pd.DataFrame(summary_rows)
summary["gain"] = base_metric["rmse"] - summary["rmse"]
summary["far_gain"] = base_metric["far_gt2000_rmse"] - summary["far_gt2000_rmse"]
summary["p95_gain"] = base_metric["p95_well_rmse"] - summary["p95_well_rmse"]
summary = summary.sort_values(["rmse", "p95_well_rmse"], kind="stable")
summary.to_csv(OUT_DIR / "domain_summary.csv", index=False)
pd.DataFrame(assignment_rows).to_csv(OUT_DIR / "cluster_assignments.csv", index=False)
panel.to_csv(OUT_DIR / "bywell_oracle_alpha.csv", index=False)

cluster_rows = summary[summary["family"].eq("typewell_cluster")]
best = cluster_rows.iloc[0]
best_k = int(best["clusters"])
best_labels = cluster_payload[best_k]["labels"]
rng = np.random.default_rng(SEED)
shuffle_rows = []
for draw in range(SHUFFLE_DRAWS):
    labels = rng.permutation(best_labels)
    alpha, _ = _loo_alpha(panel, labels)
    metric = _metric(frame, _prediction(frame, dict(zip(wells, alpha))), "shuffled_clusters")
    shuffle_rows.append({
        "draw": draw,
        "rmse": metric["rmse"],
        "gain": base_metric["rmse"] - metric["rmse"],
        "far_gt2000_rmse": metric["far_gt2000_rmse"],
        "p95_well_rmse": metric["p95_well_rmse"],
    })
shuffle = pd.DataFrame(shuffle_rows)
shuffle.to_csv(OUT_DIR / "shuffle_control.csv", index=False)

aligned_gain = float(best["gain"])
global_gain = float(base_metric["rmse"] - global_metric["rmse"])
shuffle_gain_p95 = float(shuffle["gain"].quantile(0.95))
aligned_increment = aligned_gain - shuffle_gain_p95
checks = {
    "cluster_loo_gain_ge_0p05": aligned_gain >= 0.05,
    "aligned_beats_global_ge_0p03": aligned_gain - global_gain >= 0.03,
    "aligned_beats_shuffle_p95_ge_0p03": aligned_increment >= 0.03,
    "cluster_far_gain_positive": float(best["far_gain"]) > 0.0,
    "cluster_p95_not_worse_by_gt_0p10": float(best["p95_gain"]) >= -0.10,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "typewell_domain_move_observability_v136",
    "stage": "read_only_typewell_domain_leave_one_well_out_oracle_audit",
    "deployable": False,
    "truth_usage": "cluster doses and metrics only; fingerprints are label-free",
    "wells": len(wells),
    "fingerprint_features": int(fingerprints.shape[1]),
    "pca_components": pca_components,
    "exact_fingerprint_groups": int(len(exact_uniques)),
    "base_rmse": base_metric["rmse"],
    "global_loo_gain": global_gain,
    "exact_hash_loo_gain": float(base_metric["rmse"] - exact_metric["rmse"]),
    "best_clusters": best_k,
    "best_cluster_loo_gain": aligned_gain,
    "best_cluster_far_gain": float(best["far_gain"]),
    "best_cluster_p95_gain": float(best["p95_gain"]),
    "shuffle_gain_p95": shuffle_gain_p95,
    "aligned_increment_vs_shuffle_p95": aligned_increment,
    "checks": checks,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> build fold-crossfit typewell-domain experts on the absolute trajectory"
        if strict_pass else
        "FAIL -> typewell domain does not explain move reversal; close domain routing"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV136 DOMAIN SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV136 SHUFFLE SUMMARY")
print(shuffle.describe(percentiles=[0.05, 0.50, 0.95]).to_string(), flush=True)
print("\nV136 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
