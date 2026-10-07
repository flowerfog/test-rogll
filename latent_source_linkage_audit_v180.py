"""V180: complete-well latent geological source linkage audit.

Build deployment-legal fingerprints from typewell GR, complete horizontal GR,
trajectory geometry, and the visible TVT_input prefix.  Compare their neighbour
graph with a train-only horizon fingerprint, then evaluate outer-fold-excluded
residual-trajectory transfer on every hidden row.  Truth and horizon columns
are used only for audit/oracle endpoints.
"""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


EXPERIMENT = "latent_source_linkage_audit_v180"
CWD = Path.cwd()
ROOT = CWD if (CWD / "train").is_dir() else Path("/root/autodl-tmp")
DATA_ROOT = next((p for p in (ROOT / "rogii-wellbore-geology-prediction", ROOT) if (p / "train").is_dir()), ROOT)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").is_dir() else ROOT / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
GRID = np.linspace(0.0, 1.0, 96)
TOPK = 32


def _clean(value):
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, (np.integer, int)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)): return bool(value)
    return value


def _read_base():
    candidates = (
        (ROOT / "oof_exports" / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet", "nested_sequence_coefficient"),
        (ROOT / "oof_exports" / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet", "incumbent_6p377_replay"),
        (DATA_ROOT / "output" / "tail_only_reanchored_neighbor_gateway_v0" / "tail_only_oof.csv.gz", "base"),
    )
    for path, prediction in candidates:
        if not path.is_file(): continue
        f = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        truth = next((c for c in ("tvt_true", "truth") if c in f), None); distance = next((c for c in ("md_since", "distance") if c in f), None)
        if truth is None or distance is None or prediction not in f: continue
        out = f[["well", "fold", "row_idx", truth, distance, prediction]].copy(); out.columns = ["well", "fold", "row_idx", "truth", "distance", "base"]
        out["well"] = out["well"].astype(str); out["fold"] = out["fold"].astype(np.int8); out["row_idx"] = out["row_idx"].astype(np.int32)
        return out.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True), str(path), prediction
    raise FileNotFoundError("No complete OOF base")


def _fill(axis, values):
    axis = np.asarray(axis, float); values = np.asarray(values, float); valid = np.isfinite(axis) & np.isfinite(values)
    return np.interp(axis, axis[valid], values[valid]) if valid.sum() >= 2 else np.zeros(len(axis))


def _resample(values, count=len(GRID)):
    values = np.asarray(values, float)
    if len(values) == 1: return np.full(count, values[0])
    return np.interp(np.linspace(0, 1, count), np.linspace(0, 1, len(values)), values)


def _robust(values):
    values = np.asarray(values, float); centre = float(np.median(values)); scale = float(np.quantile(values, .75) - np.quantile(values, .25))
    return np.clip((values - centre) / max(scale, float(np.std(values)) * .25, 1e-3), -8, 8)


def _embed(matrix, components):
    scaled = RobustScaler(quantile_range=(10, 90)).fit_transform(np.asarray(matrix, float))
    return PCA(n_components=min(components, scaled.shape[0] - 1, scaled.shape[1]), whiten=True, random_state=180).fit_transform(scaled)


def _metric(frame, prediction, endpoint):
    error = np.asarray(prediction, float) - frame["truth"].to_numpy(float)
    by_well = pd.DataFrame({"well": frame["well"], "e2": error * error}).groupby("well", sort=False)["e2"].mean(); far = frame["distance"].to_numpy(float) > 2000
    return {"endpoint": endpoint, "rows": int(len(frame)), "wells": int(frame["well"].nunique()), "exact_sse": float(np.dot(error, error)),
            "rmse": float(np.sqrt(np.mean(error * error))), "mae": float(np.mean(np.abs(error))), "median_well_rmse": float(np.median(np.sqrt(by_well))),
            "p95_well_rmse": float(np.quantile(np.sqrt(by_well), .95)), "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2)))}


started = time.time(); frame, base_source, base_column = _read_base()
if len(frame) != EXPECTED_ROWS or frame["well"].nunique() != EXPECTED_WELLS: raise RuntimeError("Incomplete base")
base = frame["base"].to_numpy(float); truth = frame["truth"].to_numpy(float)
wells = sorted(frame["well"].unique()); code = {w: i for i, w in enumerate(wells)}
legal_rows, type_rows, hidden_rows, residual_curves, exact_hashes, records = [], [], [], [], [], {}
for number, well in enumerate(wells, 1):
    local = frame[frame["well"].eq(well)]; horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv", usecols=["TVT", "GR"]).dropna()
    md = horizontal["MD"].to_numpy(float); x = _fill(md, horizontal["X"]); y = _fill(md, horizontal["Y"]); z = _fill(md, horizontal["Z"]); gr = _fill(md, horizontal["GR"])
    visible = np.flatnonzero(np.isfinite(horizontal["TVT_input"].to_numpy(float))); anchor = int(visible[-1]); eval_rows = local["row_idx"].to_numpy(np.int64)
    t = typewell[["TVT", "GR"]].to_numpy(float); t = t[np.argsort(t[:, 0], kind="stable")]
    t_signature = _robust(_resample(t[:, 1])); exact_hashes.append(hashlib.sha256(np.round(t, 3).tobytes()).hexdigest()[:16])
    h_signature = _robust(_resample(gr)); prefix_s = horizontal["TVT_input"].to_numpy(float)[visible] + z[visible]; prefix_shape = _resample(prefix_s - prefix_s[-1]) / 20.0
    geometry = np.r_[_resample(x - x[anchor], 24) / 1000, _resample(y - y[anchor], 24) / 1000, _resample(z - z[anchor], 24) / 1000,
                     [x[anchor] / 10000, y[anchor] / 10000, z[anchor] / 1000, (md[-1] - md[anchor]) / 1000]]
    legal_rows.append(np.r_[t_signature, h_signature, prefix_shape, geometry]); type_rows.append(np.r_[t_signature, (t[-1, 0] - t[0, 0]) / 1000, len(t) / 2000])
    horizon = np.column_stack([_fill(md, horizontal[name]) for name in HORIZONS]); hidden_rows.append(np.concatenate([_resample(horizon[:, j] - horizon[anchor, j], 48) / 100 for j in range(len(HORIZONS))]))
    residual = local["truth"].to_numpy(float) - local["base"].to_numpy(float); residual_curves.append(_resample(residual))
    records[well] = {"positions": local.index.to_numpy(np.int64), "eval_rows": eval_rows, "fold": int(local["fold"].iloc[0])}
    if number % 100 == 0 or number == EXPECTED_WELLS: print(f"V180 fingerprints {number}/{EXPECTED_WELLS}", flush=True)

legal_embedding = _embed(legal_rows, 40); type_embedding = _embed(type_rows, 24); hidden_embedding = _embed(hidden_rows, 24)
legal_distance = cdist(legal_embedding, legal_embedding); type_distance = cdist(type_embedding, type_embedding); hidden_distance = cdist(hidden_embedding, hidden_embedding)
np.fill_diagonal(legal_distance, np.inf); np.fill_diagonal(type_distance, np.inf); np.fill_diagonal(hidden_distance, np.inf)
fold_by_code = np.asarray([records[w]["fold"] for w in wells], np.int8); residual_curves = np.asarray(residual_curves)

legal_top1 = np.full(len(frame), np.nan); legal_mean8 = np.full(len(frame), np.nan); legal_oracle32 = np.full(len(frame), np.nan); hidden_top1 = np.full(len(frame), np.nan)
link_rows = []
for target_code, well in enumerate(wells):
    record = records[well]; allowed = np.flatnonzero(fold_by_code != record["fold"])
    legal_order = allowed[np.argsort(legal_distance[target_code, allowed])]; hidden_order = allowed[np.argsort(hidden_distance[target_code, allowed])]
    candidates = legal_order[:TOPK]; positions = record["positions"]; n = len(positions); progress = np.linspace(0, 1, n)
    transferred = np.vstack([np.interp(progress, GRID, residual_curves[c]) for c in candidates])
    target_truth = truth[positions]; target_base = base[positions]
    errors = np.mean(np.square(target_base[None, :] + transferred - target_truth[None, :]), axis=1); oracle_index = int(np.argmin(errors))
    legal_top1[positions] = target_base + transferred[0]; legal_mean8[positions] = target_base + np.mean(transferred[:8], axis=0); legal_oracle32[positions] = target_base + transferred[oracle_index]
    hidden_curve = np.interp(progress, GRID, residual_curves[hidden_order[0]]); hidden_top1[positions] = target_base + hidden_curve
    hidden5 = set(hidden_order[:5].tolist())
    link_rows.append({"well": well, "fold": record["fold"], "legal_top1": wells[legal_order[0]], "hidden_top1": wells[hidden_order[0]],
                      "hidden_top5_in_legal_top1": int(legal_order[0] in hidden5), "hidden_top5_in_legal_top8": int(bool(hidden5.intersection(legal_order[:8]))),
                      "hidden_top5_in_legal_top32": int(bool(hidden5.intersection(legal_order[:32]))), "oracle_rank": oracle_index + 1})

hash_counts = pd.Series(exact_hashes).value_counts(); duplicate_wells = int(hash_counts[hash_counts > 1].sum()); duplicate_groups = int((hash_counts > 1).sum())
rng = np.random.default_rng(180); pair_a = rng.integers(0, EXPECTED_WELLS, 100000); pair_b = rng.integers(0, EXPECTED_WELLS, 100000); keep = pair_a != pair_b
distance_spearman = float(spearmanr(legal_distance[pair_a[keep], pair_b[keep]], hidden_distance[pair_a[keep], pair_b[keep]]).statistic)
link_frame = pd.DataFrame(link_rows)
summary = pd.DataFrame([_metric(frame, base, "base_current"), _metric(frame, legal_top1, "legal_fingerprint_top1"), _metric(frame, legal_mean8, "legal_fingerprint_mean8"),
                        _metric(frame, legal_oracle32, "legal_fingerprint_oracle32"), _metric(frame, hidden_top1, "hidden_horizon_top1_oracle")])
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False); link_frame.to_csv(OUT_DIR / "linkage_bywell.csv", index=False)
lookup = summary.set_index("endpoint"); base_rmse = float(lookup.loc["base_current", "rmse"]); top1_rmse = float(lookup.loc["legal_fingerprint_top1", "rmse"]); oracle32_rmse = float(lookup.loc["legal_fingerprint_oracle32", "rmse"]); hidden_rmse = float(lookup.loc["hidden_horizon_top1_oracle", "rmse"])
verdict = {"experiment": EXPERIMENT, "stage": "outer_fold_excluded_latent_source_linkage_audit", "base_source": base_source, "base_column": base_column,
           "coverage": {"rows": len(frame), "wells": EXPECTED_WELLS}, "exact_typewell": {"groups": int(len(hash_counts)), "duplicate_groups": duplicate_groups, "wells_in_duplicate_groups": duplicate_wells, "fraction": duplicate_wells / EXPECTED_WELLS, "maximum_group": int(hash_counts.max())},
           "values": {"base_rmse": base_rmse, "legal_top1_rmse": top1_rmse, "legal_top1_gain": base_rmse - top1_rmse, "legal_oracle32_rmse": oracle32_rmse, "hidden_horizon_top1_rmse": hidden_rmse,
                      "legal_hidden_distance_spearman": distance_spearman, "hidden_top5_recall_at1": float(link_frame["hidden_top5_in_legal_top1"].mean()), "hidden_top5_recall_at8": float(link_frame["hidden_top5_in_legal_top8"].mean()),
                      "hidden_top5_recall_at32": float(link_frame["hidden_top5_in_legal_top32"].mean()), "median_oracle_rank_within_legal32": float(link_frame["oracle_rank"].median())},
           "checks": {"duplicate_typewell_fraction_ge_0p25": duplicate_wells / EXPECTED_WELLS >= .25, "legal_hidden_spearman_ge_0p40": distance_spearman >= .40,
                      "hidden_top5_recall_at32_ge_0p50": float(link_frame["hidden_top5_in_legal_top32"].mean()) >= .50, "legal_oracle32_rmse_lt_6": oracle32_rmse < 6.0,
                      "legal_top1_gain_ge_0p20": base_rmse - top1_rmse >= .20}, "elapsed_minutes": (time.time() - started) / 60.0}
verdict["strict_pass"] = bool(all(verdict["checks"].values())); verdict["next"] = "PASS -> learn legal-to-latent source retrieval" if verdict["strict_pass"] else "FAIL -> no recoverable train/test source linkage; stop historical transfer"
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nV180 MATCHED SUMMARY"); print(summary.sort_values("rmse").to_string(index=False), flush=True)
print("\nV180 FINAL VERDICT"); print(json.dumps(_clean(verdict), indent=2), flush=True); print("saved:", OUT_DIR, flush=True)
