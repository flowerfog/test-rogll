"""V118: frozen top-129 consensus replay on cached Fold1/2/3 panels."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_consensus_support_replay_v118"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V110_SOURCE = ROOT / "rawdata_lattice_topk_path_barycenter_v110_cell.py"
V115_PATH = OOF_ROOT / "rawdata_lattice_consensus_center_fold2_replay_v115" / "fold123_consensus_center_oof.parquet"
PANEL_DIRS = {
    1: OOF_ROOT / "rawdata_lattice_consensus_center_fold1_v114" / "fold1_panels",
    2: OOF_ROOT / "rawdata_lattice_topk_barycenter_fold2_v112" / "fold2_panels",
    3: OOF_ROOT / "rawdata_lattice_blendaware_ranker_v103" / "matched_panels",
}
TOPK = 129
FREEZE_FT = 1000.0
RAMP_FT = 250.0
CAP_FT = 6.0
TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"


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


def _bootstrap_v110():
    source = V110_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V110 EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V110 marker missing from {V110_SOURCE}")
    module = types.ModuleType("raw_lattice_v118_definitions")
    module.__file__ = str(V110_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V110_SOURCE), "exec"), module.__dict__)
    return module


def _smoothstep(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


for required in (V110_SOURCE, V115_PATH, *PANEL_DIRS.values()):
    if not required.exists():
        raise FileNotFoundError(required)
R = _bootstrap_v110()
for required in (R.V81_SOURCE, R.MODEL_PATH, R.V70_PATH, R.V76_WEIGHTS):
    if not required.exists():
        raise FileNotFoundError(required)

N = R._bootstrap_v81()
Q, V, B, prior = N.Q, N.V, N.B, N.prior
model = lgb.Booster(model_file=str(R.MODEL_PATH))
frame = pd.read_parquet(
    R.V70_PATH,
    columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
    filters=[("fold", "in", [1, 2, 3])],
).sort_values(["fold", "well", "row_idx"], kind="stable").reset_index(drop=True)
frame["well"] = frame["well"].astype(str)
weights = pd.read_parquet(R.V76_WEIGHTS)
weights = weights[weights["fold"].isin([1, 2, 3])][["fold", "well", "ensemble_weight"]].copy()
weights["well"] = weights["well"].astype(str)
weight_map = dict(zip(zip(weights["fold"], weights["well"]), weights["ensemble_weight"].astype(float)))
frame["dose"] = [weight_map.get((int(fold), well), np.nan) for fold, well in zip(frame["fold"], frame["well"])]
if frame.empty or frame["dose"].isna().any() or frame["id"].duplicated().any():
    raise RuntimeError("Incomplete exact V76 source")
frame["exact_v76"] = frame[PARENT].to_numpy(np.float64) + frame["dose"].to_numpy(np.float64) * (
    frame[COARSE].to_numpy(np.float64) - frame[PARENT].to_numpy(np.float64)
)

parts = []
reports = []
for fold in (1, 2, 3):
    fold_frame = frame[frame["fold"].eq(fold)].copy()
    wells = sorted(fold_frame["well"].unique())
    for number, well in enumerate(wells, 1):
        local = fold_frame[fold_frame["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
        try:
            panel_path = PANEL_DIRS[fold] / f"{well}.npz"
            with np.load(panel_path, allow_pickle=False) as data:
                features = data["features"].astype(np.float32, copy=False)
                paths = data["paths"].astype(np.int16, copy=False)
            record = V["_load_record"](well)
            eval_rows = record["eval_rows"]
            truth = record["truth"][eval_rows].astype(np.float64)
            if len(local) != len(eval_rows) or np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
                raise RuntimeError("truth/row alignment")
            scores = model.predict(features)
            order = R._deduplicated_order(paths, scores)
            if len(order) < TOPK:
                raise RuntimeError(f"only {len(order)} unique candidates")
            segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
            center = np.mean(
                np.vstack([
                    B["_path_prediction"](record, prior, paths[index], segments)[eval_rows]
                    for index in order[:TOPK]
                ]),
                axis=0,
            )
            distance = record["md"][eval_rows].astype(np.float64) - float(record["anchor_md"])
            taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
            exact = local["exact_v76"].to_numpy(np.float64)
            coarse = local[COARSE].to_numpy(np.float64)
            dose = float(local["dose"].iloc[0])
            move = taper * np.clip(dose * (center - coarse), -CAP_FT, CAP_FT)
            local["consensus_top129"] = (exact + move).astype(np.float32)
            status = "ok"
        except Exception as exc:
            local["consensus_top129"] = local["exact_v76"].astype(np.float32)
            status = "fallback"
            reports.append({"fold": fold, "well": well, "status": status, "reason": repr(exc)})
        else:
            reports.append({"fold": fold, "well": well, "status": status, "reason": ""})
        parts.append(local)
        if number % 25 == 0 or number == len(wells):
            print(f"V118 Fold{fold} {number}/{len(wells)} elapsed_min={(time.time()-STARTED)/60:.1f}", flush=True)

output = pd.concat(parts, ignore_index=True).sort_values(["fold", "well", "row_idx"], kind="stable")
report = pd.DataFrame(reports)
report.to_csv(OUT_DIR / "build_report.csv", index=False)
fallback_wells = int(report["status"].eq("fallback").sum())
if fallback_wells > 5:
    raise RuntimeError(f"Too many fallback wells: {fallback_wells}")

top33 = pd.read_parquet(V115_PATH, columns=["id", "exact_v76", "consensus_center"])
top33 = top33.rename(columns={"consensus_center": "consensus_top33"})
output = output.merge(top33, on="id", how="left", validate="one_to_one", suffixes=("", "_v115"))
if output[["consensus_top33", "exact_v76_v115"]].isna().any().any():
    raise RuntimeError("Incomplete V115 alignment")
parity = float(np.max(np.abs(output["exact_v76"] - output["exact_v76_v115"])))
if parity > 1e-5:
    raise RuntimeError(f"V115 exact V76 parity failed: {parity}")

summary_rows = []
metrics = {}
for fold in (0, 1, 2, 3):
    local = output if fold == 0 else output[output["fold"].eq(fold)]
    metrics[str(fold)] = {}
    for endpoint in ("exact_v76", "consensus_top33", "consensus_top129"):
        result = R._metric(local, local[endpoint].to_numpy(np.float64), endpoint)
        metrics[str(fold)][endpoint] = result
        summary_rows.append({"fold": fold, **result})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)

fold_gains = {
    str(fold): metrics[str(fold)]["exact_v76"]["rmse"] - metrics[str(fold)]["consensus_top129"]["rmse"]
    for fold in (1, 2, 3)
}
pooled_gain = metrics["0"]["exact_v76"]["rmse"] - metrics["0"]["consensus_top129"]["rmse"]
increment_vs_top33 = metrics["0"]["consensus_top33"]["rmse"] - metrics["0"]["consensus_top129"]["rmse"]
checks = {
    "positive_folds_3of3": all(value > 0.0 for value in fold_gains.values()),
    "pooled_gain_ge_0p05": pooled_gain >= 0.05,
    "pooled_increment_vs_top33_ge_0p01": increment_vs_top33 >= 0.01,
    "fold1_gain_positive": fold_gains["1"] > 0.0,
    "fold2_gain_positive": fold_gains["2"] > 0.0,
    "fallback_wells_le_5": fallback_wells <= 5,
}
strict_pass = bool(all(checks.values()))

save_columns = ["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE, "exact_v76", "consensus_top33", "consensus_top129"]
output[save_columns].to_parquet(OUT_DIR / "fold123_consensus_top129_oof.parquet", index=False, compression="zstd")
verdict = {
    "experiment": EXPERIMENT,
    "stage": "frozen_top129_Fold123_replay",
    "selection_or_tuning_count": 0,
    "recipe": {"topk": TOPK, "aggregator": "uniform_mean", "alpha": 1.0, "freeze_ft": FREEZE_FT, "ramp_ft": RAMP_FT, "cap_ft": CAP_FT},
    "fold_gains": fold_gains,
    "pooled_gain": pooled_gain,
    "pooled_increment_vs_top33": increment_vs_top33,
    "metrics": metrics,
    "fallback_wells": fallback_wells,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": "PASS -> update V116 deployment TOPK to 129; FAIL -> retain top33 and test dominant-mode clustering",
}
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2, allow_nan=False), encoding="utf-8")
print("\nV118 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nV118 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
