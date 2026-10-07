"""V105: top-K accessibility gate for blend-aware candidate reranking.

V103/V104 showed that a global blend-aware ranker promotes implausible paths.
This read-only Fold3 oracle asks whether the reliable standalone ranker already
retrieves useful blend-aware candidates in its top-K shortlist. No model is fit,
Fold2/Fold1 are untouched, and truth is used only for oracle choice and metrics.
"""

from __future__ import annotations

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
EXPERIMENT = "rawdata_lattice_blendaware_shortlist_oracle_v105"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V81_SOURCE = ROOT / "rawdata_lattice_structural_risk_ranker_v81b_cell.py"
V103_PANELS = OOF_ROOT / "rawdata_lattice_blendaware_ranker_v103" / "matched_panels"
BASE_MODEL_PATH = OOF_ROOT / "rawdata_lattice_parentrelative_ranker_v104" / "models" / "fold45_base.txt"
V76_WEIGHTS = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

FOLD = 3
DISCOVERY_WELLS = 24
TOPK_VALUES = (4, 8, 16, 32, 64, 128, 256)
PARENT = "incumbent_6p377_replay"
TRUTH = "tvt_true"


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


def _bootstrap_v81():
    source = V81_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V81B EXECUTION ---"
    if marker not in source:
        raise RuntimeError(f"V81b marker missing from {V81_SOURCE}")
    module = types.ModuleType("raw_lattice_v81_shortlist_oracle_definitions")
    module.__file__ = str(V81_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V81_SOURCE), "exec"), module.__dict__)
    return module


for required in (V81_SOURCE, V103_PANELS, BASE_MODEL_PATH, V76_WEIGHTS):
    if not required.exists():
        raise FileNotFoundError(required)

N = _bootstrap_v81()
Q, V, B, prior = N.Q, N.V, N.B, N.prior
incumbent = N.incumbent
fold_by_well = N.fold_by_well
base_model = lgb.Booster(model_file=str(BASE_MODEL_PATH))
weights = pd.read_parquet(V76_WEIGHTS)
weights["well"] = weights["well"].astype(str)
weight_map = dict(zip(weights["well"], weights["ensemble_weight"].astype(float)))


def _load_panel(well):
    path = V103_PANELS / f"{well}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _metric(frame, prediction, endpoint):
    prediction = np.asarray(prediction, np.float64)
    error = prediction - frame[TRUTH].to_numpy(np.float64)
    error2 = error * error
    bywell = pd.DataFrame({"well": frame["well"], "e2": error2})
    bywell = np.sqrt(bywell.groupby("well", sort=False)["e2"].mean().to_numpy())
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan,
    }


wells = sorted(
    well for well, fold in fold_by_well.items()
    if fold == FOLD and (V103_PANELS / f"{well}.npz").is_file()
)
if len(wells) < 153:
    raise RuntimeError(f"Only {len(wells)} matched Fold3 panels")
discovery = set(wells[:DISCOVERY_WELLS])

parts = []
selection_rows = []
for number, well in enumerate(wells, 1):
    local = incumbent[incumbent["well"].eq(well)].sort_values("row_idx", kind="stable").copy()
    record = V["_load_record"](well)
    panel = _load_panel(well)
    paths = panel["paths"].astype(np.int16, copy=False)
    segments = V["_segments"](record, float(Q.SEGMENT_LENGTH_FT))
    scores = base_model.predict(panel["features"].astype(np.float32, copy=False))
    order = np.argsort(-scores, kind="stable")
    eval_rows = record["eval_rows"]
    truth = record["truth"][eval_rows].astype(np.float64)
    parent = local[PARENT].to_numpy(np.float64)
    if len(local) != len(eval_rows) or np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
        raise RuntimeError(f"{well}: incumbent alignment")
    if well not in weight_map:
        raise RuntimeError(f"{well}: missing V76 dose")
    dose = float(weight_map[well])

    chosen_indices = {"base": int(order[0])}
    for topk in TOPK_VALUES:
        retained = order[: min(topk, len(order))]
        chosen_indices[f"standalone_top{topk}"] = int(
            retained[np.argmin(panel["mse"][retained])]
        )
        chosen_indices[f"blend_top{topk}"] = int(
            retained[np.argmin(panel["blend_mse"][retained])]
        )

    unique_indices = sorted(set(chosen_indices.values()))
    predictions = {
        index: B["_path_prediction"](record, prior, paths[index], segments)[eval_rows].astype(np.float64)
        for index in unique_indices
    }
    for endpoint, index in chosen_indices.items():
        candidate = predictions[index]
        local[endpoint] = (parent + dose * (candidate - parent)).astype(np.float32)
    parts.append(local)
    selection_rows.append({
        "well": well,
        "split": "discovery" if well in discovery else "confirmation",
        "dose": dose,
        "candidates": int(len(paths)),
        **{f"index_{name}": int(index) for name, index in chosen_indices.items()},
    })
    if number % 10 == 0 or number == len(wells):
        print(
            f"V105 shortlist oracle {number}/{len(wells)} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True,
        )

frame = pd.concat(parts, ignore_index=True)
pd.DataFrame(selection_rows).to_csv(OUT_DIR / "perwell_choices.csv", index=False)

summary_rows = []
for stage, selected_wells in {
    "fold3_discovery": discovery,
    "fold3_confirmation": set(wells).difference(discovery),
}.items():
    local = frame[frame["well"].isin(selected_wells)]
    for endpoint in [
        "base",
        *[name for topk in TOPK_VALUES for name in (f"standalone_top{topk}", f"blend_top{topk}")],
    ]:
        row = _metric(local, local[endpoint], endpoint)
        row["stage"] = stage
        summary_rows.append(row)
summary = pd.DataFrame(summary_rows)

discovery_table = summary[summary["stage"].eq("fold3_discovery")].copy()
base_discovery = discovery_table[discovery_table["endpoint"].eq("base")].iloc[0]
blend_discovery = discovery_table[discovery_table["endpoint"].str.startswith("blend_top")].copy()
blend_discovery["gain"] = float(base_discovery["rmse"]) - blend_discovery["rmse"]
blend_discovery["topk"] = blend_discovery["endpoint"].str.extract(r"(\d+)$").astype(int)
selected = blend_discovery.sort_values(["rmse", "topk"], kind="stable").iloc[0]
selected_topk = int(selected["topk"])

confirmation_table = summary[summary["stage"].eq("fold3_confirmation")]
base_confirmation = confirmation_table[confirmation_table["endpoint"].eq("base")].iloc[0]
standalone_confirmation = confirmation_table[
    confirmation_table["endpoint"].eq(f"standalone_top{selected_topk}")
].iloc[0]
blend_confirmation = confirmation_table[
    confirmation_table["endpoint"].eq(f"blend_top{selected_topk}")
].iloc[0]

values = {
    "discovery_gain_vs_base": float(base_discovery["rmse"] - selected["rmse"]),
    "confirmation_gain_vs_base": float(base_confirmation["rmse"] - blend_confirmation["rmse"]),
    "confirmation_increment_vs_standalone_oracle": float(
        standalone_confirmation["rmse"] - blend_confirmation["rmse"]
    ),
    "confirmation_far_gain_vs_base": float(
        base_confirmation["far_gt2000_rmse"] - blend_confirmation["far_gt2000_rmse"]
    ),
}
checks = {
    "discovery_gain_ge_0p30": values["discovery_gain_vs_base"] >= 0.30,
    "confirmation_gain_ge_0p30": values["confirmation_gain_vs_base"] >= 0.30,
    "confirmation_blend_increment_ge_0p10": values["confirmation_increment_vs_standalone_oracle"] >= 0.10,
    "confirmation_far_gain_nonnegative": values["confirmation_far_gain_vs_base"] >= 0.0,
}
strict_pass = bool(all(checks.values()))

summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
print("\nV105 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_topK_blendaware_accessibility_oracle",
    "truth_usage": "oracle shortlist choice and metrics only",
    "selection_search_count": len(TOPK_VALUES),
    "topk_values": list(TOPK_VALUES),
    "selected_topk": selected_topk,
    "protocol": {
        "fold": FOLD,
        "discovery_wells": len(discovery),
        "confirmation_wells": len(wells) - len(discovery),
        "fold2_touched": False,
        "fold1_touched": False,
    },
    "values": values,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> train a cross-fitted blend-aware reranker only inside the frozen top-K shortlist"
        if strict_pass
        else "FAIL -> blend-aware candidates are not retrievable; close the complete branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV105 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
