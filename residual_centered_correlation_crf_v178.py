"""V178: residual-centred native GR/typewell correlation CRF gateway.

Unlike the failed absolute-state CRF, this decoder searches only a +/-12 ft
correction around an existing complete-well base.  Native horizontal GR and
the assigned typewell GR supply multiscale NCC emissions.  Complete-well
fivefold nesting selects one fixed transition/prior recipe and blend dose.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


EXPERIMENT = "residual_centered_correlation_crf_v178"
CWD = Path.cwd()
ROOT = CWD if (CWD / "train").is_dir() else Path("/root/autodl-tmp")
DATA_ROOT = next((p for p in (ROOT / "rogii-wellbore-geology-prediction", ROOT) if (p / "train").is_dir()), ROOT)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").is_dir() else ROOT / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_ROWS = 3_783_989
EXPECTED_WELLS = 773
NODES = 128
OFFSETS = np.arange(-12.0, 12.01, 0.5, dtype=np.float64)
PATCH_GRID = np.linspace(-1.0, 1.0, 33)
SCALE_PAIRS = ((64.0, 4.0), (128.0, 8.0), (256.0, 16.0), (512.0, 32.0))
ROLL_FRACTION = 0.20
RECIPES = {
    "t025_z0005": (0.025, 0.0005),
    "t050_z0005": (0.050, 0.0005),
    "t100_z0005": (0.100, 0.0005),
    "t050_z0020": (0.050, 0.0020),
    "t100_z0020": (0.100, 0.0020),
    "t200_z0020": (0.200, 0.0020),
}


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _read_base():
    candidates = (
        (ROOT / "oof_exports" / "final_consensus_sequence_coefficient_full5_v144" / "crossfit_oof.parquet", "nested_sequence_coefficient"),
        (ROOT / "oof_exports" / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet", "incumbent_6p377_replay"),
        (DATA_ROOT / "output" / "tail_only_reanchored_neighbor_gateway_v0" / "tail_only_oof.csv.gz", "base"),
    )
    for path, prediction in candidates:
        if not path.is_file():
            continue
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        truth = next((c for c in ("tvt_true", "truth") if c in frame), None)
        distance = next((c for c in ("md_since", "distance") if c in frame), None)
        if truth is None or distance is None or prediction not in frame:
            continue
        out = frame[["well", "fold", "row_idx", truth, distance, prediction]].copy()
        out.columns = ["well", "fold", "row_idx", "truth", "distance", "base"]
        out["well"] = out["well"].astype(str)
        out["fold"] = out["fold"].astype(np.int8)
        out["row_idx"] = out["row_idx"].astype(np.int32)
        return out.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True), str(path), prediction
    raise FileNotFoundError("No complete supported OOF base")


def _fill(axis, values):
    axis = np.asarray(axis, np.float64)
    values = np.asarray(values, np.float64)
    valid = np.isfinite(axis) & np.isfinite(values)
    if valid.sum() < 2:
        return np.zeros(len(axis), np.float64)
    return np.interp(axis, axis[valid], values[valid])


def _normalise_patch(values):
    centre = np.mean(values, axis=-1, keepdims=True)
    scale = np.std(values, axis=-1, keepdims=True)
    return (values - centre) / np.maximum(scale, 0.15)


def _sample_rows(eval_rows, anchor):
    take = np.unique(np.round(np.linspace(0, len(eval_rows) - 1, min(NODES, len(eval_rows)))).astype(int))
    return np.unique(np.r_[anchor, eval_rows[take]]).astype(np.int64)


def _emission(md, horizontal_gr, type_tvt, type_gr, rows, candidate_tvt):
    scores = np.zeros(candidate_tvt.shape, np.float64)
    valid_scales = np.zeros(candidate_tvt.shape, np.float64)
    for md_half, tvt_half in SCALE_PAIRS:
        query_axis = md[rows, None] + md_half * PATCH_GRID[None, :]
        query = np.interp(query_axis.ravel(), md, horizontal_gr).reshape(query_axis.shape)
        query = _normalise_patch(query)
        reference_axis = candidate_tvt[:, :, None] + tvt_half * PATCH_GRID[None, None, :]
        reference = np.interp(reference_axis.ravel(), type_tvt, type_gr).reshape(reference_axis.shape)
        reference = _normalise_patch(reference)
        available = (reference_axis[:, :, 0] >= type_tvt[0]) & (reference_axis[:, :, -1] <= type_tvt[-1])
        ncc = np.mean(query[:, None, :] * reference, axis=-1)
        scores += np.where(available, ncc, 0.0)
        valid_scales += available
    scores /= np.maximum(valid_scales, 1.0)
    scores[valid_scales == 0] = -2.0
    return scores.astype(np.float32)


def _decode(emission, transition, zero_prior):
    states = len(OFFSETS)
    length = len(emission)
    transition_cost = transition * np.abs(OFFSETS[:, None] - OFFSETS[None, :])
    unary = emission.astype(np.float64) - zero_prior * np.square(OFFSETS)[None, :]
    score = np.full((length, states), -np.inf, np.float64)
    back = np.zeros((length, states), np.int16)
    zero = int(np.argmin(np.abs(OFFSETS)))
    score[0, zero] = unary[0, zero]
    for row in range(1, length):
        candidate = score[row - 1, :, None] - transition_cost
        back[row] = np.argmax(candidate, axis=0)
        score[row] = unary[row] + candidate[back[row], np.arange(states)]
    path = np.empty(length, np.int16)
    path[-1] = int(np.argmax(score[-1]))
    for row in range(length - 1, 0, -1):
        path[row - 1] = back[row, path[row]]
    return OFFSETS[path]


def _metric(frame, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - frame["truth"].to_numpy(np.float64)
    by_well = pd.DataFrame({"well": frame["well"], "e2": error * error}).groupby("well", sort=False)["e2"].mean()
    far = frame["distance"].to_numpy(np.float64) > 2000.0
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)), "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)), "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))), "median_well_rmse": float(np.median(np.sqrt(by_well))),
        "p95_well_rmse": float(np.quantile(np.sqrt(by_well), 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


started = time.time()
frame, base_source, base_column = _read_base()
if len(frame) != EXPECTED_ROWS or frame["well"].nunique() != EXPECTED_WELLS:
    raise RuntimeError(f"Expected {EXPECTED_ROWS}/{EXPECTED_WELLS}, found {len(frame)}/{frame['well'].nunique()}")
base = frame["base"].to_numpy(np.float64)
truth = frame["truth"].to_numpy(np.float64)
fold_array = frame["fold"].to_numpy(np.int8)
records = {}
axis_oracle = base + np.clip(truth - base, OFFSETS[0], OFFSETS[-1])

for number, (well, local) in enumerate(frame.groupby("well", sort=True), 1):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv", usecols=["MD", "GR", "TVT", "TVT_input"])
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv", usecols=["TVT", "GR"]).dropna()
    md = horizontal["MD"].to_numpy(np.float64)
    hgr = _fill(md, horizontal["GR"].to_numpy(np.float64))
    ttvt = typewell["TVT"].to_numpy(np.float64)
    tgr = typewell["GR"].to_numpy(np.float64)
    order = np.argsort(ttvt, kind="stable")
    ttvt, tgr = ttvt[order], tgr[order]
    unique, index = np.unique(ttvt, return_index=True)
    ttvt, tgr = unique, tgr[index]
    visible = np.flatnonzero(np.isfinite(horizontal["TVT_input"].to_numpy(np.float64)))
    anchor = int(visible[-1])
    eval_rows = local["row_idx"].to_numpy(np.int64)
    official = np.flatnonzero(~np.isfinite(horizontal["TVT_input"].to_numpy(np.float64)) & np.isfinite(horizontal["TVT"].to_numpy(np.float64)))
    if not np.array_equal(eval_rows, official):
        raise RuntimeError(f"{well}: evaluation mask mismatch")
    if np.max(np.abs(horizontal["TVT"].to_numpy(np.float64)[eval_rows] - local["truth"].to_numpy(np.float64))) > 0.01:
        raise RuntimeError(f"{well}: truth parity failed")
    nodes = _sample_rows(eval_rows, anchor)
    node_base = np.r_[float(horizontal["TVT_input"].iloc[anchor]), np.interp(nodes[1:], eval_rows, local["base"].to_numpy(np.float64))]
    candidate = node_base[:, None] + OFFSETS[None, :]
    emission = _emission(md, hgr, ttvt, tgr, nodes, candidate)
    records[well] = {
        "positions": local.index.to_numpy(np.int64), "eval_rows": eval_rows, "nodes": nodes,
        "emission": emission, "fold": int(local["fold"].iloc[0]),
    }
    if number % 50 == 0 or number == EXPECTED_WELLS:
        print(f"V178 emissions {number}/{EXPECTED_WELLS}", flush=True)

predictions = {name: np.full(len(frame), np.nan, np.float64) for name in RECIPES}
rolled_predictions = {name: np.full(len(frame), np.nan, np.float64) for name in RECIPES}
for number, (well, record) in enumerate(records.items(), 1):
    positions, eval_rows, nodes = record["positions"], record["eval_rows"], record["nodes"]
    emission = record["emission"]
    shift = max(1, int(round((len(emission) - 1) * ROLL_FRACTION)))
    rolled = emission.copy()
    rolled[1:] = np.roll(rolled[1:], shift, axis=0)
    for name, (transition, zero_prior) in RECIPES.items():
        correction = _decode(emission, transition, zero_prior)
        rolled_correction = _decode(rolled, transition, zero_prior)
        predictions[name][positions] = base[positions] + np.interp(eval_rows, nodes, correction)
        rolled_predictions[name][positions] = base[positions] + np.interp(eval_rows, nodes, rolled_correction)
    if number % 100 == 0 or number == EXPECTED_WELLS:
        print(f"V178 decode {number}/{EXPECTED_WELLS}", flush=True)

nested_candidate = np.full(len(frame), np.nan, np.float64)
nested_blend = np.full(len(frame), np.nan, np.float64)
nested_rolled = np.full(len(frame), np.nan, np.float64)
selection_rows = []
for held_fold in range(1, 6):
    train = fold_array != held_fold
    valid = ~train
    scores = {name: float(np.sqrt(np.mean((pred[train] - truth[train]) ** 2))) for name, pred in predictions.items()}
    selected = min(scores, key=scores.get)
    move = predictions[selected][train] - base[train]
    alpha = float(np.dot(truth[train] - base[train], move) / max(np.dot(move, move), 1e-12))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    nested_candidate[valid] = predictions[selected][valid]
    nested_blend[valid] = base[valid] + alpha * (predictions[selected][valid] - base[valid])
    nested_rolled[valid] = base[valid] + alpha * (rolled_predictions[selected][valid] - base[valid])
    selection_rows.append({"heldout_fold": held_fold, "selected_recipe": selected, "selected_alpha": alpha, "train_rmse": scores[selected]})

summary = pd.DataFrame([
    _metric(frame, base, "base_current"),
    *[_metric(frame, pred, name) for name, pred in predictions.items()],
    _metric(frame, nested_candidate, "nested_candidate"),
    _metric(frame, nested_blend, "nested_blend"),
    _metric(frame, nested_rolled, "rolled_emission_control"),
    _metric(frame, axis_oracle, "residual_axis_oracle"),
])
fold_rows = []
for fold in range(1, 6):
    mask = fold_array == fold
    local = frame.loc[mask].reset_index(drop=True)
    for endpoint, pred in (("base_current", base[mask]), ("nested_blend", nested_blend[mask]), ("rolled_emission_control", nested_rolled[mask])):
        fold_rows.append({"fold": fold, **_metric(local, pred, endpoint)})
fold_summary = pd.DataFrame(fold_rows)

summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
fold_summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
pd.DataFrame(selection_rows).to_csv(OUT_DIR / "nested_selections.csv", index=False)
output = frame[["well", "fold", "row_idx", "truth", "distance"]].copy()
output["base_current"] = base.astype(np.float32)
output["nested_candidate"] = nested_candidate.astype(np.float32)
output["nested_blend"] = nested_blend.astype(np.float32)
output.to_csv(OUT_DIR / "oof.csv.gz", index=False, compression="gzip")

lookup = summary.set_index("endpoint")
base_rmse = float(lookup.loc["base_current", "rmse"])
candidate_rmse = float(lookup.loc["nested_candidate", "rmse"])
blend_rmse = float(lookup.loc["nested_blend", "rmse"])
rolled_rmse = float(lookup.loc["rolled_emission_control", "rmse"])
axis_rmse = float(lookup.loc["residual_axis_oracle", "rmse"])
pivot = fold_summary.pivot(index="fold", columns="endpoint", values="rmse")
fold_gains = (pivot["base_current"] - pivot["nested_blend"]).to_dict()
verdict = {
    "experiment": EXPERIMENT,
    "stage": "complete_fivefold_residual_centered_native_correlation_crf",
    "base_source": base_source, "base_column": base_column,
    "legal_inputs": ["base prediction", "horizontal MD/GR", "typewell TVT/GR", "visible TVT_input anchor"],
    "state_axis": {"minimum": float(OFFSETS[0]), "maximum": float(OFFSETS[-1]), "step": 0.5},
    "coverage": {"rows": len(frame), "wells": frame["well"].nunique()},
    "nested_selections": selection_rows,
    "values": {
        "base_rmse": base_rmse, "candidate_rmse": candidate_rmse, "nested_blend_rmse": blend_rmse,
        "nested_gain": base_rmse - blend_rmse, "aligned_increment_vs_rolled": rolled_rmse - blend_rmse,
        "residual_axis_oracle_rmse": axis_rmse, "fold_gains": fold_gains,
    },
    "checks": {
        "residual_axis_oracle_rmse_lt_6": axis_rmse < 6.0,
        "nested_gain_ge_0p20": base_rmse - blend_rmse >= 0.20,
        "positive_folds_5of5": all(v > 0 for v in fold_gains.values()),
        "minimum_fold_gain_nonnegative": min(fold_gains.values()) >= 0,
        "aligned_beats_rolled_ge_0p10": rolled_rmse - blend_rmse >= 0.10,
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["checks"].values()))
verdict["next"] = "PASS -> confirm frozen residual CRF on current V144 lineage" if verdict["strict_pass"] else "FAIL -> native NCC cannot identify residual state; stop CRF/PF correction family"
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nV178 MATCHED SUMMARY")
print(summary.sort_values("rmse").to_string(index=False), flush=True)
print("\nV178 FOLD SUMMARY")
print(fold_summary.to_string(index=False), flush=True)
print("\nV178 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
