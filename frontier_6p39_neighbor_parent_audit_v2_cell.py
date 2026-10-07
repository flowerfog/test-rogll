"""Audit the original neighbor-transfer bank on two reconstructed 6.39 parents.

Run after ``frontier_6p39_existing_models_oof_v0_cell.py``. Fold 3 selects a
recipe independently for each parent, Fold 2 confirms it once, and Fold 1 is
untouched. The absolute neighbor curves are reused without regeneration.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
SOURCE = (
    OOF_ROOT
    / "frontier_6p39_existing_models_oof_v0"
    / "frontier_6p39_core_oof.parquet"
)
POOL_CANDIDATES = [
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
]
NEIGHBOR_DIR = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
NEIGHBOR_BANK = NEIGHBOR_DIR / "neighbor_transfer_oof.npz"
NEIGHBOR_META = NEIGHBOR_DIR / "selected_neighbors.csv"
OUT_DIR = OOF_ROOT / "frontier_6p39_neighbor_parent_audit_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
UNTOUCHED_FOLD = 1
MIN_DIRECTION_DOT = 0.70
METHODS = ("structure_xy", "tvt_xy", "structure_progress", "tvt_progress")
GATES = {
    "coherent150": dict(min=np.inf, median=150.0, p90=300.0, anchor=300.0),
    "coherent300": dict(min=np.inf, median=300.0, p90=600.0, anchor=600.0),
    "coherent600": dict(min=np.inf, median=600.0, p90=1200.0, anchor=1200.0),
    "touch150": dict(min=150.0, median=600.0, p90=1200.0, anchor=1200.0),
}
WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.0)
CAPS = (3.0, 6.0, 9.0, 12.0, 18.0, 24.0, np.inf)


def first_file(paths: list[Path], label: str) -> Path:
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing {label}; inspected={[str(p) for p in paths]}")
    return path


def metrics(local: pd.DataFrame, prediction: np.ndarray) -> dict:
    truth = local["truth"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    bywell = (
        pd.DataFrame({"well": local["well"].to_numpy(), "error2": error * error})
        .groupby("well")["error2"]
        .mean()
        .pow(0.5)
    )
    result = {
        "rows": int(len(local)),
        "wells": int(local["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
    }
    distance = local["md_since"].to_numpy(np.float64)
    for name, mask in {
        "near_0_500_rmse": distance <= 500.0,
        "far_gt2000_rmse": distance > 2000.0,
    }.items():
        result[name] = (
            float(np.sqrt(np.mean(error[mask] * error[mask]))) if mask.any() else None
        )
    return result


for required in (SOURCE, NEIGHBOR_BANK, NEIGHBOR_META):
    if not required.is_file():
        raise FileNotFoundError(required)
pool_path = first_file(POOL_CANDIDATES, "candidate pool used by neighbor bank")

source = pd.read_parquet(SOURCE)
required_source = {"id", "well", "row_idx", "fold", "truth", "sp45", "fle", "frontier_6p39_core"}
missing = sorted(required_source.difference(source.columns))
if missing:
    raise RuntimeError(f"6.39 OOF missing {missing}")
source = source.sort_values(["well", "row_idx"]).reset_index(drop=True)
source["id"] = source["id"].astype(str)
source["well"] = source["well"].astype(str)
if source["id"].duplicated().any():
    raise RuntimeError("Duplicate 6.39 OOF ids")

pool_header = pd.read_csv(pool_path, nrows=0).columns.tolist()
pool_columns = [column for column in ("id", "well", "row_idx") if column in pool_header]
if "id" not in pool_columns:
    raise RuntimeError(f"Candidate pool has no id column: {pool_header}")
pool = pd.read_csv(pool_path, usecols=pool_columns, dtype={"id": "string", "well": "string"})
pool["id"] = pool["id"].astype(str)
if "well" not in pool:
    pool["well"] = pool["id"].str.rsplit("_", n=1).str[0]
if "row_idx" not in pool:
    pool["row_idx"] = pool["id"].str.rsplit("_", n=1).str[1].astype(np.int32)
pool["well"] = pool["well"].astype(str)
pool = pool.sort_values(["well", "row_idx"]).reset_index(drop=True)
if pool["id"].duplicated().any():
    raise RuntimeError("Duplicate candidate-pool ids")

with np.load(NEIGHBOR_BANK, allow_pickle=False) as handle:
    bank = {key: handle[key].astype(np.float64) for key in handle.files}
for method in METHODS:
    if method not in bank or len(bank[method]) != len(pool):
        raise RuntimeError(f"Malformed neighbor bank method {method}")

source_position = pd.Series(np.arange(len(pool), dtype=np.int64), index=pool["id"])
bank_position = source["id"].map(source_position)
if bank_position.isna().any():
    raise RuntimeError(f"Neighbor bank misses {int(bank_position.isna().sum())} OOF rows")
bank_position = bank_position.to_numpy(np.int64)
for method in METHODS:
    source[method] = bank[method][bank_position]

# Scored rows are native one-foot rows. This is used only for distance-stratum
# reporting and the near/far selection guard, never as a model feature.
source["md_since"] = source.groupby("well", sort=False).cumcount().to_numpy(np.float64) + 1.0

meta = pd.read_csv(NEIGHBOR_META)
meta["well"] = meta["well"].astype(str)
needed_meta = {
    "well", "min_distance", "median_distance", "p90_distance",
    "anchor_distance_full", "abs_direction_dot",
}
if not needed_meta.issubset(meta.columns):
    raise RuntimeError(f"Neighbor metadata missing {sorted(needed_meta - set(meta.columns))}")
meta = meta.drop_duplicates("well").set_index("well")
for column in needed_meta - {"well"}:
    source[column] = source["well"].map(meta[column]).astype(float)

gate_masks = {}
for name, limits in GATES.items():
    gate_masks[name] = (
        (source["min_distance"].to_numpy() <= limits["min"])
        & (source["median_distance"].to_numpy() <= limits["median"])
        & (source["p90_distance"].to_numpy() <= limits["p90"])
        & (source["anchor_distance_full"].to_numpy() <= limits["anchor"])
        & (source["abs_direction_dot"].to_numpy() >= MIN_DIRECTION_DOT)
    )

truth = source["truth"].to_numpy(np.float64)
fold = source["fold"].to_numpy(np.int8)
distance = source["md_since"].to_numpy(np.float64)
parents = {
    "fle": source["fle"].to_numpy(np.float64),
    "published_6p39_core": source["frontier_6p39_core"].to_numpy(np.float64),
}
selection_mask = fold == SELECTION_FOLD
confirmation_mask = fold == CONFIRMATION_FOLD
if not selection_mask.any() or not confirmation_mask.any():
    raise RuntimeError("Selection or confirmation fold is empty")


def recipe_prediction(parent: np.ndarray, recipe: dict) -> np.ndarray:
    output = parent.copy()
    weight = float(recipe["weight"])
    if weight == 0.0:
        return output
    candidate = source[str(recipe["method"])].to_numpy(np.float64)
    eligible = gate_masks[str(recipe["gate"])] & np.isfinite(candidate)
    move = candidate[eligible] - parent[eligible]
    cap = float(recipe["cap_ft"])
    if np.isfinite(cap):
        move = np.clip(move, -cap, cap)
    output[eligible] += weight * move
    return output


all_scan_rows = []
parent_verdicts = {}
export = source[["id", "well", "row_idx", "fold", "truth", "md_since"]].copy()

for parent_name, parent in parents.items():
    selection_error = parent[selection_mask] - truth[selection_mask]
    selection_near = distance[selection_mask] <= 500.0
    selection_far = distance[selection_mask] > 2000.0
    base_sse = float(np.dot(selection_error, selection_error))
    base_near_sse = float(np.dot(selection_error[selection_near], selection_error[selection_near]))
    base_far_sse = float(np.dot(selection_error[selection_far], selection_error[selection_far]))
    n_selection = int(selection_mask.sum())
    n_near = int(selection_near.sum())
    n_far = int(selection_far.sum())
    base_rmse = float(np.sqrt(base_sse / n_selection))
    base_near = float(np.sqrt(base_near_sse / n_near))
    base_far = float(np.sqrt(base_far_sse / n_far))

    local_rows = []
    for method in METHODS:
        candidate = source[method].to_numpy(np.float64)
        for gate in GATES:
            eligible_full = gate_masks[gate] & np.isfinite(candidate)
            eligible = eligible_full[selection_mask]
            raw_delta = candidate[selection_mask] - parent[selection_mask]
            for cap in CAPS:
                delta = np.zeros(n_selection, dtype=np.float64)
                selected_delta = raw_delta[eligible]
                if np.isfinite(cap):
                    selected_delta = np.clip(selected_delta, -cap, cap)
                delta[eligible] = selected_delta
                cross = float(np.dot(selection_error, delta))
                delta2 = float(np.dot(delta, delta))
                near_cross = float(np.dot(selection_error[selection_near], delta[selection_near]))
                near_delta2 = float(np.dot(delta[selection_near], delta[selection_near]))
                far_cross = float(np.dot(selection_error[selection_far], delta[selection_far]))
                far_delta2 = float(np.dot(delta[selection_far], delta[selection_far]))
                for weight in WEIGHTS:
                    sse = base_sse + 2.0 * weight * cross + weight * weight * delta2
                    near_sse = base_near_sse + 2.0 * weight * near_cross + weight * weight * near_delta2
                    far_sse = base_far_sse + 2.0 * weight * far_cross + weight * weight * far_delta2
                    rmse = float(np.sqrt(max(sse, 0.0) / n_selection))
                    near_rmse = float(np.sqrt(max(near_sse, 0.0) / n_near))
                    far_rmse = float(np.sqrt(max(far_sse, 0.0) / n_far))
                    local_rows.append({
                        "parent": parent_name,
                        "method": method,
                        "gate": gate,
                        "weight": float(weight),
                        "cap_ft": "full" if not np.isfinite(cap) else float(cap),
                        "cap_order": 1e9 if not np.isfinite(cap) else float(cap),
                        "eligible_wells": int(source.loc[eligible_full & selection_mask, "well"].nunique()),
                        "eligible_rows": int((eligible_full & selection_mask).sum()),
                        "rmse": rmse,
                        "gain": base_rmse - rmse,
                        "near_0_500_rmse": near_rmse,
                        "far_gt2000_rmse": far_rmse,
                        "near_degradation": near_rmse - base_near,
                        "far_gain": base_far - far_rmse,
                    })

    local_scan = pd.DataFrame(local_rows)
    local_scan["selection_score"] = (
        local_scan["rmse"]
        + 0.25 * np.maximum(local_scan["near_degradation"], 0.0)
        - 0.05 * np.maximum(local_scan["far_gain"], 0.0)
    )
    local_scan = local_scan.sort_values(
        ["selection_score", "rmse", "weight", "cap_order"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    all_scan_rows.append(local_scan)
    selected = local_scan.iloc[0]
    recipe = {
        "method": str(selected["method"]),
        "gate": str(selected["gate"]),
        "weight": float(selected["weight"]),
        "cap_ft": np.inf if selected["cap_ft"] == "full" else float(selected["cap_ft"]),
    }
    prediction = recipe_prediction(parent, recipe)
    export[f"{parent_name}_parent"] = parent
    export[f"{parent_name}_neighbor"] = prediction
    export[f"{parent_name}_neighbor_move"] = prediction - parent

    selection_base = metrics(source.loc[selection_mask], parent[selection_mask])
    selection_result = metrics(source.loc[selection_mask], prediction[selection_mask])
    confirmation_base = metrics(source.loc[confirmation_mask], parent[confirmation_mask])
    confirmation_result = metrics(source.loc[confirmation_mask], prediction[confirmation_mask])
    selection_gain = selection_base["rmse"] - selection_result["rmse"]
    confirmation_gain = confirmation_base["rmse"] - confirmation_result["rmse"]
    confirmation_far_gain = confirmation_base["far_gt2000_rmse"] - confirmation_result["far_gt2000_rmse"]
    strict_pass = bool(
        recipe["weight"] > 0.0
        and selection_gain >= 0.05
        and confirmation_gain >= 0.05
        and confirmation_far_gain >= 0.0
        and confirmation_result["near_0_500_rmse"] <= confirmation_base["near_0_500_rmse"] + 0.05
    )
    parent_verdicts[parent_name] = {
        "selected_recipe": {
            **recipe,
            "cap_ft": None if not np.isfinite(recipe["cap_ft"]) else recipe["cap_ft"],
        },
        "selection_base": selection_base,
        "selection": selection_result,
        "selection_gain": float(selection_gain),
        "confirmation_base": confirmation_base,
        "confirmation": confirmation_result,
        "confirmation_gain": float(confirmation_gain),
        "confirmation_far_gain": float(confirmation_far_gain),
        "strict_pass": strict_pass,
    }

scan = pd.concat(all_scan_rows, ignore_index=True)
scan.to_csv(OUT_DIR / "fold3_neighbor_recipe_scan.csv", index=False)
export.to_parquet(OUT_DIR / "neighbor_parent_oof.parquet", index=False, compression="zstd")

verdict = {
    "experiment": "frontier_6p39_neighbor_parent_audit_v2",
    "stage": "Fold3_selection_to_Fold2_confirmation_on_FLE_and_published_parents",
    "source": str(SOURCE),
    "neighbor_bank": str(NEIGHBOR_BANK),
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "confirmation_fold": CONFIRMATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
    },
    "distance_note": "md_since is scored-row ordinal; used only for diagnostics and near/far guard",
    "parents": parent_verdicts,
    "any_strict_pass": bool(any(item["strict_pass"] for item in parent_verdicts.values())),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
    "next": (
        "PASS -> freeze the confirmed parent/Neighbor recipe and run CP residual training on that parent"
        if any(item["strict_pass"] for item in parent_verdicts.values())
        else "FAIL -> Neighbor gain does not migrate; run CP residual training independently on FLE"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("NEIGHBOR TOP RECIPES")
print(
    scan.groupby("parent", group_keys=False).head(10).to_string(index=False),
    flush=True,
)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
