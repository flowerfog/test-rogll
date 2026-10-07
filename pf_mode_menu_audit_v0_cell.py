"""Step-0 PF mode-menu audit using the existing s16/p200 sidecars.

Primary menu: three trajectory-level KMeans modes from the PF multimodal input
experiment. Top-five likelihood-ranked seed paths are reported as a diagnostic.
All oracle gains are measured against CP12, not the weaker fleongg retrain.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
BASE_DIR = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = BASE_DIR / "features"
PF_EXP_DIR = OOF_ROOT / "pf_multimodal_input_ablation_v0_s16_p200"
PF_CACHE_DIR = PF_EXP_DIR / "pf_seed_features"
POOL_PATH = OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv"
OUT_DIR = OOF_ROOT / "pf_mode_menu_audit_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEEP_MD_SINCE = 1500.0
RECALL_TOLERANCES = (5.0, 8.0, 12.0)
MODE_WEIGHT_FLOOR = 0.01

for path in (FEATURE_DIR, PF_CACHE_DIR, POOL_PATH):
    if not path.exists():
        raise FileNotFoundError(path)


def rmse(y, prediction):
    y = np.asarray(y, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


def feature_paths(wid):
    return (
        FEATURE_DIR / f"{wid}__sample.pkl",
        FEATURE_DIR / f"{wid}__full.pkl",
    )


meta = pd.read_csv(BASE_DIR / "well_metadata.csv")
meta["well"] = meta["well"].astype(str)

print("loading CP12 base", flush=True)
header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
if "id" not in header or "cp12" not in header:
    raise RuntimeError("candidate pool must contain id and cp12")
pool = pd.read_csv(POOL_PATH, usecols=["id", "cp12"])
pool["id"] = pool["id"].astype(str)
if pool["id"].duplicated().any():
    raise RuntimeError("duplicate ids in candidate pool")


def load_well_menu(wid):
    _, full_path = feature_paths(wid)
    frame = pd.read_pickle(full_path)[
        ["id", "well", "_row_idx", "target", "last_known_tvt", "md_since"]
    ].copy()
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)

    cached = np.load(PF_CACHE_DIR / f"{wid}.npz", allow_pickle=False)
    eval_pos = cached["eval_pos"].astype(np.int32)
    values = cached["features"].astype(np.float32)
    names = cached["names"].astype(str).tolist()
    index = {name: i for i, name in enumerate(names)}

    row_idx = frame["_row_idx"].to_numpy(np.int32)
    locations = np.searchsorted(eval_pos, row_idx)
    if (
        np.any(locations >= len(eval_pos))
        or not np.array_equal(eval_pos[locations], row_idx)
    ):
        raise RuntimeError(f"PF sidecar row mismatch for {wid}")
    values = values[locations]
    last = frame["last_known_tvt"].to_numpy(np.float64)

    required = (
        [f"pfmm_mode{k}_d" for k in range(1, 4)]
        + [f"pfmm_mode{k}_weight" for k in range(1, 4)]
        + [f"pfmm_top{k}_d" for k in range(1, 6)]
    )
    missing = [name for name in required if name not in index]
    if missing:
        raise RuntimeError(f"PF sidecar missing features for {wid}: {missing}")

    mode_paths = np.column_stack(
        [last + values[:, index[f"pfmm_mode{k}_d"]] for k in range(1, 4)]
    )
    mode_weights = np.column_stack(
        [values[:, index[f"pfmm_mode{k}_weight"]] for k in range(1, 4)]
    )
    top_paths = np.column_stack(
        [last + values[:, index[f"pfmm_top{k}_d"]] for k in range(1, 6)]
    )
    truth = last + frame["target"].to_numpy(np.float64)

    output = frame[["id", "well", "_row_idx", "md_since"]].copy()
    output["tvt_true"] = truth
    for k in range(3):
        output[f"mode{k + 1}"] = mode_paths[:, k]
        output[f"mode{k + 1}_weight"] = mode_weights[:, k]
    for k in range(5):
        output[f"top{k + 1}"] = top_paths[:, k]
    return output


started = time.time()
parts = []
for wid in tqdm(meta["well"], desc="load PF mode menus"):
    parts.append(load_well_menu(str(wid)))
menu = pd.concat(parts, ignore_index=True)
menu = menu.merge(pool, on="id", how="left", validate="one_to_one")
if menu["cp12"].isna().any():
    raise RuntimeError(f"missing CP12 predictions: {menu['cp12'].isna().sum()}")
print(
    "menu rows:",
    {"rows": len(menu), "wells": menu["well"].nunique(), "elapsed_min": (time.time() - started) / 60},
)


def menu_arrays(frame, menu_name):
    if menu_name == "mode3":
        columns = ["mode1", "mode2", "mode3"]
    elif menu_name == "top5_seed":
        columns = [f"top{k}" for k in range(1, 6)]
    else:
        raise KeyError(menu_name)
    return frame[columns].to_numpy(np.float64), columns


def row_oracle(y, candidates):
    error = np.abs(candidates - y[:, None])
    choice = np.argmin(error, axis=1)
    return candidates[np.arange(len(y)), choice], np.min(error, axis=1), choice


def trajectory_oracle_prediction(frame, menu_name):
    prediction = np.empty(len(frame), dtype=np.float64)
    choices = []
    for wid, positions in frame.groupby("well", sort=False).indices.items():
        positions = np.asarray(positions, np.int64)
        local = frame.iloc[positions]
        y = local["tvt_true"].to_numpy(np.float64)
        candidates, columns = menu_arrays(local, menu_name)
        scores = np.sqrt(np.mean((candidates - y[:, None]) ** 2, axis=0))
        best = int(np.argmin(scores))
        prediction[positions] = candidates[:, best]
        choices.append(
            {
                "well": str(wid),
                "menu": menu_name,
                "chosen_slot": columns[best],
                "chosen_rmse": float(scores[best]),
                "second_rmse": float(np.partition(scores, 1)[1]) if len(scores) > 1 else float(scores[best]),
            }
        )
    return prediction, pd.DataFrame(choices)


def evaluate_subset(frame, subset_name, menu_name):
    y = frame["tvt_true"].to_numpy(np.float64)
    base = frame["cp12"].to_numpy(np.float64)
    candidates, _ = menu_arrays(frame, menu_name)
    oracle, min_abs_error, _ = row_oracle(y, candidates)
    trajectory_oracle, choices = trajectory_oracle_prediction(frame, menu_name)
    result = {
        "subset": subset_name,
        "menu": menu_name,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "cp12_rmse": rmse(y, base),
        "row_oracle_rmse": rmse(y, oracle),
        "row_oracle_gain": rmse(y, base) - rmse(y, oracle),
        "trajectory_oracle_rmse": rmse(y, trajectory_oracle),
        "trajectory_oracle_gain": rmse(y, base) - rmse(y, trajectory_oracle),
    }
    for tolerance in RECALL_TOLERANCES:
        result[f"recall_within_{int(tolerance)}"] = float(
            np.mean(min_abs_error <= tolerance)
        )
    return result, choices


# Top40 is fixed from full-eval CP12 SSE, then evaluated separately in the deep tail.
well_sse = (
    menu.assign(cp12_se=(menu["cp12"] - menu["tvt_true"]) ** 2)
    .groupby("well", as_index=False)
    .agg(rows=("id", "size"), cp12_sse=("cp12_se", "sum"))
    .sort_values("cp12_sse", ascending=False)
)
top40_wells = set(well_sse.head(40)["well"].astype(str))

deep = menu[menu["md_since"] > DEEP_MD_SINCE].copy()
deep_top40 = deep[deep["well"].isin(top40_wells)].copy()
if len(deep) == 0:
    raise RuntimeError("no rows satisfy the deep-tail threshold")

summary_rows = []
choice_parts = []
for subset_name, frame in (
    ("all_eval", menu),
    ("deep_gt1500", deep),
    ("deep_top40", deep_top40),
):
    for menu_name in ("mode3", "top5_seed"):
        result, choices = evaluate_subset(frame, subset_name, menu_name)
        summary_rows.append(result)
        choices["subset"] = subset_name
        choice_parts.append(choices)

summary = pd.DataFrame(summary_rows)
choices = pd.concat(choice_parts, ignore_index=True)

# Mode spacing is only meaningful when the second cluster has material weight.
mode2_valid = deep["mode2_weight"].to_numpy(float) >= MODE_WEIGHT_FLOOR
mode12_gap = np.abs(
    deep["mode1"].to_numpy(float) - deep["mode2"].to_numpy(float)
)
spacing = {
    "deep_rows": len(deep),
    "mode2_material_fraction": float(np.mean(mode2_valid)),
    "mode12_gap_median": float(np.median(mode12_gap[mode2_valid])) if mode2_valid.any() else 0.0,
    "mode12_gap_p25": float(np.quantile(mode12_gap[mode2_valid], 0.25)) if mode2_valid.any() else 0.0,
    "mode12_gap_p75": float(np.quantile(mode12_gap[mode2_valid], 0.75)) if mode2_valid.any() else 0.0,
}

primary = summary[
    (summary["subset"] == "deep_gt1500") & (summary["menu"] == "mode3")
].iloc[0]
primary_top40 = summary[
    (summary["subset"] == "deep_top40") & (summary["menu"] == "mode3")
].iloc[0]
verdict = {
    "primary_menu": "mode3",
    "deep_recall_within_5": float(primary["recall_within_5"]),
    "deep_row_oracle_gain_vs_cp12": float(primary["row_oracle_gain"]),
    "deep_trajectory_oracle_gain_vs_cp12": float(primary["trajectory_oracle_gain"]),
    "top40_row_oracle_gain_vs_cp12": float(primary_top40["row_oracle_gain"]),
}
verdict["strict_pass"] = bool(
    verdict["deep_recall_within_5"] >= 0.70
    and verdict["deep_row_oracle_gain_vs_cp12"] >= 1.50
    and verdict["deep_trajectory_oracle_gain_vs_cp12"] >= 0.50
    and verdict["top40_row_oracle_gain_vs_cp12"] >= 1.50
)
verdict["next"] = (
    "build symmetric per-mode evidence table"
    if verdict["strict_pass"]
    else "stop mode-selection route or expand the menu before selecting"
)

print("\nMENU SUMMARY")
display(summary)
print("\nMODE SPACING")
print(spacing)
print("\nSTEP-0 VERDICT")
print(verdict)

summary.to_csv(OUT_DIR / "menu_summary.csv", index=False)
choices.to_csv(OUT_DIR / "trajectory_oracle_choices.csv", index=False)
well_sse.to_csv(OUT_DIR / "cp12_well_sse.csv", index=False)
pd.DataFrame([spacing]).to_csv(OUT_DIR / "mode_spacing.csv", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
