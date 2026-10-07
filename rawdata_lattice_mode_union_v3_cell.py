"""Expand and unite label-free raw-data lattice modes before learning a scorer.

V2 showed that endpoint-diverse beams preserve substantially better paths, but
the Fold-2 retained-bank oracle was still above the 5.5-ft target.  This cell
prices the next search investment without training a ranker.  Fold 4/5 fit the
transition prior, Fold 3 selects a small union of search arms, Fold 2 confirms
the frozen union, and Fold 1 remains untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_v2():
    candidates = [
        Path.cwd() / "rawdata_lattice_beam_survival_v2_cell.py",
        Path("/root/autodl-tmp/rawdata_lattice_beam_survival_v2_cell.py"),
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "Upload rawdata_lattice_beam_survival_v2_cell.py to /root/autodl-tmp"
        )
    text = source.read_text(encoding="utf-8")
    marker = "# --- V2 EXECUTION ---"
    if marker not in text:
        raise RuntimeError(f"bootstrap marker missing from {source}")
    namespace = {"__name__": "raw_lattice_v2_definitions"}
    exec(compile(text.split(marker, 1)[0], str(source), "exec"), namespace)
    return namespace, source


B, V2_SOURCE = _bootstrap_v2()
V = B["V"]
DATA_ROOT = B["DATA_ROOT"]
EXPERIMENT = "rawdata_lattice_mode_union_v3"
if os.name != "nt" and Path("/root/autodl-tmp").exists():
    OUT_DIR = Path("/root/autodl-tmp/oof_exports") / EXPERIMENT
else:
    OUT_DIR = Path.cwd() / "output" / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_LENGTH_FT = 128.0
PERSISTENCE_WEIGHT = 0.1
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
DISCOVERY_WELLS = int(os.environ.get("RAW_V3_DISCOVERY_WELLS", "24"))
MAX_UNION_ARMS = 3
MIN_INCREMENTAL_RMSE_GAIN = 0.05

# The arms differ only in label-free search mechanics.  No truth, historical
# prediction, neighbour, CP12, Gold, or spatial feature enters a candidate.
ARMS = [
    {
        "id": "raw_g_b512_e2",
        "beam": 512,
        "diversity_ft": 2.0,
        "reference": "raw",
        "likelihood": "gaussian",
        "history_diversity": False,
    },
    {
        "id": "raw_g_b512_e4",
        "beam": 512,
        "diversity_ft": 4.0,
        "reference": "raw",
        "likelihood": "gaussian",
        "history_diversity": False,
    },
    {
        "id": "raw_g_b1024_e2",
        "beam": 1024,
        "diversity_ft": 2.0,
        "reference": "raw",
        "likelihood": "gaussian",
        "history_diversity": False,
    },
    {
        "id": "raw_g_b512_shape4",
        "beam": 512,
        "diversity_ft": 4.0,
        "reference": "raw",
        "likelihood": "gaussian",
        "history_diversity": True,
    },
    {
        "id": "raw_clip_b512_shape4",
        "beam": 512,
        "diversity_ft": 4.0,
        "reference": "raw",
        "likelihood": "clipped_gaussian",
        "history_diversity": True,
    },
    {
        "id": "shift_g_b512_shape4",
        "beam": 512,
        "diversity_ft": 4.0,
        "reference": "shift",
        "likelihood": "gaussian",
        "history_diversity": True,
    },
]


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def _select_history_diverse(
    score,
    end_surface,
    inherited_history,
    beam_size,
    diversity_ft,
    update_history,
):
    """Retain paths diverse in endpoint and an earlier surface checkpoint."""
    flat_score = score.ravel()
    flat_surface = end_surface.ravel()
    candidate_history = np.repeat(inherited_history, score.shape[1])
    if update_history:
        next_history = flat_surface.copy()
    else:
        next_history = candidate_history.copy()

    order = np.argsort(flat_score)
    selected = []
    selected_set = set()
    bin_count = {}
    for index in order:
        endpoint_bin = int(np.floor(flat_surface[index] / diversity_ft))
        history_bin = int(np.floor(candidate_history[index] / diversity_ft))
        key = (endpoint_bin, history_bin)
        if bin_count.get(key, 0) >= 1:
            continue
        selected.append(int(index))
        selected_set.add(int(index))
        bin_count[key] = 1
        if len(selected) >= beam_size:
            break
    if len(selected) < beam_size:
        for index in order:
            index = int(index)
            if index not in selected_set:
                selected.append(index)
                if len(selected) >= beam_size:
                    break
    keep = np.asarray(selected, dtype=np.int64)
    return keep, next_history[keep]


def _beam_bank_v3(record, prior, arm):
    rates = prior["rate_grid"]
    segments = V["_segments"](record, SEGMENT_LENGTH_FT)
    reference_gr = B["_reference_for_record"](record, arm["reference"])
    scores = np.array([0.0], dtype=np.float64)
    surfaces = np.array([record["anchor_surface"]], dtype=np.float64)
    history_surface = surfaces.copy()
    previous_rate = np.array([-1], dtype=np.int64)
    paths = np.empty((1, 0), dtype=np.int16)
    previous_md = record["anchor_md"]

    for segment_index, rows in enumerate(segments):
        loss, end_surface = B["_loss_and_surface"](
            record,
            rows,
            previous_md,
            surfaces,
            rates,
            reference_gr,
            arm["likelihood"],
        )
        candidate_score = scores[:, None] + loss
        previous = np.where(
            previous_rate[:, None] >= 0,
            rates[np.maximum(previous_rate[:, None], 0)],
            prior["rate_centre"],
        )
        transition = (
            (rates[None, :] - previous - prior["change_centre"])
            / prior["change_scale"]
        ) ** 2
        candidate_score += PERSISTENCE_WEIGHT * transition

        if arm["history_diversity"]:
            keep, next_history = _select_history_diverse(
                candidate_score,
                end_surface,
                history_surface,
                int(arm["beam"]),
                float(arm["diversity_ft"]),
                update_history=((segment_index + 1) % 4 == 0),
            )
        else:
            keep = B["_select_diverse"](
                candidate_score,
                end_surface,
                int(arm["beam"]),
                float(arm["diversity_ft"]),
            )
            candidate_history = np.repeat(history_surface, len(rates))
            next_history = candidate_history[keep]

        parent = keep // len(rates)
        rate_index = keep % len(rates)
        scores = candidate_score.ravel()[keep]
        surfaces = end_surface.ravel()[keep]
        history_surface = next_history
        previous_rate = rate_index.astype(np.int64)
        paths = np.concatenate(
            [paths[parent], rate_index[:, None].astype(np.int16)], axis=1
        )
        previous_md = float(record["md"][rows][-1])
    return scores, paths, segments


def _evaluate_arm(records, prior, arm, stage):
    output = []
    for number, record in enumerate(records, start=1):
        scores, paths, segments = _beam_bank_v3(record, prior, arm)
        eval_rows = record["eval_rows"]
        truth = record["truth"][eval_rows]
        distance = record["md"][eval_rows] - record["anchor_md"]
        far = distance > 2000.0
        best_mse = np.inf
        best_sse = np.inf
        best_sae = np.inf
        best_far_sse = np.inf
        map_sse = map_sae = map_far_sse = np.nan
        endpoint_bins = set()

        for path_index, path in enumerate(paths):
            prediction = B["_path_prediction"](record, prior, path, segments)[eval_rows]
            error = prediction - truth
            sse = float(np.sum(error**2))
            if path_index == 0:
                map_sse = sse
                map_sae = float(np.sum(np.abs(error)))
                map_far_sse = float(np.sum(error[far] ** 2))
            if sse < best_sse:
                best_sse = sse
                best_mse = sse / len(error)
                best_sae = float(np.sum(np.abs(error)))
                best_far_sse = float(np.sum(error[far] ** 2))
            endpoint_bins.add(int(np.floor(prediction[-1] / 4.0)))

        common = {
            "arm": arm["id"],
            "stage": stage,
            "well": record["well"],
            "rows": len(truth),
            "far_rows": int(far.sum()),
            "bank_paths": int(len(paths)),
            "unique_endpoint_bins_4ft": int(len(endpoint_bins)),
            "oracle_mse": float(best_mse),
        }
        output.append(
            common
            | {
                "variant": "map",
                "sse": map_sse,
                "sae": map_sae,
                "far_sse": map_far_sse,
            }
        )
        output.append(
            common
            | {
                "variant": "bank_oracle",
                "sse": best_sse,
                "sae": best_sae,
                "far_sse": best_far_sse,
            }
        )
        if number % 12 == 0 or number == len(records):
            print(f"{stage} {arm['id']} {number}/{len(records)}", flush=True)
    return pd.DataFrame(output)


def _summary(frame):
    rows = []
    for (arm, variant), group in frame.groupby(["arm", "variant"]):
        count = int(group["rows"].sum())
        sse = float(group["sse"].sum())
        far_rows = int(group["far_rows"].sum())
        far_sse = float(group["far_sse"].sum())
        well_rmse = np.sqrt(group["sse"] / group["rows"])
        rows.append(
            {
                "arm": arm,
                "variant": variant,
                "rows": count,
                "wells": int(group["well"].nunique()),
                "exact_sse": sse,
                "rmse": math.sqrt(sse / count),
                "mae": float(group["sae"].sum() / count),
                "median_well_rmse": float(np.median(well_rmse)),
                "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
                "far_gt2000_rmse": math.sqrt(far_sse / far_rows) if far_rows else np.nan,
                "mean_bank_paths": float(group["bank_paths"].mean()),
                "mean_unique_endpoint_bins_4ft": float(
                    group["unique_endpoint_bins_4ft"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _union_rows(oracle_frame, selected_arms, label):
    subset = oracle_frame[
        oracle_frame["variant"].eq("bank_oracle")
        & oracle_frame["arm"].isin(selected_arms)
    ].copy()
    chosen = subset.loc[subset.groupby("well")["sse"].idxmin()].copy()
    chosen["arm_source"] = chosen["arm"]
    chosen["arm"] = label
    chosen["variant"] = "union_oracle"
    return chosen


def _greedy_union(discovery_frame):
    oracle = discovery_frame[discovery_frame["variant"].eq("bank_oracle")]
    remaining = sorted(oracle["arm"].unique())
    selected = []
    history = []
    previous_rmse = np.inf
    total_rows = int(oracle.drop_duplicates("well")["rows"].sum())
    while remaining and len(selected) < MAX_UNION_ARMS:
        candidates = []
        for arm in remaining:
            trial = selected + [arm]
            union = _union_rows(oracle, trial, "+".join(trial))
            rmse = math.sqrt(float(union["sse"].sum()) / total_rows)
            candidates.append((rmse, arm))
        rmse, arm = min(candidates)
        gain = previous_rmse - rmse if np.isfinite(previous_rmse) else np.nan
        if selected and gain < MIN_INCREMENTAL_RMSE_GAIN:
            break
        selected.append(arm)
        remaining.remove(arm)
        history.append(
            {
                "step": len(selected),
                "added_arm": arm,
                "union": "+".join(selected),
                "rmse": rmse,
                "gain_from_previous": gain,
            }
        )
        previous_rmse = rmse
    return selected, pd.DataFrame(history)


def _union_summary(frame, selected_arms, label):
    union = _union_rows(frame, selected_arms, label)
    count = int(union["rows"].sum())
    sse = float(union["sse"].sum())
    far_rows = int(union["far_rows"].sum())
    far_sse = float(union["far_sse"].sum())
    well_rmse = np.sqrt(union["sse"] / union["rows"])
    return union, {
        "union": label,
        "arms": list(selected_arms),
        "rows": count,
        "wells": int(union["well"].nunique()),
        "exact_sse": sse,
        "rmse": math.sqrt(sse / count),
        "mae": float(union["sae"].sum() / count),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
        "far_gt2000_rmse": math.sqrt(far_sse / far_rows) if far_rows else np.nan,
        "winning_arm_counts": union["arm_source"].value_counts().to_dict()
        if "arm_source" in union
        else {},
    }


# --- V3 EXECUTION ---
started = time.time()
wells = [
    path.name.split("__", 1)[0]
    for path in sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
]
folds = V["_stable_folds"](wells)
fold_by_well = dict(zip(folds["well"], folds["fold"]))
needed = [well for well in wells if fold_by_well[well] in (2, 3, 4, 5)]
records = {}
failures = []
for number, well in enumerate(needed, start=1):
    try:
        records[well] = V["_load_record"](well)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(needed):
        print(f"V3 load {number}/{len(needed)} failures={len(failures)}", flush=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)

prior_records = [
    records[w]
    for w in needed
    if w in records and fold_by_well[w] in PRIOR_FOLDS
]
selection_records = [
    records[w]
    for w in needed
    if w in records and fold_by_well[w] == SELECTION_FOLD
]
application_records = [
    records[w]
    for w in needed
    if w in records and fold_by_well[w] == APPLICATION_FOLD
]
prior = V["_fit_prior"](prior_records, SEGMENT_LENGTH_FT)

discovery = sorted(
    selection_records,
    key=lambda record: hashlib.sha256(
        f"raw-v3-discovery:{record['well']}".encode("ascii")
    ).digest(),
)[:DISCOVERY_WELLS]

discovery_frames = []
for arm in ARMS:
    discovery_frames.append(_evaluate_arm(discovery, prior, arm, "fold3_discovery"))
discovery_detail = pd.concat(discovery_frames, ignore_index=True)
discovery_summary = _summary(discovery_detail)
selected_arms, greedy_history = _greedy_union(discovery_detail)
discovery_union, discovery_union_metrics = _union_summary(
    discovery_detail, selected_arms, "+".join(selected_arms)
)
discovery_detail.to_csv(OUT_DIR / "fold3_discovery_bywell.csv", index=False)
discovery_summary.to_csv(OUT_DIR / "fold3_discovery_summary.csv", index=False)
greedy_history.to_csv(OUT_DIR / "fold3_greedy_union.csv", index=False)
print("\nV3 DISCOVERY SUMMARY")
print(discovery_summary.sort_values(["variant", "rmse"]).to_string(index=False), flush=True)
print("\nV3 GREEDY UNION")
print(greedy_history.to_string(index=False), flush=True)

arm_by_id = {arm["id"]: arm for arm in ARMS}
selection_frames = []
for arm_id in selected_arms:
    selection_frames.append(
        _evaluate_arm(selection_records, prior, arm_by_id[arm_id], "fold3_confirmation")
    )
selection_detail = pd.concat(selection_frames, ignore_index=True)
selection_summary = _summary(selection_detail)
selection_union, selection_union_metrics = _union_summary(
    selection_detail, selected_arms, "+".join(selected_arms)
)
selection_detail.to_csv(OUT_DIR / "fold3_confirmation_bywell.csv", index=False)
selection_summary.to_csv(OUT_DIR / "fold3_confirmation_summary.csv", index=False)
selection_union.to_csv(OUT_DIR / "fold3_union_oracle_bywell.csv", index=False)
print("\nV3 FOLD3 CONFIRMATION")
print(selection_summary.sort_values(["variant", "rmse"]).to_string(index=False), flush=True)
print(json.dumps(_clean(selection_union_metrics), indent=2), flush=True)

application_frames = []
for arm_id in selected_arms:
    application_frames.append(
        _evaluate_arm(application_records, prior, arm_by_id[arm_id], "fold2_application")
    )
application_detail = pd.concat(application_frames, ignore_index=True)
application_summary = _summary(application_detail)
application_union, application_union_metrics = _union_summary(
    application_detail, selected_arms, "+".join(selected_arms)
)
application_detail.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)
application_summary.to_csv(OUT_DIR / "fold2_summary.csv", index=False)
application_union.to_csv(OUT_DIR / "fold2_union_oracle_bywell.csv", index=False)
print("\nV3 FOLD2 APPLICATION")
print(application_summary.sort_values(["variant", "rmse"]).to_string(index=False), flush=True)
print(json.dumps(_clean(application_union_metrics), indent=2), flush=True)

application_oracles = application_summary[
    application_summary["variant"].eq("bank_oracle")
].sort_values("rmse")
best_single_oracle = application_oracles.iloc[0].to_dict()
union_gain = float(best_single_oracle["rmse"] - application_union_metrics["rmse"])
gates = {
    "minimum_union_oracle_rmse_le_5p5": application_union_metrics["rmse"] <= 5.5,
    "strong_union_oracle_rmse_le_4p8": application_union_metrics["rmse"] <= 4.8,
    "union_p95_le_9": application_union_metrics["p95_well_rmse"] <= 9.0,
    "union_far_gt2000_le_5p5": application_union_metrics["far_gt2000_rmse"] <= 5.5,
    "union_adds_gain_ge_0p05": union_gain >= 0.05,
}
if not gates["minimum_union_oracle_rmse_le_5p5"]:
    next_action = "FAIL -> search representation still lacks target coverage; do not train a scorer"
elif not gates["strong_union_oracle_rmse_le_4p8"]:
    next_action = "MARGINAL -> coverage reaches 5.5 but leaves almost no ranking budget; inspect tail modes"
else:
    next_action = "PASS -> freeze this bank and train a complete-well listwise scorer with matched controls"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_label_free_mode_union_oracle_gate",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": 1,
        "fold1_touched": False,
    },
    "arms_scanned": ARMS,
    "selected_arms": selected_arms,
    "discovery_union": discovery_union_metrics,
    "fold3_confirmation_union": selection_union_metrics,
    "fold2_best_single_oracle": best_single_oracle,
    "fold2_union_oracle": application_union_metrics,
    "fold2_union_gain_vs_best_single": union_gain,
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "next": next_action,
    "elapsed_minutes": (time.time() - started) / 60.0,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV3 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR)
