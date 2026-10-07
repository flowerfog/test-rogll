"""Beam-survival audit for the raw-data segment lattice.

This is the next investment gate after rawdata_segment_lattice_nested_v1:
the lattice oracle is excellent, but MAP decoding has a catastrophic tail.
The audit asks whether good paths survive beam pruning. It compares matched
beam widths, endpoint-mode diversity, likelihoods, and prefix-only reference
calibration. No learned model or historical prediction artifact is used.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_v1():
    candidates = [
        Path.cwd() / "rawdata_segment_lattice_nested_v1_cell.py",
        Path("/root/autodl-tmp/rawdata_segment_lattice_nested_v1_cell.py"),
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "Upload rawdata_segment_lattice_nested_v1_cell.py to /root/autodl-tmp"
        )
    text = source.read_text(encoding="utf-8")
    marker = "started = time.time()"
    if marker not in text:
        raise RuntimeError(f"bootstrap marker missing from {source}")
    namespace = {"__name__": "raw_lattice_v1_definitions"}
    exec(compile(text.split(marker, 1)[0], str(source), "exec"), namespace)
    return namespace, source


V = None
V, V1_SOURCE = _bootstrap_v1()
EXPERIMENT = "rawdata_lattice_beam_survival_v2"
DATA_ROOT = V["DATA_ROOT"]
if os.name != "nt" and Path("/root/autodl-tmp").exists():
    OUT_DIR = Path("/root/autodl-tmp/oof_exports") / EXPERIMENT
else:
    OUT_DIR = Path.cwd() / "output" / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_LENGTH_FT = 128.0
PERSISTENCE_WEIGHT = 0.1
RATE_WEIGHT = 0.0
DISCOVERY_WELLS = 32
APPLICATION_FOLD = 2
SELECTION_FOLD = 3
PRIOR_FOLDS = (4, 5)

CONFIGS = [
    {"id": "raw_g_b40", "beam": 40, "diversity_ft": 0.0, "reference": "raw", "likelihood": "gaussian"},
    {"id": "raw_g_b128", "beam": 128, "diversity_ft": 0.0, "reference": "raw", "likelihood": "gaussian"},
    {"id": "raw_g_b256", "beam": 256, "diversity_ft": 0.0, "reference": "raw", "likelihood": "gaussian"},
    {"id": "raw_g_b128_d4", "beam": 128, "diversity_ft": 4.0, "reference": "raw", "likelihood": "gaussian"},
    {"id": "raw_g_b256_d4", "beam": 256, "diversity_ft": 4.0, "reference": "raw", "likelihood": "gaussian"},
    {"id": "raw_g_b256_d8", "beam": 256, "diversity_ft": 8.0, "reference": "raw", "likelihood": "gaussian"},
    {"id": "raw_c_b128_d4", "beam": 128, "diversity_ft": 4.0, "reference": "raw", "likelihood": "cauchy"},
    {"id": "raw_clip_b128_d4", "beam": 128, "diversity_ft": 4.0, "reference": "raw", "likelihood": "clipped_gaussian"},
    {"id": "shift_g_b128_d4", "beam": 128, "diversity_ft": 4.0, "reference": "shift", "likelihood": "gaussian"},
    {"id": "shift_c_b128_d4", "beam": 128, "diversity_ft": 4.0, "reference": "shift", "likelihood": "cauchy"},
]


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _reference_for_record(record, mode):
    if mode == "raw":
        return record["type_gr"]
    known = np.isfinite(record["tvt_input"]) & np.isfinite(record["gr"])
    rows = np.flatnonzero(known)[::4]
    reference = V["_interp_nan"](
        record["tvt_input"][rows], record["type_tvt"], record["type_gr"]
    )
    valid = np.isfinite(reference) & np.isfinite(record["gr"][rows])
    if valid.sum() < 32:
        shift = 0.0
    else:
        shift = float(np.median(record["gr"][rows][valid] - reference[valid]))
    return record["type_gr"] + shift


def _loss_and_surface(record, rows, start_md, start_surfaces, rates, reference_gr, likelihood):
    md = record["md"][rows]
    delta = md[None, None, :] - float(start_md)
    surface = start_surfaces[:, None, None] + rates[None, :, None] * delta
    candidate_tvt = surface - record["z"][rows][None, None, :]
    reference = V["_interp_nan"](
        candidate_tvt.ravel(), record["type_tvt"], reference_gr
    ).reshape(candidate_tvt.shape)
    observed = record["gr"][rows][None, None, :]
    valid = np.isfinite(reference) & np.isfinite(observed)
    normalised = (reference - observed) / record["noise_scale"]
    if likelihood == "gaussian":
        point_loss = normalised**2
    elif likelihood == "cauchy":
        point_loss = np.log1p(normalised**2)
    elif likelihood == "clipped_gaussian":
        point_loss = np.minimum(normalised**2, 9.0)
    else:
        raise ValueError(likelihood)
    point_loss[~valid] = 0.0
    coverage = np.sum(valid, axis=2)
    loss = np.sum(point_loss, axis=2) / np.maximum(coverage, 1)
    loss[coverage < max(2, int(math.ceil(0.25 * len(rows))))] = 1e6
    end_surface = start_surfaces[:, None] + rates[None, :] * float(md[-1] - start_md)
    return loss, end_surface


def _select_diverse(score, end_surface, beam_size, diversity_ft):
    flat_score = score.ravel()
    flat_surface = end_surface.ravel()
    if diversity_ft <= 0:
        count = min(int(beam_size), len(flat_score))
        keep = np.argpartition(flat_score, count - 1)[:count]
        return keep[np.argsort(flat_score[keep])]

    # Keep at most two paths per endpoint-surface bin, then backfill globally.
    order = np.argsort(flat_score)
    selected = []
    bin_count = {}
    selected_set = set()
    for index in order:
        key = int(np.floor(flat_surface[index] / diversity_ft))
        if bin_count.get(key, 0) >= 2:
            continue
        selected.append(int(index))
        selected_set.add(int(index))
        bin_count[key] = bin_count.get(key, 0) + 1
        if len(selected) >= beam_size:
            break
    if len(selected) < beam_size:
        for index in order:
            index = int(index)
            if index not in selected_set:
                selected.append(index)
                if len(selected) >= beam_size:
                    break
    return np.asarray(selected, dtype=np.int64)


def _beam_bank(record, prior, config):
    rates = prior["rate_grid"]
    segments = V["_segments"](record, SEGMENT_LENGTH_FT)
    reference_gr = _reference_for_record(record, config["reference"])
    scores = np.array([0.0], dtype=np.float64)
    surfaces = np.array([record["anchor_surface"]], dtype=np.float64)
    previous_rate = np.array([-1], dtype=np.int64)
    paths = np.empty((1, 0), dtype=np.int16)
    previous_md = record["anchor_md"]

    for rows in segments:
        loss, end_surface = _loss_and_surface(
            record,
            rows,
            previous_md,
            surfaces,
            rates,
            reference_gr,
            config["likelihood"],
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
        candidate_score = candidate_score + PERSISTENCE_WEIGHT * transition
        keep = _select_diverse(
            candidate_score,
            end_surface,
            int(config["beam"]),
            float(config["diversity_ft"]),
        )
        parent = keep // len(rates)
        rate_index = keep % len(rates)
        scores = candidate_score.ravel()[keep]
        surfaces = end_surface.ravel()[keep]
        previous_rate = rate_index.astype(np.int64)
        paths = np.concatenate(
            [paths[parent], rate_index[:, None].astype(np.int16)], axis=1
        )
        previous_md = float(record["md"][rows][-1])
    return scores, paths, segments


def _path_prediction(record, prior, path, segments):
    rates = prior["rate_grid"]
    prediction = np.full(record["md"].shape, np.nan, dtype=np.float64)
    start_md = record["anchor_md"]
    start_surface = record["anchor_surface"]
    for rows, rate_index in zip(segments, path):
        rate = float(rates[int(rate_index)])
        prediction[rows] = start_surface + rate * (
            record["md"][rows] - start_md
        ) - record["z"][rows]
        start_surface += rate * float(record["md"][rows][-1] - start_md)
        start_md = float(record["md"][rows][-1])
    valid = np.flatnonzero(np.isfinite(prediction))
    eval_rows = record["eval_rows"]
    surface = V["_interp_nan"](
        record["md"][eval_rows],
        record["md"][valid],
        prediction[valid] + record["z"][valid],
    )
    first_rate = float(rates[int(path[0])])
    last_rate = float(rates[int(path[-1])])
    left = record["md"][eval_rows] < record["md"][valid[0]]
    right = record["md"][eval_rows] > record["md"][valid[-1]]
    first_surface = prediction[valid[0]] + record["z"][valid[0]]
    last_surface = prediction[valid[-1]] + record["z"][valid[-1]]
    surface[left] = first_surface + first_rate * (
        record["md"][eval_rows[left]] - record["md"][valid[0]]
    )
    surface[right] = last_surface + last_rate * (
        record["md"][eval_rows[right]] - record["md"][valid[-1]]
    )
    prediction[eval_rows] = surface - record["z"][eval_rows]
    return prediction


def _evaluate_config(records, prior, config, label):
    rows = []
    for number, record in enumerate(records, start=1):
        scores, paths, segments = _beam_bank(record, prior, config)
        eval_rows = record["eval_rows"]
        truth = record["truth"][eval_rows]
        predictions = []
        mse = []
        for path in paths:
            prediction = _path_prediction(record, prior, path, segments)[eval_rows]
            predictions.append(prediction)
            mse.append(float(np.mean((prediction - truth) ** 2)))
        predictions = np.asarray(predictions, dtype=np.float64)
        mse = np.asarray(mse, dtype=np.float64)
        oracle_index = int(np.argmin(mse))
        score_scale = max(float(np.std(scores)), 1.0)
        weight = np.exp(-(scores - scores.min()) / score_scale)
        weight /= weight.sum()
        posterior = np.sum(weight[:, None] * predictions, axis=0)
        variants = {
            "map": predictions[0],
            "bank_oracle": predictions[oracle_index],
            "posterior": posterior,
        }
        for variant, prediction in variants.items():
            error = prediction - truth
            distance = record["md"][eval_rows] - record["anchor_md"]
            far = distance > 2000.0
            rows.append(
                {
                    "config_id": config["id"],
                    "stage": label,
                    "variant": variant,
                    "well": record["well"],
                    "rows": len(error),
                    "sse": float(np.sum(error**2)),
                    "sae": float(np.sum(np.abs(error))),
                    "far_rows": int(far.sum()),
                    "far_sse": float(np.sum(error[far] ** 2)),
                    "bank_paths": int(len(paths)),
                    "unique_endpoint_bins_4ft": int(
                        len(np.unique(np.floor(predictions[:, -1] / 4.0)))
                    ),
                    "map_to_oracle_mse_gap": float(mse[0] - mse[oracle_index]),
                }
            )
        if number % 16 == 0 or number == len(records):
            print(f"{label} {config['id']} {number}/{len(records)}", flush=True)
    return pd.DataFrame(rows)


def _summaries(frame):
    output = []
    for (config_id, variant), group in frame.groupby(["config_id", "variant"]):
        count = int(group["rows"].sum())
        sse = float(group["sse"].sum())
        far_rows = int(group["far_rows"].sum())
        far_sse = float(group["far_sse"].sum())
        output.append(
            {
                "config_id": config_id,
                "variant": variant,
                "rows": count,
                "wells": int(group["well"].nunique()),
                "exact_sse": sse,
                "rmse": math.sqrt(sse / count),
                "mae": float(group["sae"].sum() / count),
                "median_well_rmse": float(np.median(np.sqrt(group["sse"] / group["rows"]))),
                "p95_well_rmse": float(np.quantile(np.sqrt(group["sse"] / group["rows"]), 0.95)),
                "far_gt2000_rmse": math.sqrt(far_sse / far_rows) if far_rows else np.nan,
                "mean_unique_endpoint_bins_4ft": float(group["unique_endpoint_bins_4ft"].mean()),
            }
        )
    return pd.DataFrame(output)


# --- V2 EXECUTION ---
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
        print(f"V2 load {number}/{len(needed)} failures={len(failures)}", flush=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)

prior_records = [records[w] for w in needed if w in records and fold_by_well[w] in PRIOR_FOLDS]
selection_records = [records[w] for w in needed if w in records and fold_by_well[w] == SELECTION_FOLD]
application_records = [records[w] for w in needed if w in records and fold_by_well[w] == APPLICATION_FOLD]
prior = V["_fit_prior"](prior_records, SEGMENT_LENGTH_FT)

discovery = sorted(
    selection_records,
    key=lambda record: __import__("hashlib").sha256(
        f"v2-discovery:{record['well']}".encode("ascii")
    ).digest(),
)[:DISCOVERY_WELLS]

discovery_frames = []
for config in CONFIGS:
    discovery_frames.append(_evaluate_config(discovery, prior, config, "fold3_discovery"))
discovery_detail = pd.concat(discovery_frames, ignore_index=True)
discovery_summary = _summaries(discovery_detail)
discovery_detail.to_csv(OUT_DIR / "fold3_discovery_bywell.csv", index=False)
discovery_summary.to_csv(OUT_DIR / "fold3_discovery_summary.csv", index=False)
print("\nFOLD3 DISCOVERY SUMMARY")
print(discovery_summary.sort_values(["variant", "rmse"]).to_string(index=False), flush=True)

map_scan = discovery_summary[discovery_summary["variant"].eq("map")].copy()
oracle_scan = discovery_summary[discovery_summary["variant"].eq("bank_oracle")][
    ["config_id", "rmse"]
].rename(columns={"rmse": "bank_oracle_rmse"})
ranked = map_scan.merge(oracle_scan, on="config_id")
eligible = ranked[ranked["bank_oracle_rmse"] <= 6.0]
if len(eligible):
    confirmation_ids = eligible.nsmallest(3, "rmse")["config_id"].tolist()
else:
    confirmation_ids = ranked.nsmallest(3, "bank_oracle_rmse")["config_id"].tolist()

config_by_id = {config["id"]: config for config in CONFIGS}
confirmation_frames = []
for config_id in confirmation_ids:
    confirmation_frames.append(
        _evaluate_config(
            selection_records, prior, config_by_id[config_id], "fold3_confirmation"
        )
    )
confirmation_detail = pd.concat(confirmation_frames, ignore_index=True)
confirmation_summary = _summaries(confirmation_detail)
confirmation_detail.to_csv(OUT_DIR / "fold3_confirmation_bywell.csv", index=False)
confirmation_summary.to_csv(OUT_DIR / "fold3_confirmation_summary.csv", index=False)
print("\nFOLD3 CONFIRMATION SUMMARY")
print(confirmation_summary.sort_values(["variant", "rmse"]).to_string(index=False), flush=True)

confirm_map = confirmation_summary[confirmation_summary["variant"].eq("map")]
confirm_oracle = confirmation_summary[confirmation_summary["variant"].eq("bank_oracle")][
    ["config_id", "rmse"]
].rename(columns={"rmse": "bank_oracle_rmse"})
confirm_ranked = confirm_map.merge(confirm_oracle, on="config_id")
confirm_eligible = confirm_ranked[confirm_ranked["bank_oracle_rmse"] <= 6.0]
selected_row = (
    confirm_eligible.nsmallest(1, "rmse").iloc[0]
    if len(confirm_eligible)
    else confirm_ranked.nsmallest(1, "bank_oracle_rmse").iloc[0]
)
selected_config = config_by_id[str(selected_row["config_id"])]

fold2_detail = _evaluate_config(
    application_records, prior, selected_config, "fold2_application"
)
fold2_summary = _summaries(fold2_detail)
fold2_detail.to_csv(OUT_DIR / "fold2_bywell.csv", index=False)
fold2_summary.to_csv(OUT_DIR / "fold2_summary.csv", index=False)
print("\nFOLD2 BEAM SURVIVAL")
print(fold2_summary.sort_values("rmse").to_string(index=False), flush=True)

fold2_map = fold2_summary[fold2_summary["variant"].eq("map")].iloc[0].to_dict()
fold2_oracle = fold2_summary[fold2_summary["variant"].eq("bank_oracle")].iloc[0].to_dict()
fold2_posterior = fold2_summary[fold2_summary["variant"].eq("posterior")].iloc[0].to_dict()
gates = {
    "retained_bank_oracle_rmse_le_5p5": fold2_oracle["rmse"] <= 5.5,
    "map_rmse_le_12": fold2_map["rmse"] <= 12.0,
    "posterior_rmse_le_12": fold2_posterior["rmse"] <= 12.0,
}
if not gates["retained_bank_oracle_rmse_le_5p5"]:
    next_action = "FAIL -> useful paths are pruned; redesign mode proposals/search before scoring"
elif not (gates["map_rmse_le_12"] or gates["posterior_rmse_le_12"]):
    next_action = "GO RANKING -> good paths survive; build a complete-well bank scorer, not a new generator"
else:
    next_action = "GO ENDPOINT -> retain decoder and perform the independent five-fold endpoint audit"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_beam_survival_and_emission_pricing",
    "data_root": str(DATA_ROOT),
    "upstream_code": str(V1_SOURCE),
    "upstream_prediction_artifacts": [],
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": 1,
        "fold1_touched": False,
    },
    "selected_config": selected_config,
    "fold2_map": fold2_map,
    "fold2_retained_bank_oracle": fold2_oracle,
    "fold2_posterior": fold2_posterior,
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "next": next_action,
    "elapsed_minutes": (time.time() - started) / 60.0,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nFINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR)
