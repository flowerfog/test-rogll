"""Raw-data-only segment-lattice gateway following the Phase-0 investment audit.

Lineage and protocol
--------------------
* Inputs: original train horizontal/typewell CSVs only.
* Fold assignment: deterministically rebuilt from well IDs (same recipe as
  rawdata_reasoning_phase0_v0).
* Folds 4/5: fit structural-rate and persistence priors.
* Fold 3: select segment length and fixed decoder weights.
* Fold 2: apply the frozen recipe exactly once.
* Fold 1: untouched.

The lattice starts at the last visible TVT_input row and represents the hidden
structural surface S = TVT + Z as piecewise-linear segments. Candidate paths
are therefore label-free. Truth enters only (a) outer-train prior fitting,
(b) lattice-oracle measurement, and (c) held-out scoring.
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


EXPERIMENT = "rawdata_segment_lattice_nested_v1"
SEED = 20260721
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1

SEGMENT_LENGTHS_FT = (128.0, 256.0, 384.0, 512.0)
RATE_STATES = 25
BEAM_SIZE = 40
ROW_STRIDE = 8
MIN_SEGMENT_ROWS = 4
DISCOVERY_WELLS = 32
ORACLE_GATE_RMSE = 5.5
DECODER_GATE_RMSE = 12.0


def _find_data_root() -> Path:
    candidates = [
        Path.cwd(),
        Path("/root/autodl-tmp"),
        Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    ]
    for candidate in candidates:
        train = candidate / "train"
        if train.is_dir() and next(train.glob("*__horizontal_well.csv"), None):
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate original competition train directory")


DATA_ROOT = _find_data_root()
if os.name != "nt" and Path("/root/autodl-tmp").exists():
    OUT_DIR = Path("/root/autodl-tmp/oof_exports") / EXPERIMENT
else:
    OUT_DIR = Path.cwd() / "output" / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _clean_json(value):
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_clean_json(payload), indent=2, allow_nan=False), encoding="utf-8")


def _stable_folds(wells):
    scored = []
    for well in wells:
        digest = hashlib.sha256(f"{SEED}:{well}".encode("ascii")).digest()
        scored.append((int.from_bytes(digest[:8], "little"), well))
    ordered = [well for _, well in sorted(scored)]
    fold = np.empty(len(ordered), dtype=np.int64)
    for number, indices in enumerate(np.array_split(np.arange(len(ordered)), 5), start=1):
        fold[indices] = number
    return pd.DataFrame({"well": ordered, "fold": fold})


def _interp_nan(x, xp, fp):
    x = np.asarray(x, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    valid = np.isfinite(xp) & np.isfinite(fp)
    if valid.sum() < 2:
        return np.full(x.shape, np.nan, dtype=np.float64)
    xp = xp[valid]
    fp = fp[valid]
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]
    unique, indices = np.unique(xp, return_index=True)
    fp = fp[indices]
    result = np.interp(x, unique, fp)
    result[(x < unique[0]) | (x > unique[-1])] = np.nan
    return result


def _load_record(well):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    horizontal = horizontal.sort_values("MD").reset_index(drop=True)
    md = horizontal["MD"].to_numpy(np.float64)
    z = horizontal["Z"].to_numpy(np.float64)
    truth = horizontal["TVT"].to_numpy(np.float64)
    gr = horizontal["GR"].to_numpy(np.float64)
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    known = np.isfinite(tvt_input) & np.isfinite(truth)
    evaluation = (~np.isfinite(tvt_input)) & np.isfinite(truth)
    if known.sum() < 100 or evaluation.sum() < 100:
        raise RuntimeError(f"{well}: known/evaluation={known.sum()}/{evaluation.sum()}")
    anchor = int(np.flatnonzero(known)[-1])
    type_tvt = typewell["TVT"].to_numpy(np.float64)
    type_gr = typewell["GR"].to_numpy(np.float64)

    known_rows = np.flatnonzero(known & np.isfinite(gr))[::4]
    known_reference = _interp_nan(tvt_input[known_rows], type_tvt, type_gr)
    residual = gr[known_rows] - known_reference
    residual = residual[np.isfinite(residual)]
    if len(residual) < 32:
        noise_scale = 15.0
    else:
        centre = np.median(residual)
        noise_scale = 1.4826 * np.median(np.abs(residual - centre))
    noise_scale = float(np.clip(noise_scale, 5.0, 35.0))

    eval_rows = np.flatnonzero(evaluation)
    # Segment geometry is defined by MD, not by GR availability. Missing-GR
    # rows remain in the lattice and simply contribute no emission evidence.
    sampled = eval_rows[::ROW_STRIDE]
    if len(sampled) < 64:
        raise RuntimeError(f"{well}: only {len(sampled)} sampled evaluation rows")
    return {
        "well": well,
        "md": md,
        "z": z,
        "truth": truth,
        "gr": gr,
        "tvt_input": tvt_input,
        "evaluation": evaluation,
        "eval_rows": eval_rows,
        "sampled_rows": sampled,
        "anchor_index": anchor,
        "anchor_md": float(md[anchor]),
        "anchor_surface": float(tvt_input[anchor] + z[anchor]),
        "type_tvt": type_tvt,
        "type_gr": type_gr,
        "noise_scale": noise_scale,
    }


def _segments(record, length_ft):
    rows = record["sampled_rows"]
    md = record["md"][rows]
    segment_id = np.floor((md - record["anchor_md"]) / length_ft).astype(np.int64)
    output = []
    for identifier in np.unique(segment_id):
        selected = rows[segment_id == identifier]
        if len(selected) >= MIN_SEGMENT_ROWS:
            output.append(selected)
    if not output:
        raise RuntimeError(f"{record['well']}: no segments at {length_ft}")
    return output


def _segment_truth_rates(record, length_ft):
    rates = []
    previous_md = record["anchor_md"]
    previous_surface = record["anchor_surface"]
    for rows in _segments(record, length_ft):
        md = record["md"][rows]
        surface = record["truth"][rows] + record["z"][rows]
        delta = md - previous_md
        denominator = float(np.dot(delta, delta))
        if denominator <= 0:
            continue
        rate = float(np.dot(delta, surface - previous_surface) / denominator)
        rates.append(rate)
        previous_surface = previous_surface + rate * float(md[-1] - previous_md)
        previous_md = float(md[-1])
    return np.asarray(rates, dtype=np.float64)


def _fit_prior(records, length_ft):
    all_rates = []
    all_changes = []
    for record in records:
        rates = _segment_truth_rates(record, length_ft)
        if len(rates):
            all_rates.extend(rates.tolist())
        if len(rates) > 1:
            all_changes.extend(np.diff(rates).tolist())
    rates = np.asarray(all_rates, dtype=np.float64)
    changes = np.asarray(all_changes, dtype=np.float64)
    if len(rates) < 100:
        raise RuntimeError(f"too few prior rates at {length_ft}: {len(rates)}")
    rate_centre = float(np.median(rates))
    rate_scale = float(max(1.4826 * np.median(np.abs(rates - rate_centre)), 0.005))
    change_centre = float(np.median(changes)) if len(changes) else 0.0
    change_scale = float(
        max(1.4826 * np.median(np.abs(changes - change_centre)), 0.005)
    )
    low = min(float(np.quantile(rates, 0.005) - 0.01), -0.01)
    high = max(float(np.quantile(rates, 0.995) + 0.01), 0.01)
    grid = np.linspace(low, high, RATE_STATES, dtype=np.float64)
    return {
        "length_ft": float(length_ft),
        "rate_centre": rate_centre,
        "rate_scale": rate_scale,
        "change_centre": change_centre,
        "change_scale": change_scale,
        "rate_grid": grid,
        "rate_count": int(len(rates)),
        "change_count": int(len(changes)),
        "rate_p005": float(np.quantile(rates, 0.005)),
        "rate_p995": float(np.quantile(rates, 0.995)),
    }


def _candidate_segment_loss(record, rows, start_md, start_surfaces, rates, mode, type_gr=None):
    md = record["md"][rows]
    delta = md[None, None, :] - float(start_md)
    surface = start_surfaces[:, None, None] + rates[None, :, None] * delta
    candidate_tvt = surface - record["z"][rows][None, None, :]
    if mode == "oracle":
        difference = candidate_tvt - record["truth"][rows][None, None, :]
        loss = np.mean(difference**2, axis=2)
    else:
        reference_gr = record["type_gr"] if type_gr is None else type_gr
        reference = _interp_nan(
            candidate_tvt.ravel(), record["type_tvt"], reference_gr
        ).reshape(candidate_tvt.shape)
        observed = record["gr"][rows][None, None, :]
        valid = np.isfinite(reference) & np.isfinite(observed)
        difference = (reference - observed) / record["noise_scale"]
        squared = difference**2
        squared[~valid] = 0.0
        coverage = np.sum(valid, axis=2)
        loss = np.sum(squared, axis=2) / np.maximum(coverage, 1)
        # Sparse logging is legal. A segment with little/no GR is propagated by
        # structural priors rather than being removed from the state sequence.
        loss[coverage < max(2, int(math.ceil(0.25 * len(rows))))] = 1e6
    end_delta = float(md[-1] - start_md)
    end_surface = start_surfaces[:, None] + rates[None, :] * end_delta
    return loss, end_surface


def _beam_decode(record, prior, config, mode="full", shuffled_type_gr=None):
    length_ft = float(config["segment_length_ft"])
    rates = prior["rate_grid"]
    segments = _segments(record, length_ft)
    beam_scores = np.array([0.0], dtype=np.float64)
    beam_surfaces = np.array([record["anchor_surface"]], dtype=np.float64)
    beam_previous_rate = np.array([-1], dtype=np.int64)
    beam_paths = np.empty((1, 0), dtype=np.int16)
    previous_md = record["anchor_md"]

    emission_weight = float(config.get("emission_weight", 1.0))
    persistence_weight = float(config.get("persistence_weight", 0.0))
    rate_weight = float(config.get("rate_weight", 0.0))
    if mode == "oracle":
        emission_weight = 1.0
        persistence_weight = 0.0
        rate_weight = 0.0
    elif mode == "no_gr":
        emission_weight = 0.0
    elif mode == "no_persistence":
        persistence_weight = 0.0

    for rows in segments:
        loss, end_surface = _candidate_segment_loss(
            record,
            rows,
            previous_md,
            beam_surfaces,
            rates,
            "oracle" if mode == "oracle" else "gr",
            type_gr=shuffled_type_gr,
        )
        score = beam_scores[:, None] + emission_weight * loss
        score = score + rate_weight * (
            (rates[None, :] - prior["rate_centre"]) / prior["rate_scale"]
        ) ** 2
        if persistence_weight > 0:
            previous = np.where(
                beam_previous_rate[:, None] >= 0,
                rates[np.maximum(beam_previous_rate[:, None], 0)],
                prior["rate_centre"],
            )
            transition = (
                (rates[None, :] - previous - prior["change_centre"])
                / prior["change_scale"]
            ) ** 2
            score = score + persistence_weight * transition

        flat_score = score.ravel()
        keep_count = min(BEAM_SIZE, len(flat_score))
        keep = np.argpartition(flat_score, keep_count - 1)[:keep_count]
        keep = keep[np.argsort(flat_score[keep])]
        parent = keep // len(rates)
        rate_index = keep % len(rates)
        beam_scores = flat_score[keep]
        beam_surfaces = end_surface[parent, rate_index]
        beam_previous_rate = rate_index.astype(np.int64)
        beam_paths = np.concatenate(
            [beam_paths[parent], rate_index[:, None].astype(np.int16)], axis=1
        )
        previous_md = float(record["md"][rows][-1])

    selected_path = beam_paths[0]
    prediction = np.full(record["md"].shape, np.nan, dtype=np.float64)
    start_md = record["anchor_md"]
    start_surface = record["anchor_surface"]
    for rows, rate_index in zip(segments, selected_path):
        rate = float(rates[int(rate_index)])
        delta = record["md"][rows] - start_md
        prediction[rows] = start_surface + rate * delta - record["z"][rows]
        start_surface = start_surface + rate * float(record["md"][rows][-1] - start_md)
        start_md = float(record["md"][rows][-1])

    # Fill unsampled official evaluation rows by interpolating the structural surface.
    sampled_valid = np.flatnonzero(np.isfinite(prediction))
    eval_rows = record["eval_rows"]
    predicted_surface = _interp_nan(
        record["md"][eval_rows],
        record["md"][sampled_valid],
        prediction[sampled_valid] + record["z"][sampled_valid],
    )
    if len(sampled_valid):
        left = record["md"][eval_rows] < record["md"][sampled_valid[0]]
        right = record["md"][eval_rows] > record["md"][sampled_valid[-1]]
        first_rate = float(rates[int(selected_path[0])])
        last_rate = float(rates[int(selected_path[-1])])
        first_surface = prediction[sampled_valid[0]] + record["z"][sampled_valid[0]]
        last_surface = prediction[sampled_valid[-1]] + record["z"][sampled_valid[-1]]
        predicted_surface[left] = first_surface + first_rate * (
            record["md"][eval_rows[left]] - record["md"][sampled_valid[0]]
        )
        predicted_surface[right] = last_surface + last_rate * (
            record["md"][eval_rows[right]] - record["md"][sampled_valid[-1]]
        )
    prediction[eval_rows] = predicted_surface - record["z"][eval_rows]
    return prediction, float(beam_scores[0]), selected_path


def _metric_rows(records, predictions, label):
    rows = []
    bands = (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    )
    for record in records:
        eval_rows = record["eval_rows"]
        prediction = predictions[record["well"]][eval_rows]
        truth = record["truth"][eval_rows]
        valid = np.isfinite(prediction) & np.isfinite(truth)
        error = prediction[valid] - truth[valid]
        item = {
            "variant": label,
            "well": record["well"],
            "rows": int(valid.sum()),
            "sse": float(np.sum(error**2)),
            "sae": float(np.sum(np.abs(error))),
        }
        distance = record["md"][eval_rows][valid] - record["anchor_md"]
        for band, low, high in bands:
            chosen = (distance >= low) & (distance < high)
            item[f"{band}_rows"] = int(chosen.sum())
            item[f"{band}_sse"] = float(np.sum(error[chosen] ** 2))
        rows.append(item)
    return pd.DataFrame(rows)


def _summary(frame):
    count = int(frame["rows"].sum())
    sse = float(frame["sse"].sum())
    result = {
        "variant": str(frame["variant"].iloc[0]),
        "rows": count,
        "wells": int(frame["well"].nunique()),
        "exact_sse": sse,
        "rmse": math.sqrt(sse / count),
        "mae": float(frame["sae"].sum() / count),
        "median_well_rmse": float(np.median(np.sqrt(frame["sse"] / frame["rows"]))),
        "p75_well_rmse": float(np.quantile(np.sqrt(frame["sse"] / frame["rows"]), 0.75)),
        "p95_well_rmse": float(np.quantile(np.sqrt(frame["sse"] / frame["rows"]), 0.95)),
        "max_well_rmse": float(np.max(np.sqrt(frame["sse"] / frame["rows"]))),
    }
    for band in ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000"):
        band_rows = int(frame[f"{band}_rows"].sum())
        band_sse = float(frame[f"{band}_sse"].sum())
        result[f"{band}_rmse"] = math.sqrt(band_sse / band_rows) if band_rows else np.nan
    return result


def _decode_many(records, prior, config, mode="full", shuffle_map=None, progress="decode"):
    predictions = {}
    for number, record in enumerate(records, start=1):
        shuffled = None
        if shuffle_map is not None:
            shuffled = shuffle_map[record["well"]]
        prediction, _, _ = _beam_decode(
            record, prior, config, mode=mode, shuffled_type_gr=shuffled
        )
        predictions[record["well"]] = prediction
        if number % 50 == 0 or number == len(records):
            print(f"{progress} {number}/{len(records)}", flush=True)
    return predictions


started = time.time()
all_wells = [
    path.name.split("__", 1)[0]
    for path in sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
]
folds = _stable_folds(all_wells)
folds.to_csv(OUT_DIR / "raw_grouped_fivefold.csv", index=False)
fold_by_well = dict(zip(folds["well"], folds["fold"]))
needed_wells = [well for well in all_wells if fold_by_well[well] in (2, 3, 4, 5)]

records = {}
failures = []
for number, well in enumerate(needed_wells, start=1):
    try:
        records[well] = _load_record(well)
    except Exception as exc:
        failures.append({"well": well, "fold": int(fold_by_well[well]), "reason": repr(exc)})
    if number % 50 == 0 or number == len(needed_wells):
        print(f"load raw records {number}/{len(needed_wells)} failures={len(failures)}", flush=True)
pd.DataFrame(failures, columns=["well", "fold", "reason"]).to_csv(
    OUT_DIR / "record_failures.csv", index=False
)

prior_records = [records[well] for well in needed_wells if well in records and fold_by_well[well] in PRIOR_FOLDS]
selection_records = [records[well] for well in needed_wells if well in records and fold_by_well[well] == SELECTION_FOLD]
application_records = [records[well] for well in needed_wells if well in records and fold_by_well[well] == APPLICATION_FOLD]
if min(len(prior_records), len(selection_records), len(application_records)) < 145:
    raise RuntimeError(
        f"insufficient fold records: {len(prior_records)}/{len(selection_records)}/{len(application_records)}"
    )

priors = {}
for length_ft in SEGMENT_LENGTHS_FT:
    priors[length_ft] = _fit_prior(prior_records, length_ft)
prior_payload = {
    str(int(length)): {key: value for key, value in prior.items() if key != "rate_grid"}
    | {"rate_grid": prior["rate_grid"].tolist()}
    for length, prior in priors.items()
}
_write_json(OUT_DIR / "outer_train_priors.json", prior_payload)
print("OUTER-TRAIN PRIORS", json.dumps(prior_payload, indent=2), flush=True)

# Representation gate: truth may select the best valid lattice path, while every
# state and prior remains fixed from Folds 4/5.
oracle_rows = []
for length_ft in SEGMENT_LENGTHS_FT:
    config = {
        "segment_length_ft": length_ft,
        "emission_weight": 1.0,
        "persistence_weight": 0.0,
        "rate_weight": 0.0,
    }
    predictions = _decode_many(
        selection_records,
        priors[length_ft],
        config,
        mode="oracle",
        progress=f"oracle L{int(length_ft)}",
    )
    metrics = _summary(_metric_rows(selection_records, predictions, f"oracle_L{int(length_ft)}"))
    oracle_rows.append(metrics)
oracle_scan = pd.DataFrame(oracle_rows).sort_values("rmse")
oracle_scan.to_csv(OUT_DIR / "fold3_lattice_oracle_scan.csv", index=False)
print("\nFOLD3 LATTICE ORACLE SCAN")
print(oracle_scan.to_string(index=False), flush=True)

best_oracle = oracle_scan.iloc[0].to_dict()
if float(best_oracle["rmse"]) > ORACLE_GATE_RMSE:
    verdict = {
        "experiment": EXPERIMENT,
        "stage": "fold3_lattice_oracle_gate",
        "protocol": {
            "prior_folds": list(PRIOR_FOLDS),
            "selection_fold": SELECTION_FOLD,
            "application_fold": APPLICATION_FOLD,
            "untouched_fold": UNTOUCHED_FOLD,
        },
        "best_oracle": best_oracle,
        "oracle_gate_rmse_le": ORACLE_GATE_RMSE,
        "strict_pass": False,
        "next": "FAIL -> state lattice cannot represent the target; stop before GR decoding",
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    _write_json(OUT_DIR / "verdict.json", verdict)
    print("FINAL VERDICT", json.dumps(_clean_json(verdict), indent=2), flush=True)
    raise RuntimeError(json.dumps(_clean_json(verdict), indent=2))

ranked_lengths = oracle_scan["variant"].str.extract(r"L(\d+)")[0].astype(float).tolist()[:2]
discovery_records = sorted(
    selection_records,
    key=lambda record: hashlib.sha256(f"discovery:{record['well']}".encode("ascii")).digest(),
)[:DISCOVERY_WELLS]
weight_recipes = [
    (0.0, 0.0),
    (0.1, 0.0),
    (0.3, 0.0),
    (1.0, 0.0),
    (0.3, 0.1),
    (1.0, 0.1),
    (3.0, 0.1),
]
discovery_rows = []
for length_ft in ranked_lengths:
    for persistence_weight, rate_weight in weight_recipes:
        config = {
            "segment_length_ft": length_ft,
            "emission_weight": 1.0,
            "persistence_weight": persistence_weight,
            "rate_weight": rate_weight,
        }
        predictions = _decode_many(
            discovery_records,
            priors[length_ft],
            config,
            progress=f"discovery L{int(length_ft)} p{persistence_weight} r{rate_weight}",
        )
        metrics = _summary(_metric_rows(discovery_records, predictions, "discovery"))
        discovery_rows.append({**config, **metrics})
discovery_scan = pd.DataFrame(discovery_rows).sort_values("rmse")
discovery_scan.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
print("\nFOLD3 DISCOVERY SCAN")
print(discovery_scan.to_string(index=False), flush=True)

confirmation_rows = []
recipe_columns = ["segment_length_ft", "emission_weight", "persistence_weight", "rate_weight"]
for _, recipe in discovery_scan.head(3).iterrows():
    config = {column: float(recipe[column]) for column in recipe_columns}
    length_ft = config["segment_length_ft"]
    predictions = _decode_many(
        selection_records,
        priors[length_ft],
        config,
        progress=f"fold3 confirm L{int(length_ft)} p{config['persistence_weight']}",
    )
    metrics = _summary(_metric_rows(selection_records, predictions, "fold3_confirmation"))
    confirmation_rows.append({**config, **metrics})
confirmation_scan = pd.DataFrame(confirmation_rows).sort_values("rmse")
confirmation_scan.to_csv(OUT_DIR / "fold3_confirmation_scan.csv", index=False)
print("\nFOLD3 CONFIRMATION SCAN")
print(confirmation_scan.to_string(index=False), flush=True)

selected = confirmation_scan.iloc[0].to_dict()
selected_config = {column: float(selected[column]) for column in recipe_columns}
selected_length = selected_config["segment_length_ft"]
selected_prior = priors[selected_length]

# Fold 2 is touched for the first and only recipe application here.
fold2_predictions = _decode_many(
    application_records, selected_prior, selected_config, progress="fold2 full"
)
full_bywell = _metric_rows(application_records, fold2_predictions, "full")
full_metrics = _summary(full_bywell)

no_gr_predictions = _decode_many(
    application_records, selected_prior, selected_config, mode="no_gr", progress="fold2 no_gr"
)
no_gr_metrics = _summary(_metric_rows(application_records, no_gr_predictions, "no_gr"))

no_persistence_predictions = _decode_many(
    application_records,
    selected_prior,
    selected_config,
    mode="no_persistence",
    progress="fold2 no_persistence",
)
no_persistence_metrics = _summary(
    _metric_rows(application_records, no_persistence_predictions, "no_persistence")
)

ordered_application = sorted(application_records, key=lambda record: record["well"])
shifted_type_gr = {
    record["well"]: ordered_application[(index + 1) % len(ordered_application)]["type_gr"]
    for index, record in enumerate(ordered_application)
}
# Shuffled references have different TVT axes. Interpolate the donor log onto
# the recipient typewell axis before passing it to the common decoder.
shifted_type_gr = {
    record["well"]: _interp_nan(
        record["type_tvt"],
        ordered_application[(index + 1) % len(ordered_application)]["type_tvt"],
        ordered_application[(index + 1) % len(ordered_application)]["type_gr"],
    )
    for index, record in enumerate(ordered_application)
}
shuffled_predictions = _decode_many(
    application_records,
    selected_prior,
    selected_config,
    shuffle_map=shifted_type_gr,
    progress="fold2 shuffled_typewell",
)
shuffled_metrics = _summary(
    _metric_rows(application_records, shuffled_predictions, "shuffled_typewell")
)

oracle_config = {
    "segment_length_ft": selected_length,
    "emission_weight": 1.0,
    "persistence_weight": 0.0,
    "rate_weight": 0.0,
}
fold2_oracle_predictions = _decode_many(
    application_records,
    selected_prior,
    oracle_config,
    mode="oracle",
    progress="fold2 oracle",
)
fold2_oracle = _summary(
    _metric_rows(application_records, fold2_oracle_predictions, "lattice_oracle")
)

controls = pd.DataFrame([full_metrics, no_gr_metrics, no_persistence_metrics, shuffled_metrics])
controls["degradation_vs_full"] = controls["rmse"] - full_metrics["rmse"]
controls.to_csv(OUT_DIR / "fold2_controls.csv", index=False)
full_bywell["well_rmse"] = np.sqrt(full_bywell["sse"] / full_bywell["rows"])
full_bywell.sort_values("well_rmse", ascending=False).to_csv(
    OUT_DIR / "fold2_full_bywell.csv", index=False
)
print("\nFOLD2 CONTROLS")
print(controls.to_string(index=False), flush=True)

gates = {
    "fold2_lattice_oracle_rmse_le_5p5": fold2_oracle["rmse"] <= 5.5,
    "fold2_decoder_rmse_le_12": full_metrics["rmse"] <= DECODER_GATE_RMSE,
    "fold2_far_rmse_le_15": full_metrics["far_gt2000_rmse"] <= 15.0,
    "no_gr_degradation_ge_1": no_gr_metrics["rmse"] - full_metrics["rmse"] >= 1.0,
    "shuffled_typewell_degradation_ge_1": shuffled_metrics["rmse"] - full_metrics["rmse"] >= 1.0,
}
if not gates["fold2_lattice_oracle_rmse_le_5p5"]:
    next_action = "FAIL -> candidate representation did not transfer; revise the lattice before scoring"
elif not gates["fold2_decoder_rmse_le_12"]:
    next_action = "HOLD -> representation transfers but GR scoring is the bottleneck; price richer sequence evidence"
elif not all((gates["no_gr_degradation_ge_1"], gates["shuffled_typewell_degradation_ge_1"])):
    next_action = "HOLD -> decoder score is not causally supported by GR/typewell alignment"
else:
    next_action = "GO -> retain this raw-data endpoint and measure its fold-clean error independence"

verdict = {
    "experiment": EXPERIMENT,
    "stage": "raw_data_nested_segment_lattice_gateway",
    "data_root": str(DATA_ROOT),
    "upstream_dependencies": [],
    "sample_unit": "one complete physical well",
    "deployable_inputs_only": True,
    "truth_usage": "outer-train priors, lattice oracle, and held-out scoring only",
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
    },
    "records": {
        "prior": len(prior_records),
        "selection": len(selection_records),
        "application": len(application_records),
        "failures": len(failures),
    },
    "selected_config": selected_config,
    "fold3_selected": selected,
    "fold2_lattice_oracle": fold2_oracle,
    "fold2_full": full_metrics,
    "controls": {
        "no_gr": no_gr_metrics,
        "no_persistence": no_persistence_metrics,
        "shuffled_typewell": shuffled_metrics,
    },
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "next": next_action,
    "elapsed_minutes": (time.time() - started) / 60.0,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nFINAL VERDICT")
print(json.dumps(_clean_json(verdict), indent=2), flush=True)
print("saved:", OUT_DIR)
