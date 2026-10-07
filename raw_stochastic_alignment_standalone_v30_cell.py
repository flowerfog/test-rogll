"""Raw-data stochastic alignment as a standalone TVT endpoint.

This experiment deliberately has no dependency on any historical prediction.
It reads only each well's horizontal log, its assigned typewell, the visible
TVT prefix, MD/Z geometry, and fold-clean structural priors fitted on outer
training wells.  CP, Neighbor, SP45, FLE, Ridge, and the 6.377 submission are
not state centres, features, priors, or training targets.

Protocol
--------
* Folds 4/5: fit a robust structural-surface rate prior.
* Fold 3: select alignment temperature, transition scale, reference mode, and
  posterior estimator.  A 24-well subset is only a compute-saving discovery
  stage; shortlisted recipes are confirmed on the complete fold.
* Fold 2: one frozen confirmation with no-GR and shuffled-typewell controls.
* Fold 1: one frozen final measurement.

The latent state is an absolute structural surface

    surface(MD) = anchor_surface + prior_rate * distance + offset_state

and TVT is recovered as surface - Z.  A forward-backward posterior and
backward path samples provide stochastic-DTW-style path uncertainty; Viterbi,
posterior mean, and sampled posterior median are all measured.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp


EXPERIMENT = "raw_stochastic_alignment_standalone_v30"
SEED = 20260726
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
FINAL_FOLD = 1

NODE_SPACING_FT = 64.0
OFFSETS_FT = np.arange(-128.0, 128.0001, 4.0, dtype=np.float64)
ZERO_STATE = int(np.argmin(np.abs(OFFSETS_FT)))
PATH_SAMPLES = 24
DISCOVERY_WELLS = int(os.environ.get("RAW_STOCHASTIC_DISCOVERY_WELLS", "24"))
MIN_PREFIX_ROWS = 64
MIN_EVAL_ROWS = 96

# Horizontal smoothing is in measured depth; typewell smoothing is in TVT.
# The paired scales intentionally span raw texture through broad motif shape.
HORIZONTAL_SCALES_FT = (0.0, 16.0, 64.0, 128.0)
TYPEWELL_SCALES_FT = (0.0, 2.0, 6.0, 12.0)
CHANNEL_WEIGHTS = np.asarray((1.0, 0.75, 0.50, 0.30), np.float64)

TEMPERATURES = (0.5, 1.0, 2.0)
TRANSITION_MULTIPLIERS = (0.75, 1.50)
REFERENCE_MODES = ("raw", "prefix_shift")
ESTIMATORS = ("map", "posterior_mean", "sample_median")


def _find_data_root() -> Path:
    candidates = (
        Path.cwd(),
        Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
        Path("/root/autodl-tmp"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    )
    for candidate in candidates:
        train = candidate / "train"
        if train.is_dir() and next(train.glob("*__horizontal_well.csv"), None):
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate the original competition train directory")


DATA_ROOT = _find_data_root()
if os.name != "nt" and Path("/root/autodl-tmp").exists():
    OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
else:
    OOF_ROOT = Path.cwd() / "output"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _stable_folds(wells):
    scored = []
    for well in wells:
        digest = hashlib.sha256(f"{SEED}:{well}".encode("ascii")).digest()
        scored.append((int.from_bytes(digest[:8], "little"), well))
    ordered = [well for _, well in sorted(scored)]
    output = []
    for fold, indices in enumerate(np.array_split(np.arange(len(ordered)), 5), 1):
        output.extend((ordered[int(index)], fold) for index in indices)
    return pd.DataFrame(output, columns=["well", "fold"])


def _deduplicate(axis, values):
    frame = pd.DataFrame({"axis": axis, "value": values}).replace(
        [np.inf, -np.inf], np.nan
    )
    frame = (
        frame.dropna()
        .groupby("axis", as_index=False)["value"]
        .median()
        .sort_values("axis")
    )
    if len(frame) < 64:
        raise RuntimeError("typewell has too few unique finite TVT samples")
    return frame["axis"].to_numpy(np.float64), frame["value"].to_numpy(np.float64)


def _robust_scale(values, floor):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float(floor)
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    return float(max(1.4826 * mad, 0.15 * np.std(values), floor))


def _smooth_irregular(axis, values, scale_ft):
    if scale_ft <= 0.0:
        return values.astype(np.float64, copy=True)
    spacing = float(np.median(np.diff(axis)))
    sigma = max(float(scale_ft) / max(spacing, 1e-6), 0.25)
    return gaussian_filter1d(values.astype(np.float64), sigma=sigma, mode="nearest")


def _load_record(well):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    horizontal = horizontal.sort_values("MD").reset_index(drop=True)
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float64)
    gr_series = pd.to_numeric(horizontal["GR"], errors="coerce").interpolate(
        limit_direction="both"
    )
    gr = gr_series.to_numpy(np.float64)
    truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md) & np.isfinite(z))
    evaluation = np.flatnonzero(
        ~np.isfinite(tvt_input) & np.isfinite(truth) & np.isfinite(md) & np.isfinite(z)
    )
    if len(known) < MIN_PREFIX_ROWS or len(evaluation) < MIN_EVAL_ROWS:
        raise RuntimeError(f"{well}: prefix/eval rows={len(known)}/{len(evaluation)}")
    anchor = int(known[-1])

    tw_tvt, tw_gr = _deduplicate(
        pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64),
        pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64),
    )
    expected_prefix = np.interp(
        tvt_input[known], tw_tvt, tw_gr, left=np.nan, right=np.nan
    )
    valid_prefix = np.isfinite(expected_prefix) & np.isfinite(gr[known])
    residual = gr[known][valid_prefix] - expected_prefix[valid_prefix]
    prefix_shift = float(np.median(residual)) if len(residual) else 0.0
    gamma = float(np.clip(_robust_scale(residual - prefix_shift, 5.0), 5.0, 35.0))

    prefix_tail = known[md[known] >= md[anchor] - 1000.0]
    if len(prefix_tail) < 32:
        prefix_tail = known[-min(256, len(known)):]
    prefix_surface = tvt_input[prefix_tail] + z[prefix_tail]
    prefix_surface_rate = float(np.polyfit(md[prefix_tail], prefix_surface, 1)[0])

    md_spacing = float(max(np.median(np.diff(md[np.isfinite(md)])), 1e-3))
    horizontal_channels = []
    for scale in HORIZONTAL_SCALES_FT:
        if scale <= 0.0:
            horizontal_channels.append(gr.copy())
        else:
            horizontal_channels.append(
                gaussian_filter1d(gr, sigma=max(scale / md_spacing, 0.25), mode="nearest")
            )
    typewell_channels = np.stack(
        [_smooth_irregular(tw_tvt, tw_gr, scale) for scale in TYPEWELL_SCALES_FT]
    )
    return {
        "well": str(well),
        "md": md,
        "z": z,
        "truth": truth,
        "tvt_input": tvt_input,
        "known": known,
        "evaluation": evaluation,
        "anchor": anchor,
        "anchor_md": float(md[anchor]),
        "anchor_tvt": float(tvt_input[anchor]),
        "anchor_surface": float(tvt_input[anchor] + z[anchor]),
        "tw_tvt": tw_tvt,
        "tw_channels": typewell_channels,
        "horizontal_channels": np.stack(horizontal_channels),
        "prefix_shift": prefix_shift,
        "gamma": gamma,
        "prefix_surface_rate": prefix_surface_rate,
    }


def _fit_surface_prior(records):
    rates = []
    changes = []
    block_ft = 128.0
    for record in records:
        rows = np.r_[record["anchor"], record["evaluation"]]
        distance = record["md"][rows] - record["anchor_md"]
        surface = record["truth"][rows] + record["z"][rows]
        targets = np.arange(0.0, float(distance[-1]) + 0.5, block_ft)
        positions = np.searchsorted(distance, targets, side="left")
        positions = np.unique(np.clip(positions, 0, len(rows) - 1))
        if positions[-1] != len(rows) - 1:
            positions = np.r_[positions, len(rows) - 1]
        local_rates = []
        for left, right in zip(positions[:-1], positions[1:]):
            delta = float(distance[right] - distance[left])
            if delta < 32.0:
                continue
            local_rates.append(float((surface[right] - surface[left]) / delta))
        rates.extend(local_rates)
        if len(local_rates) > 1:
            changes.extend(np.diff(local_rates).tolist())
    rates = np.asarray(rates, np.float64)
    changes = np.asarray(changes, np.float64)
    if len(rates) < 1000:
        raise RuntimeError(f"Too few outer-train structural rates: {len(rates)}")
    centre = float(np.median(rates))
    scale = _robust_scale(rates - centre, 0.005)
    change_scale = _robust_scale(changes, 0.005)
    return {
        "rate_centre": centre,
        "rate_scale": scale,
        "change_scale": change_scale,
        "rate_low": float(np.quantile(rates, 0.005)),
        "rate_high": float(np.quantile(rates, 0.995)),
        "rate_count": int(len(rates)),
        "change_count": int(len(changes)),
    }


def _node_rows(record):
    rows = np.r_[record["anchor"], record["evaluation"]]
    distance = record["md"][rows] - record["anchor_md"]
    targets = np.arange(0.0, float(distance[-1]) + 0.5, NODE_SPACING_FT)
    positions = np.searchsorted(distance, targets, side="left")
    positions = np.unique(np.clip(positions, 0, len(rows) - 1))
    if positions[0] != 0:
        positions = np.r_[0, positions]
    if positions[-1] != len(rows) - 1:
        positions = np.r_[positions, len(rows) - 1]
    return rows[positions]


def _reference_channels(record, mode, donor=None):
    source = record if donor is None else donor
    channels = source["tw_channels"]
    axis = source["tw_tvt"]
    if mode == "raw":
        shift = 0.0
    elif mode == "prefix_shift":
        expected = np.interp(
            record["tvt_input"][record["known"]],
            axis,
            channels[0],
            left=np.nan,
            right=np.nan,
        )
        observed = record["horizontal_channels"][0, record["known"]]
        valid = np.isfinite(expected) & np.isfinite(observed)
        shift = float(np.median(observed[valid] - expected[valid])) if valid.any() else 0.0
    else:
        raise ValueError(mode)
    return axis, channels + shift


def _emission(record, prior, config, donor=None, no_gr=False):
    rows = _node_rows(record)
    distance = record["md"][rows] - record["anchor_md"]
    centre_rate = float(
        np.clip(record["prefix_surface_rate"], prior["rate_low"], prior["rate_high"])
    )
    centre_surface = record["anchor_surface"] + centre_rate * distance
    state_surface = centre_surface[:, None] + OFFSETS_FT[None, :]
    state_tvt = state_surface - record["z"][rows, None]
    if no_gr:
        return rows, distance, state_surface, np.zeros_like(state_tvt)

    axis, reference = _reference_channels(record, config["reference"], donor)
    emission = np.zeros_like(state_tvt)
    total_weight = 0.0
    for channel, weight in enumerate(CHANNEL_WEIGHTS):
        expected = np.interp(
            state_tvt.ravel(), axis, reference[channel], left=np.nan, right=np.nan
        ).reshape(state_tvt.shape)
        observed = record["horizontal_channels"][channel, rows, None]
        residual = (observed - expected) / max(record["gamma"], 1e-6)
        square = np.minimum(residual * residual, 1e4)
        if config["likelihood"] == "cauchy":
            loss = np.log1p(square)
        elif config["likelihood"] == "gaussian":
            loss = 0.5 * np.minimum(square, 400.0)
        else:
            raise ValueError(config["likelihood"])
        loss[~np.isfinite(loss)] = 25.0
        emission += float(weight) * loss
        total_weight += float(weight)
    emission /= max(total_weight, 1e-9)
    emission -= np.min(emission, axis=1, keepdims=True)
    emission[0] = 0.0
    return rows, distance, state_surface, emission


def _transition_log(delta_md, prior, multiplier):
    # The deterministic centre already advances at rate_centre, so state
    # transitions model only departures from that raw-data prior path.
    sigma = max(
        2.0,
        float(multiplier) * float(prior["rate_scale"]) * max(float(delta_md), 1.0),
    )
    difference = OFFSETS_FT[None, :] - OFFSETS_FT[:, None]
    value = -0.5 * (difference / sigma) ** 2
    value -= logsumexp(value, axis=1, keepdims=True)
    return value


def _decode(record, prior, config, donor=None, no_gr=False, sample_paths=True):
    rows, distance, state_surface, costs = _emission(
        record, prior, config, donor=donor, no_gr=no_gr
    )
    log_emission = -costs / float(config["temperature"])
    nodes, states = log_emission.shape
    transitions = [None]
    for node in range(1, nodes):
        transitions.append(
            _transition_log(
                distance[node] - distance[node - 1],
                prior,
                config["transition_multiplier"],
            )
        )

    alpha = np.full((nodes, states), -np.inf, np.float64)
    alpha[0, ZERO_STATE] = 0.0
    for node in range(1, nodes):
        alpha[node] = log_emission[node] + logsumexp(
            alpha[node - 1, :, None] + transitions[node], axis=0
        )
        alpha[node] -= logsumexp(alpha[node])

    beta = np.zeros((nodes, states), np.float64)
    for node in range(nodes - 2, -1, -1):
        beta[node] = logsumexp(
            transitions[node + 1]
            + log_emission[node + 1][None, :]
            + beta[node + 1][None, :],
            axis=1,
        )
        beta[node] -= logsumexp(beta[node])
    log_marginal = alpha + beta
    log_marginal -= logsumexp(log_marginal, axis=1, keepdims=True)
    marginal = np.exp(log_marginal)

    viterbi = np.full((nodes, states), -np.inf, np.float64)
    backpointer = np.zeros((nodes, states), np.int16)
    viterbi[0, ZERO_STATE] = 0.0
    for node in range(1, nodes):
        candidate = viterbi[node - 1, :, None] + transitions[node]
        backpointer[node] = np.argmax(candidate, axis=0).astype(np.int16)
        viterbi[node] = log_emission[node] + np.max(candidate, axis=0)
    map_states = np.empty(nodes, np.int16)
    map_states[-1] = int(np.argmax(viterbi[-1]))
    for node in range(nodes - 2, -1, -1):
        map_states[node] = backpointer[node + 1, map_states[node + 1]]

    mean_surface = np.sum(marginal * state_surface, axis=1)
    map_surface = state_surface[np.arange(nodes), map_states]
    sample_surface = None
    if sample_paths:
        digest = hashlib.sha256(f"{SEED}:{record['well']}:{config}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        sampled_states = np.empty((PATH_SAMPLES, nodes), np.int16)
        terminal = np.exp(alpha[-1] - logsumexp(alpha[-1]))
        sampled_states[:, -1] = rng.choice(states, size=PATH_SAMPLES, p=terminal)
        for node in range(nodes - 2, -1, -1):
            for sample in range(PATH_SAMPLES):
                next_state = int(sampled_states[sample, node + 1])
                probability = alpha[node] + transitions[node + 1][:, next_state]
                probability = np.exp(probability - logsumexp(probability))
                sampled_states[sample, node] = rng.choice(states, p=probability)
        sample_surface = state_surface[np.arange(nodes)[None, :], sampled_states]

    entropy = -np.sum(marginal * np.log(np.maximum(marginal, 1e-12)), axis=1)
    entropy /= math.log(states)
    offset_mean = marginal @ OFFSETS_FT
    offset_std = np.sqrt(
        np.maximum(marginal @ (OFFSETS_FT ** 2) - offset_mean * offset_mean, 0.0)
    )
    node_surface = {
        "map": map_surface,
        "posterior_mean": mean_surface,
        "sample_median": (
            np.median(sample_surface, axis=0) if sample_surface is not None else mean_surface
        ),
    }
    predictions = {}
    eval_rows = record["evaluation"]
    for estimator, surface in node_surface.items():
        interpolated = np.interp(record["md"][eval_rows], record["md"][rows], surface)
        predictions[estimator] = interpolated - record["z"][eval_rows]
    diagnostics = {
        "mean_entropy": float(np.mean(entropy[1:])),
        "mean_offset_std": float(np.mean(offset_std[1:])),
        "terminal_offset_std": float(offset_std[-1]),
        "nodes": int(nodes),
    }
    return predictions, diagnostics


def _metric_rows(records, decoded, variant):
    rows = []
    for record in records:
        predictions, diagnostics = decoded[record["well"]]
        truth = record["truth"][record["evaluation"]]
        distance = record["md"][record["evaluation"]] - record["anchor_md"]
        for estimator, prediction in predictions.items():
            error = prediction - truth
            far = distance > 2000.0
            rows.append(
                {
                    "well": record["well"],
                    "variant": variant,
                    "estimator": estimator,
                    "rows": int(len(error)),
                    "sse": float(np.dot(error, error)),
                    "sae": float(np.sum(np.abs(error))),
                    "far_rows": int(far.sum()),
                    "far_sse": float(np.dot(error[far], error[far])) if far.any() else 0.0,
                    **diagnostics,
                }
            )
    return pd.DataFrame(rows)


def _summary(bywell):
    output = []
    for (variant, estimator), local in bywell.groupby(["variant", "estimator"], sort=False):
        rows = int(local["rows"].sum())
        far_rows = int(local["far_rows"].sum())
        well_rmse = np.sqrt(local["sse"] / local["rows"])
        output.append(
            {
                "variant": variant,
                "estimator": estimator,
                "rows": rows,
                "wells": int(len(local)),
                "exact_sse": float(local["sse"].sum()),
                "rmse": float(np.sqrt(local["sse"].sum() / rows)),
                "mae": float(local["sae"].sum() / rows),
                "median_well_rmse": float(np.median(well_rmse)),
                "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
                "far_gt2000_rmse": (
                    float(np.sqrt(local["far_sse"].sum() / far_rows))
                    if far_rows else np.nan
                ),
                "mean_entropy": float(local["mean_entropy"].mean()),
                "mean_offset_std": float(local["mean_offset_std"].mean()),
            }
        )
    return pd.DataFrame(output)


def _decode_many(records, prior, config, label, donors=None, no_gr=False):
    output = {}
    for number, record in enumerate(records, 1):
        donor = None if donors is None else donors[record["well"]]
        output[record["well"]] = _decode(
            record, prior, config, donor=donor, no_gr=no_gr, sample_paths=True
        )
        if number % 25 == 0 or number == len(records):
            print(f"{label} {number}/{len(records)}", flush=True)
    return output


def _config_id(config):
    return (
        f"{config['reference']}_{config['likelihood']}"
        f"_t{config['temperature']:g}_s{config['transition_multiplier']:g}"
    )


started = time.time()
all_wells = [
    path.name.split("__", 1)[0]
    for path in sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
]
if FOLD_PATH is not None:
    folds = pd.read_csv(FOLD_PATH, dtype={"well": str})[["well", "fold"]]
    folds = folds[folds["well"].isin(all_wells)].drop_duplicates("well")
else:
    folds = _stable_folds(all_wells)
folds.to_csv(OUT_DIR / "folds.csv", index=False)
fold_by_well = dict(zip(folds["well"], folds["fold"]))

records = {}
failures = []
for number, well in enumerate(sorted(fold_by_well), 1):
    try:
        records[well] = _load_record(well)
    except Exception as exc:
        failures.append({"well": well, "fold": int(fold_by_well[well]), "reason": repr(exc)})
    if number % 50 == 0 or number == len(fold_by_well):
        print(f"V30 load raw {number}/{len(fold_by_well)} failures={len(failures)}", flush=True)
pd.DataFrame(failures, columns=["well", "fold", "reason"]).to_csv(
    OUT_DIR / "record_failures.csv", index=False
)


def _records_for(fold_numbers):
    return [
        records[well]
        for well in sorted(records)
        if int(fold_by_well[well]) in set(fold_numbers)
    ]


prior_records = _records_for(PRIOR_FOLDS)
selection_records = _records_for((SELECTION_FOLD,))
application_records = _records_for((APPLICATION_FOLD,))
final_records = _records_for((FINAL_FOLD,))
if min(len(selection_records), len(application_records), len(final_records)) < 145:
    raise RuntimeError(
        "Insufficient held-out coverage: "
        f"{len(selection_records)}/{len(application_records)}/{len(final_records)}"
    )
prior = _fit_surface_prior(prior_records)
_write_json(OUT_DIR / "outer_train_surface_prior.json", prior)
print("V30 OUTER PRIOR", prior, flush=True)

configs = []
for reference in REFERENCE_MODES:
    for temperature in TEMPERATURES:
        for transition_multiplier in TRANSITION_MULTIPLIERS:
            configs.append(
                {
                    "reference": reference,
                    "likelihood": "cauchy",
                    "temperature": float(temperature),
                    "transition_multiplier": float(transition_multiplier),
                }
            )

discovery = sorted(
    selection_records,
    key=lambda record: hashlib.sha256(f"v30:{record['well']}".encode("ascii")).digest(),
)[:DISCOVERY_WELLS]
discovery_rows = []
for config in configs:
    config_id = _config_id(config)
    decoded = _decode_many(discovery, prior, config, f"V30 discovery {config_id}")
    summary = _summary(_metric_rows(discovery, decoded, config_id))
    discovery_rows.append(summary)
discovery_scan = pd.concat(discovery_rows, ignore_index=True).sort_values("rmse")
discovery_scan.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
print("\nV30 DISCOVERY TOP", flush=True)
print(discovery_scan.head(20).to_string(index=False), flush=True)

shortlist = []
for _, row in discovery_scan.iterrows():
    config_id = str(row["variant"])
    if config_id not in shortlist:
        shortlist.append(config_id)
    if len(shortlist) >= 3:
        break
config_by_id = {_config_id(config): config for config in configs}

confirmation_rows = []
confirmation_predictions = {}
for config_id in shortlist:
    config = config_by_id[config_id]
    decoded = _decode_many(
        selection_records, prior, config, f"V30 Fold3 confirm {config_id}"
    )
    confirmation_predictions[config_id] = decoded
    confirmation_rows.append(_summary(_metric_rows(selection_records, decoded, config_id)))
confirmation_scan = pd.concat(confirmation_rows, ignore_index=True).sort_values("rmse")
confirmation_scan.to_csv(OUT_DIR / "fold3_confirmation_scan.csv", index=False)
print("\nV30 FOLD3 CONFIRMATION", flush=True)
print(confirmation_scan.to_string(index=False), flush=True)

selected_row = confirmation_scan.iloc[0]
selected_id = str(selected_row["variant"])
selected_estimator = str(selected_row["estimator"])
selected_config = config_by_id[selected_id]
print(
    "V30 SELECTED",
    {"config": selected_config, "estimator": selected_estimator},
    flush=True,
)

fold2_decoded = _decode_many(
    application_records, prior, selected_config, "V30 Fold2 full"
)
fold1_decoded = _decode_many(final_records, prior, selected_config, "V30 Fold1 final")

no_gr_decoded = _decode_many(
    application_records, prior, selected_config, "V30 Fold2 no-GR", no_gr=True
)
ordered_fold2 = sorted(application_records, key=lambda record: record["well"])
donors = {
    record["well"]: ordered_fold2[(index + 1) % len(ordered_fold2)]
    for index, record in enumerate(ordered_fold2)
}
shuffled_decoded = _decode_many(
    application_records,
    prior,
    selected_config,
    "V30 Fold2 shuffled-typewell",
    donors=donors,
)

# Gaussian is a matched likelihood ablation, not a searched recipe.
gaussian_config = dict(selected_config)
gaussian_config["likelihood"] = "gaussian"
gaussian_decoded = _decode_many(
    application_records, prior, gaussian_config, "V30 Fold2 Gaussian"
)

fold3_selected = _summary(
    _metric_rows(selection_records, confirmation_predictions[selected_id], "fold3_selected")
)
fold2_full = _summary(_metric_rows(application_records, fold2_decoded, "fold2_full"))
fold1_full = _summary(_metric_rows(final_records, fold1_decoded, "fold1_full"))
fold2_no_gr = _summary(_metric_rows(application_records, no_gr_decoded, "no_gr"))
fold2_shuffled = _summary(
    _metric_rows(application_records, shuffled_decoded, "shuffled_typewell")
)
fold2_gaussian = _summary(
    _metric_rows(application_records, gaussian_decoded, "gaussian")
)
all_summary = pd.concat(
    [fold3_selected, fold2_full, fold1_full, fold2_no_gr, fold2_shuffled, fold2_gaussian],
    ignore_index=True,
)
all_summary.to_csv(OUT_DIR / "summary.csv", index=False)


def _selected(summary, variant):
    row = summary[
        summary["variant"].eq(variant) & summary["estimator"].eq(selected_estimator)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Missing selected estimator for {variant}")
    return row.iloc[0]


fold2_metric = _selected(all_summary, "fold2_full")
fold1_metric = _selected(all_summary, "fold1_full")
no_gr_metric = _selected(all_summary, "no_gr")
shuffled_metric = _selected(all_summary, "shuffled_typewell")

# Persist a self-contained raw endpoint OOF for later correlation auditing.
oof_parts = []
for fold, fold_records, decoded in (
    (3, selection_records, confirmation_predictions[selected_id]),
    (2, application_records, fold2_decoded),
    (1, final_records, fold1_decoded),
):
    for record in fold_records:
        prediction = decoded[record["well"]][0][selected_estimator]
        eval_rows = record["evaluation"]
        oof_parts.append(
            pd.DataFrame(
                {
                    "well": record["well"],
                    "fold": np.full(len(eval_rows), fold, np.int8),
                    "row_idx": eval_rows.astype(np.int32),
                    "md_since": record["md"][eval_rows] - record["anchor_md"],
                    "tvt_true": record["truth"][eval_rows],
                    "prediction": prediction.astype(np.float32),
                }
            )
        )
pd.concat(oof_parts, ignore_index=True).to_parquet(
    OUT_DIR / "standalone_oof.parquet", index=False, compression="zstd"
)

gates = {
    "fold2_rmse_le_11p5": bool(fold2_metric["rmse"] <= 11.5),
    "fold1_rmse_le_11p5": bool(fold1_metric["rmse"] <= 11.5),
    "fold2_far_le_14": bool(fold2_metric["far_gt2000_rmse"] <= 14.0),
    "fold1_far_le_14": bool(fold1_metric["far_gt2000_rmse"] <= 14.0),
    "no_gr_degradation_ge_1": bool(no_gr_metric["rmse"] - fold2_metric["rmse"] >= 1.0),
    "shuffled_typewell_degradation_ge_1": bool(
        shuffled_metric["rmse"] - fold2_metric["rmse"] >= 1.0
    ),
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "raw_absolute_surface_stochastic_alignment_standalone_gateway",
    "literature_direction": {
        "stochastic_dtw": "10.30632/pjv63n6-2022a11",
        "implemented_principle": "posterior concentration and sampled correlation paths",
    },
    "raw_inputs_only": True,
    "historical_prediction_dependencies": [],
    "forbidden_inputs_confirmed_absent": [
        "core", "CP8", "Neighbor", "SP45", "FLE", "Ridge", "Gold", "hidden predictions"
    ],
    "state_definition": (
        "anchor_surface + clipped visible-prefix surface rate * distance + absolute_offset; "
        "rate bounds come from outer train; TVT = surface - Z"
    ),
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "application_fold": APPLICATION_FOLD,
        "final_fold": FINAL_FOLD,
        "discovery_wells": len(discovery),
    },
    "coverage": {
        "prior_wells": len(prior_records),
        "selection_wells": len(selection_records),
        "application_wells": len(application_records),
        "final_wells": len(final_records),
        "failed_wells": len(failures),
    },
    "lattice": {
        "node_spacing_ft": NODE_SPACING_FT,
        "offset_min_ft": float(OFFSETS_FT[0]),
        "offset_max_ft": float(OFFSETS_FT[-1]),
        "offset_step_ft": float(OFFSETS_FT[1] - OFFSETS_FT[0]),
        "states": len(OFFSETS_FT),
        "posterior_path_samples": PATH_SAMPLES,
    },
    "outer_train_prior": prior,
    "selected_config": selected_config,
    "selected_estimator": selected_estimator,
    "fold3_selected": _selected(all_summary, "fold3_selected").to_dict(),
    "fold2": fold2_metric.to_dict(),
    "fold1": fold1_metric.to_dict(),
    "controls": {
        "no_gr": no_gr_metric.to_dict(),
        "shuffled_typewell": shuffled_metric.to_dict(),
        "gaussian": _selected(all_summary, "gaussian").to_dict(),
    },
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "free_gb": float(shutil.disk_usage(OUT_DIR).free / (1024 ** 3)),
    "next": (
        "PASS -> audit fold-clean error correlation and fixed blend with the 6.377 incumbent"
        if all(gates.values())
        else "FAIL -> diagnose absolute state coverage/emission; do not route back to CP or Neighbor"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV30 SUMMARY", flush=True)
print(all_summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT", flush=True)
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
