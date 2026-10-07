"""Generate a genuinely new multi-seed PF family for Track6 reproduction.

Twenty-four deterministic Fold1 wells select PF dynamics, GR calibration,
likelihood, and seed aggregation. The remaining Fold1 wells are untouched
confirmation. This is a generator gateway, not a final CV score.
"""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from numba import njit


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
TRACK6_PATH = OOF_ROOT / "track6_independent_pf_gbdt_gateway_v1" / "fold1_track6_oof.pkl"
OUT_DIR = OOF_ROOT / "track6_multiseed_pf_generator_gateway_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DISCOVERY_WELLS = 24
N_PARTICLES = 256
N_SEEDS = 16
TOP_CONFIGS_CONFIRM = 2
ORACLE_MEAN_GATE = 5.30
COSINE_GATE = 0.50
FIXED_MEAN_GATE = 6.40


# name, GR_calibration_mode, likelihood_mode, init_spread, momentum, rate_noise,
# position_noise, rough_position, rough_rate, resample_fraction
# calibration: 0=raw, 1=robust offset, 2=affine, 3=50% raw/affine blend.
CONFIGS = [
    ("raw_gauss_standard", 0, 0, 4.5, 0.998, 0.0020, 0.005, 0.10, 0.0010, 0.50),
    ("offset_gauss_standard", 1, 0, 4.5, 0.998, 0.0020, 0.005, 0.10, 0.0010, 0.50),
    ("affine_gauss_standard", 2, 0, 4.5, 0.998, 0.0020, 0.005, 0.10, 0.0010, 0.50),
    ("offset_cauchy_standard", 1, 1, 4.5, 0.998, 0.0020, 0.005, 0.10, 0.0010, 0.50),
    ("blend50_cauchy_standard", 3, 1, 4.5, 0.998, 0.0020, 0.005, 0.10, 0.0010, 0.50),
    ("affine_cauchy_standard", 2, 1, 4.5, 0.998, 0.0020, 0.005, 0.10, 0.0010, 0.50),
    ("blend50_cauchy_wide", 3, 1, 12.0, 0.997, 0.0030, 0.010, 0.20, 0.0020, 0.40),
    ("offset_cauchy_rough", 1, 1, 8.0, 0.993, 0.0060, 0.020, 0.30, 0.0040, 0.35),
]
AGGREGATIONS = ("mean", "median", "temp3", "temp12")


@njit(cache=False)
def _pf_seed(
    md,
    z,
    gr,
    tw_tvt,
    tw_gr,
    last_md,
    last_z,
    last_tvt,
    init_rate,
    gr_sigma,
    particles,
    seed,
    likelihood_mode,
    init_spread,
    momentum,
    rate_noise,
    position_noise,
    rough_position,
    rough_rate,
    resample_fraction,
):
    np.random.seed(seed)
    n_rows = len(md)
    pos = last_tvt + last_z + init_spread * np.random.randn(particles)
    rate = init_rate + 0.01 * np.random.randn(particles)
    weights = np.ones(particles) / particles
    output = np.empty(n_rows)
    previous_md = last_md
    log_likelihood = 0.0

    for row in range(n_rows):
        delta_md = md[row] - previous_md
        if delta_md < 1.0:
            delta_md = 1.0
        rate = momentum * rate + rate_noise * np.random.randn(particles)
        pos = pos + rate * delta_md + position_noise * np.random.randn(particles)
        tvt_particles = pos - z[row]
        lower = tw_tvt[0] - 100.0
        upper = tw_tvt[-1] + 100.0
        for particle in range(particles):
            if tvt_particles[particle] < lower:
                tvt_particles[particle] = lower
            elif tvt_particles[particle] > upper:
                tvt_particles[particle] = upper
        pos = tvt_particles + z[row]

        expected = np.interp(tvt_particles, tw_tvt, tw_gr)
        likelihood = np.empty(particles)
        for particle in range(particles):
            difference = (gr[row] - expected[particle]) / gr_sigma
            if likelihood_mode == 0:
                square = difference * difference
                likelihood[particle] = np.exp(-0.5 * min(square, 600.0))
            else:
                likelihood[particle] = 1.0 / (1.0 + difference * difference)
            if likelihood[particle] < 1e-300:
                likelihood[particle] = 1e-300

        average_likelihood = 0.0
        for particle in range(particles):
            average_likelihood += weights[particle] * likelihood[particle]
        log_likelihood += np.log(max(average_likelihood, 1e-300))

        total = 0.0
        for particle in range(particles):
            weights[particle] *= likelihood[particle]
            total += weights[particle]
        if total > 0.0:
            weights /= total
        else:
            weights[:] = 1.0 / particles

        inverse_effective = 0.0
        for particle in range(particles):
            inverse_effective += weights[particle] * weights[particle]
        effective = 1.0 / max(inverse_effective, 1e-300)
        if effective < resample_fraction * particles:
            cumulative = np.cumsum(weights)
            start = np.random.random() / particles
            indices = np.empty(particles, np.int64)
            cursor = 0
            for particle in range(particles):
                point = start + particle / particles
                while cursor < particles - 1 and cumulative[cursor] < point:
                    cursor += 1
                indices[particle] = cursor
            pos = pos[indices] + rough_position * np.random.randn(particles)
            rate = rate[indices] + rough_rate * np.random.randn(particles)
            weights[:] = 1.0 / particles

        estimate = 0.0
        for particle in range(particles):
            estimate += weights[particle] * (pos[particle] - z[row])
        output[row] = estimate
        previous_md = md[row]
    return output, log_likelihood


def _seed_for(well, config, seed):
    token = f"{well}|{config}|{seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little") % (2**31 - 1)


def _rmse(truth, prediction):
    error = np.asarray(prediction, float) - np.asarray(truth, float)
    return float(np.sqrt(np.mean(error * error)))


def _affine_typewell(horizontal, tw_tvt, tw_gr):
    known = horizontal[horizontal["TVT_input"].notna() & horizontal["GR"].notna()]
    if len(known) < 30:
        return tw_gr.copy()
    reference = np.interp(known["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    observed = known["GR"].to_numpy(float)
    if np.std(reference) < 1e-6:
        return tw_gr + float(np.mean(observed) - np.mean(reference))
    slope, intercept = np.polyfit(reference, observed, 1)
    slope = float(np.clip(slope, 0.50, 1.50))
    return slope * tw_gr + float(intercept)


def _offset_typewell(horizontal, tw_tvt, tw_gr):
    known = horizontal[horizontal["TVT_input"].notna() & horizontal["GR"].notna()]
    if len(known) < 10:
        return tw_gr.copy()
    reference = np.interp(known["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    observed = known["GR"].to_numpy(float)
    # A median offset is robust to local bed mismatch in the known section.
    offset = float(np.median(observed - reference))
    return tw_gr + offset


def _prepare_well(well, oof_well):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv").sort_values("TVT")
    known = horizontal[horizontal["TVT_input"].notna()]
    hidden_mask = horizontal["TVT_input"].isna().to_numpy()
    hidden_indices = np.flatnonzero(hidden_mask)
    ids = np.asarray([f"{well}_{index}" for index in hidden_indices], object)
    order = pd.DataFrame({"id": ids}).merge(
        oof_well[["id", "tvt_true", "sp45"]], on="id", how="left", validate="one_to_one"
    )
    if order[["tvt_true", "sp45"]].isna().any().any():
        raise RuntimeError(f"OOF merge failed for {well}")

    tw_tvt = typewell["TVT"].to_numpy(np.float64)
    tw_gr_raw = (
        typewell["GR"].interpolate(limit_direction="both").fillna(typewell["GR"].mean())
        .to_numpy(np.float64)
    )
    tw_gr_affine = _affine_typewell(horizontal, tw_tvt, tw_gr_raw)
    tw_gr_offset = _offset_typewell(horizontal, tw_tvt, tw_gr_raw)
    tw_gr_blend50 = 0.5 * tw_gr_raw + 0.5 * tw_gr_affine
    gr_all = (
        horizontal["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tw_gr_raw))
        .to_numpy(np.float64)
    )
    known_gr = known["GR"].to_numpy(float)
    reference_known = np.interp(known["TVT_input"].to_numpy(float), tw_tvt, tw_gr_affine)
    residual = known_gr - reference_known
    residual = residual[np.isfinite(residual)]
    gr_sigma = float(np.clip(np.std(residual), 10.0, 60.0)) if len(residual) else 20.0
    if not np.isfinite(gr_sigma):
        gr_sigma = 20.0
    tail = known.tail(30)
    delta_tvt = np.diff(tail["TVT_input"].to_numpy(float))
    delta_z = np.diff(tail["Z"].to_numpy(float))
    delta_md = np.diff(tail["MD"].to_numpy(float))
    valid = delta_md > 0
    init_rate = float(np.median((delta_tvt + delta_z)[valid] / delta_md[valid])) if valid.sum() >= 3 else 0.0
    last = known.iloc[-1]
    return {
        "well": well,
        "id": ids,
        "truth": order["tvt_true"].to_numpy(np.float64),
        "sp45": order["sp45"].to_numpy(np.float64),
        "md": horizontal.loc[hidden_mask, "MD"].to_numpy(np.float64),
        "z": horizontal.loc[hidden_mask, "Z"].to_numpy(np.float64),
        "gr": gr_all[hidden_mask],
        "tw_tvt": tw_tvt,
        "tw_gr_raw": tw_gr_raw,
        "tw_gr_offset": tw_gr_offset,
        "tw_gr_affine": tw_gr_affine,
        "tw_gr_blend50": tw_gr_blend50,
        "last_md": float(last["MD"]),
        "last_z": float(last["Z"]),
        "last_tvt": float(last["TVT_input"]),
        "init_rate": init_rate,
        "gr_sigma": gr_sigma,
    }


def _run_config(payload, config):
    (
        name,
        calibration_mode,
        likelihood_mode,
        init_spread,
        momentum,
        rate_noise,
        position_noise,
        rough_position,
        rough_rate,
        resample_fraction,
    ) = config
    if calibration_mode == 1:
        tw_gr = payload["tw_gr_offset"]
    elif calibration_mode == 2:
        tw_gr = payload["tw_gr_affine"]
    elif calibration_mode == 3:
        tw_gr = payload["tw_gr_blend50"]
    else:
        tw_gr = payload["tw_gr_raw"]
    paths = []
    likelihoods = []
    for seed in range(N_SEEDS):
        path, score = _pf_seed(
            payload["md"],
            payload["z"],
            payload["gr"],
            payload["tw_tvt"],
            tw_gr,
            payload["last_md"],
            payload["last_z"],
            payload["last_tvt"],
            payload["init_rate"],
            payload["gr_sigma"],
            N_PARTICLES,
            _seed_for(payload["well"], name, seed),
            likelihood_mode,
            init_spread,
            momentum,
            rate_noise,
            position_noise,
            rough_position,
            rough_rate,
            resample_fraction,
        )
        paths.append(path)
        likelihoods.append(score)
    paths = np.stack(paths)
    likelihoods = np.asarray(likelihoods, float)
    finite_likelihood = np.isfinite(likelihoods)
    if not finite_likelihood.all():
        fallback = float(np.min(likelihoods[finite_likelihood])) if finite_likelihood.any() else 0.0
        likelihoods = np.where(finite_likelihood, likelihoods, fallback - 100.0)
    output = {
        "mean": np.mean(paths, axis=0),
        "median": np.median(paths, axis=0),
    }
    for temperature in (3.0, 12.0):
        logits = (likelihoods - np.max(likelihoods)) / temperature
        weights = np.exp(np.clip(logits, -80.0, 0.0))
        weights /= np.sum(weights)
        output[f"temp{int(temperature)}"] = np.sum(weights[:, None] * paths, axis=0)
    return output


def _metrics(wells, truth, prediction):
    error = prediction - truth
    frame = pd.DataFrame({"well": wells, "error2": error * error})
    per_well = frame.groupby("well", sort=False)["error2"].mean().pow(0.5)
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mean_well_rmse": float(per_well.mean()),
        "median_well_rmse": float(per_well.median()),
        "p95_well_rmse": float(per_well.quantile(0.95)),
    }


def _pair(wells, truth, base, candidate):
    base_error = base - truth
    candidate_error = candidate - truth
    cosine = float(
        np.dot(base_error, candidate_error)
        / np.sqrt(np.dot(base_error, base_error) * np.dot(candidate_error, candidate_error))
    )
    work = pd.DataFrame(
        {"well": wells, "truth": truth, "base": base, "candidate": candidate}
    )
    fixed = []
    for weight in np.linspace(0.0, 1.0, 101):
        prediction = base + weight * (candidate - base)
        score = _metrics(wells, truth, prediction)
        fixed.append((weight, score["rmse"], score["mean_well_rmse"]))
    fixed_best = min(fixed, key=lambda row: (row[2], row[1]))
    oracle = np.empty(len(work), float)
    oracle_weights = []
    for well, indices in work.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, int)
        direction = candidate[indices] - base[indices]
        denominator = float(np.dot(direction, direction))
        weight = 0.0
        if denominator > 1e-12:
            weight = float(np.clip(-np.dot(base_error[indices], direction) / denominator, 0.0, 1.0))
        oracle[indices] = base[indices] + weight * direction
        oracle_weights.append(weight)
    oracle_score = _metrics(wells, truth, oracle)
    return {
        "error_cosine": cosine,
        "fixed_weight_candidate": float(fixed_best[0]),
        "fixed_rmse": float(fixed_best[1]),
        "fixed_mean_well_rmse": float(fixed_best[2]),
        "oracle_rmse": oracle_score["rmse"],
        "oracle_mean_well_rmse": oracle_score["mean_well_rmse"],
        "oracle_p95_well_rmse": oracle_score["p95_well_rmse"],
        "mean_oracle_weight_candidate": float(np.mean(oracle_weights)),
    }


def _evaluate(payloads, predictions):
    wells = np.concatenate(
        [np.repeat(payload["well"], len(payload["truth"])) for payload in payloads]
    )
    truth = np.concatenate([payload["truth"] for payload in payloads])
    sp45 = np.concatenate([payload["sp45"] for payload in payloads])
    rows = []
    for name, pieces in predictions.items():
        candidate = np.concatenate(pieces)
        rows.append(
            {
                "variant": name,
                **_metrics(wells, truth, candidate),
                **_pair(wells, truth, sp45, candidate),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["oracle_mean_well_rmse", "error_cosine", "fixed_mean_well_rmse"],
        ignore_index=True,
    )


started = time.time()
oof = pd.read_pickle(TRACK6_PATH)
oof["id"] = oof["id"].astype(str)
well_sizes = oof.groupby("well", sort=False).size().sort_values()
ordered_wells = well_sizes.index.astype(str).tolist()
discovery_positions = np.unique(
    np.round(np.linspace(0, len(ordered_wells) - 1, DISCOVERY_WELLS)).astype(int)
)
discovery_set = {ordered_wells[index] for index in discovery_positions}
discovery_wells = [well for well in ordered_wells if well in discovery_set]
confirmation_wells = [well for well in ordered_wells if well not in discovery_set]

# Numba compilation smoke test.
_pf_seed(
    np.arange(8, dtype=np.float64),
    np.zeros(8, dtype=np.float64),
    np.ones(8, dtype=np.float64) * 80.0,
    np.linspace(10000, 10100, 32),
    np.linspace(60, 100, 32),
    -1.0,
    0.0,
    10050.0,
    0.0,
    20.0,
    16,
    123,
    1,
    4.5,
    0.998,
    0.002,
    0.005,
    0.1,
    0.001,
    0.5,
)

print(
    "MULTISEED PF CONFIG",
    {
        "discovery_wells": len(discovery_wells),
        "confirmation_wells": len(confirmation_wells),
        "configs": len(CONFIGS),
        "seeds": N_SEEDS,
        "particles": N_PARTICLES,
    },
    flush=True,
)

discovery_payloads = []
discovery_predictions = {f"{config[0]}__{agg}": [] for config in CONFIGS for agg in AGGREGATIONS}
for number, well in enumerate(discovery_wells, start=1):
    payload = _prepare_well(well, oof[oof["well"].astype(str).eq(well)])
    discovery_payloads.append(payload)
    for config in CONFIGS:
        output = _run_config(payload, config)
        for aggregation, path in output.items():
            discovery_predictions[f"{config[0]}__{aggregation}"].append(path)
    print(f"discovery PF {number}/{len(discovery_wells)}", flush=True)

discovery_report = _evaluate(discovery_payloads, discovery_predictions)
discovery_report.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
eligible = discovery_report[
    discovery_report["mean_well_rmse"].le(15.0)
    & discovery_report["fixed_weight_candidate"].ge(0.05)
]
if not len(eligible):
    fail_verdict = {
        "experiment": "track6_multiseed_pf_generator_gateway_v3",
        "stage": "24_well_discovery_fail_fast",
        "generator": {
            "particles": N_PARTICLES,
            "seeds": N_SEEDS,
            "discovery_configs": len(CONFIGS),
        },
        "best_discovery": discovery_report.iloc[0].to_dict(),
        "strict_pass": False,
        "next": "FAIL -> no deployable discovery endpoint; stop before 131-well confirmation",
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    (OUT_DIR / "verdict.json").write_text(
        json.dumps(fail_verdict, indent=2), encoding="utf-8"
    )
    print("\nDISCOVERY FAIL-FAST")
    print(json.dumps(fail_verdict, indent=2), flush=True)
    raise RuntimeError(json.dumps(fail_verdict, indent=2))
ranking = eligible
selected_variants = ranking.head(8)["variant"].tolist()
selected_configs = []
for variant in selected_variants:
    config_name = variant.split("__", 1)[0]
    if config_name not in selected_configs:
        selected_configs.append(config_name)
    if len(selected_configs) >= TOP_CONFIGS_CONFIRM:
        break
selected_config_map = {config[0]: config for config in CONFIGS if config[0] in selected_configs}
selected_variants = [
    variant for variant in selected_variants if variant.split("__", 1)[0] in selected_config_map
]

print("DISCOVERY TOP")
print(discovery_report.head(20).to_string(index=False), flush=True)
print("SELECTED CONFIGS", selected_configs, flush=True)

confirmation_payloads = []
confirmation_predictions = {variant: [] for variant in selected_variants}
for number, well in enumerate(confirmation_wells, start=1):
    payload = _prepare_well(well, oof[oof["well"].astype(str).eq(well)])
    confirmation_payloads.append(payload)
    outputs = {
        name: _run_config(payload, config)
        for name, config in selected_config_map.items()
    }
    for variant in selected_variants:
        config_name, aggregation = variant.split("__", 1)
        confirmation_predictions[variant].append(outputs[config_name][aggregation])
    if number % 20 == 0 or number == len(confirmation_wells):
        print(f"confirmation PF {number}/{len(confirmation_wells)}", flush=True)

confirmation_report = _evaluate(confirmation_payloads, confirmation_predictions)
confirmation_report.to_csv(OUT_DIR / "confirmation_scan.csv", index=False)
passing = confirmation_report[
    confirmation_report["error_cosine"].le(COSINE_GATE)
    & confirmation_report["oracle_mean_well_rmse"].le(ORACLE_MEAN_GATE)
    & confirmation_report["fixed_mean_well_rmse"].le(FIXED_MEAN_GATE)
]

ids = np.concatenate([payload["id"] for payload in confirmation_payloads])
export = pd.DataFrame(
    {
        "id": ids,
        "well": np.concatenate(
            [np.repeat(payload["well"], len(payload["id"])) for payload in confirmation_payloads]
        ),
        "tvt_true": np.concatenate([payload["truth"] for payload in confirmation_payloads]),
        "sp45": np.concatenate([payload["sp45"] for payload in confirmation_payloads]),
    }
)
for variant, pieces in confirmation_predictions.items():
    export[variant] = np.concatenate(pieces).astype(np.float32)
export.to_pickle(OUT_DIR / "confirmation_multiseed_pf_oof.pkl")

best = confirmation_report.iloc[0].to_dict() if len(confirmation_report) else None
verdict = {
    "experiment": "track6_multiseed_pf_generator_gateway_v3",
    "stage": "24_well_discovery_to_131_well_confirmation",
    "generator": {
        "particles": N_PARTICLES,
        "seeds": N_SEEDS,
        "discovery_configs": len(CONFIGS),
        "selected_configs": selected_configs,
    },
    "gates": {
        "error_cosine_le": COSINE_GATE,
        "two_way_oracle_mean_well_rmse_le": ORACLE_MEAN_GATE,
        "fixed_blend_mean_well_rmse_le": FIXED_MEAN_GATE,
    },
    "best_confirmation": best,
    "passing_variants": int(len(passing)),
    "strict_pass": bool(len(passing) > 0),
    "next": (
        "PASS -> generate five-fold OOF features and train the Track6 GBDT ensemble"
        if len(passing) > 0
        else "FAIL -> this reconstructed multi-seed PF still lacks the published Track6 endpoint"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nCONFIRMATION RESULTS")
print(confirmation_report.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
