import argparse
import datetime as _dt
import gc
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from numba import njit
except ImportError:
    njit = None

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    gaussian_filter1d = None


PF_HEATMAP_CHANNELS = {"pf_prob", "pf_mean_abs_diff"}


PF_PROFILE_CFG_KEY_MAP = {
    "PF_heatmap_lik_scale": "lik_scale",
    "PF_heatmap_ref_grid_step": "ref_grid_step",
    "PF_heatmap_query_gr_mode": "query_gr_mode",
    "PF_heatmap_seen_blend_weight": "seen_blend_weight",
    "PF_heatmap_typewell_gr_calibration": "typewell_gr_calibration",
    "PF_heatmap_raw_ref_likelihood_weight": "raw_ref_likelihood_weight",
    "PF_heatmap_gr_ambiguity_power": "gr_ambiguity_power",
    "PF_heatmap_gr_ambiguity_min_power": "gr_ambiguity_min_power",
    "PF_heatmap_gr_ambiguity_contrast": "gr_ambiguity_contrast",
    "PF_heatmap_gr_ambiguity_ref_mode": "gr_ambiguity_ref_mode",
    "PF_heatmap_gr_information_power": "gr_information_power",
    "PF_heatmap_gr_information_center": "gr_information_center",
    "PF_heatmap_gr_information_min_multiplier": "gr_information_min_multiplier",
    "PF_heatmap_gr_information_max_multiplier": "gr_information_max_multiplier",
    "PF_heatmap_gr_information_slope_weight": "gr_information_slope_weight",
    "PF_heatmap_gr_information_ref_mode": "gr_information_ref_mode",
    "PF_heatmap_gr_shape_power": "gr_shape_power",
    "PF_heatmap_gr_shape_mode": "gr_shape_mode",
    "PF_heatmap_gr_shape_window": "gr_shape_window",
    "PF_heatmap_gr_shape_min_points": "gr_shape_min_points",
    "PF_heatmap_gr_shape_sigma_floor": "gr_shape_sigma_floor",
    "PF_heatmap_gr_shape_ref_mode": "gr_shape_ref_mode",
    "PF_heatmap_missing_likelihood_power": "missing_likelihood_power",
    "PF_heatmap_missing_gap_decay": "missing_gap_decay",
    "PF_heatmap_missing_min_power": "missing_min_power",
    "PF_heatmap_outlier_prob": "outlier_prob",
    "PF_heatmap_outlier_likelihood": "outlier_likelihood",
    "PF_heatmap_dynamic_sigma_alpha": "dynamic_sigma_alpha",
    "PF_heatmap_dynamic_sigma_threshold": "dynamic_sigma_threshold",
    "PF_heatmap_dynamic_sigma_power": "dynamic_sigma_power",
    "PF_heatmap_dynamic_sigma_min": "dynamic_sigma_min",
    "PF_heatmap_dynamic_sigma_max": "dynamic_sigma_max",
    "PF_heatmap_seed_path_weight_mode": "seed_path_weight_mode",
    "PF_heatmap_seed_prob_weight_mode": "seed_prob_weight_mode",
    "PF_heatmap_seed_local_weight_mode": "seed_local_weight_mode",
    "PF_heatmap_seed_local_score_source": "seed_local_score_source",
    "PF_heatmap_seed_local_lik_scale": "seed_local_lik_scale",
    "PF_heatmap_seed_local_half_life_blocks": "seed_local_half_life_blocks",
    "PF_heatmap_seed_local_global_mix": "seed_local_global_mix",
    "PF_heatmap_seed_local_min_power": "seed_local_min_power",
    "PF_heatmap_seed_local_combine_mode": "seed_local_combine_mode",
    "PF_heatmap_seed_local_residual_alpha": "seed_local_residual_alpha",
    "PF_heatmap_seed_local_residual_clip": "seed_local_residual_clip",
    "PF_heatmap_state_z_weight": "pf_state_z_weight",
    "PF_heatmap_momentum": "pf_momentum",
    "PF_heatmap_rate_noise": "pf_rate_noise",
    "PF_heatmap_pos_noise": "pf_pos_noise",
    "PF_heatmap_resample_threshold": "pf_resample_threshold",
    "PF_heatmap_resample_obs_power_adapt": "pf_resample_obs_power_adapt",
    "PF_heatmap_resample_min_threshold": "pf_resample_min_threshold",
    "PF_heatmap_rough_pos": "pf_rough_pos",
    "PF_heatmap_rough_rate": "pf_rough_rate",
    "PF_heatmap_rescue_frac": "pf_rescue_frac",
    "PF_heatmap_rescue_pos_sd": "pf_rescue_pos_sd",
    "PF_heatmap_rescue_rate_sd": "pf_rescue_rate_sd",
    "PF_heatmap_init_pos_sd": "pf_init_pos_sd",
    "PF_heatmap_init_rate_sd": "pf_init_rate_sd",
    "PF_heatmap_rate_mean_weight": "pf_rate_mean_weight",
    "PF_heatmap_jump_prob": "pf_jump_prob",
    "PF_heatmap_jump_sd": "pf_jump_sd",
    "PF_heatmap_jump_rate_sd": "pf_jump_rate_sd",
    "PF_heatmap_missing_jump_boost": "pf_missing_jump_boost",
    "PF_heatmap_jump_tail_prob": "pf_jump_tail_prob",
    "PF_heatmap_jump_tail_sd": "pf_jump_tail_sd",
    "PF_heatmap_jump_tail_rate_sd": "pf_jump_tail_rate_sd",
    "PF_heatmap_jump_tail_dist": "pf_jump_tail_dist",
    "PF_heatmap_jump_tail_clip": "pf_jump_tail_clip",
    "PF_heatmap_jump_tail_missing_boost": "pf_jump_tail_missing_boost",
    "PF_heatmap_anchor_sigma": "pf_anchor_sigma",
    "PF_heatmap_anchor_power": "pf_anchor_power",
    "PF_heatmap_correlated_rate_alpha": "pf_correlated_rate_alpha",
    "PF_heatmap_correlated_pos_alpha": "pf_correlated_pos_alpha",
    "PF_heatmap_lookahead_power": "pf_lookahead_power",
    "PF_heatmap_lookahead_steps": "pf_lookahead_steps",
    "PF_heatmap_lookahead_decay": "pf_lookahead_decay",
    "PF_heatmap_lookahead_max_gap": "pf_lookahead_max_gap",
    "PF_heatmap_lookahead_delta_power": "pf_lookahead_delta_power",
    "PF_heatmap_stratified_init": "pf_stratified_init",
    "PF_heatmap_finite_run_power_boost": "pf_finite_run_power_boost",
    "PF_heatmap_finite_run_power_cap": "pf_finite_run_power_cap",
    "PF_heatmap_finite_run_power_decay": "pf_finite_run_power_decay",
    "PF_heatmap_finite_run_power_floor": "pf_finite_run_power_floor",
    "PF_heatmap_conf_obs_power_decay": "pf_conf_obs_power_decay",
    "PF_heatmap_conf_obs_power_floor": "pf_conf_obs_power_floor",
    "PF_heatmap_conf_obs_power_power": "pf_conf_obs_power_power",
    "PF_heatmap_missing_noise_scale": "pf_missing_noise_scale",
    "PF_heatmap_missing_jump_scale": "pf_missing_jump_scale",
    "PF_heatmap_ess_jump_boost": "pf_ess_jump_boost",
    "PF_heatmap_ess_jump_power": "pf_ess_jump_power",
    "PF_heatmap_surprise_jump_boost": "pf_surprise_jump_boost",
    "PF_heatmap_surprise_jump_threshold": "pf_surprise_jump_threshold",
    "PF_heatmap_surprise_jump_power": "pf_surprise_jump_power",
    "PF_heatmap_ess_rough_boost": "pf_ess_rough_boost",
    "PF_heatmap_ess_rough_power": "pf_ess_rough_power",
    "PF_heatmap_prob_temperature": "pf_prob_temperature",
    "PF_heatmap_ffbsi_mode": "pf_ffbsi_mode",
    "PF_heatmap_ffbsi_n_paths": "pf_ffbsi_n_paths",
    "PF_heatmap_ffbsi_fallback_mode": "pf_ffbsi_fallback_mode",
    "PF_heatmap_ffbsi_max_active_bins": "pf_ffbsi_max_active_bins",
    "PF_heatmap_ffbsi_transition_model": "pf_ffbsi_transition_model",
    "PF_heatmap_ffbsi_transition_scale": "pf_ffbsi_transition_scale",
    "PF_heatmap_ffbsi_pos_floor": "pf_ffbsi_pos_floor",
    "PF_heatmap_ffbsi_rate_floor": "pf_ffbsi_rate_floor",
}


PF_PROFILE_ALLOWED_CACHE_KEYS = set(PF_PROFILE_CFG_KEY_MAP.values())


def _profile_cache_key(key):
    key = str(key)
    if key in PF_PROFILE_CFG_KEY_MAP:
        return PF_PROFILE_CFG_KEY_MAP[key]
    return key


def _json_clean_profile_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_clean_profile_value(v) for v in value]
    if isinstance(value, list):
        return [_json_clean_profile_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_clean_profile_value(v) for k, v in value.items()}
    return value


def _normalize_profile_mixture_spec(raw_spec):
    if raw_spec is None or raw_spec == "":
        return []
    if isinstance(raw_spec, str):
        raw_spec = json.loads(raw_spec)
    if not isinstance(raw_spec, (list, tuple)):
        raise ValueError("PF_heatmap_profile_mixture_spec must be a list of profile dictionaries")

    profiles = []
    for idx, raw_profile in enumerate(raw_spec):
        if not isinstance(raw_profile, dict):
            raise ValueError("each PF profile must be a dictionary")
        name = str(raw_profile.get("name", f"profile{idx:02d}"))
        weight = float(raw_profile.get("weight", 1.0))
        if weight <= 0.0 or not np.isfinite(weight):
            raise ValueError(f"PF profile {name!r} has invalid weight={weight!r}")
        raw_overrides = raw_profile.get("overrides", {})
        if not isinstance(raw_overrides, dict):
            raise ValueError(f"PF profile {name!r} overrides must be a dictionary")
        overrides = {}
        for key, value in raw_overrides.items():
            cache_key = _profile_cache_key(key)
            if cache_key not in PF_PROFILE_ALLOWED_CACHE_KEYS:
                raise ValueError(f"PF profile {name!r} override key {key!r} is not supported")
            overrides[cache_key] = _json_clean_profile_value(value)
        profiles.append(
            {
                "name": name,
                "weight": weight,
                "overrides": dict(sorted(overrides.items())),
            }
        )

    total = sum(profile["weight"] for profile in profiles)
    if total <= 0.0:
        return []
    for profile in profiles:
        profile["weight"] = float(profile["weight"] / total)
    return profiles


def _profile_indices_for_seed_count(profiles, n_seeds, shuffle_seed=0):
    n_seeds = int(n_seeds)
    if n_seeds <= 0 or not profiles:
        return np.zeros(max(n_seeds, 0), dtype=np.int64)

    weights = np.asarray([float(profile["weight"]) for profile in profiles], dtype=np.float64)
    weights /= weights.sum()
    counts = np.zeros(len(profiles), dtype=np.int64)
    positive = np.flatnonzero(weights > 0.0)
    if n_seeds >= len(positive):
        counts[positive] = 1
        remaining = n_seeds - int(counts.sum())
        base = weights * remaining
    else:
        remaining = n_seeds
        base = weights * remaining

    add = np.floor(base).astype(np.int64)
    counts += add
    left = n_seeds - int(counts.sum())
    if left > 0:
        frac = base - np.floor(base)
        order = np.argsort(frac)[::-1]
        for idx in order[:left]:
            counts[idx] += 1

    # In very small seed budgets, the guarantee-one branch can over-allocate
    # after rounding edge cases. Trim from the lowest-prior profiles first.
    while int(counts.sum()) > n_seeds:
        removable = np.flatnonzero(counts > 0)
        idx = removable[np.argmin(weights[removable])]
        counts[idx] -= 1

    profile_indices = np.repeat(np.arange(len(profiles), dtype=np.int64), counts)
    if profile_indices.size != n_seeds:
        raise RuntimeError("PF profile allocation failed to match n_seeds")
    rng = np.random.default_rng(int(shuffle_seed) + 100003 * int(n_seeds))
    rng.shuffle(profile_indices)
    return profile_indices


def _apply_profile_to_cache_cfg(cache_cfg, profile):
    if not profile:
        return cache_cfg
    overrides = profile.get("overrides", {})
    if not overrides:
        return cache_cfg
    out = dict(cache_cfg)
    out.update(overrides)
    return out


def _interpolate_horizontal_gr(gr_series, fallback):
    gr = gr_series.astype(float).interpolate(limit_direction="both")
    return gr.fillna(float(fallback))


def _prefix_gr_for_pf_sigma(gr_series):
    # Match the strong reference PF: missing prefix GR inflates the likelihood
    # scale instead of being interpolated into an overconfident GR matcher.
    return gr_series.fillna(0.0).to_numpy(dtype=float)


def run_particle_filter(hw, tw, n_particles=500, seed=42):
    """Conservative PF. Returns (predictions_array, total_log_likelihood)."""
    tw_s   = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    tw_mean_gr = float(np.nanmean(tw_gr)) if np.isfinite(tw_gr).any() else 0.0
    gr_interp = _interpolate_horizontal_gr(hw["GR"], tw_mean_gr)

    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy(), 0.0

    last     = kn.iloc[-1]
    last_tvt = float(last['TVT_input'])
    last_Z   = float(last['Z'])
    last_MD  = float(last['MD'])

    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(_prefix_gr_for_pf_sigma(kn['GR']) - tw_at_k), 10., 60.))

    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values)
    dm = np.diff(tail['MD'].values)
    m  = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N   = n_particles
    rng = np.random.default_rng(seed)
    ls   = last_tvt + last_Z
    pos  = ls + 3.0 * rng.standard_normal(N)  # wider init spread helps wells with abrupt TVT shift at PS
    rate = ir + 0.01 * rng.standard_normal(N)
    w    = np.ones(N) / N

    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5

    md_v = ev['MD'].values.astype(float)
    z_v  = ev['Z'].values.astype(float)
    # Interpolate GR gaps before tracking; high-NaN wells should still advance
    # with a continuous GR observation stream, matching the existing data-prep
    # pattern used elsewhere in this sequence pipeline.
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw['TVT_input'].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos  = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos   = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d  = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        n_eff = 1.0 / (w**2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0  = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos  = pos[idx]  + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w    = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble(hw, tw, n_particles=500, n_seeds=32, scale=5.0):
    """
    128-seed lik-weighted PF ensemble.
    More seeds give better coverage of the TVT exploration space.
    """
    preds = []
    liks  = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p)
        liks.append(ll)

    liks   = np.array(liks)
    liks_n = liks - liks.max()
    weights = np.exp(liks_n / scale)
    weights /= weights.sum()

    return (weights[:, None] * np.stack(preds, 0)).sum(0)


def run_particle_filter_pasted_v12(hw, tw, n_particles=500, seed=42):
    """User-pasted v12 PF path, kept separate for direct comparison."""
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy(), 0.0

    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_z = float(last["Z"])
    last_md = float(last["MD"])

    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))

    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    move = dm > 0
    init_rate = float(np.median((dt + dz)[move] / dm[move])) if move.sum() >= 3 else 0.0

    n = int(n_particles)
    rng = np.random.default_rng(seed)
    init_pos = last_tvt + last_z
    pos = init_pos + 2.0 * rng.standard_normal(n)
    rate = init_rate + 0.01 * rng.standard_normal(n)
    weight = np.ones(n) / n

    momentum = 0.998
    rate_noise = 0.002
    pos_noise = 0.005
    rough_pos = 0.1
    rough_rate = 0.001
    resample_threshold = 0.5

    md_v = ev["MD"].values.astype(float)
    z_v = ev["Z"].values.astype(float)
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw["TVT_input"].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_md = last_md
    log_lik = 0.0

    for i in range(len(ev)):
        dmd = max(md_v[i] - prev_md, 1.0)
        rate = momentum * rate + rate_noise * rng.standard_normal(n)
        pos = pos + rate * dmd + pos_noise * rng.standard_normal(n)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100.0, tw_tvt[-1] + 100.0)
        pos = tvt_p + z_v[i]

        expected_gr = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - expected_gr) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.0))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((weight * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        weight = weight * lk
        weight_sum = weight.sum()
        weight = weight / weight_sum if weight_sum > 0 else np.ones(n) / n

        n_eff = 1.0 / (weight**2).sum()
        if n_eff < resample_threshold * n:
            cum = np.cumsum(weight)
            u0 = rng.uniform(0, 1.0 / n)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(n) / n), 0, n - 1)
            pos = pos[idx] + rough_pos * rng.standard_normal(n)
            rate = rate[idx] + rough_rate * rng.standard_normal(n)
            weight = np.ones(n) / n

        res[i] = float(np.dot(weight, pos - z_v[i]))
        prev_md = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble_pasted_v12(hw, tw, n_particles=500, n_seeds=64, scale=5.0):
    preds = []
    liks = []
    for seed in range(int(n_seeds)):
        pred, log_lik = run_particle_filter_pasted_v12(hw, tw, n_particles=n_particles, seed=seed)
        preds.append(pred)
        liks.append(log_lik)

    liks = np.asarray(liks, dtype=np.float64)
    weights = np.exp((liks - liks.max()) / float(scale))
    weights /= weights.sum()
    return (weights[:, None] * np.stack(preds, axis=0)).sum(axis=0)


def uses_pf_heatmap_channels(cfg):
    static_channels = getattr(cfg, "unet_static_channels", getattr(cfg, "stage_unet_static_channels", ()))
    return any(name in PF_HEATMAP_CHANNELS for name in static_channels)


def pf_heatmap_split_dir(cfg, split):
    return Path(getattr(cfg, "PF_heatmap_cache_dir", Path("PF_cache"))) / split


def _pf_typewell_gr_calibration_mode(cfg):
    if hasattr(cfg, "PF_heatmap_typewell_gr_calibration"):
        return str(getattr(cfg, "PF_heatmap_typewell_gr_calibration"))
    apply_gr_calibration = bool(getattr(cfg, "PF_heatmap_apply_gr_calibration", True))
    return "seen_blend" if apply_gr_calibration else "none"


def pf_heatmap_cache_config(cfg):
    typewell_gr_calibration = _pf_typewell_gr_calibration_mode(cfg)
    profile_mixture_spec = _normalize_profile_mixture_spec(
        getattr(cfg, "PF_heatmap_profile_mixture_spec", None)
    )
    suffix_merge_likelihood_mode = str(getattr(cfg, "PF_heatmap_suffix_merge_likelihood_mode", "block"))
    suffix_merge_likelihood_mode_norm = suffix_merge_likelihood_mode.lower()
    suffix_merge_default_alpha = 1.0 if suffix_merge_likelihood_mode_norm == "block" else 0.5
    suffix_merge_default_adjust = False if suffix_merge_likelihood_mode_norm == "block" else True
    return {
        "version": int(getattr(cfg, "PF_heatmap_version", 1)),
        "n_particles": int(getattr(cfg, "PF_heatmap_n_particles", 500)),
        "n_seeds": int(getattr(cfg, "PF_heatmap_n_seeds", 16)),
        "base_seed": int(getattr(cfg, "PF_heatmap_base_seed", 202605)),
        "profile_mixture_mode": "profiles" if profile_mixture_spec else "fixed",
        "profile_mixture_spec": profile_mixture_spec,
        "lik_scale": float(getattr(cfg, "PF_heatmap_lik_scale", 5.0)),
        "axis_sigma": float(getattr(cfg, "PF_heatmap_axis_sigma", 0.0)),
        "ref_grid_step": float(getattr(cfg, "PF_heatmap_ref_grid_step", 0.2)),
        "query_gr_mode": str(getattr(cfg, "PF_heatmap_query_gr_mode", "interp")),
        "apply_gr_calibration": typewell_gr_calibration != "none",
        "typewell_gr_calibration": typewell_gr_calibration,
        "seen_blend_weight": float(getattr(cfg, "PF_heatmap_seen_blend_weight", 0.7)),
        "raw_ref_likelihood_weight": float(getattr(cfg, "PF_heatmap_raw_ref_likelihood_weight", 0.0)),
        "gr_ambiguity_power": float(getattr(cfg, "PF_heatmap_gr_ambiguity_power", 0.0)),
        "gr_ambiguity_min_power": float(getattr(cfg, "PF_heatmap_gr_ambiguity_min_power", 0.35)),
        "gr_ambiguity_contrast": float(getattr(cfg, "PF_heatmap_gr_ambiguity_contrast", 1.0)),
        "gr_ambiguity_ref_mode": str(getattr(cfg, "PF_heatmap_gr_ambiguity_ref_mode", "primary")),
        "gr_information_power": float(getattr(cfg, "PF_heatmap_gr_information_power", 0.0)),
        "gr_information_center": float(getattr(cfg, "PF_heatmap_gr_information_center", 0.75)),
        "gr_information_min_multiplier": float(getattr(cfg, "PF_heatmap_gr_information_min_multiplier", 0.75)),
        "gr_information_max_multiplier": float(getattr(cfg, "PF_heatmap_gr_information_max_multiplier", 1.25)),
        "gr_information_slope_weight": float(getattr(cfg, "PF_heatmap_gr_information_slope_weight", 0.0)),
        "gr_information_ref_mode": str(getattr(cfg, "PF_heatmap_gr_information_ref_mode", "primary")),
        "gr_shape_power": float(getattr(cfg, "PF_heatmap_gr_shape_power", 0.0)),
        "gr_shape_mode": str(getattr(cfg, "PF_heatmap_gr_shape_mode", "resid_corr")),
        "gr_shape_window": int(getattr(cfg, "PF_heatmap_gr_shape_window", 15)),
        "gr_shape_min_points": int(getattr(cfg, "PF_heatmap_gr_shape_min_points", 7)),
        "gr_shape_sigma_floor": float(getattr(cfg, "PF_heatmap_gr_shape_sigma_floor", 0.35)),
        "gr_shape_ref_mode": str(getattr(cfg, "PF_heatmap_gr_shape_ref_mode", "primary")),
        "missing_likelihood_power": float(getattr(cfg, "PF_heatmap_missing_likelihood_power", 1.0)),
        "missing_gap_decay": float(getattr(cfg, "PF_heatmap_missing_gap_decay", 0.0)),
        "missing_min_power": float(getattr(cfg, "PF_heatmap_missing_min_power", 0.05)),
        "outlier_prob": float(getattr(cfg, "PF_heatmap_outlier_prob", 0.0)),
        "outlier_likelihood": float(getattr(cfg, "PF_heatmap_outlier_likelihood", 0.05)),
        "dynamic_sigma_alpha": float(getattr(cfg, "PF_heatmap_dynamic_sigma_alpha", 0.0)),
        "dynamic_sigma_threshold": float(getattr(cfg, "PF_heatmap_dynamic_sigma_threshold", 1.25)),
        "dynamic_sigma_power": float(getattr(cfg, "PF_heatmap_dynamic_sigma_power", 1.0)),
        "dynamic_sigma_min": float(getattr(cfg, "PF_heatmap_dynamic_sigma_min", 0.85)),
        "dynamic_sigma_max": float(getattr(cfg, "PF_heatmap_dynamic_sigma_max", 2.0)),
        "seed_path_weight_mode": str(getattr(cfg, "PF_heatmap_seed_path_weight_mode", "likelihood")),
        "seed_prob_weight_mode": str(getattr(cfg, "PF_heatmap_seed_prob_weight_mode", "equal")),
        "seed_local_weight_mode": str(getattr(cfg, "PF_heatmap_seed_local_weight_mode", "off")),
        "seed_local_score_source": str(getattr(cfg, "PF_heatmap_seed_local_score_source", "pf_likelihood")),
        "seed_local_lik_scale": float(getattr(cfg, "PF_heatmap_seed_local_lik_scale", 1.0)),
        "seed_local_half_life_blocks": float(getattr(cfg, "PF_heatmap_seed_local_half_life_blocks", 8.0)),
        "seed_local_global_mix": float(getattr(cfg, "PF_heatmap_seed_local_global_mix", 0.0)),
        "seed_local_min_power": float(getattr(cfg, "PF_heatmap_seed_local_min_power", 1e-6)),
        "seed_local_combine_mode": str(getattr(cfg, "PF_heatmap_seed_local_combine_mode", "replace")),
        "seed_local_residual_alpha": float(getattr(cfg, "PF_heatmap_seed_local_residual_alpha", 0.0)),
        "seed_local_residual_clip": float(getattr(cfg, "PF_heatmap_seed_local_residual_clip", 5.0)),
        "pf_state_z_weight": float(getattr(cfg, "PF_heatmap_state_z_weight", 1.0)),
        "pf_momentum": float(getattr(cfg, "PF_heatmap_momentum", 0.998)),
        "pf_rate_noise": float(getattr(cfg, "PF_heatmap_rate_noise", 0.002)),
        "pf_pos_noise": float(getattr(cfg, "PF_heatmap_pos_noise", 0.005)),
        "pf_resample_threshold": float(getattr(cfg, "PF_heatmap_resample_threshold", 0.5)),
        "pf_resample_obs_power_adapt": float(getattr(cfg, "PF_heatmap_resample_obs_power_adapt", 0.0)),
        "pf_resample_min_threshold": float(getattr(cfg, "PF_heatmap_resample_min_threshold", 0.0)),
        "pf_rough_pos": float(getattr(cfg, "PF_heatmap_rough_pos", 0.1)),
        "pf_rough_rate": float(getattr(cfg, "PF_heatmap_rough_rate", 0.001)),
        "pf_rescue_frac": float(getattr(cfg, "PF_heatmap_rescue_frac", 0.0)),
        "pf_rescue_pos_sd": float(getattr(cfg, "PF_heatmap_rescue_pos_sd", 0.0)),
        "pf_rescue_rate_sd": float(getattr(cfg, "PF_heatmap_rescue_rate_sd", 0.0)),
        "pf_init_pos_sd": float(getattr(cfg, "PF_heatmap_init_pos_sd", 3.0)),
        "pf_init_rate_sd": float(getattr(cfg, "PF_heatmap_init_rate_sd", 0.01)),
        "pf_rate_mean_weight": float(getattr(cfg, "PF_heatmap_rate_mean_weight", 0.0)),
        "pf_jump_prob": float(getattr(cfg, "PF_heatmap_jump_prob", 0.0)),
        "pf_jump_sd": float(getattr(cfg, "PF_heatmap_jump_sd", 0.0)),
        "pf_jump_rate_sd": float(getattr(cfg, "PF_heatmap_jump_rate_sd", 0.0)),
        "pf_missing_jump_boost": float(getattr(cfg, "PF_heatmap_missing_jump_boost", 0.0)),
        "pf_jump_tail_prob": float(getattr(cfg, "PF_heatmap_jump_tail_prob", 0.0)),
        "pf_jump_tail_sd": float(getattr(cfg, "PF_heatmap_jump_tail_sd", 0.0)),
        "pf_jump_tail_rate_sd": float(getattr(cfg, "PF_heatmap_jump_tail_rate_sd", 0.0)),
        "pf_jump_tail_dist": int(getattr(cfg, "PF_heatmap_jump_tail_dist", 1)),
        "pf_jump_tail_clip": float(getattr(cfg, "PF_heatmap_jump_tail_clip", 0.0)),
        "pf_jump_tail_missing_boost": float(getattr(cfg, "PF_heatmap_jump_tail_missing_boost", 0.0)),
        "pf_anchor_sigma": float(getattr(cfg, "PF_heatmap_anchor_sigma", 0.0)),
        "pf_anchor_power": float(getattr(cfg, "PF_heatmap_anchor_power", 1.0)),
        "pf_correlated_rate_alpha": float(getattr(cfg, "PF_heatmap_correlated_rate_alpha", 0.0)),
        "pf_correlated_pos_alpha": float(getattr(cfg, "PF_heatmap_correlated_pos_alpha", 0.0)),
        "pf_lookahead_power": float(getattr(cfg, "PF_heatmap_lookahead_power", 0.0)),
        "pf_lookahead_steps": int(getattr(cfg, "PF_heatmap_lookahead_steps", 1)),
        "pf_lookahead_decay": float(getattr(cfg, "PF_heatmap_lookahead_decay", 0.5)),
        "pf_lookahead_max_gap": float(getattr(cfg, "PF_heatmap_lookahead_max_gap", 256.0)),
        "pf_lookahead_delta_power": float(getattr(cfg, "PF_heatmap_lookahead_delta_power", 0.0)),
        "pf_stratified_init": int(getattr(cfg, "PF_heatmap_stratified_init", 0)),
        "pf_finite_run_power_boost": float(getattr(cfg, "PF_heatmap_finite_run_power_boost", 0.0)),
        "pf_finite_run_power_cap": float(getattr(cfg, "PF_heatmap_finite_run_power_cap", 1.0)),
        "pf_finite_run_power_decay": float(getattr(cfg, "PF_heatmap_finite_run_power_decay", 0.0)),
        "pf_finite_run_power_floor": float(getattr(cfg, "PF_heatmap_finite_run_power_floor", 0.35)),
        "pf_conf_obs_power_decay": float(getattr(cfg, "PF_heatmap_conf_obs_power_decay", 0.0)),
        "pf_conf_obs_power_floor": float(getattr(cfg, "PF_heatmap_conf_obs_power_floor", 0.55)),
        "pf_conf_obs_power_power": float(getattr(cfg, "PF_heatmap_conf_obs_power_power", 1.0)),
        "pf_missing_noise_scale": float(getattr(cfg, "PF_heatmap_missing_noise_scale", 1.0)),
        "pf_missing_jump_scale": float(getattr(cfg, "PF_heatmap_missing_jump_scale", 1.0)),
        "pf_ess_jump_boost": float(getattr(cfg, "PF_heatmap_ess_jump_boost", 0.0)),
        "pf_ess_jump_power": float(getattr(cfg, "PF_heatmap_ess_jump_power", 1.0)),
        "pf_surprise_jump_boost": float(getattr(cfg, "PF_heatmap_surprise_jump_boost", 0.0)),
        "pf_surprise_jump_threshold": float(getattr(cfg, "PF_heatmap_surprise_jump_threshold", 1.5)),
        "pf_surprise_jump_power": float(getattr(cfg, "PF_heatmap_surprise_jump_power", 1.0)),
        "pf_ess_rough_boost": float(getattr(cfg, "PF_heatmap_ess_rough_boost", 0.0)),
        "pf_ess_rough_power": float(getattr(cfg, "PF_heatmap_ess_rough_power", 1.0)),
        "pf_prob_temperature": float(getattr(cfg, "PF_heatmap_prob_temperature", 1.0)),
        "pf_ffbsi_mode": str(getattr(cfg, "PF_heatmap_ffbsi_mode", "off")),
        "pf_ffbsi_n_paths": int(getattr(cfg, "PF_heatmap_ffbsi_n_paths", 0)),
        "pf_ffbsi_fallback_mode": str(getattr(cfg, "PF_heatmap_ffbsi_fallback_mode", "filtered")),
        "pf_ffbsi_max_active_bins": int(getattr(cfg, "PF_heatmap_ffbsi_max_active_bins", 512)),
        "pf_ffbsi_transition_model": str(getattr(cfg, "PF_heatmap_ffbsi_transition_model", "gaussian")),
        "pf_ffbsi_transition_scale": float(getattr(cfg, "PF_heatmap_ffbsi_transition_scale", 1.5)),
        "pf_ffbsi_pos_floor": float(getattr(cfg, "PF_heatmap_ffbsi_pos_floor", 0.35)),
        "pf_ffbsi_rate_floor": float(getattr(cfg, "PF_heatmap_ffbsi_rate_floor", 0.0025)),
        "suffix_merge_k": int(getattr(cfg, "PF_heatmap_suffix_merge_k", 1)),
        "suffix_merge_obs_power_alpha": float(getattr(cfg, "PF_heatmap_suffix_merge_obs_power_alpha", suffix_merge_default_alpha)),
        "suffix_merge_adjust_dynamics": bool(getattr(cfg, "PF_heatmap_suffix_merge_adjust_dynamics", suffix_merge_default_adjust)),
        "suffix_merge_likelihood_mode": suffix_merge_likelihood_mode,
        "suffix_merge_state_mode": str(getattr(cfg, "PF_heatmap_suffix_merge_state_mode", "start")),
        "typewell_window": float(cfg.typewell_window),
        "typewell_len": int(cfg.typewell_len),
        "downsample": int(cfg.downsample),
        "prefix_len": int(cfg.prefix_len),
        "target_len": int(cfg.target_len),
        "raw_len": int(cfg.raw_len),
        "num_bins": int(cfg.num_bins),
    }


def pf_heatmap_cache_digest(cache_cfg):
    payload = json.dumps(cache_cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def pf_heatmap_file(split_dir, well_id):
    return Path(split_dir) / f"{well_id}.npz"


def load_pf_heatmap_cache(cache_dir, well_id):
    path = pf_heatmap_file(cache_dir, well_id)
    if not path.exists():
        raise FileNotFoundError(f"missing PF heatmap cache for {well_id}: {path}")
    with np.load(path) as data:
        # Original visible-prefix PF rows are painted from known TVT_input.
        # Sanitize them at load time before simulation/augmentation remaps.
        last_seen_idx = int(data["last_seen_idx"])
        pf_prob = data["pf_prob"].astype("float32")
        pf_tvt_pred = data["pf_tvt_pred"].astype("float32")
        pf_tvt_rel_pred = data["pf_tvt_rel_pred"].astype("float32")
        window_orig_index = data["window_orig_index"].astype("float32")
        original_prefix = (
            np.isfinite(window_orig_index)
            & (window_orig_index >= 0.0)
            & (window_orig_index <= np.float32(last_seen_idx))
        )
        pf_prob[original_prefix, :] = 0.0
        pf_tvt_pred[original_prefix] = np.nan
        pf_tvt_rel_pred[original_prefix] = np.nan
        return {
            "well_id": str(data["well_id"]),
            "cache_digest": str(data["cache_digest"]),
            "last_seen_idx": last_seen_idx,
            "tvt0": float(data["tvt0"]),
            "prefix_start": int(data["prefix_start"]),
            "suffix_len": int(data["suffix_len"]),
            "pf_prob": pf_prob,
            "pf_tvt_pred": pf_tvt_pred,
            "pf_tvt_rel_pred": pf_tvt_rel_pred,
            "window_tvt": data["window_tvt"].astype("float32"),
            "window_tvt_rel": data["window_tvt_rel"].astype("float32"),
            "window_has_tvt": data["window_has_tvt"].astype(bool),
            "window_orig_index": window_orig_index,
            "target_mask": data["target_mask"].astype(bool),
        }


def ensure_pf_heatmap_cache(split, data_path, well_ids, cfg, log=None):
    if not uses_pf_heatmap_channels(cfg):
        return None
    if njit is None:
        raise ImportError("numba is required for PF heatmap cache generation")

    split = str(split)
    data_path = Path(data_path)
    well_ids = list(well_ids)
    split_dir = pf_heatmap_split_dir(cfg, split)
    split_dir.mkdir(parents=True, exist_ok=True)
    cache_cfg = pf_heatmap_cache_config(cfg)
    cache_digest = pf_heatmap_cache_digest(cache_cfg)
    config_path = split_dir / "config.json"
    requested_files = [pf_heatmap_file(split_dir, well_id) for well_id in well_ids]

    existing_cfg = None
    if config_path.exists():
        with config_path.open("r") as f:
            existing_cfg = json.load(f)
    cache_ready = (
        existing_cfg is not None
        and existing_cfg.get("cache_digest") == cache_digest
        and existing_cfg.get("cache_cfg") == cache_cfg
        and all(path.exists() for path in requested_files)
    )
    if cache_ready:
        _log(log, f"PF heatmap cache reuse split={split}: {split_dir} wells={len(well_ids):,} digest={cache_digest}")
        return split_dir

    if existing_cfg is not None:
        _log(log, f"PF heatmap cache config changed for split={split}; clearing {split_dir}")
        for path in split_dir.glob("*.npz"):
            path.unlink()
        summary_path = split_dir / "summary.json"
        if summary_path.exists():
            summary_path.unlink()
        diag_dir = split_dir / "diagnostics"
        if diag_dir.exists():
            shutil.rmtree(diag_dir)

    config_payload = {
        "cache_digest": cache_digest,
        "cache_cfg": cache_cfg,
    }
    with config_path.open("w") as f:
        json.dump(config_payload, f, indent=2, sort_keys=True)

    _warmup_pf_heatmap_numba()
    if split == "train" and len(well_ids) > 0:
        _benchmark_numba_vs_python(data_path, well_ids[: min(2, len(well_ids))], cfg, log)

    started = time.perf_counter()
    worker_args = [
        (
            data_path,
            well_id,
            split_dir,
            cache_cfg,
            cache_digest,
            split == "train",
        )
        for well_id in well_ids
    ]
    num_workers = min(int(getattr(cfg, "PF_heatmap_num_workers", 4)), len(worker_args))
    num_workers = max(num_workers, 1)
    _log(
        log,
        "generating PF heatmap cache "
        f"split={split} wells={len(well_ids):,} workers={num_workers} "
        f"N={cache_cfg['n_particles']} seeds={cache_cfg['n_seeds']} "
        f"shape=({cache_cfg['num_bins']},{cache_cfg['typewell_len']}) digest={cache_digest}",
    )
    if num_workers == 1:
        results = [
            _generate_pf_heatmap_cache_one(args)
            for args in tqdm(worker_args, desc=f"PF {split}", dynamic_ncols=True)
        ]
    else:
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_generate_pf_heatmap_cache_one, worker_args),
                    total=len(worker_args),
                    desc=f"PF {split}",
                    dynamic_ncols=True,
                )
            )

    elapsed = time.perf_counter() - started
    _write_pf_cache_summary(split_dir, split, results, elapsed, cache_digest, cache_cfg, log)
    return split_dir


def _log(log, message):
    if log is None:
        print(message, flush=True)
    else:
        log(message)


def _write_pf_cache_summary(split_dir, split, results, elapsed, cache_digest, cache_cfg, log):
    tvt_rmse_values = np.asarray([r["tvt_rmse"] for r in results if np.isfinite(r["tvt_rmse"])], dtype=float)
    row_counts = np.asarray([r["target_count"] for r in results], dtype=float)
    row_sum_min = np.asarray([r["prob_row_sum_min"] for r in results if np.isfinite(r["prob_row_sum_min"])], dtype=float)
    row_sum_max = np.asarray([r["prob_row_sum_max"] for r in results if np.isfinite(r["prob_row_sum_max"])], dtype=float)
    weighted_sse = float(sum(r["tvt_sse"] for r in results if np.isfinite(r["tvt_sse"])))
    weighted_count = float(sum(r["target_count"] for r in results))
    weighted_rmse = math.sqrt(weighted_sse / weighted_count) if weighted_count > 0 else math.nan
    summary = {
        "split": split,
        "cache_digest": cache_digest,
        "cache_cfg": cache_cfg,
        "elapsed_sec": elapsed,
        "sec_per_well": elapsed / max(len(results), 1),
        "well_count": len(results),
        "row_weighted_tvt_rmse": weighted_rmse,
        "target_rows": int(weighted_count),
        "well_tvt_rmse_mean": float(np.mean(tvt_rmse_values)) if tvt_rmse_values.size else math.nan,
        "well_tvt_rmse_median": float(np.median(tvt_rmse_values)) if tvt_rmse_values.size else math.nan,
        "well_tvt_rmse_q75": float(np.quantile(tvt_rmse_values, 0.75)) if tvt_rmse_values.size else math.nan,
        "well_tvt_rmse_q95": float(np.quantile(tvt_rmse_values, 0.95)) if tvt_rmse_values.size else math.nan,
        "well_tvt_rmse_q99": float(np.quantile(tvt_rmse_values, 0.99)) if tvt_rmse_values.size else math.nan,
        "target_row_count_mean": float(np.mean(row_counts)) if row_counts.size else math.nan,
        "prob_active_row_sum_min": float(np.min(row_sum_min)) if row_sum_min.size else math.nan,
        "prob_active_row_sum_max": float(np.max(row_sum_max)) if row_sum_max.size else math.nan,
    }
    with (Path(split_dir) / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    msg = (
        f"PF heatmap cache generated split={split}: wells={len(results):,}, "
        f"elapsed={elapsed:.2f}s, sec/well={elapsed / max(len(results), 1):.3f}"
    )
    if np.isfinite(weighted_rmse):
        msg += (
            f", TVT_RMSE={weighted_rmse:.4f}, "
            f"well_rmse mean/median/q75/q95/q99="
            f"{summary['well_tvt_rmse_mean']:.4f}/"
            f"{summary['well_tvt_rmse_median']:.4f}/"
            f"{summary['well_tvt_rmse_q75']:.4f}/"
            f"{summary['well_tvt_rmse_q95']:.4f}/"
            f"{summary['well_tvt_rmse_q99']:.4f}, "
            f"prob_row_sum={summary['prob_active_row_sum_min']:.6f}.."
            f"{summary['prob_active_row_sum_max']:.6f}"
        )
    _log(log, msg)


def _read_well_frames(data_path, well_id):
    h_cols = ["MD", "Z", "GR", "TVT_input"]
    h_path = Path(data_path) / f"{well_id}__horizontal_well.csv"
    h_df = pd.read_csv(h_path)
    h_df = h_df[[col for col in h_cols + ["TVT"] if col in h_df.columns]]
    tw_df = pd.read_csv(Path(data_path) / f"{well_id}__typewell.csv", usecols=["TVT", "GR"])
    return h_df, tw_df


def _gr_calibration_seen_blend(h_df, tw_df, blend_weight=0.7):
    """Copy of seq_NN_dataset.GR_calibration for opt-only PF experiments."""
    target_cond = h_df["TVT_input"].isna()
    seen_h_df = h_df[~target_cond].copy()
    blend_weight = float(blend_weight)

    tvt0 = seen_h_df["TVT_input"].iloc[-1]
    refer_cond = seen_h_df["TVT_input"].between(tvt0 - 100, tvt0 + 100)
    refer_cond &= ~seen_h_df["GR"].isna()
    refer_df = seen_h_df[refer_cond][["TVT_input", "GR"]].copy()
    refer_df["TVT"] = (refer_df["TVT_input"] * 4).round(0) / 4
    refer_df = refer_df.groupby("TVT")["GR"].mean().reset_index().sort_values("TVT")

    out = tw_df["GR"].to_numpy(dtype=np.float64, copy=True)
    cond = tw_df["TVT"].between(refer_df["TVT"].min(), refer_df["TVT"].max()).to_numpy()
    if cond.any():
        interp_gr = np.interp(
            tw_df.loc[cond, "TVT"].to_numpy(dtype=np.float64),
            refer_df["TVT"].to_numpy(dtype=np.float64),
            refer_df["GR"].to_numpy(dtype=np.float64),
            left=np.nan,
            right=np.nan,
        )
        raw_gr = tw_df.loc[cond, "GR"].to_numpy(dtype=np.float64)
        blended = blend_weight * interp_gr + (1.0 - blend_weight) * raw_gr
        out[np.flatnonzero(cond)] = np.where(np.isfinite(blended), blended, raw_gr)
    return out


def _apply_typewell_gr_calibration(h_df, tw_df, cache_cfg):
    mode = str(cache_cfg.get("typewell_gr_calibration", "none"))
    if mode in {"none", "", "raw"}:
        return tw_df
    out = tw_df.copy()
    if mode == "seen_blend":
        out["GR"] = _gr_calibration_seen_blend(
            h_df,
            tw_df,
            blend_weight=float(cache_cfg.get("seen_blend_weight", 0.7)),
        )
        return out
    raise ValueError(f"unknown PF typewell_gr_calibration={mode!r}")


def _typewell_arrays_for_cache_cfg(h_df, raw_tw_df, cache_cfg):
    tw_df = _apply_typewell_gr_calibration(h_df, raw_tw_df, cache_cfg)
    typewell = tw_df.sort_values("TVT")
    tw_tvt = typewell["TVT"].to_numpy(dtype=np.float64)
    tw_gr = typewell["GR"].fillna(typewell["GR"].mean()).to_numpy(dtype=np.float64)
    return tw_tvt, tw_gr


def _generate_pf_heatmap_cache_one(args):
    data_path, well_id, split_dir, cache_cfg, cache_digest, has_target = args
    h_df, tw_df = _read_well_frames(data_path, well_id)
    item = _build_pf_heatmap_item(h_df, tw_df, cache_cfg, has_target)
    path = pf_heatmap_file(split_dir, well_id)
    np.savez_compressed(
        path,
        well_id=np.asarray(well_id),
        cache_digest=np.asarray(cache_digest),
        last_seen_idx=np.asarray(item["last_seen_idx"], dtype=np.int64),
        tvt0=np.asarray(item["tvt0"], dtype=np.float32),
        prefix_start=np.asarray(item["prefix_start"], dtype=np.int64),
        suffix_len=np.asarray(item["suffix_len"], dtype=np.int64),
        pf_prob=item["pf_prob"].astype("float16"),
        pf_tvt_pred=item["pf_tvt_pred"].astype("float32"),
        pf_tvt_rel_pred=item["pf_tvt_rel_pred"].astype("float32"),
        pf_prob_ffbsi=item["pf_prob_ffbsi"].astype("float16"),
        pf_tvt_pred_ffbsi=item["pf_tvt_pred_ffbsi"].astype("float32"),
        pf_tvt_rel_pred_ffbsi=item["pf_tvt_rel_pred_ffbsi"].astype("float32"),
        window_tvt=item["window_tvt"].astype("float32"),
        window_tvt_rel=item["window_tvt_rel"].astype("float32"),
        window_has_tvt=item["window_has_tvt"].astype(bool),
        window_orig_index=item["window_orig_index"].astype("float32"),
        target_mask=item["target_mask"].astype(bool),
    )
    return {
        "well_id": well_id,
        "target_count": int(item["target_count"]),
        "tvt_sse": float(item["tvt_sse"]),
        "tvt_rmse": float(item["tvt_rmse"]),
        "prob_row_sum_min": float(item["prob_row_sum_min"]),
        "prob_row_sum_max": float(item["prob_row_sum_max"]),
        "target_count_ffbsi": int(item["target_count_ffbsi"]),
        "tvt_sse_ffbsi": float(item["tvt_sse_ffbsi"]),
        "tvt_rmse_ffbsi": float(item["tvt_rmse_ffbsi"]),
        "prob_row_sum_min_ffbsi": float(item["prob_row_sum_min_ffbsi"]),
        "prob_row_sum_max_ffbsi": float(item["prob_row_sum_max_ffbsi"]),
    }


def _build_pf_heatmap_item(h_df, tw_df, cache_cfg, has_target):
    raw_tw_df = tw_df.copy()
    tw_tvt_default, tw_gr_default = _typewell_arrays_for_cache_cfg(h_df, raw_tw_df, cache_cfg)
    tvt_input = h_df["TVT_input"].to_numpy(dtype=np.float64)
    finite_seen = np.flatnonzero(np.isfinite(tvt_input))
    if finite_seen.size == 0:
        raise ValueError("no finite TVT_input anchor rows")
    last_seen_idx = int(finite_seen[-1])
    tvt0 = float(tvt_input[last_seen_idx])
    suffix_start = last_seen_idx + 1
    row_count = len(h_df)
    suffix_len = row_count - suffix_start
    kept_suffix_len = min(suffix_len, int(cache_cfg["target_len"]))

    prefix_len = int(cache_cfg["prefix_len"])
    raw_len = int(cache_cfg["raw_len"])
    num_bins = int(cache_cfg["num_bins"])
    downsample = int(cache_cfg["downsample"])
    typewell_len = int(cache_cfg["typewell_len"])
    typewell_window = float(cache_cfg["typewell_window"])
    merge_k = max(1, int(cache_cfg.get("suffix_merge_k", 1)))
    merge_likelihood_mode = str(cache_cfg.get("suffix_merge_likelihood_mode", "block")).lower()
    if merge_likelihood_mode not in {"mean", "block"}:
        raise ValueError(f"unknown PF suffix merge likelihood mode={merge_likelihood_mode!r}")
    merge_default_alpha = 1.0 if merge_likelihood_mode == "block" else 0.5
    merge_power_alpha = float(cache_cfg.get("suffix_merge_obs_power_alpha", merge_default_alpha))
    merge_state_mode = str(cache_cfg.get("suffix_merge_state_mode", "start")).lower()
    if merge_state_mode not in {"start", "end"}:
        raise ValueError(f"unknown PF suffix merge state mode={merge_state_mode!r}")

    window_tvt = np.full(num_bins, np.nan, dtype=np.float32)
    window_tvt_rel = np.full(num_bins, np.nan, dtype=np.float32)
    window_has_tvt = np.zeros(num_bins, dtype=bool)
    window_orig_index = np.full(num_bins, np.nan, dtype=np.float32)
    target_mask = np.zeros(num_bins, dtype=bool)

    prefix_start = max(0, last_seen_idx + 1 - prefix_len)
    _fill_window_tvt_bins(
        h_df=h_df,
        last_seen_idx=last_seen_idx,
        prefix_start=prefix_start,
        tvt0=tvt0,
        raw_len=raw_len,
        prefix_len=prefix_len,
        target_len=int(cache_cfg["target_len"]),
        downsample=downsample,
        num_bins=num_bins,
        window_tvt=window_tvt,
        window_tvt_rel=window_tvt_rel,
        window_has_tvt=window_has_tvt,
        window_orig_index=window_orig_index,
        target_mask=target_mask,
        has_target=has_target,
    )

    pf_prob = np.zeros((num_bins, typewell_len), dtype=np.float32)
    pf_tvt_pred = np.zeros(num_bins, dtype=np.float32)
    pf_tvt_rel_pred = np.zeros(num_bins, dtype=np.float32)
    pf_prob_ffbsi = np.zeros((num_bins, typewell_len), dtype=np.float32)
    pf_tvt_pred_ffbsi = np.zeros(num_bins, dtype=np.float32)
    pf_tvt_rel_pred_ffbsi = np.zeros(num_bins, dtype=np.float32)
    if kept_suffix_len > 0:
        query_md = h_df["MD"].to_numpy(dtype=np.float64)[suffix_start : suffix_start + kept_suffix_len]
        query_z = h_df["Z"].to_numpy(dtype=np.float64)[suffix_start : suffix_start + kept_suffix_len]
        tw_mean_gr = float(np.nanmean(tw_gr_default))
        if not np.isfinite(tw_mean_gr):
            tw_mean_gr = 0.0
        raw_gr = h_df["GR"].to_numpy(dtype=np.float64)
        gr_interp = _interpolate_horizontal_gr(h_df["GR"], tw_mean_gr)
        query_gr_mode = str(cache_cfg.get("query_gr_mode", "interp"))
        if query_gr_mode not in {"interp", "soft_interp", "skip"}:
            raise ValueError(f"unknown PF query_gr_mode={query_gr_mode!r}")
        query_raw_gr = raw_gr[suffix_start : suffix_start + kept_suffix_len]
        query_gr = gr_interp.fillna(tw_mean_gr).to_numpy(dtype=np.float64)[suffix_start : suffix_start + kept_suffix_len]
        finite_query_gr = np.isfinite(query_raw_gr)
        if query_gr_mode == "interp":
            query_gr_power = np.ones_like(query_gr, dtype=np.float64)
        elif query_gr_mode == "skip":
            query_gr_power = finite_query_gr.astype(np.float64)
        else:
            missing_power = float(cache_cfg.get("missing_likelihood_power", 1.0))
            query_gr_power = np.where(finite_query_gr, 1.0, missing_power).astype(np.float64)
        gap_decay = float(cache_cfg.get("missing_gap_decay", 0.0))
        if gap_decay > 0.0 and (~finite_query_gr).any():
            missing_min_power = float(cache_cfg.get("missing_min_power", 0.05))
            missing_min_power = float(np.clip(missing_min_power, 0.0, 1.0))
            gap_len = 0
            for i in range(len(query_gr_power)):
                if finite_query_gr[i]:
                    gap_len = 0
                else:
                    gap_len += 1
                    decayed = math.exp(-float(gap_len) / gap_decay)
                    query_gr_power[i] *= max(missing_min_power, decayed)
        if merge_k > 1:
            (
                query_md_pf,
                query_z_pf,
                query_gr_pf,
                query_gr_power_pf,
                query_gr_finite_pf,
                query_obs_gr_pf,
                query_obs_power_pf,
                query_obs_finite_pf,
                query_obs_md_delta_pf,
                query_obs_z_pf,
                query_obs_count_pf,
                merge_block_starts,
                merge_block_ends,
            ) = _merge_suffix_query_blocks(
                query_md=query_md,
                query_z=query_z,
                query_gr=query_gr,
                query_gr_power=query_gr_power,
                query_gr_finite=finite_query_gr.astype(np.float64),
                merge_k=merge_k,
                power_alpha=merge_power_alpha,
                likelihood_mode=merge_likelihood_mode,
                state_mode=merge_state_mode,
            )
            seed_bin_idx = np.arange(len(query_md_pf), dtype=np.int64)
            seed_num_bins = int(len(query_md_pf))
        else:
            query_md_pf = query_md
            query_z_pf = query_z
            query_gr_pf = query_gr
            query_gr_power_pf = query_gr_power
            query_gr_finite_pf = finite_query_gr.astype(np.float64)
            query_obs_gr_pf = query_gr[:, None].astype(np.float64, copy=False)
            query_obs_power_pf = query_gr_power[:, None].astype(np.float64, copy=False)
            query_obs_finite_pf = finite_query_gr[:, None].astype(np.float64, copy=False)
            query_obs_md_delta_pf = np.zeros((len(query_md), 1), dtype=np.float64)
            query_obs_z_pf = query_z[:, None].astype(np.float64, copy=False)
            query_obs_count_pf = np.ones(len(query_md), dtype=np.int64)
            merge_block_starts = np.arange(kept_suffix_len, dtype=np.int64)
            merge_block_ends = merge_block_starts + 1
            seed_bin_idx = (prefix_len + np.arange(kept_suffix_len, dtype=np.int64)) // downsample
            seed_num_bins = int(num_bins)
        # Shape correlation is applied at seed-ensemble time, not inside the
        # per-particle emission update. Passing neutral arrays keeps the older
        # experimental kernel signature stable while avoiding row-level motif
        # over-chasing and the large runtime cost of per-particle windows.
        shape_gr = np.zeros((1, 1), dtype=np.float64)
        shape_md_delta = np.zeros((1, 1), dtype=np.float64)
        shape_z = np.zeros((1, 1), dtype=np.float64)
        shape_count = np.zeros(1, dtype=np.int64)
        known_md = h_df["MD"].to_numpy(dtype=np.float64)[: suffix_start]
        known_z = h_df["Z"].to_numpy(dtype=np.float64)[: suffix_start]
        known_gr_for_sigma = _prefix_gr_for_pf_sigma(h_df["GR"].iloc[:suffix_start])
        known_tvt = tvt_input[: suffix_start]
        raw_typewell = raw_tw_df.sort_values("TVT")
        raw_tw_tvt = raw_typewell["TVT"].to_numpy(dtype=np.float64)
        raw_tw_gr = raw_typewell["GR"].fillna(raw_typewell["GR"].mean()).to_numpy(dtype=np.float64)
        local_score_source = str(cache_cfg.get("seed_local_score_source", "pf_likelihood")).lower()
        local_weight_mode = str(cache_cfg.get("seed_local_weight_mode", "off")).lower()
        needs_gr_path_local_score = (
            local_weight_mode not in {"off", "none", "global", "likelihood"}
            and local_score_source in {"gr_path", "path_gr", "path_resid", "path_residual", "gr_resid", "gr_residual"}
        )
        if local_score_source not in {
            "pf_likelihood",
            "pf",
            "filter",
            "predictive",
            "gr_path",
            "path_gr",
            "path_resid",
            "path_residual",
            "gr_resid",
            "gr_residual",
        }:
            raise ValueError(f"unknown PF seed local score source={local_score_source!r}")
        seeds = _sample_pf_seeds(int(cache_cfg["base_seed"]), int(cache_cfg["n_seeds"]))
        profiles = list(cache_cfg.get("profile_mixture_spec", []))
        profile_indices = _profile_indices_for_seed_count(
            profiles,
            len(seeds),
            shuffle_seed=int(cache_cfg["base_seed"]),
        )
        needs_shape_bonus = float(cache_cfg.get("gr_shape_power", 0.0)) > 0.0
        if not needs_shape_bonus and profiles:
            for profile in profiles:
                profile_cfg = _apply_profile_to_cache_cfg(cache_cfg, profile)
                if float(profile_cfg.get("gr_shape_power", 0.0)) > 0.0:
                    needs_shape_bonus = True
                    break
        if needs_shape_bonus:
            query_gr_bin, query_gr_bin_count = _build_gr_shape_bin_series(
                query_gr_pf,
                query_gr_finite_pf,
                seed_bin_idx,
                seed_num_bins,
            )
        else:
            query_gr_bin = np.empty(0, dtype=np.float64)
            query_gr_bin_count = np.empty(0, dtype=np.int64)
        seed_paths = []
        seed_liks = []
        seed_probs = []
        seed_paths_ffbsi = []
        seed_probs_ffbsi = []
        seed_local_log_liks = []
        seed_local_obs_power = []
        has_ffbsi_seed = False
        for seed_idx, seed in enumerate(seeds):
            seed_cache_cfg = cache_cfg
            seed_tw_tvt = tw_tvt_default
            seed_tw_gr = tw_gr_default
            if profiles:
                seed_cache_cfg = _apply_profile_to_cache_cfg(
                    cache_cfg,
                    profiles[int(profile_indices[seed_idx])],
                )
                if (
                    seed_cache_cfg.get("typewell_gr_calibration") != cache_cfg.get("typewell_gr_calibration")
                    or abs(float(seed_cache_cfg.get("seen_blend_weight", 0.7)) - float(cache_cfg.get("seen_blend_weight", 0.7))) > 1e-12
                ):
                    seed_tw_tvt, seed_tw_gr = _typewell_arrays_for_cache_cfg(
                        h_df,
                        raw_tw_df,
                        seed_cache_cfg,
                    )
            seed_cache_cfg = _effective_pf_cache_cfg_for_merge(seed_cache_cfg)
            one = _run_numba_pf_heatmap_bins(
                query_md=query_md_pf,
                query_z=query_z_pf,
                query_gr=query_gr_pf,
                query_gr_power=query_gr_power_pf,
                query_gr_finite=query_gr_finite_pf,
                obs_gr=query_obs_gr_pf,
                obs_power=query_obs_power_pf,
                obs_finite=query_obs_finite_pf,
                obs_md_delta=query_obs_md_delta_pf,
                obs_z=query_obs_z_pf,
                obs_count=query_obs_count_pf,
                obs_likelihood_mode=1 if (merge_k > 1 and merge_likelihood_mode == "block") else 0,
                shape_gr=shape_gr,
                shape_md_delta=shape_md_delta,
                shape_z=shape_z,
                shape_count=shape_count,
                known_md=known_md,
                known_z=known_z,
                known_gr_for_sigma=known_gr_for_sigma,
                known_tvt=known_tvt,
                tw_tvt=seed_tw_tvt,
                tw_gr=seed_tw_gr,
                raw_tw_tvt=raw_tw_tvt,
                raw_tw_gr=raw_tw_gr,
                tvt0=tvt0,
                bin_idx=seed_bin_idx,
                num_bins=seed_num_bins,
                typewell_window=typewell_window,
                typewell_len=typewell_len,
                n_particles=int(cache_cfg["n_particles"]),
                seed=int(seed),
                cache_cfg=seed_cache_cfg,
            )
            seed_paths.append(one["pf_tvt_pred"])
            seed_paths_ffbsi.append(one.get("pf_tvt_pred_ffbsi", one["pf_tvt_pred"]))
            if needs_gr_path_local_score:
                seed_local_log_lik, seed_local_power = _seed_gr_path_local_evidence(
                    seed_tvt_pred=one["pf_tvt_pred"],
                    obs_gr=query_obs_gr_pf,
                    obs_power=query_obs_power_pf,
                    obs_finite=query_obs_finite_pf,
                    obs_z=query_obs_z_pf,
                    obs_count=query_obs_count_pf,
                    tw_tvt=seed_tw_tvt,
                    tw_gr=seed_tw_gr,
                    raw_tw_tvt=raw_tw_tvt,
                    raw_tw_gr=raw_tw_gr,
                    raw_ref_weight=float(seed_cache_cfg.get("raw_ref_likelihood_weight", 0.0)),
                    gr_sigma=float(one.get("gr_sigma", np.nan)),
                    state_z_weight=float(seed_cache_cfg.get("pf_state_z_weight", 1.0)),
                )
            else:
                seed_local_log_lik = np.asarray(
                    one.get("local_log_lik", np.zeros(seed_num_bins, dtype=np.float64)),
                    dtype=np.float64,
                )
                seed_local_power = np.asarray(
                    one.get("local_obs_power", np.zeros(seed_num_bins, dtype=np.float64)),
                    dtype=np.float64,
                )
            seed_shape_bonus = 0.0
            if needs_shape_bonus:
                seed_shape_bonus = _pf_seed_gr_shape_bonus(
                    query_gr_bin=query_gr_bin,
                    query_gr_bin_count=query_gr_bin_count,
                    seed_tvt_pred=one["pf_tvt_pred"],
                    tw_tvt=seed_tw_tvt,
                    tw_gr=seed_tw_gr,
                    raw_tw_tvt=raw_tw_tvt,
                    raw_tw_gr=raw_tw_gr,
                    gr_shape_power=float(seed_cache_cfg.get("gr_shape_power", 0.0)),
                    gr_shape_mode=str(seed_cache_cfg.get("gr_shape_mode", "resid_corr")),
                    gr_shape_window=int(seed_cache_cfg.get("gr_shape_window", 15)),
                    gr_shape_min_points=int(seed_cache_cfg.get("gr_shape_min_points", 7)),
                    gr_shape_sigma_floor=float(seed_cache_cfg.get("gr_shape_sigma_floor", 0.35)),
                    gr_shape_ref_mode=str(seed_cache_cfg.get("gr_shape_ref_mode", "primary")),
                    raw_ref_weight=float(seed_cache_cfg.get("raw_ref_likelihood_weight", 0.0)),
                )
            seed_liks.append(float(one["log_lik"]) + seed_shape_bonus)
            seed_probs.append(one["pf_prob"])
            seed_probs_ffbsi.append(one.get("pf_prob_ffbsi", one["pf_prob"]))
            seed_local_log_liks.append(seed_local_log_lik)
            seed_local_obs_power.append(seed_local_power)
            has_ffbsi_seed = has_ffbsi_seed or bool(one.get("ffbsi_ok", False))
        seed_liks = np.asarray(seed_liks, dtype=np.float64)
        prob_stack = np.stack(seed_probs, axis=0)
        prob_stack_ffbsi = np.stack(seed_probs_ffbsi, axis=0)
        if profiles:
            baseline_prob_weights = _profile_seed_ensemble_weights(
                seed_liks,
                profile_indices,
                profiles,
                mode=str(cache_cfg.get("seed_prob_weight_mode", "equal")),
                lik_scale=float(cache_cfg.get("lik_scale", 5.0)),
                mode_key="seed_prob_weight_mode",
            )
            baseline_path_weights = _profile_seed_ensemble_weights(
                seed_liks,
                profile_indices,
                profiles,
                mode=str(cache_cfg.get("seed_path_weight_mode", "likelihood")),
                lik_scale=float(cache_cfg.get("lik_scale", 5.0)),
                mode_key="seed_path_weight_mode",
            )
        else:
            baseline_prob_weights = _seed_ensemble_weights(
                seed_liks,
                mode=str(cache_cfg.get("seed_prob_weight_mode", "equal")),
                lik_scale=float(cache_cfg.get("lik_scale", 5.0)),
            )
            baseline_path_weights = _seed_ensemble_weights(
                seed_liks,
                mode=str(cache_cfg.get("seed_path_weight_mode", "likelihood")),
                lik_scale=float(cache_cfg.get("lik_scale", 5.0)),
            )
        local_bin_weights = _seed_local_bin_weights(
            seed_liks=seed_liks,
            local_log_liks=np.stack(seed_local_log_liks, axis=0),
            local_obs_power=np.stack(seed_local_obs_power, axis=0),
            mode=str(cache_cfg.get("seed_local_weight_mode", "off")),
            lik_scale=float(cache_cfg.get("seed_local_lik_scale", cache_cfg.get("lik_scale", 5.0))),
            half_life_blocks=float(cache_cfg.get("seed_local_half_life_blocks", 8.0)),
            global_mix=float(cache_cfg.get("seed_local_global_mix", 0.0)),
            min_power=float(cache_cfg.get("seed_local_min_power", 1e-6)),
            profile_indices=profile_indices,
            profiles=profiles,
            global_weights=baseline_path_weights,
        )
        seed_local_combine_mode = str(cache_cfg.get("seed_local_combine_mode", "replace")).lower()
        pf_prob_base = (prob_stack * baseline_prob_weights[:, None, None]).sum(axis=0).astype(np.float32)
        pf_prob_base = _normalize_prob_rows(pf_prob_base)
        pf_prob_ffbsi_base = (prob_stack_ffbsi * baseline_prob_weights[:, None, None]).sum(axis=0).astype(np.float32)
        pf_prob_ffbsi_base = _normalize_prob_rows(pf_prob_ffbsi_base)
        if local_bin_weights is not None and seed_local_combine_mode == "replace":
            pf_prob = _combine_seed_probs_with_bin_weights(prob_stack, local_bin_weights)
            pf_prob_ffbsi = _combine_seed_probs_with_bin_weights(prob_stack_ffbsi, local_bin_weights)
        else:
            pf_prob = pf_prob_base
            pf_prob_ffbsi = pf_prob_ffbsi_base
        prob_temperature = float(cache_cfg.get("pf_prob_temperature", 1.0))
        if prob_temperature > 0.0 and abs(prob_temperature - 1.0) > 1e-12:
            pf_prob = np.power(pf_prob, np.float32(1.0 / prob_temperature)).astype(np.float32)
            pf_prob = _normalize_prob_rows(pf_prob)
            pf_prob_ffbsi = np.power(pf_prob_ffbsi, np.float32(1.0 / prob_temperature)).astype(np.float32)
            pf_prob_ffbsi = _normalize_prob_rows(pf_prob_ffbsi)
        axis_sigma = float(cache_cfg.get("axis_sigma", 0.0))
        if axis_sigma > 0.0:
            if gaussian_filter1d is None:
                raise ImportError("scipy.ndimage.gaussian_filter1d is required for PF_heatmap_axis_sigma > 0")
            pf_prob = gaussian_filter1d(pf_prob, sigma=axis_sigma, axis=1, mode="constant")
            pf_prob = _normalize_prob_rows(pf_prob)
            pf_prob_ffbsi = gaussian_filter1d(pf_prob_ffbsi, sigma=axis_sigma, axis=1, mode="constant")
            pf_prob_ffbsi = _normalize_prob_rows(pf_prob_ffbsi)

        path_stack = np.stack(seed_paths, axis=0)
        path_stack_ffbsi = np.stack(seed_paths_ffbsi, axis=0)
        if profiles:
            pf_tvt_pred_base = _combine_seed_paths_with_weights(path_stack, baseline_path_weights)
            pf_tvt_pred_ffbsi_base = _combine_seed_paths_with_weights(path_stack_ffbsi, baseline_path_weights)
        else:
            pf_tvt_pred_base = _combine_seed_paths(
                path_stack,
                seed_liks,
                mode=str(cache_cfg.get("seed_path_weight_mode", "likelihood")),
                lik_scale=float(cache_cfg.get("lik_scale", 5.0)),
            )
            pf_tvt_pred_ffbsi_base = _combine_seed_paths(
                path_stack_ffbsi,
                seed_liks,
                mode=str(cache_cfg.get("seed_path_weight_mode", "likelihood")),
                lik_scale=float(cache_cfg.get("lik_scale", 5.0)),
            )
        if local_bin_weights is not None and seed_local_combine_mode == "replace":
            pf_tvt_pred = _combine_seed_paths_with_bin_weights(path_stack, local_bin_weights)
            pf_tvt_pred_ffbsi = _combine_seed_paths_with_bin_weights(path_stack_ffbsi, local_bin_weights)
        elif local_bin_weights is not None and seed_local_combine_mode in {"residual", "residual_blend", "delta"}:
            local_path = _combine_seed_paths_with_bin_weights(path_stack, local_bin_weights)
            local_path_ffbsi = _combine_seed_paths_with_bin_weights(path_stack_ffbsi, local_bin_weights)
            residual_alpha = float(np.clip(cache_cfg.get("seed_local_residual_alpha", 0.0), 0.0, 1.0))
            residual_clip = float(cache_cfg.get("seed_local_residual_clip", 5.0))
            if residual_clip > 0.0:
                delta = np.clip(local_path - pf_tvt_pred_base, -residual_clip, residual_clip)
                delta_ffbsi = np.clip(local_path_ffbsi - pf_tvt_pred_ffbsi_base, -residual_clip, residual_clip)
            else:
                delta = local_path - pf_tvt_pred_base
                delta_ffbsi = local_path_ffbsi - pf_tvt_pred_ffbsi_base
            pf_tvt_pred = (pf_tvt_pred_base + residual_alpha * delta).astype(np.float32)
            pf_tvt_pred_ffbsi = (pf_tvt_pred_ffbsi_base + residual_alpha * delta_ffbsi).astype(np.float32)
        else:
            pf_tvt_pred = pf_tvt_pred_base
            pf_tvt_pred_ffbsi = pf_tvt_pred_ffbsi_base
        if not has_ffbsi_seed:
            pf_prob_ffbsi = pf_prob.copy()
            pf_tvt_pred_ffbsi = pf_tvt_pred.copy()
        if merge_k > 1:
            pf_prob = _expand_merged_prob_to_bins(
                pf_prob,
                merge_block_starts,
                merge_block_ends,
                prefix_len,
                downsample,
                num_bins,
            )
            pf_prob_ffbsi = _expand_merged_prob_to_bins(
                pf_prob_ffbsi,
                merge_block_starts,
                merge_block_ends,
                prefix_len,
                downsample,
                num_bins,
            )
            pf_tvt_pred = _downsample_suffix_values(
                _row_path_from_merged_steps(
                    pf_tvt_pred,
                    tvt0,
                    merge_block_starts,
                    merge_block_ends,
                    kept_suffix_len,
                ),
                prefix_len,
                downsample,
                num_bins,
            )
            pf_tvt_pred_ffbsi = _downsample_suffix_values(
                _row_path_from_merged_steps(
                    pf_tvt_pred_ffbsi,
                    tvt0,
                    merge_block_starts,
                    merge_block_ends,
                    kept_suffix_len,
                ),
                prefix_len,
                downsample,
                num_bins,
            )
        pf_tvt_rel_pred = (pf_tvt_pred - np.float32(tvt0)).astype(np.float32)
        pf_tvt_rel_pred_ffbsi = (pf_tvt_pred_ffbsi - np.float32(tvt0)).astype(np.float32)
        pf_tvt_pred = np.nan_to_num(pf_tvt_pred, nan=0.0).astype(np.float32)
        pf_tvt_rel_pred = np.nan_to_num(pf_tvt_rel_pred, nan=0.0).astype(np.float32)
        pf_tvt_pred_ffbsi = np.nan_to_num(pf_tvt_pred_ffbsi, nan=0.0).astype(np.float32)
        pf_tvt_rel_pred_ffbsi = np.nan_to_num(pf_tvt_rel_pred_ffbsi, nan=0.0).astype(np.float32)

    fixed_path_bins = window_has_tvt & ~target_mask
    if fixed_path_bins.any():
        _paint_fixed_path_density(
            pf_prob=pf_prob,
            pf_tvt_pred=pf_tvt_pred,
            pf_tvt_rel_pred=pf_tvt_rel_pred,
            tvt0=tvt0,
            tvt_rel=window_tvt_rel,
            valid=fixed_path_bins,
            typewell_window=typewell_window,
            typewell_len=typewell_len,
        )
        _paint_fixed_path_density(
            pf_prob=pf_prob_ffbsi,
            pf_tvt_pred=pf_tvt_pred_ffbsi,
            pf_tvt_rel_pred=pf_tvt_rel_pred_ffbsi,
            tvt0=tvt0,
            tvt_rel=window_tvt_rel,
            valid=fixed_path_bins,
            typewell_window=typewell_window,
            typewell_len=typewell_len,
        )

    valid_rows = target_mask & np.isfinite(window_tvt) & np.isfinite(pf_tvt_pred) & (pf_tvt_pred != 0.0)
    if has_target and valid_rows.any():
        err = pf_tvt_pred[valid_rows] - window_tvt[valid_rows]
        tvt_sse = float(np.square(err).sum())
        target_count = int(valid_rows.sum())
        tvt_rmse = math.sqrt(tvt_sse / target_count)
    else:
        tvt_sse = math.nan
        target_count = 0
        tvt_rmse = math.nan
    valid_rows_ffbsi = target_mask & np.isfinite(window_tvt) & np.isfinite(pf_tvt_pred_ffbsi) & (pf_tvt_pred_ffbsi != 0.0)
    if has_target and valid_rows_ffbsi.any():
        err_ffbsi = pf_tvt_pred_ffbsi[valid_rows_ffbsi] - window_tvt[valid_rows_ffbsi]
        tvt_sse_ffbsi = float(np.square(err_ffbsi).sum())
        target_count_ffbsi = int(valid_rows_ffbsi.sum())
        tvt_rmse_ffbsi = math.sqrt(tvt_sse_ffbsi / target_count_ffbsi)
    else:
        tvt_sse_ffbsi = math.nan
        target_count_ffbsi = 0
        tvt_rmse_ffbsi = math.nan
    prob_sums = pf_prob.sum(axis=1)
    active_prob = prob_sums[prob_sums > 0]
    prob_sums_ffbsi = pf_prob_ffbsi.sum(axis=1)
    active_prob_ffbsi = prob_sums_ffbsi[prob_sums_ffbsi > 0]
    return {
        "last_seen_idx": last_seen_idx,
        "tvt0": tvt0,
        "prefix_start": prefix_start,
        "suffix_len": suffix_len,
        "pf_prob": pf_prob,
        "pf_tvt_pred": pf_tvt_pred,
        "pf_tvt_rel_pred": pf_tvt_rel_pred,
        "pf_prob_ffbsi": pf_prob_ffbsi,
        "pf_tvt_pred_ffbsi": pf_tvt_pred_ffbsi,
        "pf_tvt_rel_pred_ffbsi": pf_tvt_rel_pred_ffbsi,
        "window_tvt": np.nan_to_num(window_tvt, nan=0.0).astype(np.float32),
        "window_tvt_rel": np.nan_to_num(window_tvt_rel, nan=0.0).astype(np.float32),
        "window_has_tvt": window_has_tvt,
        "window_orig_index": np.nan_to_num(window_orig_index, nan=-1.0).astype(np.float32),
        "target_mask": target_mask,
        "target_count": target_count,
        "tvt_sse": tvt_sse,
        "tvt_rmse": tvt_rmse,
        "prob_row_sum_min": float(active_prob.min()) if active_prob.size else math.nan,
        "prob_row_sum_max": float(active_prob.max()) if active_prob.size else math.nan,
        "target_count_ffbsi": target_count_ffbsi,
        "tvt_sse_ffbsi": tvt_sse_ffbsi,
        "tvt_rmse_ffbsi": tvt_rmse_ffbsi,
        "prob_row_sum_min_ffbsi": float(active_prob_ffbsi.min()) if active_prob_ffbsi.size else math.nan,
        "prob_row_sum_max_ffbsi": float(active_prob_ffbsi.max()) if active_prob_ffbsi.size else math.nan,
    }


def _fill_window_tvt_bins(
    h_df,
    last_seen_idx,
    prefix_start,
    tvt0,
    raw_len,
    prefix_len,
    target_len,
    downsample,
    num_bins,
    window_tvt,
    window_tvt_rel,
    window_has_tvt,
    window_orig_index,
    target_mask,
    has_target,
):
    raw_orig_index = np.full(raw_len, np.nan, dtype=np.float32)
    row_orig_index = np.arange(len(h_df), dtype=np.float32)
    raw_tvt = np.full(raw_len, np.nan, dtype=np.float32)
    raw_target = np.zeros(raw_len, dtype=bool)
    prefix_size = last_seen_idx + 1 - prefix_start
    prefix_dest_start = prefix_len - prefix_size
    tvt_input = h_df["TVT_input"].to_numpy(dtype=np.float32)
    if prefix_size > 0:
        raw_tvt[prefix_dest_start:prefix_len] = tvt_input[prefix_start : last_seen_idx + 1]
        raw_orig_index[prefix_dest_start:prefix_len] = row_orig_index[prefix_start : last_seen_idx + 1]
    suffix_size = min(len(h_df) - (last_seen_idx + 1), target_len)
    if suffix_size > 0:
        dst = slice(prefix_len, prefix_len + suffix_size)
        raw_orig_index[dst] = row_orig_index[last_seen_idx + 1 : last_seen_idx + 1 + suffix_size]
        if has_target and "TVT" in h_df.columns:
            tvt = h_df["TVT"].to_numpy(dtype=np.float32)
            raw_tvt[dst] = tvt[last_seen_idx + 1 : last_seen_idx + 1 + suffix_size]
            raw_target[dst] = True
    raw_has_tvt = np.isfinite(raw_tvt)
    bin_idx = np.arange(raw_len, dtype=np.int64) // int(downsample)
    valid = raw_has_tvt & (bin_idx < num_bins)
    if valid.any():
        sums = np.bincount(bin_idx[valid], weights=raw_tvt[valid], minlength=num_bins)
        counts = np.bincount(bin_idx[valid], minlength=num_bins)
        filled = counts > 0
        window_tvt[filled] = (sums[filled] / counts[filled]).astype(np.float32)
        window_tvt_rel[filled] = window_tvt[filled] - np.float32(tvt0)
        window_has_tvt[filled] = True
    target_counts = np.bincount(bin_idx[raw_target & (bin_idx < num_bins)], minlength=num_bins)
    target_mask[:] = target_counts > 0
    _fill_orig_index_bins(raw_orig_index, downsample, num_bins, window_orig_index)


def _paint_fixed_path_density(
    pf_prob,
    pf_tvt_pred,
    pf_tvt_rel_pred,
    tvt0,
    tvt_rel,
    valid,
    typewell_window,
    typewell_len,
):
    axis0 = -float(typewell_window) + float(typewell_window) / int(typewell_len)
    axis_step = 2.0 * float(typewell_window) / int(typewell_len)
    rows = np.flatnonzero(valid & np.isfinite(tvt_rel))
    for row in rows:
        rel = float(tvt_rel[row])
        grid_pos = (rel - axis0) / axis_step
        left = int(np.floor(grid_pos))
        frac = grid_pos - left
        pf_prob[row, :] = 0.0
        if 0 <= left < int(typewell_len):
            pf_prob[row, left] += np.float32(1.0 - frac)
        right = left + 1
        if 0 <= right < int(typewell_len):
            pf_prob[row, right] += np.float32(frac)
        row_sum = float(pf_prob[row].sum())
        if row_sum > 0.0:
            pf_prob[row] /= np.float32(row_sum)
            pf_tvt_rel_pred[row] = np.float32(rel)
            pf_tvt_pred[row] = np.float32(float(tvt0) + rel)


def _fill_orig_index_bins(raw_orig_index, downsample, num_bins, window_orig_index):
    bin_idx = np.arange(len(raw_orig_index), dtype=np.int64) // int(downsample)
    valid = np.isfinite(raw_orig_index) & (bin_idx < int(num_bins))
    if not valid.any():
        return
    sums = np.bincount(bin_idx[valid], weights=raw_orig_index[valid], minlength=int(num_bins))
    counts = np.bincount(bin_idx[valid], minlength=int(num_bins))
    filled = counts > 0
    window_orig_index[filled] = (sums[filled] / counts[filled]).astype(np.float32)


def _sample_pf_seeds(base_seed, n_seeds):
    rng = np.random.default_rng(int(base_seed))
    return rng.integers(0, np.iinfo(np.int32).max, size=int(n_seeds), dtype=np.int64)


def _single_pf_seed(base_seed):
    return int(_sample_pf_seeds(base_seed, 1)[0])


def _normalize_prob_rows(prob):
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    prob[prob < 0.0] = 0.0
    row_sum = prob.sum(axis=1, keepdims=True)
    np.divide(prob, row_sum, out=prob, where=row_sum > 0.0)
    return prob


def _downsample_suffix_values(values, prefix_len, downsample, num_bins):
    out = np.full(int(num_bins), np.nan, dtype=np.float32)
    if len(values) == 0:
        return out
    bin_idx = (int(prefix_len) + np.arange(len(values), dtype=np.int64)) // int(downsample)
    valid = bin_idx < int(num_bins)
    sums = np.bincount(
        bin_idx[valid],
        weights=np.asarray(values, dtype=np.float32)[valid],
        minlength=int(num_bins),
    )
    counts = np.bincount(bin_idx[valid], minlength=int(num_bins))
    filled = counts > 0
    out[filled] = (sums[filled] / counts[filled]).astype(np.float32)
    return out


def _pf_arrays_from_hist(hist_bins, path_sums, count_bins, num_bins, typewell_len):
    pf_prob = np.zeros((int(num_bins), int(typewell_len)), dtype=np.float32)
    valid_bins = count_bins > 0
    pf_prob[valid_bins] = (hist_bins[valid_bins] / count_bins[valid_bins, None]).astype(np.float32)
    pf_prob = _normalize_prob_rows(pf_prob)
    pf_tvt_pred = np.full(int(num_bins), np.nan, dtype=np.float32)
    pf_tvt_pred[valid_bins] = (path_sums[valid_bins] / count_bins[valid_bins]).astype(np.float32)
    return pf_prob, pf_tvt_pred


def _local_gr_ambiguity_power(
    query_gr,
    query_gr_finite,
    tvt0,
    tw_tvt,
    tw_gr,
    gr_sigma,
    typewell_window,
    ambiguity_power,
    min_power,
    contrast,
):
    out = np.ones(len(query_gr), dtype=np.float64)
    ambiguity_power = float(ambiguity_power)
    if ambiguity_power <= 0.0 or len(query_gr) == 0:
        return out
    if not np.isfinite(gr_sigma) or gr_sigma <= 0.0:
        return out

    local = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    local &= (tw_tvt >= float(tvt0) - float(typewell_window))
    local &= (tw_tvt <= float(tvt0) + float(typewell_window))
    if local.sum() < 4:
        return out

    local_gr = tw_gr[local].astype(np.float64)
    min_power = float(np.clip(min_power, 0.0, 1.0))
    contrast = max(float(contrast), 1e-6)
    inv_sigma = 1.0 / float(gr_sigma)
    for i in range(len(query_gr)):
        if query_gr_finite[i] <= 0.0 or not np.isfinite(query_gr[i]):
            continue
        d = (float(query_gr[i]) - local_gr) * inv_sigma
        logits = -0.5 * d * d
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        prob_sum = float(probs.sum())
        if prob_sum <= 0.0 or not np.isfinite(prob_sum):
            continue
        probs /= prob_sum
        max_prob = float(np.max(probs))
        ess = 1.0 / max(float(np.sum(probs * probs)), 1e-12)
        unique = max(max_prob, 1.0 / ess)
        unique = float(np.clip(unique * contrast, 0.0, 1.0))
        out[i] = min_power + (1.0 - min_power) * (unique ** ambiguity_power)
    return out


def _local_gr_information_multiplier(
    query_gr,
    query_gr_finite,
    tvt0,
    tw_tvt,
    tw_gr,
    gr_sigma,
    typewell_window,
    information_power,
    center,
    min_multiplier,
    max_multiplier,
    slope_weight,
):
    out = np.ones(len(query_gr), dtype=np.float64)
    information_power = float(information_power)
    if information_power <= 0.0 or len(query_gr) == 0:
        return out
    if not np.isfinite(gr_sigma) or gr_sigma <= 0.0:
        return out

    local = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    local &= (tw_tvt >= float(tvt0) - float(typewell_window))
    local &= (tw_tvt <= float(tvt0) + float(typewell_window))
    if local.sum() < 4:
        return out

    local_tvt = tw_tvt[local].astype(np.float64)
    local_gr = tw_gr[local].astype(np.float64)
    order = np.argsort(local_tvt)
    local_tvt = local_tvt[order]
    local_gr = local_gr[order]
    min_multiplier = float(np.clip(min_multiplier, 0.05, 10.0))
    max_multiplier = float(np.clip(max_multiplier, min_multiplier, 10.0))
    center = max(float(center), 1e-6)
    slope_weight = float(np.clip(slope_weight, 0.0, 1.0))
    inv_sigma = 1.0 / float(gr_sigma)
    log_n = math.log(float(len(local_gr)))
    if slope_weight > 0.0:
        local_slope = np.abs(np.gradient(local_gr, local_tvt))
    else:
        local_slope = np.zeros_like(local_gr)

    for i in range(len(query_gr)):
        if query_gr_finite[i] <= 0.0 or not np.isfinite(query_gr[i]):
            continue
        d = (float(query_gr[i]) - local_gr) * inv_sigma
        logits = -0.5 * d * d
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        prob_sum = float(probs.sum())
        if prob_sum <= 0.0 or not np.isfinite(prob_sum):
            continue
        probs /= prob_sum
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-300)))) / max(log_n, 1e-12)
        entropy_info = float(np.clip(1.0 - entropy, 0.0, 1.0))
        if slope_weight > 0.0:
            slope_raw = float(np.sum(probs * local_slope)) * inv_sigma
            slope_info = slope_raw / (1.0 + slope_raw)
            info = (1.0 - slope_weight) * entropy_info + slope_weight * slope_info
        else:
            info = entropy_info
        multiplier = math.pow(max(info / center, 1e-6), information_power)
        out[i] = float(np.clip(multiplier, min_multiplier, max_multiplier))
    return out


def _gr_shape_mode_code(mode):
    mode = str(mode).lower()
    if mode in {"off", "none", "0"}:
        return 0
    if mode in {"resid_corr", "residual_corr", "centered", "center"}:
        return 1
    if mode in {"ncc", "pearson", "corr"}:
        return 2
    raise ValueError(f"unknown PF_heatmap_gr_shape_mode={mode!r}")


def _gr_shape_ref_mode_code(mode):
    mode = str(mode).lower()
    if mode in {"primary", "seen", "seen_blend"}:
        return 0
    if mode == "raw":
        return 1
    if mode in {"mixed", "point"}:
        return 2
    raise ValueError(f"unknown PF_heatmap_gr_shape_ref_mode={mode!r}")


def _build_gr_shape_windows(md, z, gr, gr_finite, window):
    window = int(window)
    if window <= 1:
        window = 1
    if window % 2 == 0:
        window += 1
    n = len(gr)
    shape_gr = np.zeros((n, window), dtype=np.float64)
    shape_md_delta = np.zeros((n, window), dtype=np.float64)
    shape_z = np.zeros((n, window), dtype=np.float64)
    shape_count = np.zeros(n, dtype=np.int64)
    if n == 0:
        return shape_gr, shape_md_delta, shape_z, shape_count
    md = np.asarray(md, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    gr = np.asarray(gr, dtype=np.float64)
    finite = np.asarray(gr_finite, dtype=np.float64) > 0.0
    for i in range(n):
        lo = max(0, i - window + 1)
        hi = i + 1
        slot = 0
        for row in range(lo, hi):
            if not finite[row] or not np.isfinite(gr[row]):
                continue
            if slot >= window:
                break
            shape_gr[i, slot] = gr[row]
            shape_md_delta[i, slot] = md[row] - md[i]
            shape_z[i, slot] = z[row]
            slot += 1
        shape_count[i] = slot
    return shape_gr, shape_md_delta, shape_z, shape_count


def _build_gr_shape_bin_series(gr, gr_finite, bin_idx, num_bins):
    gr = np.asarray(gr, dtype=np.float64)
    gr_finite = np.asarray(gr_finite, dtype=np.float64) > 0.0
    bin_idx = np.asarray(bin_idx, dtype=np.int64)
    num_bins = int(num_bins)
    out = np.full(num_bins, np.nan, dtype=np.float64)
    count = np.zeros(num_bins, dtype=np.int64)
    valid = gr_finite & np.isfinite(gr) & (bin_idx >= 0) & (bin_idx < num_bins)
    if valid.any():
        sums = np.bincount(bin_idx[valid], weights=gr[valid], minlength=num_bins)
        count = np.bincount(bin_idx[valid], minlength=num_bins).astype(np.int64, copy=False)
        filled = count > 0
        out[filled] = sums[filled] / count[filled]
    return out, count


def _pf_seed_gr_shape_bonus(
    query_gr_bin,
    query_gr_bin_count,
    seed_tvt_pred,
    tw_tvt,
    tw_gr,
    raw_tw_tvt,
    raw_tw_gr,
    gr_shape_power,
    gr_shape_mode,
    gr_shape_window,
    gr_shape_min_points,
    gr_shape_sigma_floor,
    gr_shape_ref_mode,
    raw_ref_weight,
):
    gr_shape_power = float(gr_shape_power)
    if gr_shape_power <= 0.0:
        return 0.0

    query_gr_bin = np.asarray(query_gr_bin, dtype=np.float64)
    query_gr_bin_count = np.asarray(query_gr_bin_count, dtype=np.int64)
    seed_tvt_pred = np.asarray(seed_tvt_pred, dtype=np.float64)
    valid = (
        np.isfinite(query_gr_bin)
        & np.isfinite(seed_tvt_pred)
        & np.isfinite(query_gr_bin_count)
        & (query_gr_bin_count > 0)
    )
    if not valid.any():
        return 0.0
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size < int(gr_shape_min_points):
        return 0.0

    gr_shape_window = int(gr_shape_window)
    if gr_shape_window > 0 and valid_idx.size > gr_shape_window:
        valid_idx = valid_idx[-gr_shape_window:]

    obs = query_gr_bin[valid_idx].astype(np.float64, copy=False)
    pred = np.interp(seed_tvt_pred[valid_idx], tw_tvt, tw_gr).astype(np.float64, copy=False)
    shape_ref_mode = str(gr_shape_ref_mode).lower()
    if shape_ref_mode == "raw" and raw_tw_tvt.size >= 3:
        pred = np.interp(seed_tvt_pred[valid_idx], raw_tw_tvt, raw_tw_gr).astype(np.float64, copy=False)
    elif shape_ref_mode in {"mixed", "point"} and raw_tw_tvt.size >= 3 and raw_ref_weight > 0.0:
        raw_pred = np.interp(seed_tvt_pred[valid_idx], raw_tw_tvt, raw_tw_gr).astype(np.float64, copy=False)
        pred = (1.0 - float(raw_ref_weight)) * pred + float(raw_ref_weight) * raw_pred

    shape_mode = str(gr_shape_mode).lower()
    if shape_mode == "resid_corr":
        if obs.size < 3 or pred.size < 3:
            return 0.0
        obs = np.diff(obs)
        pred = np.diff(pred)
        min_corr_points = max(2, int(gr_shape_min_points) - 1)
    elif shape_mode not in {"ncc", "pearson", "corr"}:
        raise ValueError(f"unknown PF_heatmap_gr_shape_mode={gr_shape_mode!r}")
    else:
        min_corr_points = int(gr_shape_min_points)

    if obs.size < min_corr_points or pred.size < min_corr_points:
        return 0.0

    obs = obs - float(np.mean(obs))
    pred = pred - float(np.mean(pred))
    obs_var = float(np.mean(obs * obs))
    pred_var = float(np.mean(pred * pred))
    if obs_var <= float(gr_shape_sigma_floor) ** 2 or pred_var <= float(gr_shape_sigma_floor) ** 2:
        return 0.0
    corr = float(np.mean(obs * pred) / math.sqrt(max(obs_var * pred_var, 1e-18)))
    corr = float(np.clip(corr, -1.0, 1.0))
    bonus = max(corr, 0.0)
    return float(gr_shape_power) * float(valid_idx.size) * bonus


def _seed_ensemble_weights(seed_liks, mode="likelihood", lik_scale=5.0):
    seed_liks = np.asarray(seed_liks, dtype=np.float64)
    if seed_liks.size == 0:
        return seed_liks
    finite = np.isfinite(seed_liks)
    if mode == "equal" or not finite.any():
        return np.ones(seed_liks.size, dtype=np.float64) / float(seed_liks.size)
    if mode == "winner":
        weights = np.zeros(seed_liks.size, dtype=np.float64)
        weights[int(np.nanargmax(seed_liks))] = 1.0
        return weights
    if mode == "rank":
        order = np.argsort(np.where(finite, seed_liks, -np.inf))[::-1]
        weights = np.zeros(seed_liks.size, dtype=np.float64)
        ranks = np.arange(seed_liks.size, 0, -1, dtype=np.float64)
        weights[order] = ranks
        return weights / weights.sum()
    if mode != "likelihood":
        raise ValueError(f"unknown seed ensemble weight mode={mode!r}")
    scale = max(float(lik_scale), 1e-6)
    shifted = np.where(finite, seed_liks, -np.inf) - np.max(seed_liks[finite])
    weights = np.exp(shifted / scale)
    weight_sum = weights.sum()
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return np.ones(seed_liks.size, dtype=np.float64) / float(seed_liks.size)
    return weights / weight_sum


def _profile_seed_ensemble_weights(
    seed_liks,
    profile_indices,
    profiles,
    mode="likelihood",
    lik_scale=5.0,
    mode_key=None,
):
    seed_liks = np.asarray(seed_liks, dtype=np.float64)
    profile_indices = np.asarray(profile_indices, dtype=np.int64)
    if not profiles or seed_liks.size == 0:
        return _seed_ensemble_weights(seed_liks, mode=mode, lik_scale=lik_scale)
    if profile_indices.size != seed_liks.size:
        raise ValueError("profile_indices must have one entry per seed likelihood")

    out = np.zeros(seed_liks.size, dtype=np.float64)
    profile_weights = np.asarray([float(profile["weight"]) for profile in profiles], dtype=np.float64)
    profile_weights /= profile_weights.sum()
    for profile_idx, profile_weight in enumerate(profile_weights):
        seed_mask = profile_indices == profile_idx
        if not seed_mask.any():
            continue
        profile_overrides = profiles[profile_idx].get("overrides", {})
        local_mode = str(profile_overrides.get(mode_key, mode)) if mode_key is not None else str(mode)
        local_lik_scale = float(profile_overrides.get("lik_scale", lik_scale))
        local_weights = _seed_ensemble_weights(
            seed_liks[seed_mask],
            mode=local_mode,
            lik_scale=local_lik_scale,
        )
        out[seed_mask] = profile_weight * local_weights

    total = out.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.ones(seed_liks.size, dtype=np.float64) / float(seed_liks.size)
    return out / total


def _combine_seed_paths(path_stack, seed_liks, mode="likelihood", lik_scale=5.0):
    path_stack = np.asarray(path_stack, dtype=np.float64)
    mode = str(mode)
    if mode == "median":
        return _nanmedian_seed_path(path_stack).astype(np.float32)
    if mode == "trimmed":
        return _trimmed_seed_path_mean(path_stack, trim_frac=0.10).astype(np.float32)
    if mode == "rank_trimmed":
        weights = _seed_ensemble_weights(seed_liks, mode="rank", lik_scale=lik_scale)
        return _weighted_trimmed_seed_path_mean(path_stack, weights, trim_frac=0.10).astype(np.float32)
    weights = _seed_ensemble_weights(seed_liks, mode=mode, lik_scale=lik_scale)
    valid_path = np.isfinite(path_stack)
    weighted = np.where(valid_path, path_stack, 0.0) * weights[:, None]
    denom = valid_path.astype(np.float64).T @ weights
    out = np.full(path_stack.shape[1], np.nan, dtype=np.float32)
    np.divide(weighted.sum(axis=0), denom, out=out, where=denom > 0)
    return out.astype(np.float32)


def _combine_seed_paths_with_weights(path_stack, weights):
    path_stack = np.asarray(path_stack, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if path_stack.ndim != 2:
        raise ValueError("path_stack must be a 2D seed x bin array")
    if weights.size != path_stack.shape[0]:
        raise ValueError("weights must have one value per seed path")
    valid_path = np.isfinite(path_stack)
    weighted = np.where(valid_path, path_stack, 0.0) * weights[:, None]
    denom = valid_path.astype(np.float64).T @ weights
    out = np.full(path_stack.shape[1], np.nan, dtype=np.float32)
    np.divide(weighted.sum(axis=0), denom, out=out, where=denom > 0)
    return out.astype(np.float32)


def _combine_seed_paths_with_bin_weights(path_stack, weights):
    path_stack = np.asarray(path_stack, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if path_stack.ndim != 2:
        raise ValueError("path_stack must be a 2D seed x bin array")
    if weights.shape != path_stack.shape:
        raise ValueError("weights must have the same seed x bin shape as path_stack")
    valid_path = np.isfinite(path_stack)
    weighted = np.where(valid_path, path_stack, 0.0) * weights
    denom = np.sum(np.where(valid_path, weights, 0.0), axis=0)
    out = np.full(path_stack.shape[1], np.nan, dtype=np.float32)
    np.divide(weighted.sum(axis=0), denom, out=out, where=denom > 0)
    return out.astype(np.float32)


def _normalize_seed_bin_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    out = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    col_sum = out.sum(axis=0)
    valid = col_sum > 0.0
    if valid.any():
        out[:, valid] /= col_sum[None, valid]
    if (~valid).any() and out.shape[0] > 0:
        out[:, ~valid] = 1.0 / float(out.shape[0])
    return out


def _seed_local_score_matrix(local_log_liks, local_obs_power, mode="off", half_life_blocks=8.0, min_power=1e-6):
    local_log_liks = np.asarray(local_log_liks, dtype=np.float64)
    local_obs_power = np.asarray(local_obs_power, dtype=np.float64)
    if local_log_liks.ndim != 2 or local_obs_power.shape != local_log_liks.shape:
        raise ValueError("local likelihood arrays must have matching seed x bin shapes")

    mode = str(mode or "off").lower()
    min_power = max(float(min_power), 1e-12)
    obs_power = np.where(np.isfinite(local_obs_power) & (local_obs_power > 0.0), local_obs_power, 0.0)
    log_lik = np.where(np.isfinite(local_log_liks), local_log_liks, 0.0)

    if mode in {"local", "point", "bin", "per_bin", "window"}:
        score = np.full_like(log_lik, np.nan, dtype=np.float64)
        np.divide(log_lik, obs_power, out=score, where=obs_power >= min_power)
        return score

    if mode in {"forward_decay", "forward", "decay"}:
        rho = _half_life_to_decay(half_life_blocks)
        score = np.full_like(log_lik, np.nan, dtype=np.float64)
        num = np.zeros(log_lik.shape[0], dtype=np.float64)
        den = np.zeros(log_lik.shape[0], dtype=np.float64)
        for b in range(log_lik.shape[1]):
            num = rho * num + log_lik[:, b]
            den = rho * den + obs_power[:, b]
            np.divide(num, den, out=score[:, b], where=den >= min_power)
        return score

    if mode in {"two_sided_decay", "twosided_decay", "two_sided", "bidirectional_decay"}:
        rho = _half_life_to_decay(half_life_blocks)
        f_num = np.zeros_like(log_lik, dtype=np.float64)
        f_den = np.zeros_like(log_lik, dtype=np.float64)
        num = np.zeros(log_lik.shape[0], dtype=np.float64)
        den = np.zeros(log_lik.shape[0], dtype=np.float64)
        for b in range(log_lik.shape[1]):
            num = rho * num + log_lik[:, b]
            den = rho * den + obs_power[:, b]
            f_num[:, b] = num
            f_den[:, b] = den

        b_num = np.zeros_like(log_lik, dtype=np.float64)
        b_den = np.zeros_like(log_lik, dtype=np.float64)
        num = np.zeros(log_lik.shape[0], dtype=np.float64)
        den = np.zeros(log_lik.shape[0], dtype=np.float64)
        for b in range(log_lik.shape[1] - 1, -1, -1):
            num = rho * num + log_lik[:, b]
            den = rho * den + obs_power[:, b]
            b_num[:, b] = num
            b_den[:, b] = den

        score = np.full_like(log_lik, np.nan, dtype=np.float64)
        total_num = f_num + b_num - log_lik
        total_den = f_den + b_den - obs_power
        np.divide(total_num, total_den, out=score, where=total_den >= min_power)
        return score

    raise ValueError(f"unknown local seed weight mode={mode!r}")


def _seed_gr_path_local_evidence(
    seed_tvt_pred,
    obs_gr,
    obs_power,
    obs_finite,
    obs_z,
    obs_count,
    tw_tvt,
    tw_gr,
    raw_tw_tvt,
    raw_tw_gr,
    raw_ref_weight,
    gr_sigma,
    state_z_weight,
):
    seed_tvt_pred = np.asarray(seed_tvt_pred, dtype=np.float64)
    obs_gr = np.asarray(obs_gr, dtype=np.float64)
    obs_power = np.asarray(obs_power, dtype=np.float64)
    obs_finite = np.asarray(obs_finite, dtype=np.float64)
    obs_z = np.asarray(obs_z, dtype=np.float64)
    obs_count = np.asarray(obs_count, dtype=np.int64)
    local_log_lik = np.zeros(seed_tvt_pred.shape[0], dtype=np.float64)
    local_obs_power = np.zeros(seed_tvt_pred.shape[0], dtype=np.float64)
    if seed_tvt_pred.size == 0:
        return local_log_lik, local_obs_power
    gr_sigma = float(gr_sigma)
    if not np.isfinite(gr_sigma) or gr_sigma <= 1e-6:
        gr_sigma = 10.0
    raw_ref_weight = float(np.clip(raw_ref_weight, 0.0, 1.0))
    use_raw = raw_ref_weight > 0.0 and len(raw_tw_tvt) >= 3
    obs_width = obs_gr.shape[1] if obs_gr.ndim == 2 else 0
    for b in range(seed_tvt_pred.shape[0]):
        base_tvt = seed_tvt_pred[b]
        if not np.isfinite(base_tvt) or obs_width == 0:
            continue
        count = int(obs_count[b]) if b < obs_count.size else obs_width
        count = max(0, min(count, obs_width))
        for obs_idx in range(count):
            power = float(obs_power[b, obs_idx])
            if power <= 0.0 or obs_finite[b, obs_idx] <= 0.0 or not np.isfinite(obs_gr[b, obs_idx]):
                continue
            output_obs_idx = count - 1
            if output_obs_idx < 0:
                output_obs_idx = 0
            output_z = obs_z[b, output_obs_idx]
            local_tvt = base_tvt + state_z_weight * (output_z - obs_z[b, obs_idx])
            expected_gr = float(np.interp(local_tvt, tw_tvt, tw_gr))
            if use_raw:
                raw_expected_gr = float(np.interp(local_tvt, raw_tw_tvt, raw_tw_gr))
                expected_gr = (1.0 - raw_ref_weight) * expected_gr + raw_ref_weight * raw_expected_gr
            d = (float(obs_gr[b, obs_idx]) - expected_gr) / gr_sigma
            d2 = min(d * d, 600.0)
            local_log_lik[b] += -0.5 * d2 * power
            local_obs_power[b] += power
    return local_log_lik, local_obs_power


def _half_life_to_decay(half_life_blocks):
    half_life_blocks = float(half_life_blocks)
    if not np.isfinite(half_life_blocks) or half_life_blocks <= 0.0:
        return 0.0
    return float(math.exp(math.log(0.5) / half_life_blocks))


def _seed_local_bin_weights(
    seed_liks,
    local_log_liks,
    local_obs_power,
    mode="off",
    lik_scale=1.0,
    half_life_blocks=8.0,
    global_mix=0.0,
    min_power=1e-6,
    profile_indices=None,
    profiles=None,
    global_weights=None,
):
    mode = str(mode or "off").lower()
    if mode in {"off", "none", "global", "likelihood"}:
        return None

    seed_liks = np.asarray(seed_liks, dtype=np.float64)
    score = _seed_local_score_matrix(
        local_log_liks,
        local_obs_power,
        mode=mode,
        half_life_blocks=half_life_blocks,
        min_power=min_power,
    )
    if score.shape[0] != seed_liks.size:
        raise ValueError("local seed likelihood arrays must have one row per seed")

    scale = max(float(lik_scale), 1e-6)
    weights = np.zeros_like(score, dtype=np.float64)
    if profiles:
        profile_indices = np.asarray(profile_indices, dtype=np.int64)
        if profile_indices.size != seed_liks.size:
            raise ValueError("profile_indices must have one entry per seed likelihood")
        profile_weights = np.asarray([float(profile["weight"]) for profile in profiles], dtype=np.float64)
        profile_weights /= profile_weights.sum()
        for profile_idx, profile_weight in enumerate(profile_weights):
            seed_mask = profile_indices == profile_idx
            if not seed_mask.any():
                continue
            local_score = score[seed_mask]
            for b in range(score.shape[1]):
                col = local_score[:, b]
                finite = np.isfinite(col)
                if not finite.any():
                    weights[seed_mask, b] = float(profile_weight) / float(seed_mask.sum())
                    continue
                shifted = np.where(finite, col, -np.inf) - np.max(col[finite])
                w = np.exp(shifted / scale)
                w_sum = w.sum()
                if w_sum <= 0.0 or not np.isfinite(w_sum):
                    weights[seed_mask, b] = float(profile_weight) / float(seed_mask.sum())
                else:
                    weights[seed_mask, b] = float(profile_weight) * w / w_sum
    else:
        for b in range(score.shape[1]):
            col = score[:, b]
            finite = np.isfinite(col)
            if not finite.any():
                weights[:, b] = 1.0 / float(score.shape[0])
                continue
            shifted = np.where(finite, col, -np.inf) - np.max(col[finite])
            w = np.exp(shifted / scale)
            w_sum = w.sum()
            if w_sum <= 0.0 or not np.isfinite(w_sum):
                weights[:, b] = 1.0 / float(score.shape[0])
            else:
                weights[:, b] = w / w_sum

    global_mix = float(np.clip(global_mix, 0.0, 1.0))
    if global_mix > 0.0:
        if global_weights is None:
            global_weights = _seed_ensemble_weights(seed_liks, mode="likelihood", lik_scale=lik_scale)
        else:
            global_weights = np.asarray(global_weights, dtype=np.float64)
        weights = (1.0 - global_mix) * weights + global_mix * global_weights[:, None]
    return _normalize_seed_bin_weights(weights)


def _combine_seed_probs_with_bin_weights(prob_stack, weights):
    prob_stack = np.asarray(prob_stack, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if prob_stack.ndim != 3:
        raise ValueError("prob_stack must be a 3D seed x bin x typewell array")
    if weights.shape != prob_stack.shape[:2]:
        raise ValueError("weights must have shape seed x bin")
    out = np.sum(prob_stack * weights[:, :, None], axis=0).astype(np.float32)
    return _normalize_prob_rows(out)


def _trimmed_seed_path_mean(path_stack, trim_frac=0.10):
    out = np.full(path_stack.shape[1], np.nan, dtype=np.float64)
    trim_frac = float(np.clip(trim_frac, 0.0, 0.45))
    for col in range(path_stack.shape[1]):
        vals = path_stack[:, col]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        vals.sort()
        k = int(math.floor(vals.size * trim_frac))
        if 2 * k < vals.size:
            vals = vals[k : vals.size - k]
        out[col] = float(np.mean(vals))
    return out


def _weighted_trimmed_seed_path_mean(path_stack, weights, trim_frac=0.10):
    out = np.full(path_stack.shape[1], np.nan, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    trim_frac = float(np.clip(trim_frac, 0.0, 0.45))
    for col in range(path_stack.shape[1]):
        vals = path_stack[:, col]
        valid = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
        vals = vals[valid]
        w = weights[valid]
        if vals.size == 0:
            continue
        order = np.argsort(vals)
        vals = vals[order]
        w = w[order]
        if trim_frac > 0.0 and vals.size >= 5:
            cdf = np.cumsum(w) / np.sum(w)
            keep = (cdf >= trim_frac) & (cdf <= 1.0 - trim_frac)
            if keep.any():
                vals = vals[keep]
                w = w[keep]
        out[col] = float(np.sum(vals * w) / np.sum(w))
    return out


def _nanmedian_seed_path(path_stack):
    out = np.full(path_stack.shape[1], np.nan, dtype=np.float64)
    for col in range(path_stack.shape[1]):
        vals = path_stack[:, col]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[col] = float(np.median(vals))
    return out



def _prepare_ref_grid(tw_tvt, tw_gr, step=0.2):
    valid = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    tw_tvt = tw_tvt[valid]
    tw_gr = tw_gr[valid]
    order = np.argsort(tw_tvt)
    tw_tvt = tw_tvt[order]
    tw_gr = tw_gr[order]
    grid_min = float(tw_tvt[0])
    grid_max = float(tw_tvt[-1])
    grid_tvt = np.arange(grid_min, grid_max + float(step), float(step), dtype=np.float64)
    return np.interp(grid_tvt, tw_tvt, tw_gr).astype(np.float64), grid_min, float(step)


def _run_numba_pf_heatmap_bins(
    query_md,
    query_z,
    query_gr,
    query_gr_power,
    query_gr_finite,
    obs_gr,
    obs_power,
    obs_finite,
    obs_md_delta,
    obs_z,
    obs_count,
    obs_likelihood_mode,
    shape_gr,
    shape_md_delta,
    shape_z,
    shape_count,
    known_md,
    known_z,
    known_gr_for_sigma,
    known_tvt,
    tw_tvt,
    tw_gr,
    raw_tw_tvt,
    raw_tw_gr,
    tvt0,
    bin_idx,
    num_bins,
    typewell_window,
    typewell_len,
    n_particles,
    seed,
    cache_cfg=None,
):
    known_mask = np.isfinite(known_md) & np.isfinite(known_z) & np.isfinite(known_tvt)
    known_md = known_md[known_mask]
    known_z = known_z[known_mask]
    known_tvt = known_tvt[known_mask]
    known_gr_for_sigma = known_gr_for_sigma[known_mask]
    if len(query_md) == 0 or len(known_tvt) < 2 or len(tw_tvt) < 3:
        return {
            "pf_prob": np.zeros((num_bins, typewell_len), dtype=np.float32),
            "pf_tvt_pred": np.full(num_bins, np.nan, dtype=np.float32),
            "log_lik": -np.inf,
            "gr_sigma": math.nan,
            "local_log_lik": np.zeros(num_bins, dtype=np.float64),
            "local_obs_power": np.zeros(num_bins, dtype=np.float64),
            "ffbsi_ok": False,
        }

    if cache_cfg is None:
        cache_cfg = {}
    shape_power = 0.0
    shape_mode_code = 0
    shape_ref_mode_code = 0
    shape_min_points = 7
    shape_sigma_floor = 0.35
    ref_grid, grid_min, grid_step = _prepare_ref_grid(tw_tvt, tw_gr, step=float(cache_cfg.get("ref_grid_step", 0.2)))
    tw_at_k = np.interp(known_tvt, tw_tvt, tw_gr)
    sigma = float(np.clip(np.nanstd(known_gr_for_sigma - tw_at_k), 10.0, 60.0))
    raw_ref_weight = float(cache_cfg.get("raw_ref_likelihood_weight", 0.0))
    need_raw_ref_grid = raw_ref_weight > 0.0
    if need_raw_ref_grid and len(raw_tw_tvt) >= 3:
        raw_ref_grid, raw_grid_min, raw_grid_step = _prepare_ref_grid(
            raw_tw_tvt,
            raw_tw_gr,
            step=float(cache_cfg.get("ref_grid_step", 0.2)),
        )
        raw_ref_weight = float(np.clip(raw_ref_weight, 0.0, 1.0))
        if shape_ref_mode_code == 1 and raw_ref_weight <= 0.0:
            raw_ref_weight = 0.0
    else:
        raw_ref_grid = np.asarray([0.0], dtype=np.float64)
        raw_grid_min = 0.0
        raw_grid_step = 1.0
        raw_ref_weight = 0.0
    ambiguity_power = float(cache_cfg.get("gr_ambiguity_power", 0.0))
    if ambiguity_power > 0.0:
        amb_mode = str(cache_cfg.get("gr_ambiguity_ref_mode", "primary"))
        if amb_mode == "raw" and len(raw_tw_tvt) >= 3:
            amb_tvt = raw_tw_tvt
            amb_gr = raw_tw_gr
        else:
            amb_tvt = tw_tvt
            amb_gr = tw_gr
        query_gr_power = query_gr_power * _local_gr_ambiguity_power(
            query_gr=query_gr,
            query_gr_finite=query_gr_finite,
            tvt0=tvt0,
            tw_tvt=amb_tvt,
            tw_gr=amb_gr,
            gr_sigma=sigma,
            typewell_window=typewell_window,
            ambiguity_power=ambiguity_power,
            min_power=float(cache_cfg.get("gr_ambiguity_min_power", 0.35)),
            contrast=float(cache_cfg.get("gr_ambiguity_contrast", 1.0)),
        )
    information_power = float(cache_cfg.get("gr_information_power", 0.0))
    if information_power > 0.0:
        info_mode = str(cache_cfg.get("gr_information_ref_mode", "primary"))
        if info_mode == "raw" and len(raw_tw_tvt) >= 3:
            info_tvt = raw_tw_tvt
            info_gr = raw_tw_gr
        else:
            info_tvt = tw_tvt
            info_gr = tw_gr
        query_gr_power = query_gr_power * _local_gr_information_multiplier(
            query_gr=query_gr,
            query_gr_finite=query_gr_finite,
            tvt0=tvt0,
            tw_tvt=info_tvt,
            tw_gr=info_gr,
            gr_sigma=sigma,
            typewell_window=typewell_window,
            information_power=information_power,
            center=float(cache_cfg.get("gr_information_center", 0.75)),
            min_multiplier=float(cache_cfg.get("gr_information_min_multiplier", 0.75)),
            max_multiplier=float(cache_cfg.get("gr_information_max_multiplier", 1.25)),
            slope_weight=float(cache_cfg.get("gr_information_slope_weight", 0.0)),
        )
    tail_start = max(0, len(known_tvt) - 30)
    dt = np.diff(known_tvt[tail_start:])
    dz = np.diff(known_z[tail_start:])
    dm = np.diff(known_md[tail_start:])
    move = dm > 0
    state_z_weight = float(cache_cfg.get("pf_state_z_weight", 1.0))
    init_rate = float(np.median((dt + state_z_weight * dz)[move] / dm[move])) if move.sum() >= 3 else 0.0
    init_pos = float(known_tvt[-1] + state_z_weight * known_z[-1])
    initial_prev_md = float(known_md[-1])
    tvt_min = float(tw_tvt[0] - 100.0)
    tvt_max = float(tw_tvt[-1] + 100.0)
    ffbsi_mode = str(cache_cfg.get("pf_ffbsi_mode", "off")).lower()
    ffbsi_n_paths = int(cache_cfg.get("pf_ffbsi_n_paths", 0))
    ffbsi_fallback_mode = str(cache_cfg.get("pf_ffbsi_fallback_mode", "filtered")).lower()
    ffbsi_transition_model_code = _pf_ffbsi_transition_model_code(
        cache_cfg.get("pf_ffbsi_transition_model", "gaussian")
    )
    if ffbsi_mode in {"on", "ffbsi", "fallback"} and ffbsi_n_paths > 0:
        try:
            (
                filtered_hist_bins,
                filtered_path_sums,
                filtered_count_bins,
                ffbsi_hist_bins,
                ffbsi_path_sums,
                ffbsi_count_bins,
                log_lik,
                local_log_lik,
                local_obs_power,
                ffbsi_ok,
            ) = _pf_heatmap_ffbsi_kernel(
                query_md.astype(np.float64),
                query_z.astype(np.float64),
                query_gr.astype(np.float64),
                query_gr_power.astype(np.float64),
                query_gr_finite.astype(np.float64),
                obs_gr.astype(np.float64),
                obs_power.astype(np.float64),
                obs_finite.astype(np.float64),
                obs_md_delta.astype(np.float64),
                obs_z.astype(np.float64),
                obs_count.astype(np.int64),
                int(obs_likelihood_mode),
                shape_gr.astype(np.float64),
                shape_md_delta.astype(np.float64),
                shape_z.astype(np.float64),
                shape_count.astype(np.int64),
                ref_grid,
                grid_min,
                grid_step,
                raw_ref_grid,
                raw_grid_min,
                raw_grid_step,
                raw_ref_weight,
                sigma,
                float(cache_cfg.get("outlier_prob", 0.0)),
                float(cache_cfg.get("outlier_likelihood", 0.05)),
                float(cache_cfg.get("dynamic_sigma_alpha", 0.0)),
                float(cache_cfg.get("dynamic_sigma_threshold", 1.25)),
                float(cache_cfg.get("dynamic_sigma_power", 1.0)),
                float(cache_cfg.get("dynamic_sigma_min", 0.85)),
                float(cache_cfg.get("dynamic_sigma_max", 2.0)),
                float(shape_power),
                int(shape_mode_code),
                int(shape_min_points),
                float(shape_sigma_floor),
                int(shape_ref_mode_code),
                init_pos,
                init_rate,
                initial_prev_md,
                float(tvt0),
                bin_idx.astype(np.int64),
                int(num_bins),
                float(typewell_window),
                int(typewell_len),
                int(n_particles),
                tvt_min,
                tvt_max,
                int(seed),
                float(cache_cfg.get("pf_momentum", 0.998)),
                float(cache_cfg.get("pf_rate_noise", 0.002)),
                float(cache_cfg.get("pf_pos_noise", 0.005)),
                float(cache_cfg.get("pf_resample_threshold", 0.5)),
                float(cache_cfg.get("pf_resample_obs_power_adapt", 0.0)),
                float(cache_cfg.get("pf_resample_min_threshold", 0.0)),
                float(cache_cfg.get("pf_rough_pos", 0.1)),
                float(cache_cfg.get("pf_rough_rate", 0.001)),
                float(cache_cfg.get("pf_rescue_frac", 0.0)),
                float(cache_cfg.get("pf_rescue_pos_sd", 0.0)),
                float(cache_cfg.get("pf_rescue_rate_sd", 0.0)),
                float(cache_cfg.get("pf_init_pos_sd", 3.0)),
                float(cache_cfg.get("pf_init_rate_sd", 0.01)),
                float(cache_cfg.get("pf_rate_mean_weight", 0.0)),
                float(cache_cfg.get("pf_jump_prob", 0.0)),
                float(cache_cfg.get("pf_jump_sd", 0.0)),
                float(cache_cfg.get("pf_jump_rate_sd", 0.0)),
                float(cache_cfg.get("pf_missing_jump_boost", 0.0)),
                float(cache_cfg.get("pf_jump_tail_prob", 0.0)),
                float(cache_cfg.get("pf_jump_tail_sd", 0.0)),
                float(cache_cfg.get("pf_jump_tail_rate_sd", 0.0)),
                int(cache_cfg.get("pf_jump_tail_dist", 1)),
                float(cache_cfg.get("pf_jump_tail_clip", 0.0)),
                float(cache_cfg.get("pf_jump_tail_missing_boost", 0.0)),
                float(cache_cfg.get("pf_anchor_sigma", 0.0)),
                float(cache_cfg.get("pf_anchor_power", 1.0)),
                float(cache_cfg.get("pf_correlated_rate_alpha", 0.0)),
                float(cache_cfg.get("pf_correlated_pos_alpha", 0.0)),
                float(cache_cfg.get("pf_lookahead_power", 0.0)),
                int(cache_cfg.get("pf_lookahead_steps", 1)),
                float(cache_cfg.get("pf_lookahead_decay", 0.5)),
                float(cache_cfg.get("pf_lookahead_max_gap", 256.0)),
                float(cache_cfg.get("pf_lookahead_delta_power", 0.0)),
                int(cache_cfg.get("pf_stratified_init", 0)),
                float(cache_cfg.get("pf_finite_run_power_boost", 0.0)),
                float(cache_cfg.get("pf_finite_run_power_cap", 1.0)),
                float(cache_cfg.get("pf_finite_run_power_decay", 0.0)),
                float(cache_cfg.get("pf_finite_run_power_floor", 0.35)),
                float(cache_cfg.get("pf_conf_obs_power_decay", 0.0)),
                float(cache_cfg.get("pf_conf_obs_power_floor", 0.55)),
                float(cache_cfg.get("pf_conf_obs_power_power", 1.0)),
                float(cache_cfg.get("pf_missing_noise_scale", 1.0)),
                float(cache_cfg.get("pf_missing_jump_scale", 1.0)),
                float(cache_cfg.get("pf_ess_jump_boost", 0.0)),
                float(cache_cfg.get("pf_ess_jump_power", 1.0)),
                float(cache_cfg.get("pf_surprise_jump_boost", 0.0)),
                float(cache_cfg.get("pf_surprise_jump_threshold", 1.5)),
                float(cache_cfg.get("pf_surprise_jump_power", 1.0)),
                float(cache_cfg.get("pf_ess_rough_boost", 0.0)),
                float(cache_cfg.get("pf_ess_rough_power", 1.0)),
                state_z_weight,
                ffbsi_n_paths,
                int(cache_cfg.get("pf_ffbsi_max_active_bins", 512)),
                int(ffbsi_transition_model_code),
                float(cache_cfg.get("pf_ffbsi_transition_scale", 1.5)),
                float(cache_cfg.get("pf_ffbsi_pos_floor", 0.35)),
                float(cache_cfg.get("pf_ffbsi_rate_floor", 0.0025)),
            )
            if ffbsi_ok > 0:
                pf_prob, pf_tvt_pred = _pf_arrays_from_hist(
                    filtered_hist_bins,
                    filtered_path_sums,
                    filtered_count_bins,
                    num_bins,
                    typewell_len,
                )
                pf_prob_ffbsi, pf_tvt_pred_ffbsi = _pf_arrays_from_hist(
                    ffbsi_hist_bins,
                    ffbsi_path_sums,
                    ffbsi_count_bins,
                    num_bins,
                    typewell_len,
                )
                return {
                    "pf_prob": pf_prob,
                    "pf_tvt_pred": pf_tvt_pred,
                    "pf_prob_ffbsi": pf_prob_ffbsi,
                    "pf_tvt_pred_ffbsi": pf_tvt_pred_ffbsi,
                    "log_lik": float(log_lik),
                    "gr_sigma": float(sigma),
                    "local_log_lik": np.asarray(local_log_lik, dtype=np.float64),
                    "local_obs_power": np.asarray(local_obs_power, dtype=np.float64),
                    "ffbsi_ok": True,
                }
            if ffbsi_fallback_mode not in {"filtered", "on", "true", "1"}:
                raise RuntimeError("FFBSi heatmap kernel could not run for this well")
        except Exception:
            if ffbsi_fallback_mode not in {"filtered", "on", "true", "1"}:
                raise
    hist_bins, path_sums, count_bins, log_lik, local_log_lik, local_obs_power = _pf_heatmap_kernel(
        query_md.astype(np.float64),
        query_z.astype(np.float64),
        query_gr.astype(np.float64),
        query_gr_power.astype(np.float64),
        query_gr_finite.astype(np.float64),
        obs_gr.astype(np.float64),
        obs_power.astype(np.float64),
        obs_finite.astype(np.float64),
        obs_md_delta.astype(np.float64),
        obs_z.astype(np.float64),
        obs_count.astype(np.int64),
        int(obs_likelihood_mode),
        shape_gr.astype(np.float64),
        shape_md_delta.astype(np.float64),
        shape_z.astype(np.float64),
        shape_count.astype(np.int64),
        ref_grid,
        grid_min,
        grid_step,
        raw_ref_grid,
        raw_grid_min,
        raw_grid_step,
        raw_ref_weight,
        sigma,
        float(cache_cfg.get("outlier_prob", 0.0)),
        float(cache_cfg.get("outlier_likelihood", 0.05)),
        float(cache_cfg.get("dynamic_sigma_alpha", 0.0)),
        float(cache_cfg.get("dynamic_sigma_threshold", 1.25)),
        float(cache_cfg.get("dynamic_sigma_power", 1.0)),
        float(cache_cfg.get("dynamic_sigma_min", 0.85)),
        float(cache_cfg.get("dynamic_sigma_max", 2.0)),
        float(shape_power),
        int(shape_mode_code),
        int(shape_min_points),
        float(shape_sigma_floor),
        int(shape_ref_mode_code),
        init_pos,
        init_rate,
        initial_prev_md,
        float(tvt0),
        bin_idx.astype(np.int64),
        int(num_bins),
        float(typewell_window),
        int(typewell_len),
        int(n_particles),
        tvt_min,
        tvt_max,
        int(seed),
        float(cache_cfg.get("pf_momentum", 0.998)),
        float(cache_cfg.get("pf_rate_noise", 0.002)),
        float(cache_cfg.get("pf_pos_noise", 0.005)),
        float(cache_cfg.get("pf_resample_threshold", 0.5)),
        float(cache_cfg.get("pf_resample_obs_power_adapt", 0.0)),
        float(cache_cfg.get("pf_resample_min_threshold", 0.0)),
        float(cache_cfg.get("pf_rough_pos", 0.1)),
        float(cache_cfg.get("pf_rough_rate", 0.001)),
        float(cache_cfg.get("pf_rescue_frac", 0.0)),
        float(cache_cfg.get("pf_rescue_pos_sd", 0.0)),
        float(cache_cfg.get("pf_rescue_rate_sd", 0.0)),
        float(cache_cfg.get("pf_init_pos_sd", 3.0)),
        float(cache_cfg.get("pf_init_rate_sd", 0.01)),
        float(cache_cfg.get("pf_rate_mean_weight", 0.0)),
        float(cache_cfg.get("pf_jump_prob", 0.0)),
        float(cache_cfg.get("pf_jump_sd", 0.0)),
        float(cache_cfg.get("pf_jump_rate_sd", 0.0)),
        float(cache_cfg.get("pf_missing_jump_boost", 0.0)),
        float(cache_cfg.get("pf_jump_tail_prob", 0.0)),
        float(cache_cfg.get("pf_jump_tail_sd", 0.0)),
        float(cache_cfg.get("pf_jump_tail_rate_sd", 0.0)),
        int(cache_cfg.get("pf_jump_tail_dist", 1)),
        float(cache_cfg.get("pf_jump_tail_clip", 0.0)),
        float(cache_cfg.get("pf_jump_tail_missing_boost", 0.0)),
        float(cache_cfg.get("pf_anchor_sigma", 0.0)),
        float(cache_cfg.get("pf_anchor_power", 1.0)),
        float(cache_cfg.get("pf_correlated_rate_alpha", 0.0)),
        float(cache_cfg.get("pf_correlated_pos_alpha", 0.0)),
        float(cache_cfg.get("pf_lookahead_power", 0.0)),
        int(cache_cfg.get("pf_lookahead_steps", 1)),
        float(cache_cfg.get("pf_lookahead_decay", 0.5)),
        float(cache_cfg.get("pf_lookahead_max_gap", 256.0)),
        float(cache_cfg.get("pf_lookahead_delta_power", 0.0)),
        int(cache_cfg.get("pf_stratified_init", 0)),
        float(cache_cfg.get("pf_finite_run_power_boost", 0.0)),
        float(cache_cfg.get("pf_finite_run_power_cap", 1.0)),
        float(cache_cfg.get("pf_finite_run_power_decay", 0.0)),
        float(cache_cfg.get("pf_finite_run_power_floor", 0.35)),
        float(cache_cfg.get("pf_conf_obs_power_decay", 0.0)),
        float(cache_cfg.get("pf_conf_obs_power_floor", 0.55)),
        float(cache_cfg.get("pf_conf_obs_power_power", 1.0)),
        float(cache_cfg.get("pf_missing_noise_scale", 1.0)),
        float(cache_cfg.get("pf_missing_jump_scale", 1.0)),
        float(cache_cfg.get("pf_ess_jump_boost", 0.0)),
        float(cache_cfg.get("pf_ess_jump_power", 1.0)),
        float(cache_cfg.get("pf_surprise_jump_boost", 0.0)),
        float(cache_cfg.get("pf_surprise_jump_threshold", 1.5)),
        float(cache_cfg.get("pf_surprise_jump_power", 1.0)),
        float(cache_cfg.get("pf_ess_rough_boost", 0.0)),
        float(cache_cfg.get("pf_ess_rough_power", 1.0)),
        state_z_weight,
    )
    pf_prob = np.zeros((num_bins, typewell_len), dtype=np.float32)
    valid_bins = count_bins > 0
    pf_prob[valid_bins] = (hist_bins[valid_bins] / count_bins[valid_bins, None]).astype(np.float32)
    pf_prob = _normalize_prob_rows(pf_prob)
    pf_tvt_pred = np.full(num_bins, np.nan, dtype=np.float32)
    pf_tvt_pred[valid_bins] = (path_sums[valid_bins] / count_bins[valid_bins]).astype(np.float32)
    return {
        "pf_prob": pf_prob,
        "pf_tvt_pred": pf_tvt_pred,
        "log_lik": float(log_lik),
        "gr_sigma": float(sigma),
        "local_log_lik": np.asarray(local_log_lik, dtype=np.float64),
        "local_obs_power": np.asarray(local_obs_power, dtype=np.float64),
        "ffbsi_ok": False,
    }


def _merge_suffix_query_blocks(
    query_md,
    query_z,
    query_gr,
    query_gr_power,
    query_gr_finite,
    merge_k,
    power_alpha,
    likelihood_mode,
    state_mode,
):
    query_md = np.asarray(query_md, dtype=np.float64)
    query_z = np.asarray(query_z, dtype=np.float64)
    query_gr = np.asarray(query_gr, dtype=np.float64)
    query_gr_power = np.asarray(query_gr_power, dtype=np.float64)
    query_gr_finite = np.asarray(query_gr_finite, dtype=np.float64)
    merge_k = max(1, int(merge_k))
    power_alpha = float(power_alpha)
    likelihood_mode = str(likelihood_mode).lower()
    state_mode = str(state_mode).lower()

    n_rows = int(query_md.size)
    if merge_k <= 1 or n_rows <= 1:
        starts = np.arange(n_rows, dtype=np.int64)
        ends = starts + 1
        return (
            query_md.copy(),
            query_z.copy(),
            query_gr.copy(),
            query_gr_power.copy(),
            query_gr_finite.copy(),
            query_gr[:, None].copy(),
            query_gr_power[:, None].copy(),
            query_gr_finite[:, None].copy(),
            np.zeros((n_rows, 1), dtype=np.float64),
            query_z[:, None].copy(),
            np.ones(n_rows, dtype=np.int64),
            starts,
            ends,
        )

    starts = np.arange(0, n_rows, merge_k, dtype=np.int64)
    ends = np.minimum(starts + merge_k, n_rows).astype(np.int64, copy=False)
    n_blocks = int(starts.size)
    merged_md = np.zeros(n_blocks, dtype=np.float64)
    merged_z = np.zeros(n_blocks, dtype=np.float64)
    merged_gr = np.zeros(n_blocks, dtype=np.float64)
    merged_power = np.zeros(n_blocks, dtype=np.float64)
    merged_finite = np.zeros(n_blocks, dtype=np.float64)
    obs_width = int(merge_k)
    obs_gr = np.zeros((n_blocks, obs_width), dtype=np.float64)
    obs_power = np.zeros((n_blocks, obs_width), dtype=np.float64)
    obs_finite = np.zeros((n_blocks, obs_width), dtype=np.float64)
    obs_md_delta = np.zeros((n_blocks, obs_width), dtype=np.float64)
    obs_z = np.zeros((n_blocks, obs_width), dtype=np.float64)
    obs_count = np.zeros(n_blocks, dtype=np.int64)

    for block_idx, (start, end) in enumerate(zip(starts, ends)):
        if state_mode == "end":
            state_row = end - 1
        else:
            state_row = start
        merged_md[block_idx] = query_md[state_row]
        merged_z[block_idx] = query_z[state_row]
        block_gr = query_gr[start:end]
        block_power = np.clip(query_gr_power[start:end], 0.0, None)
        block_finite = np.clip(query_gr_finite[start:end], 0.0, 1.0)
        block_len = int(end - start)
        obs_count[block_idx] = block_len
        obs_gr[block_idx, :block_len] = block_gr
        obs_power[block_idx, :block_len] = block_power
        obs_finite[block_idx, :block_len] = block_finite
        obs_md_delta[block_idx, :block_len] = query_md[start:end] - merged_md[block_idx]
        obs_z[block_idx, :block_len] = query_z[start:end]
        weight_sum = float(np.sum(block_power))
        if weight_sum > 0.0 and np.isfinite(weight_sum):
            merged_gr[block_idx] = float(np.sum(block_gr * block_power) / weight_sum)
        else:
            merged_gr[block_idx] = float(np.mean(block_gr)) if block_gr.size else 0.0
        if weight_sum > 0.0:
            merged_power[block_idx] = float(math.pow(max(weight_sum, 0.0), power_alpha))
            if likelihood_mode == "block":
                obs_power[block_idx, :block_len] *= merged_power[block_idx] / weight_sum
        else:
            merged_power[block_idx] = 0.0
        merged_finite[block_idx] = float(np.mean(block_finite)) if block_finite.size else 0.0

    return (
        merged_md,
        merged_z,
        merged_gr,
        merged_power,
        merged_finite,
        obs_gr,
        obs_power,
        obs_finite,
        obs_md_delta,
        obs_z,
        obs_count,
        starts,
        ends,
    )


def _row_path_from_merged_steps(step_tvt_pred, tvt0, block_starts, block_ends, total_len):
    total_len = int(total_len)
    out = np.full(total_len, np.nan, dtype=np.float32)
    if total_len <= 0:
        return out
    prev_tvt = float(tvt0)
    for step_tvt, start, end in zip(step_tvt_pred, block_starts, block_ends):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        cur_tvt = float(step_tvt) if np.isfinite(step_tvt) else prev_tvt
        span = end - start
        delta = (cur_tvt - prev_tvt) / float(span)
        for offset in range(span):
            out[start + offset] = np.float32(prev_tvt + delta * float(offset + 1))
        prev_tvt = cur_tvt
    last_filled = np.flatnonzero(np.isfinite(out))
    if last_filled.size > 0 and int(last_filled[-1]) + 1 < total_len:
        out[int(last_filled[-1]) + 1 :] = out[int(last_filled[-1])]
    return out


def _expand_merged_prob_to_bins(step_prob, block_starts, block_ends, prefix_len, downsample, num_bins):
    step_prob = np.asarray(step_prob, dtype=np.float32)
    out = np.zeros((int(num_bins), int(step_prob.shape[1])), dtype=np.float32)
    counts = np.zeros(int(num_bins), dtype=np.float32)
    for prob_row, start, end in zip(step_prob, block_starts, block_ends):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        row_bins = (int(prefix_len) + np.arange(start, end, dtype=np.int64)) // int(downsample)
        row_bins = row_bins[(row_bins >= 0) & (row_bins < int(num_bins))]
        if row_bins.size == 0:
            continue
        uniq, uniq_counts = np.unique(row_bins, return_counts=True)
        for bin_idx, bin_count in zip(uniq, uniq_counts):
            out[int(bin_idx)] += np.float32(bin_count) * prob_row
            counts[int(bin_idx)] += np.float32(bin_count)
    valid = counts > 0.0
    out[valid] /= counts[valid, None]
    return _normalize_prob_rows(out)


def _effective_pf_cache_cfg_for_merge(cache_cfg):
    merge_k = max(1, int(cache_cfg.get("suffix_merge_k", 1)))
    if merge_k <= 1:
        return dict(cache_cfg)
    out = dict(cache_cfg)
    if not bool(cache_cfg.get("suffix_merge_adjust_dynamics", False)):
        return out

    row_momentum = float(np.clip(out.get("pf_momentum", 0.998), 0.0, 0.999999999))
    out["pf_momentum"] = float(np.clip(math.pow(row_momentum, merge_k), 0.0, 0.999999999))

    innovation_scale = math.sqrt(
        float(sum(math.pow(row_momentum, 2 * step_idx) for step_idx in range(merge_k)))
    )
    if innovation_scale > 0.0:
        out["pf_rate_noise"] = float(out.get("pf_rate_noise", 0.0)) * innovation_scale
    out["pf_pos_noise"] = float(out.get("pf_pos_noise", 0.0)) * math.sqrt(float(merge_k))

    for key in ("pf_correlated_rate_alpha", "pf_correlated_pos_alpha"):
        alpha = float(np.clip(out.get(key, 0.0), 0.0, 0.999999999))
        if alpha > 0.0:
            out[key] = float(math.pow(alpha, merge_k))

    rate_mean_weight = float(np.clip(out.get("pf_rate_mean_weight", 0.0), 0.0, 1.0))
    if rate_mean_weight > 0.0:
        out["pf_rate_mean_weight"] = float(1.0 - math.pow(1.0 - rate_mean_weight, merge_k))

    return out


def _warmup_pf_heatmap_numba():
    if njit is None:
        return
    _run_numba_pf_heatmap_bins(
        query_md=np.asarray([1.0, 2.0], dtype=np.float64),
        query_z=np.asarray([-9000.0, -9000.1], dtype=np.float64),
        query_gr=np.asarray([80.0, 81.0], dtype=np.float64),
        query_gr_power=np.asarray([1.0, 1.0], dtype=np.float64),
        query_gr_finite=np.asarray([1.0, 1.0], dtype=np.float64),
        obs_gr=np.asarray([[80.0], [81.0]], dtype=np.float64),
        obs_power=np.asarray([[1.0], [1.0]], dtype=np.float64),
        obs_finite=np.asarray([[1.0], [1.0]], dtype=np.float64),
        obs_md_delta=np.zeros((2, 1), dtype=np.float64),
        obs_z=np.asarray([[-9000.0], [-9000.1]], dtype=np.float64),
        obs_count=np.ones(2, dtype=np.int64),
        obs_likelihood_mode=0,
        shape_gr=np.zeros((2, 1), dtype=np.float64),
        shape_md_delta=np.zeros((2, 1), dtype=np.float64),
        shape_z=np.zeros((2, 1), dtype=np.float64),
        shape_count=np.zeros(2, dtype=np.int64),
        known_md=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        known_z=np.asarray([-9000.0, -9000.05, -9000.1], dtype=np.float64),
        known_gr_for_sigma=np.asarray([80.0, 81.0, 82.0], dtype=np.float64),
        known_tvt=np.asarray([11000.0, 11000.0, 11000.1], dtype=np.float64),
        tw_tvt=np.asarray([10999.0, 11000.0, 11001.0], dtype=np.float64),
        tw_gr=np.asarray([79.0, 80.0, 82.0], dtype=np.float64),
        raw_tw_tvt=np.asarray([10999.0, 11000.0, 11001.0], dtype=np.float64),
        raw_tw_gr=np.asarray([79.0, 80.0, 82.0], dtype=np.float64),
        tvt0=11000.1,
        bin_idx=np.asarray([0, 0], dtype=np.int64),
        num_bins=1,
        typewell_window=10.0,
        typewell_len=20,
        n_particles=4,
        seed=0,
        cache_cfg={},
    )


def _benchmark_numba_vs_python(data_path, well_ids, cfg, log):
    bench_n = min(128, int(getattr(cfg, "PF_heatmap_n_particles", 500)))
    if bench_n <= 0:
        return
    rows = []
    for well_id in well_ids:
        h_df, tw_df = _read_well_frames(data_path, well_id)
        if "TVT" not in h_df.columns:
            continue
        bench_seed = _single_pf_seed(int(getattr(cfg, "PF_heatmap_base_seed", 202605)))
        t0 = time.perf_counter()
        slow_pred, _ = run_particle_filter(h_df, tw_df, n_particles=bench_n, seed=bench_seed)
        slow_sec = time.perf_counter() - t0
        cache_cfg = pf_heatmap_cache_config(cfg)
        cache_cfg["n_particles"] = bench_n
        cache_cfg["n_seeds"] = 1
        t1 = time.perf_counter()
        fast = _build_pf_heatmap_item(h_df, tw_df, cache_cfg, has_target=True)
        fast_sec = time.perf_counter() - t1
        last_seen_idx = int(np.flatnonzero(np.isfinite(h_df["TVT_input"].to_numpy(dtype=np.float64)))[-1])
        suffix_start = last_seen_idx + 1
        kept_len = min(len(h_df) - suffix_start, int(cfg.target_len))
        slow_bin = _downsample_suffix_values_for_benchmark(
            slow_pred[suffix_start : suffix_start + kept_len],
            cfg,
        )
        target_bin = fast["window_tvt"]
        mask = fast["target_mask"] & np.isfinite(target_bin) & np.isfinite(slow_bin) & (fast["pf_tvt_pred"] != 0.0)
        if mask.any():
            slow_target = float(np.sqrt(np.mean(np.square(slow_bin[mask] - target_bin[mask]))))
            fast_target = float(np.sqrt(np.mean(np.square(fast["pf_tvt_pred"][mask] - target_bin[mask]))))
            path_gap = float(np.sqrt(np.mean(np.square(slow_bin[mask] - fast["pf_tvt_pred"][mask]))))
        else:
            slow_target = fast_target = path_gap = math.nan
        rows.append((slow_sec, fast_sec, slow_target, fast_target, path_gap))
    if not rows:
        return
    arr = np.asarray(rows, dtype=float)
    _log(
        log,
        "PF numba smoke: "
        f"slow_sec/well={np.nanmean(arr[:, 0]):.3f}, "
        f"numba_sec/well={np.nanmean(arr[:, 1]):.3f}, "
        f"speedup={np.nanmean(arr[:, 0]) / max(np.nanmean(arr[:, 1]), 1e-9):.2f}x, "
        f"slow_TVT_RMSE={np.nanmean(arr[:, 2]):.4f}, "
        f"numba_TVT_RMSE={np.nanmean(arr[:, 3]):.4f}, "
        f"path_gap_RMSE={np.nanmean(arr[:, 4]):.4f}",
    )


def _downsample_suffix_values_for_benchmark(values, cfg):
    return _downsample_suffix_values(values, cfg.prefix_len, cfg.downsample, cfg.num_bins)


def _pf_ffbsi_transition_model_code(model):
    model = str(model or "gaussian").lower()
    if model in {"gaussian", "collapsed", "variance", "var"}:
        return 0
    if model in {"mixture", "jump_mixture", "mixture_independent", "jump_mixture_independent"}:
        return 1
    if model in {"mixture_norm", "jump_mixture_norm", "mixture_independent_norm", "jump_mixture_independent_norm"}:
        return 2
    if model in {"mixture_joint", "jump_mixture_joint", "mixture_cov", "jump_mixture_cov"}:
        return 3
    if model in {"mixture_joint_norm", "jump_mixture_joint_norm", "mixture_cov_norm", "jump_mixture_cov_norm"}:
        return 4
    if model in {"gaussian_joint", "collapsed_joint", "joint", "cov"}:
        return 5
    raise ValueError(f"unknown PF FFBSi transition model={model!r}")


if njit is not None:

    @njit(cache=False)
    def _interp_grid_numba(grid, tvt, grid_min, grid_step):
        idx = int((tvt - grid_min) / grid_step)
        if idx < 0:
            return grid[0]
        last = len(grid) - 1
        if idx >= last:
            return grid[last]
        frac = (tvt - grid_min) / grid_step - idx
        return grid[idx] * (1.0 - frac) + grid[idx + 1] * frac

    @njit(cache=False)
    def _pf_gr_shape_likelihood_numba(
        row,
        tvt,
        rate,
        z_row,
        state_z_weight,
        gr_center,
        sigma_eff,
        ref_grid,
        grid_min,
        grid_step,
        raw_ref_grid,
        raw_grid_min,
        raw_grid_step,
        raw_ref_weight,
        shape_gr,
        shape_md_delta,
        shape_z,
        shape_count,
        shape_power,
        shape_mode,
        shape_min_points,
        shape_sigma_floor,
        shape_ref_mode,
    ):
        if shape_power <= 0.0 or shape_mode <= 0:
            return 1.0
        n = shape_count[row]
        if n < shape_min_points or n <= 1:
            return 1.0
        shape_width = shape_gr.shape[1]
        if n > shape_width:
            n = shape_width
        if n < shape_min_points or n <= 1:
            return 1.0
        if sigma_eff <= 1e-6:
            sigma_eff = 1e-6
        sigma_floor = shape_sigma_floor * sigma_eff
        if sigma_floor <= 1e-6:
            sigma_floor = 1e-6

        center_pred = _interp_grid_numba(ref_grid, tvt, grid_min, grid_step)
        if shape_ref_mode == 1:
            center_pred = _interp_grid_numba(raw_ref_grid, tvt, raw_grid_min, raw_grid_step)
        elif shape_ref_mode == 2 and raw_ref_weight > 0.0:
            raw_center_pred = _interp_grid_numba(raw_ref_grid, tvt, raw_grid_min, raw_grid_step)
            center_pred = (1.0 - raw_ref_weight) * center_pred + raw_ref_weight * raw_center_pred

        obs_mean = 0.0
        pred_mean = 0.0
        for k in range(n):
            pred_tvt = (tvt + rate * shape_md_delta[row, k]) + state_z_weight * (z_row - shape_z[row, k])
            pred = _interp_grid_numba(ref_grid, pred_tvt, grid_min, grid_step)
            if shape_ref_mode == 1:
                pred = _interp_grid_numba(raw_ref_grid, pred_tvt, raw_grid_min, raw_grid_step)
            elif shape_ref_mode == 2 and raw_ref_weight > 0.0:
                raw_pred = _interp_grid_numba(raw_ref_grid, pred_tvt, raw_grid_min, raw_grid_step)
                pred = (1.0 - raw_ref_weight) * pred + raw_ref_weight * raw_pred
            if shape_mode == 1:
                obs_mean += 0.0
                pred_mean += 0.0
            else:
                obs_mean += shape_gr[row, k]
                pred_mean += pred
        if shape_mode != 1:
            obs_mean /= n
            pred_mean /= n

        obs_var = 0.0
        pred_var = 0.0
        cross = 0.0
        for k in range(n):
            pred_tvt = (tvt + rate * shape_md_delta[row, k]) + state_z_weight * (z_row - shape_z[row, k])
            pred = _interp_grid_numba(ref_grid, pred_tvt, grid_min, grid_step)
            if shape_ref_mode == 1:
                pred = _interp_grid_numba(raw_ref_grid, pred_tvt, raw_grid_min, raw_grid_step)
            elif shape_ref_mode == 2 and raw_ref_weight > 0.0:
                raw_pred = _interp_grid_numba(raw_ref_grid, pred_tvt, raw_grid_min, raw_grid_step)
                pred = (1.0 - raw_ref_weight) * pred + raw_ref_weight * raw_pred
            if shape_mode == 1:
                obs = shape_gr[row, k] - gr_center
                pred = pred - center_pred
            else:
                obs = shape_gr[row, k]
            obs_d = obs - obs_mean
            pred_d = pred - pred_mean
            obs_var += obs_d * obs_d
            pred_var += pred_d * pred_d
            cross += obs_d * pred_d
        if obs_var <= sigma_floor * sigma_floor or pred_var <= sigma_floor * sigma_floor:
            return 1.0
        corr = cross / math.sqrt(obs_var * pred_var)
        if corr > 1.0:
            corr = 1.0
        elif corr < -1.0:
            corr = -1.0
        bonus = max(corr, 0.0)
        if shape_mode == 2:
            bonus = 0.5 * (1.0 + corr)
        lk = 1.0 + shape_power * bonus
        if lk < 1e-300:
            lk = 1e-300
        return lk

    @njit(cache=False)
    def _pf_gr_point_likelihood_numba(
        obs_gr,
        tvt,
        obs_power,
        obs_finite,
        sigma_eff,
        ref_grid,
        grid_min,
        grid_step,
        raw_ref_grid,
        raw_grid_min,
        raw_grid_step,
        raw_ref_weight,
        outlier_prob,
        outlier_likelihood,
    ):
        if obs_power <= 0.0:
            return 1.0
        expected_gr = _interp_grid_numba(ref_grid, tvt, grid_min, grid_step)
        d = (obs_gr - expected_gr) / sigma_eff
        d2 = d * d
        if d2 > 600.0:
            d2 = 600.0
        primary_lk = math.exp(-0.5 * d2 * obs_power)
        if raw_ref_weight > 0.0:
            raw_expected_gr = _interp_grid_numba(raw_ref_grid, tvt, raw_grid_min, raw_grid_step)
            raw_d = (obs_gr - raw_expected_gr) / sigma_eff
            raw_d2 = raw_d * raw_d
            if raw_d2 > 600.0:
                raw_d2 = 600.0
            raw_lk = math.exp(-0.5 * raw_d2 * obs_power)
            primary_lk = (1.0 - raw_ref_weight) * primary_lk + raw_ref_weight * raw_lk
        if outlier_prob > 0.0 and obs_finite > 0.0:
            primary_lk = (1.0 - outlier_prob) * primary_lk + outlier_prob * outlier_likelihood
        if primary_lk < 1e-300:
            primary_lk = 1e-300
        return primary_lk

    @njit(cache=False)
    def _pf_resample_numba(pos, rate, weight, rough_pos, rough_rate):
        n = len(pos)
        cum = np.empty(n)
        total = 0.0
        for j in range(n):
            total += weight[j]
            cum[j] = total
        start = np.random.uniform(0.0, 1.0 / n)
        new_pos = np.empty(n)
        new_rate = np.empty(n)
        cursor = 0
        for j in range(n):
            u = start + j / n
            while cursor < n - 1 and cum[cursor] < u:
                cursor += 1
            new_pos[j] = pos[cursor] + rough_pos * np.random.randn()
            new_rate[j] = rate[cursor] + rough_rate * np.random.randn()
        return new_pos, new_rate

    @njit(cache=False)
    def _pf_jump_noise_numba(sd, dist, clip_abs):
        if sd <= 0.0:
            return 0.0
        if dist == 1:
            # Laplace noise parameterized by standard deviation.
            u = np.random.random() - 0.5
            scale = sd * 0.7071067811865475
            if u >= 0.0:
                x = -scale * math.log(max(1e-12, 1.0 - 2.0 * u))
            else:
                x = scale * math.log(max(1e-12, 1.0 + 2.0 * u))
        elif dist == 2:
            # Clipped Cauchy-like tail. Use only with a finite clip.
            x = sd * 0.25 * math.tan(math.pi * (np.random.random() - 0.5))
        else:
            x = sd * np.random.randn()
        if clip_abs > 0.0:
            if x > clip_abs:
                x = clip_abs
            elif x < -clip_abs:
                x = -clip_abs
        return x

    @njit(cache=False)
    def _pf_sample_weight_index_numba(weight, n):
        total = 0.0
        for j in range(n):
            if weight[j] > 0.0 and np.isfinite(weight[j]):
                total += weight[j]
        if total <= 0.0:
            return int(np.random.randint(0, n))
        u = np.random.random() * total
        c = 0.0
        for j in range(n):
            if weight[j] > 0.0 and np.isfinite(weight[j]):
                c += weight[j]
                if c >= u:
                    return j
        return n - 1

    @njit(cache=False)
    def _pf_sample_log_score_index_numba(log_score, n):
        best = 0
        best_score = -1e300
        for j in range(n):
            if log_score[j] > best_score:
                best_score = log_score[j]
                best = j
        if best_score <= -1e290:
            return best
        total = 0.0
        for j in range(n):
            total += math.exp(log_score[j] - best_score)
        if total <= 0.0 or not np.isfinite(total):
            return best
        u = np.random.random() * total
        c = 0.0
        for j in range(n):
            c += math.exp(log_score[j] - best_score)
            if c >= u:
                return j
        return best

    @njit(cache=False)
    def _pf_transition_local_params_numba(
        dmd,
        gr_power,
        gr_finite,
        rate_noise,
        pos_noise,
        jump_prob,
        jump_sd,
        jump_rate_sd,
        missing_jump_boost,
        jump_tail_prob,
        jump_tail_sd,
        jump_tail_rate_sd,
        jump_tail_missing_boost,
        missing_noise_scale,
        missing_jump_scale,
    ):
        missing_frac = 1.0 - gr_finite
        if missing_frac < 0.0:
            missing_frac = 0.0
        elif missing_frac > 1.0:
            missing_frac = 1.0
        local_rate_noise = rate_noise
        local_pos_noise = pos_noise
        if missing_frac > 0.0 and missing_noise_scale != 1.0:
            noise_scale = 1.0 + (missing_noise_scale - 1.0) * missing_frac
            local_rate_noise *= noise_scale
            local_pos_noise *= noise_scale
        step_jump_prob = jump_prob * dmd
        if missing_frac > 0.0 and missing_jump_scale != 1.0:
            step_jump_prob *= 1.0 + (missing_jump_scale - 1.0) * missing_frac
        if gr_power < 1.0 and missing_jump_boost > 0.0:
            step_jump_prob += missing_jump_boost * (1.0 - gr_power) * dmd
        if step_jump_prob > 1.0:
            step_jump_prob = 1.0
        elif step_jump_prob < 0.0:
            step_jump_prob = 0.0

        step_tail_jump_prob = jump_tail_prob * dmd
        if missing_frac > 0.0 and missing_jump_scale != 1.0:
            step_tail_jump_prob *= 1.0 + (missing_jump_scale - 1.0) * missing_frac
        if gr_power < 1.0 and jump_tail_missing_boost > 0.0:
            step_tail_jump_prob += jump_tail_missing_boost * (1.0 - gr_power) * dmd
        if step_tail_jump_prob > 1.0:
            step_tail_jump_prob = 1.0
        elif step_tail_jump_prob < 0.0:
            step_tail_jump_prob = 0.0

        return local_rate_noise, local_pos_noise, step_jump_prob, step_tail_jump_prob

    @njit(cache=False)
    def _pf_transition_scale_numba(
        dmd,
        gr_power,
        gr_finite,
        rate_noise,
        pos_noise,
        jump_prob,
        jump_sd,
        jump_rate_sd,
        missing_jump_boost,
        jump_tail_prob,
        jump_tail_sd,
        jump_tail_rate_sd,
        jump_tail_missing_boost,
        missing_noise_scale,
        missing_jump_scale,
        transition_scale,
        pos_floor,
        rate_floor,
    ):
        local_rate_noise, local_pos_noise, step_jump_prob, step_tail_jump_prob = _pf_transition_local_params_numba(
            dmd,
            gr_power,
            gr_finite,
            rate_noise,
            pos_noise,
            jump_prob,
            jump_sd,
            jump_rate_sd,
            missing_jump_boost,
            jump_tail_prob,
            jump_tail_sd,
            jump_tail_rate_sd,
            jump_tail_missing_boost,
            missing_noise_scale,
            missing_jump_scale,
        )
        rate_var = local_rate_noise * local_rate_noise
        pos_var = local_pos_noise * local_pos_noise + dmd * dmd * local_rate_noise * local_rate_noise
        if step_jump_prob > 0.0:
            pos_var += step_jump_prob * jump_sd * jump_sd
            rate_var += step_jump_prob * jump_rate_sd * jump_rate_sd
        if step_tail_jump_prob > 0.0:
            pos_var += step_tail_jump_prob * jump_tail_sd * jump_tail_sd
            rate_var += step_tail_jump_prob * jump_tail_rate_sd * jump_tail_rate_sd
        pos_sd = math.sqrt(max(pos_var, 1e-18)) * transition_scale
        rate_sd = math.sqrt(max(rate_var, 1e-18)) * transition_scale
        if pos_sd < pos_floor:
            pos_sd = pos_floor
        if rate_sd < rate_floor:
            rate_sd = rate_floor
        return pos_sd, rate_sd

    @njit(cache=False)
    def _pf_transition_independent_log_lk_numba(
        pos_delta,
        rate_delta,
        raw_pos_var,
        raw_rate_var,
        transition_scale,
        pos_floor,
        rate_floor,
        include_norm,
    ):
        scale2 = transition_scale * transition_scale
        pos_sd = math.sqrt(max(raw_pos_var * scale2, 1e-18))
        rate_sd = math.sqrt(max(raw_rate_var * scale2, 1e-18))
        if pos_sd < pos_floor:
            pos_sd = pos_floor
        if rate_sd < rate_floor:
            rate_sd = rate_floor
        rate_d = rate_delta / rate_sd
        pos_d = pos_delta / pos_sd
        rate_log_lk = -0.5 * rate_d * rate_d
        if rate_log_lk < -300.0:
            rate_log_lk = -300.0
        pos_log_lk = -0.5 * pos_d * pos_d
        if pos_log_lk < -300.0:
            pos_log_lk = -300.0
        out = rate_log_lk + pos_log_lk
        if include_norm != 0:
            out -= math.log(max(pos_sd, 1e-300)) + math.log(max(rate_sd, 1e-300))
        return out

    @njit(cache=False)
    def _pf_transition_joint_log_lk_numba(
        pos_delta,
        rate_delta,
        raw_pos_var,
        raw_rate_var,
        raw_cov,
        transition_scale,
        pos_floor,
        rate_floor,
        include_norm,
    ):
        scale2 = transition_scale * transition_scale
        pos_var = max(raw_pos_var * scale2, 1e-18)
        rate_var = max(raw_rate_var * scale2, 1e-18)
        cov = raw_cov * scale2
        pos_floor2 = pos_floor * pos_floor
        rate_floor2 = rate_floor * rate_floor
        if pos_var < pos_floor2:
            pos_var = pos_floor2
        if rate_var < rate_floor2:
            rate_var = rate_floor2
        max_abs_cov = 0.999999 * math.sqrt(max(pos_var * rate_var, 1e-300))
        if cov > max_abs_cov:
            cov = max_abs_cov
        elif cov < -max_abs_cov:
            cov = -max_abs_cov
        det = pos_var * rate_var - cov * cov
        if det <= 1e-300 or not np.isfinite(det):
            return _pf_transition_independent_log_lk_numba(
                pos_delta,
                rate_delta,
                raw_pos_var,
                raw_rate_var,
                transition_scale,
                pos_floor,
                rate_floor,
                include_norm,
            )
        quad = (
            rate_var * pos_delta * pos_delta
            + pos_var * rate_delta * rate_delta
            - 2.0 * cov * pos_delta * rate_delta
        ) / det
        if quad > 1200.0:
            quad = 1200.0
        out = -0.5 * quad
        if include_norm != 0:
            out -= 0.5 * math.log(max(det, 1e-300))
        return out

    @njit(cache=False)
    def _pf_logaddexp_numba(a, b):
        if a <= -1e290:
            return b
        if b <= -1e290:
            return a
        if a >= b:
            return a + math.log1p(math.exp(b - a))
        return b + math.log1p(math.exp(a - b))

    @njit(cache=False)
    def _pf_transition_component_log_lk_numba(
        pos_delta,
        rate_delta,
        local_pos_noise,
        local_rate_noise,
        dmd,
        jump_pos_sd,
        jump_rate_sd,
        tail_pos_sd,
        tail_rate_sd,
        transition_scale,
        pos_floor,
        rate_floor,
        use_joint,
        include_norm,
    ):
        raw_rate_var = local_rate_noise * local_rate_noise + jump_rate_sd * jump_rate_sd + tail_rate_sd * tail_rate_sd
        raw_pos_var = (
            local_pos_noise * local_pos_noise
            + dmd * dmd * local_rate_noise * local_rate_noise
            + jump_pos_sd * jump_pos_sd
            + tail_pos_sd * tail_pos_sd
        )
        raw_cov = dmd * local_rate_noise * local_rate_noise
        if use_joint != 0:
            return _pf_transition_joint_log_lk_numba(
                pos_delta,
                rate_delta,
                raw_pos_var,
                raw_rate_var,
                raw_cov,
                transition_scale,
                pos_floor,
                rate_floor,
                include_norm,
            )
        return _pf_transition_independent_log_lk_numba(
            pos_delta,
            rate_delta,
            raw_pos_var,
            raw_rate_var,
            transition_scale,
            pos_floor,
            rate_floor,
            include_norm,
        )

    @njit(cache=False)
    def _pf_transition_mixture_log_lk_numba(
        pos_delta,
        rate_delta,
        local_pos_noise,
        local_rate_noise,
        dmd,
        step_jump_prob,
        jump_sd,
        jump_rate_sd,
        step_tail_jump_prob,
        jump_tail_sd,
        jump_tail_rate_sd,
        transition_scale,
        pos_floor,
        rate_floor,
        use_joint,
        include_norm,
    ):
        pj = step_jump_prob
        if pj < 0.0:
            pj = 0.0
        elif pj > 1.0:
            pj = 1.0
        pt = step_tail_jump_prob
        if pt < 0.0:
            pt = 0.0
        elif pt > 1.0:
            pt = 1.0

        out = -1e300
        w_none = (1.0 - pj) * (1.0 - pt)
        if w_none > 1e-300:
            comp = math.log(w_none) + _pf_transition_component_log_lk_numba(
                pos_delta,
                rate_delta,
                local_pos_noise,
                local_rate_noise,
                dmd,
                0.0,
                0.0,
                0.0,
                0.0,
                transition_scale,
                pos_floor,
                rate_floor,
                use_joint,
                include_norm,
            )
            out = _pf_logaddexp_numba(out, comp)
        w_jump = pj * (1.0 - pt)
        if w_jump > 1e-300:
            comp = math.log(w_jump) + _pf_transition_component_log_lk_numba(
                pos_delta,
                rate_delta,
                local_pos_noise,
                local_rate_noise,
                dmd,
                jump_sd,
                jump_rate_sd,
                0.0,
                0.0,
                transition_scale,
                pos_floor,
                rate_floor,
                use_joint,
                include_norm,
            )
            out = _pf_logaddexp_numba(out, comp)
        w_tail = (1.0 - pj) * pt
        if w_tail > 1e-300:
            comp = math.log(w_tail) + _pf_transition_component_log_lk_numba(
                pos_delta,
                rate_delta,
                local_pos_noise,
                local_rate_noise,
                dmd,
                0.0,
                0.0,
                jump_tail_sd,
                jump_tail_rate_sd,
                transition_scale,
                pos_floor,
                rate_floor,
                use_joint,
                include_norm,
            )
            out = _pf_logaddexp_numba(out, comp)
        w_both = pj * pt
        if w_both > 1e-300:
            comp = math.log(w_both) + _pf_transition_component_log_lk_numba(
                pos_delta,
                rate_delta,
                local_pos_noise,
                local_rate_noise,
                dmd,
                jump_sd,
                jump_rate_sd,
                jump_tail_sd,
                jump_tail_rate_sd,
                transition_scale,
                pos_floor,
                rate_floor,
                use_joint,
                include_norm,
            )
            out = _pf_logaddexp_numba(out, comp)
        return out

    @njit(cache=False)
    def _pf_transition_collapsed_joint_log_lk_numba(
        pos_delta,
        rate_delta,
        local_pos_noise,
        local_rate_noise,
        dmd,
        step_jump_prob,
        jump_sd,
        jump_rate_sd,
        step_tail_jump_prob,
        jump_tail_sd,
        jump_tail_rate_sd,
        transition_scale,
        pos_floor,
        rate_floor,
    ):
        raw_rate_var = local_rate_noise * local_rate_noise
        raw_pos_var = local_pos_noise * local_pos_noise + dmd * dmd * local_rate_noise * local_rate_noise
        if step_jump_prob > 0.0:
            raw_rate_var += step_jump_prob * jump_rate_sd * jump_rate_sd
            raw_pos_var += step_jump_prob * jump_sd * jump_sd
        if step_tail_jump_prob > 0.0:
            raw_rate_var += step_tail_jump_prob * jump_tail_rate_sd * jump_tail_rate_sd
            raw_pos_var += step_tail_jump_prob * jump_tail_sd * jump_tail_sd
        raw_cov = dmd * local_rate_noise * local_rate_noise
        return _pf_transition_joint_log_lk_numba(
            pos_delta,
            rate_delta,
            raw_pos_var,
            raw_rate_var,
            raw_cov,
            transition_scale,
            pos_floor,
            rate_floor,
            0,
        )

    @njit(cache=False)
    def _pf_heatmap_kernel(
        md,
        z,
        gr,
        gr_power,
        gr_finite,
        block_obs_gr,
        block_obs_power,
        block_obs_finite,
        block_obs_md_delta,
        block_obs_z,
        block_obs_count,
        obs_likelihood_mode,
        shape_gr,
        shape_md_delta,
        shape_z,
        shape_count,
        ref_grid,
        grid_min,
        grid_step,
        raw_ref_grid,
        raw_grid_min,
        raw_grid_step,
        raw_ref_weight,
        gr_sigma,
        outlier_prob,
        outlier_likelihood,
        dynamic_sigma_alpha,
        dynamic_sigma_threshold,
        dynamic_sigma_power,
        dynamic_sigma_min,
        dynamic_sigma_max,
        shape_power,
        shape_mode,
        shape_min_points,
        shape_sigma_floor,
        shape_ref_mode,
        init_pos,
        init_rate,
        initial_prev_md,
        tvt0,
        bin_idx,
        num_bins,
        typewell_window,
        typewell_len,
        n_particles,
        tvt_min,
        tvt_max,
        seed,
        momentum,
        rate_noise,
        pos_noise,
        resample_threshold,
        resample_obs_power_adapt,
        resample_min_threshold,
        rough_pos,
        rough_rate,
        rescue_frac,
        rescue_pos_sd,
        rescue_rate_sd,
        init_pos_sd,
        init_rate_sd,
        rate_mean_weight,
        jump_prob,
        jump_sd,
        jump_rate_sd,
        missing_jump_boost,
        jump_tail_prob,
        jump_tail_sd,
        jump_tail_rate_sd,
        jump_tail_dist,
        jump_tail_clip,
        jump_tail_missing_boost,
        anchor_sigma,
        anchor_power,
        correlated_rate_alpha,
        correlated_pos_alpha,
        lookahead_power,
        lookahead_steps,
        lookahead_decay,
        lookahead_max_gap,
        lookahead_delta_power,
        stratified_init,
        finite_run_power_boost,
        finite_run_power_cap,
        finite_run_power_decay,
        finite_run_power_floor,
        conf_obs_power_decay,
        conf_obs_power_floor,
        conf_obs_power_power,
        missing_noise_scale,
        missing_jump_scale,
        ess_jump_boost,
        ess_jump_power,
        surprise_jump_boost,
        surprise_jump_threshold,
        surprise_jump_power,
        ess_rough_boost,
        ess_rough_power,
        state_z_weight,
    ):
        np.random.seed(seed)
        hist_bins = np.zeros((num_bins, typewell_len))
        path_sums = np.zeros(num_bins)
        count_bins = np.zeros(num_bins)
        local_log_lik = np.zeros(num_bins)
        local_obs_power = np.zeros(num_bins)
        pos = np.empty(n_particles)
        rate = np.empty(n_particles)
        weight = np.ones(n_particles) / n_particles
        for j in range(n_particles):
            if stratified_init != 0:
                u = (j + np.random.random()) / n_particles
                pos[j] = init_pos + init_pos_sd * (2.0 * u - 1.0) * 1.7320508075688772
                u_rate = ((j * 37) % n_particles + np.random.random()) / n_particles
                rate[j] = init_rate + init_rate_sd * (2.0 * u_rate - 1.0) * 1.7320508075688772
            else:
                pos[j] = init_pos + init_pos_sd * np.random.randn()
                rate[j] = init_rate + init_rate_sd * np.random.randn()

        axis0 = -typewell_window + typewell_window / typewell_len
        axis_step = 2.0 * typewell_window / typewell_len
        prev_md = initial_prev_md
        anchor_z = z[0]
        if abs(state_z_weight) > 1e-12:
            anchor_z = (init_pos - tvt0) / state_z_weight
        log_lik = 0.0
        rate_shock = 0.0
        pos_shock = 0.0
        finite_run = 0.0
        sigma_mult = 1.0
        if outlier_prob < 0.0:
            outlier_prob = 0.0
        elif outlier_prob > 1.0:
            outlier_prob = 1.0
        if outlier_likelihood < 1e-300:
            outlier_likelihood = 1e-300
        elif outlier_likelihood > 1.0:
            outlier_likelihood = 1.0
        if dynamic_sigma_min <= 0.0:
            dynamic_sigma_min = 1e-6
        if dynamic_sigma_max < dynamic_sigma_min:
            dynamic_sigma_max = dynamic_sigma_min
        if dynamic_sigma_threshold <= 0.0:
            dynamic_sigma_threshold = 1.0
        if dynamic_sigma_power <= 0.0:
            dynamic_sigma_power = 1.0
        if jump_tail_prob < 0.0:
            jump_tail_prob = 0.0
        if jump_tail_sd < 0.0:
            jump_tail_sd = 0.0
        if jump_tail_rate_sd < 0.0:
            jump_tail_rate_sd = 0.0
        if jump_tail_sd <= 0.0 and jump_tail_rate_sd <= 0.0:
            jump_tail_prob = 0.0
            jump_tail_missing_boost = 0.0
        if jump_tail_prob <= 0.0 and jump_tail_missing_boost <= 0.0:
            jump_tail_prob = 0.0
            jump_tail_missing_boost = 0.0
        if jump_tail_dist < 0 or jump_tail_dist > 2:
            jump_tail_dist = 1
        if jump_tail_clip < 0.0:
            jump_tail_clip = 0.0
        if jump_tail_dist == 2 and jump_tail_clip <= 0.0 and jump_tail_sd > 0.0:
            jump_tail_clip = 6.0 * jump_tail_sd
        if lookahead_steps < 1:
            lookahead_steps = 1
        if lookahead_decay <= 0.0:
            lookahead_decay = 1.0
        if finite_run_power_decay < 0.0:
            finite_run_power_decay = 0.0
        if finite_run_power_floor < 0.0:
            finite_run_power_floor = 0.0
        elif finite_run_power_floor > 1.0:
            finite_run_power_floor = 1.0
        if conf_obs_power_decay < 0.0:
            conf_obs_power_decay = 0.0
        if conf_obs_power_floor < 0.0:
            conf_obs_power_floor = 0.0
        elif conf_obs_power_floor > 1.0:
            conf_obs_power_floor = 1.0
        if conf_obs_power_power <= 0.0:
            conf_obs_power_power = 1.0
        if ess_jump_boost < 0.0:
            ess_jump_boost = 0.0
        if ess_jump_power <= 0.0:
            ess_jump_power = 1.0
        if surprise_jump_boost < 0.0:
            surprise_jump_boost = 0.0
        if surprise_jump_threshold <= 0.0:
            surprise_jump_threshold = 1.0
        if surprise_jump_power <= 0.0:
            surprise_jump_power = 1.0
        prev_ess_frac = 1.0
        prev_surprise = 0.0
        obs_width = block_obs_power.shape[1]
        local_obs_power_arr = np.empty(obs_width)
        for i in range(len(md)):
            dmd = md[i] - prev_md
            if dmd < 1.0:
                dmd = 1.0
            missing_frac = 1.0 - gr_finite[i]
            if missing_frac < 0.0:
                missing_frac = 0.0
            elif missing_frac > 1.0:
                missing_frac = 1.0
            local_rate_noise = rate_noise
            local_pos_noise = pos_noise
            if missing_frac > 0.0 and missing_noise_scale != 1.0:
                noise_scale = 1.0 + (missing_noise_scale - 1.0) * missing_frac
                local_rate_noise *= noise_scale
                local_pos_noise *= noise_scale
            step_jump_prob = jump_prob * dmd
            if missing_frac > 0.0 and missing_jump_scale != 1.0:
                step_jump_prob *= 1.0 + (missing_jump_scale - 1.0) * missing_frac
            if gr_power[i] < 1.0 and missing_jump_boost > 0.0:
                step_jump_prob += missing_jump_boost * (1.0 - gr_power[i]) * dmd
            if ess_jump_boost > 0.0:
                ess_deficit = 1.0 - prev_ess_frac
                if ess_deficit > 0.0:
                    if ess_jump_power != 1.0:
                        ess_deficit = math.pow(ess_deficit, ess_jump_power)
                    step_jump_prob += ess_jump_boost * ess_deficit * dmd
            if surprise_jump_boost > 0.0 and prev_surprise > surprise_jump_threshold:
                surprise_excess = prev_surprise / surprise_jump_threshold - 1.0
                if surprise_excess > 0.0:
                    if surprise_jump_power != 1.0:
                        surprise_excess = math.pow(surprise_excess, surprise_jump_power)
                    step_jump_prob += surprise_jump_boost * surprise_excess * dmd
            if step_jump_prob > 1.0:
                step_jump_prob = 1.0
            step_tail_jump_prob = jump_tail_prob * dmd
            if missing_frac > 0.0 and missing_jump_scale != 1.0:
                step_tail_jump_prob *= 1.0 + (missing_jump_scale - 1.0) * missing_frac
            if gr_power[i] < 1.0 and jump_tail_missing_boost > 0.0:
                step_tail_jump_prob += jump_tail_missing_boost * (1.0 - gr_power[i]) * dmd
            if step_tail_jump_prob > 1.0:
                step_tail_jump_prob = 1.0
            if correlated_rate_alpha > 0.0:
                rate_shock = correlated_rate_alpha * rate_shock + math.sqrt(max(0.0, 1.0 - correlated_rate_alpha * correlated_rate_alpha)) * np.random.randn()
            if correlated_pos_alpha > 0.0:
                pos_shock = correlated_pos_alpha * pos_shock + math.sqrt(max(0.0, 1.0 - correlated_pos_alpha * correlated_pos_alpha)) * np.random.randn()
            for j in range(n_particles):
                rate_eps = np.random.randn()
                if correlated_rate_alpha > 0.0:
                    rate_eps = 0.5 * rate_eps + 0.8660254037844386 * rate_shock
                rate[j] = momentum * rate[j] + local_rate_noise * rate_eps
                if rate_mean_weight > 0.0:
                    rate[j] += rate_mean_weight * (init_rate - rate[j])
                pos_eps = np.random.randn()
                if correlated_pos_alpha > 0.0:
                    pos_eps = 0.5 * pos_eps + 0.8660254037844386 * pos_shock
                pos[j] = pos[j] + rate[j] * dmd + local_pos_noise * pos_eps
                if step_jump_prob > 0.0 and np.random.random() < step_jump_prob:
                    pos[j] += jump_sd * np.random.randn()
                    rate[j] += jump_rate_sd * np.random.randn()
                if step_tail_jump_prob > 0.0 and np.random.random() < step_tail_jump_prob:
                    pos[j] += _pf_jump_noise_numba(jump_tail_sd, jump_tail_dist, jump_tail_clip)
                    rate_clip = 0.0
                    if jump_tail_clip > 0.0 and jump_tail_sd > 0.0 and jump_tail_rate_sd > 0.0:
                        rate_clip = jump_tail_clip * jump_tail_rate_sd / jump_tail_sd
                    elif jump_tail_dist == 2 and jump_tail_rate_sd > 0.0:
                        rate_clip = 6.0 * jump_tail_rate_sd
                    rate[j] += _pf_jump_noise_numba(jump_tail_rate_sd, jump_tail_dist, rate_clip)
                tvt = pos[j] - state_z_weight * z[i]
                if tvt < tvt_min:
                    tvt = tvt_min
                elif tvt > tvt_max:
                    tvt = tvt_max
                pos[j] = tvt + state_z_weight * z[i]

            weight_sum = 0.0
            avg_lk = 0.0
            obs_power = gr_power[i]
            if obs_power < 0.0:
                obs_power = 0.0
            obs_count_i = block_obs_count[i]
            if obs_count_i < 1:
                obs_count_i = 1
            if obs_count_i > obs_width:
                obs_count_i = obs_width
            total_local_obs_power = 0.0
            if obs_likelihood_mode == 1:
                block_obs_power_sum = 0.0
                block_obs_power_adjusted_sum = 0.0
                for obs_idx in range(obs_count_i):
                    local_power = block_obs_power[i, obs_idx]
                    if local_power < 0.0:
                        local_power = 0.0
                    block_obs_power_sum += local_power
                    if block_obs_finite[i, obs_idx] > 0.0:
                        finite_run += 1.0
                    else:
                        finite_run = 0.0
                    if finite_run_power_boost > 0.0 and local_power > 0.0 and finite_run > 1.0:
                        local_power *= 1.0 + finite_run_power_boost * math.log(1.0 + finite_run)
                        if finite_run_power_cap > 0.0 and local_power > finite_run_power_cap:
                            local_power = finite_run_power_cap
                    if finite_run_power_decay > 0.0 and local_power > 0.0 and finite_run > 1.0:
                        run_mult = 1.0 / math.sqrt(1.0 + finite_run_power_decay * (finite_run - 1.0))
                        if run_mult < finite_run_power_floor:
                            run_mult = finite_run_power_floor
                        local_power *= run_mult
                    local_obs_power_arr[obs_idx] = local_power
                    block_obs_power_adjusted_sum += local_power
                obs_power = block_obs_power_adjusted_sum
                if block_obs_power_adjusted_sum > 0.0 and block_obs_power_sum > 0.0:
                    scale_to_merged_power = gr_power[i] / block_obs_power_sum
                    if scale_to_merged_power < 0.0:
                        scale_to_merged_power = 0.0
                    obs_power = block_obs_power_adjusted_sum * scale_to_merged_power
                    for obs_idx in range(obs_count_i):
                        local_obs_power_arr[obs_idx] *= scale_to_merged_power
                    total_local_obs_power = obs_power
            else:
                if gr_finite[i] > 0.0:
                    finite_run += 1.0
                else:
                    finite_run = 0.0
                if finite_run_power_boost > 0.0 and obs_power > 0.0 and finite_run > 1.0:
                    obs_power *= 1.0 + finite_run_power_boost * math.log(1.0 + finite_run)
                    if finite_run_power_cap > 0.0 and obs_power > finite_run_power_cap:
                        obs_power = finite_run_power_cap
                if finite_run_power_decay > 0.0 and obs_power > 0.0 and finite_run > 1.0:
                    run_mult = 1.0 / math.sqrt(1.0 + finite_run_power_decay * (finite_run - 1.0))
                    if run_mult < finite_run_power_floor:
                        run_mult = finite_run_power_floor
                    obs_power *= run_mult
                local_obs_power_arr[0] = obs_power
                total_local_obs_power = obs_power
            if conf_obs_power_decay > 0.0 and obs_power > 0.0 and prev_ess_frac < 1.0:
                conf_mult = 1.0 / (1.0 + conf_obs_power_decay * math.pow(1.0 - prev_ess_frac, conf_obs_power_power))
                if conf_mult < conf_obs_power_floor:
                    conf_mult = conf_obs_power_floor
                obs_power *= conf_mult
                if total_local_obs_power > 0.0:
                    for obs_idx in range(obs_count_i):
                        local_obs_power_arr[obs_idx] *= conf_mult
            sigma_eff = gr_sigma * sigma_mult
            if sigma_eff <= 1e-6:
                sigma_eff = 1e-6
            pred_gr_mean = 0.0
            pred_gr_weight = 0.0
            if obs_power > 0.0 and dynamic_sigma_alpha > 0.0 and gr_finite[i] > 0.0:
                for j in range(n_particles):
                    tvt_dyn = pos[j] - state_z_weight * z[i]
                    expected_gr_dyn = _interp_grid_numba(ref_grid, tvt_dyn, grid_min, grid_step)
                    if raw_ref_weight > 0.0:
                        raw_expected_gr_dyn = _interp_grid_numba(raw_ref_grid, tvt_dyn, raw_grid_min, raw_grid_step)
                        expected_gr_dyn = (1.0 - raw_ref_weight) * expected_gr_dyn + raw_ref_weight * raw_expected_gr_dyn
                    pred_gr_mean += weight[j] * expected_gr_dyn
                    pred_gr_weight += weight[j]
                if pred_gr_weight > 0.0:
                    pred_gr_mean /= pred_gr_weight
            for j in range(n_particles):
                tvt = pos[j] - state_z_weight * z[i]
                lk = 1.0
                if obs_power > 0.0:
                    if obs_likelihood_mode == 1:
                        for obs_idx in range(obs_count_i):
                            local_power = local_obs_power_arr[obs_idx]
                            if local_power <= 0.0:
                                continue
                            local_tvt = tvt + rate[j] * block_obs_md_delta[i, obs_idx] + state_z_weight * (z[i] - block_obs_z[i, obs_idx])
                            lk *= _pf_gr_point_likelihood_numba(
                                block_obs_gr[i, obs_idx],
                                local_tvt,
                                local_power,
                                block_obs_finite[i, obs_idx],
                                sigma_eff,
                                ref_grid,
                                grid_min,
                                grid_step,
                                raw_ref_grid,
                                raw_grid_min,
                                raw_grid_step,
                                raw_ref_weight,
                                outlier_prob,
                                outlier_likelihood,
                            )
                    else:
                        lk *= _pf_gr_point_likelihood_numba(
                            gr[i],
                            tvt,
                            obs_power,
                            gr_finite[i],
                            sigma_eff,
                            ref_grid,
                            grid_min,
                            grid_step,
                            raw_ref_grid,
                            raw_grid_min,
                            raw_grid_step,
                            raw_ref_weight,
                            outlier_prob,
                            outlier_likelihood,
                        )
                if shape_power > 0.0 and shape_mode > 0:
                    lk *= _pf_gr_shape_likelihood_numba(
                        i,
                        tvt,
                        rate[j],
                        z[i],
                        state_z_weight,
                        gr[i],
                        sigma_eff,
                        ref_grid,
                        grid_min,
                        grid_step,
                        raw_ref_grid,
                        raw_grid_min,
                        raw_grid_step,
                        raw_ref_weight,
                        shape_gr,
                        shape_md_delta,
                        shape_z,
                        shape_count,
                        shape_power,
                        shape_mode,
                        shape_min_points,
                        shape_sigma_floor,
                        shape_ref_mode,
                    )
                if lookahead_power > 0.0:
                    next_i = i + 1
                    found = 0
                    lookahead_weight = lookahead_power
                    while next_i < len(md) and found < lookahead_steps:
                        if gr_power[next_i] > 0.0 and gr_finite[next_i] > 0.0:
                            gap = md[next_i] - md[i]
                            if gap < 1.0:
                                gap = 1.0
                            if lookahead_max_gap > 0.0 and gap > lookahead_max_gap:
                                break
                            if lookahead_max_gap <= 0.0 or gap <= lookahead_max_gap:
                                future_tvt = pos[j] + rate[j] * gap - state_z_weight * z[next_i]
                                expected_gr_next = _interp_grid_numba(ref_grid, future_tvt, grid_min, grid_step)
                                d_next = (gr[next_i] - expected_gr_next) / sigma_eff
                                d2_next = d_next * d_next
                                if d2_next > 600.0:
                                    d2_next = 600.0
                                next_power = lookahead_weight * gr_power[next_i]
                                if lookahead_delta_power > 0.0 and gr_finite[i] > 0.0:
                                    expected_gr_now = _interp_grid_numba(ref_grid, tvt, grid_min, grid_step)
                                    obs_delta = gr[next_i] - gr[i]
                                    pred_delta = expected_gr_next - expected_gr_now
                                    delta_d = (obs_delta - pred_delta) / sigma_eff
                                    delta_d2 = delta_d * delta_d
                                    if delta_d2 > 600.0:
                                        delta_d2 = 600.0
                                    d2_next = (1.0 - lookahead_delta_power) * d2_next + lookahead_delta_power * delta_d2
                                next_lk = math.exp(-0.5 * d2_next * next_power)
                                if raw_ref_weight > 0.0:
                                    raw_expected_gr_next = _interp_grid_numba(raw_ref_grid, future_tvt, raw_grid_min, raw_grid_step)
                                    raw_d_next = (gr[next_i] - raw_expected_gr_next) / sigma_eff
                                    raw_d2_next = raw_d_next * raw_d_next
                                    if raw_d2_next > 600.0:
                                        raw_d2_next = 600.0
                                    raw_next_lk = math.exp(-0.5 * raw_d2_next * next_power)
                                    next_lk = (1.0 - raw_ref_weight) * next_lk + raw_ref_weight * raw_next_lk
                                if outlier_prob > 0.0:
                                    next_lk = (1.0 - outlier_prob) * next_lk + outlier_prob * outlier_likelihood
                                lk *= next_lk
                                found += 1
                                lookahead_weight *= lookahead_decay
                                if lookahead_weight <= 0.0:
                                    break
                        next_i += 1
                if anchor_sigma > 0.0 and anchor_power > 0.0:
                    if obs_likelihood_mode == 1:
                        for obs_idx in range(obs_count_i):
                            local_anchor_tvt = tvt + rate[j] * block_obs_md_delta[i, obs_idx] + state_z_weight * (z[i] - block_obs_z[i, obs_idx])
                            da = (local_anchor_tvt - tvt0) / anchor_sigma
                            da2 = da * da
                            if da2 > 600.0:
                                da2 = 600.0
                            lk *= math.exp(-0.5 * da2 * anchor_power)
                    else:
                        da = (tvt - tvt0) / anchor_sigma
                        da2 = da * da
                        if da2 > 600.0:
                            da2 = 600.0
                        lk *= math.exp(-0.5 * da2 * anchor_power)
                if lk < 1e-300:
                    lk = 1e-300
                avg_lk += weight[j] * lk
                weight[j] *= lk
                weight_sum += weight[j]
            if avg_lk < 1e-300:
                avg_lk = 1e-300
            local_log = math.log(avg_lk)
            log_lik += local_log
            b_local = bin_idx[i]
            if 0 <= b_local < num_bins:
                local_log_lik[b_local] += local_log
                local_obs_power[b_local] += obs_power
            if obs_power > 0.0 and dynamic_sigma_alpha > 0.0 and gr_finite[i] > 0.0 and pred_gr_weight > 0.0:
                surprise = abs(gr[i] - pred_gr_mean) / gr_sigma
                target_mult = math.pow(max(surprise / dynamic_sigma_threshold, 1e-6), dynamic_sigma_power)
                if target_mult < dynamic_sigma_min:
                    target_mult = dynamic_sigma_min
                elif target_mult > dynamic_sigma_max:
                    target_mult = dynamic_sigma_max
                sigma_mult = (1.0 - dynamic_sigma_alpha) * sigma_mult + dynamic_sigma_alpha * target_mult
                if sigma_mult < dynamic_sigma_min:
                    sigma_mult = dynamic_sigma_min
                elif sigma_mult > dynamic_sigma_max:
                    sigma_mult = dynamic_sigma_max
            if weight_sum > 0.0:
                for j in range(n_particles):
                    weight[j] /= weight_sum
            else:
                for j in range(n_particles):
                    weight[j] = 1.0 / n_particles

            b = bin_idx[i]
            if 0 <= b < num_bins:
                output_obs_idx = obs_count_i - 1
                if output_obs_idx < 0:
                    output_obs_idx = 0
                elif output_obs_idx >= obs_width:
                    output_obs_idx = obs_width - 1
                tvt_mean = 0.0
                for j in range(n_particles):
                    tvt = pos[j] - state_z_weight * z[i]
                    if obs_likelihood_mode == 1:
                        tvt = tvt + rate[j] * block_obs_md_delta[i, output_obs_idx] + state_z_weight * (z[i] - block_obs_z[i, output_obs_idx])
                    if tvt < tvt_min:
                        tvt = tvt_min
                    elif tvt > tvt_max:
                        tvt = tvt_max
                    tvt_mean += weight[j] * tvt
                    rel = tvt - tvt0
                    grid_pos = (rel - axis0) / axis_step
                    left = math.floor(grid_pos)
                    frac = grid_pos - left
                    if left >= 0 and left < typewell_len:
                        hist_bins[b, left] += weight[j] * (1.0 - frac)
                    right = left + 1
                    if right >= 0 and right < typewell_len:
                        hist_bins[b, right] += weight[j] * frac
                path_sums[b] += tvt_mean
                count_bins[b] += 1.0

            effective_n_inv = 0.0
            for j in range(n_particles):
                effective_n_inv += weight[j] * weight[j]
            if effective_n_inv > 0.0:
                prev_ess_frac = 1.0 / (effective_n_inv * n_particles)
                if prev_ess_frac < 0.0:
                    prev_ess_frac = 0.0
                elif prev_ess_frac > 1.0:
                    prev_ess_frac = 1.0
            else:
                prev_ess_frac = 0.0
            if gr_finite[i] > 0.0 and pred_gr_weight > 0.0 and gr_sigma > 1e-6:
                prev_surprise = abs(gr[i] - pred_gr_mean) / gr_sigma
            elif gr_finite[i] <= 0.0:
                prev_surprise *= 0.85
            local_resample_threshold = resample_threshold
            if resample_obs_power_adapt > 0.0 and local_resample_threshold > 0.0:
                adapt_scale = 1.0 / (1.0 + resample_obs_power_adapt * max(obs_power, 0.0))
                local_resample_threshold *= adapt_scale
                if resample_min_threshold > 0.0 and local_resample_threshold < resample_min_threshold:
                    local_resample_threshold = resample_min_threshold
            if local_resample_threshold > 0.0 and 1.0 / effective_n_inv < local_resample_threshold * n_particles:
                local_rough_pos = rough_pos
                local_rough_rate = rough_rate
                if ess_rough_boost > 0.0:
                    ess_frac = 1.0 / (effective_n_inv * n_particles)
                    deficit = local_resample_threshold - ess_frac
                    if deficit > 0.0:
                        if ess_rough_power != 1.0:
                            deficit = math.pow(deficit / max(local_resample_threshold, 1e-12), ess_rough_power) * local_resample_threshold
                        local_rough_pos *= 1.0 + ess_rough_boost * deficit / max(local_resample_threshold, 1e-12)
                        local_rough_rate *= 1.0 + ess_rough_boost * deficit / max(local_resample_threshold, 1e-12)
                pos, rate = _pf_resample_numba(pos, rate, weight, local_rough_pos, local_rough_rate)
                if rescue_frac > 0.0 and rescue_pos_sd > 0.0 and rescue_rate_sd > 0.0:
                    n_rescue = int(math.floor(rescue_frac * n_particles))
                    if n_rescue > 0:
                        rescue_center = init_pos + init_rate * (md[i] - md[0])
                        for j in range(n_rescue):
                            idx = n_particles - 1 - j
                            pos[idx] = rescue_center + rescue_pos_sd * np.random.randn()
                            rate[idx] = init_rate + rescue_rate_sd * np.random.randn()
                for j in range(n_particles):
                    tvt = pos[j] - state_z_weight * z[i]
                    if tvt < tvt_min:
                        tvt = tvt_min
                    elif tvt > tvt_max:
                        tvt = tvt_max
                    pos[j] = tvt + state_z_weight * z[i]
                    weight[j] = 1.0 / n_particles

            prev_md = md[i]
        return hist_bins, path_sums, count_bins, log_lik, local_log_lik, local_obs_power

    @njit(cache=False)
    def _pf_heatmap_ffbsi_kernel(
        md,
        z,
        gr,
        gr_power,
        gr_finite,
        block_obs_gr,
        block_obs_power,
        block_obs_finite,
        block_obs_md_delta,
        block_obs_z,
        block_obs_count,
        obs_likelihood_mode,
        shape_gr,
        shape_md_delta,
        shape_z,
        shape_count,
        ref_grid,
        grid_min,
        grid_step,
        raw_ref_grid,
        raw_grid_min,
        raw_grid_step,
        raw_ref_weight,
        gr_sigma,
        outlier_prob,
        outlier_likelihood,
        dynamic_sigma_alpha,
        dynamic_sigma_threshold,
        dynamic_sigma_power,
        dynamic_sigma_min,
        dynamic_sigma_max,
        shape_power,
        shape_mode,
        shape_min_points,
        shape_sigma_floor,
        shape_ref_mode,
        init_pos,
        init_rate,
        initial_prev_md,
        tvt0,
        bin_idx,
        num_bins,
        typewell_window,
        typewell_len,
        n_particles,
        tvt_min,
        tvt_max,
        seed,
        momentum,
        rate_noise,
        pos_noise,
        resample_threshold,
        resample_obs_power_adapt,
        resample_min_threshold,
        rough_pos,
        rough_rate,
        rescue_frac,
        rescue_pos_sd,
        rescue_rate_sd,
        init_pos_sd,
        init_rate_sd,
        rate_mean_weight,
        jump_prob,
        jump_sd,
        jump_rate_sd,
        missing_jump_boost,
        jump_tail_prob,
        jump_tail_sd,
        jump_tail_rate_sd,
        jump_tail_dist,
        jump_tail_clip,
        jump_tail_missing_boost,
        anchor_sigma,
        anchor_power,
        correlated_rate_alpha,
        correlated_pos_alpha,
        lookahead_power,
        lookahead_steps,
        lookahead_decay,
        lookahead_max_gap,
        lookahead_delta_power,
        stratified_init,
        finite_run_power_boost,
        finite_run_power_cap,
        finite_run_power_decay,
        finite_run_power_floor,
        conf_obs_power_decay,
        conf_obs_power_floor,
        conf_obs_power_power,
        missing_noise_scale,
        missing_jump_scale,
        ess_jump_boost,
        ess_jump_power,
        surprise_jump_boost,
        surprise_jump_threshold,
        surprise_jump_power,
        ess_rough_boost,
        ess_rough_power,
        state_z_weight,
        n_paths,
        max_active_bins,
        transition_model_code,
        transition_scale,
        pos_floor,
        rate_floor,
    ):
        np.random.seed(seed)
        hist_bins = np.zeros((num_bins, typewell_len))
        path_sums = np.zeros(num_bins)
        count_bins = np.zeros(num_bins)
        filtered_hist = np.zeros((num_bins, typewell_len))
        filtered_path_sums = np.zeros(num_bins)
        filtered_count_bins = np.zeros(num_bins)
        local_log_lik = np.zeros(num_bins)
        local_obs_power = np.zeros(num_bins)
        pos = np.empty(n_particles)
        rate = np.empty(n_particles)
        weight = np.ones(n_particles) / n_particles

        bin_map = -np.ones(num_bins, dtype=np.int64)
        active_bins = np.empty(num_bins, dtype=np.int64)
        active_row_start = np.empty(num_bins, dtype=np.int64)
        active_row_end = np.empty(num_bins, dtype=np.int64)
        active_count = 0
        prev_bin = -1
        for i0 in range(len(md)):
            b0 = bin_idx[i0]
            if b0 >= 0 and b0 < num_bins:
                if b0 != prev_bin:
                    if active_count >= max_active_bins:
                        return (
                            filtered_hist,
                            filtered_path_sums,
                            filtered_count_bins,
                            hist_bins,
                            path_sums,
                            count_bins,
                            -1e300,
                            local_log_lik,
                            local_obs_power,
                            0,
                        )
                    bin_map[b0] = active_count
                    active_bins[active_count] = b0
                    active_row_start[active_count] = i0
                    active_row_end[active_count] = i0
                    active_count += 1
                    prev_bin = b0
                else:
                    snap_idx0 = bin_map[b0]
                    if snap_idx0 >= 0:
                        active_row_end[snap_idx0] = i0
        if active_count <= 0 or n_paths <= 0:
            return (
                filtered_hist,
                filtered_path_sums,
                filtered_count_bins,
                hist_bins,
                path_sums,
                count_bins,
                -1e300,
                local_log_lik,
                local_obs_power,
                0,
            )

        snap_pos = np.empty((active_count, n_particles))
        snap_rate = np.empty((active_count, n_particles))
        snap_weight = np.empty((active_count, n_particles))
        snap_md = np.empty(active_count)
        snap_z = np.empty(active_count)
        snap_output_md_delta = np.zeros(active_count)
        snap_output_z = np.empty(active_count)
        snap_gr_power = np.empty(active_count)
        snap_gr_finite = np.empty(active_count)
        row_to_snap = -np.ones(len(md), dtype=np.int64)
        for a0 in range(active_count):
            row_to_snap[(active_row_start[a0] + active_row_end[a0]) // 2] = a0

        for j in range(n_particles):
            if stratified_init != 0:
                u = (j + np.random.random()) / n_particles
                pos[j] = init_pos + init_pos_sd * (2.0 * u - 1.0) * 1.7320508075688772
                u_rate = ((j * 37) % n_particles + np.random.random()) / n_particles
                rate[j] = init_rate + init_rate_sd * (2.0 * u_rate - 1.0) * 1.7320508075688772
            else:
                pos[j] = init_pos + init_pos_sd * np.random.randn()
                rate[j] = init_rate + init_rate_sd * np.random.randn()

        axis0 = -typewell_window + typewell_window / typewell_len
        axis_step = 2.0 * typewell_window / typewell_len
        prev_md = initial_prev_md
        anchor_z = z[0]
        if abs(state_z_weight) > 1e-12:
            anchor_z = (init_pos - tvt0) / state_z_weight
        log_lik = 0.0
        rate_shock = 0.0
        pos_shock = 0.0
        finite_run = 0.0
        sigma_mult = 1.0
        if outlier_prob < 0.0:
            outlier_prob = 0.0
        elif outlier_prob > 1.0:
            outlier_prob = 1.0
        if outlier_likelihood < 1e-300:
            outlier_likelihood = 1e-300
        elif outlier_likelihood > 1.0:
            outlier_likelihood = 1.0
        if dynamic_sigma_min <= 0.0:
            dynamic_sigma_min = 1e-6
        if dynamic_sigma_max < dynamic_sigma_min:
            dynamic_sigma_max = dynamic_sigma_min
        if dynamic_sigma_threshold <= 0.0:
            dynamic_sigma_threshold = 1.0
        if dynamic_sigma_power <= 0.0:
            dynamic_sigma_power = 1.0
        if jump_tail_prob < 0.0:
            jump_tail_prob = 0.0
        if jump_tail_sd < 0.0:
            jump_tail_sd = 0.0
        if jump_tail_rate_sd < 0.0:
            jump_tail_rate_sd = 0.0
        if jump_tail_sd <= 0.0 and jump_tail_rate_sd <= 0.0:
            jump_tail_prob = 0.0
            jump_tail_missing_boost = 0.0
        if jump_tail_prob <= 0.0 and jump_tail_missing_boost <= 0.0:
            jump_tail_prob = 0.0
            jump_tail_missing_boost = 0.0
        if jump_tail_dist < 0 or jump_tail_dist > 2:
            jump_tail_dist = 1
        if jump_tail_clip < 0.0:
            jump_tail_clip = 0.0
        if jump_tail_dist == 2 and jump_tail_clip <= 0.0 and jump_tail_sd > 0.0:
            jump_tail_clip = 6.0 * jump_tail_sd
        if lookahead_steps < 1:
            lookahead_steps = 1
        if lookahead_decay <= 0.0:
            lookahead_decay = 1.0
        if finite_run_power_decay < 0.0:
            finite_run_power_decay = 0.0
        if finite_run_power_floor < 0.0:
            finite_run_power_floor = 0.0
        elif finite_run_power_floor > 1.0:
            finite_run_power_floor = 1.0
        if conf_obs_power_decay < 0.0:
            conf_obs_power_decay = 0.0
        if conf_obs_power_floor < 0.0:
            conf_obs_power_floor = 0.0
        elif conf_obs_power_floor > 1.0:
            conf_obs_power_floor = 1.0
        if conf_obs_power_power <= 0.0:
            conf_obs_power_power = 1.0
        if ess_jump_boost < 0.0:
            ess_jump_boost = 0.0
        if ess_jump_power <= 0.0:
            ess_jump_power = 1.0
        if surprise_jump_boost < 0.0:
            surprise_jump_boost = 0.0
        if surprise_jump_threshold <= 0.0:
            surprise_jump_threshold = 1.0
        if surprise_jump_power <= 0.0:
            surprise_jump_power = 1.0
        prev_ess_frac = 1.0
        prev_surprise = 0.0
        obs_width = block_obs_power.shape[1]
        local_obs_power_arr = np.empty(obs_width)

        for i in range(len(md)):
            dmd = md[i] - prev_md
            if dmd < 1.0:
                dmd = 1.0
            missing_frac = 1.0 - gr_finite[i]
            if missing_frac < 0.0:
                missing_frac = 0.0
            elif missing_frac > 1.0:
                missing_frac = 1.0
            local_rate_noise = rate_noise
            local_pos_noise = pos_noise
            if missing_frac > 0.0 and missing_noise_scale != 1.0:
                noise_scale = 1.0 + (missing_noise_scale - 1.0) * missing_frac
                local_rate_noise *= noise_scale
                local_pos_noise *= noise_scale
            step_jump_prob = jump_prob * dmd
            if missing_frac > 0.0 and missing_jump_scale != 1.0:
                step_jump_prob *= 1.0 + (missing_jump_scale - 1.0) * missing_frac
            if gr_power[i] < 1.0 and missing_jump_boost > 0.0:
                step_jump_prob += missing_jump_boost * (1.0 - gr_power[i]) * dmd
            if ess_jump_boost > 0.0:
                ess_deficit = 1.0 - prev_ess_frac
                if ess_deficit > 0.0:
                    if ess_jump_power != 1.0:
                        ess_deficit = math.pow(ess_deficit, ess_jump_power)
                    step_jump_prob += ess_jump_boost * ess_deficit * dmd
            if surprise_jump_boost > 0.0 and prev_surprise > surprise_jump_threshold:
                surprise_excess = prev_surprise / surprise_jump_threshold - 1.0
                if surprise_excess > 0.0:
                    if surprise_jump_power != 1.0:
                        surprise_excess = math.pow(surprise_excess, surprise_jump_power)
                    step_jump_prob += surprise_jump_boost * surprise_excess * dmd
            if step_jump_prob > 1.0:
                step_jump_prob = 1.0
            step_tail_jump_prob = jump_tail_prob * dmd
            if missing_frac > 0.0 and missing_jump_scale != 1.0:
                step_tail_jump_prob *= 1.0 + (missing_jump_scale - 1.0) * missing_frac
            if gr_power[i] < 1.0 and jump_tail_missing_boost > 0.0:
                step_tail_jump_prob += jump_tail_missing_boost * (1.0 - gr_power[i]) * dmd
            if step_tail_jump_prob > 1.0:
                step_tail_jump_prob = 1.0
            if correlated_rate_alpha > 0.0:
                rate_shock = correlated_rate_alpha * rate_shock + math.sqrt(max(0.0, 1.0 - correlated_rate_alpha * correlated_rate_alpha)) * np.random.randn()
            if correlated_pos_alpha > 0.0:
                pos_shock = correlated_pos_alpha * pos_shock + math.sqrt(max(0.0, 1.0 - correlated_pos_alpha * correlated_pos_alpha)) * np.random.randn()
            for j in range(n_particles):
                rate_eps = np.random.randn()
                if correlated_rate_alpha > 0.0:
                    rate_eps = 0.5 * rate_eps + 0.8660254037844386 * rate_shock
                rate[j] = momentum * rate[j] + local_rate_noise * rate_eps
                if rate_mean_weight > 0.0:
                    rate[j] += rate_mean_weight * (init_rate - rate[j])
                pos_eps = np.random.randn()
                if correlated_pos_alpha > 0.0:
                    pos_eps = 0.5 * pos_eps + 0.8660254037844386 * pos_shock
                pos[j] = pos[j] + rate[j] * dmd + local_pos_noise * pos_eps
                if step_jump_prob > 0.0 and np.random.random() < step_jump_prob:
                    pos[j] += jump_sd * np.random.randn()
                    rate[j] += jump_rate_sd * np.random.randn()
                if step_tail_jump_prob > 0.0 and np.random.random() < step_tail_jump_prob:
                    pos[j] += _pf_jump_noise_numba(jump_tail_sd, jump_tail_dist, jump_tail_clip)
                    rate_clip = 0.0
                    if jump_tail_clip > 0.0 and jump_tail_sd > 0.0 and jump_tail_rate_sd > 0.0:
                        rate_clip = jump_tail_clip * jump_tail_rate_sd / jump_tail_sd
                    elif jump_tail_dist == 2 and jump_tail_rate_sd > 0.0:
                        rate_clip = 6.0 * jump_tail_rate_sd
                    rate[j] += _pf_jump_noise_numba(jump_tail_rate_sd, jump_tail_dist, rate_clip)
                tvt = pos[j] - state_z_weight * z[i]
                if tvt < tvt_min:
                    tvt = tvt_min
                elif tvt > tvt_max:
                    tvt = tvt_max
                pos[j] = tvt + state_z_weight * z[i]

            weight_sum = 0.0
            avg_lk = 0.0
            obs_power = gr_power[i]
            if obs_power < 0.0:
                obs_power = 0.0
            obs_count_i = block_obs_count[i]
            if obs_count_i < 1:
                obs_count_i = 1
            if obs_count_i > obs_width:
                obs_count_i = obs_width
            total_local_obs_power = 0.0
            if obs_likelihood_mode == 1:
                block_obs_power_sum = 0.0
                block_obs_power_adjusted_sum = 0.0
                for obs_idx in range(obs_count_i):
                    local_power = block_obs_power[i, obs_idx]
                    if local_power < 0.0:
                        local_power = 0.0
                    block_obs_power_sum += local_power
                    if block_obs_finite[i, obs_idx] > 0.0:
                        finite_run += 1.0
                    else:
                        finite_run = 0.0
                    if finite_run_power_boost > 0.0 and local_power > 0.0 and finite_run > 1.0:
                        local_power *= 1.0 + finite_run_power_boost * math.log(1.0 + finite_run)
                        if finite_run_power_cap > 0.0 and local_power > finite_run_power_cap:
                            local_power = finite_run_power_cap
                    if finite_run_power_decay > 0.0 and local_power > 0.0 and finite_run > 1.0:
                        run_mult = 1.0 / math.sqrt(1.0 + finite_run_power_decay * (finite_run - 1.0))
                        if run_mult < finite_run_power_floor:
                            run_mult = finite_run_power_floor
                        local_power *= run_mult
                    local_obs_power_arr[obs_idx] = local_power
                    block_obs_power_adjusted_sum += local_power
                obs_power = block_obs_power_adjusted_sum
                if block_obs_power_adjusted_sum > 0.0 and block_obs_power_sum > 0.0:
                    scale_to_merged_power = gr_power[i] / block_obs_power_sum
                    if scale_to_merged_power < 0.0:
                        scale_to_merged_power = 0.0
                    obs_power = block_obs_power_adjusted_sum * scale_to_merged_power
                    for obs_idx in range(obs_count_i):
                        local_obs_power_arr[obs_idx] *= scale_to_merged_power
                    total_local_obs_power = obs_power
            else:
                if gr_finite[i] > 0.0:
                    finite_run += 1.0
                else:
                    finite_run = 0.0
                if finite_run_power_boost > 0.0 and obs_power > 0.0 and finite_run > 1.0:
                    obs_power *= 1.0 + finite_run_power_boost * math.log(1.0 + finite_run)
                    if finite_run_power_cap > 0.0 and obs_power > finite_run_power_cap:
                        obs_power = finite_run_power_cap
                if finite_run_power_decay > 0.0 and obs_power > 0.0 and finite_run > 1.0:
                    run_mult = 1.0 / math.sqrt(1.0 + finite_run_power_decay * (finite_run - 1.0))
                    if run_mult < finite_run_power_floor:
                        run_mult = finite_run_power_floor
                    obs_power *= run_mult
                local_obs_power_arr[0] = obs_power
                total_local_obs_power = obs_power
            if conf_obs_power_decay > 0.0 and obs_power > 0.0 and prev_ess_frac < 1.0:
                conf_mult = 1.0 / (1.0 + conf_obs_power_decay * math.pow(1.0 - prev_ess_frac, conf_obs_power_power))
                if conf_mult < conf_obs_power_floor:
                    conf_mult = conf_obs_power_floor
                obs_power *= conf_mult
                if total_local_obs_power > 0.0:
                    for obs_idx in range(obs_count_i):
                        local_obs_power_arr[obs_idx] *= conf_mult
            sigma_eff = gr_sigma * sigma_mult
            if sigma_eff <= 1e-6:
                sigma_eff = 1e-6
            pred_gr_mean = 0.0
            pred_gr_weight = 0.0
            if obs_power > 0.0 and dynamic_sigma_alpha > 0.0 and gr_finite[i] > 0.0:
                for j in range(n_particles):
                    tvt_dyn = pos[j] - state_z_weight * z[i]
                    expected_gr_dyn = _interp_grid_numba(ref_grid, tvt_dyn, grid_min, grid_step)
                    if raw_ref_weight > 0.0:
                        raw_expected_gr_dyn = _interp_grid_numba(raw_ref_grid, tvt_dyn, raw_grid_min, raw_grid_step)
                        expected_gr_dyn = (1.0 - raw_ref_weight) * expected_gr_dyn + raw_ref_weight * raw_expected_gr_dyn
                    pred_gr_mean += weight[j] * expected_gr_dyn
                    pred_gr_weight += weight[j]
                if pred_gr_weight > 0.0:
                    pred_gr_mean /= pred_gr_weight
            for j in range(n_particles):
                tvt = pos[j] - state_z_weight * z[i]
                lk = 1.0
                if obs_power > 0.0:
                    if obs_likelihood_mode == 1:
                        for obs_idx in range(obs_count_i):
                            local_power = local_obs_power_arr[obs_idx]
                            if local_power <= 0.0:
                                continue
                            local_tvt = tvt + rate[j] * block_obs_md_delta[i, obs_idx] + state_z_weight * (z[i] - block_obs_z[i, obs_idx])
                            lk *= _pf_gr_point_likelihood_numba(
                                block_obs_gr[i, obs_idx],
                                local_tvt,
                                local_power,
                                block_obs_finite[i, obs_idx],
                                sigma_eff,
                                ref_grid,
                                grid_min,
                                grid_step,
                                raw_ref_grid,
                                raw_grid_min,
                                raw_grid_step,
                                raw_ref_weight,
                                outlier_prob,
                                outlier_likelihood,
                            )
                    else:
                        lk *= _pf_gr_point_likelihood_numba(
                            gr[i],
                            tvt,
                            obs_power,
                            gr_finite[i],
                            sigma_eff,
                            ref_grid,
                            grid_min,
                            grid_step,
                            raw_ref_grid,
                            raw_grid_min,
                            raw_grid_step,
                            raw_ref_weight,
                            outlier_prob,
                            outlier_likelihood,
                        )
                if shape_power > 0.0 and shape_mode > 0:
                    lk *= _pf_gr_shape_likelihood_numba(
                        i,
                        tvt,
                        rate[j],
                        z[i],
                        state_z_weight,
                        gr[i],
                        sigma_eff,
                        ref_grid,
                        grid_min,
                        grid_step,
                        raw_ref_grid,
                        raw_grid_min,
                        raw_grid_step,
                        raw_ref_weight,
                        shape_gr,
                        shape_md_delta,
                        shape_z,
                        shape_count,
                        shape_power,
                        shape_mode,
                        shape_min_points,
                        shape_sigma_floor,
                        shape_ref_mode,
                    )
                if lookahead_power > 0.0:
                    next_i = i + 1
                    found = 0
                    lookahead_weight = lookahead_power
                    while next_i < len(md) and found < lookahead_steps:
                        if gr_power[next_i] > 0.0 and gr_finite[next_i] > 0.0:
                            gap = md[next_i] - md[i]
                            if gap < 1.0:
                                gap = 1.0
                            if lookahead_max_gap > 0.0 and gap > lookahead_max_gap:
                                break
                            if lookahead_max_gap <= 0.0 or gap <= lookahead_max_gap:
                                future_tvt = pos[j] + rate[j] * gap - state_z_weight * z[next_i]
                                expected_gr_next = _interp_grid_numba(ref_grid, future_tvt, grid_min, grid_step)
                                d_next = (gr[next_i] - expected_gr_next) / sigma_eff
                                d2_next = d_next * d_next
                                if d2_next > 600.0:
                                    d2_next = 600.0
                                next_power = lookahead_weight * gr_power[next_i]
                                if lookahead_delta_power > 0.0 and gr_finite[i] > 0.0:
                                    expected_gr_now = _interp_grid_numba(ref_grid, tvt, grid_min, grid_step)
                                    obs_delta = gr[next_i] - gr[i]
                                    pred_delta = expected_gr_next - expected_gr_now
                                    delta_d = (obs_delta - pred_delta) / sigma_eff
                                    delta_d2 = delta_d * delta_d
                                    if delta_d2 > 600.0:
                                        delta_d2 = 600.0
                                    d2_next = (1.0 - lookahead_delta_power) * d2_next + lookahead_delta_power * delta_d2
                                next_lk = math.exp(-0.5 * d2_next * next_power)
                                if raw_ref_weight > 0.0:
                                    raw_expected_gr_next = _interp_grid_numba(raw_ref_grid, future_tvt, raw_grid_min, raw_grid_step)
                                    raw_d_next = (gr[next_i] - raw_expected_gr_next) / sigma_eff
                                    raw_d2_next = raw_d_next * raw_d_next
                                    if raw_d2_next > 600.0:
                                        raw_d2_next = 600.0
                                    raw_next_lk = math.exp(-0.5 * raw_d2_next * next_power)
                                    next_lk = (1.0 - raw_ref_weight) * next_lk + raw_ref_weight * raw_next_lk
                                if outlier_prob > 0.0:
                                    next_lk = (1.0 - outlier_prob) * next_lk + outlier_prob * outlier_likelihood
                                lk *= next_lk
                                found += 1
                                lookahead_weight *= lookahead_decay
                                if lookahead_weight <= 0.0:
                                    break
                        next_i += 1
                if anchor_sigma > 0.0 and anchor_power > 0.0:
                    if obs_likelihood_mode == 1:
                        for obs_idx in range(obs_count_i):
                            local_anchor_tvt = tvt + rate[j] * block_obs_md_delta[i, obs_idx] + state_z_weight * (z[i] - block_obs_z[i, obs_idx])
                            da = (local_anchor_tvt - tvt0) / anchor_sigma
                            da2 = da * da
                            if da2 > 600.0:
                                da2 = 600.0
                            lk *= math.exp(-0.5 * da2 * anchor_power)
                    else:
                        da = (tvt - tvt0) / anchor_sigma
                        da2 = da * da
                        if da2 > 600.0:
                            da2 = 600.0
                        lk *= math.exp(-0.5 * da2 * anchor_power)
                if lk < 1e-300:
                    lk = 1e-300
                avg_lk += weight[j] * lk
                weight[j] *= lk
                weight_sum += weight[j]
            if avg_lk < 1e-300:
                avg_lk = 1e-300
            local_log = math.log(avg_lk)
            log_lik += local_log
            b_local = bin_idx[i]
            if 0 <= b_local < num_bins:
                local_log_lik[b_local] += local_log
                local_obs_power[b_local] += obs_power
            if obs_power > 0.0 and dynamic_sigma_alpha > 0.0 and gr_finite[i] > 0.0 and pred_gr_weight > 0.0:
                surprise = abs(gr[i] - pred_gr_mean) / gr_sigma
                target_mult = math.pow(max(surprise / dynamic_sigma_threshold, 1e-6), dynamic_sigma_power)
                if target_mult < dynamic_sigma_min:
                    target_mult = dynamic_sigma_min
                elif target_mult > dynamic_sigma_max:
                    target_mult = dynamic_sigma_max
                sigma_mult = (1.0 - dynamic_sigma_alpha) * sigma_mult + dynamic_sigma_alpha * target_mult
                if sigma_mult < dynamic_sigma_min:
                    sigma_mult = dynamic_sigma_min
                elif sigma_mult > dynamic_sigma_max:
                    sigma_mult = dynamic_sigma_max
            if weight_sum > 0.0:
                for j in range(n_particles):
                    weight[j] /= weight_sum
            else:
                for j in range(n_particles):
                    weight[j] = 1.0 / n_particles

            b = bin_idx[i]
            if 0 <= b < num_bins:
                output_obs_idx = obs_count_i - 1
                if output_obs_idx < 0:
                    output_obs_idx = 0
                elif output_obs_idx >= obs_width:
                    output_obs_idx = obs_width - 1
                tvt_mean = 0.0
                for j in range(n_particles):
                    tvt = pos[j] - state_z_weight * z[i]
                    if obs_likelihood_mode == 1:
                        tvt = tvt + rate[j] * block_obs_md_delta[i, output_obs_idx] + state_z_weight * (z[i] - block_obs_z[i, output_obs_idx])
                    if tvt < tvt_min:
                        tvt = tvt_min
                    elif tvt > tvt_max:
                        tvt = tvt_max
                    tvt_mean += weight[j] * tvt
                    rel = tvt - tvt0
                    grid_pos = (rel - axis0) / axis_step
                    left = math.floor(grid_pos)
                    frac = grid_pos - left
                    if left >= 0 and left < typewell_len:
                        filtered_hist[b, left] += weight[j] * (1.0 - frac)
                    right = left + 1
                    if right >= 0 and right < typewell_len:
                        filtered_hist[b, right] += weight[j] * frac
                filtered_path_sums[b] += tvt_mean
                filtered_count_bins[b] += 1.0
                snap_idx = row_to_snap[i]
                if snap_idx >= 0:
                    snap_md[snap_idx] = md[i]
                    snap_z[snap_idx] = z[i]
                    if obs_likelihood_mode == 1:
                        snap_output_md_delta[snap_idx] = block_obs_md_delta[i, output_obs_idx]
                        snap_output_z[snap_idx] = block_obs_z[i, output_obs_idx]
                    else:
                        snap_output_md_delta[snap_idx] = 0.0
                        snap_output_z[snap_idx] = z[i]
                    snap_gr_power[snap_idx] = gr_power[i]
                    snap_gr_finite[snap_idx] = gr_finite[i]
                    for j in range(n_particles):
                        snap_pos[snap_idx, j] = pos[j]
                        snap_rate[snap_idx, j] = rate[j]
                        snap_weight[snap_idx, j] = weight[j]

            effective_n_inv = 0.0
            for j in range(n_particles):
                effective_n_inv += weight[j] * weight[j]
            if effective_n_inv > 0.0:
                prev_ess_frac = 1.0 / (effective_n_inv * n_particles)
                if prev_ess_frac < 0.0:
                    prev_ess_frac = 0.0
                elif prev_ess_frac > 1.0:
                    prev_ess_frac = 1.0
            else:
                prev_ess_frac = 0.0
            if gr_finite[i] > 0.0 and pred_gr_weight > 0.0 and gr_sigma > 1e-6:
                prev_surprise = abs(gr[i] - pred_gr_mean) / gr_sigma
            elif gr_finite[i] <= 0.0:
                prev_surprise *= 0.85
            local_resample_threshold = resample_threshold
            if resample_obs_power_adapt > 0.0 and local_resample_threshold > 0.0:
                adapt_scale = 1.0 / (1.0 + resample_obs_power_adapt * max(obs_power, 0.0))
                local_resample_threshold *= adapt_scale
                if resample_min_threshold > 0.0 and local_resample_threshold < resample_min_threshold:
                    local_resample_threshold = resample_min_threshold
            if local_resample_threshold > 0.0 and 1.0 / effective_n_inv < local_resample_threshold * n_particles:
                local_rough_pos = rough_pos
                local_rough_rate = rough_rate
                if ess_rough_boost > 0.0:
                    ess_frac = 1.0 / (effective_n_inv * n_particles)
                    deficit = local_resample_threshold - ess_frac
                    if deficit > 0.0:
                        if ess_rough_power != 1.0:
                            deficit = math.pow(deficit / max(local_resample_threshold, 1e-12), ess_rough_power) * local_resample_threshold
                        local_rough_pos *= 1.0 + ess_rough_boost * deficit / max(local_resample_threshold, 1e-12)
                        local_rough_rate *= 1.0 + ess_rough_boost * deficit / max(local_resample_threshold, 1e-12)
                pos, rate = _pf_resample_numba(pos, rate, weight, local_rough_pos, local_rough_rate)
                if rescue_frac > 0.0 and rescue_pos_sd > 0.0 and rescue_rate_sd > 0.0:
                    n_rescue = int(math.floor(rescue_frac * n_particles))
                    if n_rescue > 0:
                        rescue_center = init_pos + init_rate * (md[i] - md[0])
                        for j in range(n_rescue):
                            idx = n_particles - 1 - j
                            pos[idx] = rescue_center + rescue_pos_sd * np.random.randn()
                            rate[idx] = init_rate + rescue_rate_sd * np.random.randn()
                for j in range(n_particles):
                    tvt = pos[j] - state_z_weight * z[i]
                    if tvt < tvt_min:
                        tvt = tvt_min
                    elif tvt > tvt_max:
                        tvt = tvt_max
                    pos[j] = tvt + state_z_weight * z[i]
                    weight[j] = 1.0 / n_particles

            prev_md = md[i]

        if active_count <= 1:
            return (
                filtered_hist,
                filtered_path_sums,
                filtered_count_bins,
                hist_bins,
                path_sums,
                count_bins,
                log_lik,
                local_log_lik,
                local_obs_power,
                0,
            )

        log_score = np.empty(n_particles)
        for s in range(n_paths):
            next_idx = _pf_sample_weight_index_numba(snap_weight[active_count - 1], n_particles)
            next_pos = snap_pos[active_count - 1, next_idx]
            next_rate = snap_rate[active_count - 1, next_idx]
            b_last = active_bins[active_count - 1]
            tvt_last = next_pos - state_z_weight * snap_z[active_count - 1]
            if obs_likelihood_mode == 1:
                tvt_last = tvt_last + next_rate * snap_output_md_delta[active_count - 1] + state_z_weight * (snap_z[active_count - 1] - snap_output_z[active_count - 1])
            if tvt_last < tvt_min:
                tvt_last = tvt_min
            elif tvt_last > tvt_max:
                tvt_last = tvt_max
            rel_last = tvt_last - tvt0
            grid_pos_last = (rel_last - axis0) / axis_step
            left_last = math.floor(grid_pos_last)
            frac_last = grid_pos_last - left_last
            if left_last >= 0 and left_last < typewell_len:
                hist_bins[b_last, left_last] += 1.0 - frac_last
            right_last = left_last + 1
            if right_last >= 0 and right_last < typewell_len:
                hist_bins[b_last, right_last] += frac_last
            path_sums[b_last] += tvt_last
            count_bins[b_last] += 1.0

            for a in range(active_count - 2, -1, -1):
                dmd = snap_md[a + 1] - snap_md[a]
                if dmd < 1.0:
                    dmd = 1.0
                if transition_model_code == 0:
                    pos_sd, rate_sd = _pf_transition_scale_numba(
                        dmd,
                        snap_gr_power[a + 1],
                        snap_gr_finite[a + 1],
                        rate_noise,
                        pos_noise,
                        jump_prob,
                        jump_sd,
                        jump_rate_sd,
                        missing_jump_boost,
                        jump_tail_prob,
                        jump_tail_sd,
                        jump_tail_rate_sd,
                        jump_tail_missing_boost,
                        missing_noise_scale,
                        missing_jump_scale,
                        transition_scale,
                        pos_floor,
                        rate_floor,
                    )
                    local_rate_noise = 0.0
                    local_pos_noise = 0.0
                    step_jump_prob = 0.0
                    step_tail_jump_prob = 0.0
                else:
                    local_rate_noise, local_pos_noise, step_jump_prob, step_tail_jump_prob = _pf_transition_local_params_numba(
                        dmd,
                        snap_gr_power[a + 1],
                        snap_gr_finite[a + 1],
                        rate_noise,
                        pos_noise,
                        jump_prob,
                        jump_sd,
                        jump_rate_sd,
                        missing_jump_boost,
                        jump_tail_prob,
                        jump_tail_sd,
                        jump_tail_rate_sd,
                        jump_tail_missing_boost,
                        missing_noise_scale,
                        missing_jump_scale,
                    )
                    pos_sd = 1.0
                    rate_sd = 1.0
                for j in range(n_particles):
                    wj = snap_weight[a, j]
                    if wj <= 0.0 or not np.isfinite(wj):
                        log_score[j] = -1e300
                        continue
                    pred_rate = momentum * snap_rate[a, j]
                    if rate_mean_weight > 0.0:
                        pred_rate += rate_mean_weight * (init_rate - pred_rate)
                    pred_pos = snap_pos[a, j] + pred_rate * dmd
                    rate_delta = next_rate - pred_rate
                    pos_delta = next_pos - pred_pos
                    if transition_model_code == 0:
                        rate_d = rate_delta / rate_sd
                        pos_d = pos_delta / pos_sd
                        rate_log_lk = -0.5 * rate_d * rate_d
                        if rate_log_lk < -300.0:
                            rate_log_lk = -300.0
                        pos_log_lk = -0.5 * pos_d * pos_d
                        if pos_log_lk < -300.0:
                            pos_log_lk = -300.0
                        transition_log_lk = rate_log_lk + pos_log_lk
                    elif transition_model_code == 5:
                        transition_log_lk = _pf_transition_collapsed_joint_log_lk_numba(
                            pos_delta,
                            rate_delta,
                            local_pos_noise,
                            local_rate_noise,
                            dmd,
                            step_jump_prob,
                            jump_sd,
                            jump_rate_sd,
                            step_tail_jump_prob,
                            jump_tail_sd,
                            jump_tail_rate_sd,
                            transition_scale,
                            pos_floor,
                            rate_floor,
                        )
                    else:
                        use_joint = 0
                        include_norm = 0
                        if transition_model_code == 2 or transition_model_code == 4:
                            include_norm = 1
                        if transition_model_code == 3 or transition_model_code == 4:
                            use_joint = 1
                        transition_log_lk = _pf_transition_mixture_log_lk_numba(
                            pos_delta,
                            rate_delta,
                            local_pos_noise,
                            local_rate_noise,
                            dmd,
                            step_jump_prob,
                            jump_sd,
                            jump_rate_sd,
                            step_tail_jump_prob,
                            jump_tail_sd,
                            jump_tail_rate_sd,
                            transition_scale,
                            pos_floor,
                            rate_floor,
                            use_joint,
                            include_norm,
                        )
                    log_score[j] = math.log(max(wj, 1e-300)) + transition_log_lk
                idx = _pf_sample_log_score_index_numba(log_score, n_particles)
                next_idx = idx
                next_pos = snap_pos[a, idx]
                next_rate = snap_rate[a, idx]
                b = active_bins[a]
                tvt = next_pos - state_z_weight * snap_z[a]
                if obs_likelihood_mode == 1:
                    tvt = tvt + next_rate * snap_output_md_delta[a] + state_z_weight * (snap_z[a] - snap_output_z[a])
                if tvt < tvt_min:
                    tvt = tvt_min
                elif tvt > tvt_max:
                    tvt = tvt_max
                rel = tvt - tvt0
                grid_pos = (rel - axis0) / axis_step
                left = math.floor(grid_pos)
                frac = grid_pos - left
                if left >= 0 and left < typewell_len:
                    hist_bins[b, left] += 1.0 - frac
                right = left + 1
                if right >= 0 and right < typewell_len:
                    hist_bins[b, right] += frac
                path_sums[b] += tvt
                count_bins[b] += 1.0
        return (
            filtered_hist,
            filtered_path_sums,
            filtered_count_bins,
            hist_bins,
            path_sums,
            count_bins,
            log_lik,
            local_log_lik,
            local_obs_power,
            1,
        )


PF_OPT_PROGRESS_PATH = Path(__file__).resolve().parents[1] / "PFOptProgress.md"


PF_FAILURE_RELIABILITY_FEATURES = (
    "entropy_mean",
    "ess_mean",
    "maxp_mean",
    "anchor_mass5_mean",
    "anchor_mass15_mean",
    "pred_rel_abs_mean",
    "mode_anchor_abs_mean",
    "mean_anchor_abs_mean",
    "pred_end_abs",
    "pred_step_rms",
    "pf_gr_rmse",
    "pf_gr_minus_anchor_gr_rmse",
)

PF_FAILURE_HIGH_RISK_FEATURES = {
    "entropy_mean",
    "ess_mean",
    "pred_rel_abs_mean",
    "mode_anchor_abs_mean",
    "mean_anchor_abs_mean",
    "pred_end_abs",
    "pred_step_rms",
    "pf_gr_rmse",
    "pf_gr_minus_anchor_gr_rmse",
}


PF_OPT_FIXED_CANDIDATES = (
    ("baseline", {}),
    ("missing_skip", {"PF_heatmap_query_gr_mode": "skip"}),
    ("missing_soft_p025", {"PF_heatmap_query_gr_mode": "soft_interp", "PF_heatmap_missing_likelihood_power": 0.25}),
    ("missing_soft_p050", {"PF_heatmap_query_gr_mode": "soft_interp", "PF_heatmap_missing_likelihood_power": 0.50}),
    ("path_equal", {"PF_heatmap_seed_path_weight_mode": "equal"}),
    ("prob_likelihood", {"PF_heatmap_seed_prob_weight_mode": "likelihood"}),
    ("path_equal_prob_likelihood", {"PF_heatmap_seed_path_weight_mode": "equal", "PF_heatmap_seed_prob_weight_mode": "likelihood"}),
    ("lik_scale_10", {"PF_heatmap_lik_scale": 10.0}),
    ("lik_scale_20", {"PF_heatmap_lik_scale": 20.0}),
    ("lik_scale_2p5", {"PF_heatmap_lik_scale": 2.5}),
    ("rate_noise_001", {"PF_heatmap_rate_noise": 0.001}),
    ("rate_noise_004", {"PF_heatmap_rate_noise": 0.004}),
    ("pos_noise_002", {"PF_heatmap_pos_noise": 0.002}),
    ("pos_noise_010", {"PF_heatmap_pos_noise": 0.010}),
    ("momentum_995", {"PF_heatmap_momentum": 0.995}),
    ("momentum_999", {"PF_heatmap_momentum": 0.999}),
    ("resamp_030", {"PF_heatmap_resample_threshold": 0.30}),
    ("resamp_070", {"PF_heatmap_resample_threshold": 0.70}),
    ("rough_0050_0005", {"PF_heatmap_rough_pos": 0.05, "PF_heatmap_rough_rate": 0.0005}),
    ("rough_0200_0020", {"PF_heatmap_rough_pos": 0.20, "PF_heatmap_rough_rate": 0.0020}),
    ("init_pos_sd_1", {"PF_heatmap_init_pos_sd": 1.0}),
    ("init_pos_sd_5", {"PF_heatmap_init_pos_sd": 5.0}),
    ("rate_mean_0005", {"PF_heatmap_rate_mean_weight": 0.0005}),
    ("rate_mean_0010", {"PF_heatmap_rate_mean_weight": 0.0010}),
    ("anchor_s50_p002", {"PF_heatmap_anchor_sigma": 50.0, "PF_heatmap_anchor_power": 0.002}),
    ("anchor_s50_p005", {"PF_heatmap_anchor_sigma": 50.0, "PF_heatmap_anchor_power": 0.005}),
    ("anchor_s35_p002", {"PF_heatmap_anchor_sigma": 35.0, "PF_heatmap_anchor_power": 0.002}),
    ("jump_0002_s5", {"PF_heatmap_jump_prob": 0.0002, "PF_heatmap_jump_sd": 5.0, "PF_heatmap_jump_rate_sd": 0.001}),
    ("jump_0005_s8", {"PF_heatmap_jump_prob": 0.0005, "PF_heatmap_jump_sd": 8.0, "PF_heatmap_jump_rate_sd": 0.002}),
    ("missing_skip_jump", {"PF_heatmap_query_gr_mode": "skip", "PF_heatmap_jump_prob": 0.0003, "PF_heatmap_jump_sd": 6.0, "PF_heatmap_jump_rate_sd": 0.0015}),
)


PF_OPT_CURRENT_BEST_NAME = "rs0003_rand07"
PF_OPT_CURRENT_BEST_FULL_NAME = "rs0003_rand07_full_seed20262640"
PF_OPT_CURRENT_BEST_FULL_RMSE = 8.540895
PF_OPT_CURRENT_BEST_FULL_BASE_SEED = 20262640
PF_OPT_FULL_CHECK_DEFAULT_PARTICLES = 500
PF_OPT_FULL_CHECK_DEFAULT_SEEDS = 128
PF_OPT_FULL_CHECK_DEFAULT_BASE_SEED = 20260622
PF_OPT_BUDGET_META_N_PARTICLES = "__opt_n_particles"
PF_OPT_BUDGET_META_N_SEEDS = "__opt_n_seeds"
PF_OPT_MERGE64_ALLOWED_BUDGETS = ((500, 128), (1000, 64))
PF_OPT_CURRENT_BEST_OVERRIDES = {
    "PF_heatmap_apply_gr_calibration": True,
    "PF_heatmap_anchor_power": 0.002,
    "PF_heatmap_anchor_sigma": 80.0,
    "PF_heatmap_correlated_pos_alpha": 0.2,
    "PF_heatmap_finite_run_power_boost": 0.06,
    "PF_heatmap_finite_run_power_decay": 1.0,
    "PF_heatmap_finite_run_power_floor": 0.70,
    "PF_heatmap_init_pos_sd": 2.0,
    "PF_heatmap_init_rate_sd": 0.02,
    "PF_heatmap_jump_prob": 0.00015,
    "PF_heatmap_jump_rate_sd": 0.0004,
    "PF_heatmap_jump_sd": 7.0,
    "PF_heatmap_lik_scale": 5.5,
    "PF_heatmap_missing_jump_boost": 0.00015,
    "PF_heatmap_momentum": 0.99945,
    "PF_heatmap_pos_noise": 0.0035,
    "PF_heatmap_query_gr_mode": "interp",
    "PF_heatmap_raw_ref_likelihood_weight": 0.075,
    "PF_heatmap_rate_mean_weight": 0.00100,
    "PF_heatmap_rate_noise": 0.0009,
    "PF_heatmap_ref_grid_step": 0.05,
    "PF_heatmap_resample_obs_power_adapt": 0.60,
    "PF_heatmap_resample_min_threshold": 0.35,
    "PF_heatmap_resample_threshold": 0.35,
    "PF_heatmap_rough_pos": 0.08,
    "PF_heatmap_rough_rate": 0.006,
    "PF_heatmap_seed_path_weight_mode": "likelihood",
    "PF_heatmap_seed_prob_weight_mode": "likelihood",
    "PF_heatmap_seen_blend_weight": 0.75,
    "PF_heatmap_stratified_init": 1,
    "PF_heatmap_typewell_gr_calibration": "seen_blend",
    "PF_heatmap_profile_mixture_spec": [
        {
            "name": "base",
            "weight": 0.6514002115702939,
            "overrides": {},
        },
        {
            "name": "wide_init",
            "weight": 0.19545809861466074,
            "overrides": {
                "pf_init_pos_sd": 2.5,
                "pf_init_rate_sd": 0.025,
                "pf_resample_threshold": 0.35,
                "pf_rough_pos": 0.10,
                "pf_rough_rate": 0.006,
            },
        },
        {
            "name": "smooth_stiff",
            "weight": 0.053120440286232196,
            "overrides": {
                "pf_jump_prob": 0.000075,
                "pf_jump_rate_sd": 0.00030,
                "pf_jump_sd": 9.0,
                "pf_momentum": 0.9995,
                "pf_pos_noise": 0.0030,
                "pf_rate_mean_weight": 0.00150,
                "pf_rate_noise": 0.0008,
                "pf_rough_pos": 0.06,
                "pf_rough_rate": 0.0045,
            },
        },
        {
            "name": "jump_rare_big",
            "weight": 0.07370294173329474,
            "overrides": {
                "pf_jump_prob": 0.000075,
                "pf_jump_rate_sd": 0.00060,
                "pf_jump_sd": 10.0,
                "pf_missing_jump_boost": 0.000075,
                "pf_resample_threshold": 0.35,
            },
        },
        {
            "name": "tight_init",
            "weight": 0.02631830779551842,
            "overrides": {
                "pf_init_pos_sd": 1.0,
                "pf_init_rate_sd": 0.015,
                "pf_pos_noise": 0.0035,
                "pf_rough_pos": 0.06,
            },
        },
    ],
}


PF_OPT_MERGE16_CURRENT_BEST_NAME = "m16rs0002_m16_alpha110_resamp035"
PF_OPT_MERGE16_CURRENT_BEST_FULL_NAME = "m16rs0002_check001_m16rs0002_m16_alpha110_resamp035_full_seed20262640"
PF_OPT_MERGE16_CURRENT_BEST_FULL_RMSE = 8.029434
PF_OPT_MERGE16_CURRENT_BEST_FULL_BASE_SEED = 20262640
PF_OPT_MERGE16_CURRENT_BEST_OVERRIDES = {
    **PF_OPT_CURRENT_BEST_OVERRIDES,
    "PF_heatmap_base_seed": PF_OPT_MERGE16_CURRENT_BEST_FULL_BASE_SEED,
    "PF_heatmap_resample_obs_power_adapt": 0.45,
    "PF_heatmap_resample_min_threshold": 0.30,
    "PF_heatmap_suffix_merge_adjust_dynamics": False,
    "PF_heatmap_suffix_merge_likelihood_mode": "block",
    "PF_heatmap_suffix_merge_k": 16,
    "PF_heatmap_suffix_merge_obs_power_alpha": 1.1,
    "PF_heatmap_suffix_merge_state_mode": "start",
}

PF_OPT_MERGE64_CURRENT_BEST_NAME = "m64rs0011_base_alpha105"
PF_OPT_MERGE64_CURRENT_BEST_FULL_NAME = "m64rs0011_check004_m64rs0011_base_alpha105_full_seed20265667"
PF_OPT_MERGE64_CURRENT_BEST_FULL_RMSE = 7.949323
PF_OPT_MERGE64_CURRENT_BEST_FULL_FFBSI_RMSE = 7.486534
PF_OPT_MERGE64_CURRENT_BEST_FULL_BASE_SEED = 20265667
PF_OPT_MERGE64_CURRENT_BEST_OVERRIDES = {
    **PF_OPT_MERGE16_CURRENT_BEST_OVERRIDES,
    "PF_heatmap_base_seed": PF_OPT_MERGE64_CURRENT_BEST_FULL_BASE_SEED,
    "PF_heatmap_suffix_merge_k": 64,
    "PF_heatmap_suffix_merge_obs_power_alpha": 1.05,
    "PF_heatmap_lik_scale": 6.0,
    "PF_heatmap_finite_run_power_floor": 0.60,
    "PF_heatmap_momentum": 0.9995,
    "PF_heatmap_pos_noise": 0.0025,
    "PF_heatmap_resample_threshold": 0.30,
    "PF_heatmap_jump_prob": 0.000075,
    "PF_heatmap_jump_rate_sd": 0.00030,
    "PF_heatmap_jump_sd": 7.0,
    "PF_heatmap_missing_jump_boost": 0.00020,
    "PF_heatmap_ffbsi_mode": "fallback",
    "PF_heatmap_ffbsi_n_paths": 64,
    "PF_heatmap_ffbsi_fallback_mode": "filtered",
}


def _pf_opt_with_candidate_budget(overrides, n_particles, n_seeds):
    out = dict(overrides or {})
    out[PF_OPT_BUDGET_META_N_PARTICLES] = int(n_particles)
    out[PF_OPT_BUDGET_META_N_SEEDS] = int(n_seeds)
    return out


def _pf_opt_runtime_overrides(overrides):
    out = dict(overrides or {})
    out.pop(PF_OPT_BUDGET_META_N_PARTICLES, None)
    out.pop(PF_OPT_BUDGET_META_N_SEEDS, None)
    return out


def _pf_opt_candidate_budget(overrides, default_n_particles=500, default_n_seeds=128):
    overrides = dict(overrides or {})
    return (
        int(overrides.get(PF_OPT_BUDGET_META_N_PARTICLES, default_n_particles)),
        int(overrides.get(PF_OPT_BUDGET_META_N_SEEDS, default_n_seeds)),
    )


def _pf_opt_budget_label(n_particles, n_seeds):
    return f"n{int(n_particles)}k{int(n_seeds)}"


def _pf_opt_validate_merge64_budget(n_particles, n_seeds):
    budget = (int(n_particles), int(n_seeds))
    if budget not in set(PF_OPT_MERGE64_ALLOWED_BUDGETS):
        allowed = ", ".join(f"{n}x{k}" for n, k in PF_OPT_MERGE64_ALLOWED_BUDGETS)
        raise ValueError(f"merge64 search budget `{budget[0]}x{budget[1]}` is unsupported; allowed budgets are {allowed}")
    return budget


def _pf_opt_apply_n1000_budget_center(overrides, changed=()):
    changed = set(changed or ())
    out = _pf_opt_with_candidate_budget(overrides, 1000, 64)
    center = {
        "PF_heatmap_resample_threshold": 0.25,
        "PF_heatmap_resample_obs_power_adapt": 0.30,
        "PF_heatmap_resample_min_threshold": 0.125,
        "PF_heatmap_rough_pos": 0.040,
        "PF_heatmap_rough_rate": 0.0030,
        "PF_heatmap_momentum": 0.99955,
        "PF_heatmap_rate_noise": 0.00070,
        "PF_heatmap_pos_noise": 0.0025,
        "PF_heatmap_jump_prob": 0.000075,
        "PF_heatmap_missing_jump_boost": 0.00020,
        "PF_heatmap_jump_sd": 7.0,
        "PF_heatmap_jump_rate_sd": 0.00030,
    }
    for key, value in center.items():
        if key not in changed:
            out[key] = value
    if (
        "PF_heatmap_resample_obs_power_adapt" in out
        and "PF_heatmap_resample_min_threshold" not in changed
        and "PF_heatmap_resample_min_threshold" not in out
    ):
        out["PF_heatmap_resample_min_threshold"] = 0.125
    return out

PF_OPT_MERGE16_PROFILE_MIXTURE_NAMES = (
    "profile_rs0026_pm0002",
    "profile_pm0002_gfrdecay050_raw_all",
    "profile_pm0002_gfrdecay050_amb_all",
    "profile_pm0002_gout_dyn_info_all",
    "profile_pm0002_8x_dyn_micro",
    "profile_pm0002_8x_gobs_micro",
)


PF_OPT_PROFILE_MIX_COMMON_OVERRIDES = {
    # The requested design compares forward-filtered/direct and FFBSi paths in
    # one run, so every seed profile must keep fallback FFBSi enabled.
    "PF_heatmap_ffbsi_mode": "fallback",
    "PF_heatmap_ffbsi_n_paths": 64,
    "PF_heatmap_ffbsi_fallback_mode": "filtered",
    "PF_heatmap_ffbsi_max_active_bins": 512,
    "PF_heatmap_ffbsi_transition_scale": 1.35,
    "PF_heatmap_ffbsi_pos_floor": 0.15,
    "PF_heatmap_ffbsi_rate_floor": 0.0025,
}


def _pf_profile(name, weight, **overrides):
    return {
        "name": name,
        "weight": float(weight),
        "overrides": dict(overrides),
    }


PF_OPT_GLOBAL_PROFILE_COMPONENT_KEYS = (
    "outlier_prob",
    "outlier_likelihood",
    "dynamic_sigma_alpha",
    "dynamic_sigma_threshold",
    "dynamic_sigma_power",
    "dynamic_sigma_min",
    "dynamic_sigma_max",
    "gr_information_power",
    "gr_information_center",
    "gr_information_min_multiplier",
    "gr_information_max_multiplier",
    "gr_information_slope_weight",
    "gr_information_ref_mode",
    "pf_lookahead_power",
    "pf_lookahead_steps",
    "pf_lookahead_decay",
    "pf_lookahead_max_gap",
    "pf_lookahead_delta_power",
    "gr_ambiguity_power",
    "gr_ambiguity_min_power",
    "gr_ambiguity_contrast",
    "gr_ambiguity_ref_mode",
    "raw_ref_likelihood_weight",
    "seen_blend_weight",
    "pf_state_z_weight",
    "pf_finite_run_power_decay",
    "pf_finite_run_power_floor",
    "pf_jump_tail_prob",
    "pf_jump_tail_sd",
    "pf_jump_tail_rate_sd",
    "pf_jump_tail_dist",
    "pf_jump_tail_clip",
    "pf_jump_tail_missing_boost",
)


PF_OPT_GLOBAL_NEW_DIM_KEYS = (
    "outlier_prob",
    "outlier_likelihood",
    "dynamic_sigma_alpha",
    "dynamic_sigma_threshold",
    "dynamic_sigma_power",
    "dynamic_sigma_min",
    "dynamic_sigma_max",
    "gr_information_power",
    "gr_information_center",
    "gr_information_min_multiplier",
    "gr_information_max_multiplier",
    "gr_information_slope_weight",
    "gr_information_ref_mode",
    "pf_lookahead_power",
    "pf_lookahead_steps",
    "pf_lookahead_decay",
    "pf_lookahead_max_gap",
    "pf_lookahead_delta_power",
    "pf_finite_run_power_decay",
    "pf_finite_run_power_floor",
    "pf_jump_tail_prob",
    "pf_jump_tail_sd",
    "pf_jump_tail_rate_sd",
    "pf_jump_tail_dist",
    "pf_jump_tail_clip",
    "pf_jump_tail_missing_boost",
)


def _normalize_profile_override_dict(overrides):
    cleaned = {}
    for key, value in dict(overrides or {}).items():
        cache_key = _profile_cache_key(key)
        if cache_key not in PF_PROFILE_ALLOWED_CACHE_KEYS:
            raise ValueError(f"PF profile override key {key!r} is not supported")
        cleaned[cache_key] = _json_clean_profile_value(value)
    return dict(sorted(cleaned.items()))


def _apply_global_profile_component(profiles, global_overrides):
    global_overrides = _normalize_profile_override_dict(global_overrides)
    if not global_overrides:
        return _normalize_profile_mixture_spec(profiles)
    out = []
    for profile in _normalize_profile_mixture_spec(profiles):
        merged = dict(profile["overrides"])
        merged.update(global_overrides)
        out.append(
            {
                "name": profile["name"],
                "weight": float(profile["weight"]),
                "overrides": dict(sorted(merged.items())),
            }
        )
    return out


def _pf_top_level_cache_overrides_from_pf_heatmap(overrides):
    out = {}
    for key, value in _pf_opt_runtime_overrides(overrides).items():
        if key == "PF_heatmap_profile_mixture_spec":
            continue
        cache_key = _profile_cache_key(key)
        if cache_key in PF_PROFILE_ALLOWED_CACHE_KEYS:
            out[cache_key] = _json_clean_profile_value(value)
    return dict(sorted(out.items()))


def _fresh_seq_nn_cfg(overrides=None):
    try:
        from seq_NN_cfg import CFG
    except Exception:
        import importlib.util

        cfg_path = Path(__file__).resolve().with_name("seq_NN_cfg.py")
        spec = importlib.util.spec_from_file_location("_pf_opt_seq_NN_cfg", cfg_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load seq_NN_cfg.CFG for opt profile-comb experiment")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        CFG = module.CFG
    cfg = CFG()
    for key, value in _pf_opt_runtime_overrides(overrides).items():
        setattr(cfg, key, value)
    if hasattr(cfg, "refresh"):
        cfg.refresh()
    return cfg


def _pf_profile_cache_overrides_from_cfg(cfg):
    cache_cfg = pf_heatmap_cache_config(cfg)
    return {
        key: _json_clean_profile_value(cache_cfg[key])
        for key in sorted(PF_PROFILE_ALLOWED_CACHE_KEYS)
        if key in cache_cfg
    }


def _cfg_default_pf_profile_mixture_spec():
    cfg = _fresh_seq_nn_cfg()
    return _normalize_profile_mixture_spec(getattr(cfg, "PF_heatmap_profile_mixture_spec", None))


def _assert_profile_comb_cache_coverage(cfg_cache, sota_cache):
    covered_or_global = {
        "version",
        "n_particles",
        "n_seeds",
        "base_seed",
        "profile_mixture_mode",
        "profile_mixture_spec",
        "apply_gr_calibration",
        "typewell_window",
        "typewell_len",
        "downsample",
        "prefix_len",
        "target_len",
        "raw_len",
        "num_bins",
    }
    for key in sorted(set(cfg_cache) | set(sota_cache)):
        if key in PF_PROFILE_ALLOWED_CACHE_KEYS or key in covered_or_global:
            continue
        if cfg_cache.get(key) != sota_cache.get(key):
            raise RuntimeError(
                "profile-comb cannot cover differing non-profile PF cache key "
                f"{key!r}: cfg={cfg_cache.get(key)!r} sota={sota_cache.get(key)!r}"
            )


def _profiles_with_prefix_and_global_overrides(profiles, prefix, global_overrides, forced_overrides=None):
    global_overrides = _pf_top_level_cache_overrides_from_pf_heatmap(global_overrides)
    forced_overrides = _pf_top_level_cache_overrides_from_pf_heatmap(forced_overrides)
    out = []
    for profile in _normalize_profile_mixture_spec(profiles):
        overrides = dict(global_overrides)
        overrides.update(profile["overrides"])
        overrides.update(forced_overrides)
        out.append(
            {
                "name": f"{prefix}_{profile['name']}",
                "weight": float(profile["weight"]),
                "overrides": dict(sorted(overrides.items())),
            }
        )
    return out


def _profile_mixture_half_cfg_default_half_sota():
    cfg = _fresh_seq_nn_cfg()
    sota_cfg = _fresh_seq_nn_cfg(PF_OPT_CURRENT_BEST_OVERRIDES)
    cfg_cache = pf_heatmap_cache_config(cfg)
    sota_cache = pf_heatmap_cache_config(sota_cfg)
    _assert_profile_comb_cache_coverage(cfg_cache, sota_cache)
    cfg_overrides = _pf_profile_cache_overrides_from_cfg(cfg)
    sota_overrides = _pf_profile_cache_overrides_from_cfg(sota_cfg)
    cfg_profiles = _profiles_with_prefix_and_global_overrides(
        getattr(cfg, "PF_heatmap_profile_mixture_spec", None),
        "cfg",
        cfg_overrides,
        forced_overrides=PF_OPT_PROFILE_MIX_COMMON_OVERRIDES,
    )
    if not cfg_profiles:
        raise RuntimeError("could not load CFG default PF profile mixture for opt profile-comb experiment")
    sota_profiles = _profiles_with_prefix_and_global_overrides(
        getattr(sota_cfg, "PF_heatmap_profile_mixture_spec", None),
        "sota",
        sota_overrides,
        forced_overrides=PF_OPT_PROFILE_MIX_COMMON_OVERRIDES,
    )
    profiles = []
    for profile in cfg_profiles:
        item = dict(profile)
        item["weight"] = 0.5 * float(item["weight"])
        profiles.append(item)
    for profile in sota_profiles:
        item = dict(profile)
        item["weight"] = 0.5 * float(item["weight"])
        profiles.append(item)
    return _normalize_profile_mixture_spec(profiles)


PF_OPT_PROFILE_COMB_OVERRIDES = {
    # Each profile carries a full PF scalar cache config; top-level settings are
    # only the shared experiment controls and equal global-only cache fields.
    "PF_heatmap_profile_mixture_spec": _profile_mixture_half_cfg_default_half_sota(),
    **PF_OPT_PROFILE_MIX_COMMON_OVERRIDES,
}


def _pm0002_accepted_profiles():
    return [
        _pf_profile("base", 0.6514002115702939),
        _pf_profile(
            "wide_init",
            0.19545809861466074,
            PF_heatmap_init_pos_sd=2.5,
            PF_heatmap_init_rate_sd=0.025,
            PF_heatmap_resample_threshold=0.35,
            PF_heatmap_rough_pos=0.10,
            PF_heatmap_rough_rate=0.006,
        ),
        _pf_profile(
            "smooth_stiff",
            0.053120440286232196,
            PF_heatmap_jump_prob=0.000075,
            PF_heatmap_jump_rate_sd=0.00030,
            PF_heatmap_jump_sd=9.0,
            PF_heatmap_momentum=0.9995,
            PF_heatmap_pos_noise=0.0030,
            PF_heatmap_rate_mean_weight=0.00150,
            PF_heatmap_rate_noise=0.0008,
            PF_heatmap_rough_pos=0.06,
            PF_heatmap_rough_rate=0.0045,
        ),
        _pf_profile(
            "jump_rare_big",
            0.07370294173329474,
            PF_heatmap_jump_prob=0.000075,
            PF_heatmap_jump_rate_sd=0.00060,
            PF_heatmap_jump_sd=10.0,
            PF_heatmap_missing_jump_boost=0.000075,
            PF_heatmap_resample_threshold=0.35,
        ),
        _pf_profile(
            "tight_init",
            0.02631830779551842,
            PF_heatmap_init_pos_sd=1.0,
            PF_heatmap_init_rate_sd=0.015,
            PF_heatmap_pos_noise=0.0035,
            PF_heatmap_rough_pos=0.06,
        ),
    ]


def _global_obs_component(label):
    components = {
        "frdecay025": {
            "PF_heatmap_finite_run_power_decay": 0.25,
            "PF_heatmap_finite_run_power_floor": 0.60,
        },
        "frdecay050": {
            "PF_heatmap_finite_run_power_decay": 0.50,
            "PF_heatmap_finite_run_power_floor": 0.55,
        },
        "frdecay100": {
            "PF_heatmap_finite_run_power_decay": 1.00,
            "PF_heatmap_finite_run_power_floor": 0.50,
        },
        "out005": {
            "PF_heatmap_outlier_prob": 0.005,
            "PF_heatmap_outlier_likelihood": 0.050,
        },
        "out010": {
            "PF_heatmap_outlier_prob": 0.010,
            "PF_heatmap_outlier_likelihood": 0.050,
        },
        "dyn010": {
            "PF_heatmap_dynamic_sigma_alpha": 0.010,
            "PF_heatmap_dynamic_sigma_threshold": 1.35,
            "PF_heatmap_dynamic_sigma_power": 0.60,
            "PF_heatmap_dynamic_sigma_min": 0.90,
            "PF_heatmap_dynamic_sigma_max": 1.75,
        },
        "dyn020": {
            "PF_heatmap_dynamic_sigma_alpha": 0.020,
            "PF_heatmap_dynamic_sigma_threshold": 1.35,
            "PF_heatmap_dynamic_sigma_power": 0.75,
            "PF_heatmap_dynamic_sigma_min": 0.85,
            "PF_heatmap_dynamic_sigma_max": 2.00,
        },
        "info025": {
            "PF_heatmap_gr_information_power": 0.25,
            "PF_heatmap_gr_information_center": 0.70,
            "PF_heatmap_gr_information_min_multiplier": 0.75,
            "PF_heatmap_gr_information_max_multiplier": 1.20,
        },
        "info035": {
            "PF_heatmap_gr_information_power": 0.35,
            "PF_heatmap_gr_information_center": 0.70,
            "PF_heatmap_gr_information_min_multiplier": 0.70,
            "PF_heatmap_gr_information_max_multiplier": 1.25,
            "PF_heatmap_gr_information_slope_weight": 0.25,
        },
        "look0005": {
            "PF_heatmap_lookahead_power": 0.0005,
            "PF_heatmap_lookahead_steps": 2,
            "PF_heatmap_lookahead_decay": 0.35,
            "PF_heatmap_lookahead_max_gap": 64.0,
        },
        "raw020": {
            "PF_heatmap_raw_ref_likelihood_weight": 0.20,
            "PF_heatmap_seen_blend_weight": 0.65,
        },
        "raw025": {
            "PF_heatmap_raw_ref_likelihood_weight": 0.25,
            "PF_heatmap_seen_blend_weight": 0.60,
        },
        "amb035": {
            "PF_heatmap_gr_ambiguity_power": 0.35,
            "PF_heatmap_gr_ambiguity_min_power": 0.45,
            "PF_heatmap_gr_ambiguity_contrast": 1.00,
        },
        "z095": {
            "PF_heatmap_state_z_weight": 0.95,
        },
        "jmix_data_lap": {
            "PF_heatmap_jump_prob": 0.00018,
            "PF_heatmap_jump_sd": 2.5,
            "PF_heatmap_jump_rate_sd": 0.00035,
            "PF_heatmap_missing_jump_boost": 0.00003,
            "PF_heatmap_jump_tail_prob": 0.000025,
            "PF_heatmap_jump_tail_sd": 12.0,
            "PF_heatmap_jump_tail_rate_sd": 0.0010,
            "PF_heatmap_jump_tail_dist": 1,
            "PF_heatmap_jump_tail_clip": 35.0,
            "PF_heatmap_jump_tail_missing_boost": 0.000015,
        },
        "jmix_small_freq": {
            "PF_heatmap_jump_prob": 0.00050,
            "PF_heatmap_jump_sd": 1.8,
            "PF_heatmap_jump_rate_sd": 0.00025,
            "PF_heatmap_missing_jump_boost": 0.00005,
            "PF_heatmap_jump_tail_prob": 0.000020,
            "PF_heatmap_jump_tail_sd": 10.0,
            "PF_heatmap_jump_tail_rate_sd": 0.0008,
            "PF_heatmap_jump_tail_dist": 1,
            "PF_heatmap_jump_tail_clip": 28.0,
            "PF_heatmap_jump_tail_missing_boost": 0.000010,
        },
        "jmix_tail_mid": {
            "PF_heatmap_jump_prob": 0.00012,
            "PF_heatmap_jump_sd": 3.0,
            "PF_heatmap_jump_rate_sd": 0.00030,
            "PF_heatmap_missing_jump_boost": 0.00002,
            "PF_heatmap_jump_tail_prob": 0.000060,
            "PF_heatmap_jump_tail_sd": 18.0,
            "PF_heatmap_jump_tail_rate_sd": 0.0015,
            "PF_heatmap_jump_tail_dist": 1,
            "PF_heatmap_jump_tail_clip": 55.0,
            "PF_heatmap_jump_tail_missing_boost": 0.000030,
        },
        "jmix_tail_rare_big": {
            "PF_heatmap_jump_prob": 0.00006,
            "PF_heatmap_jump_sd": 4.0,
            "PF_heatmap_jump_rate_sd": 0.00040,
            "PF_heatmap_missing_jump_boost": 0.00001,
            "PF_heatmap_jump_tail_prob": 0.000010,
            "PF_heatmap_jump_tail_sd": 30.0,
            "PF_heatmap_jump_tail_rate_sd": 0.0025,
            "PF_heatmap_jump_tail_dist": 1,
            "PF_heatmap_jump_tail_clip": 90.0,
            "PF_heatmap_jump_tail_missing_boost": 0.000020,
        },
        "jmix_tail_very_rare_extreme": {
            "PF_heatmap_jump_prob": 0.00004,
            "PF_heatmap_jump_sd": 5.0,
            "PF_heatmap_jump_rate_sd": 0.00050,
            "PF_heatmap_missing_jump_boost": 0.00000,
            "PF_heatmap_jump_tail_prob": 0.000004,
            "PF_heatmap_jump_tail_sd": 55.0,
            "PF_heatmap_jump_tail_rate_sd": 0.0040,
            "PF_heatmap_jump_tail_dist": 1,
            "PF_heatmap_jump_tail_clip": 140.0,
            "PF_heatmap_jump_tail_missing_boost": 0.000010,
        },
        "jmix_cauchy_mid": {
            "PF_heatmap_jump_prob": 0.00010,
            "PF_heatmap_jump_sd": 2.5,
            "PF_heatmap_jump_rate_sd": 0.00030,
            "PF_heatmap_missing_jump_boost": 0.00002,
            "PF_heatmap_jump_tail_prob": 0.000020,
            "PF_heatmap_jump_tail_sd": 18.0,
            "PF_heatmap_jump_tail_rate_sd": 0.0015,
            "PF_heatmap_jump_tail_dist": 2,
            "PF_heatmap_jump_tail_clip": 60.0,
            "PF_heatmap_jump_tail_missing_boost": 0.000015,
        },
        "jmix_aggressive": {
            "PF_heatmap_jump_prob": 0.00075,
            "PF_heatmap_jump_sd": 2.5,
            "PF_heatmap_jump_rate_sd": 0.00060,
            "PF_heatmap_missing_jump_boost": 0.00008,
            "PF_heatmap_jump_tail_prob": 0.000100,
            "PF_heatmap_jump_tail_sd": 20.0,
            "PF_heatmap_jump_tail_rate_sd": 0.0030,
            "PF_heatmap_jump_tail_dist": 1,
            "PF_heatmap_jump_tail_clip": 80.0,
            "PF_heatmap_jump_tail_missing_boost": 0.000050,
        },
        "jmix_rate_tail": {
            "PF_heatmap_jump_prob": 0.00020,
            "PF_heatmap_jump_sd": 2.0,
            "PF_heatmap_jump_rate_sd": 0.0010,
            "PF_heatmap_missing_jump_boost": 0.00003,
            "PF_heatmap_jump_tail_prob": 0.000040,
            "PF_heatmap_jump_tail_sd": 8.0,
            "PF_heatmap_jump_tail_rate_sd": 0.0045,
            "PF_heatmap_jump_tail_dist": 1,
            "PF_heatmap_jump_tail_clip": 35.0,
            "PF_heatmap_jump_tail_missing_boost": 0.000020,
        },
        "jmix_pos_tail": {
            "PF_heatmap_jump_prob": 0.00018,
            "PF_heatmap_jump_sd": 3.0,
            "PF_heatmap_jump_rate_sd": 0.00020,
            "PF_heatmap_missing_jump_boost": 0.00002,
            "PF_heatmap_jump_tail_prob": 0.000050,
            "PF_heatmap_jump_tail_sd": 24.0,
            "PF_heatmap_jump_tail_rate_sd": 0.0004,
            "PF_heatmap_jump_tail_dist": 1,
            "PF_heatmap_jump_tail_clip": 70.0,
            "PF_heatmap_jump_tail_missing_boost": 0.000020,
        },
    }
    return dict(components[str(label)])


PF_OPT_PROFILE_MIXTURES = {
    "profile_dyn_6x_balanced": [
        _pf_profile("base", 0.34),
        _pf_profile(
            "smooth_stiff",
            0.14,
            PF_heatmap_momentum=0.9995,
            PF_heatmap_rate_noise=0.0008,
            PF_heatmap_pos_noise=0.0030,
            PF_heatmap_rough_pos=0.06,
            PF_heatmap_rough_rate=0.0045,
            PF_heatmap_jump_prob=0.000075,
            PF_heatmap_jump_sd=9.0,
            PF_heatmap_jump_rate_sd=0.00030,
            PF_heatmap_rate_mean_weight=0.00150,
        ),
        _pf_profile(
            "rough_rate",
            0.16,
            PF_heatmap_momentum=0.99935,
            PF_heatmap_rate_noise=0.0013,
            PF_heatmap_pos_noise=0.0045,
            PF_heatmap_rough_pos=0.10,
            PF_heatmap_rough_rate=0.0080,
            PF_heatmap_init_rate_sd=0.025,
            PF_heatmap_rate_mean_weight=0.00100,
        ),
        _pf_profile(
            "wide_init",
            0.14,
            PF_heatmap_init_pos_sd=2.5,
            PF_heatmap_init_rate_sd=0.025,
            PF_heatmap_resample_threshold=0.35,
            PF_heatmap_rough_pos=0.10,
            PF_heatmap_rough_rate=0.0060,
        ),
        _pf_profile(
            "jump_rescue",
            0.12,
            PF_heatmap_jump_prob=0.000075,
            PF_heatmap_jump_sd=10.0,
            PF_heatmap_jump_rate_sd=0.00060,
            PF_heatmap_missing_jump_boost=0.000075,
            PF_heatmap_resample_threshold=0.35,
        ),
        _pf_profile(
            "tight_init",
            0.10,
            PF_heatmap_init_pos_sd=1.0,
            PF_heatmap_init_rate_sd=0.015,
            PF_heatmap_pos_noise=0.0035,
            PF_heatmap_rough_pos=0.06,
        ),
    ],
    "profile_dyn_6x_stiff": [
        _pf_profile("base", 0.40),
        _pf_profile(
            "smooth_stiff",
            0.20,
            PF_heatmap_momentum=0.9995,
            PF_heatmap_rate_noise=0.0008,
            PF_heatmap_pos_noise=0.0030,
            PF_heatmap_rough_pos=0.06,
            PF_heatmap_rough_rate=0.0045,
            PF_heatmap_jump_prob=0.000075,
            PF_heatmap_jump_sd=9.0,
            PF_heatmap_jump_rate_sd=0.00030,
            PF_heatmap_rate_mean_weight=0.00150,
        ),
        _pf_profile(
            "tight_init",
            0.16,
            PF_heatmap_init_pos_sd=1.0,
            PF_heatmap_init_rate_sd=0.015,
            PF_heatmap_pos_noise=0.0035,
            PF_heatmap_rough_pos=0.06,
        ),
        _pf_profile(
            "low_jump",
            0.10,
            PF_heatmap_jump_prob=0.00005,
            PF_heatmap_jump_sd=8.0,
            PF_heatmap_jump_rate_sd=0.00025,
            PF_heatmap_missing_jump_boost=0.00002,
        ),
        _pf_profile(
            "wide_init",
            0.08,
            PF_heatmap_init_pos_sd=2.5,
            PF_heatmap_init_rate_sd=0.025,
            PF_heatmap_resample_threshold=0.35,
        ),
        _pf_profile(
            "rough_rate",
            0.06,
            PF_heatmap_momentum=0.99935,
            PF_heatmap_rate_noise=0.0013,
            PF_heatmap_rough_rate=0.0080,
            PF_heatmap_init_rate_sd=0.025,
        ),
    ],
    "profile_dyn_6x_rescue": [
        _pf_profile("base", 0.30),
        _pf_profile(
            "wide_init",
            0.20,
            PF_heatmap_init_pos_sd=2.5,
            PF_heatmap_init_rate_sd=0.025,
            PF_heatmap_resample_threshold=0.35,
            PF_heatmap_rough_pos=0.10,
            PF_heatmap_rough_rate=0.0060,
        ),
        _pf_profile(
            "jump_rescue",
            0.18,
            PF_heatmap_jump_prob=0.000075,
            PF_heatmap_jump_sd=10.0,
            PF_heatmap_jump_rate_sd=0.00060,
            PF_heatmap_missing_jump_boost=0.000075,
            PF_heatmap_resample_threshold=0.35,
        ),
        _pf_profile(
            "rough_rate",
            0.14,
            PF_heatmap_momentum=0.99935,
            PF_heatmap_rate_noise=0.0013,
            PF_heatmap_pos_noise=0.0045,
            PF_heatmap_rough_pos=0.10,
            PF_heatmap_rough_rate=0.0080,
            PF_heatmap_init_rate_sd=0.025,
            PF_heatmap_rate_mean_weight=0.00100,
        ),
        _pf_profile(
            "medium_jump",
            0.10,
            PF_heatmap_jump_prob=0.000225,
            PF_heatmap_jump_sd=5.8,
            PF_heatmap_jump_rate_sd=0.00040,
            PF_heatmap_missing_jump_boost=0.00005,
        ),
        _pf_profile(
            "smooth_stiff",
            0.08,
            PF_heatmap_momentum=0.9995,
            PF_heatmap_rate_noise=0.0008,
            PF_heatmap_pos_noise=0.0030,
            PF_heatmap_rough_rate=0.0045,
        ),
    ],
    "profile_dyn_obs_8x_balanced": [
        _pf_profile("base", 0.28),
        _pf_profile(
            "smooth_stiff",
            0.12,
            PF_heatmap_momentum=0.9995,
            PF_heatmap_rate_noise=0.0008,
            PF_heatmap_pos_noise=0.0030,
            PF_heatmap_rough_pos=0.06,
            PF_heatmap_rough_rate=0.0045,
            PF_heatmap_jump_prob=0.000075,
            PF_heatmap_jump_sd=9.0,
            PF_heatmap_jump_rate_sd=0.00030,
        ),
        _pf_profile(
            "rough_rate",
            0.12,
            PF_heatmap_momentum=0.99935,
            PF_heatmap_rate_noise=0.0013,
            PF_heatmap_pos_noise=0.0045,
            PF_heatmap_rough_pos=0.10,
            PF_heatmap_rough_rate=0.0080,
            PF_heatmap_init_rate_sd=0.025,
        ),
        _pf_profile(
            "wide_init",
            0.12,
            PF_heatmap_init_pos_sd=2.5,
            PF_heatmap_init_rate_sd=0.025,
            PF_heatmap_resample_threshold=0.35,
        ),
        _pf_profile(
            "jump_rescue",
            0.10,
            PF_heatmap_jump_prob=0.000075,
            PF_heatmap_jump_sd=10.0,
            PF_heatmap_jump_rate_sd=0.00060,
            PF_heatmap_missing_jump_boost=0.000075,
        ),
        _pf_profile(
            "raw_only_mild",
            0.08,
            PF_heatmap_raw_ref_likelihood_weight=0.0,
        ),
        _pf_profile(
            "rawmix_high",
            0.08,
            PF_heatmap_raw_ref_likelihood_weight=0.25,
            PF_heatmap_seen_blend_weight=0.60,
        ),
        _pf_profile(
            "seen_heavy",
            0.10,
            PF_heatmap_seen_blend_weight=0.85,
            PF_heatmap_raw_ref_likelihood_weight=0.10,
        ),
    ],
    "profile_dyn_z_8x": [
        _pf_profile("base", 0.30),
        _pf_profile(
            "z095",
            0.12,
            PF_heatmap_state_z_weight=0.95,
            PF_heatmap_rate_noise=0.0012,
            PF_heatmap_rough_rate=0.0070,
        ),
        _pf_profile(
            "z090",
            0.10,
            PF_heatmap_state_z_weight=0.90,
            PF_heatmap_rate_noise=0.0012,
            PF_heatmap_rough_rate=0.0070,
            PF_heatmap_raw_ref_likelihood_weight=0.15,
        ),
        _pf_profile(
            "smooth_stiff",
            0.14,
            PF_heatmap_momentum=0.9995,
            PF_heatmap_rate_noise=0.0008,
            PF_heatmap_pos_noise=0.0030,
            PF_heatmap_rough_rate=0.0045,
        ),
        _pf_profile(
            "wide_init",
            0.12,
            PF_heatmap_init_pos_sd=2.5,
            PF_heatmap_init_rate_sd=0.025,
        ),
        _pf_profile(
            "jump_rescue",
            0.10,
            PF_heatmap_jump_prob=0.000075,
            PF_heatmap_jump_sd=10.0,
            PF_heatmap_jump_rate_sd=0.00060,
        ),
        _pf_profile(
            "tight_init",
            0.06,
            PF_heatmap_init_pos_sd=1.0,
            PF_heatmap_init_rate_sd=0.015,
        ),
        _pf_profile(
            "rough_rate",
            0.06,
            PF_heatmap_momentum=0.99935,
            PF_heatmap_rate_noise=0.0013,
            PF_heatmap_rough_rate=0.0080,
        ),
    ],
    "profile_pm0007_obs_balanced": [
        _pf_profile("base", 0.50),
        _pf_profile(
            "ambiguity_primary",
            0.14,
            PF_heatmap_gr_ambiguity_power=0.50,
            PF_heatmap_gr_ambiguity_min_power=0.45,
            PF_heatmap_gr_ambiguity_contrast=1.00,
        ),
        _pf_profile(
            "seen_light",
            0.07,
            PF_heatmap_raw_ref_likelihood_weight=0.20,
            PF_heatmap_seen_blend_weight=0.55,
        ),
        _pf_profile(
            "seen_heavy",
            0.08,
            PF_heatmap_raw_ref_likelihood_weight=0.10,
            PF_heatmap_seen_blend_weight=0.85,
        ),
        _pf_profile(
            "outlier_light",
            0.07,
            PF_heatmap_outlier_prob=0.010,
            PF_heatmap_outlier_likelihood=0.050,
        ),
        _pf_profile(
            "dyn_sigma_slow",
            0.06,
            PF_heatmap_dynamic_sigma_alpha=0.020,
            PF_heatmap_dynamic_sigma_threshold=1.35,
            PF_heatmap_dynamic_sigma_power=0.60,
            PF_heatmap_dynamic_sigma_min=0.90,
            PF_heatmap_dynamic_sigma_max=1.75,
        ),
        _pf_profile(
            "info_entropy",
            0.05,
            PF_heatmap_gr_information_power=0.35,
            PF_heatmap_gr_information_center=0.70,
            PF_heatmap_gr_information_min_multiplier=0.70,
            PF_heatmap_gr_information_max_multiplier=1.20,
        ),
        _pf_profile(
            "lookahead_short",
            0.03,
            PF_heatmap_lookahead_power=0.0010,
            PF_heatmap_lookahead_steps=2,
            PF_heatmap_lookahead_decay=0.35,
            PF_heatmap_lookahead_max_gap=64.0,
        ),
    ],
    "profile_pm0007_outlier_sigma": [
        _pf_profile("base", 0.56),
        _pf_profile(
            "ambiguity_primary",
            0.15,
            PF_heatmap_gr_ambiguity_power=0.50,
            PF_heatmap_gr_ambiguity_min_power=0.45,
        ),
        _pf_profile(
            "seen_light",
            0.07,
            PF_heatmap_raw_ref_likelihood_weight=0.20,
            PF_heatmap_seen_blend_weight=0.55,
        ),
        _pf_profile(
            "seen_heavy",
            0.08,
            PF_heatmap_raw_ref_likelihood_weight=0.10,
            PF_heatmap_seen_blend_weight=0.85,
        ),
        _pf_profile(
            "outlier_light",
            0.08,
            PF_heatmap_outlier_prob=0.010,
            PF_heatmap_outlier_likelihood=0.050,
        ),
        _pf_profile(
            "dyn_sigma_slow",
            0.06,
            PF_heatmap_dynamic_sigma_alpha=0.020,
            PF_heatmap_dynamic_sigma_threshold=1.35,
            PF_heatmap_dynamic_sigma_power=0.60,
            PF_heatmap_dynamic_sigma_min=0.90,
            PF_heatmap_dynamic_sigma_max=1.75,
        ),
    ],
    "profile_pm0007_info_lookahead": [
        _pf_profile("base", 0.54),
        _pf_profile(
            "ambiguity_primary",
            0.15,
            PF_heatmap_gr_ambiguity_power=0.50,
            PF_heatmap_gr_ambiguity_min_power=0.45,
        ),
        _pf_profile(
            "seen_light",
            0.07,
            PF_heatmap_raw_ref_likelihood_weight=0.20,
            PF_heatmap_seen_blend_weight=0.55,
        ),
        _pf_profile(
            "seen_heavy",
            0.08,
            PF_heatmap_raw_ref_likelihood_weight=0.10,
            PF_heatmap_seen_blend_weight=0.85,
        ),
        _pf_profile(
            "info_entropy",
            0.07,
            PF_heatmap_gr_information_power=0.35,
            PF_heatmap_gr_information_center=0.70,
            PF_heatmap_gr_information_min_multiplier=0.70,
            PF_heatmap_gr_information_max_multiplier=1.20,
        ),
        _pf_profile(
            "info_slope",
            0.05,
            PF_heatmap_gr_information_power=0.50,
            PF_heatmap_gr_information_center=0.70,
            PF_heatmap_gr_information_min_multiplier=0.65,
            PF_heatmap_gr_information_max_multiplier=1.30,
            PF_heatmap_gr_information_slope_weight=0.35,
        ),
        _pf_profile(
            "lookahead_short",
            0.04,
            PF_heatmap_lookahead_power=0.0010,
            PF_heatmap_lookahead_steps=2,
            PF_heatmap_lookahead_decay=0.35,
            PF_heatmap_lookahead_max_gap=64.0,
        ),
    ],
}


for _legacy_profile_mixture_name in (
    "profile_dyn_obs_8x_balanced",
    "profile_dyn_z_8x",
    "profile_pm0007_obs_balanced",
    "profile_pm0007_outlier_sigma",
    "profile_pm0007_info_lookahead",
):
    PF_OPT_PROFILE_MIXTURES.pop(_legacy_profile_mixture_name, None)

PF_OPT_PROFILE_MIXTURES.update(
    {
        "profile_pm0002_gout005_all": _apply_global_profile_component(
            _pm0002_accepted_profiles(),
            _global_obs_component("out005"),
        ),
        "profile_pm0002_gdyn010_all": _apply_global_profile_component(
            _pm0002_accepted_profiles(),
            _global_obs_component("dyn010"),
        ),
        "profile_pm0002_ginfo025_all": _apply_global_profile_component(
            _pm0002_accepted_profiles(),
            _global_obs_component("info025"),
        ),
        "profile_pm0002_gout_dyn_info_all": _apply_global_profile_component(
            _pm0002_accepted_profiles(),
            {
                **_global_obs_component("out005"),
                **_global_obs_component("dyn010"),
                **_global_obs_component("info025"),
            },
        ),
        "profile_pm0002_graw020_amb_all": _apply_global_profile_component(
            _pm0002_accepted_profiles(),
            {
                **_global_obs_component("raw020"),
                **_global_obs_component("amb035"),
            },
        ),
        "profile_pm0002_gfrdecay025_all": _apply_global_profile_component(
            _pm0002_accepted_profiles(),
            _global_obs_component("frdecay025"),
        ),
        "profile_pm0002_gfrdecay050_all": _apply_global_profile_component(
            _pm0002_accepted_profiles(),
            _global_obs_component("frdecay050"),
        ),
        "profile_pm0002_gfrdecay100_all": _apply_global_profile_component(
            _pm0002_accepted_profiles(),
            _global_obs_component("frdecay100"),
        ),
        "profile_pm0002_gfrdecay050_amb_all": _apply_global_profile_component(
            _pm0002_accepted_profiles(),
            {
                **_global_obs_component("frdecay050"),
                **_global_obs_component("amb035"),
            },
        ),
        "profile_pm0002_gfrdecay050_raw_all": _apply_global_profile_component(
            _pm0002_accepted_profiles(),
            {
                **_global_obs_component("frdecay050"),
                **_global_obs_component("raw020"),
            },
        ),
        "profile_pm0002_8x_dyn_micro": [
            _pf_profile("base", 0.45),
            _pf_profile(
                "wide_init",
                0.16,
                PF_heatmap_init_pos_sd=2.5,
                PF_heatmap_init_rate_sd=0.025,
                PF_heatmap_resample_threshold=0.35,
                PF_heatmap_rough_pos=0.10,
                PF_heatmap_rough_rate=0.006,
            ),
            _pf_profile(
                "wide_init_light",
                0.08,
                PF_heatmap_init_pos_sd=2.0,
                PF_heatmap_init_rate_sd=0.0225,
                PF_heatmap_resample_threshold=0.375,
                PF_heatmap_rough_pos=0.09,
                PF_heatmap_rough_rate=0.0055,
            ),
            _pf_profile(
                "smooth_stiff",
                0.08,
                PF_heatmap_momentum=0.9995,
                PF_heatmap_rate_noise=0.0008,
                PF_heatmap_pos_noise=0.0030,
                PF_heatmap_rough_pos=0.06,
                PF_heatmap_rough_rate=0.0045,
                PF_heatmap_jump_prob=0.000075,
                PF_heatmap_jump_sd=9.0,
                PF_heatmap_jump_rate_sd=0.00030,
                PF_heatmap_rate_mean_weight=0.00150,
            ),
            _pf_profile(
                "smooth_stiff_light",
                0.06,
                PF_heatmap_momentum=0.99945,
                PF_heatmap_rate_noise=0.0009,
                PF_heatmap_pos_noise=0.0035,
                PF_heatmap_rough_pos=0.07,
                PF_heatmap_rough_rate=0.0050,
                PF_heatmap_jump_prob=0.00010,
                PF_heatmap_jump_sd=8.0,
                PF_heatmap_jump_rate_sd=0.00030,
                PF_heatmap_rate_mean_weight=0.00125,
            ),
            _pf_profile(
                "jump_rare_big",
                0.07,
                PF_heatmap_jump_prob=0.000075,
                PF_heatmap_jump_sd=10.0,
                PF_heatmap_jump_rate_sd=0.00060,
                PF_heatmap_missing_jump_boost=0.000075,
                PF_heatmap_resample_threshold=0.35,
            ),
            _pf_profile(
                "rough_rate_light",
                0.06,
                PF_heatmap_momentum=0.99935,
                PF_heatmap_rate_noise=0.0012,
                PF_heatmap_pos_noise=0.0042,
                PF_heatmap_rough_pos=0.09,
                PF_heatmap_rough_rate=0.0070,
                PF_heatmap_init_rate_sd=0.0225,
                PF_heatmap_rate_mean_weight=0.00100,
            ),
            _pf_profile(
                "tight_init",
                0.04,
                PF_heatmap_init_pos_sd=1.0,
                PF_heatmap_init_rate_sd=0.015,
                PF_heatmap_pos_noise=0.0035,
                PF_heatmap_rough_pos=0.06,
            ),
        ],
        "profile_pm0002_8x_gobs_micro": _apply_global_profile_component(
            [
                _pf_profile("base", 0.45),
                _pf_profile(
                    "wide_init",
                    0.16,
                    PF_heatmap_init_pos_sd=2.5,
                    PF_heatmap_init_rate_sd=0.025,
                    PF_heatmap_resample_threshold=0.35,
                    PF_heatmap_rough_pos=0.10,
                    PF_heatmap_rough_rate=0.006,
                ),
                _pf_profile(
                    "wide_init_light",
                    0.08,
                    PF_heatmap_init_pos_sd=2.0,
                    PF_heatmap_init_rate_sd=0.0225,
                    PF_heatmap_resample_threshold=0.375,
                    PF_heatmap_rough_pos=0.09,
                    PF_heatmap_rough_rate=0.0055,
                ),
                _pf_profile(
                    "smooth_stiff",
                    0.08,
                    PF_heatmap_momentum=0.9995,
                    PF_heatmap_rate_noise=0.0008,
                    PF_heatmap_pos_noise=0.0030,
                    PF_heatmap_rough_pos=0.06,
                    PF_heatmap_rough_rate=0.0045,
                    PF_heatmap_jump_prob=0.000075,
                    PF_heatmap_jump_sd=9.0,
                    PF_heatmap_jump_rate_sd=0.00030,
                    PF_heatmap_rate_mean_weight=0.00150,
                ),
                _pf_profile(
                    "smooth_stiff_light",
                    0.06,
                    PF_heatmap_momentum=0.99945,
                    PF_heatmap_rate_noise=0.0009,
                    PF_heatmap_pos_noise=0.0035,
                    PF_heatmap_rough_pos=0.07,
                    PF_heatmap_rough_rate=0.0050,
                    PF_heatmap_jump_prob=0.00010,
                    PF_heatmap_jump_sd=8.0,
                    PF_heatmap_jump_rate_sd=0.00030,
                    PF_heatmap_rate_mean_weight=0.00125,
                ),
                _pf_profile(
                    "jump_rare_big",
                    0.07,
                    PF_heatmap_jump_prob=0.000075,
                    PF_heatmap_jump_sd=10.0,
                    PF_heatmap_jump_rate_sd=0.00060,
                    PF_heatmap_missing_jump_boost=0.000075,
                    PF_heatmap_resample_threshold=0.35,
                ),
                _pf_profile(
                    "rough_rate_light",
                    0.06,
                    PF_heatmap_momentum=0.99935,
                    PF_heatmap_rate_noise=0.0012,
                    PF_heatmap_pos_noise=0.0042,
                    PF_heatmap_rough_pos=0.09,
                    PF_heatmap_rough_rate=0.0070,
                    PF_heatmap_init_rate_sd=0.0225,
                    PF_heatmap_rate_mean_weight=0.00100,
                ),
                _pf_profile(
                    "tight_init",
                    0.04,
                    PF_heatmap_init_pos_sd=1.0,
                    PF_heatmap_init_rate_sd=0.015,
                    PF_heatmap_pos_noise=0.0035,
                    PF_heatmap_rough_pos=0.06,
                ),
            ],
            {
                **_global_obs_component("out005"),
                **_global_obs_component("dyn010"),
                **_global_obs_component("info025"),
            },
        ),
    }
)

for _overfit_profile_mixture_name in (
    "profile_pm0002_gout005_all",
    "profile_pm0002_gdyn010_all",
    "profile_pm0002_ginfo025_all",
    "profile_pm0002_gout_dyn_info_all",
    "profile_pm0002_graw020_amb_all",
    "profile_pm0002_gfrdecay050_amb_all",
    "profile_pm0002_8x_gobs_micro",
):
    PF_OPT_PROFILE_MIXTURES.pop(_overfit_profile_mixture_name, None)


def _profiles_from_weight_plan(weight_plan):
    base_profiles = {profile["name"]: profile for profile in _pm0002_accepted_profiles()}
    out = []
    for name, weight in weight_plan:
        if name not in base_profiles:
            raise KeyError(f"unknown pm0002 profile {name!r}")
        profile = dict(base_profiles[name])
        profile["weight"] = float(weight)
        out.append(profile)
    return _normalize_profile_mixture_spec(out)


PF_OPT_FINE_PROFILE_MIXTURES = {
    "profile_rs0026_pm0002": _pm0002_accepted_profiles(),
    "profile_rs0026_base_heavy": _profiles_from_weight_plan(
        [
            ("base", 0.70),
            ("wide_init", 0.16),
            ("smooth_stiff", 0.05),
            ("jump_rare_big", 0.06),
            ("tight_init", 0.03),
        ]
    ),
    "profile_rs0026_wide_light": _profiles_from_weight_plan(
        [
            ("base", 0.60),
            ("wide_init", 0.24),
            ("smooth_stiff", 0.05),
            ("jump_rare_big", 0.07),
            ("tight_init", 0.04),
        ]
    ),
    "profile_rs0026_smooth_light": _profiles_from_weight_plan(
        [
            ("base", 0.62),
            ("wide_init", 0.18),
            ("smooth_stiff", 0.10),
            ("jump_rare_big", 0.06),
            ("tight_init", 0.04),
        ]
    ),
    "profile_rs0026_jump_light": _profiles_from_weight_plan(
        [
            ("base", 0.62),
            ("wide_init", 0.17),
            ("smooth_stiff", 0.05),
            ("jump_rare_big", 0.12),
            ("tight_init", 0.04),
        ]
    ),
    "profile_rs0026_less_jump": _profiles_from_weight_plan(
        [
            ("base", 0.67),
            ("wide_init", 0.20),
            ("smooth_stiff", 0.07),
            ("jump_rare_big", 0.03),
            ("tight_init", 0.03),
        ]
    ),
}

PF_OPT_BROAD_PROFILE_MIXTURES = {
    name: PF_OPT_PROFILE_MIXTURES[name]
    for name in (
        "profile_dyn_6x_balanced",
        "profile_dyn_6x_stiff",
        "profile_dyn_6x_rescue",
        "profile_pm0002_8x_dyn_micro",
        "profile_pm0002_gfrdecay025_all",
        "profile_pm0002_gfrdecay100_all",
        "profile_pm0002_gfrdecay050_raw_all",
    )
    if name in PF_OPT_PROFILE_MIXTURES
}

# Keep the accepted rs0026/pm0002 mixture as the center, but add broad dynamic,
# finite-run, and reference-balance profile plans so the active loops are not
# trapped in only tiny local profile-weight reshapes.
PF_OPT_PROFILE_MIXTURES.clear()
PF_OPT_PROFILE_MIXTURES.update(PF_OPT_FINE_PROFILE_MIXTURES)
PF_OPT_PROFILE_MIXTURES.update(PF_OPT_BROAD_PROFILE_MIXTURES)


PF_OPT_FFBSI_BASE_OVERRIDES = {
    "PF_heatmap_ffbsi_mode": "fallback",
    "PF_heatmap_ffbsi_n_paths": 32,
    "PF_heatmap_ffbsi_fallback_mode": "filtered",
    "PF_heatmap_ffbsi_max_active_bins": 512,
    "PF_heatmap_ffbsi_transition_scale": 1.35,
    "PF_heatmap_ffbsi_pos_floor": 0.15,
    "PF_heatmap_ffbsi_rate_floor": 0.0025,
}


PF_OPT_FFBSI_PROFILE_PLANS = {
    "pm0002": _pm0002_accepted_profiles(),
    "more_smooth": _profiles_from_weight_plan(
        [
            ("base", 0.55),
            ("wide_init", 0.15),
            ("smooth_stiff", 0.18),
            ("jump_rare_big", 0.08),
            ("tight_init", 0.04),
        ]
    ),
    "more_wide": _profiles_from_weight_plan(
        [
            ("base", 0.55),
            ("wide_init", 0.28),
            ("smooth_stiff", 0.06),
            ("jump_rare_big", 0.08),
            ("tight_init", 0.03),
        ]
    ),
    "less_jump": _profiles_from_weight_plan(
        [
            ("base", 0.68),
            ("wide_init", 0.20),
            ("smooth_stiff", 0.08),
            ("jump_rare_big", 0.02),
            ("tight_init", 0.02),
        ]
    ),
    "more_jump": _profiles_from_weight_plan(
        [
            ("base", 0.58),
            ("wide_init", 0.17),
            ("smooth_stiff", 0.06),
            ("jump_rare_big", 0.16),
            ("tight_init", 0.03),
        ]
    ),
    "base_heavy": _profiles_from_weight_plan(
        [
            ("base", 0.78),
            ("wide_init", 0.10),
            ("smooth_stiff", 0.05),
            ("jump_rare_big", 0.05),
            ("tight_init", 0.02),
        ]
    ),
}


PF_OPT_FFBSI_PROFILE_MIXTURES = {
    f"ffbsi_prof_{name}": profiles
    for name, profiles in PF_OPT_FFBSI_PROFILE_PLANS.items()
    if name != "pm0002"
}


PF_OPT_PROFILE_CANDIDATES = tuple(
    (
        name,
        {
            "PF_heatmap_profile_mixture_spec": profiles,
        },
    )
    for name, profiles in PF_OPT_PROFILE_MIXTURES.items()
)


PF_OPT_FINITE_RUN_CURATED_CANDIDATES = (
    (
        "frdecay015_floor055",
        {
            "PF_heatmap_finite_run_power_decay": 0.15,
            "PF_heatmap_finite_run_power_floor": 0.55,
        },
    ),
    (
        "frdecay035_floor055",
        {
            "PF_heatmap_finite_run_power_decay": 0.35,
            "PF_heatmap_finite_run_power_floor": 0.55,
        },
    ),
    (
        "frdecay050_floor055",
        {
            "PF_heatmap_finite_run_power_decay": 0.50,
            "PF_heatmap_finite_run_power_floor": 0.55,
        },
    ),
    (
        "raw0125_seen075",
        {
            "PF_heatmap_raw_ref_likelihood_weight": 0.125,
            "PF_heatmap_seen_blend_weight": 0.75,
        },
    ),
    (
        "raw010_seen080",
        {
            "PF_heatmap_raw_ref_likelihood_weight": 0.10,
            "PF_heatmap_seen_blend_weight": 0.80,
        },
    ),
    (
        "lik575_anchor85",
        {
            "PF_heatmap_lik_scale": 5.75,
            "PF_heatmap_anchor_sigma": 85.0,
        },
    ),
)


PF_OPT_PROFILE_CANDIDATES = PF_OPT_PROFILE_CANDIDATES + PF_OPT_FINITE_RUN_CURATED_CANDIDATES


PF_OPT_FFBSI_CURATED_CANDIDATES = (
    ("ffbsi_floor010", {"PF_heatmap_ffbsi_pos_floor": 0.10}),
    ("ffbsi_floor020", {"PF_heatmap_ffbsi_pos_floor": 0.20}),
    ("ffbsi_floor025", {"PF_heatmap_ffbsi_pos_floor": 0.25}),
    ("ffbsi_scale120", {"PF_heatmap_ffbsi_transition_scale": 1.20}),
    ("ffbsi_scale150", {"PF_heatmap_ffbsi_transition_scale": 1.50}),
    ("ffbsi_scale165", {"PF_heatmap_ffbsi_transition_scale": 1.65}),
    ("ffbsi_paths064", {"PF_heatmap_ffbsi_n_paths": 64}),
    ("ffbsi_paths096", {"PF_heatmap_ffbsi_n_paths": 96}),
    ("ffbsi_jump08", {"PF_heatmap_jump_sd": 8.0}),
    ("ffbsi_jump10", {"PF_heatmap_jump_sd": 10.0}),
    ("ffbsi_lik55", {"PF_heatmap_lik_scale": 5.5}),
    ("ffbsi_lik65", {"PF_heatmap_lik_scale": 6.5}),
    ("ffbsi_more_wide", {"PF_heatmap_profile_mixture_spec": PF_OPT_FFBSI_PROFILE_PLANS["more_wide"]}),
    ("ffbsi_more_jump", {"PF_heatmap_profile_mixture_spec": PF_OPT_FFBSI_PROFILE_PLANS["more_jump"]}),
    ("ffbsi_more_wide_scale150", {"PF_heatmap_profile_mixture_spec": PF_OPT_FFBSI_PROFILE_PLANS["more_wide"], "PF_heatmap_ffbsi_transition_scale": 1.50}),
    ("ffbsi_more_jump_scale150", {"PF_heatmap_profile_mixture_spec": PF_OPT_FFBSI_PROFILE_PLANS["more_jump"], "PF_heatmap_ffbsi_transition_scale": 1.50}),
    ("ffbsi_pm0002_paths096_rate14", {"PF_heatmap_profile_mixture_spec": PF_OPT_FFBSI_PROFILE_PLANS["pm0002"], "PF_heatmap_ffbsi_n_paths": 96, "PF_heatmap_rate_noise": 0.0014}),
    ("ffbsi_smooth_jump10_scale150", {"PF_heatmap_jump_sd": 10.0, "PF_heatmap_ffbsi_transition_scale": 1.50}),
)


PF_OPT_FFBSI_PROFILE_CANDIDATES = tuple(
    (
        name,
        {
            "PF_heatmap_profile_mixture_spec": profiles,
        },
    )
    for name, profiles in PF_OPT_FFBSI_PROFILE_MIXTURES.items()
) + PF_OPT_FFBSI_CURATED_CANDIDATES


PF_PROFILE_ARCHETYPES = {
    "base": {},
    "smooth_stiff": {
        "PF_heatmap_momentum": 0.9995,
        "PF_heatmap_rate_noise": 0.0008,
        "PF_heatmap_pos_noise": 0.0030,
        "PF_heatmap_rough_pos": 0.06,
        "PF_heatmap_rough_rate": 0.0045,
        "PF_heatmap_jump_prob": 0.000075,
        "PF_heatmap_jump_sd": 9.0,
        "PF_heatmap_jump_rate_sd": 0.00030,
        "PF_heatmap_rate_mean_weight": 0.00150,
    },
    "ultra_stiff": {
        "PF_heatmap_momentum": 0.99955,
        "PF_heatmap_rate_noise": 0.00065,
        "PF_heatmap_pos_noise": 0.0025,
        "PF_heatmap_rough_pos": 0.05,
        "PF_heatmap_rough_rate": 0.0035,
        "PF_heatmap_jump_prob": 0.00005,
        "PF_heatmap_jump_sd": 8.0,
        "PF_heatmap_jump_rate_sd": 0.00025,
        "PF_heatmap_rate_mean_weight": 0.00175,
    },
    "rough_rate": {
        "PF_heatmap_momentum": 0.99935,
        "PF_heatmap_rate_noise": 0.0013,
        "PF_heatmap_pos_noise": 0.0045,
        "PF_heatmap_rough_pos": 0.10,
        "PF_heatmap_rough_rate": 0.0080,
        "PF_heatmap_init_rate_sd": 0.025,
        "PF_heatmap_rate_mean_weight": 0.00100,
    },
    "mobile_rate": {
        "PF_heatmap_momentum": 0.99925,
        "PF_heatmap_rate_noise": 0.0015,
        "PF_heatmap_pos_noise": 0.0050,
        "PF_heatmap_rough_pos": 0.12,
        "PF_heatmap_rough_rate": 0.0080,
        "PF_heatmap_init_rate_sd": 0.025,
        "PF_heatmap_rate_mean_weight": 0.00075,
    },
    "wide_init": {
        "PF_heatmap_init_pos_sd": 2.5,
        "PF_heatmap_init_rate_sd": 0.025,
        "PF_heatmap_resample_threshold": 0.35,
        "PF_heatmap_rough_pos": 0.10,
        "PF_heatmap_rough_rate": 0.0060,
    },
    "very_wide_init": {
        "PF_heatmap_init_pos_sd": 3.0,
        "PF_heatmap_init_rate_sd": 0.025,
        "PF_heatmap_resample_threshold": 0.35,
        "PF_heatmap_rough_pos": 0.12,
        "PF_heatmap_rough_rate": 0.0070,
    },
    "tight_init": {
        "PF_heatmap_init_pos_sd": 1.0,
        "PF_heatmap_init_rate_sd": 0.015,
        "PF_heatmap_pos_noise": 0.0035,
        "PF_heatmap_rough_pos": 0.06,
    },
    "jump_rare_big": {
        "PF_heatmap_jump_prob": 0.000075,
        "PF_heatmap_jump_sd": 10.0,
        "PF_heatmap_jump_rate_sd": 0.00060,
        "PF_heatmap_missing_jump_boost": 0.000075,
        "PF_heatmap_resample_threshold": 0.35,
    },
    "jump_medium": {
        "PF_heatmap_jump_prob": 0.000225,
        "PF_heatmap_jump_sd": 5.8,
        "PF_heatmap_jump_rate_sd": 0.00040,
        "PF_heatmap_missing_jump_boost": 0.00005,
    },
    "missing_loose": {
        "PF_heatmap_missing_jump_boost": 0.00010,
        "PF_heatmap_missing_noise_scale": 1.10,
        "PF_heatmap_missing_jump_scale": 1.10,
        "PF_heatmap_resample_threshold": 0.35,
    },
    "missing_tight": {
        "PF_heatmap_missing_jump_boost": 0.00001,
        "PF_heatmap_missing_noise_scale": 0.90,
        "PF_heatmap_missing_jump_scale": 0.90,
        "PF_heatmap_resample_threshold": 0.425,
    },
    "raw_low": {
        "PF_heatmap_raw_ref_likelihood_weight": 0.0,
    },
    "raw_high": {
        "PF_heatmap_raw_ref_likelihood_weight": 0.25,
        "PF_heatmap_seen_blend_weight": 0.60,
    },
    "seen_heavy": {
        "PF_heatmap_seen_blend_weight": 0.85,
        "PF_heatmap_raw_ref_likelihood_weight": 0.10,
    },
    "seen_light": {
        "PF_heatmap_seen_blend_weight": 0.55,
        "PF_heatmap_raw_ref_likelihood_weight": 0.20,
    },
    "ambiguity_primary": {
        "PF_heatmap_gr_ambiguity_power": 0.50,
        "PF_heatmap_gr_ambiguity_min_power": 0.45,
        "PF_heatmap_gr_ambiguity_contrast": 1.00,
    },
    "ambiguity_raw": {
        "PF_heatmap_gr_ambiguity_power": 0.50,
        "PF_heatmap_gr_ambiguity_min_power": 0.35,
        "PF_heatmap_gr_ambiguity_ref_mode": "raw",
    },
    "outlier_light": {
        "PF_heatmap_outlier_prob": 0.010,
        "PF_heatmap_outlier_likelihood": 0.050,
    },
    "outlier_med": {
        "PF_heatmap_outlier_prob": 0.025,
        "PF_heatmap_outlier_likelihood": 0.075,
    },
    "dyn_sigma_slow": {
        "PF_heatmap_dynamic_sigma_alpha": 0.020,
        "PF_heatmap_dynamic_sigma_threshold": 1.35,
        "PF_heatmap_dynamic_sigma_power": 0.60,
        "PF_heatmap_dynamic_sigma_min": 0.90,
        "PF_heatmap_dynamic_sigma_max": 1.75,
    },
    "dyn_sigma_med": {
        "PF_heatmap_dynamic_sigma_alpha": 0.050,
        "PF_heatmap_dynamic_sigma_threshold": 1.25,
        "PF_heatmap_dynamic_sigma_power": 0.75,
        "PF_heatmap_dynamic_sigma_min": 0.85,
        "PF_heatmap_dynamic_sigma_max": 2.00,
    },
    "info_entropy": {
        "PF_heatmap_gr_information_power": 0.35,
        "PF_heatmap_gr_information_center": 0.70,
        "PF_heatmap_gr_information_min_multiplier": 0.70,
        "PF_heatmap_gr_information_max_multiplier": 1.20,
    },
    "info_slope": {
        "PF_heatmap_gr_information_power": 0.50,
        "PF_heatmap_gr_information_center": 0.70,
        "PF_heatmap_gr_information_min_multiplier": 0.65,
        "PF_heatmap_gr_information_max_multiplier": 1.30,
        "PF_heatmap_gr_information_slope_weight": 0.35,
    },
    "lookahead_short": {
        "PF_heatmap_lookahead_power": 0.0010,
        "PF_heatmap_lookahead_steps": 2,
        "PF_heatmap_lookahead_decay": 0.35,
        "PF_heatmap_lookahead_max_gap": 64.0,
    },
    "lookahead_delta": {
        "PF_heatmap_lookahead_power": 0.0015,
        "PF_heatmap_lookahead_steps": 2,
        "PF_heatmap_lookahead_decay": 0.35,
        "PF_heatmap_lookahead_max_gap": 64.0,
        "PF_heatmap_lookahead_delta_power": 0.25,
    },
    "z095": {
        "PF_heatmap_state_z_weight": 0.95,
        "PF_heatmap_rate_noise": 0.0012,
        "PF_heatmap_rough_rate": 0.0070,
    },
    "z090": {
        "PF_heatmap_state_z_weight": 0.90,
        "PF_heatmap_rate_noise": 0.0012,
        "PF_heatmap_rough_rate": 0.0070,
        "PF_heatmap_raw_ref_likelihood_weight": 0.15,
    },
    "anchor_tighter": {
        "PF_heatmap_anchor_sigma": 60.0,
        "PF_heatmap_anchor_power": 0.0025,
    },
    "anchor_looser": {
        "PF_heatmap_anchor_sigma": 90.0,
        "PF_heatmap_anchor_power": 0.0015,
    },
}


PF_PROFILE_ARCHETYPES.update(
    {
        "smooth_stiff_light": {
            "PF_heatmap_momentum": 0.99945,
            "PF_heatmap_rate_noise": 0.0009,
            "PF_heatmap_pos_noise": 0.0035,
            "PF_heatmap_rough_pos": 0.07,
            "PF_heatmap_rough_rate": 0.0050,
            "PF_heatmap_jump_prob": 0.00010,
            "PF_heatmap_jump_sd": 8.0,
            "PF_heatmap_jump_rate_sd": 0.00030,
            "PF_heatmap_rate_mean_weight": 0.00125,
        },
        "smooth_stiff_heavy": {
            "PF_heatmap_momentum": 0.99955,
            "PF_heatmap_rate_noise": 0.0007,
            "PF_heatmap_pos_noise": 0.0028,
            "PF_heatmap_rough_pos": 0.055,
            "PF_heatmap_rough_rate": 0.0040,
            "PF_heatmap_jump_prob": 0.00005,
            "PF_heatmap_jump_sd": 9.0,
            "PF_heatmap_jump_rate_sd": 0.00025,
            "PF_heatmap_rate_mean_weight": 0.00175,
        },
        "rough_rate_light": {
            "PF_heatmap_momentum": 0.99935,
            "PF_heatmap_rate_noise": 0.0012,
            "PF_heatmap_pos_noise": 0.0042,
            "PF_heatmap_rough_pos": 0.09,
            "PF_heatmap_rough_rate": 0.0070,
            "PF_heatmap_init_rate_sd": 0.0225,
            "PF_heatmap_rate_mean_weight": 0.00100,
        },
        "rough_rate_heavy": {
            "PF_heatmap_momentum": 0.99925,
            "PF_heatmap_rate_noise": 0.0015,
            "PF_heatmap_pos_noise": 0.0050,
            "PF_heatmap_rough_pos": 0.12,
            "PF_heatmap_rough_rate": 0.0090,
            "PF_heatmap_init_rate_sd": 0.0250,
            "PF_heatmap_rate_mean_weight": 0.00075,
        },
        "wide_init_light": {
            "PF_heatmap_init_pos_sd": 2.0,
            "PF_heatmap_init_rate_sd": 0.0225,
            "PF_heatmap_resample_threshold": 0.375,
            "PF_heatmap_rough_pos": 0.09,
            "PF_heatmap_rough_rate": 0.0055,
        },
        "wide_init_heavy": {
            "PF_heatmap_init_pos_sd": 2.75,
            "PF_heatmap_init_rate_sd": 0.0275,
            "PF_heatmap_resample_threshold": 0.325,
            "PF_heatmap_rough_pos": 0.11,
            "PF_heatmap_rough_rate": 0.0065,
        },
        "tight_init_stiff": {
            "PF_heatmap_init_pos_sd": 0.85,
            "PF_heatmap_init_rate_sd": 0.0125,
            "PF_heatmap_pos_noise": 0.0030,
            "PF_heatmap_rough_pos": 0.055,
            "PF_heatmap_rough_rate": 0.0045,
        },
        "jump_rare_soft": {
            "PF_heatmap_jump_prob": 0.00005,
            "PF_heatmap_jump_sd": 8.0,
            "PF_heatmap_jump_rate_sd": 0.00030,
            "PF_heatmap_missing_jump_boost": 0.000025,
            "PF_heatmap_resample_threshold": 0.375,
        },
        "jump_rare_mid": {
            "PF_heatmap_jump_prob": 0.00010,
            "PF_heatmap_jump_sd": 9.0,
            "PF_heatmap_jump_rate_sd": 0.00045,
            "PF_heatmap_missing_jump_boost": 0.00005,
            "PF_heatmap_resample_threshold": 0.35,
        },
        "raw_mid": {
            "PF_heatmap_raw_ref_likelihood_weight": 0.20,
            "PF_heatmap_seen_blend_weight": 0.65,
        },
        "raw_very_low": {
            "PF_heatmap_raw_ref_likelihood_weight": 0.05,
            "PF_heatmap_seen_blend_weight": 0.75,
        },
        "seen_mid": {
            "PF_heatmap_seen_blend_weight": 0.75,
            "PF_heatmap_raw_ref_likelihood_weight": 0.15,
        },
        "seen_very_heavy": {
            "PF_heatmap_seen_blend_weight": 0.90,
            "PF_heatmap_raw_ref_likelihood_weight": 0.05,
        },
        "ambiguity_soft": {
            "PF_heatmap_gr_ambiguity_power": 0.35,
            "PF_heatmap_gr_ambiguity_min_power": 0.45,
            "PF_heatmap_gr_ambiguity_contrast": 1.00,
        },
        "ambiguity_hard": {
            "PF_heatmap_gr_ambiguity_power": 0.75,
            "PF_heatmap_gr_ambiguity_min_power": 0.50,
            "PF_heatmap_gr_ambiguity_contrast": 1.15,
        },
        "z0975": {
            "PF_heatmap_state_z_weight": 0.975,
            "PF_heatmap_rate_noise": 0.0011,
            "PF_heatmap_rough_rate": 0.0065,
        },
        "z0925": {
            "PF_heatmap_state_z_weight": 0.925,
            "PF_heatmap_rate_noise": 0.0012,
            "PF_heatmap_rough_rate": 0.0070,
            "PF_heatmap_raw_ref_likelihood_weight": 0.15,
        },
        "anchor_mid_tight": {
            "PF_heatmap_anchor_sigma": 65.0,
            "PF_heatmap_anchor_power": 0.0025,
        },
        "anchor_mid_loose": {
            "PF_heatmap_anchor_sigma": 85.0,
            "PF_heatmap_anchor_power": 0.00175,
        },
        "missing_balanced": {
            "PF_heatmap_missing_jump_boost": 0.00005,
            "PF_heatmap_missing_noise_scale": 1.00,
            "PF_heatmap_missing_jump_scale": 1.00,
            "PF_heatmap_resample_threshold": 0.40,
        },
        "rate_mean_low": {
            "PF_heatmap_rate_mean_weight": 0.00075,
            "PF_heatmap_rate_noise": 0.0012,
            "PF_heatmap_rough_rate": 0.0065,
        },
        "rate_mean_high": {
            "PF_heatmap_rate_mean_weight": 0.00175,
            "PF_heatmap_rate_noise": 0.0009,
            "PF_heatmap_rough_rate": 0.0050,
        },
        "resample_low": {
            "PF_heatmap_resample_threshold": 0.30,
            "PF_heatmap_rough_pos": 0.10,
            "PF_heatmap_rough_rate": 0.0065,
        },
        "resample_high": {
            "PF_heatmap_resample_threshold": 0.45,
            "PF_heatmap_rough_pos": 0.07,
            "PF_heatmap_rough_rate": 0.0050,
        },
    }
)


for _legacy_obs_profile_name in (
    "outlier_light",
    "outlier_med",
    "dyn_sigma_slow",
    "dyn_sigma_med",
    "info_entropy",
    "info_slope",
    "lookahead_short",
    "lookahead_delta",
):
    PF_PROFILE_ARCHETYPES.pop(_legacy_obs_profile_name, None)


PF_PROFILE_DIRECTIONS = {
    "dyn_balanced": (
        "base",
        "wide_init",
        "smooth_stiff",
        "jump_rare_big",
        "tight_init",
        "rough_rate_light",
        "wide_init_light",
        "smooth_stiff_light",
    ),
    "dyn_stiff": (
        "base",
        "smooth_stiff",
        "smooth_stiff_light",
        "smooth_stiff_heavy",
        "ultra_stiff",
        "tight_init",
        "tight_init_stiff",
        "missing_tight",
        "wide_init_light",
    ),
    "dyn_rescue": (
        "base",
        "wide_init",
        "wide_init_light",
        "wide_init_heavy",
        "very_wide_init",
        "jump_rare_big",
        "jump_rare_mid",
        "jump_rare_soft",
        "jump_medium",
        "rough_rate",
        "rough_rate_light",
    ),
    "dyn_micro": (
        "base",
        "smooth_stiff",
        "smooth_stiff_light",
        "smooth_stiff_heavy",
        "ultra_stiff",
        "rough_rate",
        "rough_rate_light",
        "rough_rate_heavy",
        "mobile_rate",
        "wide_init",
        "wide_init_light",
        "wide_init_heavy",
        "very_wide_init",
        "tight_init",
        "tight_init_stiff",
        "jump_rare_big",
        "jump_rare_mid",
        "jump_rare_soft",
        "jump_medium",
        "rate_mean_low",
        "rate_mean_high",
        "resample_low",
        "resample_high",
    ),
    "calibration": (
        "base",
        "raw_very_low",
        "raw_low",
        "raw_mid",
        "raw_high",
        "seen_mid",
        "seen_light",
        "seen_heavy",
        "seen_very_heavy",
        "ambiguity_soft",
        "ambiguity_primary",
        "ambiguity_hard",
        "ambiguity_raw",
    ),
    "calib_micro": (
        "base",
        "wide_init",
        "smooth_stiff",
        "jump_rare_big",
        "tight_init",
        "raw_very_low",
        "raw_low",
        "raw_mid",
        "raw_high",
        "seen_mid",
        "seen_light",
        "seen_heavy",
        "seen_very_heavy",
        "ambiguity_soft",
        "ambiguity_primary",
        "ambiguity_hard",
        "ambiguity_raw",
    ),
    "obs_global_dyn": (
        "base",
        "wide_init",
        "wide_init_light",
        "smooth_stiff",
        "smooth_stiff_light",
        "jump_rare_big",
        "jump_rare_mid",
        "tight_init",
        "rough_rate_light",
        "rate_mean_high",
    ),
    "obs_global_wide": (
        "base",
        "smooth_stiff",
        "smooth_stiff_light",
        "smooth_stiff_heavy",
        "ultra_stiff",
        "rough_rate",
        "rough_rate_light",
        "rough_rate_heavy",
        "mobile_rate",
        "wide_init",
        "wide_init_light",
        "wide_init_heavy",
        "very_wide_init",
        "tight_init",
        "tight_init_stiff",
        "jump_rare_big",
        "jump_rare_mid",
        "jump_rare_soft",
        "jump_medium",
        "missing_balanced",
        "missing_tight",
        "missing_loose",
        "raw_very_low",
        "raw_low",
        "raw_mid",
        "raw_high",
        "seen_mid",
        "seen_light",
        "seen_heavy",
        "seen_very_heavy",
        "ambiguity_soft",
        "ambiguity_primary",
        "ambiguity_hard",
        "ambiguity_raw",
        "z0975",
        "z095",
        "z0925",
        "z090",
        "anchor_mid_tight",
        "anchor_tighter",
        "anchor_mid_loose",
        "anchor_looser",
        "rate_mean_low",
        "rate_mean_high",
        "resample_low",
        "resample_high",
    ),
    "z_state": (
        "base",
        "z0975",
        "z095",
        "z0925",
        "z090",
        "smooth_stiff",
        "rough_rate_light",
        "wide_init",
        "jump_rare_big",
    ),
    "missing": (
        "base",
        "missing_balanced",
        "missing_tight",
        "missing_loose",
        "smooth_stiff",
        "jump_rare_big",
        "jump_rare_soft",
        "wide_init",
    ),
    "anchor": (
        "base",
        "anchor_mid_tight",
        "anchor_tighter",
        "anchor_mid_loose",
        "anchor_looser",
        "smooth_stiff",
        "rough_rate_light",
        "jump_rare_big",
        "tight_init",
    ),
    "wide32": (
        "base",
        "smooth_stiff",
        "smooth_stiff_light",
        "smooth_stiff_heavy",
        "ultra_stiff",
        "rough_rate",
        "rough_rate_light",
        "rough_rate_heavy",
        "mobile_rate",
        "wide_init",
        "wide_init_light",
        "wide_init_heavy",
        "very_wide_init",
        "tight_init",
        "tight_init_stiff",
        "jump_rare_big",
        "jump_rare_mid",
        "jump_rare_soft",
        "jump_medium",
        "missing_balanced",
        "missing_tight",
        "missing_loose",
        "raw_very_low",
        "raw_low",
        "raw_mid",
        "raw_high",
        "seen_mid",
        "seen_light",
        "seen_heavy",
        "seen_very_heavy",
        "ambiguity_soft",
        "ambiguity_primary",
        "ambiguity_hard",
        "ambiguity_raw",
        "z0975",
        "z095",
        "z0925",
        "z090",
        "anchor_mid_tight",
        "anchor_tighter",
        "anchor_mid_loose",
        "anchor_looser",
        "rate_mean_low",
        "rate_mean_high",
        "resample_low",
        "resample_high",
    ),
    "jump_tail_global": (
        "base",
        "wide_init",
        "wide_init_light",
        "smooth_stiff",
        "smooth_stiff_light",
        "smooth_stiff_heavy",
        "ultra_stiff",
        "jump_rare_big",
        "jump_rare_mid",
        "jump_rare_soft",
        "tight_init",
        "tight_init_stiff",
        "rough_rate_light",
        "rate_mean_low",
        "rate_mean_high",
        "resample_low",
        "resample_high",
        "missing_balanced",
        "missing_tight",
        "missing_loose",
    ),
    "all": (
        "base",
        "smooth_stiff",
        "smooth_stiff_light",
        "smooth_stiff_heavy",
        "ultra_stiff",
        "rough_rate",
        "rough_rate_light",
        "rough_rate_heavy",
        "mobile_rate",
        "wide_init",
        "wide_init_light",
        "wide_init_heavy",
        "very_wide_init",
        "tight_init",
        "tight_init_stiff",
        "jump_rare_big",
        "jump_rare_mid",
        "jump_rare_soft",
        "jump_medium",
        "missing_balanced",
        "missing_tight",
        "missing_loose",
        "raw_very_low",
        "raw_low",
        "raw_mid",
        "raw_high",
        "seen_mid",
        "seen_light",
        "seen_heavy",
        "seen_very_heavy",
        "ambiguity_soft",
        "ambiguity_primary",
        "ambiguity_hard",
        "ambiguity_raw",
        "z0975",
        "z095",
        "z0925",
        "z090",
        "anchor_mid_tight",
        "anchor_tighter",
        "anchor_mid_loose",
        "anchor_looser",
        "rate_mean_low",
        "rate_mean_high",
        "resample_low",
        "resample_high",
    ),
}


def _make_profile_mixture_from_archetypes(profile_names, weights):
    if len(profile_names) != len(weights):
        raise ValueError("profile_names and weights must have the same length")
    profiles = []
    for name, weight in zip(profile_names, weights):
        profiles.append(_pf_profile(str(name), float(weight), **PF_PROFILE_ARCHETYPES[str(name)]))
    return profiles


def _profile_pool_from_mixture(source_name, source_prior, profiles, global_overrides=None):
    source_name = str(source_name)
    source_prior = float(source_prior)
    if source_prior <= 0.0:
        return []
    global_overrides = _normalize_profile_override_dict(global_overrides or {})
    out = []
    for profile in _normalize_profile_mixture_spec(profiles):
        overrides = dict(global_overrides)
        overrides.update(profile["overrides"])
        out.append(
            {
                "name": f"{source_name}__{profile['name']}",
                "prior": source_prior * float(profile["weight"]),
                "overrides": dict(sorted(overrides.items())),
            }
        )
    return out


def _merge64_profile_sampling_pool():
    pool = []
    # Accepted filtered and merge SOTA lineage. Keep the active pm0002 family as
    # the center of mass, then let source priors perturb it rather than replacing
    # it with one symmetric broad mixture.
    pool.extend(_profile_pool_from_mixture("m64_sota_pm0002", 0.30, _pm0002_accepted_profiles()))
    pool.extend(
        _profile_pool_from_mixture(
            "m16_sota_pm0002",
            0.12,
            _pm0002_accepted_profiles(),
            {
                "PF_heatmap_resample_obs_power_adapt": 0.45,
                "PF_heatmap_resample_min_threshold": 0.30,
                "PF_heatmap_finite_run_power_floor": 0.70,
            },
        )
    )
    pool.extend(
        _profile_pool_from_mixture(
            "rs0026_filtered",
            0.14,
            _pm0002_accepted_profiles(),
            {
                "PF_heatmap_raw_ref_likelihood_weight": 0.10,
                "PF_heatmap_seen_blend_weight": 0.75,
                "PF_heatmap_finite_run_power_decay": 0.25,
                "PF_heatmap_finite_run_power_floor": 0.55,
                "PF_heatmap_momentum": 0.99945,
                "PF_heatmap_pos_noise": 0.0035,
                "PF_heatmap_rate_noise": 0.0009,
                "PF_heatmap_jump_prob": 0.00015,
                "PF_heatmap_jump_sd": 7.0,
            },
        )
    )
    pool.extend(
        _profile_pool_from_mixture(
            "fs0003_ffbsi_more_smooth",
            0.11,
            PF_OPT_FFBSI_PROFILE_PLANS["more_smooth"],
            {
                "PF_heatmap_lik_scale": 6.0,
                "PF_heatmap_pos_noise": 0.0060,
                "PF_heatmap_jump_sd": 9.0,
                "PF_heatmap_ffbsi_transition_scale": 1.35,
                "PF_heatmap_ffbsi_pos_floor": 0.15,
            },
        )
    )
    pool.extend(
        _profile_pool_from_mixture(
            "ffbsi_more_wide",
            0.045,
            PF_OPT_FFBSI_PROFILE_PLANS["more_wide"],
            {
                "PF_heatmap_lik_scale": 6.0,
                "PF_heatmap_pos_noise": 0.0060,
                "PF_heatmap_ffbsi_transition_scale": 1.50,
                "PF_heatmap_ffbsi_pos_floor": 0.15,
            },
        )
    )
    pool.extend(
        _profile_pool_from_mixture(
            "ffbsi_more_jump",
            0.045,
            PF_OPT_FFBSI_PROFILE_PLANS["more_jump"],
            {
                "PF_heatmap_lik_scale": 6.0,
                "PF_heatmap_jump_sd": 10.0,
                "PF_heatmap_ffbsi_transition_scale": 1.50,
                "PF_heatmap_ffbsi_pos_floor": 0.15,
            },
        )
    )
    pool.extend(_profile_pool_from_mixture("pm0002_8x_dyn_micro", 0.08, PF_OPT_PROFILE_MIXTURES["profile_pm0002_8x_dyn_micro"]))
    if "profile_pm0002_gfrdecay025_all" in PF_OPT_PROFILE_MIXTURES:
        pool.extend(_profile_pool_from_mixture("gfr025", 0.045, PF_OPT_PROFILE_MIXTURES["profile_pm0002_gfrdecay025_all"]))
    if "profile_pm0002_gfrdecay050_raw_all" in PF_OPT_PROFILE_MIXTURES:
        pool.extend(_profile_pool_from_mixture("gfr050_raw", 0.04, PF_OPT_PROFILE_MIXTURES["profile_pm0002_gfrdecay050_raw_all"]))
    if "profile_pm0002_gfrdecay100_all" in PF_OPT_PROFILE_MIXTURES:
        pool.extend(_profile_pool_from_mixture("gfr100", 0.025, PF_OPT_PROFILE_MIXTURES["profile_pm0002_gfrdecay100_all"]))

    # Low-prior targeted rescue profiles from historical negative/near-neutral
    # domains. They stay available for recombination without letting broad hooks
    # dominate the search.
    rescue_names = (
        "rough_rate_heavy",
        "mobile_rate",
        "very_wide_init",
        "jump_medium",
        "missing_loose",
        "missing_tight",
        "raw_high",
        "seen_very_heavy",
        "ambiguity_soft",
        "z0975",
        "z095",
        "anchor_mid_tight",
        "anchor_mid_loose",
        "resample_low",
        "resample_high",
    )
    rescue_weight = 0.05 / float(len(rescue_names))
    for name in rescue_names:
        pool.extend(_profile_pool_from_mixture(f"rescue_{name}", rescue_weight, [_pf_profile(name, 1.0, **PF_PROFILE_ARCHETYPES[name])]))

    priors = np.asarray([float(item["prior"]) for item in pool], dtype=np.float64)
    if not pool or not np.isfinite(priors).all() or priors.sum() <= 0.0:
        raise RuntimeError("merge64 profile sampling pool is empty or invalid")
    return pool


def _sample_merge64_profile_mixture(rng, n_slots=32):
    pool = _merge64_profile_sampling_pool()
    priors = np.asarray([float(item["prior"]) for item in pool], dtype=np.float64)
    priors /= priors.sum()
    picked = rng.choice(np.arange(len(pool), dtype=np.int64), size=int(n_slots), replace=True, p=priors)
    profiles = []
    for slot_idx, pool_idx in enumerate(picked.tolist()):
        item = pool[int(pool_idx)]
        profiles.append(
            {
                "name": f"s{slot_idx:02d}_{item['name']}",
                "weight": 1.0,
                "overrides": dict(item["overrides"]),
            }
        )
    return _normalize_profile_mixture_spec(profiles)


def _sample_global_profile_component(rng, direction):
    direction = str(direction)
    if direction == "jump_tail_global":
        label = str(
            rng.choice(
                [
                    "jmix_data_lap",
                    "jmix_small_freq",
                    "jmix_tail_mid",
                    "jmix_tail_rare_big",
                    "jmix_tail_very_rare_extreme",
                    "jmix_cauchy_mid",
                    "jmix_aggressive",
                    "jmix_rate_tail",
                    "jmix_pos_tail",
                ],
                p=[0.18, 0.14, 0.16, 0.12, 0.06, 0.08, 0.08, 0.10, 0.08],
            )
        )
        return label, _global_obs_component(label)

    force_obs = direction.startswith("obs_global")
    labels = []
    overrides = {}

    def add_component(label):
        labels.append(str(label))
        overrides.update(_global_obs_component(str(label)))

    obs_mode = "none"
    if force_obs or rng.random() < 0.58:
        obs_mode = str(
            rng.choice(
                [
                    "out",
                    "dyn",
                    "info",
                    "out_dyn",
                    "out_info",
                    "dyn_info",
                    "mild_stack",
                ],
                p=[0.18, 0.22, 0.19, 0.16, 0.10, 0.10, 0.05],
            )
        )

    if obs_mode == "out":
        add_component(rng.choice(["out005", "out010"], p=[0.70, 0.30]))
    elif obs_mode == "dyn":
        add_component(rng.choice(["dyn010", "dyn020"], p=[0.75, 0.25]))
    elif obs_mode == "info":
        add_component(rng.choice(["info025", "info035"], p=[0.70, 0.30]))
    elif obs_mode == "out_dyn":
        add_component(rng.choice(["out005", "out010"], p=[0.75, 0.25]))
        add_component(rng.choice(["dyn010", "dyn020"], p=[0.80, 0.20]))
    elif obs_mode == "out_info":
        add_component(rng.choice(["out005", "out010"], p=[0.75, 0.25]))
        add_component(rng.choice(["info025", "info035"], p=[0.75, 0.25]))
    elif obs_mode == "dyn_info":
        add_component(rng.choice(["dyn010", "dyn020"], p=[0.80, 0.20]))
        add_component(rng.choice(["info025", "info035"], p=[0.75, 0.25]))
    elif obs_mode == "mild_stack":
        add_component("out005")
        add_component("dyn010")
        add_component("info025")

    cal_prob = 0.28 if direction in {"calibration", "calib_micro", "obs_global_wide"} else 0.14
    if rng.random() < cal_prob:
        add_component(rng.choice(["raw020", "raw025"], p=[0.65, 0.35]))
    amb_prob = 0.22 if direction in {"calibration", "calib_micro", "obs_global_wide"} else 0.10
    if rng.random() < amb_prob:
        add_component("amb035")
    if rng.random() < 0.08:
        add_component("z095")

    if not labels:
        return "gbase", {}
    uniq_labels = []
    seen = set()
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        uniq_labels.append(label)
    label = "g" + "+".join(uniq_labels[:4])
    if len(uniq_labels) > 4:
        label += f"+n{len(uniq_labels)}"
    return label, overrides


PF_OPT_FFBSI_RANDOM_SPACE = {
    "PF_heatmap_ffbsi_n_paths": [32, 32, 64, 64, 96],
    "PF_heatmap_ffbsi_max_active_bins": [512, 512, 768],
    "PF_heatmap_ffbsi_transition_scale": [1.15, 1.20, 1.35, 1.35, 1.50, 1.50, 1.65],
    "PF_heatmap_ffbsi_pos_floor": [0.10, 0.15, 0.15, 0.20, 0.25, 0.35],
    "PF_heatmap_ffbsi_rate_floor": [0.0018, 0.0025, 0.0025, 0.0040],
    "PF_heatmap_lik_scale": [5.5, 6.0, 6.0, 6.5, 7.0],
    "PF_heatmap_raw_ref_likelihood_weight": [0.10, 0.15, 0.15, 0.20, 0.25],
    "PF_heatmap_seen_blend_weight": [0.65, 0.70, 0.70, 0.80],
    "PF_heatmap_finite_run_power_boost": [0.020, 0.030, 0.030, 0.040],
    "PF_heatmap_finite_run_power_decay": [0.0, 0.25, 0.50, 1.00],
    "PF_heatmap_finite_run_power_floor": [0.50, 0.55, 0.60, 0.70],
    "PF_heatmap_pos_noise": [0.0050, 0.0060, 0.0060, 0.0070],
    "PF_heatmap_rate_noise": [0.0008, 0.0011, 0.0011, 0.0014],
    "PF_heatmap_rate_mean_weight": [0.0010, 0.00125, 0.00125, 0.0015],
    "PF_heatmap_rough_pos": [0.06, 0.08, 0.08, 0.10],
    "PF_heatmap_rough_rate": [0.0045, 0.0060, 0.0060, 0.0075],
    "PF_heatmap_momentum": [0.99935, 0.99940, 0.99940, 0.99950],
    "PF_heatmap_jump_prob": [0.000075, 0.00015, 0.00015, 0.00020],
    "PF_heatmap_jump_sd": [8.0, 9.0, 9.0, 10.0, 11.0],
    "PF_heatmap_jump_rate_sd": [0.00030, 0.00040, 0.00040, 0.00060],
    "PF_heatmap_missing_jump_boost": [0.000025, 0.00005, 0.00005, 0.000075],
    "PF_heatmap_gr_ambiguity_power": [0.0, 0.0, 0.25],
    "PF_heatmap_gr_ambiguity_min_power": [0.45, 0.55],
    "PF_heatmap_outlier_prob": [0.0, 0.0, 0.005],
    "PF_heatmap_dynamic_sigma_alpha": [0.0, 0.0, 0.010],
}


PF_OPT_FFBSI_KEY_DIMS = (
    "PF_heatmap_ffbsi_transition_scale",
    "PF_heatmap_ffbsi_pos_floor",
    "PF_heatmap_ffbsi_rate_floor",
    "PF_heatmap_ffbsi_n_paths",
    "PF_heatmap_profile_mixture_spec",
)


PF_OPT_FFBSI_SIDE_DIMS = (
    "PF_heatmap_lik_scale",
    "PF_heatmap_finite_run_power_boost",
    "PF_heatmap_finite_run_power_decay",
    "PF_heatmap_pos_noise",
    "PF_heatmap_rate_noise",
    "PF_heatmap_rate_mean_weight",
    "PF_heatmap_rough_pos",
    "PF_heatmap_rough_rate",
    "PF_heatmap_momentum",
    "PF_heatmap_jump_prob",
    "PF_heatmap_jump_sd",
    "PF_heatmap_jump_rate_sd",
    "PF_heatmap_missing_jump_boost",
    "PF_heatmap_raw_ref_likelihood_weight",
    "PF_heatmap_seen_blend_weight",
    "PF_heatmap_gr_ambiguity_power",
    "PF_heatmap_outlier_prob",
    "PF_heatmap_dynamic_sigma_alpha",
)


def _random_profile_mixture_candidates(round_idx, current_overrides, n_candidates=32):
    rng = np.random.default_rng(20260613 + 6151 * int(round_idx))
    candidates = []
    seen = set()
    attempts = 0
    profile_plan_names = tuple(PF_OPT_PROFILE_MIXTURES)
    space = PF_OPT_RANDOM_SEARCH_SPACE
    while len(candidates) < int(n_candidates) and attempts < int(n_candidates) * 120:
        attempts += 1
        overrides = dict(current_overrides)

        changed = []
        n_key_changes = int(rng.choice([2, 3, 4], p=[0.50, 0.40, 0.10]))
        changed.extend(rng.choice(PF_OPT_RANDOM_KEY_DIMS, size=n_key_changes, replace=False).tolist())
        n_side_changes = int(rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15]))
        changed.extend(rng.choice(PF_OPT_RANDOM_SIDE_DIMS, size=n_side_changes, replace=False).tolist())
        if (
            "PF_heatmap_finite_run_power_decay" not in changed
            and len(candidates) % 2 == 0
            and rng.random() < 0.85
        ):
            changed.append("PF_heatmap_finite_run_power_decay")
        changed = sorted(set(changed))

        profile_label = "pm0002"
        for key in changed:
            if key == "PF_heatmap_profile_mixture_spec":
                profile_label = str(rng.choice(profile_plan_names))
                overrides[key] = PF_OPT_PROFILE_MIXTURES[profile_label]
                continue
            values = space[key]
            cur_value = overrides.get(key, None)
            for _ in range(8):
                value = values[int(rng.integers(0, len(values)))]
                if value != cur_value:
                    overrides[key] = value
                    break
            else:
                overrides[key] = values[int(rng.integers(0, len(values)))]

        ambiguity_power = float(overrides.get("PF_heatmap_gr_ambiguity_power", 0.0))
        if ambiguity_power > 0.0 and "PF_heatmap_gr_ambiguity_min_power" not in changed:
            overrides["PF_heatmap_gr_ambiguity_min_power"] = float(rng.choice(space["PF_heatmap_gr_ambiguity_min_power"]))
        finite_run_decay = float(overrides.get("PF_heatmap_finite_run_power_decay", 0.0))
        if finite_run_decay > 0.0 and "PF_heatmap_finite_run_power_floor" not in changed:
            overrides["PF_heatmap_finite_run_power_floor"] = float(
                rng.choice(_pf_opt_finite_run_floor_values(finite_run_decay))
            )
        elif finite_run_decay > 0.0:
            floor_values = _pf_opt_finite_run_floor_values(finite_run_decay)
            floor = float(overrides.get("PF_heatmap_finite_run_power_floor", floor_values[0]))
            if floor not in floor_values:
                overrides["PF_heatmap_finite_run_power_floor"] = float(floor_values[0])
        outlier_prob = float(overrides.get("PF_heatmap_outlier_prob", 0.0))
        if outlier_prob > 0.0:
            overrides["PF_heatmap_outlier_likelihood"] = 0.05
        dynamic_sigma_alpha = float(overrides.get("PF_heatmap_dynamic_sigma_alpha", 0.0))
        if dynamic_sigma_alpha > 0.0:
            overrides.setdefault("PF_heatmap_dynamic_sigma_threshold", 1.35)
            overrides.setdefault("PF_heatmap_dynamic_sigma_power", 0.60)
            overrides.setdefault("PF_heatmap_dynamic_sigma_min", 0.90)
            overrides.setdefault("PF_heatmap_dynamic_sigma_max", 1.75)

        overrides = _clean_pf_random_overrides(overrides)
        payload = json.dumps(_normalize_profile_mixture_spec(overrides.get("PF_heatmap_profile_mixture_spec")), sort_keys=True)
        payload += json.dumps({k: v for k, v in overrides.items() if k != "PF_heatmap_profile_mixture_spec"}, sort_keys=True)
        if payload in seen:
            continue
        seen.add(payload)
        name = f"fp{int(round_idx):04d}_{profile_label}_{len(candidates):02d}"
        candidates.append((name, overrides))
    if len(candidates) != int(n_candidates):
        raise RuntimeError(f"only generated {len(candidates)} unique filtered profile candidates")
    return candidates


PF_OPT_DESIGN_CANDIDATES = (
    (
        "design_amb_p050_m035",
        {"PF_heatmap_gr_ambiguity_power": 0.50, "PF_heatmap_gr_ambiguity_min_power": 0.35},
        "Temper finite-GR likelihood when local Typewell GR has repeated motifs.",
    ),
    (
        "design_amb_p075_m050",
        {"PF_heatmap_gr_ambiguity_power": 0.75, "PF_heatmap_gr_ambiguity_min_power": 0.50},
        "Stronger ambiguity gate but retain at least half observation power.",
    ),
    (
        "design_amb_raw_p050",
        {
            "PF_heatmap_gr_ambiguity_power": 0.50,
            "PF_heatmap_gr_ambiguity_min_power": 0.35,
            "PF_heatmap_gr_ambiguity_ref_mode": "raw",
        },
        "Use raw paired Typewell rather than seen-blended curve to measure motif ambiguity.",
    ),
    (
        "design_frdecay025_floor060",
        {
            "PF_heatmap_finite_run_power_decay": 0.25,
            "PF_heatmap_finite_run_power_floor": 0.60,
        },
        "Downweight long contiguous finite-GR runs to avoid treating correlated 1 ft rows as independent.",
    ),
    (
        "design_frdecay050_floor055",
        {
            "PF_heatmap_finite_run_power_decay": 0.50,
            "PF_heatmap_finite_run_power_floor": 0.55,
        },
        "Medium finite-run ESS tempering while preserving at least half-strength GR evidence.",
    ),
    (
        "design_frdecay100_floor050",
        {
            "PF_heatmap_finite_run_power_decay": 1.00,
            "PF_heatmap_finite_run_power_floor": 0.50,
        },
        "Stronger finite-run ESS tempering for wells with very long uninterrupted GR stretches.",
    ),
    (
        "design_frdecay050_noboost",
        {
            "PF_heatmap_finite_run_power_boost": 0.0,
            "PF_heatmap_finite_run_power_decay": 0.50,
            "PF_heatmap_finite_run_power_floor": 0.55,
        },
        "Replace the old finite-run boost with ESS-style tempering rather than stacking both effects.",
    ),
    (
        "design_frdecay050_amb035",
        {
            "PF_heatmap_finite_run_power_decay": 0.50,
            "PF_heatmap_finite_run_power_floor": 0.55,
            "PF_heatmap_gr_ambiguity_power": 0.35,
            "PF_heatmap_gr_ambiguity_min_power": 0.45,
        },
        "Combine run-level correlation tempering with motif-ambiguity point tempering.",
    ),
    (
        "design_rawmix025",
        {"PF_heatmap_raw_ref_likelihood_weight": 0.25},
        "Mix raw paired-Typewell likelihood with seen-prefix blended likelihood.",
    ),
    (
        "design_rawmix035_amb",
        {
            "PF_heatmap_raw_ref_likelihood_weight": 0.35,
            "PF_heatmap_gr_ambiguity_power": 0.50,
            "PF_heatmap_gr_ambiguity_min_power": 0.45,
        },
        "Combine raw-reference likelihood diversity with ambiguity tempering.",
    ),
    (
        "design_seenblend055",
        {"PF_heatmap_seen_blend_weight": 0.55},
        "Weaken same-well pseudo-Typewell replacement while keeping calibration active.",
    ),
    (
        "design_seenblend085",
        {"PF_heatmap_seen_blend_weight": 0.85},
        "Strengthen same-well pseudo-Typewell texture inside supported TVT range.",
    ),
    (
        "design_zweight090",
        {"PF_heatmap_state_z_weight": 0.90},
        "Partial direct-TVT dynamics while retaining most notebook TVT+Z behavior.",
    ),
    (
        "design_zweight085_rawmix",
        {"PF_heatmap_state_z_weight": 0.85, "PF_heatmap_raw_ref_likelihood_weight": 0.25},
        "Complement TVT+Z dynamics with a mild direct-TVT state and raw-reference likelihood.",
    ),
    (
        "design_all_mild",
        {
            "PF_heatmap_gr_ambiguity_power": 0.50,
            "PF_heatmap_gr_ambiguity_min_power": 0.45,
            "PF_heatmap_raw_ref_likelihood_weight": 0.25,
            "PF_heatmap_state_z_weight": 0.90,
        },
        "Mild structural stack: ambiguity gate, raw reference mixture, and partial direct-TVT dynamics.",
    ),
)


def _pf_opt_project_root():
    return Path(__file__).resolve().parents[1]


def _append_progress(message):
    PF_OPT_PROGRESS_PATH.read_text()
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with PF_OPT_PROGRESS_PATH.open("a") as f:
        f.write(f"\n[{timestamp}] {message.rstrip()}\n")


def _update_active_best_progress(text):
    body = PF_OPT_PROGRESS_PATH.read_text()
    if "## Active Best" not in body or "## Round Log" not in body:
        _append_progress("Active best update:\n" + text.rstrip())
        return
    start = body.index("## Active Best")
    end = body.index("## Round Log")
    replacement = f"## Active Best\n\n{text.rstrip()}\n\n"
    PF_OPT_PROGRESS_PATH.write_text(body[:start] + replacement + body[end:])


def _discover_well_ids(data_path):
    data_path = Path(data_path)
    return sorted(path.name.split("__horizontal_well.csv")[0] for path in data_path.glob("*__horizontal_well.csv"))


def _sample_pf_opt_wells(all_well_ids, sample_size, round_idx, seed_base=20260605, exclude_well_ids=None):
    all_well_ids = list(all_well_ids)
    if exclude_well_ids:
        excluded = {str(well_id) for well_id in exclude_well_ids}
        pool = [well_id for well_id in all_well_ids if str(well_id) not in excluded]
        if len(pool) >= min(int(sample_size or len(all_well_ids)), len(all_well_ids)):
            all_well_ids = pool
    if sample_size is None or int(sample_size) >= len(all_well_ids):
        return all_well_ids
    rng = np.random.default_rng(int(seed_base) + 1009 * int(round_idx))
    picked = rng.choice(np.asarray(all_well_ids, dtype=object), size=int(sample_size), replace=False)
    return sorted(str(well_id) for well_id in picked.tolist())


def _confirm_pf_opt_candidate_on_second_shard(
    round_label,
    current_name,
    current_overrides,
    candidate_name,
    candidate_overrides,
    all_wells,
    sample_size,
    round_idx,
    workers,
    data_path,
    n_particles,
    n_seeds,
    gate_margin=0.01,
    seed_base=20260623,
    exclude_well_ids=None,
    objective_metric="direct",
):
    n_particles, n_seeds = _pf_opt_candidate_budget(
        candidate_overrides,
        default_n_particles=int(n_particles),
        default_n_seeds=int(n_seeds),
    )
    confirm_wells = _sample_pf_opt_wells(
        all_wells,
        sample_size,
        round_idx=round_idx,
        seed_base=seed_base,
        exclude_well_ids=exclude_well_ids,
    )
    baseline = benchmark_pf_opt_candidate(
        f"{round_label}_confirm_baseline_{current_name}",
        current_overrides,
        workers=workers,
        data_path=data_path,
        progress=True,
        well_ids=confirm_wells,
        n_particles=int(n_particles),
        n_seeds=int(n_seeds),
        enforce_budget=(int(n_particles), int(n_seeds)),
    )
    candidate = benchmark_pf_opt_candidate(
        f"{round_label}_confirm_{candidate_name}",
        candidate_overrides,
        workers=workers,
        data_path=data_path,
        progress=True,
        well_ids=confirm_wells,
        n_particles=int(n_particles),
        n_seeds=int(n_seeds),
        enforce_budget=(int(n_particles), int(n_seeds)),
    )
    _, gains = _pf_opt_passes_metric_gate(
        baseline,
        candidate,
        global_margin=0.0,
        aux_margin=0.0,
        objective_metric=objective_metric,
    )
    ok = (
        float(gains["global"]) >= 0.5 * float(gate_margin)
        and float(gains["well_mean"]) >= 0.0
        and float(gains["well_trim90"]) >= 0.0
    )
    return {
        "ok": bool(ok),
        "baseline": baseline,
        "candidate": candidate,
        "gains": gains,
        "well_count": len(confirm_wells),
        "seed_base": int(seed_base),
        "overlap_count": len(set(str(well_id) for well_id in confirm_wells) & {str(well_id) for well_id in (exclude_well_ids or [])}),
    }


def _make_pf_opt_cfg(data_path=None, overrides=None, n_particles=1000, n_seeds=64, workers=8):
    try:
        from seq_NN_cfg import CFG

        cfg = CFG()
    except Exception:
        class _Cfg:
            project_root = _pf_opt_project_root()
            train_path = project_root / "data" / "train"
            test_path = project_root / "data" / "test"
            downsample = 64
            prefix_len = 1024
            target_len = 10000
            raw_len = prefix_len + target_len
            num_bins = raw_len // downsample + 1
            typewell_window = 100.0
            typewell_len = 400
            PF_heatmap_cache_dir = project_root / "PF_cache"
            PF_heatmap_n_particles = 1000
            PF_heatmap_n_seeds = 64
            PF_heatmap_base_seed = 202605
            PF_heatmap_lik_scale = 5.0
            PF_heatmap_axis_sigma = 0.0
            PF_heatmap_ref_grid_step = 0.2
            PF_heatmap_num_workers = 8
            PF_heatmap_version = 6
            unet_static_channels = ("pf_prob",)

            def refresh(self):
                self.raw_len = self.prefix_len + self.target_len
                self.num_bins = self.raw_len // self.downsample + 1
                return self

        cfg = _Cfg()
    if data_path is not None:
        cfg.train_path = Path(data_path)
    cfg.PF_heatmap_n_particles = int(n_particles)
    cfg.PF_heatmap_n_seeds = int(n_seeds)
    cfg.PF_heatmap_num_workers = int(workers)
    cfg.unet_static_channels = tuple(set(getattr(cfg, "unet_static_channels", getattr(cfg, "stage_unet_static_channels", ()))) | {"pf_prob"})
    runtime_overrides = _pf_opt_runtime_overrides(overrides)
    if runtime_overrides:
        for key, value in runtime_overrides.items():
            setattr(cfg, key, value)
    if hasattr(cfg, "refresh"):
        cfg.refresh()
    cfg.PF_heatmap_n_particles = int(n_particles)
    cfg.PF_heatmap_n_seeds = int(n_seeds)
    cfg.PF_heatmap_num_workers = int(workers)
    return cfg


def _benchmark_pf_opt_one(args):
    data_path, well_id, cache_cfg = args
    h_df, tw_df = _read_well_frames(data_path, well_id)
    item = _build_pf_heatmap_item(h_df, tw_df, cache_cfg, has_target=True)
    tvt0 = float(item["tvt0"])
    mask = item["target_mask"] & np.isfinite(item["window_tvt"])
    anchor_sse = float(np.square(item["window_tvt"][mask] - np.float32(tvt0)).sum()) if mask.any() else math.nan
    gr = h_df["GR"].to_numpy(dtype=np.float64)
    finite_seen = np.flatnonzero(np.isfinite(h_df["TVT_input"].to_numpy(dtype=np.float64)))
    suffix_start = int(finite_seen[-1]) + 1
    kept_suffix_len = min(len(h_df) - suffix_start, int(cache_cfg["target_len"]))
    suffix_gr = gr[suffix_start : suffix_start + kept_suffix_len]
    return {
        "well_id": well_id,
        "target_count": int(item["target_count"]),
        "tvt_sse": float(item["tvt_sse"]),
        "tvt_rmse": float(item["tvt_rmse"]),
        "target_count_ffbsi": int(item["target_count_ffbsi"]),
        "tvt_sse_ffbsi": float(item["tvt_sse_ffbsi"]),
        "tvt_rmse_ffbsi": float(item["tvt_rmse_ffbsi"]),
        "anchor_sse": anchor_sse,
        "anchor_count": int(mask.sum()),
        "suffix_gr_missing_frac": float(np.mean(~np.isfinite(suffix_gr))) if suffix_gr.size else math.nan,
        "prob_row_sum_min": float(item["prob_row_sum_min"]),
        "prob_row_sum_max": float(item["prob_row_sum_max"]),
        "prob_row_sum_min_ffbsi": float(item["prob_row_sum_min_ffbsi"]),
        "prob_row_sum_max_ffbsi": float(item["prob_row_sum_max_ffbsi"]),
    }


def _summarize_pf_opt_results(name, overrides, cache_cfg, results, elapsed):
    target_count = int(sum(r["target_count"] for r in results))
    sse = float(sum(r["tvt_sse"] for r in results if np.isfinite(r["tvt_sse"])))
    rmse = math.sqrt(sse / target_count) if target_count > 0 else math.nan
    target_count_ffbsi = int(sum(r["target_count_ffbsi"] for r in results))
    sse_ffbsi = float(sum(r["tvt_sse_ffbsi"] for r in results if np.isfinite(r["tvt_sse_ffbsi"])))
    rmse_ffbsi = math.sqrt(sse_ffbsi / target_count_ffbsi) if target_count_ffbsi > 0 else math.nan
    anchor_count = int(sum(r["anchor_count"] for r in results))
    anchor_sse = float(sum(r["anchor_sse"] for r in results if np.isfinite(r["anchor_sse"])))
    anchor_rmse = math.sqrt(anchor_sse / anchor_count) if anchor_count > 0 else math.nan
    well_rmse = np.asarray([r["tvt_rmse"] for r in results if np.isfinite(r["tvt_rmse"])], dtype=np.float64)
    well_rmse_ffbsi = np.asarray([r["tvt_rmse_ffbsi"] for r in results if np.isfinite(r["tvt_rmse_ffbsi"])], dtype=np.float64)
    miss = np.asarray([r["suffix_gr_missing_frac"] for r in results if np.isfinite(r["suffix_gr_missing_frac"])], dtype=np.float64)
    prob_min = np.asarray([r["prob_row_sum_min"] for r in results if np.isfinite(r["prob_row_sum_min"])], dtype=np.float64)
    prob_max = np.asarray([r["prob_row_sum_max"] for r in results if np.isfinite(r["prob_row_sum_max"])], dtype=np.float64)
    prob_min_ffbsi = np.asarray([r["prob_row_sum_min_ffbsi"] for r in results if np.isfinite(r["prob_row_sum_min_ffbsi"])], dtype=np.float64)
    prob_max_ffbsi = np.asarray([r["prob_row_sum_max_ffbsi"] for r in results if np.isfinite(r["prob_row_sum_max_ffbsi"])], dtype=np.float64)
    if well_rmse.size:
        keep_count = max(1, int(math.floor(0.90 * float(well_rmse.size))))
        well_rmse_trim90_mean = float(np.mean(np.sort(well_rmse)[:keep_count]))
    else:
        well_rmse_trim90_mean = math.nan
    if well_rmse_ffbsi.size:
        keep_count_ffbsi = max(1, int(math.floor(0.90 * float(well_rmse_ffbsi.size))))
        well_rmse_trim90_mean_ffbsi = float(np.mean(np.sort(well_rmse_ffbsi)[:keep_count_ffbsi]))
    else:
        well_rmse_trim90_mean_ffbsi = math.nan
    return {
        "name": name,
        "overrides": dict(overrides),
        "cache_cfg": cache_cfg,
        "well_count": len(results),
        "target_count": target_count,
        "rmse": rmse,
        "target_count_ffbsi": target_count_ffbsi,
        "rmse_ffbsi": rmse_ffbsi,
        "anchor_rmse": anchor_rmse,
        "elapsed_sec": float(elapsed),
        "sec_per_well": float(elapsed / max(len(results), 1)),
        "well_rmse_mean": float(np.mean(well_rmse)) if well_rmse.size else math.nan,
        "well_rmse_median": float(np.median(well_rmse)) if well_rmse.size else math.nan,
        "well_rmse_q75": float(np.quantile(well_rmse, 0.75)) if well_rmse.size else math.nan,
        "well_rmse_q95": float(np.quantile(well_rmse, 0.95)) if well_rmse.size else math.nan,
        "well_rmse_trim90_mean": well_rmse_trim90_mean,
        "well_rmse_mean_ffbsi": float(np.mean(well_rmse_ffbsi)) if well_rmse_ffbsi.size else math.nan,
        "well_rmse_median_ffbsi": float(np.median(well_rmse_ffbsi)) if well_rmse_ffbsi.size else math.nan,
        "well_rmse_q75_ffbsi": float(np.quantile(well_rmse_ffbsi, 0.75)) if well_rmse_ffbsi.size else math.nan,
        "well_rmse_q95_ffbsi": float(np.quantile(well_rmse_ffbsi, 0.95)) if well_rmse_ffbsi.size else math.nan,
        "well_rmse_trim90_mean_ffbsi": well_rmse_trim90_mean_ffbsi,
        "suffix_gr_missing_frac_mean": float(np.mean(miss)) if miss.size else math.nan,
        "prob_row_sum_min": float(np.min(prob_min)) if prob_min.size else math.nan,
        "prob_row_sum_max": float(np.max(prob_max)) if prob_max.size else math.nan,
        "prob_row_sum_min_ffbsi": float(np.min(prob_min_ffbsi)) if prob_min_ffbsi.size else math.nan,
        "prob_row_sum_max_ffbsi": float(np.max(prob_max_ffbsi)) if prob_max_ffbsi.size else math.nan,
    }


def benchmark_pf_opt_candidate(
    name,
    overrides=None,
    well_limit=50,
    workers=8,
    data_path=None,
    progress=True,
    well_ids=None,
    n_particles=1000,
    n_seeds=64,
    enforce_budget=(1000, 64),
):
    if njit is None:
        raise ImportError("numba is required for PF optimization benchmarks")
    overrides = dict(overrides or {})
    cfg = _make_pf_opt_cfg(
        data_path=data_path,
        overrides=overrides,
        n_particles=int(n_particles),
        n_seeds=int(n_seeds),
        workers=int(workers),
    )
    data_path = Path(data_path or cfg.train_path)
    if well_ids is None:
        well_ids = _discover_well_ids(data_path)
        if well_limit is not None:
            well_ids = well_ids[: int(well_limit)]
    else:
        well_ids = list(well_ids)
    if not well_ids:
        raise ValueError(f"no train wells found under {data_path}")
    cache_cfg = pf_heatmap_cache_config(cfg)
    if enforce_budget is not None:
        expected_particles, expected_seeds = enforce_budget
        if cache_cfg["n_particles"] != int(expected_particles) or cache_cfg["n_seeds"] != int(expected_seeds):
            raise ValueError(
                "PF optimization benchmark must use "
                f"{int(expected_particles)} particles and {int(expected_seeds)} seeds"
            )
    _warmup_pf_heatmap_numba()
    started = time.perf_counter()
    worker_args = [(data_path, well_id, cache_cfg) for well_id in well_ids]
    n_workers = max(1, min(int(workers), len(worker_args)))
    if progress:
        print(
            f"benchmark {name}: wells={len(well_ids)} workers={n_workers} "
            f"N={cache_cfg['n_particles']} seeds={cache_cfg['n_seeds']} overrides={overrides}",
            flush=True,
        )
    if n_workers == 1:
        results = [_benchmark_pf_opt_one(args) for args in tqdm(worker_args, desc=name, dynamic_ncols=True)]
    else:
        with Pool(processes=n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_benchmark_pf_opt_one, worker_args),
                    total=len(worker_args),
                    desc=name,
                    dynamic_ncols=True,
                )
            )
    summary = _summarize_pf_opt_results(name, overrides, cache_cfg, results, time.perf_counter() - started)
    if progress:
        print(_format_pf_opt_summary(summary), flush=True)
    return summary


def _pf_opt_particle_budget_combo_label(n_particles, n_seeds):
    return f"{int(n_particles)}x{int(n_seeds)}"


def _pf_opt_particle_budget_record(
    stage,
    group_label,
    budget_label,
    requested_name,
    baseline_summary,
    summary,
    n_particles,
    n_seeds,
    rank,
):
    baseline_rmse = float(baseline_summary["rmse"])
    baseline_ffbsi_rmse = float(baseline_summary.get("rmse_ffbsi", math.nan))
    rmse = float(summary["rmse"])
    rmse_ffbsi = float(summary.get("rmse_ffbsi", math.nan))
    return {
        "stage": str(stage),
        "group_label": str(group_label),
        "budget_label": str(budget_label),
        "requested_name": str(requested_name),
        "budget_total": int(int(n_particles) * int(n_seeds)),
        "n_particles": int(n_particles),
        "n_seeds": int(n_seeds),
        "rank_in_group": int(rank),
        "name": str(requested_name),
        "cache_name": summary["name"],
        "rmse": rmse,
        "rmse_ffbsi": rmse_ffbsi,
        "well_rmse_mean": float(summary["well_rmse_mean"]),
        "well_rmse_trim90_mean": float(summary["well_rmse_trim90_mean"]),
        "well_rmse_mean_ffbsi": float(summary.get("well_rmse_mean_ffbsi", math.nan)),
        "well_rmse_trim90_mean_ffbsi": float(summary.get("well_rmse_trim90_mean_ffbsi", math.nan)),
        "gain_vs_baseline": baseline_rmse - rmse,
        "gain_vs_baseline_ffbsi": baseline_ffbsi_rmse - rmse_ffbsi,
        "baseline_name": baseline_summary["name"],
        "baseline_rmse": baseline_rmse,
        "baseline_rmse_ffbsi": baseline_ffbsi_rmse,
        "baseline_well_mean": float(baseline_summary["well_rmse_mean"]),
        "baseline_well_trim90": float(baseline_summary["well_rmse_trim90_mean"]),
        "baseline_well_mean_ffbsi": float(baseline_summary.get("well_rmse_mean_ffbsi", math.nan)),
        "baseline_well_trim90_ffbsi": float(baseline_summary.get("well_rmse_trim90_mean_ffbsi", math.nan)),
        "well_count": int(summary["well_count"]),
        "sec_per_well": float(summary["sec_per_well"]),
        "anchor_rmse": float(summary["anchor_rmse"]),
        "suffix_gr_missing_frac_mean": float(summary["suffix_gr_missing_frac_mean"]),
    }


def _pf_opt_particle_budget_best_record(records):
    if not records:
        return None
    ordered = sorted(records, key=lambda row: (float(row["rmse"]), float(row["well_rmse_mean"]), float(row["well_rmse_trim90_mean"])))
    return ordered[0]


def _pf_particle_budget_center_for_n(n_particles):
    n_particles = int(n_particles)
    if n_particles <= 1000:
        return {
            "PF_heatmap_resample_threshold": 0.25,
            "PF_heatmap_resample_obs_power_adapt": 0.30,
            "PF_heatmap_resample_min_threshold": 0.125,
            "PF_heatmap_rough_pos": 0.04,
            "PF_heatmap_rough_rate": 0.0030,
        }
    if n_particles <= 2000:
        return {
            "PF_heatmap_resample_threshold": 0.16,
            "PF_heatmap_resample_obs_power_adapt": 0.25,
            "PF_heatmap_resample_min_threshold": 0.08,
            "PF_heatmap_rough_pos": 0.02,
            "PF_heatmap_rough_rate": 0.0015,
        }
    return {
        "PF_heatmap_resample_threshold": 0.10,
        "PF_heatmap_resample_obs_power_adapt": 0.15,
        "PF_heatmap_resample_min_threshold": 0.05,
        "PF_heatmap_rough_pos": 0.012,
        "PF_heatmap_rough_rate": 0.0008,
    }


def _pf_particle_budget_focus_overrides(
    center,
    threshold_mul=1.0,
    adapt_mul=1.0,
    min_mul=1.0,
    rough_pos_mul=1.0,
    rough_rate_mul=1.0,
):
    overrides = dict(center)
    overrides["PF_heatmap_resample_threshold"] = float(center["PF_heatmap_resample_threshold"] * float(threshold_mul))
    overrides["PF_heatmap_resample_obs_power_adapt"] = float(center["PF_heatmap_resample_obs_power_adapt"] * float(adapt_mul))
    overrides["PF_heatmap_resample_min_threshold"] = float(center["PF_heatmap_resample_min_threshold"] * float(min_mul))
    overrides["PF_heatmap_rough_pos"] = float(center["PF_heatmap_rough_pos"] * float(rough_pos_mul))
    overrides["PF_heatmap_rough_rate"] = float(center["PF_heatmap_rough_rate"] * float(rough_rate_mul))
    overrides["PF_heatmap_resample_threshold"] = float(np.clip(overrides["PF_heatmap_resample_threshold"], 0.05, 0.60))
    overrides["PF_heatmap_resample_obs_power_adapt"] = float(np.clip(overrides["PF_heatmap_resample_obs_power_adapt"], 0.10, 0.60))
    overrides["PF_heatmap_resample_min_threshold"] = float(
        np.clip(overrides["PF_heatmap_resample_min_threshold"], 0.025, overrides["PF_heatmap_resample_threshold"])
    )
    overrides["PF_heatmap_rough_pos"] = float(np.clip(overrides["PF_heatmap_rough_pos"], 0.005, 0.08))
    overrides["PF_heatmap_rough_rate"] = float(np.clip(overrides["PF_heatmap_rough_rate"], 0.00025, 0.010))
    return overrides


def _pf_particle_budget_grid_specs_for_n(n_particles):
    center = _pf_particle_budget_center_for_n(n_particles)
    name_prefix = f"n{int(n_particles)}"
    raw_specs = [
        ("center", {}),
        ("resample_low", {"threshold_mul": 0.80, "adapt_mul": 0.85, "min_mul": 0.80}),
        ("resample_high", {"threshold_mul": 1.20, "adapt_mul": 1.15, "min_mul": 1.10}),
        ("adapt_low", {"adapt_mul": 0.85}),
        ("adapt_high", {"adapt_mul": 1.15}),
        ("min_low", {"min_mul": 0.80}),
        ("min_high", {"min_mul": 1.20}),
        ("rough_low", {"rough_pos_mul": 0.75, "rough_rate_mul": 0.75}),
        ("rough_high", {"rough_pos_mul": 1.25, "rough_rate_mul": 1.25}),
        ("resample_low_rough_low", {"threshold_mul": 0.80, "adapt_mul": 0.85, "min_mul": 0.80, "rough_pos_mul": 0.75, "rough_rate_mul": 0.75}),
        ("resample_low_rough_high", {"threshold_mul": 0.80, "adapt_mul": 0.85, "min_mul": 0.80, "rough_pos_mul": 1.15, "rough_rate_mul": 1.15}),
        ("resample_high_rough_low", {"threshold_mul": 1.20, "adapt_mul": 1.15, "min_mul": 1.10, "rough_pos_mul": 0.75, "rough_rate_mul": 0.75}),
        ("resample_high_rough_high", {"threshold_mul": 1.20, "adapt_mul": 1.15, "min_mul": 1.10, "rough_pos_mul": 1.15, "rough_rate_mul": 1.15}),
        ("adapt_low_rough_low", {"adapt_mul": 0.85, "rough_pos_mul": 0.75, "rough_rate_mul": 0.75}),
        ("adapt_low_rough_high", {"adapt_mul": 0.85, "rough_pos_mul": 1.15, "rough_rate_mul": 1.15}),
        ("adapt_high_rough_low", {"adapt_mul": 1.15, "rough_pos_mul": 0.75, "rough_rate_mul": 0.75}),
        ("adapt_high_rough_high", {"adapt_mul": 1.15, "rough_pos_mul": 1.15, "rough_rate_mul": 1.15}),
        ("all_low", {"threshold_mul": 0.80, "adapt_mul": 0.85, "min_mul": 0.80, "rough_pos_mul": 0.75, "rough_rate_mul": 0.75}),
        ("all_mid_low", {"threshold_mul": 0.90, "adapt_mul": 0.90, "min_mul": 0.90, "rough_pos_mul": 0.85, "rough_rate_mul": 0.85}),
        ("all_high", {"threshold_mul": 1.20, "adapt_mul": 1.15, "min_mul": 1.20, "rough_pos_mul": 1.25, "rough_rate_mul": 1.25}),
    ]
    out = []
    for suffix, kwargs in raw_specs:
        out.append((f"{name_prefix}_{suffix}", _pf_particle_budget_focus_overrides(center, **kwargs)))
    return out


def _pf_particle_budget_grid_candidate_limit(n_particles):
    n_particles = int(n_particles)
    if n_particles <= 1000:
        return 8
    if n_particles <= 2000:
        return 20
    return 20


def _pf_particle_budget_search_plan_note(total_particles, total_seeds, n_particles, n_seeds, n_candidates):
    total_budget = int(total_particles) * int(total_seeds)
    n_budget = int(n_particles) * int(n_seeds)
    return (
        f"fixed total budget `{int(total_particles)}x{int(total_seeds)}={total_budget:,}`; "
        f"running `N={int(n_particles)}` with `K={int(n_seeds)}` (`{n_budget:,}`) and "
        f"{int(n_candidates)} candidate grids around the N-aware resample/roughening center."
    )


def _pf_merge_overrides(base_overrides, delta_overrides=None):
    merged = dict(base_overrides or {})
    if delta_overrides:
        merged.update(delta_overrides)
    return merged


def run_pf_particle_budget_full_grid_search(
    workers=8,
    data_path=None,
    full_well_limit=773,
    total_particles=500,
    total_seeds=128,
    n_grid=(1000, 2000, 4000),
    output_dir=None,
    baseline_base_seed=PF_OPT_CURRENT_BEST_FULL_BASE_SEED,
):
    PF_OPT_PROGRESS_PATH.read_text()
    current_overrides = _clean_pf_random_overrides(PF_OPT_CURRENT_BEST_OVERRIDES)
    current_overrides["PF_heatmap_base_seed"] = int(baseline_base_seed)
    cfg = _make_pf_opt_cfg(
        data_path=data_path,
        overrides=current_overrides,
        n_particles=int(total_particles),
        n_seeds=int(total_seeds),
        workers=int(workers),
    )
    data_path = Path(data_path or cfg.train_path)
    all_wells = _discover_well_ids(data_path)
    if not all_wells:
        raise ValueError(f"no train wells found under {data_path}")
    full_wells = all_wells if full_well_limit is None else all_wells[: int(full_well_limit)]
    if not full_wells:
        raise ValueError("no wells available for full-grid particle-budget research")

    total_particles = int(total_particles)
    total_seeds = int(total_seeds)
    total_budget = int(total_particles * total_seeds)
    n_grid = tuple(int(n) for n in n_grid)

    output_dir = Path(
        output_dir
        or (_pf_opt_project_root() / "logs" / f"pf_particle_grid_full_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    _append_progress(
        "Full-well particle-number grid search start: "
        f"baseline budget `{int(total_particles)}x{int(total_seeds)}` across all `{len(full_wells)}` wells, "
        f"baseline seed `{int(baseline_base_seed)}`, workers={int(workers)}. "
        "The search keeps raw/seen reference balance, finite-run tempering, anchor sigma/power, "
        "and likelihood scale fixed, and only moves the N-aware resample/roughening interaction."
    )
    _append_progress(
        "Full-well particle-number grid plan: "
        + ", ".join(
            _pf_particle_budget_search_plan_note(total_particles, total_seeds, n, max(total_budget // int(n), 1), _pf_particle_budget_grid_candidate_limit(n))
            for n in n_grid
        )
    )

    baseline_summary = benchmark_pf_opt_candidate(
        PF_OPT_CURRENT_BEST_FULL_NAME,
        current_overrides,
        well_ids=full_wells,
        workers=workers,
        data_path=data_path,
        progress=True,
        n_particles=int(total_particles),
        n_seeds=int(total_seeds),
        enforce_budget=(int(total_particles), int(total_seeds)),
    )
    _append_progress(
        "Baseline full-well result: "
        + _format_pf_opt_summary(baseline_summary)
        + f" using seed `{int(baseline_base_seed)}`."
    )
    _update_active_best_progress(
        f"- Active baseline: `{baseline_summary['name']}` RMSE `{baseline_summary['rmse']:.6f}` "
        f"FFBSi RMSE `{baseline_summary.get('rmse_ffbsi', math.nan):.6f}` across `{len(full_wells)}` wells.\n"
        f"- Baseline overrides: `{json.dumps(current_overrides, sort_keys=True)}`."
    )

    rows = []
    candidate_plan_rows = []
    best_by_n = {}
    all_candidate_summaries = []
    result_cache = {}

    def _run_and_cache(candidate_name, overrides, n_particles, n_seeds):
        key = (int(n_particles), int(n_seeds), json.dumps(_clean_pf_random_overrides(overrides), sort_keys=True))
        if key not in result_cache:
            result_cache[key] = benchmark_pf_opt_candidate(
                candidate_name,
                overrides,
                well_ids=full_wells,
                workers=workers,
                data_path=data_path,
                progress=True,
                n_particles=int(n_particles),
                n_seeds=int(n_seeds),
                enforce_budget=(int(n_particles), int(n_seeds)),
            )
        return result_cache[key]

    for n_particles in n_grid:
        n_seeds = int(total_budget // int(n_particles))
        if n_seeds <= 0 or int(n_particles) * int(n_seeds) != total_budget:
            _append_progress(
                f"Skipping N={int(n_particles)} because fixed budget `{int(total_particles)}x{int(total_seeds)}` "
                f"does not divide evenly into K={n_seeds}."
            )
            continue
        specs = _pf_particle_budget_grid_specs_for_n(int(n_particles))
        limit = _pf_particle_budget_grid_candidate_limit(int(n_particles))
        _append_progress(
            f"Grid slice N={int(n_particles)} K={int(n_seeds)} candidate_limit={int(limit)} "
            f"center={json.dumps(_pf_particle_budget_center_for_n(int(n_particles)), sort_keys=True)}"
        )
        for rank, (cand_name, cand_overrides) in enumerate(specs[: int(limit)]):
            applied_overrides = _pf_merge_overrides(current_overrides, cand_overrides)
            candidate_plan_rows.append(
                {
                    "n_particles": int(n_particles),
                    "n_seeds": int(n_seeds),
                    "budget_total": int(total_budget),
                    "group_label": f"n{int(n_particles)}",
                    "rank_in_group": int(rank),
                    "candidate_name": cand_name,
                    "delta_overrides": json.dumps(cand_overrides, sort_keys=True),
                    "candidate_overrides": json.dumps(applied_overrides, sort_keys=True),
                    "will_run": True,
                }
            )
            summary = _run_and_cache(cand_name, applied_overrides, int(n_particles), int(n_seeds))
            record = _pf_opt_particle_budget_record(
                stage="fixed_budget_full",
                group_label=f"n{int(n_particles)}",
                budget_label=f"{int(n_particles)}x{int(n_seeds)}",
                requested_name=cand_name,
                baseline_summary=baseline_summary,
                summary=summary,
                n_particles=int(n_particles),
                n_seeds=int(n_seeds),
                rank=rank,
            )
            rows.append(record)
            all_candidate_summaries.append(summary)
        group_records = [row for row in rows if row["group_label"] == f"n{int(n_particles)}"]
        best_record = _pf_opt_particle_budget_best_record(group_records)
        if best_record is not None:
            best_by_n[f"n{int(n_particles)}"] = best_record
            _append_progress(
                f"N={int(n_particles)} summary: best `{best_record['requested_name']}` "
                f"direct_RMSE `{best_record['rmse']:.6f}` FFBSi_RMSE `{best_record['rmse_ffbsi']:.6f}` "
                f"gain_vs_baseline `{best_record['gain_vs_baseline']:.6f}` "
                f"gain_vs_baseline_ffbsi `{best_record['gain_vs_baseline_ffbsi']:.6f}`."
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["group_label", "rmse", "well_rmse_mean", "well_rmse_trim90_mean"]).reset_index(drop=True)
    df.to_csv(output_dir / "particle_budget_full_grid_results.csv", index=False)
    pd.DataFrame(candidate_plan_rows).to_csv(output_dir / "particle_budget_full_grid_plan.csv", index=False)

    summary = {
        "recipe_name": PF_OPT_CURRENT_BEST_NAME,
        "baseline_name": baseline_summary["name"],
        "baseline_base_seed": int(baseline_base_seed),
        "baseline_overrides": current_overrides,
        "baseline_rmse": float(baseline_summary["rmse"]),
        "baseline_rmse_ffbsi": float(baseline_summary.get("rmse_ffbsi", math.nan)),
        "well_limit": int(len(full_wells)),
        "worker_count": int(workers),
        "budget_total_particles": int(total_particles),
        "budget_total_seeds": int(total_seeds),
        "budget_total": int(total_budget),
        "n_grid": [int(n) for n in n_grid],
        "candidate_limit_by_n": {str(int(n)): int(_pf_particle_budget_grid_candidate_limit(int(n))) for n in n_grid},
        "best_by_n": best_by_n,
        "result_csv": str(output_dir / "particle_budget_full_grid_results.csv"),
        "plan_csv": str(output_dir / "particle_budget_full_grid_plan.csv"),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _append_progress(
        "Full-well particle-number grid summary:\n"
        f"- Output dir: `{output_dir}`.\n"
        f"- Baseline: `{baseline_summary['name']}` direct_RMSE `{baseline_summary['rmse']:.6f}` "
        f"FFBSi_RMSE `{baseline_summary.get('rmse_ffbsi', math.nan):.6f}`.\n"
        f"- Best-by-N: `{json.dumps({k: {'candidate': v['requested_name'], 'rmse': v['rmse'], 'rmse_ffbsi': v['rmse_ffbsi']} for k, v in best_by_n.items()}, sort_keys=True)}`."
    )
    return summary


def run_pf_particle_budget_research(
    well_limit=50,
    workers=8,
    data_path=None,
    fixed_k=32,
    n_grid=(500, 1000, 2000),
    budget_bases=((500, 32), (500, 64), (500, 128)),
    output_dir=None,
):
    PF_OPT_PROGRESS_PATH.read_text()
    current_overrides = _clean_pf_random_overrides(PF_OPT_CURRENT_BEST_OVERRIDES)
    cfg = _make_pf_opt_cfg(
        data_path=data_path,
        overrides=current_overrides,
        n_particles=int(max(n_grid)),
        n_seeds=int(max(fixed_k, max(int(base_k) for _, base_k in budget_bases))),
        workers=int(workers),
    )
    data_path = Path(data_path or cfg.train_path)
    all_wells = _discover_well_ids(data_path)
    if not all_wells:
        raise ValueError(f"no train wells found under {data_path}")
    sampled_wells = all_wells[: int(well_limit)] if well_limit is not None else all_wells
    if not sampled_wells:
        raise ValueError("no wells available for particle-budget research")

    output_dir = Path(
        output_dir
        or (_pf_opt_project_root() / "logs" / f"pf_particle_budget_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_k = int(fixed_k)
    n_grid = tuple(int(n) for n in n_grid)
    budget_bases = tuple((int(base_n), int(base_k)) for base_n, base_k in budget_bases)
    started = time.perf_counter()
    _append_progress(
        "Particle-number research start: "
        f"wells={len(sampled_wells)}, current recipe `{PF_OPT_CURRENT_BEST_NAME}`, "
        f"fixed-K sweep `K={fixed_k}` over N={list(n_grid)}, "
        f"fixed-budget groups {[f'{int(n)}x{int(k)}' for n, k in budget_bases]}, "
        "all other PF knobs held fixed initially."
    )

    result_cache = {}
    rows = []

    def _run_and_cache(name, n_particles, n_seeds):
        key = (int(n_particles), int(n_seeds))
        if key not in result_cache:
            result_cache[key] = benchmark_pf_opt_candidate(
                name,
                current_overrides,
                well_limit=well_limit,
                workers=workers,
                data_path=data_path,
                progress=True,
                well_ids=sampled_wells,
                n_particles=int(n_particles),
                n_seeds=int(n_seeds),
                enforce_budget=key,
            )
        return result_cache[key]

    stage1_label = f"fixed_k_{fixed_k}"
    stage1_baseline = _run_and_cache(f"particle_{stage1_label}_n{n_grid[0]}_baseline", n_grid[0], fixed_k)
    stage1_records = []
    for rank, n_particles in enumerate(n_grid):
        summary = _run_and_cache(
            f"particle_{stage1_label}_n{int(n_particles)}_k{fixed_k}",
            int(n_particles),
            fixed_k,
        )
        record = _pf_opt_particle_budget_record(
            stage="fixed_k",
            group_label=stage1_label,
            budget_label=f"{int(n_particles)}x{fixed_k}",
            requested_name=f"particle_{stage1_label}_n{int(n_particles)}_k{fixed_k}",
            baseline_summary=stage1_baseline,
            summary=summary,
            n_particles=int(n_particles),
            n_seeds=fixed_k,
            rank=rank,
        )
        rows.append(record)
        stage1_records.append(record)

    stage1_best = _pf_opt_particle_budget_best_record(stage1_records)
    _append_progress(
        "Fixed-K sweep result: "
        f"baseline `{_pf_opt_particle_budget_combo_label(n_grid[0], fixed_k)}` "
        f"direct_RMSE `{stage1_baseline['rmse']:.6f}`; "
        f"best `{stage1_best['budget_label']}` direct_RMSE `{stage1_best['rmse']:.6f}` "
        f"(gain `{stage1_best['gain_vs_baseline']:.6f}`), "
        f"FFBSi gain `{stage1_best['gain_vs_baseline_ffbsi']:.6f}`; "
        f"top3 `{_format_brief_candidate_ranking([result_cache[(int(n), fixed_k)] for n in n_grid], limit=3)}`."
    )
    _update_active_best_progress(
        f"- Particle-budget stage 1 best: `{stage1_best['budget_label']}` direct RMSE `{stage1_best['rmse']:.6f}` "
        f"FFBSi RMSE `{stage1_best['rmse_ffbsi']:.6f}` on `{len(sampled_wells)}` wells.\n"
        f"- Fixed-K baseline: `{_pf_opt_particle_budget_combo_label(n_grid[0], fixed_k)}` direct RMSE `{stage1_baseline['rmse']:.6f}`.\n"
        f"- Current recipe: `{PF_OPT_CURRENT_BEST_NAME}` with overrides `{json.dumps(current_overrides, sort_keys=True)}`."
    )

    stage2_group_best = {}
    for base_n, base_k in budget_bases:
        total_budget = int(base_n) * int(base_k)
        group_label = f"budget_{int(base_n)}x{int(base_k)}"
        baseline_n = int(n_grid[0])
        if total_budget % baseline_n != 0:
            continue
        baseline_k = int(total_budget // baseline_n)
        baseline_summary = _run_and_cache(
            f"particle_{group_label}_n{baseline_n}_k{baseline_k}_baseline",
            baseline_n,
            baseline_k,
        )
        group_records = []
        for rank, n_particles in enumerate(n_grid):
            if total_budget % int(n_particles) != 0:
                continue
            n_seeds = int(total_budget // int(n_particles))
            summary = _run_and_cache(
                f"particle_{group_label}_n{int(n_particles)}_k{n_seeds}",
                int(n_particles),
                n_seeds,
            )
            record = _pf_opt_particle_budget_record(
                stage="fixed_budget",
                group_label=group_label,
                budget_label=f"{int(n_particles)}x{n_seeds}",
                requested_name=f"particle_{group_label}_n{int(n_particles)}_k{n_seeds}",
                baseline_summary=baseline_summary,
                summary=summary,
                n_particles=int(n_particles),
                n_seeds=n_seeds,
                rank=rank,
            )
            rows.append(record)
            group_records.append(record)
        best_record = _pf_opt_particle_budget_best_record(group_records)
        if best_record is None:
            continue
        stage2_group_best[group_label] = best_record
        _append_progress(
            f"Fixed-budget group `{group_label}` result: baseline `{_pf_opt_particle_budget_combo_label(baseline_n, baseline_k)}` "
            f"direct_RMSE `{baseline_summary['rmse']:.6f}`; best `{best_record['budget_label']}` direct_RMSE "
            f"`{best_record['rmse']:.6f}` (gain `{best_record['gain_vs_baseline']:.6f}`), "
            f"FFBSi gain `{best_record['gain_vs_baseline_ffbsi']:.6f}`; "
            f"top3 `{_format_brief_candidate_ranking([result_cache[(int(n), int(total_budget // int(n)))] for n in n_grid if total_budget % int(n) == 0], limit=3)}`."
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["stage", "group_label", "rmse", "well_rmse_mean", "well_rmse_trim90_mean"]).reset_index(drop=True)
    df.to_csv(output_dir / "particle_budget_results.csv", index=False)

    summary = {
        "recipe_name": PF_OPT_CURRENT_BEST_NAME,
        "recipe_overrides": current_overrides,
        "well_limit": int(len(sampled_wells)),
        "worker_count": int(workers),
        "fixed_k": fixed_k,
        "n_grid": [int(n) for n in n_grid],
        "budget_bases": [{"base_n": int(base_n), "base_k": int(base_k), "total_budget": int(base_n) * int(base_k)} for base_n, base_k in budget_bases],
        "stage1_best": stage1_best,
        "fixed_budget_best": stage2_group_best,
        "result_csv": str(output_dir / "particle_budget_results.csv"),
        "elapsed_sec": float(time.perf_counter() - started),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _append_progress(
        "Particle-number research summary:\n"
        f"- Output dir: `{output_dir}`.\n"
        f"- Fixed-K best: `{stage1_best['budget_label']}` direct_RMSE `{stage1_best['rmse']:.6f}` "
        f"(gain `{stage1_best['gain_vs_baseline']:.6f}` vs `{_pf_opt_particle_budget_combo_label(n_grid[0], fixed_k)}`).\n"
        f"- Fixed-budget bests: `{json.dumps({k: {'budget': v['budget_label'], 'rmse': v['rmse'], 'gain': v['gain_vs_baseline']} for k, v in stage2_group_best.items()}, sort_keys=True)}`."
    )
    return summary


@dataclass
class PFFailureWellItem:
    well_id: str
    target: np.ndarray
    pf_pred: np.ndarray
    anchor_pred: np.ndarray
    features: dict


def _rmse_np(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(target)
    if not valid.any():
        return math.nan
    diff = pred[valid] - target[valid]
    return float(math.sqrt(float(np.mean(diff * diff))))


def _sse_count_np(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(target)
    if not valid.any():
        return 0.0, 0
    diff = pred[valid] - target[valid]
    return float(np.dot(diff, diff)), int(valid.sum())


def _rmse_from_sse_count(sse, count):
    return float(math.sqrt(float(sse) / int(count))) if int(count) > 0 else math.nan


def _finite_run_lengths(mask):
    mask = np.asarray(mask, dtype=bool)
    max_true = 0
    max_false = 0
    cur_true = 0
    cur_false = 0
    for value in mask:
        if value:
            cur_true += 1
            cur_false = 0
        else:
            cur_false += 1
            cur_true = 0
        if cur_true > max_true:
            max_true = cur_true
        if cur_false > max_false:
            max_false = cur_false
    return int(max_true), int(max_false)


def _lookup_typewell_gr(tw_tvt, tw_gr, tvt):
    tvt = np.asarray(tvt, dtype=np.float64)
    out = np.full(tvt.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(tvt) & (tvt >= float(tw_tvt[0])) & (tvt <= float(tw_tvt[-1]))
    if valid.any():
        out[valid] = np.interp(tvt[valid], tw_tvt, tw_gr)
    return out


def _make_cached_pf_failure_item(cache_dir, data_path, well_id, cache_cfg):
    cache_path = pf_heatmap_file(cache_dir, well_id)
    h_df, tw_df = _read_well_frames(data_path, well_id)
    with np.load(cache_path) as data:
        target = data["window_tvt"].astype(np.float64)
        target_mask = data["target_mask"].astype(bool) & np.isfinite(target)
        if not target_mask.any():
            return None, None
        tvt0 = float(data["tvt0"])
        pf_pred_all = data["pf_tvt_pred"].astype(np.float64)
        anchor_all = np.full_like(pf_pred_all, tvt0, dtype=np.float64)
        pf_prob = data["pf_prob"].astype(np.float64)[target_mask]
        window_orig_index = data["window_orig_index"].astype(np.float64)

    target_tvt = target[target_mask]
    pf_pred = pf_pred_all[target_mask]
    anchor_pred = anchor_all[target_mask]
    row_sum = pf_prob.sum(axis=1)
    prob = pf_prob / np.clip(row_sum[:, None], 1e-12, None)
    typewell_len = int(cache_cfg["typewell_len"])
    typewell_window = float(cache_cfg["typewell_window"])
    grid_step = 2.0 * typewell_window / float(typewell_len)
    grid_rel = -typewell_window + typewell_window / float(typewell_len) + grid_step * np.arange(typewell_len, dtype=np.float64)
    prob_clip = np.clip(prob, 1e-300, None)
    entropy = -np.sum(np.where(prob > 0.0, prob * np.log(prob_clip), 0.0), axis=1)
    norm_entropy = entropy / max(math.log(float(typewell_len)), 1e-12)
    ess = 1.0 / np.maximum(np.square(prob).sum(axis=1), 1e-12)
    maxp = prob.max(axis=1)
    anchor_mass5 = prob[:, np.abs(grid_rel) <= 5.0].sum(axis=1)
    anchor_mass15 = prob[:, np.abs(grid_rel) <= 15.0].sum(axis=1)
    mode_rel = grid_rel[np.argmax(prob, axis=1)]
    mean_rel = np.sum(prob * grid_rel[None, :], axis=1)
    pred_rel = pf_pred - tvt0
    target_rel = target_tvt - tvt0

    finite_seen = np.flatnonzero(np.isfinite(h_df["TVT_input"].to_numpy(dtype=np.float64)))
    suffix_start = int(finite_seen[-1]) + 1
    kept_suffix_len = min(len(h_df) - suffix_start, int(cache_cfg["target_len"]))
    suffix_gr = h_df["GR"].to_numpy(dtype=np.float64)[suffix_start : suffix_start + kept_suffix_len]
    finite_gr = np.isfinite(suffix_gr)
    max_finite_gr_run, max_nan_gr_run = _finite_run_lengths(finite_gr)

    raw_tw = tw_df.sort_values("TVT")
    tw_tvt = raw_tw["TVT"].to_numpy(dtype=np.float64)
    tw_gr = raw_tw["GR"].fillna(raw_tw["GR"].mean()).to_numpy(dtype=np.float64)
    cal_tw_tvt, cal_tw_gr = _typewell_arrays_for_cache_cfg(h_df, tw_df, cache_cfg)

    suffix_gr_rows = h_df["GR"].to_numpy(dtype=np.float64)[suffix_start : suffix_start + kept_suffix_len]
    suffix_bin_idx = (int(cache_cfg["prefix_len"]) + np.arange(kept_suffix_len, dtype=np.int64)) // int(
        cache_cfg["downsample"]
    )
    observed_gr = np.full(target_tvt.shape, np.nan, dtype=np.float64)
    gr_sums = np.bincount(suffix_bin_idx, weights=np.where(np.isfinite(suffix_gr_rows), suffix_gr_rows, 0.0), minlength=int(cache_cfg["num_bins"]))
    gr_counts = np.bincount(suffix_bin_idx, weights=np.isfinite(suffix_gr_rows).astype(np.int64), minlength=int(cache_cfg["num_bins"]))
    gr_valid = (gr_counts > 0) & target_mask
    if gr_valid.any():
        observed_gr[gr_valid[target_mask]] = gr_sums[gr_valid] / gr_counts[gr_valid]
    finite_observed_gr = np.isfinite(observed_gr)
    if finite_observed_gr.any():
        pf_gr = _lookup_typewell_gr(cal_tw_tvt, cal_tw_gr, pf_pred)
        anchor_gr = _lookup_typewell_gr(cal_tw_tvt, cal_tw_gr, anchor_pred)
        raw_pf_gr = _lookup_typewell_gr(tw_tvt, tw_gr, pf_pred)
        pf_gr_rmse = _rmse_np(pf_gr[finite_observed_gr], observed_gr[finite_observed_gr])
        anchor_gr_rmse = _rmse_np(anchor_gr[finite_observed_gr], observed_gr[finite_observed_gr])
        raw_pf_gr_rmse = _rmse_np(raw_pf_gr[finite_observed_gr], observed_gr[finite_observed_gr])
    else:
        pf_gr_rmse = math.nan
        anchor_gr_rmse = math.nan
        raw_pf_gr_rmse = math.nan

    pf_rmse = _rmse_np(pf_pred, target_tvt)
    anchor_rmse = _rmse_np(anchor_pred, target_tvt)
    features = {
        "well_id": well_id,
        "target_bins": int(target_mask.sum()),
        "suffix_len": int(kept_suffix_len),
        "suffix_gr_missing_frac": float(np.mean(~finite_gr)) if finite_gr.size else math.nan,
        "max_finite_gr_run": int(max_finite_gr_run),
        "max_nan_gr_run": int(max_nan_gr_run),
        "finite_target_gr_frac": float(np.mean(finite_observed_gr)) if finite_observed_gr.size else math.nan,
        "pf_rmse": float(pf_rmse),
        "anchor_rmse": float(anchor_rmse),
        "pf_minus_anchor_rmse": float(pf_rmse - anchor_rmse),
        "pf_bias": float(np.mean(pf_pred - target_tvt)),
        "pf_mae": float(np.mean(np.abs(pf_pred - target_tvt))),
        "anchor_mae": float(np.mean(np.abs(anchor_pred - target_tvt))),
        "pred_rel_abs_mean": float(np.mean(np.abs(pred_rel))),
        "true_rel_abs_mean": float(np.mean(np.abs(target_rel))),
        "pred_rel_end": float(pred_rel[-1]),
        "true_rel_end": float(target_rel[-1]),
        "pred_end_abs": float(abs(pred_rel[-1])),
        "pred_step_rms": float(math.sqrt(float(np.mean(np.diff(pf_pred) ** 2)))) if pf_pred.size > 1 else 0.0,
        "true_step_rms": float(math.sqrt(float(np.mean(np.diff(target_tvt) ** 2)))) if target_tvt.size > 1 else 0.0,
        "entropy_mean": float(np.mean(norm_entropy)),
        "ess_mean": float(np.mean(ess)),
        "maxp_mean": float(np.mean(maxp)),
        "anchor_mass5_mean": float(np.mean(anchor_mass5)),
        "anchor_mass15_mean": float(np.mean(anchor_mass15)),
        "target_mass5_mean": float(np.mean([p[np.abs(grid_rel - rel) <= 5.0].sum() for p, rel in zip(prob, target_rel)])),
        "mode_target_abs_mean": float(np.mean(np.abs(mode_rel - target_rel))),
        "mode_anchor_abs_mean": float(np.mean(np.abs(mode_rel))),
        "mean_anchor_abs_mean": float(np.mean(np.abs(mean_rel))),
        "pf_gr_rmse": float(pf_gr_rmse),
        "anchor_gr_rmse": float(anchor_gr_rmse),
        "raw_pf_gr_rmse": float(raw_pf_gr_rmse),
        "pf_gr_minus_anchor_gr_rmse": float(pf_gr_rmse - anchor_gr_rmse) if np.isfinite(pf_gr_rmse) and np.isfinite(anchor_gr_rmse) else math.nan,
    }
    item = PFFailureWellItem(
        well_id=well_id,
        target=target_tvt.astype(np.float64),
        pf_pred=pf_pred.astype(np.float64),
        anchor_pred=anchor_pred.astype(np.float64),
        features=features,
    )
    return item, features


def _make_cached_pf_failure_folds(well_ids, fold_count=5, seed=7):
    well_ids = np.asarray(well_ids, dtype=str)
    fold_count = min(int(fold_count), len(well_ids))
    if fold_count <= 1:
        return np.zeros(len(well_ids), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    shuffled = np.array(well_ids, copy=True)
    rng.shuffle(shuffled)
    val_count = int(math.ceil(len(shuffled) / float(fold_count)))
    fold_by_well = {}
    for fold in range(fold_count):
        for well_id in shuffled[val_count * fold : val_count * (fold + 1)]:
            fold_by_well[str(well_id)] = int(fold)
    return np.asarray([fold_by_well[str(well_id)] for well_id in well_ids], dtype=np.int64)


def _pf_failure_rule_alpha(features, feature_name, threshold, direction, alpha_bad, alpha_good=1.0):
    value = float(features.get(feature_name, math.nan))
    if not np.isfinite(value):
        return float(alpha_good)
    if direction == "high":
        risky = value >= float(threshold)
    elif direction == "low":
        risky = value <= float(threshold)
    else:
        raise ValueError(f"unknown PF failure rule direction={direction!r}")
    return float(alpha_bad) if risky else float(alpha_good)


def _score_pf_failure_rule(items, indices, rule=None):
    sse = 0.0
    count = 0
    for idx in indices:
        item = items[int(idx)]
        if rule is None:
            pred = item.pf_pred
        else:
            alpha = float(rule(item.features))
            pred = item.anchor_pred + alpha * (item.pf_pred - item.anchor_pred)
        item_sse, item_count = _sse_count_np(pred, item.target)
        sse += item_sse
        count += item_count
    return _rmse_from_sse_count(sse, count), sse, count


def _pf_failure_alpha_quadratic(items):
    sse_anchor = np.zeros(len(items), dtype=np.float64)
    cross = np.zeros(len(items), dtype=np.float64)
    delta_sq = np.zeros(len(items), dtype=np.float64)
    counts = np.zeros(len(items), dtype=np.int64)
    for idx, item in enumerate(items):
        anchor_err = item.anchor_pred - item.target
        delta = item.pf_pred - item.anchor_pred
        valid = np.isfinite(anchor_err) & np.isfinite(delta)
        if not valid.any():
            continue
        anchor_err = anchor_err[valid]
        delta = delta[valid]
        sse_anchor[idx] = float(np.dot(anchor_err, anchor_err))
        cross[idx] = float(np.dot(anchor_err, delta))
        delta_sq[idx] = float(np.dot(delta, delta))
        counts[idx] = int(valid.sum())
    return {
        "sse_anchor": sse_anchor,
        "cross": cross,
        "delta_sq": delta_sq,
        "counts": counts,
    }


def _pf_failure_sse_for_alpha(quad, alpha):
    alpha = float(alpha)
    return quad["sse_anchor"] + 2.0 * alpha * quad["cross"] + alpha * alpha * quad["delta_sq"]


def _score_pf_failure_alpha_rule(quad, indices, selected, alpha_bad=1.0, alpha_good=1.0):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return math.nan, 0.0, 0
    selected = np.asarray(selected, dtype=bool)
    risky = selected[indices]
    sse_bad = _pf_failure_sse_for_alpha(quad, alpha_bad)
    sse_good = _pf_failure_sse_for_alpha(quad, alpha_good)
    sse_vec = np.where(risky, sse_bad[indices], sse_good[indices])
    count = int(quad["counts"][indices].sum())
    sse = float(sse_vec.sum())
    return _rmse_from_sse_count(sse, count), sse, count


def _build_pf_failure_rule_grid(well_df):
    rules = []
    alpha_bad_values = (0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875)
    alpha_good_values = (1.0, 0.925, 0.875)
    for feature_name in PF_FAILURE_RELIABILITY_FEATURES:
        if feature_name not in well_df.columns:
            continue
        values = well_df[feature_name].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 10:
            continue
        thresholds = np.unique(np.quantile(values, np.linspace(0.05, 0.95, 19)))
        direction = "high" if feature_name in PF_FAILURE_HIGH_RISK_FEATURES else "low"
        for threshold in thresholds:
            for alpha_good in alpha_good_values:
                for alpha_bad in alpha_bad_values:
                    if alpha_bad > alpha_good:
                        continue
                    rules.append(
                        {
                            "feature": feature_name,
                            "direction": direction,
                            "threshold": float(threshold),
                            "alpha_bad": float(alpha_bad),
                            "alpha_good": float(alpha_good),
                        }
                )
    return rules


def _pf_failure_rule_selected(well_df, rule_spec):
    values = well_df[rule_spec["feature"]].to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    if rule_spec["direction"] == "high":
        return finite & (values >= float(rule_spec["threshold"]))
    if rule_spec["direction"] == "low":
        return finite & (values <= float(rule_spec["threshold"]))
    raise ValueError(f"unknown PF failure rule direction={rule_spec['direction']!r}")


def _score_pf_failure_rules(items, well_df, rules):
    quad = _pf_failure_alpha_quadratic(items)
    all_indices = np.arange(len(items), dtype=np.int64)
    bad50_indices = well_df.sort_values("pf_rmse", ascending=False).head(min(50, len(well_df)))["item_index"].to_numpy(dtype=np.int64)
    pf_worse_anchor_indices = well_df.loc[well_df["pf_rmse"] > well_df["anchor_rmse"], "item_index"].to_numpy(dtype=np.int64)
    base_rmse, _, _ = _score_pf_failure_alpha_rule(quad, all_indices, np.zeros(len(items), dtype=bool), alpha_good=1.0)
    base_bad50_rmse, _, _ = _score_pf_failure_alpha_rule(quad, bad50_indices, np.zeros(len(items), dtype=bool), alpha_good=1.0)
    base_pf_worse_rmse, _, _ = _score_pf_failure_alpha_rule(
        quad,
        pf_worse_anchor_indices,
        np.zeros(len(items), dtype=bool),
        alpha_good=1.0,
    )
    rows = []
    for rule_spec in rules:
        selected = _pf_failure_rule_selected(well_df, rule_spec)
        alpha_bad = float(rule_spec["alpha_bad"])
        alpha_good = float(rule_spec.get("alpha_good", 1.0))
        rmse, _, _ = _score_pf_failure_alpha_rule(quad, all_indices, selected, alpha_bad, alpha_good)
        bad50_rmse, _, _ = _score_pf_failure_alpha_rule(quad, bad50_indices, selected, alpha_bad, alpha_good)
        pf_worse_rmse, _, _ = _score_pf_failure_alpha_rule(
            quad,
            pf_worse_anchor_indices,
            selected,
            alpha_bad,
            alpha_good,
        )
        rows.append(
            {
                **rule_spec,
                "rmse": rmse,
                "bad50_rmse": bad50_rmse,
                "pf_worse_anchor_rmse": pf_worse_rmse,
                "global_gain": float(base_rmse - rmse),
                "bad50_gain": float(base_bad50_rmse - bad50_rmse),
                "pf_worse_anchor_gain": float(base_pf_worse_rmse - pf_worse_rmse),
                "selected_wells": int(selected.sum()),
                "selected_rows": int(well_df.loc[selected, "target_bins"].sum()),
            }
        )
    return pd.DataFrame(rows), {
        "base_rmse": base_rmse,
        "base_bad50_rmse": base_bad50_rmse,
        "base_pf_worse_anchor_rmse": base_pf_worse_rmse,
        "bad50_indices": bad50_indices.tolist(),
        "pf_worse_anchor_indices": pf_worse_anchor_indices.tolist(),
    }


def _nested_cv_pf_failure_rules(items, well_df, rules, fold_count=5, seed=7):
    quad = _pf_failure_alpha_quadratic(items)
    fold_values = _make_cached_pf_failure_folds(well_df["well_id"].to_numpy(dtype=str), fold_count=fold_count, seed=seed)
    well_df = well_df.copy()
    well_df["fold"] = fold_values
    rows = []
    total_sse = 0.0
    total_count = 0
    choices = []
    all_indices = well_df["item_index"].to_numpy(dtype=np.int64)
    empty_selected = np.zeros(len(items), dtype=bool)
    base_rmse, _, _ = _score_pf_failure_alpha_rule(quad, all_indices, empty_selected, alpha_good=1.0)
    for fold in sorted(np.unique(fold_values)):
        train_indices = well_df.loc[well_df["fold"] != fold, "item_index"].to_numpy(dtype=np.int64)
        val_indices = well_df.loc[well_df["fold"] == fold, "item_index"].to_numpy(dtype=np.int64)
        best = None
        for rule_spec in rules:
            selected = _pf_failure_rule_selected(well_df, rule_spec)
            train_rmse, _, _ = _score_pf_failure_alpha_rule(
                quad,
                train_indices,
                selected,
                alpha_bad=float(rule_spec["alpha_bad"]),
                alpha_good=float(rule_spec.get("alpha_good", 1.0)),
            )
            if best is None or train_rmse < best["train_rmse"]:
                best = {**rule_spec, "train_rmse": train_rmse, "selected": selected}
        if best is None:
            continue
        val_rmse, val_sse, val_count = _score_pf_failure_alpha_rule(
            quad,
            val_indices,
            best["selected"],
            alpha_bad=float(best["alpha_bad"]),
            alpha_good=float(best.get("alpha_good", 1.0)),
        )
        val_base_rmse, _, _ = _score_pf_failure_alpha_rule(quad, val_indices, empty_selected, alpha_good=1.0)
        total_sse += val_sse
        total_count += val_count
        choice_name = (
            f"{best['feature']}:{best['direction']}:{best['threshold']:.6g}:"
            f"{best['alpha_bad']:.3f}:{float(best.get('alpha_good', 1.0)):.3f}"
        )
        choices.append(choice_name)
        rows.append(
            {
                "fold": int(fold),
                "feature": best["feature"],
                "direction": best["direction"],
                "threshold": float(best["threshold"]),
                "alpha_bad": float(best["alpha_bad"]),
                "train_rmse": float(best["train_rmse"]),
                "val_rmse": float(val_rmse),
                "val_base_rmse": float(val_base_rmse),
                "val_gain": float(val_base_rmse - val_rmse),
                "val_count": int(val_count),
            }
        )
    nested_rmse = _rmse_from_sse_count(total_sse, total_count)
    nested_df = pd.DataFrame(rows)
    nested_df.attrs["nested_rmse"] = nested_rmse
    nested_df.attrs["base_rmse"] = base_rmse
    nested_df.attrs["nested_gain"] = float(base_rmse - nested_rmse) if np.isfinite(nested_rmse) else math.nan
    nested_df.attrs["chosen_per_fold"] = "|".join(choices)
    return nested_df, well_df


def run_cached_pf_failure_analysis(cache_dir=None, data_path=None, output_dir=None, top_k=50, fold_count=5, seed=7):
    cache_dir = Path(cache_dir or (_pf_opt_project_root() / "PF_cache" / "train"))
    data_path = Path(data_path or (_pf_opt_project_root() / "data" / "train"))
    output_dir = Path(output_dir or (_pf_opt_project_root() / "logs" / f"pf_failure_analysis_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    config_path = cache_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"PF cache config not found: {config_path}")
    with config_path.open("r") as f:
        config_payload = json.load(f)
    cache_cfg = dict(config_payload["cache_cfg"])
    well_ids = _discover_well_ids(data_path)
    cached_wells = {path.stem for path in cache_dir.glob("*.npz")}
    well_ids = [well_id for well_id in well_ids if well_id in cached_wells]
    if not well_ids:
        raise ValueError(f"no cached train PF wells found under {cache_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    feature_rows = []
    for well_id in tqdm(well_ids, desc="pf_failure_cache", dynamic_ncols=True):
        item, features = _make_cached_pf_failure_item(cache_dir, data_path, well_id, cache_cfg)
        if item is None:
            continue
        features = dict(features)
        features["item_index"] = int(len(items))
        items.append(item)
        feature_rows.append(features)
    if not items:
        raise ValueError("cached PF failure analysis found no target rows")

    well_df = pd.DataFrame(feature_rows).sort_values("pf_rmse", ascending=False).reset_index(drop=True)
    all_indices = np.arange(len(items), dtype=np.int64)
    pf_rmse, _, target_count = _score_pf_failure_rule(items, all_indices)
    anchor_sse = 0.0
    anchor_count = 0
    for item in items:
        sse, count = _sse_count_np(item.anchor_pred, item.target)
        anchor_sse += sse
        anchor_count += count
    anchor_rmse = _rmse_from_sse_count(anchor_sse, anchor_count)

    corr_rows = []
    for feature_name in PF_FAILURE_RELIABILITY_FEATURES + ("suffix_gr_missing_frac", "max_finite_gr_run", "max_nan_gr_run", "true_rel_abs_mean"):
        if feature_name not in well_df.columns:
            continue
        x = well_df[feature_name].to_numpy(dtype=np.float64)
        y = well_df["pf_rmse"].to_numpy(dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        corr = float(np.corrcoef(x[valid], y[valid])[0, 1]) if valid.sum() > 2 else math.nan
        corr_rows.append({"feature": feature_name, "corr_with_pf_rmse": corr})
    corr_df = pd.DataFrame(corr_rows).sort_values("corr_with_pf_rmse", key=lambda s: s.abs(), ascending=False)

    rules = _build_pf_failure_rule_grid(well_df)
    rule_df, base_stats = _score_pf_failure_rules(items, well_df, rules)
    rule_df = rule_df.sort_values(["rmse", "bad50_rmse"]).reset_index(drop=True)
    nested_df, well_df_with_folds = _nested_cv_pf_failure_rules(
        items,
        well_df,
        rules,
        fold_count=int(fold_count),
        seed=int(seed),
    )
    best_global = rule_df.iloc[0].to_dict() if len(rule_df) else {}
    best_bad50 = rule_df.sort_values(["bad50_rmse", "rmse"]).iloc[0].to_dict() if len(rule_df) else {}

    well_df_with_folds.to_csv(output_dir / "well_diagnostics.csv", index=False)
    well_df_with_folds.head(int(top_k)).to_csv(output_dir / f"top{int(top_k)}_bad_pf_wells.csv", index=False)
    corr_df.to_csv(output_dir / "feature_correlations.csv", index=False)
    rule_df.to_csv(output_dir / "reliability_rule_scan.csv", index=False)
    nested_df.to_csv(output_dir / "reliability_rule_nested_cv.csv", index=False)

    top_cols = [
        "well_id",
        "target_bins",
        "pf_rmse",
        "anchor_rmse",
        "pf_minus_anchor_rmse",
        "suffix_gr_missing_frac",
        "entropy_mean",
        "ess_mean",
        "maxp_mean",
        "anchor_mass15_mean",
        "pf_gr_rmse",
        "anchor_gr_rmse",
        "pred_rel_end",
        "true_rel_end",
        "mode_target_abs_mean",
    ]
    top_cols = [col for col in top_cols if col in well_df.columns]
    summary = {
        "cache_dir": str(cache_dir),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "cache_digest": config_payload.get("cache_digest"),
        "well_count": int(len(items)),
        "target_count": int(target_count),
        "base_pf_rmse": float(pf_rmse),
        "anchor_rmse": float(anchor_rmse),
        "pf_worse_than_anchor_wells": int((well_df["pf_rmse"] > well_df["anchor_rmse"]).sum()),
        "top_bad_pf_wells": well_df[top_cols].head(int(top_k)).to_dict(orient="records"),
        "feature_correlations": corr_df.to_dict(orient="records"),
        "best_global_rule": best_global,
        "best_bad50_rule": best_bad50,
        "nested_cv_rule_rmse": float(nested_df.attrs.get("nested_rmse", math.nan)),
        "nested_cv_rule_gain": float(nested_df.attrs.get("nested_gain", math.nan)),
        "nested_cv_chosen_per_fold": nested_df.attrs.get("chosen_per_fold", ""),
        "artifacts": {
            "well_diagnostics": str(output_dir / "well_diagnostics.csv"),
            "top_bad_pf_wells": str(output_dir / f"top{int(top_k)}_bad_pf_wells.csv"),
            "feature_correlations": str(output_dir / "feature_correlations.csv"),
            "reliability_rule_scan": str(output_dir / "reliability_rule_scan.csv"),
            "reliability_rule_nested_cv": str(output_dir / "reliability_rule_nested_cv.csv"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)
    _append_progress(
        "Cached PF failure analysis result:\n"
        f"- Cache `{cache_dir}` digest `{summary['cache_digest']}` wells={summary['well_count']} target_bins={summary['target_count']}.\n"
        f"- Cached PF RMSE `{summary['base_pf_rmse']:.6f}`, anchor `{summary['anchor_rmse']:.6f}`, "
        f"PF worse than anchor on `{summary['pf_worse_than_anchor_wells']}` wells.\n"
        f"- Best global reliability rule: `{json.dumps(best_global, sort_keys=True, default=str)}`.\n"
        f"- Nested-CV rule RMSE `{summary['nested_cv_rule_rmse']:.6f}` "
        f"(gain `{summary['nested_cv_rule_gain']:.6f}` vs cached PF).\n"
        f"- Artifacts: `{output_dir}`."
    )
    return summary


def _empty_compare_acc(name):
    return {
        "name": name,
        "sse": 0.0,
        "count": 0,
        "well_rmse": [],
    }


def _add_compare_metric(acc, pred_bin, target_bin, mask):
    valid = mask & np.isfinite(pred_bin) & np.isfinite(target_bin)
    if not valid.any():
        return
    err = np.asarray(pred_bin[valid], dtype=np.float64) - np.asarray(target_bin[valid], dtype=np.float64)
    sse = float(np.square(err).sum())
    count = int(valid.sum())
    acc["sse"] += sse
    acc["count"] += count
    acc["well_rmse"].append(math.sqrt(sse / count))


def _finalize_compare_acc(acc, well_count, elapsed):
    well_rmse = np.asarray(acc["well_rmse"], dtype=np.float64)
    return {
        "name": acc["name"],
        "well_count": int(well_count),
        "target_count": int(acc["count"]),
        "rmse": math.sqrt(float(acc["sse"]) / int(acc["count"])) if acc["count"] > 0 else math.nan,
        "well_rmse_mean": float(np.mean(well_rmse)) if well_rmse.size else math.nan,
        "well_rmse_median": float(np.median(well_rmse)) if well_rmse.size else math.nan,
        "well_rmse_q75": float(np.quantile(well_rmse, 0.75)) if well_rmse.size else math.nan,
        "well_rmse_q95": float(np.quantile(well_rmse, 0.95)) if well_rmse.size else math.nan,
        "elapsed_sec": float(elapsed),
        "sec_per_well": float(elapsed / max(int(well_count), 1)),
    }


def _format_compare_method(summary):
    return (
        f"{summary['name']}: RMSE={summary['rmse']:.6f} "
        f"well_mean/median/q75/q95="
        f"{summary['well_rmse_mean']:.4f}/{summary['well_rmse_median']:.4f}/"
        f"{summary['well_rmse_q75']:.4f}/{summary['well_rmse_q95']:.4f} "
        f"sec/well={summary['sec_per_well']:.3f}"
    )


def _benchmark_pf_compare_one(args):
    data_path, well_id, cache_cfg, cfg = args
    h_df, tw_df = _read_well_frames(data_path, well_id)
    item = _build_pf_heatmap_item(h_df, tw_df, cache_cfg, has_target=True)
    finite_seen = np.flatnonzero(np.isfinite(h_df["TVT_input"].to_numpy(dtype=np.float64)))
    suffix_start = int(finite_seen[-1]) + 1
    kept_len = min(len(h_df) - suffix_start, int(cache_cfg["target_len"]))

    target_bin = item["window_tvt"]
    target_mask = item["target_mask"] & np.isfinite(target_bin)
    anchor_pred = np.full_like(target_bin, np.float32(item["tvt0"]), dtype=np.float32)
    numba_pred = item["pf_tvt_pred"]

    pasted_row_pred = run_pf_lik_ensemble_pasted_v12(
        h_df,
        tw_df,
        n_particles=int(cache_cfg["n_particles"]),
        n_seeds=int(cache_cfg["n_seeds"]),
        scale=float(cache_cfg.get("lik_scale", 5.0)),
    )
    pasted_pred = _downsample_suffix_values_for_benchmark(
        pasted_row_pred[suffix_start : suffix_start + kept_len],
        cfg,
    )

    metrics = {}
    for name, pred in (
        ("anchor", anchor_pred),
        ("default_numba_pf", numba_pred),
        ("pasted_v12_pf", pasted_pred),
    ):
        valid = target_mask & np.isfinite(pred)
        if valid.any():
            err = np.asarray(pred[valid], dtype=np.float64) - np.asarray(target_bin[valid], dtype=np.float64)
            metrics[name] = {
                "sse": float(np.square(err).sum()),
                "count": int(valid.sum()),
                "rmse": float(math.sqrt(np.square(err).mean())),
            }
        else:
            metrics[name] = {"sse": math.nan, "count": 0, "rmse": math.nan}

    finite_compare = target_mask & np.isfinite(numba_pred) & np.isfinite(pasted_pred)
    if finite_compare.any():
        path_gap = np.asarray(pasted_pred[finite_compare], dtype=np.float64) - np.asarray(numba_pred[finite_compare], dtype=np.float64)
        path_gap_rmse = float(math.sqrt(np.square(path_gap).mean()))
        path_gap_mean = float(np.mean(path_gap))
        numba_abs_err = np.abs(np.asarray(numba_pred[finite_compare], dtype=np.float64) - np.asarray(target_bin[finite_compare], dtype=np.float64))
        pasted_abs_err = np.abs(np.asarray(pasted_pred[finite_compare], dtype=np.float64) - np.asarray(target_bin[finite_compare], dtype=np.float64))
        pasted_better_abs_mae = float(np.mean(pasted_abs_err < numba_abs_err))
    else:
        path_gap_rmse = math.nan
        path_gap_mean = math.nan
        pasted_better_abs_mae = math.nan

    suffix_gr = h_df["GR"].to_numpy(dtype=np.float64)[suffix_start : suffix_start + kept_len]
    return {
        "well_id": well_id,
        "metrics": metrics,
        "path_gap_rmse": path_gap_rmse,
        "path_gap_mean": path_gap_mean,
        "pasted_better_abs_mae_frac": pasted_better_abs_mae,
        "suffix_gr_missing_frac": float(np.mean(~np.isfinite(suffix_gr))) if suffix_gr.size else math.nan,
        "target_bins": int(target_mask.sum()),
    }


def benchmark_pf_pasted_vs_numba(well_limit=100, workers=8, data_path=None):
    if njit is None:
        raise ImportError("numba is required for PF comparison benchmarks")
    cfg = _make_pf_opt_cfg(data_path=data_path)
    cfg.PF_heatmap_n_particles = 500
    cfg.PF_heatmap_n_seeds = 64
    cfg.PF_heatmap_num_workers = int(workers)
    if hasattr(cfg, "refresh"):
        cfg.refresh()
    data_path = Path(data_path or cfg.train_path)
    well_ids = _discover_well_ids(data_path)
    if well_limit is not None:
        well_ids = well_ids[: int(well_limit)]
    if not well_ids:
        raise ValueError(f"no train wells found under {data_path}")
    cache_cfg = pf_heatmap_cache_config(cfg)
    if cache_cfg["n_particles"] != 500 or cache_cfg["n_seeds"] != 64:
        raise ValueError("PF comparison benchmark must use 500 particles and 64 seeds")

    _warmup_pf_heatmap_numba()
    worker_args = [(data_path, well_id, cache_cfg, cfg) for well_id in well_ids]
    n_workers = max(1, min(int(workers), len(worker_args)))
    print(
        f"compare pasted_v12_pf vs default_numba_pf: wells={len(well_ids)} "
        f"workers={n_workers} N={cache_cfg['n_particles']} seeds={cache_cfg['n_seeds']}",
        flush=True,
    )
    started = time.perf_counter()
    if n_workers == 1:
        results = [_benchmark_pf_compare_one(args) for args in tqdm(worker_args, desc="pf_compare", dynamic_ncols=True)]
    else:
        with Pool(processes=n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_benchmark_pf_compare_one, worker_args),
                    total=len(worker_args),
                    desc="pf_compare",
                    dynamic_ncols=True,
                )
            )
    elapsed = time.perf_counter() - started

    accum = {
        "anchor": _empty_compare_acc("anchor"),
        "default_numba_pf": _empty_compare_acc("default_numba_pf"),
        "pasted_v12_pf": _empty_compare_acc("pasted_v12_pf"),
    }
    for result in results:
        for name, acc in accum.items():
            metric = result["metrics"][name]
            if metric["count"] <= 0 or not np.isfinite(metric["sse"]):
                continue
            acc["sse"] += float(metric["sse"])
            acc["count"] += int(metric["count"])
            acc["well_rmse"].append(float(metric["rmse"]))

    summaries = {name: _finalize_compare_acc(acc, len(results), elapsed) for name, acc in accum.items()}
    numba_wins = 0
    pasted_wins = 0
    ties = 0
    deltas = []
    miss = []
    better_frac = []
    path_gap_rmse = []
    path_gap_mean = []
    for result in results:
        n_rmse = result["metrics"]["default_numba_pf"]["rmse"]
        p_rmse = result["metrics"]["pasted_v12_pf"]["rmse"]
        if np.isfinite(n_rmse) and np.isfinite(p_rmse):
            delta = float(p_rmse - n_rmse)
            deltas.append(delta)
            if delta < -1e-12:
                pasted_wins += 1
            elif delta > 1e-12:
                numba_wins += 1
            else:
                ties += 1
        if np.isfinite(result["suffix_gr_missing_frac"]):
            miss.append(float(result["suffix_gr_missing_frac"]))
        if np.isfinite(result["pasted_better_abs_mae_frac"]):
            better_frac.append(float(result["pasted_better_abs_mae_frac"]))
        if np.isfinite(result["path_gap_rmse"]):
            path_gap_rmse.append(float(result["path_gap_rmse"]))
        if np.isfinite(result["path_gap_mean"]):
            path_gap_mean.append(float(result["path_gap_mean"]))

    deltas_arr = np.asarray(deltas, dtype=np.float64)
    miss_arr = np.asarray(miss, dtype=np.float64)
    corr = float(np.corrcoef(deltas_arr, miss_arr[: len(deltas_arr)])[0, 1]) if deltas_arr.size > 1 and miss_arr.size >= deltas_arr.size else math.nan
    diagnostics = {
        "well_wins": {
            "pasted_v12_pf": int(pasted_wins),
            "default_numba_pf": int(numba_wins),
            "ties": int(ties),
        },
        "delta_pasted_minus_numba_well_rmse_mean": float(np.mean(deltas_arr)) if deltas_arr.size else math.nan,
        "delta_pasted_minus_numba_well_rmse_median": float(np.median(deltas_arr)) if deltas_arr.size else math.nan,
        "delta_pasted_minus_numba_well_rmse_q25": float(np.quantile(deltas_arr, 0.25)) if deltas_arr.size else math.nan,
        "delta_pasted_minus_numba_well_rmse_q75": float(np.quantile(deltas_arr, 0.75)) if deltas_arr.size else math.nan,
        "suffix_gr_missing_frac_mean": float(np.mean(miss_arr)) if miss_arr.size else math.nan,
        "delta_vs_missing_corr": corr,
        "pasted_better_bin_abs_error_frac_mean": float(np.mean(better_frac)) if better_frac else math.nan,
        "path_gap_rmse_mean": float(np.mean(path_gap_rmse)) if path_gap_rmse else math.nan,
        "path_gap_mean_mean": float(np.mean(path_gap_mean)) if path_gap_mean else math.nan,
    }
    output = {
        "well_count": len(results),
        "budget": {"n_particles": cache_cfg["n_particles"], "n_seeds": cache_cfg["n_seeds"]},
        "workers": n_workers,
        "methods": summaries,
        "diagnostics": diagnostics,
    }

    print(_format_compare_method(summaries["anchor"]), flush=True)
    print(_format_compare_method(summaries["default_numba_pf"]), flush=True)
    print(_format_compare_method(summaries["pasted_v12_pf"]), flush=True)
    print(
        "compare diagnostics: "
        f"wins pasted/default/tie={pasted_wins}/{numba_wins}/{ties} "
        f"delta_mean/median={diagnostics['delta_pasted_minus_numba_well_rmse_mean']:.4f}/"
        f"{diagnostics['delta_pasted_minus_numba_well_rmse_median']:.4f} "
        f"bin_abs_win_frac={diagnostics['pasted_better_bin_abs_error_frac_mean']:.3f} "
        f"path_gap_rmse={diagnostics['path_gap_rmse_mean']:.4f}",
        flush=True,
    )
    return output


def _merge64_local_seed_blend_candidates():
    base = _clean_pf_random_overrides(PF_OPT_MERGE64_CURRENT_BEST_OVERRIDES)
    candidates = [("m64_local_sota_base", base)]
    candidates.extend(
        [
            (
                "m64_seed_equal",
                {
                    **base,
                    "PF_heatmap_seed_path_weight_mode": "equal",
                    "PF_heatmap_seed_prob_weight_mode": "equal",
                },
            ),
            (
                "m64_local_bin_s050_gmix075",
                {
                    **base,
                    "PF_heatmap_seed_local_weight_mode": "local",
                    "PF_heatmap_seed_local_lik_scale": 0.50,
                    "PF_heatmap_seed_local_global_mix": 0.75,
                },
            ),
            (
                "m64_local_bin_s100_gmix075",
                {
                    **base,
                    "PF_heatmap_seed_local_weight_mode": "local",
                    "PF_heatmap_seed_local_lik_scale": 1.00,
                    "PF_heatmap_seed_local_global_mix": 0.75,
                },
            ),
            (
                "m64_fdecay_h4_s050_gmix075",
                {
                    **base,
                    "PF_heatmap_seed_local_weight_mode": "forward_decay",
                    "PF_heatmap_seed_local_half_life_blocks": 4.0,
                    "PF_heatmap_seed_local_lik_scale": 0.50,
                    "PF_heatmap_seed_local_global_mix": 0.75,
                },
            ),
            (
                "m64_fdecay_h8_s050_gmix075",
                {
                    **base,
                    "PF_heatmap_seed_local_weight_mode": "forward_decay",
                    "PF_heatmap_seed_local_half_life_blocks": 8.0,
                    "PF_heatmap_seed_local_lik_scale": 0.50,
                    "PF_heatmap_seed_local_global_mix": 0.75,
                },
            ),
            (
                "m64_fdecay_h16_s050_gmix075",
                {
                    **base,
                    "PF_heatmap_seed_local_weight_mode": "forward_decay",
                    "PF_heatmap_seed_local_half_life_blocks": 16.0,
                    "PF_heatmap_seed_local_lik_scale": 0.50,
                    "PF_heatmap_seed_local_global_mix": 0.75,
                },
            ),
            (
                "m64_fdecay_h8_s100_gmix075",
                {
                    **base,
                    "PF_heatmap_seed_local_weight_mode": "forward_decay",
                    "PF_heatmap_seed_local_half_life_blocks": 8.0,
                    "PF_heatmap_seed_local_lik_scale": 1.00,
                    "PF_heatmap_seed_local_global_mix": 0.75,
                },
            ),
            (
                "m64_2decay_h4_s050_gmix075",
                {
                    **base,
                    "PF_heatmap_seed_local_weight_mode": "two_sided_decay",
                    "PF_heatmap_seed_local_half_life_blocks": 4.0,
                    "PF_heatmap_seed_local_lik_scale": 0.50,
                    "PF_heatmap_seed_local_global_mix": 0.75,
                },
            ),
            (
                "m64_2decay_h8_s050_gmix075",
                {
                    **base,
                    "PF_heatmap_seed_local_weight_mode": "two_sided_decay",
                    "PF_heatmap_seed_local_half_life_blocks": 8.0,
                    "PF_heatmap_seed_local_lik_scale": 0.50,
                    "PF_heatmap_seed_local_global_mix": 0.75,
                },
            ),
            (
                "m64_2decay_h16_s050_gmix075",
                {
                    **base,
                    "PF_heatmap_seed_local_weight_mode": "two_sided_decay",
                    "PF_heatmap_seed_local_half_life_blocks": 16.0,
                    "PF_heatmap_seed_local_lik_scale": 0.50,
                    "PF_heatmap_seed_local_global_mix": 0.75,
                },
            ),
            (
                "m64_2decay_h8_s100_gmix075",
                {
                    **base,
                    "PF_heatmap_seed_local_weight_mode": "two_sided_decay",
                    "PF_heatmap_seed_local_half_life_blocks": 8.0,
                    "PF_heatmap_seed_local_lik_scale": 1.00,
                    "PF_heatmap_seed_local_global_mix": 0.75,
                },
            ),
            (
                "m64_2decay_h8_s050_gmix050",
                {
                    **base,
                    "PF_heatmap_seed_local_weight_mode": "two_sided_decay",
                    "PF_heatmap_seed_local_half_life_blocks": 8.0,
                    "PF_heatmap_seed_local_lik_scale": 0.50,
                    "PF_heatmap_seed_local_global_mix": 0.50,
                },
            ),
        ]
    )
    return [(name, _clean_pf_random_overrides(overrides)) for name, overrides in candidates]


def run_pf_merge64_local_seed_blend_test(
    sample_size=100,
    workers=8,
    data_path=None,
    n_particles=500,
    n_seeds=128,
):
    PF_OPT_PROGRESS_PATH.read_text()
    data_cfg = _make_pf_opt_cfg(
        data_path=data_path,
        overrides=PF_OPT_MERGE64_CURRENT_BEST_OVERRIDES,
        n_particles=int(n_particles),
        n_seeds=int(n_seeds),
        workers=int(workers),
    )
    data_path = Path(data_path or data_cfg.train_path)
    all_wells = _discover_well_ids(data_path)
    if not all_wells:
        raise ValueError(f"no train wells found under {data_path}")
    sampled_wells = _sample_pf_opt_wells(all_wells, sample_size, round_idx=1, seed_base=20260630)
    candidates = _merge64_local_seed_blend_candidates()
    summaries = []
    for name, overrides in candidates:
        summary = benchmark_pf_opt_candidate(
            name,
            overrides,
            workers=workers,
            data_path=data_path,
            progress=True,
            well_ids=sampled_wells,
            n_particles=int(n_particles),
            n_seeds=int(n_seeds),
            enforce_budget=(int(n_particles), int(n_seeds)),
        )
        summaries.append(summary)

    baseline = summaries[0]
    ranked = sorted(summaries[1:], key=lambda item: float(item.get("rmse_ffbsi", math.inf)))
    best = ranked[0] if ranked else baseline
    lines = [
        "Merge64 local/decayed seed-blend fixed-slice test.",
        f"- Wells: `{len(sampled_wells)}` deterministic random sample, seed base `20260630`.",
        f"- Budget: `{int(n_particles)} particles x {int(n_seeds)} seeds`, workers `{int(workers)}`.",
        f"- Baseline `{baseline['name']}`: direct `{baseline['rmse']:.6f}`, FFBSi `{baseline.get('rmse_ffbsi', math.nan):.6f}`, "
        f"FFBSi mean/trim90 `{baseline.get('well_rmse_mean_ffbsi', math.nan):.4f}/{baseline.get('well_rmse_trim90_mean_ffbsi', math.nan):.4f}`.",
        f"- Best variant `{best['name']}`: direct `{best['rmse']:.6f}`, FFBSi `{best.get('rmse_ffbsi', math.nan):.6f}`, "
        f"FFBSi mean/trim90 `{best.get('well_rmse_mean_ffbsi', math.nan):.4f}/{best.get('well_rmse_trim90_mean_ffbsi', math.nan):.4f}`.",
        f"- Best FFBSi gain vs baseline: `{float(baseline.get('rmse_ffbsi', math.nan)) - float(best.get('rmse_ffbsi', math.nan)):.6f}`; "
        f"direct gain `{float(baseline['rmse']) - float(best['rmse']):.6f}`.",
        "- Ranked FFBSi results:",
    ]
    for summary in [baseline] + ranked[:8]:
        lines.append(
            f"  - `{summary['name']}` direct `{summary['rmse']:.6f}` FFBSi `{summary.get('rmse_ffbsi', math.nan):.6f}` "
            f"FFBSi mean/trim90 `{summary.get('well_rmse_mean_ffbsi', math.nan):.4f}/{summary.get('well_rmse_trim90_mean_ffbsi', math.nan):.4f}` "
            f"sec/well `{summary['sec_per_well']:.3f}`"
        )
    _append_progress("\n".join(lines))
    return {"baseline": baseline, "best": best, "summaries": summaries, "well_ids": sampled_wells}


def run_pf_ffbsi_feature_test(
    sample_size=100,
    workers=8,
    data_path=None,
    n_particles=500,
    n_seeds=32,
    n_paths=64,
):
    PF_OPT_PROGRESS_PATH.read_text()
    current_overrides = _clean_pf_random_overrides(PF_OPT_CURRENT_BEST_OVERRIDES)
    cfg = _make_pf_opt_cfg(
        data_path=data_path,
        overrides=current_overrides,
        n_particles=int(n_particles),
        n_seeds=int(n_seeds),
        workers=int(workers),
    )
    data_path = Path(data_path or cfg.train_path)
    all_wells = _discover_well_ids(data_path)
    if not all_wells:
        raise ValueError(f"no train wells found under {data_path}")
    sampled_wells = _sample_pf_opt_wells(all_wells, sample_size, round_idx=1, seed_base=20260612)
    base_summary = benchmark_pf_opt_candidate(
        "ffbsi_baseline_filtered_100",
        current_overrides,
        workers=workers,
        data_path=data_path,
        progress=True,
        well_ids=sampled_wells,
        n_particles=int(n_particles),
        n_seeds=int(n_seeds),
        enforce_budget=(int(n_particles), int(n_seeds)),
    )
    ffbsi_overrides = dict(current_overrides)
    ffbsi_overrides.update(
        {
            "PF_heatmap_ffbsi_mode": "fallback",
            "PF_heatmap_ffbsi_n_paths": int(n_paths),
            "PF_heatmap_ffbsi_fallback_mode": "filtered",
            "PF_heatmap_ffbsi_max_active_bins": 512,
            "PF_heatmap_ffbsi_transition_scale": 1.5,
            "PF_heatmap_ffbsi_pos_floor": 0.35,
            "PF_heatmap_ffbsi_rate_floor": 0.0025,
        }
    )
    ffbsi_summary = benchmark_pf_opt_candidate(
        f"ffbsi_paths{int(n_paths)}_100",
        ffbsi_overrides,
        workers=workers,
        data_path=data_path,
        progress=True,
        well_ids=sampled_wells,
        n_particles=int(n_particles),
        n_seeds=int(n_seeds),
        enforce_budget=(int(n_particles), int(n_seeds)),
    )
    gain = float(base_summary["rmse"] - ffbsi_summary["rmse"])
    message = (
        "FFBSi feature test on fixed 100-well sample, not random search:\n"
        f"- Baseline filtered: {_format_pf_opt_summary(base_summary)}\n"
        f"- FFBSi fallback mode, paths={int(n_paths)}: {_format_pf_opt_summary(ffbsi_summary)}\n"
        f"- Sample gain FFBSi vs filtered: `{gain:.6f}`.\n"
        f"- FFBSi overrides: `{json.dumps(ffbsi_overrides, sort_keys=True)}`."
    )
    _append_progress(message)
    return {
        "baseline": base_summary,
        "ffbsi": ffbsi_summary,
        "gain": gain,
        "well_ids": sampled_wells,
    }


def _with_pf_ffbsi_overrides(overrides, n_paths):
    out = dict(overrides)
    out.update(
        {
            "PF_heatmap_ffbsi_mode": "fallback",
            "PF_heatmap_ffbsi_n_paths": int(n_paths),
            "PF_heatmap_ffbsi_fallback_mode": "filtered",
            "PF_heatmap_ffbsi_max_active_bins": 512,
            "PF_heatmap_ffbsi_transition_scale": 1.5,
            "PF_heatmap_ffbsi_pos_floor": 0.35,
            "PF_heatmap_ffbsi_rate_floor": 0.0025,
        }
    )
    return out


def _format_pf_compare_for_progress(result):
    methods = result["methods"]
    diag = result["diagnostics"]
    lines = [
        (
            f"PF comparison result at `500 x 64`, wells={result['well_count']}, "
            f"workers={result['workers']}:"
        ),
        f"- {_format_compare_method(methods['anchor'])}",
        f"- {_format_compare_method(methods['default_numba_pf'])}",
        f"- {_format_compare_method(methods['pasted_v12_pf'])}",
        (
            "- Well wins pasted/default/tie: "
            f"{diag['well_wins']['pasted_v12_pf']}/"
            f"{diag['well_wins']['default_numba_pf']}/"
            f"{diag['well_wins']['ties']}."
        ),
        (
            "- Pasted minus numba per-well RMSE delta mean/median/q25/q75: "
            f"{diag['delta_pasted_minus_numba_well_rmse_mean']:.6f}/"
            f"{diag['delta_pasted_minus_numba_well_rmse_median']:.6f}/"
            f"{diag['delta_pasted_minus_numba_well_rmse_q25']:.6f}/"
            f"{diag['delta_pasted_minus_numba_well_rmse_q75']:.6f}."
        ),
        (
            "- Diagnostics: "
            f"suffix GR missing mean={diag['suffix_gr_missing_frac_mean']:.3f}, "
            f"delta-vs-missing corr={diag['delta_vs_missing_corr']:.4f}, "
            f"mean bin abs-error win fraction for pasted={diag['pasted_better_bin_abs_error_frac_mean']:.3f}, "
            f"mean path-gap RMSE={diag['path_gap_rmse_mean']:.4f}, "
            f"mean signed path gap pasted-numba={diag['path_gap_mean_mean']:.4f}."
        ),
    ]
    return "\n".join(lines)


def _format_pf_opt_summary(summary):
    return (
        f"{summary['name']}: wells={summary['well_count']} "
        f"direct_RMSE={summary['rmse']:.6f} ffbsi_RMSE={summary.get('rmse_ffbsi', math.nan):.6f} "
        f"anchor={summary['anchor_rmse']:.6f} "
        f"direct well_mean/trim90/median/q75/q95="
        f"{summary['well_rmse_mean']:.4f}/{summary['well_rmse_trim90_mean']:.4f}/"
        f"{summary['well_rmse_median']:.4f}/"
        f"{summary['well_rmse_q75']:.4f}/{summary['well_rmse_q95']:.4f} "
        f"ffbsi well_mean/trim90/median/q75/q95="
        f"{summary.get('well_rmse_mean_ffbsi', math.nan):.4f}/"
        f"{summary.get('well_rmse_trim90_mean_ffbsi', math.nan):.4f}/"
        f"{summary.get('well_rmse_median_ffbsi', math.nan):.4f}/"
        f"{summary.get('well_rmse_q75_ffbsi', math.nan):.4f}/"
        f"{summary.get('well_rmse_q95_ffbsi', math.nan):.4f} "
        f"missGR={summary['suffix_gr_missing_frac_mean']:.3f} "
        f"sec/well={summary['sec_per_well']:.3f}"
    )


PF_OPT_GATE_METRICS = (
    ("rmse", "global"),
    ("well_rmse_mean", "well_mean"),
    ("well_rmse_trim90_mean", "well_trim90"),
)

PF_OPT_FFBSI_GATE_METRICS = (
    ("rmse_ffbsi", "global"),
    ("well_rmse_mean_ffbsi", "well_mean"),
    ("well_rmse_trim90_mean_ffbsi", "well_trim90"),
)


def _pf_opt_gate_metrics(objective_metric="direct"):
    objective_metric = str(objective_metric or "direct").strip().lower()
    if objective_metric in {"ffbsi", "rmse_ffbsi", "ffbsi_rmse"}:
        return PF_OPT_FFBSI_GATE_METRICS
    if objective_metric in {"direct", "filtered", "rmse"}:
        return PF_OPT_GATE_METRICS
    raise ValueError(f"unknown PF opt objective metric `{objective_metric}`")


def _pf_opt_objective_key(objective_metric="direct"):
    return _pf_opt_gate_metrics(objective_metric)[0][0]


def _pf_opt_metric_value(summary, key):
    value = float(summary.get(key, math.nan))
    return value if np.isfinite(value) else math.inf


def _pf_opt_metric_gains(baseline, candidate, objective_metric="direct"):
    return {
        label: _pf_opt_metric_value(baseline, key) - _pf_opt_metric_value(candidate, key)
        for key, label in _pf_opt_gate_metrics(objective_metric)
    }


def _pf_opt_passes_metric_gate(baseline, candidate, global_margin=0.0, aux_margin=0.0, objective_metric="direct"):
    gains = _pf_opt_metric_gains(baseline, candidate, objective_metric=objective_metric)
    global_margin = float(global_margin)
    aux_margin = float(aux_margin)
    if global_margin > 0.0:
        global_ok = gains["global"] >= global_margin
    else:
        global_ok = gains["global"] > 0.0
    aux_ok = all(gains[name] > aux_margin for name in gains if name != "global")
    return global_ok and aux_ok, gains


def _format_pf_opt_metric_gains(gains):
    return ", ".join(f"{name}={gain:.6f}" for name, gain in gains.items())


def _pf_opt_full_check_seed(full_check_idx, base_seed=PF_OPT_FULL_CHECK_DEFAULT_BASE_SEED):
    return int(base_seed) + 1009 * max(0, int(full_check_idx) - 1)


def _pf_opt_with_full_check_seed(overrides, base_seed):
    out = dict(overrides or {})
    out["PF_heatmap_base_seed"] = int(base_seed)
    return _clean_pf_random_overrides(out)


def _run_pf_opt_full_pair_check(
    check_label,
    sota_name,
    sota_overrides,
    candidate_name,
    candidate_overrides,
    full_wells,
    workers,
    data_path,
    full_check_idx,
    n_particles=PF_OPT_FULL_CHECK_DEFAULT_PARTICLES,
    n_seeds=PF_OPT_FULL_CHECK_DEFAULT_SEEDS,
    base_seed=PF_OPT_FULL_CHECK_DEFAULT_BASE_SEED,
    objective_metric="direct",
):
    n_particles, n_seeds = _pf_opt_candidate_budget(
        candidate_overrides,
        default_n_particles=int(n_particles),
        default_n_seeds=int(n_seeds),
    )
    check_seed = _pf_opt_full_check_seed(full_check_idx, base_seed=base_seed)
    seeded_sota_overrides = _pf_opt_with_full_check_seed(sota_overrides, check_seed)
    seeded_candidate_overrides = _pf_opt_with_full_check_seed(candidate_overrides, check_seed)
    sota_full = benchmark_pf_opt_candidate(
        f"{check_label}_sota_{sota_name}_full_seed{check_seed}",
        seeded_sota_overrides,
        workers=workers,
        data_path=data_path,
        progress=True,
        well_ids=full_wells,
        n_particles=int(n_particles),
        n_seeds=int(n_seeds),
        enforce_budget=(int(n_particles), int(n_seeds)),
    )
    cand_full = benchmark_pf_opt_candidate(
        f"{check_label}_{candidate_name}_full_seed{check_seed}",
        seeded_candidate_overrides,
        workers=workers,
        data_path=data_path,
        progress=True,
        well_ids=full_wells,
        n_particles=int(n_particles),
        n_seeds=int(n_seeds),
        enforce_budget=(int(n_particles), int(n_seeds)),
    )
    full_ok, full_gains = _pf_opt_passes_metric_gate(
        sota_full,
        cand_full,
        global_margin=0.0,
        aux_margin=0.0,
        objective_metric=objective_metric,
    )
    decision = "ACCEPTED" if full_ok else "rejected"
    _append_progress(
        f"Full-check SOTA trail `{check_label}`: budget `{int(n_particles)} x {int(n_seeds)}`, "
        f"base_seed `{int(check_seed)}`. Current SOTA `{sota_name}` direct_RMSE `{sota_full['rmse']:.6f}`, "
        f"FFBSi_RMSE `{sota_full.get('rmse_ffbsi', math.nan):.6f}`, "
        f"well_mean/trim90 `{sota_full['well_rmse_mean']:.6f}/{sota_full['well_rmse_trim90_mean']:.6f}`; "
        f"candidate `{candidate_name}` direct_RMSE `{cand_full['rmse']:.6f}`, "
        f"FFBSi_RMSE `{cand_full.get('rmse_ffbsi', math.nan):.6f}`, "
        f"well_mean/trim90 `{cand_full['well_rmse_mean']:.6f}/{cand_full['well_rmse_trim90_mean']:.6f}`; "
        f"{str(objective_metric)} metric gains `{_format_pf_opt_metric_gains(full_gains)}`; decision `{decision}`."
    )
    return {
        "sota": sota_full,
        "candidate": cand_full,
        "ok": bool(full_ok),
        "gains": full_gains,
        "base_seed": int(check_seed),
        "budget": (int(n_particles), int(n_seeds)),
    }


def _failure_guided_slice_ids_from_summary(summary, top_k=50):
    top_rows = summary.get("top_bad_pf_wells", []) or []
    ids = []
    for row in top_rows[: int(top_k)]:
        well_id = row.get("well_id")
        if well_id:
            ids.append(str(well_id))
    return ids


def _failure_guided_load_well_diagnostics(summary):
    artifacts = summary.get("artifacts", {}) or {}
    diag_path = artifacts.get("well_diagnostics")
    if not diag_path:
        output_dir = summary.get("output_dir")
        if output_dir:
            diag_path = str(Path(output_dir) / "well_diagnostics.csv")
    if not diag_path:
        return pd.DataFrame()
    diag_path = Path(diag_path)
    if not diag_path.exists():
        return pd.DataFrame()
    return pd.read_csv(diag_path)


def _failure_guided_top_wells(well_df, n_wells, score_col, ascending=False, mask=None):
    if well_df.empty or score_col not in well_df.columns:
        return []
    frame = well_df
    if mask is not None:
        frame = frame.loc[np.asarray(mask, dtype=bool)]
    frame = frame[np.isfinite(frame[score_col].to_numpy(dtype=np.float64))]
    if frame.empty:
        return []
    return (
        frame.sort_values(score_col, ascending=bool(ascending))["well_id"]
        .head(int(n_wells))
        .astype(str)
        .tolist()
    )


def _failure_guided_build_focus_slices(summary, sample_size=50):
    well_df = _failure_guided_load_well_diagnostics(summary)
    n_wells = max(1, int(sample_size))
    if well_df.empty:
        top_bad = _failure_guided_slice_ids_from_summary(summary, top_k=n_wells)
        return {
            "top_pf_failures": {
                "wells": top_bad,
                "reason": "highest cached PF RMSE from summary fallback",
            }
        }

    slices = {}
    top_bad = _failure_guided_top_wells(well_df, n_wells, "pf_rmse", ascending=False)
    slices["top_pf_failures"] = {
        "wells": top_bad,
        "reason": "highest cached PF RMSE",
    }

    pf_worse = well_df["pf_rmse"].to_numpy(dtype=np.float64) > well_df["anchor_rmse"].to_numpy(dtype=np.float64)
    anchor_losing = _failure_guided_top_wells(
        well_df,
        n_wells,
        "pf_minus_anchor_rmse",
        ascending=False,
        mask=pf_worse,
    )
    if anchor_losing:
        slices["anchor_losing_motif"] = {
            "wells": anchor_losing,
            "reason": "PF is worse than last-anchor, usually motif over-chase or drift away from the continuation prior",
        }

    if "suffix_gr_missing_frac" in well_df.columns:
        finite_mask = well_df["suffix_gr_missing_frac"].to_numpy(dtype=np.float64) <= 0.20
        finite_bad = _failure_guided_top_wells(well_df, n_wells, "pf_rmse", ascending=False, mask=finite_mask)
        if finite_bad:
            slices["finite_gr_bad"] = {
                "wells": finite_bad,
                "reason": "bad PF with finite suffix GR available, pointing at repeated-motif likelihood problems",
            }

        missing_mask = well_df["suffix_gr_missing_frac"].to_numpy(dtype=np.float64) >= 0.50
        missing_bad = _failure_guided_top_wells(well_df, n_wells, "pf_rmse", ascending=False, mask=missing_mask)
        if missing_bad:
            slices["missing_heavy_bad"] = {
                "wells": missing_bad,
                "reason": "bad PF with heavy suffix GR missingness, useful for continuity and jump/gap behavior",
            }

    diffuse = _failure_guided_top_wells(well_df, n_wells, "entropy_mean", ascending=False)
    if diffuse:
        slices["diffuse_posterior"] = {
            "wells": diffuse,
            "reason": "high posterior entropy/ESS and low confidence",
        }

    if {"anchor_mass15_mean", "mode_anchor_abs_mean"}.issubset(well_df.columns):
        tmp = well_df.copy()
        tmp["_low_anchor_mass_drift_score"] = (
            -tmp["anchor_mass15_mean"].to_numpy(dtype=np.float64)
            + 0.02 * tmp["mode_anchor_abs_mean"].to_numpy(dtype=np.float64)
        )
        low_anchor = _failure_guided_top_wells(tmp, n_wells, "_low_anchor_mass_drift_score", ascending=False)
        if low_anchor:
            slices["low_anchor_mass_drift"] = {
                "wells": low_anchor,
                "reason": "posterior mass is away from the anchor neighborhood",
            }

    bad_gr_col = "pf_gr_minus_anchor_gr_rmse" if "pf_gr_minus_anchor_gr_rmse" in well_df.columns else "pf_gr_rmse"
    bad_gr = _failure_guided_top_wells(well_df, n_wells, bad_gr_col, ascending=False)
    if bad_gr:
        slices["bad_gr_fit"] = {
            "wells": bad_gr,
            "reason": "PF path has poor Typewell-GR fitness or loses to anchor on GR fitness",
        }

    rough = _failure_guided_top_wells(well_df, n_wells, "pred_step_rms", ascending=False)
    if rough:
        slices["rough_path"] = {
            "wells": rough,
            "reason": "PF path roughness is high relative to the suffix continuity prior",
        }

    return {name: item for name, item in slices.items() if item.get("wells")}


def _pf_focus_value_equal(left, right):
    try:
        return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
    except TypeError:
        return left == right


def _failure_guided_delta_overrides(overrides, base_overrides=None):
    base_overrides = dict(base_overrides or PF_OPT_CURRENT_BEST_OVERRIDES)
    delta = {}
    for key, value in dict(overrides or {}).items():
        if key not in base_overrides or not _pf_focus_value_equal(value, base_overrides[key]):
            delta[key] = value
    return delta


def _failure_guided_candidate_focus(name, overrides, base_overrides=None):
    name = str(name)
    overrides = _failure_guided_delta_overrides(overrides, base_overrides=base_overrides)
    has_motif_temper = (
        float(overrides.get("PF_heatmap_gr_ambiguity_power", 0.0)) > 0.0
        or float(overrides.get("PF_heatmap_finite_run_power_decay", 0.0)) > 0.0
    )
    has_anchor_repair = (
        "PF_heatmap_anchor_sigma" in overrides
        or "PF_heatmap_anchor_power" in overrides
        or "anchor" in name
        or "lowmass" in name
        or "low_anchor" in name
    )
    dynamic_keys = {
        "PF_heatmap_momentum",
        "PF_heatmap_rate_noise",
        "PF_heatmap_pos_noise",
        "PF_heatmap_rough_pos",
        "PF_heatmap_rough_rate",
        "PF_heatmap_jump_prob",
        "PF_heatmap_jump_sd",
        "PF_heatmap_jump_rate_sd",
        "PF_heatmap_init_pos_sd",
        "PF_heatmap_init_rate_sd",
    }
    has_dynamic_repair = bool(dynamic_keys.intersection(overrides))
    if has_anchor_repair and (has_motif_temper or has_dynamic_repair or "lowmass" in name):
        return "low_anchor_mass_drift"
    if (
        has_motif_temper
    ):
        return "finite_gr_bad"
    if (
        float(overrides.get("PF_heatmap_dynamic_sigma_alpha", 0.0)) > 0.0
        or float(overrides.get("PF_heatmap_outlier_prob", 0.0)) > 0.0
    ):
        return "bad_gr_fit"
    if float(overrides.get("PF_heatmap_state_z_weight", 1.0)) < 0.999:
        return "rough_path"
    if (
        "PF_heatmap_seen_blend_weight" in overrides
        or "PF_heatmap_raw_ref_likelihood_weight" in overrides
        or "raw" in name
        or "seen" in name
    ):
        return "bad_gr_fit"
    if has_anchor_repair:
        return "anchor_losing_motif"
    if has_dynamic_repair:
        return "rough_path"
    if float(overrides.get("PF_heatmap_lik_scale", 5.0)) != 5.0:
        return "diffuse_posterior"
    return "top_pf_failures"


def _failure_guided_direction_note(sample_summary):
    if not sample_summary:
        return "no_sample"
    focus_gain = float(sample_summary.get("focus_gain", math.nan))
    context_gain = float(sample_summary.get("context_gain", math.nan))
    combined_gain = float(sample_summary.get("combined_gain", math.nan))
    if not np.isfinite(focus_gain):
        return "no_focus_score"
    if focus_gain <= 0.0:
        return "no_focus_gain"
    if np.isfinite(context_gain) and context_gain < 0.0:
        return f"directed_gain_context_loss={-context_gain:.6f}"
    if np.isfinite(combined_gain) and combined_gain > 0.0:
        return "directed_gain_context_ok"
    return "directed_gain"


def _failure_guided_context_loss_limit(gate_margin):
    return max(0.05, 2.0 * float(gate_margin))


def _select_pf_failure_guided_best(sample_results, gate_margin=0.0, context_loss_limit=0.05):
    passing = []
    best = None
    for item in sample_results:
        focus_gain = float(item.get("focus_gain", -math.inf))
        combined_gain = float(item.get("combined_gain", -math.inf))
        context_gain = float(item.get("context_gain", -math.inf))
        focus_gate_ok = bool(item.get("focus_gate_ok", False))
        context_ok = context_gain >= -float(context_loss_limit)
        combined_ok = combined_gain > 0.0
        gate_ok = focus_gate_ok and focus_gain >= float(gate_margin) and context_ok and combined_ok
        item["context_gate_ok"] = bool(context_ok)
        item["combined_gate_ok"] = bool(combined_ok)
        item["directed_gate_ok"] = bool(gate_ok)
        if gate_ok:
            passing.append(item)
        if best is None or combined_gain > float(best.get("combined_gain", -math.inf)):
            best = item
    if passing:
        passing.sort(
            key=lambda item: (
                -float(item.get("combined_gain", -math.inf)),
                -float(item.get("focus_gain", -math.inf)),
                float(item.get("context_rmse", math.inf)),
            )
        )
        return passing[0], True, passing
    return best, False, []


def _format_failure_focus_slices(focus_slices, limit=8):
    parts = []
    for name, item in list(focus_slices.items())[: int(limit)]:
        parts.append(f"{name}:{len(item.get('wells', []))}")
    return ", ".join(parts)


def _format_failure_guided_ranking(summaries, limit=3):
    ordered = sorted(summaries, key=lambda item: float(item.get("combined_gain", -math.inf)), reverse=True)
    return ", ".join(
        f"{item['name']}[{item.get('focus_name', 'na')}]="
        f"fgain {float(item.get('focus_gain', math.nan)):.6f}/"
        f"cgain {float(item.get('context_gain', math.nan)):.6f}/"
        f"mix {float(item.get('combined_gain', math.nan)):.6f}"
        for item in ordered[: int(limit)]
    )


def _select_pf_opt_gated_best(baseline, sample_results, global_margin=0.0, aux_margin=0.0, objective_metric="direct"):
    passing = []
    for item in sample_results:
        item_baseline = item.get("_baseline_summary", baseline)
        ok, gains = _pf_opt_passes_metric_gate(
            item_baseline,
            item,
            global_margin=global_margin,
            aux_margin=aux_margin,
            objective_metric=objective_metric,
        )
        if ok:
            passing.append((item, gains))
    if not passing:
        objective_key = _pf_opt_objective_key(objective_metric)
        best = min(sample_results, key=lambda item: _pf_opt_metric_value(item, objective_key))
        best_baseline = best.get("_baseline_summary", baseline)
        _, gains = _pf_opt_passes_metric_gate(
            best_baseline,
            best,
            global_margin=global_margin,
            aux_margin=aux_margin,
            objective_metric=objective_metric,
        )
        return best, gains, False, []
    objective_key = _pf_opt_objective_key(objective_metric)
    best, gains = min(passing, key=lambda pair: _pf_opt_metric_value(pair[0], objective_key))
    return best, gains, True, passing


PF_FAILURE_GUIDED_CURATED_CANDIDATES = (
    (
        "fg_resample_adapt030_min035",
        {
            "PF_heatmap_resample_obs_power_adapt": 0.30,
            "PF_heatmap_resample_min_threshold": 0.35,
        },
        "Delay ESS resampling when correlated/tempered observations carry low independent information.",
    ),
    (
        "fg_rescue020_s8",
        {
            "PF_heatmap_rescue_frac": 0.02,
            "PF_heatmap_rescue_pos_sd": 8.0,
            "PF_heatmap_rescue_rate_sd": 0.004,
        },
        "Inject a small continuation-prior rescue cloud after collapse so motif-trap failures can recover.",
    ),
    (
        "fg_resample_adapt_rescue",
        {
            "PF_heatmap_resample_obs_power_adapt": 0.30,
            "PF_heatmap_resample_min_threshold": 0.35,
            "PF_heatmap_rescue_frac": 0.02,
            "PF_heatmap_rescue_pos_sd": 8.0,
            "PF_heatmap_rescue_rate_sd": 0.004,
        },
        "Combine less eager resampling under correlated evidence with a small continuation rescue cloud.",
    ),
    (
        "fg_frdecay050_floor055",
        {
            "PF_heatmap_finite_run_power_decay": 0.50,
            "PF_heatmap_finite_run_power_floor": 0.55,
        },
        "Temper long finite-GR runs so repeated 1 ft correlated evidence cannot dominate the state prior.",
    ),
    (
        "fg_amb035_min045",
        {
            "PF_heatmap_gr_ambiguity_power": 0.35,
            "PF_heatmap_gr_ambiguity_min_power": 0.45,
        },
        "Downweight point observations in locally repeated Typewell-GR motifs without removing GR evidence.",
    ),
    (
        "fg_amb050_raw",
        {
            "PF_heatmap_gr_ambiguity_power": 0.50,
            "PF_heatmap_gr_ambiguity_min_power": 0.35,
            "PF_heatmap_gr_ambiguity_ref_mode": "raw",
        },
        "Measure motif ambiguity on the raw paired Typewell so seen-prefix blending does not hide repeats.",
    ),
    (
        "fg_lowmass_anchor_amb035",
        {
            "PF_heatmap_anchor_sigma": 65.0,
            "PF_heatmap_anchor_power": 0.0025,
            "PF_heatmap_gr_ambiguity_power": 0.35,
            "PF_heatmap_gr_ambiguity_min_power": 0.45,
        },
        "Repair low anchor-neighborhood posterior mass by tempering repeated motifs while keeping a soft continuation prior.",
    ),
    (
        "fg_lowmass_stiff_frdecay",
        {
            "PF_heatmap_anchor_sigma": 65.0,
            "PF_heatmap_anchor_power": 0.0025,
            "PF_heatmap_finite_run_power_decay": 0.50,
            "PF_heatmap_finite_run_power_floor": 0.55,
            "PF_heatmap_momentum": 0.99950,
            "PF_heatmap_rate_noise": 0.0009,
            "PF_heatmap_pos_noise": 0.0035,
            "PF_heatmap_rough_pos": 0.06,
            "PF_heatmap_rough_rate": 0.0045,
        },
        "Target low anchor mass plus rough drift with correlated-run tempering and stiffer in-PF motion.",
    ),
    (
        "fg_anchor_s65_p0025",
        {
            "PF_heatmap_anchor_sigma": 65.0,
            "PF_heatmap_anchor_power": 0.0025,
        },
        "Use a slightly stronger soft anchor likelihood inside PF for motif-drift failures.",
    ),
    (
        "fg_anchor_frdecay",
        {
            "PF_heatmap_anchor_sigma": 65.0,
            "PF_heatmap_anchor_power": 0.0025,
            "PF_heatmap_finite_run_power_decay": 0.50,
            "PF_heatmap_finite_run_power_floor": 0.55,
        },
        "Combine a stronger anchor likelihood with correlated-run tempering for anchor-disagreement failures.",
    ),
    (
        "fg_stiff_anchor",
        {
            "PF_heatmap_anchor_sigma": 65.0,
            "PF_heatmap_anchor_power": 0.0025,
            "PF_heatmap_momentum": 0.99950,
            "PF_heatmap_rate_noise": 0.0009,
            "PF_heatmap_pos_noise": 0.0035,
            "PF_heatmap_rough_pos": 0.06,
            "PF_heatmap_rough_rate": 0.0045,
        },
        "Tighten motion dynamics when PF posterior roughness and anchor drift flag misspecification.",
    ),
    (
        "fg_z0925_raw020",
        {
            "PF_heatmap_state_z_weight": 0.925,
            "PF_heatmap_rate_noise": 0.0012,
            "PF_heatmap_rough_rate": 0.0070,
            "PF_heatmap_raw_ref_likelihood_weight": 0.20,
            "PF_heatmap_seen_blend_weight": 0.65,
        },
        "Partial direct-TVT state plus raw-reference support for TVT+Z state misspecification cases.",
    ),
    (
        "fg_seenlight_raw020",
        {
            "PF_heatmap_seen_blend_weight": 0.55,
            "PF_heatmap_raw_ref_likelihood_weight": 0.20,
        },
        "Reduce same-well pseudo-Typewell dominance when calibrated reference fit looks unreliable.",
    ),
    (
        "fg_seenheavy_raw010",
        {
            "PF_heatmap_seen_blend_weight": 0.85,
            "PF_heatmap_raw_ref_likelihood_weight": 0.10,
        },
        "Lean harder on same-well prefix texture for raw paired-Typewell mismatch cases.",
    ),
    (
        "fg_dynsigma010",
        {
            "PF_heatmap_dynamic_sigma_alpha": 0.010,
            "PF_heatmap_dynamic_sigma_threshold": 1.35,
            "PF_heatmap_dynamic_sigma_power": 0.60,
            "PF_heatmap_dynamic_sigma_min": 0.90,
            "PF_heatmap_dynamic_sigma_max": 1.75,
        },
        "Mild posterior-predictive sigma adaptation for high PF-vs-observed-GR residual cases.",
    ),
    (
        "fg_outlier005",
        {
            "PF_heatmap_outlier_prob": 0.005,
            "PF_heatmap_outlier_likelihood": 0.05,
        },
        "Add a small contaminated likelihood for isolated bad GR readings without broad sigma inflation.",
    ),
    (
        "fg_lik45_raw020",
        {
            "PF_heatmap_lik_scale": 4.5,
            "PF_heatmap_raw_ref_likelihood_weight": 0.20,
            "PF_heatmap_seen_blend_weight": 0.65,
        },
        "Sharper seed likelihood selection plus more raw-reference support for diffuse posterior cases.",
    ),
)


PF_FAILURE_GUIDED_FOCUS_SPACES = {
    "finite_gr_bad": {
        "PF_heatmap_gr_ambiguity_power": [0.25, 0.35, 0.50, 0.65],
        "PF_heatmap_gr_ambiguity_min_power": [0.35, 0.45, 0.55],
        "PF_heatmap_gr_ambiguity_ref_mode": ["primary", "raw"],
        "PF_heatmap_finite_run_power_decay": [0.25, 0.50, 1.00],
        "PF_heatmap_finite_run_power_floor": [0.50, 0.55, 0.60],
        "PF_heatmap_resample_obs_power_adapt": [0.0, 0.15, 0.30, 0.60],
        "PF_heatmap_resample_min_threshold": [0.30, 0.35, 0.40],
        "PF_heatmap_lik_scale": [4.5, 5.0, 5.5],
    },
    "anchor_losing_motif": {
        "PF_heatmap_anchor_sigma": [55.0, 60.0, 65.0, 70.0],
        "PF_heatmap_anchor_power": [0.0020, 0.0025, 0.0030, 0.0035],
        "PF_heatmap_momentum": [0.99940, 0.99945, 0.99950],
        "PF_heatmap_rate_noise": [0.0008, 0.0009, 0.0011],
        "PF_heatmap_pos_noise": [0.0030, 0.0035, 0.0040],
        "PF_heatmap_rough_pos": [0.055, 0.060, 0.080],
        "PF_heatmap_rough_rate": [0.0040, 0.0045, 0.0050],
        "PF_heatmap_resample_obs_power_adapt": [0.0, 0.15, 0.30],
        "PF_heatmap_resample_min_threshold": [0.30, 0.35, 0.40],
        "PF_heatmap_rescue_frac": [0.0, 0.01, 0.02, 0.05],
        "PF_heatmap_rescue_pos_sd": [4.0, 8.0, 12.0],
        "PF_heatmap_rescue_rate_sd": [0.002, 0.004, 0.006],
    },
    "bad_gr_fit": {
        "PF_heatmap_raw_ref_likelihood_weight": [0.10, 0.15, 0.20, 0.25],
        "PF_heatmap_seen_blend_weight": [0.55, 0.65, 0.70, 0.75, 0.85],
        "PF_heatmap_lik_scale": [4.5, 5.0, 5.5],
        "PF_heatmap_outlier_prob": [0.0, 0.005, 0.010],
        "PF_heatmap_dynamic_sigma_alpha": [0.0, 0.005, 0.010],
    },
    "diffuse_posterior": {
        "PF_heatmap_lik_scale": [4.0, 4.5, 5.0, 5.5, 6.0],
        "PF_heatmap_resample_threshold": [0.35, 0.375, 0.40, 0.425],
        "PF_heatmap_resample_obs_power_adapt": [0.0, 0.15, 0.30, 0.60],
        "PF_heatmap_resample_min_threshold": [0.30, 0.35, 0.40],
        "PF_heatmap_rescue_frac": [0.0, 0.01, 0.02],
        "PF_heatmap_rescue_pos_sd": [4.0, 8.0],
        "PF_heatmap_rescue_rate_sd": [0.002, 0.004],
        "PF_heatmap_init_pos_sd": [1.0, 1.25, 1.5, 2.0],
        "PF_heatmap_init_rate_sd": [0.0125, 0.0150, 0.0200, 0.0225],
        "PF_heatmap_rough_pos": [0.060, 0.080, 0.100],
        "PF_heatmap_rough_rate": [0.0045, 0.0050, 0.0060],
    },
    "low_anchor_mass_drift": {
        "PF_heatmap_anchor_sigma": [55.0, 60.0, 65.0, 70.0],
        "PF_heatmap_anchor_power": [0.0020, 0.0025, 0.0030],
        "PF_heatmap_gr_ambiguity_power": [0.0, 0.25, 0.35, 0.50],
        "PF_heatmap_gr_ambiguity_min_power": [0.35, 0.45, 0.55],
        "PF_heatmap_finite_run_power_decay": [0.0, 0.25, 0.50],
        "PF_heatmap_finite_run_power_floor": [0.55, 0.60],
        "PF_heatmap_momentum": [0.99940, 0.99945, 0.99950],
        "PF_heatmap_rate_noise": [0.0008, 0.0009, 0.0011],
        "PF_heatmap_pos_noise": [0.0030, 0.0035, 0.0040],
        "PF_heatmap_rough_pos": [0.055, 0.060, 0.080],
        "PF_heatmap_rough_rate": [0.0040, 0.0045, 0.0050],
        "PF_heatmap_resample_obs_power_adapt": [0.0, 0.15, 0.30],
        "PF_heatmap_resample_min_threshold": [0.30, 0.35, 0.40],
        "PF_heatmap_rescue_frac": [0.0, 0.01, 0.02, 0.05],
        "PF_heatmap_rescue_pos_sd": [4.0, 8.0, 12.0],
        "PF_heatmap_rescue_rate_sd": [0.002, 0.004, 0.006],
    },
    "rough_path": {
        "PF_heatmap_state_z_weight": [0.90, 0.925, 0.95, 0.975, 1.0],
        "PF_heatmap_momentum": [0.99935, 0.99940, 0.99945, 0.99950],
        "PF_heatmap_rate_noise": [0.0008, 0.0009, 0.0011, 0.0012],
        "PF_heatmap_pos_noise": [0.0030, 0.0035, 0.0040, 0.0045],
        "PF_heatmap_rough_pos": [0.055, 0.060, 0.080],
        "PF_heatmap_rough_rate": [0.0040, 0.0045, 0.0050, 0.0060],
        "PF_heatmap_rescue_frac": [0.0, 0.01, 0.02],
        "PF_heatmap_rescue_pos_sd": [4.0, 8.0],
        "PF_heatmap_rescue_rate_sd": [0.002, 0.004],
        "PF_heatmap_jump_prob": [0.000075, 0.00010, 0.00015],
        "PF_heatmap_jump_sd": [6.0, 7.0, 8.0, 9.0],
        "PF_heatmap_jump_rate_sd": [0.00025, 0.00030, 0.00040],
    },
    "missing_heavy_bad": {
        "PF_heatmap_momentum": [0.99935, 0.99940, 0.99945],
        "PF_heatmap_rate_noise": [0.0009, 0.0011, 0.0012, 0.0014],
        "PF_heatmap_pos_noise": [0.0035, 0.0040, 0.0045, 0.0050],
        "PF_heatmap_jump_prob": [0.00010, 0.00015, 0.00020],
        "PF_heatmap_jump_sd": [7.0, 8.0, 9.0],
        "PF_heatmap_jump_rate_sd": [0.00030, 0.00040, 0.00050],
        "PF_heatmap_anchor_sigma": [65.0, 70.0, 75.0, 80.0],
        "PF_heatmap_anchor_power": [0.0015, 0.0020, 0.0025],
    },
    "top_pf_failures": {
        "PF_heatmap_raw_ref_likelihood_weight": [0.10, 0.15, 0.20],
        "PF_heatmap_seen_blend_weight": [0.65, 0.70, 0.75],
        "PF_heatmap_lik_scale": [4.5, 5.0, 5.5],
        "PF_heatmap_anchor_sigma": [65.0, 70.0, 75.0],
        "PF_heatmap_anchor_power": [0.0020, 0.0025],
        "PF_heatmap_finite_run_power_decay": [0.0, 0.25, 0.50],
        "PF_heatmap_finite_run_power_floor": [0.55, 0.60],
        "PF_heatmap_resample_obs_power_adapt": [0.0, 0.15, 0.30],
        "PF_heatmap_resample_min_threshold": [0.30, 0.35, 0.40],
        "PF_heatmap_rescue_frac": [0.0, 0.01, 0.02],
        "PF_heatmap_rescue_pos_sd": [4.0, 8.0],
        "PF_heatmap_rescue_rate_sd": [0.002, 0.004],
    },
}


PF_FAILURE_GUIDED_FOCUS_CHOICES = (
    "finite_gr_bad",
    "anchor_losing_motif",
    "bad_gr_fit",
    "diffuse_posterior",
    "low_anchor_mass_drift",
    "rough_path",
    "missing_heavy_bad",
    "top_pf_failures",
)


PF_OPT_RANDOM_SEARCH_SPACE = {
    # Tight local surface around the accepted fresh-seed rs0003_rand07 chain.
    # Broad profile plans and high-variance observation hooks are handled by the
    # failure-guided loop; this random loop is for one/two-knob confirmation
    # around finite-run tempering, raw/seen reference balance, resampling, and
    # mild local dynamics.
    "PF_heatmap_raw_ref_likelihood_weight": [0.05, 0.075, 0.075, 0.10, 0.10, 0.125],
    "PF_heatmap_seen_blend_weight": [0.70, 0.75, 0.75, 0.80],
    "PF_heatmap_lik_scale": [5.25, 5.50, 5.50, 5.75, 6.00],
    "PF_heatmap_state_z_weight": [0.975, 1.0, 1.0],
    "PF_heatmap_momentum": [0.99940, 0.99945, 0.99945, 0.99950],
    "PF_heatmap_rate_noise": [0.0008, 0.0009, 0.0009, 0.0011],
    "PF_heatmap_pos_noise": [0.0030, 0.0035, 0.0035, 0.0042],
    "PF_heatmap_rate_mean_weight": [0.00075, 0.00100, 0.00100, 0.00125],
    "PF_heatmap_init_pos_sd": [1.5, 2.0, 2.0, 2.5],
    "PF_heatmap_init_rate_sd": [0.0150, 0.0200, 0.0200, 0.0250],
    "PF_heatmap_resample_threshold": [0.35, 0.35, 0.375, 0.40, 0.425],
    "PF_heatmap_resample_obs_power_adapt": [0.30, 0.60, 0.60],
    "PF_heatmap_resample_min_threshold": [0.30, 0.35, 0.35, 0.40],
    "PF_heatmap_rough_pos": [0.060, 0.080, 0.080, 0.100],
    "PF_heatmap_rough_rate": [0.0045, 0.0060, 0.0060, 0.0075],
    "PF_heatmap_anchor_sigma": [75.0, 80.0, 80.0, 85.0, 90.0],
    "PF_heatmap_anchor_power": [0.00175, 0.00200, 0.00200, 0.00250],
    "PF_heatmap_jump_prob": [0.00010, 0.00015, 0.00015, 0.000225],
    "PF_heatmap_missing_jump_boost": [0.00005, 0.00010, 0.00015, 0.00015],
    "PF_heatmap_jump_sd": [6.5, 7.0, 7.0, 8.5],
    "PF_heatmap_jump_rate_sd": [0.00030, 0.00040, 0.00040, 0.00060],
    "PF_heatmap_finite_run_power_boost": [0.025, 0.030, 0.040, 0.060, 0.060],
    "PF_heatmap_finite_run_power_decay": [0.25, 0.50, 0.50, 0.75, 1.00],
    "PF_heatmap_finite_run_power_floor": [0.50, 0.55, 0.55, 0.60, 0.70],
    "PF_heatmap_conf_obs_power_decay": [0.0, 0.0, 0.50, 1.00, 1.50],
    "PF_heatmap_conf_obs_power_floor": [0.55, 0.60, 0.70, 0.80],
    "PF_heatmap_conf_obs_power_power": [1.0, 1.0, 1.5, 2.0],
    "PF_heatmap_missing_noise_scale": [0.90, 1.0, 1.0, 1.10],
    "PF_heatmap_missing_jump_scale": [0.90, 1.0, 1.0, 1.10],
    "PF_heatmap_ess_jump_boost": [0.0, 0.0, 0.000025, 0.000050, 0.000075],
    "PF_heatmap_ess_jump_power": [1.0, 1.0, 1.5, 2.0],
    "PF_heatmap_surprise_jump_boost": [0.0, 0.0, 0.000020, 0.000040, 0.000060],
    "PF_heatmap_surprise_jump_threshold": [1.25, 1.50, 2.00, 2.50],
    "PF_heatmap_surprise_jump_power": [1.0, 1.0, 1.5],
    "PF_heatmap_jump_tail_prob": [0.0, 0.0, 0.000010, 0.000025, 0.000050],
    "PF_heatmap_jump_tail_sd": [8.0, 12.0, 18.0, 24.0],
    "PF_heatmap_jump_tail_rate_sd": [0.0004, 0.0008, 0.0015, 0.0025],
    "PF_heatmap_jump_tail_dist": [1, 1, 2],
    "PF_heatmap_jump_tail_clip": [24.0, 36.0, 60.0, 90.0],
    "PF_heatmap_jump_tail_missing_boost": [0.0, 0.000010, 0.000020, 0.000040],
    "PF_heatmap_gr_ambiguity_power": [0.0, 0.0, 0.25, 0.35],
    "PF_heatmap_gr_ambiguity_min_power": [0.45, 0.55],
    "PF_heatmap_gr_ambiguity_ref_mode": ["primary", "raw"],
    "PF_heatmap_gr_information_power": [0.0, 0.0, 0.25, 0.35],
    "PF_heatmap_gr_information_center": [0.65, 0.70, 0.75],
    "PF_heatmap_gr_information_min_multiplier": [0.65, 0.70, 0.75],
    "PF_heatmap_gr_information_max_multiplier": [1.15, 1.25, 1.35],
    "PF_heatmap_gr_information_slope_weight": [0.0, 0.20, 0.35],
    "PF_heatmap_outlier_prob": [0.0, 0.0, 0.005, 0.010],
    "PF_heatmap_dynamic_sigma_alpha": [0.0, 0.0, 0.005, 0.010],
    "PF_heatmap_dynamic_sigma_threshold": [1.25, 1.35, 1.50],
    "PF_heatmap_dynamic_sigma_power": [0.60, 0.75, 1.00],
    "PF_heatmap_dynamic_sigma_min": [0.85, 0.90],
    "PF_heatmap_dynamic_sigma_max": [1.75, 2.00],
    "PF_heatmap_lookahead_power": [0.0, 0.0, 0.00035, 0.00070],
    "PF_heatmap_lookahead_steps": [1, 2, 2],
    "PF_heatmap_lookahead_decay": [0.30, 0.50],
    "PF_heatmap_lookahead_max_gap": [48.0, 64.0, 96.0],
    "PF_heatmap_lookahead_delta_power": [0.0, 0.25, 0.50],
    "PF_heatmap_ffbsi_mode": ["fallback"],
    "PF_heatmap_ffbsi_n_paths": [32, 64, 64, 96],
    "PF_heatmap_ffbsi_fallback_mode": ["filtered"],
    "PF_heatmap_ffbsi_max_active_bins": [512, 512, 768],
    "PF_heatmap_ffbsi_transition_scale": [1.20, 1.35, 1.50, 1.65],
    "PF_heatmap_ffbsi_pos_floor": [0.15, 0.20, 0.25, 0.35],
    "PF_heatmap_ffbsi_rate_floor": [0.0018, 0.0025, 0.0040],
}


PF_OPT_RANDOM_KEY_DIMS = (
    "PF_heatmap_jump_prob",
    "PF_heatmap_jump_sd",
    "PF_heatmap_jump_rate_sd",
    "PF_heatmap_missing_jump_boost",
    "PF_heatmap_momentum",
    "PF_heatmap_rate_noise",
    "PF_heatmap_pos_noise",
    "PF_heatmap_rate_mean_weight",
    "PF_heatmap_init_pos_sd",
    "PF_heatmap_init_rate_sd",
    "PF_heatmap_resample_threshold",
    "PF_heatmap_resample_obs_power_adapt",
    "PF_heatmap_resample_min_threshold",
    "PF_heatmap_rough_pos",
    "PF_heatmap_rough_rate",
    "PF_heatmap_lik_scale",
    "PF_heatmap_state_z_weight",
    "PF_heatmap_raw_ref_likelihood_weight",
    "PF_heatmap_seen_blend_weight",
    "PF_heatmap_finite_run_power_decay",
    "PF_heatmap_conf_obs_power_decay",
    "PF_heatmap_ess_jump_boost",
    "PF_heatmap_surprise_jump_boost",
    "PF_heatmap_jump_tail_prob",
    "PF_heatmap_gr_ambiguity_power",
    "PF_heatmap_gr_information_power",
    "PF_heatmap_outlier_prob",
    "PF_heatmap_dynamic_sigma_alpha",
    "PF_heatmap_lookahead_power",
    "PF_heatmap_ffbsi_transition_scale",
    "PF_heatmap_missing_noise_scale",
    "PF_heatmap_missing_jump_scale",
)


PF_OPT_STRUCTURAL_KEY_DIMS = (
    "PF_heatmap_raw_ref_likelihood_weight",
    "PF_heatmap_seen_blend_weight",
    "PF_heatmap_finite_run_power_decay",
    "PF_heatmap_finite_run_power_floor",
    "PF_heatmap_conf_obs_power_decay",
    "PF_heatmap_conf_obs_power_floor",
    "PF_heatmap_resample_threshold",
    "PF_heatmap_resample_obs_power_adapt",
    "PF_heatmap_ffbsi_transition_scale",
)


PF_OPT_STRUCTURAL_ACTIVE_VALUES = {
    "PF_heatmap_raw_ref_likelihood_weight": [0.05, 0.075, 0.075, 0.10, 0.10, 0.125],
    "PF_heatmap_seen_blend_weight": [0.70, 0.75, 0.75, 0.80],
    "PF_heatmap_finite_run_power_decay": [0.25, 0.50, 0.75, 1.00, 1.00],
    "PF_heatmap_finite_run_power_floor": [0.55, 0.60, 0.70, 0.70],
    "PF_heatmap_conf_obs_power_decay": [0.0, 0.50, 1.00, 1.50],
    "PF_heatmap_conf_obs_power_floor": [0.55, 0.60, 0.70, 0.80],
    "PF_heatmap_resample_threshold": [0.35, 0.35, 0.375, 0.40, 0.425],
    "PF_heatmap_resample_obs_power_adapt": [0.30, 0.60, 0.60],
    "PF_heatmap_ffbsi_transition_scale": [1.20, 1.35, 1.50, 1.65],
}


PF_OPT_RANDOM_SIDE_DIMS = tuple(
    key for key in PF_OPT_RANDOM_SEARCH_SPACE if key not in set(PF_OPT_RANDOM_KEY_DIMS)
) + ("PF_heatmap_profile_mixture_spec",)


def _pf_opt_finite_run_floor_values(decay):
    decay = float(decay)
    if decay <= 0.20:
        return (0.50, 0.55, 0.55, 0.60)
    if decay <= 0.35:
        return (0.50, 0.55, 0.55, 0.60, 0.70)
    if decay <= 0.60:
        return (0.45, 0.50, 0.55, 0.55, 0.60, 0.70)
    return (0.45, 0.50, 0.55, 0.60, 0.70)


def _pf_random_search_space_note(last_round):
    if not last_round:
        lead = "starting from accepted rs0003_rand07 fresh-seed filtered PF with a conservative local grid"
    elif last_round["status"] == "no_sample_gate":
        lead = "last sample had no gated winner, resampling wells and keeping compact local mutations"
    elif last_round["status"] == "full_rejected":
        lead = "last confirmed sample winner failed paired full check, keeping one/two-knob mutations"
    elif last_round["status"] == "confirm_rejected":
        lead = "last sample winner failed repeat-shard confirmation, keeping local mutations conservative"
    else:
        lead = "last paired full check accepted a new base, recentering compact mutations around that base"
    return (
        f"{lead}; active candidates are anchored on rs0003_rand07 and move mostly across raw/seen "
        "reference balance, likelihood softness, finite-run evidence tempering, adaptive resampling, "
        "anchor sigma/power, particle mobility, local jumps, roughening, and mild missing-gap mobility. "
        "GR-shape/NCC, lookahead-heavy, coarser-grid, large outlier/dynamic-sigma, and non-likelihood "
        "seed-weighting hooks remain manual diagnostics rather than active default axes. Sample winners "
        "must pass a disjoint repeat-shard confirmation before paired fresh-seed full validation."
    )


PF_OPT_MERGE16_RANDOM_SEARCH_SPACE = {
    # Merge-specific axes first. The 50-well tuning showed that preserving
    # row evidence inside each block is the right abstraction, but evidence
    # strength can overfit small slices, so keep a tight alpha grid around 1.
    "PF_heatmap_suffix_merge_obs_power_alpha": [0.95, 1.0, 1.0, 1.05],
    # Observation/reference balance around merge16 SOTA.
    "PF_heatmap_lik_scale": [5.25, 5.50, 5.50, 5.75],
    "PF_heatmap_raw_ref_likelihood_weight": [0.05, 0.075, 0.075, 0.10, 0.125],
    "PF_heatmap_seen_blend_weight": [0.70, 0.75, 0.75, 0.80],
    "PF_heatmap_finite_run_power_boost": [0.040, 0.060, 0.060],
    "PF_heatmap_finite_run_power_decay": [0.25, 0.50, 1.00, 1.00],
    "PF_heatmap_finite_run_power_floor": [0.50, 0.55, 0.60],
    "PF_heatmap_conf_obs_power_decay": [0.0, 0.0, 0.25, 0.50, 1.00],
    "PF_heatmap_conf_obs_power_floor": [0.55, 0.60],
    "PF_heatmap_conf_obs_power_power": [1.0],
    # Step dynamics and roughening. Keep jump probability on the row-rate scale;
    # explicit k-fold jump transforms were already rejected for merged updates.
    "PF_heatmap_momentum": [0.99940, 0.99945, 0.99945, 0.99950],
    "PF_heatmap_rate_noise": [0.00090, 0.00100, 0.00110],
    "PF_heatmap_pos_noise": [0.0030, 0.0035, 0.0040],
    "PF_heatmap_rate_mean_weight": [0.00075, 0.00100, 0.00125],
    "PF_heatmap_resample_threshold": [0.35, 0.375, 0.40],
    "PF_heatmap_resample_obs_power_adapt": [0.30, 0.45, 0.60],
    "PF_heatmap_resample_min_threshold": [0.30, 0.35],
    "PF_heatmap_rough_pos": [0.060, 0.080, 0.100],
    "PF_heatmap_rough_rate": [0.0045, 0.0060, 0.0075],
    "PF_heatmap_init_pos_sd": [1.5, 2.0, 2.5],
    "PF_heatmap_init_rate_sd": [0.0150, 0.0200, 0.0250],
    "PF_heatmap_jump_prob": [0.00010, 0.00015, 0.00020],
    "PF_heatmap_missing_jump_boost": [0.00005, 0.00010, 0.00015],
    "PF_heatmap_jump_sd": [7.0, 8.0, 9.0],
    "PF_heatmap_jump_rate_sd": [0.00030, 0.00040, 0.00055],
    "PF_heatmap_anchor_sigma": [75.0, 80.0, 85.0],
    "PF_heatmap_anchor_power": [0.00175, 0.00200, 0.00225],
    # Reasonable legacy axes retained with low sampling pressure.
    "PF_heatmap_state_z_weight": [0.975, 1.0, 1.0],
    "PF_heatmap_gr_ambiguity_power": [0.0, 0.0, 0.25],
    "PF_heatmap_gr_ambiguity_min_power": [0.45, 0.55],
    "PF_heatmap_gr_ambiguity_ref_mode": ["primary", "raw"],
    "PF_heatmap_gr_information_power": [0.0, 0.25],
    "PF_heatmap_gr_information_center": [0.70, 0.75],
    "PF_heatmap_gr_information_min_multiplier": [0.70, 0.75],
    "PF_heatmap_gr_information_max_multiplier": [1.15, 1.25],
    "PF_heatmap_gr_information_slope_weight": [0.0, 0.20],
    "PF_heatmap_outlier_prob": [0.0, 0.0, 0.005],
    "PF_heatmap_dynamic_sigma_alpha": [0.0, 0.0, 0.005],
    "PF_heatmap_dynamic_sigma_threshold": [1.25, 1.35],
    "PF_heatmap_dynamic_sigma_power": [0.60],
    "PF_heatmap_dynamic_sigma_min": [0.85, 0.90],
    "PF_heatmap_dynamic_sigma_max": [1.75],
    "PF_heatmap_missing_noise_scale": [0.90, 1.0, 1.0, 1.10],
    "PF_heatmap_missing_jump_scale": [0.90, 1.0, 1.0, 1.10],
    "PF_heatmap_profile_mixture_spec": tuple(
        name for name in PF_OPT_MERGE16_PROFILE_MIXTURE_NAMES if name in PF_OPT_PROFILE_MIXTURES
    )
    or tuple(PF_OPT_PROFILE_MIXTURES),
}


PF_OPT_MERGE16_KEY_DIMS = (
    "PF_heatmap_suffix_merge_obs_power_alpha",
    "PF_heatmap_lik_scale",
    "PF_heatmap_raw_ref_likelihood_weight",
    "PF_heatmap_seen_blend_weight",
    "PF_heatmap_finite_run_power_decay",
    "PF_heatmap_conf_obs_power_decay",
    "PF_heatmap_resample_threshold",
    "PF_heatmap_resample_obs_power_adapt",
    "PF_heatmap_momentum",
    "PF_heatmap_rate_noise",
    "PF_heatmap_pos_noise",
    "PF_heatmap_rough_pos",
    "PF_heatmap_rough_rate",
    "PF_heatmap_jump_prob",
    "PF_heatmap_jump_sd",
    "PF_heatmap_anchor_power",
    "PF_heatmap_profile_mixture_spec",
)


PF_OPT_MERGE16_SIDE_DIMS = (
    "PF_heatmap_finite_run_power_boost",
    "PF_heatmap_finite_run_power_floor",
    "PF_heatmap_conf_obs_power_floor",
    "PF_heatmap_conf_obs_power_power",
    "PF_heatmap_rate_mean_weight",
    "PF_heatmap_resample_min_threshold",
    "PF_heatmap_init_pos_sd",
    "PF_heatmap_init_rate_sd",
    "PF_heatmap_missing_jump_boost",
    "PF_heatmap_jump_rate_sd",
    "PF_heatmap_anchor_sigma",
    "PF_heatmap_state_z_weight",
    "PF_heatmap_gr_ambiguity_power",
    "PF_heatmap_gr_information_power",
    "PF_heatmap_outlier_prob",
    "PF_heatmap_dynamic_sigma_alpha",
    "PF_heatmap_missing_noise_scale",
    "PF_heatmap_missing_jump_scale",
)


PF_OPT_MERGE16_CURATED_CANDIDATES = (
    (
        "m16_alpha095",
        {"PF_heatmap_suffix_merge_obs_power_alpha": 0.95},
    ),
    (
        "m16_alpha105",
        {"PF_heatmap_suffix_merge_obs_power_alpha": 1.05},
    ),
    (
        "m16_alpha110_resamp035",
        {
            "PF_heatmap_suffix_merge_obs_power_alpha": 1.10,
            "PF_heatmap_resample_obs_power_adapt": 0.45,
            "PF_heatmap_resample_min_threshold": 0.30,
        },
    ),
    (
        "m16_lik575_raw010",
        {
            "PF_heatmap_lik_scale": 5.75,
            "PF_heatmap_raw_ref_likelihood_weight": 0.10,
        },
    ),
    (
        "m16_frdecay050_floor055",
        {
            "PF_heatmap_finite_run_power_decay": 0.50,
            "PF_heatmap_finite_run_power_floor": 0.55,
        },
    ),
    (
        "m16_stiffer_motion",
        {
            "PF_heatmap_momentum": 0.99950,
            "PF_heatmap_rate_noise": 0.00080,
            "PF_heatmap_pos_noise": 0.0030,
            "PF_heatmap_rough_pos": 0.060,
            "PF_heatmap_rough_rate": 0.0045,
        },
    ),
    (
        "m16_more_jump",
        {
            "PF_heatmap_jump_prob": 0.00020,
            "PF_heatmap_missing_jump_boost": 0.00020,
            "PF_heatmap_jump_sd": 8.0,
            "PF_heatmap_jump_rate_sd": 0.00055,
        },
    ),
    (
        "m16_pm0002_fine",
        {
            "PF_heatmap_profile_mixture_spec": PF_OPT_PROFILE_MIXTURES.get(
                "profile_rs0026_pm0002",
                _pm0002_accepted_profiles(),
            ),
        },
    ),
)


def _pf_merge16_random_search_space_note(last_round):
    if not last_round:
        lead = "starting from validated merge16 full SOTA"
    elif last_round["status"] == "no_sample_gate":
        lead = "last 200-well shard had no gated winner, resampling and keeping merge16-local moves"
    elif last_round["status"] == "confirm_rejected":
        lead = "last 200-well winner failed the disjoint confirmation shard, tightening around merge16"
    elif last_round["status"] == "full_rejected":
        lead = "last confirmed winner failed full 500x128 validation, keeping one/two-cluster mutations"
    else:
        lead = "last paired full check accepted, recentering around the new merge16 base"
    return (
        f"{lead}; merge_k is fixed at 16 with block/start/no-adjust semantics. The active space is focused on "
        "the dimensions that mattered in the merge tuning: block evidence alpha around 1.0, likelihood/reference "
        "balance, finite-run and confidence evidence tempering, resample/roughening interaction, step mobility, "
        "row-rate jump probability/std, anchor softness, and profile-mixture allocation. Explicit merge dynamics "
        "scaling, endpoint scoring, GR averaging, and k-fold jump-prob transforms are excluded because prior tests "
        "were negative."
    )


def _random_pf_merge16_candidates(round_idx, current_overrides, n_candidates=32):
    rng = np.random.default_rng(20260626 + 7919 * int(round_idx))
    space = PF_OPT_MERGE16_RANDOM_SEARCH_SPACE
    key_dims = [key for key in PF_OPT_MERGE16_KEY_DIMS if key in space]
    side_dims = [key for key in PF_OPT_MERGE16_SIDE_DIMS if key in space]
    candidates = []
    seen = {json.dumps(_clean_pf_random_overrides(current_overrides), sort_keys=True)}
    attempts = 0
    while len(candidates) < int(n_candidates) and attempts < int(n_candidates) * 120:
        attempts += 1
        overrides = dict(current_overrides)
        n_key_changes = int(rng.choice([2, 3, 4], p=[0.40, 0.45, 0.15]))
        n_side_changes = int(rng.choice([0, 1, 2], p=[0.55, 0.35, 0.10]))
        changed = []
        if key_dims:
            changed.extend(rng.choice(key_dims, size=min(n_key_changes, len(key_dims)), replace=False).tolist())
        if side_dims and n_side_changes > 0:
            changed.extend(rng.choice(side_dims, size=min(n_side_changes, len(side_dims)), replace=False).tolist())
        if "PF_heatmap_suffix_merge_obs_power_alpha" not in changed and len(candidates) % 3 == 0:
            changed.append("PF_heatmap_suffix_merge_obs_power_alpha")
        if "PF_heatmap_resample_obs_power_adapt" in changed and "PF_heatmap_resample_min_threshold" not in changed:
            changed.append("PF_heatmap_resample_min_threshold")
        if "PF_heatmap_resample_min_threshold" in changed and "PF_heatmap_resample_obs_power_adapt" not in changed:
            changed.append("PF_heatmap_resample_obs_power_adapt")
        if "PF_heatmap_finite_run_power_decay" in changed and "PF_heatmap_finite_run_power_floor" not in changed:
            changed.append("PF_heatmap_finite_run_power_floor")
        if "PF_heatmap_conf_obs_power_decay" in changed:
            if "PF_heatmap_conf_obs_power_floor" not in changed:
                changed.append("PF_heatmap_conf_obs_power_floor")
            if "PF_heatmap_conf_obs_power_power" not in changed and rng.random() < 0.40:
                changed.append("PF_heatmap_conf_obs_power_power")
        if "PF_heatmap_jump_prob" in changed:
            for jump_key in ("PF_heatmap_jump_sd", "PF_heatmap_jump_rate_sd", "PF_heatmap_missing_jump_boost"):
                if jump_key not in changed and rng.random() < 0.70:
                    changed.append(jump_key)
        if (
            "PF_heatmap_jump_sd" in changed
            or "PF_heatmap_jump_rate_sd" in changed
            or "PF_heatmap_missing_jump_boost" in changed
        ) and "PF_heatmap_jump_prob" not in changed:
            changed.append("PF_heatmap_jump_prob")
        if "PF_heatmap_anchor_power" in changed and "PF_heatmap_anchor_sigma" not in changed and rng.random() < 0.55:
            changed.append("PF_heatmap_anchor_sigma")
        if "PF_heatmap_gr_ambiguity_power" in changed:
            if "PF_heatmap_gr_ambiguity_min_power" not in changed:
                changed.append("PF_heatmap_gr_ambiguity_min_power")
            if "PF_heatmap_gr_ambiguity_ref_mode" not in changed and rng.random() < 0.25:
                changed.append("PF_heatmap_gr_ambiguity_ref_mode")
        if "PF_heatmap_gr_information_power" in changed:
            for info_key in (
                "PF_heatmap_gr_information_center",
                "PF_heatmap_gr_information_min_multiplier",
                "PF_heatmap_gr_information_max_multiplier",
                "PF_heatmap_gr_information_slope_weight",
            ):
                if info_key not in changed:
                    changed.append(info_key)
        if "PF_heatmap_dynamic_sigma_alpha" in changed:
            for sigma_key in (
                "PF_heatmap_dynamic_sigma_threshold",
                "PF_heatmap_dynamic_sigma_power",
                "PF_heatmap_dynamic_sigma_min",
                "PF_heatmap_dynamic_sigma_max",
            ):
                if sigma_key not in changed and rng.random() < 0.45:
                    changed.append(sigma_key)
        changed = sorted(set(changed))
        profile_label = "base"
        for key in changed:
            if key == "PF_heatmap_profile_mixture_spec":
                profile_label = str(rng.choice(tuple(space[key])))
                overrides[key] = PF_OPT_PROFILE_MIXTURES[profile_label]
                continue
            values = space[key]
            cur_value = overrides.get(key, None)
            for _ in range(8):
                value = values[int(rng.integers(0, len(values)))]
                if value != cur_value:
                    overrides[key] = value
                    break
            else:
                overrides[key] = values[int(rng.integers(0, len(values)))]
        overrides["PF_heatmap_suffix_merge_k"] = 16
        finite_run_decay = float(overrides.get("PF_heatmap_finite_run_power_decay", 0.0))
        if finite_run_decay > 0.0 and "PF_heatmap_finite_run_power_floor" not in changed:
            overrides["PF_heatmap_finite_run_power_floor"] = float(
                rng.choice(_pf_opt_finite_run_floor_values(finite_run_decay))
            )
        ambiguity_power = float(overrides.get("PF_heatmap_gr_ambiguity_power", 0.0))
        if ambiguity_power > 0.0:
            overrides.setdefault("PF_heatmap_gr_ambiguity_min_power", float(rng.choice(space["PF_heatmap_gr_ambiguity_min_power"])))
        information_power = float(overrides.get("PF_heatmap_gr_information_power", 0.0))
        if information_power > 0.0:
            overrides.setdefault("PF_heatmap_gr_information_center", 0.75)
            overrides.setdefault("PF_heatmap_gr_information_min_multiplier", 0.75)
            overrides.setdefault("PF_heatmap_gr_information_max_multiplier", 1.25)
        conf_obs_power_decay = float(overrides.get("PF_heatmap_conf_obs_power_decay", 0.0))
        if conf_obs_power_decay > 0.0:
            overrides.setdefault("PF_heatmap_conf_obs_power_floor", 0.60)
        outlier_prob = float(overrides.get("PF_heatmap_outlier_prob", 0.0))
        if outlier_prob > 0.0:
            overrides.setdefault("PF_heatmap_outlier_likelihood", 0.05)
        dynamic_sigma_alpha = float(overrides.get("PF_heatmap_dynamic_sigma_alpha", 0.0))
        if dynamic_sigma_alpha > 0.0:
            overrides.setdefault("PF_heatmap_dynamic_sigma_threshold", 1.35)
            overrides.setdefault("PF_heatmap_dynamic_sigma_power", 0.60)
            overrides.setdefault("PF_heatmap_dynamic_sigma_min", 0.90)
            overrides.setdefault("PF_heatmap_dynamic_sigma_max", 1.75)
        overrides = _clean_pf_random_overrides(overrides)
        payload = json.dumps(overrides, sort_keys=True)
        if payload in seen:
            continue
        seen.add(payload)
        name = f"m16rs{int(round_idx):04d}_{profile_label}_{len(candidates):02d}"
        candidates.append((name, overrides))
    if len(candidates) != int(n_candidates):
        raise RuntimeError(f"only generated {len(candidates)} unique merge16 random PF candidates")
    return candidates


def _random_pf_merge16_round_candidates(round_idx, current_overrides, n_candidates=32):
    candidates = [
        (f"m16rs{int(round_idx):04d}_{name}", _clean_pf_random_overrides({**current_overrides, **delta}))
        for name, delta in PF_OPT_MERGE16_CURATED_CANDIDATES
    ]
    candidates.extend(_random_pf_merge16_candidates(round_idx, current_overrides, n_candidates=n_candidates))
    deduped_candidates = []
    seen_payloads = {json.dumps(_clean_pf_random_overrides(current_overrides), sort_keys=True)}
    for cand_name, cand_overrides in candidates:
        cand_overrides = _clean_pf_random_overrides({**cand_overrides, "PF_heatmap_suffix_merge_k": 16})
        payload = json.dumps(cand_overrides, sort_keys=True)
        if payload in seen_payloads:
            continue
        seen_payloads.add(payload)
        deduped_candidates.append((cand_name, cand_overrides))
    return deduped_candidates


PF_OPT_MERGE64_SEARCH_BASE_NAME = PF_OPT_MERGE64_CURRENT_BEST_NAME
PF_OPT_MERGE64_SEARCH_BASE_OVERRIDES = PF_OPT_MERGE64_CURRENT_BEST_OVERRIDES

PF_OPT_MERGE64_ESCAPE_PROFILE_MIXTURES = {
    "m64_escape_dyn_micro": PF_OPT_PROFILE_MIXTURES["profile_pm0002_8x_dyn_micro"],
    "m64_escape_fr025": PF_OPT_PROFILE_MIXTURES["profile_pm0002_gfrdecay025_all"],
    "m64_escape_fr050_raw": PF_OPT_PROFILE_MIXTURES["profile_pm0002_gfrdecay050_raw_all"],
    "m64_escape_stiff_wide": _make_profile_mixture_from_archetypes(
        ("base", "smooth_stiff", "smooth_stiff_light", "wide_init", "rough_rate_light", "jump_rare_big"),
        (0.46, 0.16, 0.10, 0.12, 0.08, 0.08),
    ),
    "m64_escape_ref_state": _make_profile_mixture_from_archetypes(
        ("base", "raw_low", "raw_mid", "seen_heavy", "z0975", "z095", "smooth_stiff"),
        (0.42, 0.10, 0.10, 0.12, 0.10, 0.08, 0.08),
    ),
}

PF_OPT_MERGE64_RANDOM_SEARCH_SPACE = {
    # Alpha105 remains the paired baseline, but the tight local loop repeatedly
    # produced same-shape shard wins. This space deliberately re-opens only the
    # historically high-ROI escape axes: rs0026-style dynamics, finite-run ESS
    # tempering, calibrated raw/seen reference balance, targeted profile
    # diversity, moderate FFBSi smoother settings, direct-TVT/Z-state mixing,
    # and the tuned 1000x64 resample/roughening interaction.
    "PF_heatmap_suffix_merge_obs_power_alpha": [1.00, 1.02, 1.03, 1.05, 1.05, 1.07, 1.08, 1.10],
    "PF_heatmap_lik_scale": [5.40, 5.50, 5.75, 6.00, 6.00, 6.25, 6.50],
    "PF_heatmap_raw_ref_likelihood_weight": [0.025, 0.05, 0.075, 0.10, 0.125, 0.15],
    "PF_heatmap_seen_blend_weight": [0.60, 0.65, 0.70, 0.75, 0.75, 0.80, 0.85],
    "PF_heatmap_finite_run_power_boost": [0.0, 0.030, 0.040, 0.060, 0.080],
    "PF_heatmap_finite_run_power_decay": [0.25, 0.50, 0.75, 1.00, 1.25],
    "PF_heatmap_finite_run_power_floor": [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70],
    "PF_heatmap_momentum": [0.99935, 0.99940, 0.99945, 0.99950, 0.99955],
    "PF_heatmap_rate_noise": [0.00070, 0.00080, 0.00090, 0.00100, 0.00120],
    "PF_heatmap_pos_noise": [0.0020, 0.0023, 0.0025, 0.0028, 0.0032, 0.0035, 0.0042],
    "PF_heatmap_rate_mean_weight": [0.00050, 0.00075, 0.00100, 0.00125, 0.00150],
    "PF_heatmap_resample_threshold": [0.20, 0.25, 0.30, 0.30, 0.35, 0.40],
    "PF_heatmap_resample_obs_power_adapt": [0.25, 0.30, 0.45, 0.45, 0.60],
    "PF_heatmap_resample_min_threshold": [0.125, 0.20, 0.25, 0.30, 0.35],
    "PF_heatmap_rough_pos": [0.035, 0.040, 0.050, 0.060, 0.080, 0.10],
    "PF_heatmap_rough_rate": [0.0025, 0.0030, 0.0035, 0.0045, 0.0060, 0.0070],
    "PF_heatmap_init_pos_sd": [1.0, 2.0, 2.5, 3.0],
    "PF_heatmap_init_rate_sd": [0.0150, 0.0200, 0.0250, 0.0300],
    "PF_heatmap_jump_prob": [0.00005, 0.000075, 0.00010, 0.00015, 0.00020],
    "PF_heatmap_missing_jump_boost": [0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030],
    "PF_heatmap_jump_sd": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "PF_heatmap_jump_rate_sd": [0.00020, 0.00025, 0.00030, 0.00040, 0.00060],
    "PF_heatmap_anchor_sigma": [60.0, 70.0, 80.0, 90.0, 100.0],
    "PF_heatmap_anchor_power": [0.00150, 0.00175, 0.00200, 0.00225, 0.00250, 0.00300],
    "PF_heatmap_ffbsi_transition_scale": [1.00, 1.15, 1.25, 1.35, 1.50],
    "PF_heatmap_ffbsi_pos_floor": [0.10, 0.15, 0.20, 0.25],
    "PF_heatmap_ffbsi_rate_floor": [0.0018, 0.0025, 0.0035, 0.0040],
    "PF_heatmap_state_z_weight": [0.90, 0.925, 0.95, 0.975, 1.0, 1.0],
    "PF_heatmap_missing_noise_scale": [0.80, 0.90, 1.0, 1.10, 1.25],
    "PF_heatmap_missing_jump_scale": [0.85, 0.95, 1.0, 1.10, 1.25],
    "PF_heatmap_profile_mixture_spec": tuple(PF_OPT_MERGE64_ESCAPE_PROFILE_MIXTURES),
    PF_OPT_BUDGET_META_N_PARTICLES: [500, 500, 500, 1000],
    # This preset is explicitly optimized on FFBSi TVT quality.
    "PF_heatmap_ffbsi_mode": ["fallback"],
    "PF_heatmap_ffbsi_n_paths": [64],
    "PF_heatmap_ffbsi_fallback_mode": ["filtered"],
}


PF_OPT_MERGE64_KEY_DIMS = (
    "PF_heatmap_suffix_merge_obs_power_alpha",
    "PF_heatmap_lik_scale",
    "PF_heatmap_raw_ref_likelihood_weight",
    "PF_heatmap_seen_blend_weight",
    "PF_heatmap_finite_run_power_decay",
    "PF_heatmap_finite_run_power_floor",
    "PF_heatmap_resample_threshold",
    "PF_heatmap_resample_obs_power_adapt",
    "PF_heatmap_momentum",
    "PF_heatmap_rate_noise",
    "PF_heatmap_pos_noise",
    "PF_heatmap_rate_mean_weight",
    "PF_heatmap_rough_pos",
    "PF_heatmap_rough_rate",
    "PF_heatmap_jump_prob",
    "PF_heatmap_jump_sd",
    "PF_heatmap_jump_rate_sd",
    "PF_heatmap_missing_jump_boost",
    "PF_heatmap_anchor_power",
    "PF_heatmap_ffbsi_transition_scale",
    "PF_heatmap_state_z_weight",
    "PF_heatmap_profile_mixture_spec",
    PF_OPT_BUDGET_META_N_PARTICLES,
)


PF_OPT_MERGE64_SIDE_DIMS = (
    "PF_heatmap_finite_run_power_boost",
    "PF_heatmap_resample_min_threshold",
    "PF_heatmap_init_pos_sd",
    "PF_heatmap_init_rate_sd",
    "PF_heatmap_anchor_sigma",
    "PF_heatmap_ffbsi_pos_floor",
    "PF_heatmap_ffbsi_rate_floor",
    "PF_heatmap_missing_noise_scale",
    "PF_heatmap_missing_jump_scale",
)


PF_OPT_MERGE64_DYNAMICS_DIMS = (
    "PF_heatmap_momentum",
    "PF_heatmap_rate_noise",
    "PF_heatmap_pos_noise",
    "PF_heatmap_rough_pos",
    "PF_heatmap_rough_rate",
    "PF_heatmap_jump_prob",
    "PF_heatmap_jump_sd",
    "PF_heatmap_jump_rate_sd",
    "PF_heatmap_missing_jump_boost",
)


PF_OPT_MERGE64_CURATED_CANDIDATES = (
    (
        "escape_rs0026_core",
        {
            "PF_heatmap_suffix_merge_obs_power_alpha": 1.05,
            "PF_heatmap_lik_scale": 5.50,
            "PF_heatmap_raw_ref_likelihood_weight": 0.10,
            "PF_heatmap_seen_blend_weight": 0.75,
            "PF_heatmap_finite_run_power_decay": 0.25,
            "PF_heatmap_finite_run_power_floor": 0.55,
            "PF_heatmap_momentum": 0.99945,
            "PF_heatmap_rate_noise": 0.00090,
            "PF_heatmap_pos_noise": 0.0035,
            "PF_heatmap_resample_threshold": 0.35,
            "PF_heatmap_resample_obs_power_adapt": 0.60,
            "PF_heatmap_resample_min_threshold": 0.35,
            "PF_heatmap_jump_prob": 0.00015,
            "PF_heatmap_jump_sd": 7.0,
            "PF_heatmap_jump_rate_sd": 0.00040,
            "PF_heatmap_missing_jump_boost": 0.00015,
        },
    ),
    (
        "escape_alpha100_rs0026",
        {
            "PF_heatmap_suffix_merge_obs_power_alpha": 1.00,
            "PF_heatmap_lik_scale": 5.50,
            "PF_heatmap_raw_ref_likelihood_weight": 0.10,
            "PF_heatmap_seen_blend_weight": 0.75,
            "PF_heatmap_finite_run_power_decay": 0.25,
            "PF_heatmap_finite_run_power_floor": 0.55,
            "PF_heatmap_momentum": 0.99945,
            "PF_heatmap_rate_noise": 0.00090,
            "PF_heatmap_pos_noise": 0.0035,
            "PF_heatmap_jump_prob": 0.00015,
            "PF_heatmap_jump_sd": 7.0,
            "PF_heatmap_jump_rate_sd": 0.00040,
        },
    ),
    (
        "escape_profile_dyn_micro",
        {
            "PF_heatmap_profile_mixture_spec": PF_OPT_MERGE64_ESCAPE_PROFILE_MIXTURES["m64_escape_dyn_micro"],
            "PF_heatmap_suffix_merge_obs_power_alpha": 1.05,
        },
    ),
    (
        "escape_profile_fr050_raw",
        {
            "PF_heatmap_profile_mixture_spec": PF_OPT_MERGE64_ESCAPE_PROFILE_MIXTURES["m64_escape_fr050_raw"],
            "PF_heatmap_suffix_merge_obs_power_alpha": 1.05,
            "PF_heatmap_raw_ref_likelihood_weight": 0.10,
            "PF_heatmap_seen_blend_weight": 0.70,
        },
    ),
    (
        "escape_profile_stiff_wide",
        {
            "PF_heatmap_profile_mixture_spec": PF_OPT_MERGE64_ESCAPE_PROFILE_MIXTURES["m64_escape_stiff_wide"],
            "PF_heatmap_suffix_merge_obs_power_alpha": 1.05,
            "PF_heatmap_pos_noise": 0.0032,
            "PF_heatmap_rate_noise": 0.0010,
        },
    ),
    (
        "escape_n1000_resamp_low",
        {
            PF_OPT_BUDGET_META_N_PARTICLES: 1000,
            PF_OPT_BUDGET_META_N_SEEDS: 64,
            "PF_heatmap_resample_threshold": 0.25,
            "PF_heatmap_resample_obs_power_adapt": 0.30,
            "PF_heatmap_resample_min_threshold": 0.125,
            "PF_heatmap_rough_pos": 0.040,
            "PF_heatmap_rough_rate": 0.0030,
            "PF_heatmap_momentum": 0.99955,
            "PF_heatmap_rate_noise": 0.00070,
            "PF_heatmap_pos_noise": 0.0025,
        },
    ),
    (
        "escape_ffbsi_smooth135",
        {
            "PF_heatmap_ffbsi_transition_scale": 1.35,
            "PF_heatmap_ffbsi_pos_floor": 0.15,
            "PF_heatmap_ffbsi_rate_floor": 0.0025,
            "PF_heatmap_profile_mixture_spec": PF_OPT_FFBSI_PROFILE_PLANS["more_smooth"],
            "PF_heatmap_pos_noise": 0.0035,
            "PF_heatmap_jump_sd": 9.0,
        },
    ),
    (
        "escape_zstate0925_raw125",
        {
            "PF_heatmap_state_z_weight": 0.925,
            "PF_heatmap_raw_ref_likelihood_weight": 0.125,
            "PF_heatmap_seen_blend_weight": 0.65,
            "PF_heatmap_suffix_merge_obs_power_alpha": 1.05,
        },
    ),
    (
        "escape_alpha108_fr055",
        {
            "PF_heatmap_suffix_merge_obs_power_alpha": 1.08,
            "PF_heatmap_finite_run_power_decay": 0.50,
            "PF_heatmap_finite_run_power_floor": 0.55,
            "PF_heatmap_lik_scale": 5.75,
        },
    ),
    (
        "escape_mobile_missing",
        {
            "PF_heatmap_momentum": 0.99940,
            "PF_heatmap_rate_noise": 0.00120,
            "PF_heatmap_pos_noise": 0.0042,
            "PF_heatmap_rough_pos": 0.10,
            "PF_heatmap_rough_rate": 0.0070,
            "PF_heatmap_jump_prob": 0.00015,
            "PF_heatmap_missing_jump_boost": 0.00030,
            "PF_heatmap_missing_noise_scale": 1.25,
            "PF_heatmap_missing_jump_scale": 1.25,
        },
    ),
    (
        "alpha103",
        {"PF_heatmap_suffix_merge_obs_power_alpha": 1.03},
    ),
    (
        "alpha107",
        {"PF_heatmap_suffix_merge_obs_power_alpha": 1.07},
    ),
    (
        "alpha105_lik590",
        {"PF_heatmap_lik_scale": 5.90},
    ),
    (
        "alpha105_lik610",
        {"PF_heatmap_lik_scale": 6.10},
    ),
    (
        "alpha105_fr062",
        {
            "PF_heatmap_finite_run_power_decay": 1.00,
            "PF_heatmap_finite_run_power_floor": 0.62,
        },
    ),
    (
        "alpha105_fr065",
        {
            "PF_heatmap_finite_run_power_decay": 1.00,
            "PF_heatmap_finite_run_power_floor": 0.65,
        },
    ),
    (
        "alpha105_resamp035_mild",
        {
            "PF_heatmap_resample_threshold": 0.35,
            "PF_heatmap_resample_obs_power_adapt": 0.45,
            "PF_heatmap_resample_min_threshold": 0.30,
            "PF_heatmap_rough_pos": 0.060,
            "PF_heatmap_rough_rate": 0.0045,
        },
    ),
    (
        "alpha105_resamp025_tight",
        {
            "PF_heatmap_resample_threshold": 0.25,
            "PF_heatmap_resample_obs_power_adapt": 0.30,
            "PF_heatmap_resample_min_threshold": 0.20,
            "PF_heatmap_rough_pos": 0.050,
            "PF_heatmap_rough_rate": 0.0035,
            "PF_heatmap_momentum": 0.99950,
            "PF_heatmap_rate_noise": 0.00090,
            "PF_heatmap_pos_noise": 0.0025,
        },
    ),
    (
        "alpha105_jump075_miss250",
        {
            "PF_heatmap_jump_prob": 0.000075,
            "PF_heatmap_missing_jump_boost": 0.00025,
            "PF_heatmap_jump_sd": 8.0,
            "PF_heatmap_jump_rate_sd": 0.00030,
        },
    ),
    (
        "alpha105_jump100_miss250",
        {
            "PF_heatmap_jump_prob": 0.00010,
            "PF_heatmap_missing_jump_boost": 0.00025,
            "PF_heatmap_jump_sd": 7.0,
            "PF_heatmap_jump_rate_sd": 0.00030,
        },
    ),
    (
        "alpha105_anchor_soft",
        {
            "PF_heatmap_anchor_sigma": 85.0,
            "PF_heatmap_anchor_power": 0.00175,
        },
    ),
    (
        "alpha105_anchor_firm",
        {
            "PF_heatmap_anchor_sigma": 75.0,
            "PF_heatmap_anchor_power": 0.00225,
        },
    ),
    (
        "alpha105_ref_raw100_seen700",
        {
            "PF_heatmap_raw_ref_likelihood_weight": 0.10,
            "PF_heatmap_seen_blend_weight": 0.70,
        },
    ),
    (
        "alpha105_ref_raw050_seen800",
        {
            "PF_heatmap_raw_ref_likelihood_weight": 0.05,
            "PF_heatmap_seen_blend_weight": 0.80,
        },
    ),
    (
        "alpha105_missing_mild",
        {
            "PF_heatmap_missing_noise_scale": 0.90,
            "PF_heatmap_missing_jump_scale": 1.10,
        },
    ),
)


def _pf_merge64_random_search_space_note(last_round):
    if not last_round:
        lead = "starting from accepted alpha105 merge64 SOTA with an escape search after local alpha105 rounds stalled"
    elif last_round["status"] == "no_sample_gate":
        lead = "last 200-well shard had no FFBSi-gated winner, resampling wells and trying broader high-ROI axes"
    elif last_round["status"] == "confirm_rejected":
        lead = "last 200-well winner failed the disjoint confirmation shard, keeping confirmation-gated broad escape moves"
    elif last_round["status"] == "full_rejected":
        lead = "last confirmed merge64 winner failed paired full validation, recentering but preserving broad escape families"
    else:
        lead = "last paired full check accepted, recentering merge64 mutations around the new base"
    return (
        f"{lead}; objective is FFBSi TVT RMSE at `500x128`. The space keeps the validated block/start/no-adjust "
        "merge contract and compares against `m64rs0011_base_alpha105`, which beat `m64rs0001_base_15_manual_full` "
        "on the paired full seed `20265667`. The escape space is broader than the stalled local loop but still "
        "uses repeated-shard/full gates: alpha `1.00-1.10`, likelihood scale `5.4-6.5`, finite-run ESS floors "
        "`0.50-0.70`, rs0026-style mobile dynamics, tuned `1000x64` low-resample candidates, targeted profile "
        "mixtures, moderate FFBSi transition settings, raw/seen reference balance, direct-TVT/Z-state mixing, "
        "anchor range, and missing-gap mobility. Still excluded from ordinary search: GR averaging, endpoint-state "
        "scoring, explicit merge dynamics scaling, GR correlation/NCC likelihood, lookahead-heavy hooks, broad "
        "outlier/information stacks, and jump-tail rescue."
    )


def _random_pf_merge64_candidates(round_idx, current_overrides, n_candidates=32):
    rng = np.random.default_rng(20260628 + 7919 * int(round_idx))
    space = PF_OPT_MERGE64_RANDOM_SEARCH_SPACE
    key_dims = [key for key in PF_OPT_MERGE64_KEY_DIMS if key in space]
    side_dims = [key for key in PF_OPT_MERGE64_SIDE_DIMS if key in space]
    candidates = []
    seen = {json.dumps(_clean_pf_random_overrides(current_overrides), sort_keys=True)}
    attempts = 0
    while len(candidates) < int(n_candidates) and attempts < int(n_candidates) * 160:
        attempts += 1
        overrides = dict(current_overrides)
        n_key_changes = int(rng.choice([2, 3, 4], p=[0.40, 0.45, 0.15]))
        n_side_changes = int(rng.choice([0, 1, 2], p=[0.35, 0.45, 0.20]))
        changed = []
        if key_dims:
            changed.extend(rng.choice(key_dims, size=min(n_key_changes, len(key_dims)), replace=False).tolist())
        if side_dims and n_side_changes > 0:
            changed.extend(rng.choice(side_dims, size=min(n_side_changes, len(side_dims)), replace=False).tolist())
        active_dynamics_dims = [key for key in PF_OPT_MERGE64_DYNAMICS_DIMS if key in space]
        if active_dynamics_dims and not any(key in changed for key in active_dynamics_dims) and rng.random() < 0.50:
            changed.extend(
                rng.choice(
                    active_dynamics_dims,
                    size=int(rng.choice([1, 2], p=[0.70, 0.30])),
                    replace=False,
                ).tolist()
            )
        if "PF_heatmap_jump_prob" in space and "PF_heatmap_jump_prob" not in changed and len(candidates) % 4 == 0:
            changed.append("PF_heatmap_jump_prob")
        if (
            "PF_heatmap_suffix_merge_obs_power_alpha" in space
            and "PF_heatmap_suffix_merge_obs_power_alpha" not in changed
            and len(candidates) % 3 == 0
        ):
            changed.append("PF_heatmap_suffix_merge_obs_power_alpha")
        if (
            PF_OPT_BUDGET_META_N_PARTICLES in space
            and PF_OPT_BUDGET_META_N_PARTICLES not in changed
            and len(candidates) % 3 == 1
        ):
            changed.append(PF_OPT_BUDGET_META_N_PARTICLES)
        if "PF_heatmap_profile_mixture_spec" in space and "PF_heatmap_profile_mixture_spec" not in changed and rng.random() < 0.35:
            changed.append("PF_heatmap_profile_mixture_spec")
        if PF_OPT_BUDGET_META_N_PARTICLES in changed:
            for budget_key in (
                "PF_heatmap_resample_threshold",
                "PF_heatmap_resample_obs_power_adapt",
                "PF_heatmap_resample_min_threshold",
                "PF_heatmap_rough_pos",
                "PF_heatmap_rough_rate",
            ):
                if budget_key not in changed and rng.random() < 0.70:
                    changed.append(budget_key)
        if "PF_heatmap_ffbsi_transition_scale" in changed:
            if "PF_heatmap_ffbsi_pos_floor" not in changed:
                changed.append("PF_heatmap_ffbsi_pos_floor")
            if "PF_heatmap_ffbsi_rate_floor" not in changed and rng.random() < 0.55:
                changed.append("PF_heatmap_ffbsi_rate_floor")
        if "PF_heatmap_ffbsi_pos_floor" in changed and "PF_heatmap_ffbsi_transition_scale" not in changed and "PF_heatmap_ffbsi_transition_scale" in space:
            changed.append("PF_heatmap_ffbsi_transition_scale")
        if "PF_heatmap_resample_obs_power_adapt" in changed and "PF_heatmap_resample_min_threshold" not in changed:
            changed.append("PF_heatmap_resample_min_threshold")
        if "PF_heatmap_resample_min_threshold" in changed and "PF_heatmap_resample_obs_power_adapt" not in changed:
            changed.append("PF_heatmap_resample_obs_power_adapt")
        if "PF_heatmap_finite_run_power_decay" in changed and "PF_heatmap_finite_run_power_floor" not in changed:
            changed.append("PF_heatmap_finite_run_power_floor")
        if "PF_heatmap_conf_obs_power_decay" in changed:
            if "PF_heatmap_conf_obs_power_floor" not in changed:
                changed.append("PF_heatmap_conf_obs_power_floor")
            if "PF_heatmap_conf_obs_power_power" not in changed and rng.random() < 0.40:
                changed.append("PF_heatmap_conf_obs_power_power")
        if "PF_heatmap_jump_prob" in changed:
            for jump_key in ("PF_heatmap_jump_sd", "PF_heatmap_jump_rate_sd", "PF_heatmap_missing_jump_boost"):
                if jump_key not in changed and rng.random() < 0.70:
                    changed.append(jump_key)
        if (
            "PF_heatmap_jump_sd" in changed
            or "PF_heatmap_jump_rate_sd" in changed
            or "PF_heatmap_missing_jump_boost" in changed
        ) and "PF_heatmap_jump_prob" not in changed:
            changed.append("PF_heatmap_jump_prob")
        if "PF_heatmap_jump_tail_prob" in changed:
            for tail_key in (
                "PF_heatmap_jump_tail_sd",
                "PF_heatmap_jump_tail_rate_sd",
                "PF_heatmap_jump_tail_clip",
            ):
                if tail_key not in changed:
                    changed.append(tail_key)
            if "PF_heatmap_jump_tail_missing_boost" not in changed and rng.random() < 0.35:
                changed.append("PF_heatmap_jump_tail_missing_boost")
        if (
            "PF_heatmap_jump_tail_sd" in changed
            or "PF_heatmap_jump_tail_rate_sd" in changed
            or "PF_heatmap_jump_tail_clip" in changed
            or "PF_heatmap_jump_tail_missing_boost" in changed
        ) and "PF_heatmap_jump_tail_prob" not in changed:
            changed.append("PF_heatmap_jump_tail_prob")
        if "PF_heatmap_anchor_power" in changed and "PF_heatmap_anchor_sigma" not in changed and "PF_heatmap_anchor_sigma" in space and rng.random() < 0.55:
            changed.append("PF_heatmap_anchor_sigma")
        if "PF_heatmap_gr_ambiguity_power" in changed:
            if "PF_heatmap_gr_ambiguity_min_power" not in changed:
                changed.append("PF_heatmap_gr_ambiguity_min_power")
            if "PF_heatmap_gr_ambiguity_ref_mode" not in changed and rng.random() < 0.25:
                changed.append("PF_heatmap_gr_ambiguity_ref_mode")
        if "PF_heatmap_gr_information_power" in changed:
            for info_key in (
                "PF_heatmap_gr_information_center",
                "PF_heatmap_gr_information_min_multiplier",
                "PF_heatmap_gr_information_max_multiplier",
                "PF_heatmap_gr_information_slope_weight",
            ):
                if info_key not in changed:
                    changed.append(info_key)
        if "PF_heatmap_dynamic_sigma_alpha" in changed:
            for sigma_key in (
                "PF_heatmap_dynamic_sigma_threshold",
                "PF_heatmap_dynamic_sigma_power",
                "PF_heatmap_dynamic_sigma_min",
                "PF_heatmap_dynamic_sigma_max",
            ):
                if sigma_key not in changed and rng.random() < 0.45:
                    changed.append(sigma_key)
        changed = sorted(set(changed))
        profile_label = "base"
        for key in changed:
            if key == "PF_heatmap_profile_mixture_spec":
                profile_options = tuple(space.get(key, ()))
                if profile_options and rng.random() < 0.65:
                    profile_key = str(profile_options[int(rng.integers(0, len(profile_options)))])
                    profile_label = profile_key.replace("m64_escape_", "prof_")
                    overrides[key] = PF_OPT_MERGE64_ESCAPE_PROFILE_MIXTURES[profile_key]
                else:
                    profile_label = "profile32"
                    overrides[key] = _sample_merge64_profile_mixture(rng, n_slots=32)
                continue
            if key == PF_OPT_BUDGET_META_N_PARTICLES:
                n_particles = int(space[key][int(rng.integers(0, len(space[key])))])
                if n_particles == 1000:
                    profile_label = f"{profile_label}_n1000"
                    overrides = _pf_opt_apply_n1000_budget_center(overrides, changed=changed)
                else:
                    overrides = _pf_opt_with_candidate_budget(overrides, 500, 128)
                continue
            values = space[key]
            cur_value = overrides.get(key, None)
            for _ in range(8):
                value = values[int(rng.integers(0, len(values)))]
                if value != cur_value:
                    overrides[key] = value
                    break
            else:
                overrides[key] = values[int(rng.integers(0, len(values)))]
        overrides.update(
            {
                "PF_heatmap_ffbsi_mode": "fallback",
                "PF_heatmap_ffbsi_n_paths": 64,
                "PF_heatmap_ffbsi_fallback_mode": "filtered",
                "PF_heatmap_suffix_merge_adjust_dynamics": False,
                "PF_heatmap_suffix_merge_likelihood_mode": "block",
                "PF_heatmap_suffix_merge_k": 64,
                "PF_heatmap_suffix_merge_state_mode": "start",
            }
        )
        finite_run_decay = float(overrides.get("PF_heatmap_finite_run_power_decay", 0.0))
        if finite_run_decay > 0.0 and "PF_heatmap_finite_run_power_floor" not in changed:
            overrides["PF_heatmap_finite_run_power_floor"] = float(
                rng.choice(_pf_opt_finite_run_floor_values(finite_run_decay))
            )
        ambiguity_power = float(overrides.get("PF_heatmap_gr_ambiguity_power", 0.0))
        if ambiguity_power > 0.0:
            overrides.setdefault("PF_heatmap_gr_ambiguity_min_power", float(rng.choice(space["PF_heatmap_gr_ambiguity_min_power"])))
        information_power = float(overrides.get("PF_heatmap_gr_information_power", 0.0))
        if information_power > 0.0:
            overrides.setdefault("PF_heatmap_gr_information_center", 0.75)
            overrides.setdefault("PF_heatmap_gr_information_min_multiplier", 0.75)
            overrides.setdefault("PF_heatmap_gr_information_max_multiplier", 1.25)
        conf_obs_power_decay = float(overrides.get("PF_heatmap_conf_obs_power_decay", 0.0))
        if conf_obs_power_decay > 0.0:
            overrides.setdefault("PF_heatmap_conf_obs_power_floor", 0.60)
        jump_tail_prob = float(overrides.get("PF_heatmap_jump_tail_prob", 0.0))
        jump_tail_missing_boost = float(overrides.get("PF_heatmap_jump_tail_missing_boost", 0.0))
        if jump_tail_prob > 0.0 or jump_tail_missing_boost > 0.0:
            overrides.setdefault("PF_heatmap_jump_tail_sd", float(rng.choice(space["PF_heatmap_jump_tail_sd"])))
            overrides.setdefault("PF_heatmap_jump_tail_rate_sd", float(rng.choice(space["PF_heatmap_jump_tail_rate_sd"])))
            overrides.setdefault("PF_heatmap_jump_tail_clip", float(rng.choice(space["PF_heatmap_jump_tail_clip"])))
        n_particles_meta, _ = _pf_opt_candidate_budget(overrides, 500, 128)
        if n_particles_meta == 1000:
            overrides = _pf_opt_apply_n1000_budget_center(overrides, changed=changed)
        outlier_prob = float(overrides.get("PF_heatmap_outlier_prob", 0.0))
        if outlier_prob > 0.0:
            overrides.setdefault("PF_heatmap_outlier_likelihood", 0.05)
        dynamic_sigma_alpha = float(overrides.get("PF_heatmap_dynamic_sigma_alpha", 0.0))
        if dynamic_sigma_alpha > 0.0:
            overrides.setdefault("PF_heatmap_dynamic_sigma_threshold", 1.35)
            overrides.setdefault("PF_heatmap_dynamic_sigma_power", 0.60)
            overrides.setdefault("PF_heatmap_dynamic_sigma_min", 0.90)
            overrides.setdefault("PF_heatmap_dynamic_sigma_max", 1.75)
        overrides = _clean_pf_random_overrides(overrides)
        payload = json.dumps(overrides, sort_keys=True)
        if payload in seen:
            continue
        seen.add(payload)
        name = f"m64rs{int(round_idx):04d}_{profile_label}_{len(candidates):02d}"
        candidates.append((name, overrides))
    if len(candidates) != int(n_candidates):
        raise RuntimeError(f"only generated {len(candidates)} unique merge64 random PF candidates")
    return candidates


def _random_pf_merge64_round_candidates(round_idx, current_overrides, n_candidates=32):
    base_candidate_overrides = {
        **current_overrides,
        "PF_heatmap_ffbsi_mode": "fallback",
        "PF_heatmap_ffbsi_n_paths": 64,
        "PF_heatmap_ffbsi_fallback_mode": "filtered",
        "PF_heatmap_suffix_merge_adjust_dynamics": False,
        "PF_heatmap_suffix_merge_likelihood_mode": "block",
        "PF_heatmap_suffix_merge_k": 64,
        "PF_heatmap_suffix_merge_state_mode": "start",
    }
    candidates = []
    for name, delta in PF_OPT_MERGE64_CURATED_CANDIDATES:
        candidates.append(
            (
                f"m64rs{int(round_idx):04d}_{name}",
                _clean_pf_random_overrides(
                    {
                        **base_candidate_overrides,
                        **delta,
                    }
                ),
            )
        )
    candidates.extend(_random_pf_merge64_candidates(round_idx, current_overrides, n_candidates=n_candidates))
    deduped_candidates = []
    seen_payloads = {json.dumps(_clean_pf_random_overrides(current_overrides), sort_keys=True)}
    for cand_name, cand_overrides in candidates:
        cand_overrides = _clean_pf_random_overrides({**cand_overrides, "PF_heatmap_suffix_merge_k": 64})
        payload = json.dumps(cand_overrides, sort_keys=True)
        if payload in seen_payloads:
            continue
        seen_payloads.add(payload)
        deduped_candidates.append((cand_name, cand_overrides))
    return deduped_candidates


def _pf_random_search_space_spec(search_space):
    search_space = str(search_space or "broad").strip().lower()
    if search_space == "merge16":
        return {
            "name": "merge16",
            "round_prefix": "m16rs",
            "current_name": PF_OPT_MERGE16_CURRENT_BEST_NAME,
            "current_overrides": PF_OPT_MERGE16_CURRENT_BEST_OVERRIDES,
            "sample_seed_base": 20260626,
            "confirm_seed_base": 20260627,
            "candidate_fn": _random_pf_merge16_round_candidates,
            "space_note_fn": _pf_merge16_random_search_space_note,
        }
    if search_space == "merge64":
        return {
            "name": "merge64",
            "round_prefix": "m64rs",
            "current_name": PF_OPT_MERGE64_SEARCH_BASE_NAME,
            "current_overrides": PF_OPT_MERGE64_SEARCH_BASE_OVERRIDES,
            "sample_seed_base": 20260628,
            "confirm_seed_base": 20260629,
            "candidate_fn": _random_pf_merge64_round_candidates,
            "space_note_fn": _pf_merge64_random_search_space_note,
            "objective_metric": "ffbsi",
        }
    if search_space == "broad":
        return {
            "name": "broad",
            "round_prefix": "rs",
            "current_name": PF_OPT_CURRENT_BEST_NAME,
            "current_overrides": PF_OPT_CURRENT_BEST_OVERRIDES,
            "sample_seed_base": 20260605,
            "confirm_seed_base": 20260623,
            "candidate_fn": _random_pf_broad_candidates,
            "space_note_fn": _pf_random_search_space_note,
        }
    raise ValueError(f"unknown PF random search space `{search_space}`")


def _pf_profile_search_space_note(last_round):
    if not last_round:
        lead = "starting filtered profile-mixture search from the active opt baseline"
    elif last_round["status"] == "no_sample_gate":
        lead = "last filtered-profile sample had no gated winner, resampling wells and filtered profile settings"
    elif last_round["status"] == "full_rejected":
        lead = "last filtered-profile sample winner failed full validation, keeping profile and smoothing mutations conservative"
    else:
        lead = "last filtered-profile full validation accepted, searching around the new heatmap baseline"
    return (
        f"{lead}; candidates allocate seeds by profile weights so the same profile prior scales naturally "
        "from 32 to 128 seeds. The active dimensions include local pm0002 weight reshapes plus broader "
        "dynamic, finite-run, and raw/seen-reference profile plans, with the same scalar axes as the random "
        "loop: likelihood softness, finite-run tempering, forward position/rate mobility, local Gaussian jump "
        "support, Z-state/direct-TVT mixing, adaptive resample thresholding, mild ambiguity tempering, "
        "missing-gap mobility, and rescue particles. The loop still excludes soft-interp missing-GR mode, "
        "non-likelihood seed weighting, coarser Typewell grids, GR-shape/NCC, and lookahead-heavy hooks. "
        "FFBSi-specific posterior smoothing is now legacy/diagnostic only, because the downstream NN benefits "
        "more from a useful filtered posterior than from the narrowest backward-smoothed path. Full validation is "
        "triggered only when a sampled candidate beats the active baseline on global RMSE, per-well mean RMSE, "
        "and per-well mean RMSE after excluding the worst 10% wells; promotion uses a paired fresh-seed SOTA "
        "check at the configured full-check budget."
    )


def _pf_failure_guided_search_space_note(last_round):
    if not last_round:
        lead = "starting failure-guided PF search from accepted rs0003_rand07 fresh-seed baseline"
    elif last_round["status"] == "analysis_ready":
        lead = "cached failure diagnostics are ready, starting targeted cfg repair by misspecification class"
    elif last_round["status"] == "no_sample_gate":
        lead = "last failure-guided sample had no directed winner, resampling bad wells and compact repair cfgs"
    elif last_round["status"] == "directed_rejected":
        lead = "last sample won the combined gate but failed the bad-vs-context misspecification check, tightening the repair space"
    elif last_round["status"] == "confirm_rejected":
        lead = "last directed winner failed independent repeat-shard confirmation, keeping repair changes compact"
    elif last_round["status"] == "full_rejected":
        lead = "last failure-guided candidate passed the directed gate but failed full validation, keeping the repair space narrow"
    else:
        lead = "last failure-guided full validation accepted, searching around the new misspecification-aware base"
    return (
        f"{lead}; the sampled wells are a mix of cached PF failures and a smaller random context slice so the search "
        "targets misspecified cases without becoming a pure shrink-to-anchor bias. The active repair space stays focused "
        "on reliability and model mismatch signals: finite-run ESS tempering, motif ambiguity tempering, raw/seen "
        "reference balance, soft anchor likelihood, partial direct-TVT state, mild outlier handling, and modest "
        "dynamics retunes. The loop still requires a candidate to improve the combined sample and the bad-case slice "
        "and then pass a repeat-shard confirmation before full validation. Final acceptance still uses global RMSE "
        "plus the well-level gates on the full train set."
    )


def _clean_pf_random_overrides(overrides):
    cleaned = dict(overrides)
    n_particles_meta = cleaned.pop(PF_OPT_BUDGET_META_N_PARTICLES, None)
    n_seeds_meta = cleaned.pop(PF_OPT_BUDGET_META_N_SEEDS, None)
    query_mode = str(cleaned.get("PF_heatmap_query_gr_mode", "interp"))
    if query_mode != "soft_interp":
        cleaned.pop("PF_heatmap_missing_likelihood_power", None)
        cleaned.pop("PF_heatmap_missing_gap_decay", None)
        cleaned.pop("PF_heatmap_missing_min_power", None)

    if abs(float(cleaned.get("PF_heatmap_seen_blend_weight", 0.7)) - 0.7) <= 1e-12:
        cleaned.pop("PF_heatmap_seen_blend_weight", None)
    if abs(float(cleaned.get("PF_heatmap_raw_ref_likelihood_weight", 0.0))) <= 1e-12:
        cleaned.pop("PF_heatmap_raw_ref_likelihood_weight", None)
    ambiguity_power = float(cleaned.get("PF_heatmap_gr_ambiguity_power", 0.0))
    if ambiguity_power <= 0.0:
        cleaned.pop("PF_heatmap_gr_ambiguity_power", None)
        cleaned.pop("PF_heatmap_gr_ambiguity_min_power", None)
        cleaned.pop("PF_heatmap_gr_ambiguity_contrast", None)
        cleaned.pop("PF_heatmap_gr_ambiguity_ref_mode", None)
    else:
        if abs(float(cleaned.get("PF_heatmap_gr_ambiguity_min_power", 0.35)) - 0.35) <= 1e-12:
            cleaned.pop("PF_heatmap_gr_ambiguity_min_power", None)
        if abs(float(cleaned.get("PF_heatmap_gr_ambiguity_contrast", 1.0)) - 1.0) <= 1e-12:
            cleaned.pop("PF_heatmap_gr_ambiguity_contrast", None)
        if str(cleaned.get("PF_heatmap_gr_ambiguity_ref_mode", "primary")) == "primary":
            cleaned.pop("PF_heatmap_gr_ambiguity_ref_mode", None)

    information_power = float(cleaned.get("PF_heatmap_gr_information_power", 0.0))
    if information_power <= 0.0:
        cleaned.pop("PF_heatmap_gr_information_power", None)
        cleaned.pop("PF_heatmap_gr_information_center", None)
        cleaned.pop("PF_heatmap_gr_information_min_multiplier", None)
        cleaned.pop("PF_heatmap_gr_information_max_multiplier", None)
        cleaned.pop("PF_heatmap_gr_information_slope_weight", None)
        cleaned.pop("PF_heatmap_gr_information_ref_mode", None)
    else:
        if abs(float(cleaned.get("PF_heatmap_gr_information_center", 0.75)) - 0.75) <= 1e-12:
            cleaned.pop("PF_heatmap_gr_information_center", None)
        if abs(float(cleaned.get("PF_heatmap_gr_information_min_multiplier", 0.75)) - 0.75) <= 1e-12:
            cleaned.pop("PF_heatmap_gr_information_min_multiplier", None)
        if abs(float(cleaned.get("PF_heatmap_gr_information_max_multiplier", 1.25)) - 1.25) <= 1e-12:
            cleaned.pop("PF_heatmap_gr_information_max_multiplier", None)
        if abs(float(cleaned.get("PF_heatmap_gr_information_slope_weight", 0.0))) <= 1e-12:
            cleaned.pop("PF_heatmap_gr_information_slope_weight", None)
        if str(cleaned.get("PF_heatmap_gr_information_ref_mode", "primary")) == "primary":
            cleaned.pop("PF_heatmap_gr_information_ref_mode", None)

    shape_power = float(cleaned.get("PF_heatmap_gr_shape_power", 0.0))
    if shape_power <= 0.0:
        cleaned.pop("PF_heatmap_gr_shape_power", None)
        cleaned.pop("PF_heatmap_gr_shape_mode", None)
        cleaned.pop("PF_heatmap_gr_shape_window", None)
        cleaned.pop("PF_heatmap_gr_shape_min_points", None)
        cleaned.pop("PF_heatmap_gr_shape_sigma_floor", None)
        cleaned.pop("PF_heatmap_gr_shape_ref_mode", None)
    else:
        mode = str(cleaned.get("PF_heatmap_gr_shape_mode", "resid_corr"))
        if mode == "resid_corr":
            cleaned.pop("PF_heatmap_gr_shape_mode", None)
        if int(cleaned.get("PF_heatmap_gr_shape_window", 15)) == 15:
            cleaned.pop("PF_heatmap_gr_shape_window", None)
        if int(cleaned.get("PF_heatmap_gr_shape_min_points", 7)) == 7:
            cleaned.pop("PF_heatmap_gr_shape_min_points", None)
        if abs(float(cleaned.get("PF_heatmap_gr_shape_sigma_floor", 0.35)) - 0.35) <= 1e-12:
            cleaned.pop("PF_heatmap_gr_shape_sigma_floor", None)
        if str(cleaned.get("PF_heatmap_gr_shape_ref_mode", "primary")) == "primary":
            cleaned.pop("PF_heatmap_gr_shape_ref_mode", None)

    outlier_prob = float(cleaned.get("PF_heatmap_outlier_prob", 0.0))
    if outlier_prob <= 0.0:
        cleaned.pop("PF_heatmap_outlier_prob", None)
        cleaned.pop("PF_heatmap_outlier_likelihood", None)
    elif abs(float(cleaned.get("PF_heatmap_outlier_likelihood", 0.05)) - 0.05) <= 1e-12:
        cleaned.pop("PF_heatmap_outlier_likelihood", None)

    dynamic_sigma_alpha = float(cleaned.get("PF_heatmap_dynamic_sigma_alpha", 0.0))
    if dynamic_sigma_alpha <= 0.0:
        cleaned.pop("PF_heatmap_dynamic_sigma_alpha", None)
        cleaned.pop("PF_heatmap_dynamic_sigma_threshold", None)
        cleaned.pop("PF_heatmap_dynamic_sigma_power", None)
        cleaned.pop("PF_heatmap_dynamic_sigma_min", None)
        cleaned.pop("PF_heatmap_dynamic_sigma_max", None)
    else:
        if abs(float(cleaned.get("PF_heatmap_dynamic_sigma_threshold", 1.25)) - 1.25) <= 1e-12:
            cleaned.pop("PF_heatmap_dynamic_sigma_threshold", None)
        if abs(float(cleaned.get("PF_heatmap_dynamic_sigma_power", 1.0)) - 1.0) <= 1e-12:
            cleaned.pop("PF_heatmap_dynamic_sigma_power", None)
        if abs(float(cleaned.get("PF_heatmap_dynamic_sigma_min", 0.85)) - 0.85) <= 1e-12:
            cleaned.pop("PF_heatmap_dynamic_sigma_min", None)
        if abs(float(cleaned.get("PF_heatmap_dynamic_sigma_max", 2.0)) - 2.0) <= 1e-12:
            cleaned.pop("PF_heatmap_dynamic_sigma_max", None)

    seed_local_mode = str(cleaned.get("PF_heatmap_seed_local_weight_mode", "off")).lower()
    if seed_local_mode in {"off", "none", "global", "likelihood"}:
        cleaned.pop("PF_heatmap_seed_local_weight_mode", None)
        cleaned.pop("PF_heatmap_seed_local_score_source", None)
        cleaned.pop("PF_heatmap_seed_local_lik_scale", None)
        cleaned.pop("PF_heatmap_seed_local_half_life_blocks", None)
        cleaned.pop("PF_heatmap_seed_local_global_mix", None)
        cleaned.pop("PF_heatmap_seed_local_min_power", None)
        cleaned.pop("PF_heatmap_seed_local_combine_mode", None)
        cleaned.pop("PF_heatmap_seed_local_residual_alpha", None)
        cleaned.pop("PF_heatmap_seed_local_residual_clip", None)
    else:
        if str(cleaned.get("PF_heatmap_seed_local_score_source", "pf_likelihood")).lower() == "pf_likelihood":
            cleaned.pop("PF_heatmap_seed_local_score_source", None)
        if abs(float(cleaned.get("PF_heatmap_seed_local_lik_scale", 1.0)) - 1.0) <= 1e-12:
            cleaned.pop("PF_heatmap_seed_local_lik_scale", None)
        if abs(float(cleaned.get("PF_heatmap_seed_local_half_life_blocks", 8.0)) - 8.0) <= 1e-12:
            cleaned.pop("PF_heatmap_seed_local_half_life_blocks", None)
        if abs(float(cleaned.get("PF_heatmap_seed_local_global_mix", 0.0))) <= 1e-12:
            cleaned.pop("PF_heatmap_seed_local_global_mix", None)
        if abs(float(cleaned.get("PF_heatmap_seed_local_min_power", 1e-6)) - 1e-6) <= 1e-12:
            cleaned.pop("PF_heatmap_seed_local_min_power", None)
        seed_local_combine_mode = str(cleaned.get("PF_heatmap_seed_local_combine_mode", "replace")).lower()
        if seed_local_combine_mode == "replace":
            cleaned.pop("PF_heatmap_seed_local_combine_mode", None)
            cleaned.pop("PF_heatmap_seed_local_residual_alpha", None)
            cleaned.pop("PF_heatmap_seed_local_residual_clip", None)
        else:
            if abs(float(cleaned.get("PF_heatmap_seed_local_residual_alpha", 0.0))) <= 1e-12:
                cleaned.pop("PF_heatmap_seed_local_residual_alpha", None)
            if abs(float(cleaned.get("PF_heatmap_seed_local_residual_clip", 5.0)) - 5.0) <= 1e-12:
                cleaned.pop("PF_heatmap_seed_local_residual_clip", None)

    if abs(float(cleaned.get("PF_heatmap_state_z_weight", 1.0)) - 1.0) <= 1e-12:
        cleaned.pop("PF_heatmap_state_z_weight", None)

    resample_obs_power_adapt = float(cleaned.get("PF_heatmap_resample_obs_power_adapt", 0.0))
    if resample_obs_power_adapt <= 0.0:
        cleaned.pop("PF_heatmap_resample_obs_power_adapt", None)
        cleaned.pop("PF_heatmap_resample_min_threshold", None)
    else:
        min_threshold = float(cleaned.get("PF_heatmap_resample_min_threshold", 0.0))
        if min_threshold <= 0.0:
            cleaned.pop("PF_heatmap_resample_min_threshold", None)
        else:
            cleaned["PF_heatmap_resample_min_threshold"] = float(
                min(min_threshold, float(cleaned.get("PF_heatmap_resample_threshold", 0.5)))
            )
    rescue_frac = float(cleaned.get("PF_heatmap_rescue_frac", 0.0))
    rescue_pos_sd = float(cleaned.get("PF_heatmap_rescue_pos_sd", 0.0))
    rescue_rate_sd = float(cleaned.get("PF_heatmap_rescue_rate_sd", 0.0))
    if rescue_frac <= 0.0 or rescue_pos_sd <= 0.0 or rescue_rate_sd <= 0.0:
        cleaned.pop("PF_heatmap_rescue_frac", None)
        cleaned.pop("PF_heatmap_rescue_pos_sd", None)
        cleaned.pop("PF_heatmap_rescue_rate_sd", None)
    else:
        cleaned["PF_heatmap_rescue_frac"] = float(np.clip(rescue_frac, 0.0, 0.25))

    anchor_sigma = float(cleaned.get("PF_heatmap_anchor_sigma", 0.0))
    anchor_power = float(cleaned.get("PF_heatmap_anchor_power", 0.0))
    if anchor_sigma <= 0.0 or anchor_power <= 0.0:
        cleaned["PF_heatmap_anchor_sigma"] = 0.0
        cleaned["PF_heatmap_anchor_power"] = 0.0

    jump_prob = float(cleaned.get("PF_heatmap_jump_prob", 0.0))
    missing_jump_boost = float(cleaned.get("PF_heatmap_missing_jump_boost", 0.0))
    jump_sd = float(cleaned.get("PF_heatmap_jump_sd", 0.0))
    jump_rate_sd = float(cleaned.get("PF_heatmap_jump_rate_sd", 0.0))
    if jump_prob <= 0.0 and missing_jump_boost <= 0.0:
        cleaned["PF_heatmap_jump_prob"] = 0.0
        cleaned["PF_heatmap_missing_jump_boost"] = 0.0
        cleaned["PF_heatmap_jump_sd"] = 0.0
        cleaned["PF_heatmap_jump_rate_sd"] = 0.0
    elif jump_sd <= 0.0 and jump_rate_sd <= 0.0:
        cleaned["PF_heatmap_jump_prob"] = 0.0
        cleaned["PF_heatmap_missing_jump_boost"] = 0.0
        cleaned["PF_heatmap_jump_sd"] = 0.0
        cleaned["PF_heatmap_jump_rate_sd"] = 0.0

    jump_tail_prob = float(cleaned.get("PF_heatmap_jump_tail_prob", 0.0))
    jump_tail_missing_boost = float(cleaned.get("PF_heatmap_jump_tail_missing_boost", 0.0))
    jump_tail_sd = float(cleaned.get("PF_heatmap_jump_tail_sd", 0.0))
    jump_tail_rate_sd = float(cleaned.get("PF_heatmap_jump_tail_rate_sd", 0.0))
    if jump_tail_prob <= 0.0 and jump_tail_missing_boost <= 0.0:
        cleaned.pop("PF_heatmap_jump_tail_prob", None)
        cleaned.pop("PF_heatmap_jump_tail_sd", None)
        cleaned.pop("PF_heatmap_jump_tail_rate_sd", None)
        cleaned.pop("PF_heatmap_jump_tail_dist", None)
        cleaned.pop("PF_heatmap_jump_tail_clip", None)
        cleaned.pop("PF_heatmap_jump_tail_missing_boost", None)
    elif jump_tail_sd <= 0.0 and jump_tail_rate_sd <= 0.0:
        cleaned.pop("PF_heatmap_jump_tail_prob", None)
        cleaned.pop("PF_heatmap_jump_tail_sd", None)
        cleaned.pop("PF_heatmap_jump_tail_rate_sd", None)
        cleaned.pop("PF_heatmap_jump_tail_dist", None)
        cleaned.pop("PF_heatmap_jump_tail_clip", None)
        cleaned.pop("PF_heatmap_jump_tail_missing_boost", None)
    else:
        if int(cleaned.get("PF_heatmap_jump_tail_dist", 1)) == 1:
            cleaned.pop("PF_heatmap_jump_tail_dist", None)
        if abs(float(cleaned.get("PF_heatmap_jump_tail_clip", 0.0))) <= 1e-12:
            cleaned.pop("PF_heatmap_jump_tail_clip", None)
        if abs(float(cleaned.get("PF_heatmap_jump_tail_missing_boost", 0.0))) <= 1e-12:
            cleaned.pop("PF_heatmap_jump_tail_missing_boost", None)

    if int(cleaned.get("PF_heatmap_stratified_init", 0)) == 0:
        cleaned.pop("PF_heatmap_stratified_init", None)
    for key in ("PF_heatmap_correlated_rate_alpha", "PF_heatmap_correlated_pos_alpha"):
        if abs(float(cleaned.get(key, 0.0))) <= 0.0:
            cleaned.pop(key, None)
    lookahead_power = float(cleaned.get("PF_heatmap_lookahead_power", 0.0))
    if lookahead_power <= 0.0:
        cleaned.pop("PF_heatmap_lookahead_power", None)
        cleaned.pop("PF_heatmap_lookahead_steps", None)
        cleaned.pop("PF_heatmap_lookahead_decay", None)
        cleaned.pop("PF_heatmap_lookahead_max_gap", None)
        cleaned.pop("PF_heatmap_lookahead_delta_power", None)
    else:
        if int(cleaned.get("PF_heatmap_lookahead_steps", 1)) == 1:
            cleaned.pop("PF_heatmap_lookahead_steps", None)
        if abs(float(cleaned.get("PF_heatmap_lookahead_decay", 0.5)) - 0.5) <= 1e-12:
            cleaned.pop("PF_heatmap_lookahead_decay", None)
        if abs(float(cleaned.get("PF_heatmap_lookahead_max_gap", 256.0)) - 256.0) <= 1e-12:
            cleaned.pop("PF_heatmap_lookahead_max_gap", None)
        if abs(float(cleaned.get("PF_heatmap_lookahead_delta_power", 0.0))) <= 1e-12:
            cleaned.pop("PF_heatmap_lookahead_delta_power", None)
    if float(cleaned.get("PF_heatmap_ref_grid_step", 0.2)) == 0.2:
        cleaned.pop("PF_heatmap_ref_grid_step", None)
    if abs(float(cleaned.get("PF_heatmap_finite_run_power_boost", 0.0))) <= 0.0:
        cleaned.pop("PF_heatmap_finite_run_power_boost", None)
        cleaned.pop("PF_heatmap_finite_run_power_cap", None)
    else:
        cap = float(cleaned.get("PF_heatmap_finite_run_power_cap", 1.0))
        if cap < 1.0:
            cleaned["PF_heatmap_finite_run_power_cap"] = 1.0
    finite_run_decay = float(cleaned.get("PF_heatmap_finite_run_power_decay", 0.0))
    if finite_run_decay <= 0.0:
        cleaned.pop("PF_heatmap_finite_run_power_decay", None)
        cleaned.pop("PF_heatmap_finite_run_power_floor", None)
    else:
        floor = float(cleaned.get("PF_heatmap_finite_run_power_floor", 0.35))
        cleaned["PF_heatmap_finite_run_power_floor"] = float(np.clip(floor, 0.0, 1.0))
    conf_obs_power_decay = float(cleaned.get("PF_heatmap_conf_obs_power_decay", 0.0))
    if conf_obs_power_decay <= 0.0:
        cleaned.pop("PF_heatmap_conf_obs_power_decay", None)
        cleaned.pop("PF_heatmap_conf_obs_power_floor", None)
        cleaned.pop("PF_heatmap_conf_obs_power_power", None)
    else:
        cleaned["PF_heatmap_conf_obs_power_floor"] = float(
            np.clip(float(cleaned.get("PF_heatmap_conf_obs_power_floor", 0.55)), 0.0, 1.0)
        )
        if abs(float(cleaned.get("PF_heatmap_conf_obs_power_power", 1.0)) - 1.0) <= 1e-12:
            cleaned.pop("PF_heatmap_conf_obs_power_power", None)
    if abs(float(cleaned.get("PF_heatmap_missing_noise_scale", 1.0)) - 1.0) <= 1e-12:
        cleaned.pop("PF_heatmap_missing_noise_scale", None)
    if abs(float(cleaned.get("PF_heatmap_missing_jump_scale", 1.0)) - 1.0) <= 1e-12:
        cleaned.pop("PF_heatmap_missing_jump_scale", None)
    ess_jump_boost = float(cleaned.get("PF_heatmap_ess_jump_boost", 0.0))
    if ess_jump_boost <= 0.0:
        cleaned.pop("PF_heatmap_ess_jump_boost", None)
        cleaned.pop("PF_heatmap_ess_jump_power", None)
    elif abs(float(cleaned.get("PF_heatmap_ess_jump_power", 1.0)) - 1.0) <= 1e-12:
        cleaned.pop("PF_heatmap_ess_jump_power", None)
    surprise_jump_boost = float(cleaned.get("PF_heatmap_surprise_jump_boost", 0.0))
    if surprise_jump_boost <= 0.0:
        cleaned.pop("PF_heatmap_surprise_jump_boost", None)
        cleaned.pop("PF_heatmap_surprise_jump_threshold", None)
        cleaned.pop("PF_heatmap_surprise_jump_power", None)
    else:
        if abs(float(cleaned.get("PF_heatmap_surprise_jump_threshold", 1.5)) - 1.5) <= 1e-12:
            cleaned.pop("PF_heatmap_surprise_jump_threshold", None)
        if abs(float(cleaned.get("PF_heatmap_surprise_jump_power", 1.0)) - 1.0) <= 1e-12:
            cleaned.pop("PF_heatmap_surprise_jump_power", None)
    if abs(float(cleaned.get("PF_heatmap_ess_rough_boost", 0.0))) <= 0.0:
        cleaned.pop("PF_heatmap_ess_rough_boost", None)
        cleaned.pop("PF_heatmap_ess_rough_power", None)
    elif float(cleaned.get("PF_heatmap_ess_rough_power", 1.0)) == 1.0:
        cleaned.pop("PF_heatmap_ess_rough_power", None)
    if abs(float(cleaned.get("PF_heatmap_prob_temperature", 1.0)) - 1.0) <= 1e-12:
        cleaned.pop("PF_heatmap_prob_temperature", None)
    if str(cleaned.get("PF_heatmap_ffbsi_transition_model", "gaussian")).lower() == "gaussian":
        cleaned.pop("PF_heatmap_ffbsi_transition_model", None)
    merge_k = int(cleaned.get("PF_heatmap_suffix_merge_k", 1))
    if merge_k <= 1:
        cleaned.pop("PF_heatmap_suffix_merge_k", None)
        cleaned.pop("PF_heatmap_suffix_merge_obs_power_alpha", None)
        cleaned.pop("PF_heatmap_suffix_merge_adjust_dynamics", None)
        cleaned.pop("PF_heatmap_suffix_merge_likelihood_mode", None)
        cleaned.pop("PF_heatmap_suffix_merge_state_mode", None)
    else:
        cleaned["PF_heatmap_suffix_merge_k"] = int(merge_k)
        merge_likelihood_mode = str(cleaned.get("PF_heatmap_suffix_merge_likelihood_mode", "block")).lower()
        merge_default_alpha = 1.0 if merge_likelihood_mode == "block" else 0.5
        merge_default_adjust = False if merge_likelihood_mode == "block" else True
        if abs(float(cleaned.get("PF_heatmap_suffix_merge_obs_power_alpha", merge_default_alpha)) - merge_default_alpha) <= 1e-12:
            cleaned.pop("PF_heatmap_suffix_merge_obs_power_alpha", None)
        if bool(cleaned.get("PF_heatmap_suffix_merge_adjust_dynamics", merge_default_adjust)) == merge_default_adjust:
            cleaned.pop("PF_heatmap_suffix_merge_adjust_dynamics", None)
        if merge_likelihood_mode == "block":
            cleaned.pop("PF_heatmap_suffix_merge_likelihood_mode", None)
        if str(cleaned.get("PF_heatmap_suffix_merge_state_mode", "start")).lower() == "start":
            cleaned.pop("PF_heatmap_suffix_merge_state_mode", None)
    if "PF_heatmap_profile_mixture_spec" in cleaned:
        spec = _normalize_profile_mixture_spec(cleaned["PF_heatmap_profile_mixture_spec"])
        if spec:
            cleaned["PF_heatmap_profile_mixture_spec"] = spec
        else:
            cleaned.pop("PF_heatmap_profile_mixture_spec", None)
    if n_particles_meta is not None:
        n_particles_meta = int(n_particles_meta)
        n_seeds_meta = 64 if n_seeds_meta is None and n_particles_meta == 1000 else n_seeds_meta
        n_seeds_meta = 128 if n_seeds_meta is None and n_particles_meta == 500 else n_seeds_meta
        if n_seeds_meta is None:
            raise ValueError("candidate budget metadata requires both particles and seeds")
        n_particles_meta, n_seeds_meta = _pf_opt_validate_merge64_budget(n_particles_meta, n_seeds_meta)
        cleaned[PF_OPT_BUDGET_META_N_PARTICLES] = int(n_particles_meta)
        cleaned[PF_OPT_BUDGET_META_N_SEEDS] = int(n_seeds_meta)
    return cleaned


def _random_pf_broad_candidates(round_idx, current_overrides, n_candidates=32):
    rng = np.random.default_rng(20260605 + 7919 * int(round_idx))
    space = PF_OPT_RANDOM_SEARCH_SPACE
    keys = list(space)
    key_dims = [key for key in PF_OPT_RANDOM_KEY_DIMS if key in space]
    structural_dims = [key for key in PF_OPT_STRUCTURAL_KEY_DIMS if key in space]
    side_dims = [key for key in keys if key not in set(key_dims)] + ["PF_heatmap_profile_mixture_spec"]
    candidates = []
    seen = {json.dumps(_clean_pf_random_overrides(current_overrides), sort_keys=True)}
    attempts = 0
    while len(candidates) < int(n_candidates) and attempts < int(n_candidates) * 80:
        attempts += 1
        overrides = dict(current_overrides)
        if rng.random() < 0.85:
            n_key_changes = int(rng.choice([1, 2, 3], p=[0.35, 0.50, 0.15]))
            n_side_changes = int(rng.choice([0, 1], p=[0.82, 0.18]))
        else:
            n_key_changes = int(rng.choice([3, 4], p=[0.65, 0.35]))
            n_side_changes = int(rng.choice([0, 1, 2], p=[0.55, 0.35, 0.10]))
        n_key_changes = min(n_key_changes, len(key_dims))
        n_side_changes = min(n_side_changes, len(side_dims))
        changed = []
        if n_key_changes > 0:
            changed.extend(rng.choice(key_dims, size=n_key_changes, replace=False).tolist())
        if structural_dims and rng.random() < 0.45:
            n_struct_changes = 1
            n_struct_changes = min(n_struct_changes, len(structural_dims))
            changed.extend(rng.choice(structural_dims, size=n_struct_changes, replace=False).tolist())
        if n_side_changes > 0:
            changed.extend(rng.choice(side_dims, size=n_side_changes, replace=False).tolist())
        if "PF_heatmap_resample_obs_power_adapt" in changed and "PF_heatmap_resample_min_threshold" not in changed:
            changed.append("PF_heatmap_resample_min_threshold")
        if "PF_heatmap_resample_min_threshold" in changed and "PF_heatmap_resample_obs_power_adapt" not in changed:
            changed.append("PF_heatmap_resample_obs_power_adapt")
        if "PF_heatmap_gr_ambiguity_power" in changed:
            if "PF_heatmap_gr_ambiguity_min_power" not in changed:
                changed.append("PF_heatmap_gr_ambiguity_min_power")
            if "PF_heatmap_gr_ambiguity_ref_mode" not in changed and rng.random() < 0.35:
                changed.append("PF_heatmap_gr_ambiguity_ref_mode")
        if ("PF_heatmap_gr_ambiguity_min_power" in changed or "PF_heatmap_gr_ambiguity_ref_mode" in changed) and "PF_heatmap_gr_ambiguity_power" not in changed:
            changed.append("PF_heatmap_gr_ambiguity_power")
        if "PF_heatmap_gr_information_power" in changed:
            if "PF_heatmap_gr_information_center" not in changed:
                changed.append("PF_heatmap_gr_information_center")
            if "PF_heatmap_gr_information_min_multiplier" not in changed:
                changed.append("PF_heatmap_gr_information_min_multiplier")
            if "PF_heatmap_gr_information_max_multiplier" not in changed:
                changed.append("PF_heatmap_gr_information_max_multiplier")
            if "PF_heatmap_gr_information_slope_weight" not in changed:
                changed.append("PF_heatmap_gr_information_slope_weight")
        if (
            "PF_heatmap_gr_information_center" in changed
            or "PF_heatmap_gr_information_min_multiplier" in changed
            or "PF_heatmap_gr_information_max_multiplier" in changed
            or "PF_heatmap_gr_information_slope_weight" in changed
        ) and "PF_heatmap_gr_information_power" not in changed:
            changed.append("PF_heatmap_gr_information_power")
        if "PF_heatmap_conf_obs_power_decay" in changed:
            if "PF_heatmap_conf_obs_power_floor" not in changed:
                changed.append("PF_heatmap_conf_obs_power_floor")
            if "PF_heatmap_conf_obs_power_power" not in changed and rng.random() < 0.45:
                changed.append("PF_heatmap_conf_obs_power_power")
        if ("PF_heatmap_conf_obs_power_floor" in changed or "PF_heatmap_conf_obs_power_power" in changed) and "PF_heatmap_conf_obs_power_decay" not in changed:
            changed.append("PF_heatmap_conf_obs_power_decay")
        if "PF_heatmap_ess_jump_boost" in changed and "PF_heatmap_ess_jump_power" not in changed and rng.random() < 0.45:
            changed.append("PF_heatmap_ess_jump_power")
        if "PF_heatmap_ess_jump_power" in changed and "PF_heatmap_ess_jump_boost" not in changed:
            changed.append("PF_heatmap_ess_jump_boost")
        if "PF_heatmap_surprise_jump_boost" in changed:
            if "PF_heatmap_surprise_jump_threshold" not in changed:
                changed.append("PF_heatmap_surprise_jump_threshold")
            if "PF_heatmap_surprise_jump_power" not in changed and rng.random() < 0.45:
                changed.append("PF_heatmap_surprise_jump_power")
        if ("PF_heatmap_surprise_jump_threshold" in changed or "PF_heatmap_surprise_jump_power" in changed) and "PF_heatmap_surprise_jump_boost" not in changed:
            changed.append("PF_heatmap_surprise_jump_boost")
        if "PF_heatmap_jump_tail_prob" in changed:
            for tail_key in (
                "PF_heatmap_jump_tail_sd",
                "PF_heatmap_jump_tail_rate_sd",
                "PF_heatmap_jump_tail_clip",
            ):
                if tail_key not in changed:
                    changed.append(tail_key)
            if rng.random() < 0.35 and "PF_heatmap_jump_tail_missing_boost" not in changed:
                changed.append("PF_heatmap_jump_tail_missing_boost")
        if (
            "PF_heatmap_jump_tail_sd" in changed
            or "PF_heatmap_jump_tail_rate_sd" in changed
            or "PF_heatmap_jump_tail_dist" in changed
            or "PF_heatmap_jump_tail_clip" in changed
            or "PF_heatmap_jump_tail_missing_boost" in changed
        ) and "PF_heatmap_jump_tail_prob" not in changed:
            changed.append("PF_heatmap_jump_tail_prob")
        if "PF_heatmap_outlier_prob" in changed and "PF_heatmap_outlier_likelihood" in space and rng.random() < 0.40:
            changed.append("PF_heatmap_outlier_likelihood")
        if "PF_heatmap_dynamic_sigma_alpha" in changed:
            for sigma_key in (
                "PF_heatmap_dynamic_sigma_threshold",
                "PF_heatmap_dynamic_sigma_power",
                "PF_heatmap_dynamic_sigma_min",
                "PF_heatmap_dynamic_sigma_max",
            ):
                if sigma_key not in changed and rng.random() < 0.50:
                    changed.append(sigma_key)
        if (
            "PF_heatmap_dynamic_sigma_threshold" in changed
            or "PF_heatmap_dynamic_sigma_power" in changed
            or "PF_heatmap_dynamic_sigma_min" in changed
            or "PF_heatmap_dynamic_sigma_max" in changed
        ) and "PF_heatmap_dynamic_sigma_alpha" not in changed:
            changed.append("PF_heatmap_dynamic_sigma_alpha")
        if "PF_heatmap_lookahead_power" in changed:
            for look_key in (
                "PF_heatmap_lookahead_steps",
                "PF_heatmap_lookahead_decay",
                "PF_heatmap_lookahead_max_gap",
                "PF_heatmap_lookahead_delta_power",
            ):
                if look_key not in changed and rng.random() < 0.60:
                    changed.append(look_key)
        if (
            "PF_heatmap_lookahead_steps" in changed
            or "PF_heatmap_lookahead_decay" in changed
            or "PF_heatmap_lookahead_max_gap" in changed
            or "PF_heatmap_lookahead_delta_power" in changed
        ) and "PF_heatmap_lookahead_power" not in changed:
            changed.append("PF_heatmap_lookahead_power")
        if "PF_heatmap_ffbsi_transition_scale" in changed:
            if "PF_heatmap_ffbsi_pos_floor" not in changed and rng.random() < 0.45:
                changed.append("PF_heatmap_ffbsi_pos_floor")
            if "PF_heatmap_ffbsi_rate_floor" not in changed and rng.random() < 0.45:
                changed.append("PF_heatmap_ffbsi_rate_floor")
            if "PF_heatmap_ffbsi_n_paths" not in changed and rng.random() < 0.30:
                changed.append("PF_heatmap_ffbsi_n_paths")
        changed = sorted(set(changed))
        for key in changed:
            if key == "PF_heatmap_profile_mixture_spec":
                profile_names = tuple(PF_OPT_PROFILE_MIXTURES)
                overrides[key] = PF_OPT_PROFILE_MIXTURES[str(rng.choice(profile_names))]
                continue
            values = PF_OPT_STRUCTURAL_ACTIVE_VALUES.get(key, space[key])
            cur_value = overrides.get(key, None)
            if len(values) <= 1:
                overrides[key] = values[0]
                continue
            for _ in range(6):
                value = values[int(rng.integers(0, len(values)))]
                if value != cur_value:
                    overrides[key] = value
                    break
            else:
                overrides[key] = values[int(rng.integers(0, len(values)))]
        finite_run_decay = float(overrides.get("PF_heatmap_finite_run_power_decay", 0.0))
        if finite_run_decay > 0.0 and "PF_heatmap_finite_run_power_floor" not in changed:
            overrides["PF_heatmap_finite_run_power_floor"] = float(
                rng.choice(_pf_opt_finite_run_floor_values(finite_run_decay))
            )
        elif finite_run_decay > 0.0:
            floor_values = _pf_opt_finite_run_floor_values(finite_run_decay)
            floor = float(overrides.get("PF_heatmap_finite_run_power_floor", floor_values[0]))
            if floor not in floor_values:
                overrides["PF_heatmap_finite_run_power_floor"] = float(floor_values[0])
        ambiguity_power = float(overrides.get("PF_heatmap_gr_ambiguity_power", 0.0))
        if ambiguity_power > 0.0:
            if "PF_heatmap_gr_ambiguity_min_power" not in overrides:
                overrides["PF_heatmap_gr_ambiguity_min_power"] = float(
                    rng.choice(space["PF_heatmap_gr_ambiguity_min_power"])
                )
            if "PF_heatmap_gr_ambiguity_ref_mode" not in overrides and rng.random() < 0.20:
                overrides["PF_heatmap_gr_ambiguity_ref_mode"] = str(
                    rng.choice(space["PF_heatmap_gr_ambiguity_ref_mode"])
                )
        information_power = float(overrides.get("PF_heatmap_gr_information_power", 0.0))
        if information_power > 0.0:
            overrides.setdefault("PF_heatmap_gr_information_center", float(rng.choice(space["PF_heatmap_gr_information_center"])))
            overrides.setdefault("PF_heatmap_gr_information_min_multiplier", float(rng.choice(space["PF_heatmap_gr_information_min_multiplier"])))
            overrides.setdefault("PF_heatmap_gr_information_max_multiplier", float(rng.choice(space["PF_heatmap_gr_information_max_multiplier"])))
        conf_obs_power_decay = float(overrides.get("PF_heatmap_conf_obs_power_decay", 0.0))
        if conf_obs_power_decay > 0.0:
            overrides.setdefault("PF_heatmap_conf_obs_power_floor", float(rng.choice(space["PF_heatmap_conf_obs_power_floor"])))
        ess_jump_boost = float(overrides.get("PF_heatmap_ess_jump_boost", 0.0))
        if ess_jump_boost > 0.0:
            overrides.setdefault("PF_heatmap_ess_jump_power", float(rng.choice(space["PF_heatmap_ess_jump_power"])))
        surprise_jump_boost = float(overrides.get("PF_heatmap_surprise_jump_boost", 0.0))
        if surprise_jump_boost > 0.0:
            overrides.setdefault("PF_heatmap_surprise_jump_threshold", float(rng.choice(space["PF_heatmap_surprise_jump_threshold"])))
        jump_tail_prob = float(overrides.get("PF_heatmap_jump_tail_prob", 0.0))
        jump_tail_missing_boost = float(overrides.get("PF_heatmap_jump_tail_missing_boost", 0.0))
        if jump_tail_prob > 0.0 or jump_tail_missing_boost > 0.0:
            overrides.setdefault("PF_heatmap_jump_tail_sd", float(rng.choice(space["PF_heatmap_jump_tail_sd"])))
            overrides.setdefault("PF_heatmap_jump_tail_rate_sd", float(rng.choice(space["PF_heatmap_jump_tail_rate_sd"])))
            overrides.setdefault("PF_heatmap_jump_tail_clip", float(rng.choice(space["PF_heatmap_jump_tail_clip"])))
        outlier_prob = float(overrides.get("PF_heatmap_outlier_prob", 0.0))
        if outlier_prob > 0.0:
            overrides.setdefault("PF_heatmap_outlier_likelihood", 0.05)
        dynamic_sigma_alpha = float(overrides.get("PF_heatmap_dynamic_sigma_alpha", 0.0))
        if dynamic_sigma_alpha > 0.0:
            overrides.setdefault("PF_heatmap_dynamic_sigma_threshold", 1.35)
            overrides.setdefault("PF_heatmap_dynamic_sigma_power", 0.60)
            overrides.setdefault("PF_heatmap_dynamic_sigma_min", 0.90)
            overrides.setdefault("PF_heatmap_dynamic_sigma_max", 1.75)
        lookahead_power = float(overrides.get("PF_heatmap_lookahead_power", 0.0))
        if lookahead_power > 0.0:
            overrides.setdefault("PF_heatmap_lookahead_steps", 2)
            overrides.setdefault("PF_heatmap_lookahead_decay", 0.50)
            overrides.setdefault("PF_heatmap_lookahead_max_gap", 64.0)
        overrides = _clean_pf_random_overrides(overrides)
        payload = json.dumps(overrides, sort_keys=True)
        if payload in seen:
            continue
        seen.add(payload)
        name = f"rs{int(round_idx):04d}_rand{len(candidates):02d}"
        candidates.append((name, overrides))
    if len(candidates) != int(n_candidates):
        raise RuntimeError(f"only generated {len(candidates)} unique random PF candidates")
    return candidates


def _format_brief_candidate_ranking(summaries, limit=3, objective_metric="direct"):
    objective_key = _pf_opt_objective_key(objective_metric)
    ordered = sorted(summaries, key=lambda item: _pf_opt_metric_value(item, objective_key))
    return ", ".join(f"{item['name']}={_pf_opt_metric_value(item, objective_key):.6f}" for item in ordered[: int(limit)])


def _random_pf_opt_candidates(round_idx, current_overrides, n_candidates=12):
    rng = np.random.default_rng(20260605 + int(round_idx))
    candidate_space = {
        "PF_heatmap_query_gr_mode": ["interp", "skip", "soft_interp"],
        "PF_heatmap_missing_likelihood_power": [0.10, 0.25, 0.50, 0.75],
        "PF_heatmap_seed_path_weight_mode": ["likelihood", "equal", "rank"],
        "PF_heatmap_seed_prob_weight_mode": ["equal", "likelihood", "rank"],
        "PF_heatmap_lik_scale": [2.5, 5.0, 7.5, 10.0, 15.0, 20.0],
        "PF_heatmap_momentum": [0.994, 0.996, 0.998, 0.999, 0.9995],
        "PF_heatmap_rate_noise": [0.0008, 0.0012, 0.0020, 0.0030, 0.0040],
        "PF_heatmap_pos_noise": [0.0015, 0.0030, 0.0050, 0.0080, 0.0120],
        "PF_heatmap_resample_threshold": [0.30, 0.40, 0.50, 0.65, 0.80],
        "PF_heatmap_rough_pos": [0.03, 0.05, 0.10, 0.16, 0.25],
        "PF_heatmap_rough_rate": [0.0003, 0.0005, 0.0010, 0.0016, 0.0025],
        "PF_heatmap_init_pos_sd": [1.0, 2.0, 3.0, 4.5, 6.0],
        "PF_heatmap_init_rate_sd": [0.004, 0.007, 0.010, 0.015],
        "PF_heatmap_rate_mean_weight": [0.0, 0.00025, 0.0005, 0.0010, 0.0020],
        "PF_heatmap_anchor_sigma": [0.0, 35.0, 50.0, 65.0],
        "PF_heatmap_anchor_power": [0.0, 0.001, 0.002, 0.005],
        "PF_heatmap_jump_prob": [0.0, 0.0001, 0.0002, 0.0005, 0.0010],
        "PF_heatmap_jump_sd": [0.0, 3.0, 5.0, 8.0, 12.0],
        "PF_heatmap_jump_rate_sd": [0.0, 0.0005, 0.0010, 0.0020],
    }
    names = []
    candidates = []
    keys = list(candidate_space)
    for k in range(int(n_candidates)):
        overrides = dict(current_overrides)
        n_changes = int(rng.integers(2, 6))
        changed = sorted(rng.choice(keys, size=n_changes, replace=False).tolist())
        for key in changed:
            values = candidate_space[key]
            overrides[key] = values[int(rng.integers(0, len(values)))]
        if overrides.get("PF_heatmap_query_gr_mode") != "soft_interp":
            overrides.pop("PF_heatmap_missing_likelihood_power", None)
        if float(overrides.get("PF_heatmap_anchor_sigma", 0.0)) <= 0.0:
            overrides["PF_heatmap_anchor_power"] = 0.0
        if float(overrides.get("PF_heatmap_jump_prob", 0.0)) <= 0.0:
            overrides["PF_heatmap_jump_sd"] = 0.0
            overrides["PF_heatmap_jump_rate_sd"] = 0.0
        name_bits = [f"r{round_idx:03d}", f"rand{k:02d}"]
        names.append("_".join(name_bits))
        candidates.append((names[-1], overrides))
    return candidates


def _random_pf_failure_guided_candidates(round_idx, current_overrides, n_candidates=16, focus_names=None):
    rng = np.random.default_rng(20260616 + int(round_idx))
    focus_names = list(focus_names or PF_FAILURE_GUIDED_FOCUS_CHOICES)
    focus_names = [name for name in focus_names if name in PF_FAILURE_GUIDED_FOCUS_SPACES]
    if not focus_names:
        focus_names = ["top_pf_failures"]
    candidates = []
    seen = {json.dumps(_clean_pf_random_overrides(current_overrides), sort_keys=True)}
    attempts = 0
    while len(candidates) < int(n_candidates) and attempts < int(n_candidates) * 100:
        attempts += 1
        focus_name = focus_names[int(rng.integers(0, len(focus_names)))]
        space = PF_FAILURE_GUIDED_FOCUS_SPACES[focus_name]
        keys = list(space)
        overrides = dict(current_overrides)
        n_changes = int(rng.integers(2, min(5, len(keys)) + 1))
        changed = rng.choice(keys, size=min(n_changes, len(keys)), replace=False).tolist()
        if focus_name == "finite_gr_bad" and "PF_heatmap_finite_run_power_decay" not in changed and rng.random() < 0.75:
            changed.append("PF_heatmap_finite_run_power_decay")
        if "PF_heatmap_resample_obs_power_adapt" in changed and "PF_heatmap_resample_min_threshold" not in changed:
            changed.append("PF_heatmap_resample_min_threshold")
        if "PF_heatmap_resample_min_threshold" in changed and "PF_heatmap_resample_obs_power_adapt" not in changed:
            changed.append("PF_heatmap_resample_obs_power_adapt")
        if "PF_heatmap_rescue_frac" in changed:
            if "PF_heatmap_rescue_pos_sd" not in changed:
                changed.append("PF_heatmap_rescue_pos_sd")
            if "PF_heatmap_rescue_rate_sd" not in changed:
                changed.append("PF_heatmap_rescue_rate_sd")
        if ("PF_heatmap_rescue_pos_sd" in changed or "PF_heatmap_rescue_rate_sd" in changed) and "PF_heatmap_rescue_frac" not in changed:
            changed.append("PF_heatmap_rescue_frac")
        if focus_name == "low_anchor_mass_drift":
            changed.append("PF_heatmap_anchor_power")
            if rng.random() < 0.65:
                changed.append("PF_heatmap_anchor_sigma")
            if not any(
                key in changed
                for key in (
                    "PF_heatmap_gr_ambiguity_power",
                    "PF_heatmap_finite_run_power_decay",
                    "PF_heatmap_momentum",
                    "PF_heatmap_rate_noise",
                    "PF_heatmap_pos_noise",
                    "PF_heatmap_rough_pos",
                    "PF_heatmap_rough_rate",
                )
            ):
                changed.append("PF_heatmap_finite_run_power_decay")
        changed = sorted(set(changed))
        for key in changed:
            values = space[key]
            cur_value = overrides.get(key, None)
            for _ in range(8):
                value = values[int(rng.integers(0, len(values)))]
                if value != cur_value:
                    overrides[key] = value
                    break
            else:
                overrides[key] = values[int(rng.integers(0, len(values)))]
        finite_run_decay = float(overrides.get("PF_heatmap_finite_run_power_decay", 0.0))
        if finite_run_decay > 0.0:
            floor_choices = space.get(
                "PF_heatmap_finite_run_power_floor",
                PF_FAILURE_GUIDED_FOCUS_SPACES["top_pf_failures"]["PF_heatmap_finite_run_power_floor"],
            )
            overrides["PF_heatmap_finite_run_power_floor"] = float(rng.choice(floor_choices))
        if float(overrides.get("PF_heatmap_gr_ambiguity_power", 0.0)) <= 0.0:
            overrides.pop("PF_heatmap_gr_ambiguity_min_power", None)
            overrides.pop("PF_heatmap_gr_ambiguity_ref_mode", None)
        if float(overrides.get("PF_heatmap_dynamic_sigma_alpha", 0.0)) <= 0.0:
            overrides.pop("PF_heatmap_dynamic_sigma_threshold", None)
            overrides.pop("PF_heatmap_dynamic_sigma_power", None)
            overrides.pop("PF_heatmap_dynamic_sigma_min", None)
            overrides.pop("PF_heatmap_dynamic_sigma_max", None)
        elif "PF_heatmap_dynamic_sigma_alpha" in overrides:
            overrides.setdefault("PF_heatmap_dynamic_sigma_threshold", 1.35)
            overrides.setdefault("PF_heatmap_dynamic_sigma_power", 0.60)
            overrides.setdefault("PF_heatmap_dynamic_sigma_min", 0.90)
            overrides.setdefault("PF_heatmap_dynamic_sigma_max", 1.75)
        if float(overrides.get("PF_heatmap_outlier_prob", 0.0)) > 0.0:
            overrides.setdefault("PF_heatmap_outlier_likelihood", 0.05)
        overrides = _clean_pf_random_overrides(overrides)
        payload = json.dumps(overrides, sort_keys=True)
        if payload in seen:
            continue
        seen.add(payload)
        candidates.append((f"fg{int(round_idx):04d}_{focus_name}_{len(candidates):02d}", overrides))
    if len(candidates) != int(n_candidates):
        raise RuntimeError(f"only generated {len(candidates)} unique failure-guided PF candidates")
    return candidates


def run_pf_opt_loop(well_limit=50, full_well_limit=None, workers=8, data_path=None, max_rounds=None, random_candidates=12):
    _append_progress(
        "Optimization loop start: fixed budget `1000 x 64`, "
        f"first-stage wells={well_limit}, full_well_limit={full_well_limit}, workers={workers}."
    )
    current_name = "baseline"
    current_overrides = {}
    baseline = benchmark_pf_opt_candidate(current_name, current_overrides, well_limit=well_limit, workers=workers, data_path=data_path)
    current_50 = baseline
    current_full = None
    _update_active_best_progress(
        f"- Active first-50: `{current_name}` RMSE `{current_50['rmse']:.6f}` with overrides `{json.dumps(current_overrides, sort_keys=True)}`.\n"
        f"- Active full: pending."
    )
    _append_progress("Round 1 baseline result: " + _format_pf_opt_summary(baseline))
    fixed_candidates = [(name, overrides) for name, overrides in PF_OPT_FIXED_CANDIDATES if name != "baseline"]
    round_idx = 1
    while max_rounds is None or round_idx <= int(max_rounds):
        PF_OPT_PROGRESS_PATH.read_text()
        if round_idx == 1:
            candidates = fixed_candidates
        else:
            candidates = _random_pf_opt_candidates(round_idx, current_overrides, n_candidates=random_candidates)
        _append_progress(f"Round {round_idx} start: base `{current_name}` first-50 RMSE `{current_50['rmse']:.6f}`, candidates={len(candidates)}.")
        for cand_name, cand_delta in candidates:
            cand_overrides = dict(current_overrides)
            cand_overrides.update(cand_delta)
            cand = benchmark_pf_opt_candidate(cand_name, cand_overrides, well_limit=well_limit, workers=workers, data_path=data_path)
            improved_50 = cand["rmse"] < current_50["rmse"] - 1e-9
            _append_progress(
                f"Candidate `{cand_name}` first-stage: {_format_pf_opt_summary(cand)} "
                f"overrides={json.dumps(cand_overrides, sort_keys=True)} "
                f"{'IMPROVED_50' if improved_50 else 'rejected_50'}."
            )
            if not improved_50:
                continue
            if full_well_limit is None:
                accepted = True
                cand_full = None
            else:
                if current_full is None:
                    current_full = benchmark_pf_opt_candidate(
                        current_name + "_full",
                        current_overrides,
                        well_limit=full_well_limit,
                        workers=workers,
                        data_path=data_path,
                    )
                    _append_progress("Current base full result: " + _format_pf_opt_summary(current_full))
                cand_full = benchmark_pf_opt_candidate(
                    cand_name + "_full",
                    cand_overrides,
                    well_limit=full_well_limit,
                    workers=workers,
                    data_path=data_path,
                )
                accepted = cand_full["rmse"] < current_full["rmse"] - 1e-9
                _append_progress(
                    f"Candidate `{cand_name}` full-stage: {_format_pf_opt_summary(cand_full)} "
                    f"{'ACCEPTED_FULL' if accepted else 'rejected_full'}."
                )
            if accepted:
                current_name = cand_name
                current_overrides = cand_overrides
                current_50 = cand
                if cand_full is not None:
                    current_full = cand_full
                full_text = (
                    f"`{current_full['name']}` RMSE `{current_full['rmse']:.6f}`"
                    if current_full is not None
                    else "pending"
                )
                _update_active_best_progress(
                    f"- Active first-50: `{current_name}` RMSE `{current_50['rmse']:.6f}` with overrides `{json.dumps(current_overrides, sort_keys=True)}`.\n"
                    f"- Active full: {full_text}."
                )
        round_idx += 1
    _append_progress("Optimization loop stopped by max_rounds.")


def run_pf_design_loop(well_limit=50, full_well_limit=773, workers=8, data_path=None, max_rounds=None):
    _append_progress(
        "Design optimization loop start from current best "
        f"`{PF_OPT_CURRENT_BEST_NAME}`: fixed budget `1000 x 64`, "
        f"first-stage wells={well_limit}, full_well_limit={full_well_limit}, workers={workers}."
    )
    current_name = PF_OPT_CURRENT_BEST_NAME
    current_overrides = dict(PF_OPT_CURRENT_BEST_OVERRIDES)
    current_50 = benchmark_pf_opt_candidate(
        current_name + "_design50_recheck",
        current_overrides,
        well_limit=well_limit,
        workers=workers,
        data_path=data_path,
    )
    current_full = {
        "name": PF_OPT_CURRENT_BEST_FULL_NAME,
        "rmse": PF_OPT_CURRENT_BEST_FULL_RMSE,
    }
    _append_progress("Design loop current-best first-stage recheck: " + _format_pf_opt_summary(current_50))
    _update_active_best_progress(
        f"- Active first-50: `{current_name}` RMSE `{current_50['rmse']:.6f}` with overrides `{json.dumps(current_overrides, sort_keys=True)}`.\n"
        f"- Active full: `{PF_OPT_CURRENT_BEST_FULL_NAME}` RMSE `{PF_OPT_CURRENT_BEST_FULL_RMSE:.6f}`."
    )

    round_idx = 1
    while max_rounds is None or round_idx <= int(max_rounds):
        PF_OPT_PROGRESS_PATH.read_text()
        _append_progress(
            f"Design round {round_idx} start: base `{current_name}` first-50 RMSE `{current_50['rmse']:.6f}`, "
            f"full RMSE `{current_full['rmse']:.6f}`, candidates={len(PF_OPT_DESIGN_CANDIDATES)}."
        )
        for cand_name, cand_delta, motivation in PF_OPT_DESIGN_CANDIDATES:
            cand_overrides = dict(current_overrides)
            cand_overrides.update(cand_delta)
            cand = benchmark_pf_opt_candidate(
                cand_name,
                cand_overrides,
                well_limit=well_limit,
                workers=workers,
                data_path=data_path,
            )
            improved_50 = cand["rmse"] < current_50["rmse"] - 1e-9
            _append_progress(
                f"Design candidate `{cand_name}` first-stage: {_format_pf_opt_summary(cand)} "
                f"motivation={json.dumps(motivation)} "
                f"overrides={json.dumps(cand_overrides, sort_keys=True)} "
                f"{'IMPROVED_50' if improved_50 else 'rejected_50'}."
            )
            if not improved_50:
                continue
            cand_full = benchmark_pf_opt_candidate(
                cand_name + "_full",
                cand_overrides,
                well_limit=full_well_limit,
                workers=workers,
                data_path=data_path,
            )
            accepted = cand_full["rmse"] < current_full["rmse"] - 1e-9
            _append_progress(
                f"Design candidate `{cand_name}` full-stage: {_format_pf_opt_summary(cand_full)} "
                f"{'ACCEPTED_FULL' if accepted else 'rejected_full'}."
            )
            if accepted:
                current_name = cand_name
                current_overrides = cand_overrides
                current_50 = cand
                current_full = cand_full
                _update_active_best_progress(
                    f"- Active first-50: `{current_name}` RMSE `{current_50['rmse']:.6f}` with overrides `{json.dumps(current_overrides, sort_keys=True)}`.\n"
                    f"- Active full: `{current_full['name']}` RMSE `{current_full['rmse']:.6f}`."
                )
        round_idx += 1
    _append_progress("Design optimization loop stopped by max_rounds.")


def run_pf_random_search_loop(
    sample_size=100,
    full_well_limit=None,
    workers=8,
    data_path=None,
    max_rounds=None,
    random_candidates=32,
    gate_margin=0.01,
    n_particles=500,
    n_seeds=32,
    full_check_n_particles=PF_OPT_FULL_CHECK_DEFAULT_PARTICLES,
    full_check_n_seeds=PF_OPT_FULL_CHECK_DEFAULT_SEEDS,
    full_check_base_seed=PF_OPT_FULL_CHECK_DEFAULT_BASE_SEED,
    search_space="broad",
):
    search_spec = _pf_random_search_space_spec(search_space)
    if search_spec["name"] == "merge64" and sample_size is not None and int(sample_size) == 50:
        sample_size = 200
    PF_OPT_PROGRESS_PATH.read_text()
    current_name = search_spec["current_name"]
    current_overrides = _clean_pf_random_overrides(search_spec["current_overrides"])
    current_budget = (int(n_particles), int(n_seeds))
    objective_metric = str(search_spec.get("objective_metric", "direct"))
    objective_key = _pf_opt_objective_key(objective_metric)
    gate_metric_keys = {label: key for key, label in _pf_opt_gate_metrics(objective_metric)}
    well_mean_key = gate_metric_keys["well_mean"]
    well_trim90_key = gate_metric_keys["well_trim90"]
    sample_seed_base = int(search_spec["sample_seed_base"])
    confirm_seed_base = int(search_spec["confirm_seed_base"])
    candidate_fn = search_spec["candidate_fn"]
    space_note_fn = search_spec["space_note_fn"]
    round_prefix = search_spec["round_prefix"]
    if int(full_check_base_seed) == int(PF_OPT_FULL_CHECK_DEFAULT_BASE_SEED):
        if search_spec["name"] == "merge16":
            full_check_base_seed = int(PF_OPT_MERGE16_CURRENT_BEST_FULL_BASE_SEED)
        elif search_spec["name"] == "merge64":
            full_check_base_seed = int(PF_OPT_MERGE64_CURRENT_BEST_FULL_BASE_SEED)
    _append_progress(
        "Random search loop start: "
        f"space=`{search_spec['name']}`, "
            f"default candidate budget `{int(n_particles)} x {int(n_seeds)}`, sample_wells={int(sample_size)}, "
            f"candidate/full-check budgets may be overridden by opt metadata, "
            f"default full_check_budget `{int(full_check_n_particles)} x {int(full_check_n_seeds)}`, "
        f"full_check_base_seed={int(full_check_base_seed)}, "
        f"random_candidates={int(random_candidates)}, gate_margin={float(gate_margin):.4f}, "
        f"objective=`{objective_metric}`, "
        "sample gate=`run all candidates on the sample shard, then send the best global/well gated candidate to repeat-shard confirmation`; "
        "full validation requires confirmation global gain >= half margin plus nonnegative well_mean/well_trim90, "
        f"workers={int(workers)}, full_well_limit={full_well_limit or 'all'}."
    )
    full_check_count = 0
    cfg = _make_pf_opt_cfg(
        data_path=data_path,
        overrides=current_overrides,
        n_particles=int(n_particles),
        n_seeds=int(n_seeds),
        workers=int(workers),
    )
    data_path = Path(data_path or cfg.train_path)
    all_wells = _discover_well_ids(data_path)
    if not all_wells:
        raise ValueError(f"no train wells found under {data_path}")
    full_wells = all_wells if full_well_limit is None else all_wells[: int(full_well_limit)]
    last_round = None
    round_idx = 1

    while max_rounds is None or round_idx <= int(max_rounds):
        gc.collect()
        PF_OPT_PROGRESS_PATH.read_text()
        sampled_wells = _sample_pf_opt_wells(all_wells, sample_size, round_idx, seed_base=sample_seed_base)
        round_label = f"{round_prefix}{round_idx:04d}"
        space_note = space_note_fn(last_round)
        candidates = candidate_fn(round_idx, current_overrides, n_candidates=random_candidates)
        candidate_budgets = sorted(
            {
                _pf_opt_candidate_budget(
                    cand_overrides,
                    default_n_particles=int(current_budget[0]),
                    default_n_seeds=int(current_budget[1]),
                )
                for _, cand_overrides in candidates
            }
        )
        if search_spec["name"] == "merge64":
            candidate_budgets = sorted(_pf_opt_validate_merge64_budget(*budget) for budget in candidate_budgets)
        baseline_by_budget = {}
        for base_n_particles, base_n_seeds in candidate_budgets:
            baseline_by_budget[(int(base_n_particles), int(base_n_seeds))] = benchmark_pf_opt_candidate(
                f"{round_label}_baseline_{_pf_opt_budget_label(base_n_particles, base_n_seeds)}",
                current_overrides,
                workers=workers,
                data_path=data_path,
                progress=True,
                well_ids=sampled_wells,
                n_particles=int(base_n_particles),
                n_seeds=int(base_n_seeds),
                enforce_budget=(int(base_n_particles), int(base_n_seeds)),
            )
        baseline = baseline_by_budget.get(
            current_budget,
            next(iter(baseline_by_budget.values())),
        )
        _append_progress(
            f"Round {round_idx} start: baseline `{current_name}` sample RMSE `{_pf_opt_metric_value(baseline, objective_key):.6f}` "
            f"on {len(sampled_wells)} random wells, budgets={','.join(_pf_opt_budget_label(*budget) for budget in candidate_budgets)}, "
            f"max_sample_attempts={len(candidates)}. Space: {space_note}"
        )

        sample_results = []
        for cand_name, cand_overrides in candidates:
            cand_n_particles, cand_n_seeds = _pf_opt_candidate_budget(
                cand_overrides,
                default_n_particles=int(current_budget[0]),
                default_n_seeds=int(current_budget[1]),
            )
            if search_spec["name"] == "merge64":
                cand_n_particles, cand_n_seeds = _pf_opt_validate_merge64_budget(cand_n_particles, cand_n_seeds)
            cand_baseline = baseline_by_budget[(int(cand_n_particles), int(cand_n_seeds))]
            cand = benchmark_pf_opt_candidate(
                cand_name,
                cand_overrides,
                workers=workers,
                data_path=data_path,
                progress=True,
                well_ids=sampled_wells,
                n_particles=int(cand_n_particles),
                n_seeds=int(cand_n_seeds),
                enforce_budget=(int(cand_n_particles), int(cand_n_seeds)),
            )
            cand["_baseline_summary"] = cand_baseline
            cand["_budget"] = (int(cand_n_particles), int(cand_n_seeds))
            sample_results.append(cand)
        best_sample, sample_gains, sample_gate_ok, passing_sample = _select_pf_opt_gated_best(
            baseline,
            sample_results,
            global_margin=float(gate_margin),
            aux_margin=0.0,
            objective_metric=objective_metric,
        )
        best_n_particles, best_n_seeds = _pf_opt_candidate_budget(
            best_sample["overrides"],
            default_n_particles=int(current_budget[0]),
            default_n_seeds=int(current_budget[1]),
        )
        if search_spec["name"] == "merge64":
            best_n_particles, best_n_seeds = _pf_opt_validate_merge64_budget(best_n_particles, best_n_seeds)
        best_baseline = best_sample.get("_baseline_summary", baseline)
        sample_gain = float(sample_gains["global"])
        top3 = _format_brief_candidate_ranking(sample_results, limit=3, objective_metric=objective_metric)
        best_overrides = _clean_pf_random_overrides(best_sample["overrides"])
        round_message = [
            f"Round {round_idx} sample summary: best-budget baseline `{_pf_opt_metric_value(best_baseline, objective_key):.6f}` "
            f"(well_mean `{_pf_opt_metric_value(best_baseline, well_mean_key):.6f}`, "
            f"trim90 `{_pf_opt_metric_value(best_baseline, well_trim90_key):.6f}`), "
            f"best sampled `{best_sample['name']}` `{_pf_opt_metric_value(best_sample, objective_key):.6f}` "
            f"at budget `{int(best_n_particles)}x{int(best_n_seeds)}`, "
            f"metric gains `{_format_pf_opt_metric_gains(sample_gains)}`, "
            f"attempts={len(sample_results)}/{len(candidates)}, passing={len(passing_sample)}, top3 `{top3}`.",
            f"Round {round_idx} best overrides: `{json.dumps(best_overrides, sort_keys=True)}`.",
        ]

        confirm = None
        confirm_ok = False
        if sample_gate_ok:
            confirm = _confirm_pf_opt_candidate_on_second_shard(
                round_label,
                current_name,
                current_overrides,
                best_sample["name"],
                best_overrides,
                all_wells,
                sample_size,
                round_idx,
                workers,
                data_path,
                current_budget[0],
                current_budget[1],
                gate_margin=float(gate_margin),
                seed_base=confirm_seed_base,
                exclude_well_ids=sampled_wells,
                objective_metric=objective_metric,
            )
            confirm_ok = bool(confirm["ok"])
            round_message.append(
                f"Round {round_idx} repeat-shard confirmation: baseline `{_pf_opt_metric_value(confirm['baseline'], objective_key):.6f}` "
                f"(well_mean `{_pf_opt_metric_value(confirm['baseline'], well_mean_key):.6f}`, "
                f"trim90 `{_pf_opt_metric_value(confirm['baseline'], well_trim90_key):.6f}`), "
                f"candidate `{_pf_opt_metric_value(confirm['candidate'], objective_key):.6f}` "
                f"(well_mean `{_pf_opt_metric_value(confirm['candidate'], well_mean_key):.6f}`, "
                f"trim90 `{_pf_opt_metric_value(confirm['candidate'], well_trim90_key):.6f}`), "
                f"wells `{confirm['well_count']}` overlap `{confirm['overlap_count']}`, "
                f"metric gains `{_format_pf_opt_metric_gains(confirm['gains'])}`, decision `{'passed' if confirm_ok else 'rejected'}`."
            )

        if sample_gate_ok and confirm_ok:
            full_check_count += 1
            full_check = _run_pf_opt_full_pair_check(
                f"{round_label}_check{full_check_count:03d}",
                current_name,
                current_overrides,
                best_sample["name"],
                best_overrides,
                full_wells,
                workers,
                data_path,
                full_check_idx=full_check_count,
                n_particles=int(full_check_n_particles),
                n_seeds=int(full_check_n_seeds),
                base_seed=int(full_check_base_seed),
                objective_metric=objective_metric,
            )
            sota_full = full_check["sota"]
            cand_full = full_check["candidate"]
            full_ok = bool(full_check["ok"])
            full_gains = full_check["gains"]
            full_gain = float(full_gains["global"])
            if full_ok:
                prev_name = current_name
                prev_full_rmse = sota_full["rmse"]
                current_name = best_sample["name"]
                current_overrides = best_overrides
                current_budget = (int(best_n_particles), int(best_n_seeds))
                round_message.append(
                    f"Round {round_idx} paired full check ACCEPTED: `{prev_name}` `{_pf_opt_metric_value(sota_full, objective_key):.6f}` -> "
                    f"`{cand_full['name']}` `{_pf_opt_metric_value(cand_full, objective_key):.6f}`, "
                    f"budget `{int(full_check['budget'][0])}x{int(full_check['budget'][1])}`, base_seed `{full_check['base_seed']}`, "
                    f"metric gains `{_format_pf_opt_metric_gains(full_gains)}`."
                )
                _update_active_best_progress(
                    f"- Active SOTA: `{current_name}` promoted by paired full check at "
                    f"`{int(full_check['budget'][0])} x {int(full_check['budget'][1])}`, "
                    f"base_seed `{full_check['base_seed']}`, objective `{objective_metric}`, direct RMSE `{cand_full['rmse']:.6f}`, "
                    f"FFBSi RMSE `{cand_full.get('rmse_ffbsi', math.nan):.6f}`.\n"
                    f"- Previous SOTA in same check: `{prev_name}` direct RMSE `{sota_full['rmse']:.6f}`.\n"
                    f"- Active overrides: `{json.dumps(current_overrides, sort_keys=True)}`."
                )
                last_round = {"status": "full_accepted", "sample_gain": sample_gain, "full_gain": full_gain}
            else:
                round_message.append(
                    f"Round {round_idx} paired full check rejected: SOTA `{current_name}` `{_pf_opt_metric_value(sota_full, objective_key):.6f}` "
                    f"(well_mean `{_pf_opt_metric_value(sota_full, well_mean_key):.6f}`, "
                    f"trim90 `{_pf_opt_metric_value(sota_full, well_trim90_key):.6f}`), "
                    f"candidate `{_pf_opt_metric_value(cand_full, objective_key):.6f}` "
                    f"(well_mean `{_pf_opt_metric_value(cand_full, well_mean_key):.6f}`, "
                    f"trim90 `{_pf_opt_metric_value(cand_full, well_trim90_key):.6f}`), "
                    f"budget `{int(full_check['budget'][0])}x{int(full_check['budget'][1])}`, base_seed `{full_check['base_seed']}`, "
                    f"metric gains `{_format_pf_opt_metric_gains(full_gains)}`."
                )
                last_round = {"status": "full_rejected", "sample_gain": sample_gain, "full_gain": full_gain}
        elif sample_gate_ok:
            round_message.append(
                f"Round {round_idx} no full validation: first shard passed, but repeat-shard confirmation did not meet "
                f"{objective_metric} global gain >= `{0.5 * float(gate_margin):.6f}` with nonnegative well_mean and well_trim90."
            )
            last_round = {
                "status": "confirm_rejected",
                "sample_gain": sample_gain,
                "confirm_gain": float(confirm["gains"]["global"]) if confirm else math.nan,
                "full_gain": math.nan,
            }
        else:
            round_message.append(
                f"Round {round_idx} no full validation: no candidate cleared {objective_metric} global gain "
                f"`{float(gate_margin):.6f}` while also improving well_mean and well_trim90. "
                f"Best global candidate metric gains "
                f"`{_format_pf_opt_metric_gains(sample_gains)}`."
            )
            last_round = {"status": "no_sample_gate", "sample_gain": sample_gain, "full_gain": math.nan}

        _append_progress("\n".join(round_message))
        gc.collect()
        round_idx += 1

    _append_progress("Random search loop stopped by max_rounds.")


def run_pf_failure_guided_search_loop(
    sample_size=50,
    full_well_limit=None,
    workers=8,
    data_path=None,
    cache_dir=None,
    output_dir=None,
    max_rounds=None,
    random_candidates=16,
    gate_margin=0.01,
    n_particles=500,
    n_seeds=32,
    full_check_n_particles=PF_OPT_FULL_CHECK_DEFAULT_PARTICLES,
    full_check_n_seeds=PF_OPT_FULL_CHECK_DEFAULT_SEEDS,
    full_check_base_seed=PF_OPT_FULL_CHECK_DEFAULT_BASE_SEED,
):
    _append_progress(
        "Failure-guided PF search loop start: "
        f"budget `{int(n_particles)} x {int(n_seeds)}`, sample_wells={int(sample_size)}, "
        f"full_check_budget `{int(full_check_n_particles)} x {int(full_check_n_seeds)}`, "
        f"full_check_base_seed={int(full_check_base_seed)}, "
        f"random_candidates={int(random_candidates)}, gate_margin={float(gate_margin):.4f}, "
        "full validation requires the directed bad-case/context gate plus an independent repeat-shard confirmation, "
        f"workers={int(workers)}, full_well_limit={full_well_limit or 'all'}."
    )
    current_name = PF_OPT_CURRENT_BEST_NAME
    current_overrides = _clean_pf_random_overrides(PF_OPT_CURRENT_BEST_OVERRIDES)
    full_check_count = 0
    cfg = _make_pf_opt_cfg(
        data_path=data_path,
        overrides=current_overrides,
        n_particles=int(n_particles),
        n_seeds=int(n_seeds),
        workers=int(workers),
    )
    data_path = Path(data_path or cfg.train_path)
    all_wells = _discover_well_ids(data_path)
    if not all_wells:
        raise ValueError(f"no train wells found under {data_path}")
    full_wells = all_wells if full_well_limit is None else all_wells[: int(full_well_limit)]

    analysis = run_cached_pf_failure_analysis(
        cache_dir=cache_dir,
        data_path=data_path,
        output_dir=output_dir,
        top_k=max(50, int(sample_size)),
        fold_count=5,
        seed=7,
    )
    focus_slices = _failure_guided_build_focus_slices(analysis, sample_size=max(1, int(sample_size)))
    if not focus_slices:
        raise ValueError("failure-guided search requires at least one non-empty bad-case slice")
    focus_well_set = set()
    for item in focus_slices.values():
        focus_well_set.update(str(well_id) for well_id in item.get("wells", []))
    context_pool = [well_id for well_id in all_wells if well_id not in focus_well_set]
    context_wells = _sample_pf_opt_wells(
        context_pool,
        sample_size,
        round_idx=1,
        seed_base=20260616,
    )
    if not context_wells:
        context_wells = _sample_pf_opt_wells(all_wells, sample_size, round_idx=1, seed_base=20260616)

    last_round = {"status": "analysis_ready", "sample_gain": math.nan, "full_gain": math.nan}
    round_idx = 1
    while max_rounds is None or round_idx <= int(max_rounds):
        gc.collect()
        PF_OPT_PROGRESS_PATH.read_text()
        round_label = f"fg{round_idx:04d}"
        space_note = _pf_failure_guided_search_space_note(last_round)
        baseline_ctx = benchmark_pf_opt_candidate(
            f"{round_label}_baseline_ctx",
            current_overrides,
            workers=workers,
            data_path=data_path,
            progress=True,
            well_ids=context_wells,
            n_particles=int(n_particles),
            n_seeds=int(n_seeds),
            enforce_budget=(int(n_particles), int(n_seeds)),
        )
        baseline_by_focus = {}

        def _baseline_for_focus(focus_name):
            if focus_name not in focus_slices:
                focus_name = "top_pf_failures"
            if focus_name not in baseline_by_focus:
                focus_wells = focus_slices[focus_name]["wells"]
                baseline_by_focus[focus_name] = benchmark_pf_opt_candidate(
                    f"{round_label}_baseline_{focus_name}",
                    current_overrides,
                    workers=workers,
                    data_path=data_path,
                    progress=True,
                    well_ids=focus_wells,
                    n_particles=int(n_particles),
                    n_seeds=int(n_seeds),
                    enforce_budget=(int(n_particles), int(n_seeds)),
                )
            return baseline_by_focus[focus_name]

        top_baseline = _baseline_for_focus("top_pf_failures")
        focus_names = list(focus_slices)
        candidates = [
            (f"{round_label}_{name}", _clean_pf_random_overrides({**current_overrides, **delta}))
            for name, delta, _ in PF_FAILURE_GUIDED_CURATED_CANDIDATES
        ]
        candidates.extend(
            _random_pf_failure_guided_candidates(
                round_idx,
                current_overrides,
                n_candidates=random_candidates,
                focus_names=focus_names,
            )
        )
        context_loss_limit = _failure_guided_context_loss_limit(gate_margin)
        _append_progress(
            f"Failure-guided round {round_idx} start: top_failure baseline `{top_baseline['rmse']:.6f}`, "
            f"context baseline `{baseline_ctx['rmse']:.6f}`, focus_slices `{_format_failure_focus_slices(focus_slices)}`, "
            f"context_loss_limit `{context_loss_limit:.6f}`, candidates={len(candidates)}. Space: {space_note}"
        )

        sample_results = []
        for cand_name, cand_overrides in candidates:
            focus_name = _failure_guided_candidate_focus(cand_name, cand_overrides, base_overrides=current_overrides)
            if focus_name not in focus_slices:
                focus_name = "top_pf_failures"
            focus_info = focus_slices[focus_name]
            baseline_focus = _baseline_for_focus(focus_name)
            cand_focus = benchmark_pf_opt_candidate(
                cand_name + "_" + focus_name,
                cand_overrides,
                workers=workers,
                data_path=data_path,
                progress=True,
                well_ids=focus_info["wells"],
                n_particles=int(n_particles),
                n_seeds=int(n_seeds),
                enforce_budget=(int(n_particles), int(n_seeds)),
            )
            cand_ctx = benchmark_pf_opt_candidate(
                cand_name + "_ctx",
                cand_overrides,
                workers=workers,
                data_path=data_path,
                progress=True,
                well_ids=context_wells,
                n_particles=int(n_particles),
                n_seeds=int(n_seeds),
                enforce_budget=(int(n_particles), int(n_seeds)),
            )
            focus_gate_ok, focus_metric_gains = _pf_opt_passes_metric_gate(
                baseline_focus,
                cand_focus,
                global_margin=float(gate_margin),
                aux_margin=0.0,
            )
            focus_gain = float(focus_metric_gains["global"])
            context_gain = float(baseline_ctx["rmse"] - cand_ctx["rmse"])
            combined_baseline = 0.70 * float(baseline_focus["rmse"]) + 0.30 * float(baseline_ctx["rmse"])
            combined_candidate = 0.70 * float(cand_focus["rmse"]) + 0.30 * float(cand_ctx["rmse"])
            cand_focus["focus_name"] = focus_name
            cand_focus["focus_reason"] = focus_info.get("reason", "")
            cand_focus["focus_well_count"] = len(focus_info.get("wells", []))
            cand_focus["focus_baseline_rmse"] = float(baseline_focus["rmse"])
            cand_focus["focus_metric_gains"] = focus_metric_gains
            cand_focus["focus_gain"] = focus_gain
            cand_focus["focus_gate_ok"] = bool(focus_gate_ok)
            cand_focus["context_rmse"] = float(cand_ctx["rmse"])
            cand_focus["context_baseline_rmse"] = float(baseline_ctx["rmse"])
            cand_focus["context_gain"] = context_gain
            cand_focus["combined_baseline_rmse"] = combined_baseline
            cand_focus["combined_rmse"] = combined_candidate
            cand_focus["combined_gain"] = combined_baseline - combined_candidate
            sample_results.append(cand_focus)

        best_sample, sample_gate_ok, passing_sample = _select_pf_failure_guided_best(
            sample_results,
            gate_margin=float(gate_margin),
            context_loss_limit=context_loss_limit,
        )
        best_overrides = _clean_pf_random_overrides(best_sample["overrides"])
        sample_gains = best_sample.get("focus_metric_gains", {})
        sample_gain = float(best_sample.get("focus_gain", math.nan))
        best_ctx = float(best_sample.get("context_rmse", math.nan))
        combined_candidate = float(best_sample.get("combined_rmse", math.nan))
        combined_gain = float(best_sample.get("combined_gain", math.nan))
        best_focus_name = str(best_sample.get("focus_name", "top_pf_failures"))
        best_focus_base = float(best_sample.get("focus_baseline_rmse", math.nan))
        top3 = _format_failure_guided_ranking(sample_results, limit=3)
        round_message = [
            f"Failure-guided round {round_idx} sample summary: best `{best_sample['name']}` "
            f"focus `{best_focus_name}` baseline `{best_focus_base:.6f}` -> `{best_sample['rmse']:.6f}`, "
            f"context `{baseline_ctx['rmse']:.6f}` -> `{best_ctx:.6f}`, "
            f"combined `{combined_candidate:.6f}`, combined_gain `{combined_gain:.6f}`, "
            f"direction `{_failure_guided_direction_note(best_sample)}`, "
            f"focus metric gains `{_format_pf_opt_metric_gains(sample_gains)}`, "
            f"passing={len(passing_sample)}, top3 `{top3}`.",
            f"Failure-guided round {round_idx} best overrides: `{json.dumps(best_overrides, sort_keys=True)}`.",
        ]

        confirm = None
        confirm_ok = False
        if sample_gate_ok:
            confirm = _confirm_pf_opt_candidate_on_second_shard(
                round_label,
                current_name,
                current_overrides,
                best_sample["name"],
                best_overrides,
                all_wells,
                sample_size,
                round_idx,
                workers,
                data_path,
                n_particles,
                n_seeds,
                gate_margin=float(gate_margin),
                seed_base=20260624,
                exclude_well_ids=list(focus_well_set) + list(context_wells),
            )
            confirm_ok = bool(confirm["ok"])
            round_message.append(
                f"Failure-guided round {round_idx} repeat-shard confirmation: baseline `{confirm['baseline']['rmse']:.6f}` "
                f"(well_mean `{confirm['baseline']['well_rmse_mean']:.6f}`, trim90 `{confirm['baseline']['well_rmse_trim90_mean']:.6f}`), "
                f"candidate `{confirm['candidate']['rmse']:.6f}` "
                f"(well_mean `{confirm['candidate']['well_rmse_mean']:.6f}`, trim90 `{confirm['candidate']['well_rmse_trim90_mean']:.6f}`), "
                f"wells `{confirm['well_count']}` overlap `{confirm['overlap_count']}`, "
                f"metric gains `{_format_pf_opt_metric_gains(confirm['gains'])}`, decision `{'passed' if confirm_ok else 'rejected'}`."
            )

        if sample_gate_ok and confirm_ok:
            full_check_count += 1
            full_check = _run_pf_opt_full_pair_check(
                f"{round_label}_check{full_check_count:03d}",
                current_name,
                current_overrides,
                best_sample["name"],
                best_overrides,
                full_wells,
                workers,
                data_path,
                full_check_idx=full_check_count,
                n_particles=int(full_check_n_particles),
                n_seeds=int(full_check_n_seeds),
                base_seed=int(full_check_base_seed),
            )
            sota_full = full_check["sota"]
            cand_full = full_check["candidate"]
            full_ok = bool(full_check["ok"])
            full_gains = full_check["gains"]
            full_gain = float(full_gains["global"])
            if full_ok:
                prev_name = current_name
                prev_full_rmse = sota_full["rmse"]
                current_name = best_sample["name"]
                current_overrides = best_overrides
                round_message.append(
                    f"Failure-guided round {round_idx} paired full check ACCEPTED: `{prev_name}` "
                    f"`{prev_full_rmse:.6f}` -> `{cand_full['name']}` `{cand_full['rmse']:.6f}`, "
                    f"base_seed `{full_check['base_seed']}`, "
                    f"metric gains `{_format_pf_opt_metric_gains(full_gains)}`."
                )
                _update_active_best_progress(
                    f"- Active SOTA: `{current_name}` promoted by failure-guided paired full check at "
                    f"`{int(full_check_n_particles)} x {int(full_check_n_seeds)}`, "
                    f"base_seed `{full_check['base_seed']}`, direct RMSE `{cand_full['rmse']:.6f}`, "
                    f"FFBSi RMSE `{cand_full.get('rmse_ffbsi', math.nan):.6f}`.\n"
                    f"- Previous SOTA in same check: `{prev_name}` direct RMSE `{sota_full['rmse']:.6f}`.\n"
                    f"- Active overrides: `{json.dumps(current_overrides, sort_keys=True)}`."
                )
                last_round = {"status": "full_accepted", "sample_gain": sample_gain, "full_gain": full_gain}
            else:
                round_message.append(
                    f"Failure-guided round {round_idx} paired full check rejected: SOTA `{current_name}` `{sota_full['rmse']:.6f}` "
                    f"(well_mean `{sota_full['well_rmse_mean']:.6f}`, trim90 `{sota_full['well_rmse_trim90_mean']:.6f}`), "
                    f"candidate `{cand_full['rmse']:.6f}` "
                    f"(well_mean `{cand_full['well_rmse_mean']:.6f}`, trim90 `{cand_full['well_rmse_trim90_mean']:.6f}`), "
                    f"base_seed `{full_check['base_seed']}`, "
                    f"metric gains `{_format_pf_opt_metric_gains(full_gains)}`."
                )
                last_round = {"status": "full_rejected", "sample_gain": sample_gain, "full_gain": full_gain}
        elif sample_gate_ok:
            round_message.append(
                f"Failure-guided round {round_idx} no full validation: directed gate passed, but repeat-shard confirmation "
                f"did not meet global gain >= `{0.5 * float(gate_margin):.6f}` with nonnegative well_mean and well_trim90."
            )
            last_round = {
                "status": "confirm_rejected",
                "sample_gain": sample_gain,
                "confirm_gain": float(confirm["gains"]["global"]) if confirm else math.nan,
                "full_gain": math.nan,
            }
        else:
            status = "directed_rejected" if bool(best_sample.get("focus_gate_ok", False)) else "no_sample_gate"
            round_message.append(
                f"Failure-guided round {round_idx} no full validation: no candidate cleared the directed gate "
                f"(focus gain >= `{float(gate_margin):.6f}`, positive well metrics, context loss <= "
                f"`{context_loss_limit:.6f}`, and positive combined gain). Best focus gains "
                f"`{_format_pf_opt_metric_gains(sample_gains)}`; context gain "
                f"`{float(best_sample.get('context_gain', math.nan)):.6f}`; combined gain `{combined_gain:.6f}`."
            )
            last_round = {"status": status, "sample_gain": sample_gain, "full_gain": math.nan}

        _append_progress("\n".join(round_message))
        gc.collect()
        round_idx += 1

    _append_progress("Failure-guided PF search loop stopped by max_rounds.")


def run_pf_profile_mixture_search_loop(
    sample_size=100,
    full_well_limit=None,
    workers=8,
    data_path=None,
    max_rounds=None,
    random_candidates=32,
    gate_margin=0.01,
    n_particles=500,
    n_seeds=32,
    full_check_n_particles=PF_OPT_FULL_CHECK_DEFAULT_PARTICLES,
    full_check_n_seeds=PF_OPT_FULL_CHECK_DEFAULT_SEEDS,
    full_check_base_seed=PF_OPT_FULL_CHECK_DEFAULT_BASE_SEED,
):
    PF_OPT_PROGRESS_PATH.read_text()
    _append_progress(
        "Profile-mixture search loop start: "
        f"budget `{int(n_particles)} x {int(n_seeds)}`, sample_wells={int(sample_size)}, "
        f"full_check_budget `{int(full_check_n_particles)} x {int(full_check_n_seeds)}`, "
        f"full_check_base_seed={int(full_check_base_seed)}, "
        f"curated_candidates={len(PF_OPT_PROFILE_CANDIDATES)}, random_candidates={int(random_candidates)}, "
        f"gate_margin={float(gate_margin):.4f} on global RMSE, with positive well_mean/well_trim90 gains, "
        f"`global/well_mean/well_trim90`, workers={int(workers)}, "
        f"full_well_limit={full_well_limit or 'all'}."
    )
    current_name = PF_OPT_CURRENT_BEST_NAME
    current_overrides = _clean_pf_random_overrides(PF_OPT_CURRENT_BEST_OVERRIDES)
    full_check_count = 0
    cfg = _make_pf_opt_cfg(
        data_path=data_path,
        overrides=current_overrides,
        n_particles=int(n_particles),
        n_seeds=int(n_seeds),
        workers=int(workers),
    )
    data_path = Path(data_path or cfg.train_path)
    all_wells = _discover_well_ids(data_path)
    if not all_wells:
        raise ValueError(f"no train wells found under {data_path}")
    full_wells = all_wells if full_well_limit is None else all_wells[: int(full_well_limit)]
    last_round = None
    round_idx = 1

    while max_rounds is None or round_idx <= int(max_rounds):
        gc.collect()
        PF_OPT_PROGRESS_PATH.read_text()
        sampled_wells = _sample_pf_opt_wells(all_wells, sample_size, round_idx, seed_base=20260610)
        round_label = f"pm{round_idx:04d}"
        space_note = _pf_profile_search_space_note(last_round)
        baseline = benchmark_pf_opt_candidate(
            f"{round_label}_baseline",
            current_overrides,
            workers=workers,
            data_path=data_path,
            progress=True,
            well_ids=sampled_wells,
            n_particles=int(n_particles),
            n_seeds=int(n_seeds),
            enforce_budget=(int(n_particles), int(n_seeds)),
        )
        curated = [
            (f"{round_label}_{name}", _clean_pf_random_overrides({**current_overrides, **delta}))
            for name, delta in PF_OPT_PROFILE_CANDIDATES
        ]
        random_mix = _random_profile_mixture_candidates(
            round_idx,
            current_overrides,
            n_candidates=random_candidates,
        )
        candidates = curated + random_mix
        _append_progress(
            f"Profile round {round_idx} start: baseline `{current_name}` sample RMSE `{baseline['rmse']:.6f}` "
            f"on {len(sampled_wells)} random wells, candidates={len(candidates)}. Space: {space_note}"
        )

        sample_results = []
        for cand_name, cand_overrides in candidates:
            cand = benchmark_pf_opt_candidate(
                cand_name,
                cand_overrides,
                workers=workers,
                data_path=data_path,
                progress=True,
                well_ids=sampled_wells,
                n_particles=int(n_particles),
                n_seeds=int(n_seeds),
                enforce_budget=(int(n_particles), int(n_seeds)),
            )
            sample_results.append(cand)

        best_sample, sample_gains, sample_gate_ok, passing_sample = _select_pf_opt_gated_best(
            baseline,
            sample_results,
            global_margin=float(gate_margin),
            aux_margin=0.0,
        )
        sample_gain = float(sample_gains["global"])
        top3 = _format_brief_candidate_ranking(sample_results, limit=3)
        best_overrides = _clean_pf_random_overrides(best_sample["overrides"])
        profile_count = len(best_overrides.get("PF_heatmap_profile_mixture_spec", []))
        round_message = [
            f"Profile round {round_idx} sample summary: baseline `{baseline['rmse']:.6f}` "
            f"(well_mean `{baseline['well_rmse_mean']:.6f}`, trim90 `{baseline['well_rmse_trim90_mean']:.6f}`), "
            f"best gated `{best_sample['name']}` `{best_sample['rmse']:.6f}`, "
            f"metric gains `{_format_pf_opt_metric_gains(sample_gains)}`, "
            f"passing={len(passing_sample)}, profiles={profile_count}, top3 `{top3}`.",
            f"Profile round {round_idx} best overrides: `{json.dumps(best_overrides, sort_keys=True)}`.",
        ]

        if sample_gate_ok:
            full_check_count += 1
            full_check = _run_pf_opt_full_pair_check(
                f"{round_label}_check{full_check_count:03d}",
                current_name,
                current_overrides,
                best_sample["name"],
                best_overrides,
                full_wells,
                workers,
                data_path,
                full_check_idx=full_check_count,
                n_particles=int(full_check_n_particles),
                n_seeds=int(full_check_n_seeds),
                base_seed=int(full_check_base_seed),
            )
            sota_full = full_check["sota"]
            cand_full = full_check["candidate"]
            full_ok = bool(full_check["ok"])
            full_gains = full_check["gains"]
            full_gain = float(full_gains["global"])
            if full_ok:
                prev_name = current_name
                prev_full_rmse = sota_full["rmse"]
                current_name = best_sample["name"]
                current_overrides = best_overrides
                round_message.append(
                    f"Profile round {round_idx} paired full check ACCEPTED: `{prev_name}` "
                    f"`{prev_full_rmse:.6f}` -> `{cand_full['name']}` `{cand_full['rmse']:.6f}`, "
                    f"base_seed `{full_check['base_seed']}`, "
                    f"metric gains `{_format_pf_opt_metric_gains(full_gains)}`."
                )
                _update_active_best_progress(
                    f"- Active SOTA: `{current_name}` promoted by profile paired full check at "
                    f"`{int(full_check_n_particles)} x {int(full_check_n_seeds)}`, "
                    f"base_seed `{full_check['base_seed']}`, direct RMSE `{cand_full['rmse']:.6f}`, "
                    f"FFBSi RMSE `{cand_full.get('rmse_ffbsi', math.nan):.6f}`.\n"
                    f"- Previous SOTA in same check: `{prev_name}` direct RMSE `{sota_full['rmse']:.6f}`.\n"
                    f"- Active overrides: `{json.dumps(current_overrides, sort_keys=True)}`."
                )
                last_round = {"status": "full_accepted", "sample_gain": sample_gain, "full_gain": full_gain}
            else:
                round_message.append(
                    f"Profile round {round_idx} paired full check rejected: SOTA `{current_name}` `{sota_full['rmse']:.6f}` "
                    f"(well_mean `{sota_full['well_rmse_mean']:.6f}`, trim90 `{sota_full['well_rmse_trim90_mean']:.6f}`), "
                    f"candidate `{cand_full['rmse']:.6f}` "
                    f"(well_mean `{cand_full['well_rmse_mean']:.6f}`, trim90 `{cand_full['well_rmse_trim90_mean']:.6f}`), "
                    f"base_seed `{full_check['base_seed']}`, "
                    f"metric gains `{_format_pf_opt_metric_gains(full_gains)}`."
                )
                last_round = {"status": "full_rejected", "sample_gain": sample_gain, "full_gain": full_gain}
        else:
            round_message.append(
                f"Profile round {round_idx} no full validation: no candidate cleared global gain "
                f"`{float(gate_margin):.6f}` while also improving well_mean and well_trim90. "
                f"Best global candidate metric gains "
                f"`{_format_pf_opt_metric_gains(sample_gains)}`."
            )
            last_round = {"status": "no_sample_gate", "sample_gain": sample_gain, "full_gain": math.nan}

        _append_progress("\n".join(round_message))
        gc.collect()
        round_idx += 1

    _append_progress("Profile-mixture search loop stopped by max_rounds.")


def _parse_json_overrides(raw):
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--overrides-json must decode to a JSON object")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Experimental PF optimizer for the copied seq_NN_data_prep module.")
    parser.add_argument("--benchmark", action="store_true", help="run one in-memory benchmark candidate")
    parser.add_argument("--compare-pasted", action="store_true", help="compare pasted v12 PF against default numba PF")
    parser.add_argument("--opt-loop", action="store_true", help="run the endless first-50/full-batch optimization loop")
    parser.add_argument("--design-loop", action="store_true", help="run curated design-choice candidates from current best")
    parser.add_argument("--random-search-loop", action="store_true", help="run PF random search from the active search-space preset")
    parser.add_argument(
        "--search-space",
        default="broad",
        choices=("broad", "merge16", "merge64"),
        help="random-search preset: broad, merge16, or merge64",
    )
    parser.add_argument("--profile-mixture-search-loop", action="store_true", help="run scalable PF profile-mixture search from current best")
    parser.add_argument("--profile-comb-benchmark", action="store_true", help="benchmark half CFG-default and half SOTA PF profiles")
    parser.add_argument("--failure-guided-search-loop", action="store_true", help="run PF search on cached PF failures plus context wells")
    parser.add_argument("--ffbsi-feature-test", action="store_true", help="test FFBSi heatmap mode on a fixed 100-well sample")
    parser.add_argument("--merge64-local-seed-blend-test", action="store_true", help="test local/decayed seed blending on current merge64 SOTA")
    parser.add_argument("--particle-budget-research", action="store_true", help="run the fixed-K and fixed-budget particle-number sweep")
    parser.add_argument("--particle-budget-full-grid-search", action="store_true", help="run the full-well fixed-budget N-aware particle-number grid")
    parser.add_argument("--cached-pf-failure-analysis", action="store_true", help="analyze existing PF cache bad cases and reliability rules")
    parser.add_argument("--name", default="manual", help="candidate name for --benchmark")
    parser.add_argument("--overrides-json", default="", help="JSON object of PF_heatmap_* overrides for --benchmark")
    parser.add_argument("--data-path", default=None, help="train directory path; defaults to CFG.train_path")
    parser.add_argument("--pf-cache-dir", default=None, help="PF cache split directory for --cached-pf-failure-analysis")
    parser.add_argument("--output-dir", default=None, help="output directory for analysis artifacts")
    parser.add_argument("--well-limit", type=int, default=50, help="first-stage well count")
    parser.add_argument("--full-well-limit", type=int, default=None, help="full-stage well count; use 773 for all train wells")
    parser.add_argument("--workers", type=int, default=8, help="multiprocessing workers")
    parser.add_argument("--max-rounds", type=int, default=None, help="stop loop after this many rounds; default is endless")
    parser.add_argument("--random-candidates", type=int, default=32, help="random candidates per random-search round")
    parser.add_argument("--gate-margin", type=float, default=0.01, help="sample RMSE improvement needed before full validation")
    parser.add_argument("--n-particles", type=int, default=500, help="PF particles for --random-search-loop")
    parser.add_argument("--n-seeds", type=int, default=32, help="PF seeds for --random-search-loop")
    parser.add_argument(
        "--full-check-n-particles",
        type=int,
        default=PF_OPT_FULL_CHECK_DEFAULT_PARTICLES,
        help="PF particles for paired full validation checks",
    )
    parser.add_argument(
        "--full-check-n-seeds",
        type=int,
        default=PF_OPT_FULL_CHECK_DEFAULT_SEEDS,
        help="PF seeds for paired full validation checks",
    )
    parser.add_argument(
        "--full-check-base-seed",
        type=int,
        default=PF_OPT_FULL_CHECK_DEFAULT_BASE_SEED,
        help="base seed for paired full validation checks; each full check uses a deterministic offset",
    )
    parser.add_argument("--ffbsi-paths", type=int, default=64, help="FFBSi backward sampled paths per PF seed")
    parser.add_argument("--particle-budget-total-particles", type=int, default=500, help="total particle budget numerator for full-grid search")
    parser.add_argument("--particle-budget-total-seeds", type=int, default=128, help="total particle budget denominator for full-grid search")
    parser.add_argument("--particle-budget-n-grid", default="1000,2000,4000", help="comma-separated N grid for full-grid search")
    parser.add_argument(
        "--particle-budget-base-seed",
        type=int,
        default=PF_OPT_CURRENT_BEST_FULL_BASE_SEED,
        help="baseline base seed for the full-grid search baseline run",
    )
    args = parser.parse_args()

    if args.random_search_loop and args.search_space == "merge16":
        if args.well_limit == 50:
            args.well_limit = 200
        if args.full_well_limit is None:
            args.full_well_limit = 773
        if args.n_seeds == 32:
            args.n_seeds = 128
        if args.random_candidates == 32:
            args.random_candidates = 16
        if args.full_check_base_seed == PF_OPT_FULL_CHECK_DEFAULT_BASE_SEED:
            args.full_check_base_seed = PF_OPT_MERGE16_CURRENT_BEST_FULL_BASE_SEED
    if args.random_search_loop and args.search_space == "merge64":
        if args.well_limit == 50:
            args.well_limit = 200
        if args.full_well_limit is None:
            args.full_well_limit = 773
        if args.n_seeds == 32:
            args.n_seeds = 128
        if args.random_candidates == 32:
            args.random_candidates = 16
        if args.full_check_base_seed == PF_OPT_FULL_CHECK_DEFAULT_BASE_SEED:
            args.full_check_base_seed = PF_OPT_MERGE64_CURRENT_BEST_FULL_BASE_SEED

    if args.benchmark:
        summary = benchmark_pf_opt_candidate(
            args.name,
            _parse_json_overrides(args.overrides_json),
            well_limit=args.well_limit,
            workers=args.workers,
            data_path=args.data_path,
            n_particles=args.n_particles,
            n_seeds=args.n_seeds,
            enforce_budget=(args.n_particles, args.n_seeds),
        )
        _append_progress("Manual benchmark result: " + _format_pf_opt_summary(summary))
    elif args.compare_pasted:
        result = benchmark_pf_pasted_vs_numba(
            well_limit=args.well_limit,
            workers=args.workers,
            data_path=args.data_path,
        )
        _append_progress(_format_pf_compare_for_progress(result))
    elif args.opt_loop:
        run_pf_opt_loop(
            well_limit=args.well_limit,
            full_well_limit=args.full_well_limit,
            workers=args.workers,
            data_path=args.data_path,
            max_rounds=args.max_rounds,
            random_candidates=args.random_candidates,
        )
    elif args.design_loop:
        run_pf_design_loop(
            well_limit=args.well_limit,
            full_well_limit=args.full_well_limit if args.full_well_limit is not None else 773,
            workers=args.workers,
            data_path=args.data_path,
            max_rounds=args.max_rounds,
        )
    elif args.random_search_loop:
        run_pf_random_search_loop(
            sample_size=args.well_limit,
            full_well_limit=args.full_well_limit,
            workers=args.workers,
            data_path=args.data_path,
            max_rounds=args.max_rounds,
            random_candidates=args.random_candidates,
            gate_margin=args.gate_margin,
            n_particles=args.n_particles,
            n_seeds=args.n_seeds,
            full_check_n_particles=args.full_check_n_particles,
            full_check_n_seeds=args.full_check_n_seeds,
            full_check_base_seed=args.full_check_base_seed,
            search_space=args.search_space,
        )
    elif args.profile_mixture_search_loop:
        run_pf_profile_mixture_search_loop(
            sample_size=args.well_limit,
            full_well_limit=args.full_well_limit,
            workers=args.workers,
            data_path=args.data_path,
            max_rounds=args.max_rounds,
            random_candidates=args.random_candidates,
            gate_margin=args.gate_margin,
            n_particles=args.n_particles,
            n_seeds=args.n_seeds,
            full_check_n_particles=args.full_check_n_particles,
            full_check_n_seeds=args.full_check_n_seeds,
            full_check_base_seed=args.full_check_base_seed,
        )
    elif args.profile_comb_benchmark:
        summary = benchmark_pf_opt_candidate(
            args.name if args.name != "manual" else "profile_comb_cfg_default_sota",
            PF_OPT_PROFILE_COMB_OVERRIDES,
            well_limit=args.well_limit,
            workers=args.workers,
            data_path=args.data_path,
            n_particles=args.n_particles,
            n_seeds=args.n_seeds,
            enforce_budget=(args.n_particles, args.n_seeds),
        )
        _append_progress("Profile-comb benchmark result: " + _format_pf_opt_summary(summary))
    elif args.failure_guided_search_loop:
        run_pf_failure_guided_search_loop(
            sample_size=args.well_limit,
            full_well_limit=args.full_well_limit,
            workers=args.workers,
            data_path=args.data_path,
            cache_dir=args.pf_cache_dir,
            output_dir=args.output_dir,
            max_rounds=args.max_rounds,
            random_candidates=args.random_candidates,
            gate_margin=args.gate_margin,
            n_particles=args.n_particles,
            n_seeds=args.n_seeds,
            full_check_n_particles=args.full_check_n_particles,
            full_check_n_seeds=args.full_check_n_seeds,
            full_check_base_seed=args.full_check_base_seed,
        )
    elif args.ffbsi_feature_test:
        run_pf_ffbsi_feature_test(
            sample_size=args.well_limit,
            workers=args.workers,
            data_path=args.data_path,
            n_particles=args.n_particles,
            n_seeds=args.n_seeds,
            n_paths=args.ffbsi_paths,
        )
    elif args.merge64_local_seed_blend_test:
        run_pf_merge64_local_seed_blend_test(
            sample_size=args.well_limit,
            workers=args.workers,
            data_path=args.data_path,
            n_particles=args.n_particles,
            n_seeds=args.n_seeds,
        )
    elif args.particle_budget_research:
        run_pf_particle_budget_research(
            well_limit=args.well_limit,
            workers=args.workers,
            data_path=args.data_path,
            output_dir=args.output_dir,
        )
    elif args.particle_budget_full_grid_search:
        n_grid = tuple(int(x) for x in str(args.particle_budget_n_grid).split(",") if x.strip())
        run_pf_particle_budget_full_grid_search(
            workers=int(args.workers),
            data_path=args.data_path,
            full_well_limit=args.full_well_limit if args.full_well_limit is not None else 773,
            total_particles=args.particle_budget_total_particles,
            total_seeds=args.particle_budget_total_seeds,
            n_grid=n_grid,
            output_dir=args.output_dir,
            baseline_base_seed=args.particle_budget_base_seed,
        )
    elif args.cached_pf_failure_analysis:
        run_cached_pf_failure_analysis(
            cache_dir=args.pf_cache_dir,
            data_path=args.data_path,
            output_dir=args.output_dir,
            top_k=args.well_limit,
            fold_count=5,
            seed=7,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
