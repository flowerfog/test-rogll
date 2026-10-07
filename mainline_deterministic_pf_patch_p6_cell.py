"""Deterministic single-PF patch for rogii-cp8-neighbor.ipynb.

Run after the PF definitions (notebook cell 41) and before ``main()``.  The
128-seed lik-PF is already deterministic; this patch controls only ANCC/Z PF.
"""

import zlib

import numpy as np
import pandas as pd
from numba import njit


required_p6 = {
    "_pf_ancc", "_pf_z", "_gr_sig", "_grid",
    "ANCC_N", "PF_N", "ANCC_ALPHA", "ANCC_RN", "ANCC_PN",
    "ANCC_IS", "ANCC_RP", "ANCC_RR", "PF_RESAMP", "PF_GR_WIN",
    "PF_MOM", "PF_VN", "PF_PN", "PF_GR_WT", "PF_ROUGH_P", "PF_ROUGH_V",
}
missing_p6 = sorted(required_p6.difference(globals()))
if missing_p6:
    raise RuntimeError("Run the mainline PF definition cells first: " + repr(missing_p6))


def _well_seed_p6(hw, stream):
    md = hw["MD"].to_numpy(np.float64)
    z = hw["Z"].to_numpy(np.float64)
    known = hw["TVT_input"].dropna().to_numpy(np.float64)
    payload = (
        f"{len(hw)}|{md[0]:.6f}|{md[-1]:.6f}|"
        f"{z[0]:.6f}|{z[-1]:.6f}|{known[-1] if len(known) else 0.0:.6f}|{stream}"
    )
    return int(zlib.crc32(payload.encode("ascii")) & 0x7FFFFFFF)


@njit(cache=False)
def _pf_ancc_seeded_p6(seed, md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, n_particles,
                        alpha, rate_noise, position_noise, init_spread,
                        rough_position, rough_rate, resample_fraction):
    np.random.seed(seed)
    return _pf_ancc(
        md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, n_particles,
        alpha, rate_noise, position_noise, init_spread,
        rough_position, rough_rate, resample_fraction,
    )


@njit(cache=False)
def _pf_z_seeded_p6(seed, md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step,
                    gs, initial_position, initial_velocity, beta, intercept,
                    z_sigma, n_particles, momentum, velocity_noise,
                    position_noise, gr_weight, rough_position, rough_velocity,
                    resample_fraction):
    np.random.seed(seed)
    return _pf_z(
        md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step,
        gs, initial_position, initial_velocity, beta, intercept,
        z_sigma, n_particles, momentum, velocity_noise, position_noise,
        gr_weight, rough_position, rough_velocity, resample_fraction,
    )


def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    known = hw[hw["TVT_input"].notna()]
    evaluation = hw[hw["TVT_input"].isna()]
    if len(evaluation) == 0:
        return np.array([]), np.array([])
    last_surface = float(known["TVT_input"].iloc[-1] + known["Z"].iloc[-1])
    tail = known.tail(30)
    delta_tvt = np.diff(tail["TVT_input"].to_numpy())
    delta_z = np.diff(tail["Z"].to_numpy())
    delta_md = np.diff(tail["MD"].to_numpy())
    valid = delta_md > 0
    initial_rate = float(np.median((delta_tvt + delta_z)[valid] / delta_md[valid])) if valid.sum() >= 3 else 0.0
    gr_grid, grid_min, grid_step = _grid(tw_tvt, tw_gr)
    prediction, spread = _pf_ancc_seeded_p6(
        _well_seed_p6(hw, "ancc"),
        evaluation["MD"].to_numpy(np.float64),
        evaluation["Z"].to_numpy(np.float64),
        evaluation["GR"].to_numpy(np.float64),
        gr_grid, grid_min, grid_step, gs, last_surface, initial_rate, N,
        ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP,
    )
    return prediction.astype(np.float32), spread.astype(np.float32)


def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    tw_smoothed = (
        pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().to_numpy(np.float32)
    )
    known = hw[hw["TVT_input"].notna()]
    evaluation = hw[hw["TVT_input"].isna()]
    if len(evaluation) == 0:
        return np.array([]), np.array([])

    delta_z = np.diff(known["Z"].to_numpy())
    delta_tvt = np.diff(known["TVT_input"].to_numpy())
    delta_md = np.diff(known["MD"].to_numpy())
    valid = delta_md > 0
    if valid.sum() >= 10:
        z_rate = delta_z[valid] / delta_md[valid]
        tvt_rate = delta_tvt[valid] / delta_md[valid]
        design = np.column_stack([z_rate, np.ones_like(z_rate)])
        coefficients, _, _, _ = np.linalg.lstsq(design, tvt_rate, rcond=None)
        beta, intercept = float(coefficients[0]), float(coefficients[1])
        z_sigma = max(float(np.std(tvt_rate - (beta * z_rate + intercept))), 0.001)
    else:
        beta, intercept, z_sigma = -1.0, 0.0, 0.1

    tail = known.tail(20)
    tail_delta_tvt = np.diff(tail["TVT_input"].to_numpy())
    tail_delta_md = np.diff(tail["MD"].to_numpy())
    tail_valid = tail_delta_md > 0
    initial_velocity = float(np.median(tail_delta_tvt[tail_valid] / tail_delta_md[tail_valid])) if tail_valid.sum() >= 3 else 0.0
    gr_grid, grid_min, grid_step = _grid(tw_tvt, tw_gr)
    smooth_grid, _, _ = _grid(tw_tvt, tw_smoothed)
    horizontal_smoothed = hw["GR"].rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    prediction, spread = _pf_z_seeded_p6(
        _well_seed_p6(hw, "z"),
        evaluation["MD"].to_numpy(np.float64),
        evaluation["Z"].to_numpy(np.float64),
        evaluation["GR"].to_numpy(np.float64),
        horizontal_smoothed.loc[evaluation.index].to_numpy(np.float64),
        gr_grid, smooth_grid, grid_min, grid_step, gs,
        float(known["TVT_input"].iloc[-1]), initial_velocity,
        beta, intercept, z_sigma, N,
        PF_MOM, PF_VN, PF_PN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP,
    )
    return prediction.astype(np.float32), spread.astype(np.float32)


print("P6 deterministic ANCC/Z PF patch installed; lik-PF remains unchanged", flush=True)
