"""Two notebook cells for the generic 128-seed PF branch hedge.

CELL A replaces the existing likelihood-PF call inside the per-well inference
loop. CELL B runs last, after Gold/Neighbor has written submission.csv.

Do not include the A24 target-well continuation. It is tied to the three-well
example artifact and is not a hidden-test inference rule.
"""


# %% CELL A1: define before the per-well test inference loop
import numpy as np
import pandas as pd


PF_SEED_BRANCH_STATS = {}


def _pf_branch_statistics(pred_arr, log_likelihoods, eval_mask):
    """Reproduce the source notebook's weighted two-branch summary."""
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    log_likelihoods = np.asarray(log_likelihoods, dtype=np.float64)
    eval_mask = np.asarray(eval_mask, dtype=bool)

    if pred_arr.ndim != 2 or pred_arr.shape[0] != len(log_likelihoods):
        raise RuntimeError("PF seed trajectory/likelihood shape mismatch")
    if pred_arr.shape[1] != len(eval_mask):
        raise RuntimeError("PF seed trajectory/horizontal-row shape mismatch")
    if int(eval_mask.sum()) < 10:
        raise RuntimeError("too few hidden rows for branch statistics")

    centred = log_likelihoods - np.nanmax(log_likelihoods)
    seed_weight = np.exp(centred / 5.0)
    seed_weight /= max(float(seed_weight.sum()), 1e-12)

    # One datum level per complete seed trajectory. This is deliberately not a
    # row-wise clustering operation.
    level = np.nanmedian(pred_arr[:, eval_mask], axis=1)
    valid = np.isfinite(level) & np.isfinite(seed_weight) & (seed_weight > 0.0)
    level = level[valid]
    seed_weight = seed_weight[valid]
    if len(level) < 4:
        raise RuntimeError("too few finite PF seeds for branch statistics")
    seed_weight /= max(float(seed_weight.sum()), 1e-12)

    order = np.argsort(level)
    x = level[order]
    w = seed_weight[order]
    cw = np.cumsum(w)
    cx = np.cumsum(w * x)
    cx2 = np.cumsum(w * x * x)
    total_w = float(cw[-1])
    total_x = float(cx[-1])
    total_x2 = float(cx2[-1])

    best = None
    for cut in range(1, len(x)):
        mass_low = float(cw[cut - 1])
        mass_high = total_w - mass_low
        if mass_low < 0.05 or mass_high < 0.05:
            continue
        sum_low = float(cx[cut - 1])
        sum_high = total_x - sum_low
        sse_low = float(cx2[cut - 1] - sum_low * sum_low / mass_low)
        sse_high = float(
            total_x2 - cx2[cut - 1] - sum_high * sum_high / mass_high
        )
        score = max(0.0, sse_low) + max(0.0, sse_high)
        candidate = (
            score,
            mass_low,
            mass_high,
            sum_low / mass_low,
            sum_high / mass_high,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise RuntimeError("no valid weighted branch split")

    _, mass_low, mass_high, center_low, center_high = best
    return {
        "center_low": float(center_low),
        "center_high": float(center_high),
        "mass_low": float(mass_low),
        "mass_high": float(mass_high),
        "weighted_center": float(np.sum(seed_weight * level)),
        "eval_rows": np.flatnonzero(eval_mask).astype(int).tolist(),
        "seed_count": int(len(level)),
        "loglik_max": float(np.nanmax(log_likelihoods)),
        "loglik_std": float(np.nanstd(log_likelihoods)),
    }


def run_pf_lik_ensemble_scales_with_branch(
    hw,
    tw,
    scales=SELECTOR_SCALES,
    n_particles=500,
    n_seeds=128,
):
    """Run the existing PF once and retain the seed-level branch evidence."""
    predictions = []
    log_likelihoods = []
    for seed in range(int(n_seeds)):
        prediction, log_likelihood = run_particle_filter(
            hw,
            tw,
            n_particles=int(n_particles),
            seed=int(seed),
        )
        predictions.append(np.asarray(prediction, dtype=np.float64))
        log_likelihoods.append(float(log_likelihood))

    pred_arr = np.stack(predictions, axis=0)
    log_likelihoods = np.asarray(log_likelihoods, dtype=np.float64)
    centred = log_likelihoods - np.nanmax(log_likelihoods)

    output = {}
    for scale in scales:
        weight = np.exp(centred / float(scale))
        weight /= max(float(weight.sum()), 1e-12)
        output[f"pf_scale_{float(scale):g}"] = np.sum(
            weight[:, None] * pred_arr,
            axis=0,
        )
    output["pf_mean"] = np.mean(pred_arr, axis=0)
    if bool(globals().get("SELECTOR_PF_RETURN_STD", False)):
        output["pf_seed_std"] = np.std(pred_arr, axis=0)

    eval_mask = pd.to_numeric(hw["TVT_input"], errors="coerce").isna().to_numpy()
    branch = _pf_branch_statistics(pred_arr, log_likelihoods, eval_mask)
    return output, branch


# %% CELL A2: replace the PF + Beam + selector section inside the test-well loop
#
# IMPORTANT: keep the indentation below. Replace the section beginning at
# selector_well_code(...) and ending at the selector status print. Do not run
# the old PF call as well, or PF runtime doubles.
"""
    selector_code, selector_variant, selector_n_eval, selector_z_span = selector_well_code(hw_te)
    tw_ref = tw_tr if tw_tr is not None else tw_te

    # 128-seed likelihood-weighted PF ensemble, retaining branch statistics.
    try:
        pf_by_scale, _seed_branch = run_pf_lik_ensemble_scales_with_branch(
            hw_te,
            tw_ref,
            scales=SELECTOR_SCALES,
            n_particles=int(globals().get("SP45_SELECTOR_N_PARTICLES", 500)),
            n_seeds=128,
        )
        PF_SEED_BRANCH_STATS[str(wid)] = _seed_branch
        tvt_pf = pf_by_scale["pf_scale_8"]
        print(f"  PF 128-seed lik-ensemble OK scales={SELECTOR_SCALES}")
    except Exception as exc:
        print(f"  PF failed: {exc}")
        PF_SEED_BRANCH_STATS.pop(str(wid), None)
        last_known = hw_te["TVT_input"].dropna()
        last_val = float(last_known.iloc[-1]) if len(last_known) else 0.0
        tvt_pf = hw_te["TVT_input"].fillna(last_val).to_numpy(dtype=float)
        pf_by_scale = {
            f"pf_scale_{float(scale):g}": tvt_pf.copy()
            for scale in SELECTOR_SCALES
        }

    # Beam must be assigned independently of PF success.
    try:
        tvt_beam = run_beam_ensemble(hw_te, tw_ref)
        print("  Beam 14-config ensemble OK")
    except Exception as exc:
        print(f"  Beam failed: {exc}")
        tvt_beam = tvt_pf.copy()

    last_known = hw_te["TVT_input"].dropna()
    last_known_tvt = (
        float(last_known.iloc[-1])
        if len(last_known)
        else float(np.nanmean(tvt_pf))
    )
    tvt_selector = apply_selector_variant(
        selector_variant,
        pf_by_scale,
        tvt_beam,
        last_known_tvt,
    )
    print(
        f"  Selector code={selector_code} variant={selector_variant} "
        f"n_eval={selector_n_eval:.0f} z_span={selector_z_span:.3f}"
    )
"""


# %% CELL B: run last, after Neighbor writes /kaggle/working/submission.csv
from pathlib import Path

import numpy as np
import pandas as pd


BRANCH_STRENGTH = 0.60
BRANCH_MIN_MASS = 0.25
BRANCH_SEPARATION_LOW_FT = 4.0
BRANCH_SEPARATION_HIGH_FT = 40.0
BRANCH_SHIFT_CAP_FT = 2.0
BRANCH_MIN_STAT_COVERAGE = 0.98

work = Path("/kaggle/working")
submission_path = work / "submission.csv"
if not submission_path.exists():
    raise FileNotFoundError(submission_path)

submission = pd.read_csv(submission_path, dtype={"id": "string"})
if list(submission.columns) != ["id", "tvt"]:
    raise RuntimeError(
        f"PF branch hedge expected id,tvt columns, got {list(submission.columns)}"
    )
submission["tvt"] = pd.to_numeric(submission["tvt"], errors="coerce")
if not np.isfinite(submission["tvt"].to_numpy(float)).all():
    raise RuntimeError("non-finite base submission")

submission.to_csv(work / "submission_before_pf_seed_branch.csv", index=False)
row_well = submission["id"].str.rsplit("_", n=1).str[0]
row_index = pd.to_numeric(
    submission["id"].str.rsplit("_", n=1).str[-1],
    errors="coerce",
)
if row_index.isna().any():
    raise RuntimeError("cannot parse row index from submission ids")

expected_wells = sorted(row_well.unique().tolist())
statistics = globals().get("PF_SEED_BRANCH_STATS")
if not isinstance(statistics, dict):
    raise RuntimeError("PF_SEED_BRANCH_STATS is missing or is not a dict")

required_keys = {
    "center_low",
    "center_high",
    "mass_low",
    "mass_high",
    "weighted_center",
    "eval_rows",
}
valid_statistics = {
    str(well): payload
    for well, payload in statistics.items()
    if isinstance(payload, dict) and required_keys.issubset(payload)
}
coverage = len(set(expected_wells) & set(valid_statistics)) / max(len(expected_wells), 1)
print(
    "PF branch-stat coverage:",
    f"{len(set(expected_wells) & set(valid_statistics))}/{len(expected_wells)}",
    f"= {coverage:.3%}",
    flush=True,
)
if coverage < BRANCH_MIN_STAT_COVERAGE:
    missing = sorted(set(expected_wells) - set(valid_statistics))[:20]
    raise RuntimeError(
        f"PF branch-stat coverage {coverage:.3%} is below "
        f"{BRANCH_MIN_STAT_COVERAGE:.1%}; missing={missing}"
    )

values = submission["tvt"].to_numpy(dtype=np.float64).copy()
report = []
for well in expected_wells:
    payload = valid_statistics.get(well)
    if payload is None:
        report.append({"well": well, "reason": "missing_statistics", "shift": 0.0})
        continue

    center_low = float(payload["center_low"])
    center_high = float(payload["center_high"])
    mass_low = float(payload["mass_low"])
    mass_high = float(payload["mass_high"])
    weighted_center = float(payload["weighted_center"])
    separation = abs(center_high - center_low)
    minor_mass = min(mass_low, mass_high)
    shift = 0.0
    reason = "not_qualified"
    moved_rows = 0

    if minor_mass < BRANCH_MIN_MASS:
        reason = "skip_minor_mass"
    elif not (
        BRANCH_SEPARATION_LOW_FT
        <= separation
        <= BRANCH_SEPARATION_HIGH_FT
    ):
        reason = "skip_separation"
    else:
        midpoint = 0.5 * (center_low + center_high)
        shift = float(
            np.clip(
                BRANCH_STRENGTH * (midpoint - weighted_center),
                -BRANCH_SHIFT_CAP_FT,
                BRANCH_SHIFT_CAP_FT,
            )
        )
        eval_rows = set(int(value) for value in payload["eval_rows"])
        mask = row_well.eq(well).to_numpy()
        mask &= row_index.isin(eval_rows).to_numpy()
        if abs(shift) >= 0.01 and bool(mask.any()):
            values[mask] += shift
            moved_rows = int(mask.sum())
            reason = "applied"
        else:
            shift = 0.0
            reason = "skip_zero_or_missing_rows"

    report.append(
        {
            "well": well,
            "reason": reason,
            "center_low": center_low,
            "center_high": center_high,
            "mass_low": mass_low,
            "mass_high": mass_high,
            "minor_mass": minor_mass,
            "separation_ft": separation,
            "weighted_center": weighted_center,
            "shift": shift,
            "moved_rows": moved_rows,
            "seed_count": int(payload.get("seed_count", 0)),
        }
    )

if not np.isfinite(values).all():
    raise RuntimeError("PF branch hedge produced non-finite predictions")
move = values - submission["tvt"].to_numpy(dtype=np.float64)
if float(np.max(np.abs(move))) > BRANCH_SHIFT_CAP_FT + 1e-8:
    raise RuntimeError("PF branch hedge exceeded its movement cap")

report = pd.DataFrame(report)
report.to_csv(work / "pf_seed_branch_hedge_report.csv", index=False)
submission["tvt"] = values
submission.to_csv(work / "submission_pf_seed_branch_hedge.csv", index=False)
submission.to_csv(submission_path, index=False)

summary = {
    "statistics_wells": int(len(valid_statistics)),
    "submission_wells": int(len(expected_wells)),
    "coverage": float(coverage),
    "applied_wells": int(report["reason"].eq("applied").sum()),
    "moved_rows": int((np.abs(move) > 0.0).sum()),
    "mean_abs_move": float(np.mean(np.abs(move))),
    "max_abs_move": float(np.max(np.abs(move))),
    "reason_counts": report["reason"].value_counts().to_dict(),
}
print("PF SEED BRANCH HEDGE", summary, flush=True)
print(report.to_string(index=False), flush=True)
