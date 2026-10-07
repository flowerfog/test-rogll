"""Direct Kaggle cells for the exact 6.781 mainline notebook.

Source notebook:
    fork-of-fork-of-fork-of-rogii-wellbore-geol-ecf6db (1).ipynb
    SHA256 ECA8BF99480B0C6AAD9524455BDB087546F90EEA635CD06D80C41DA8463CA7E2

These are notebook cells, not a standalone script. Follow the CELL placement
comments. Keep the existing 0.30 Ridge + 0.70 selector, quartic projections,
0.55 SP45 + 0.45 FLE, CP12, Gold, and geometric Neighbor settings unchanged.
"""


# %% CELL A0: replace the current first profile cell
import os

# Keep the proven 6.781 profile while isolating the untested seed-branch hedge.
os.environ["ROGII_GOLD_PROFILE"] = "conservative"

# A1+A2+B with 1.30 scored 6.806, so the multiplier is closed on this mainline.
LEARNED_LIKPF_GR_SIGMA_MULTIPLIER = 1.00

# The branch hedge consumes the SP45 seed bank after Neighbor has completed.
RUN_PF_SEED_BRANCH_HEDGE = True


# %% CELL A1: paste immediately AFTER the original SP45 function cell (cell 4)
# This redefines only run_pf_lik_ensemble_scales. It preserves the original PF
# outputs and optionally retains the already-computed seed trajectories.
import numpy as np
import pandas as pd


PF_SEED_BRANCH_STATS = {}


def _pf_branch_statistics(pred_arr, log_likelihoods, eval_mask):
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
    level = np.nanmedian(pred_arr[:, eval_mask], axis=1)
    valid = np.isfinite(level) & np.isfinite(seed_weight) & (seed_weight > 0.0)
    level = level[valid]
    seed_weight = seed_weight[valid]
    if len(level) < 4:
        raise RuntimeError("too few finite PF seeds for branch statistics")
    seed_weight /= max(float(seed_weight.sum()), 1e-12)

    order = np.argsort(level)
    x = level[order]
    weight = seed_weight[order]
    cumulative_weight = np.cumsum(weight)
    cumulative_x = np.cumsum(weight * x)
    cumulative_x2 = np.cumsum(weight * x * x)
    total_weight = float(cumulative_weight[-1])
    total_x = float(cumulative_x[-1])
    total_x2 = float(cumulative_x2[-1])

    best = None
    for cut in range(1, len(x)):
        mass_low = float(cumulative_weight[cut - 1])
        mass_high = total_weight - mass_low
        if mass_low < 0.05 or mass_high < 0.05:
            continue
        sum_low = float(cumulative_x[cut - 1])
        sum_high = total_x - sum_low
        sse_low = float(
            cumulative_x2[cut - 1] - sum_low * sum_low / mass_low
        )
        sse_high = float(
            total_x2
            - cumulative_x2[cut - 1]
            - sum_high * sum_high / mass_high
        )
        candidate = (
            max(0.0, sse_low) + max(0.0, sse_high),
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


def run_pf_lik_ensemble_scales(
    hw,
    tw,
    scales=SELECTOR_SCALES,
    n_particles=500,
    n_seeds=128,
    branch_stats=None,
):
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

    if branch_stats is not None:
        eval_mask = (
            pd.to_numeric(hw["TVT_input"], errors="coerce").isna().to_numpy()
        )
        branch_stats.update(
            _pf_branch_statistics(pred_arr, log_likelihoods, eval_mask)
        )
    return output


# %% CELL A2: paste immediately AFTER the original learned lik-PF cell (cell 23)
# It redefines only lik_pf. Keep the multiplier at 1.00 on the 6.781 mainline;
# 1.30 was tested by A1+A2+B and scored 6.806.
def lik_pf(
    hw,
    tw,
    n_particles=CFG.PF_PARTICLES,
    n_seeds=CFG.PF_SEEDS,
    scales=CFG.PF_SCALES,
    init_spr=4.5,
    seed_base=0,
    with_quality=False,
):
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]
    ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0:
        return {}, np.array([]), {}

    last = kn.iloc[-1]
    structure_last = float(last.TVT_input) + float(last.Z)
    typewell_at_known = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gr_sigma = float(
        np.clip(
            np.nanstd(kn.GR.fillna(0).values - typewell_at_known),
            10.0,
            60.0,
        )
    )
    gr_sigma *= float(
        globals().get("LEARNED_LIKPF_GR_SIGMA_MULTIPLIER", 1.00)
    )

    tail = kn.tail(30)
    delta_tvt = np.diff(tail.TVT_input.values)
    delta_z = np.diff(tail.Z.values)
    delta_md = np.diff(tail.MD.values)
    valid_step = delta_md > 0
    initial_rate = (
        float(np.median((delta_tvt + delta_z)[valid_step] / delta_md[valid_step]))
        if valid_step.sum() >= 3
        else 0.0
    )
    gr_grid, grid_min, grid_step = _grid(tw_tvt, tw_gr)
    gr_values = (
        hw.GR.interpolate(limit_direction="both")
        .fillna(tw_gr.mean())
        .values.astype(float)[ev.index]
    )
    predictions, log_likelihoods = _pf_lik_allseeds(
        ev.MD.values.astype(float),
        ev.Z.values.astype(float),
        gr_values,
        gr_grid,
        grid_min,
        grid_step,
        gr_sigma,
        structure_last,
        initial_rate,
        n_particles,
        n_seeds,
        seed_base,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        init_spr,
    )
    centred = log_likelihoods - log_likelihoods.max()
    output = {}
    for scale in scales:
        weight = np.exp(centred / float(scale))
        weight /= weight.sum()
        output[f"pf_scale_{scale:g}"] = (
            weight[:, None] * predictions
        ).sum(0)
    output["pf_mean"] = predictions.mean(0)

    quality = {}
    if with_quality:
        quality = {
            "pf_best_ll": float(log_likelihoods.max()) / len(ev),
            "pf_ll_spread": float(log_likelihoods.std()),
            "pf_pt_std": predictions.std(0).astype(np.float32),
            "pf_gr_sig": gr_sigma,
            "pf_gr_sig_multiplier": float(
                globals().get("LEARNED_LIKPF_GR_SIGMA_MULTIPLIER", 1.00)
            ),
        }
    return output, ev.index.values, quality


# %% CELL B: replace the complete original SP45 test-inference cell (cell 15)
sample = pd.read_csv(CFG.dataset_path / "sample_submission.csv")
sample["well"] = sample["id"].str[:8]
sample["row_idx"] = sample["id"].str[9:].astype(int)

train_hw_files = sorted(
    glob.glob(str(CFG.dataset_path / "train" / "*__horizontal_well.csv"))
)
train_wells = [os.path.basename(path).split("__")[0] for path in train_hw_files]
test_hw_files = sorted(
    glob.glob(str(CFG.dataset_path / "test" / "*__horizontal_well.csv"))
)
test_wells = [os.path.basename(path).split("__")[0] for path in test_hw_files]

rows = []
PF_SEED_BRANCH_STATS = {}
for index, wid in enumerate(test_wells):
    print(f"\nProcessing {index + 1}/{len(test_wells)}: {wid}...")
    hw_te, tw_te = load_well(wid, "test")

    tvt_phys = None
    hw_tr = None
    tw_tr = None
    if wid in train_wells:
        try:
            hw_tr, tw_tr = load_well(wid, "train")
            hw_te["TVT_input"] = hw_tr["TVT_input"].values
            tvt_phys = tvt_from_contacts(hw_tr, tw_tr)
            print("  Physical model OK")
        except Exception as exc:
            print(f"  Physical model failed: {exc}")
            tvt_phys = None

    selector_code, selector_variant, selector_n_eval, selector_z_span = (
        selector_well_code(hw_te)
    )
    tw_ref = tw_tr if tw_tr is not None else tw_te
    try:
        seed_branch = {}
        pf_by_scale = run_pf_lik_ensemble_scales(
            hw_te,
            tw_ref,
            scales=SELECTOR_SCALES,
            n_particles=500,
            n_seeds=128,
            branch_stats=seed_branch,
        )
        if seed_branch:
            PF_SEED_BRANCH_STATS[str(wid)] = seed_branch
        tvt_pf = pf_by_scale["pf_scale_8"]
        print(f"  PF 128-seed lik-ensemble OK scales={SELECTOR_SCALES}")
    except Exception as exc:
        print(f"  PF failed: {exc}")
        PF_SEED_BRANCH_STATS.pop(str(wid), None)
        last_known = hw_te["TVT_input"].dropna()
        last_value = float(last_known.iloc[-1]) if len(last_known) else 0.0
        tvt_pf = hw_te["TVT_input"].fillna(last_value).to_numpy(dtype=float)
        pf_by_scale = {
            f"pf_scale_{float(scale):g}": tvt_pf.copy()
            for scale in SELECTOR_SCALES
        }

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

    well_submission = sample[sample["well"] == wid]
    for _, row in well_submission.iterrows():
        row_index = int(row["row_idx"])
        value = (
            float(tvt_phys.iloc[row_index])
            if tvt_phys is not None
            else float(tvt_selector[row_index])
        )
        rows.append({"id": row["id"], "tvt": value})
    print(f"  Added {len(well_submission)} rows")


# %% CELL C: paste immediately AFTER active geometric Neighbor cell (cell 42)
# It must be the final prediction-changing cell.
from pathlib import Path as _BranchPath

import numpy as _branch_np
import pandas as _branch_pd


if not bool(globals().get("RUN_PF_SEED_BRANCH_HEDGE", True)):
    print("PF seed-branch hedge disabled")
else:
    _branch_work = (
        _BranchPath("/kaggle/working")
        if _BranchPath("/kaggle/working").exists()
        else _BranchPath(".")
    )
    _branch_submission_path = _branch_work / "submission.csv"
    _branch_submission = _branch_pd.read_csv(_branch_submission_path)
    if list(_branch_submission.columns) != ["id", "tvt"]:
        raise RuntimeError("branch hedge expected id,tvt submission schema")
    _branch_submission.to_csv(
        _branch_work / "submission_before_pf_seed_branch.csv",
        index=False,
    )
    _branch_well = _branch_submission["id"].astype(str).str.rsplit(
        "_", n=1
    ).str[0]
    _branch_row = _branch_pd.to_numeric(
        _branch_submission["id"].astype(str).str.rsplit("_", n=1).str[-1],
        errors="coerce",
    )
    _branch_tvt = _branch_pd.to_numeric(
        _branch_submission["tvt"], errors="coerce"
    ).to_numpy(dtype=float)
    if not _branch_np.isfinite(_branch_tvt).all():
        raise RuntimeError("non-finite base submission before branch hedge")

    _branch_strength = 0.60
    _branch_min_mass = 0.25
    _branch_separation_low = 4.0
    _branch_separation_high = 40.0
    _branch_cap = 2.0
    _branch_report = []
    for well, stats in sorted(PF_SEED_BRANCH_STATS.items()):
        reason = "not_qualified"
        shift = 0.0
        moved_rows = 0
        try:
            low = float(stats["center_low"])
            high = float(stats["center_high"])
            mass_low = float(stats["mass_low"])
            mass_high = float(stats["mass_high"])
            weighted_center = float(stats["weighted_center"])
            separation = abs(high - low)
            minor_mass = min(mass_low, mass_high)
            if minor_mass < _branch_min_mass:
                reason = "skip_minor_mass"
            elif not (
                _branch_separation_low
                <= separation
                <= _branch_separation_high
            ):
                reason = "skip_separation"
            else:
                midpoint = 0.5 * (low + high)
                shift = float(
                    _branch_np.clip(
                        _branch_strength * (midpoint - weighted_center),
                        -_branch_cap,
                        _branch_cap,
                    )
                )
                eval_rows = set(int(value) for value in stats.get("eval_rows", []))
                mask = (_branch_well == str(well)).to_numpy()
                if eval_rows:
                    mask &= _branch_row.isin(eval_rows).to_numpy()
                if abs(shift) >= 0.01 and bool(mask.any()):
                    _branch_tvt[mask] += shift
                    moved_rows = int(mask.sum())
                    reason = "applied"
                else:
                    shift = 0.0
                    reason = "skip_zero_or_missing_rows"
            _branch_report.append(
                {
                    "well": str(well),
                    "reason": reason,
                    "center_low": low,
                    "center_high": high,
                    "mass_low": mass_low,
                    "mass_high": mass_high,
                    "minor_mass": minor_mass,
                    "separation_ft": separation,
                    "weighted_center": weighted_center,
                    "shift": shift,
                    "moved_rows": moved_rows,
                    "seed_count": int(stats.get("seed_count", 0)),
                }
            )
        except Exception as exc:
            _branch_report.append(
                {"well": str(well), "reason": "error", "error": repr(exc)}
            )

    if not _branch_np.isfinite(_branch_tvt).all():
        raise RuntimeError("branch hedge produced non-finite predictions")
    _branch_submission["tvt"] = _branch_tvt
    _branch_submission.to_csv(
        _branch_work / "submission_pf_seed_branch_hedge.csv",
        index=False,
    )
    _branch_submission.to_csv(_branch_submission_path, index=False)
    _branch_report = _branch_pd.DataFrame(_branch_report)
    _branch_report.to_csv(
        _branch_work / "pf_seed_branch_hedge_report.csv", index=False
    )
    expected_wells = int(_branch_well.nunique())
    statistics_wells = int(len(PF_SEED_BRANCH_STATS))
    applied_wells = (
        int(_branch_report["reason"].eq("applied").sum())
        if "reason" in _branch_report.columns
        else 0
    )
    moved_rows_total = (
        int(
            _branch_pd.to_numeric(
                _branch_report["moved_rows"], errors="coerce"
            ).fillna(0).sum()
        )
        if "moved_rows" in _branch_report.columns
        else 0
    )
    reason_counts = (
        _branch_report["reason"].value_counts().to_dict()
        if "reason" in _branch_report.columns
        else {}
    )
    print(
        "PF SEED-BRANCH HEDGE",
        {
            "statistics_wells": statistics_wells,
            "submission_wells": expected_wells,
            "coverage": statistics_wells / max(expected_wells, 1),
            "applied_wells": applied_wells,
            "moved_rows": moved_rows_total,
            "reason_counts": reason_counts,
        },
        flush=True,
    )


# %% CELL D: paste last, after CELL C
from pathlib import Path as _FinalAuditPath
import hashlib as _final_hashlib
import json as _final_json
import numpy as _final_np
import pandas as _final_pd


_final_work = _FinalAuditPath("/kaggle/working")
_final_path = _final_work / "submission.csv"
_sample_path = (
    _FinalAuditPath("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
    / "sample_submission.csv"
)
_final = _final_pd.read_csv(_final_path)
_sample = _final_pd.read_csv(_sample_path)
if list(_final.columns) != ["id", "tvt"]:
    raise RuntimeError(f"bad final columns: {list(_final.columns)}")
if len(_final) != len(_sample):
    raise RuntimeError("final row count mismatch")
if not _final["id"].astype(str).equals(_sample["id"].astype(str)):
    raise RuntimeError("final id order mismatch")
_values = _final["tvt"].to_numpy(dtype=float)
if not _final_np.isfinite(_values).all():
    raise RuntimeError("final submission contains non-finite TVT")
_digest = _final_hashlib.sha256(_final_path.read_bytes()).hexdigest()
_audit = {
    "source_mainline_sha256": (
        "ECA8BF99480B0C6AAD9524455BDB087546F90EEA635CD06D80C41DA8463CA7E2"
    ),
    "gold_profile": os.environ.get("ROGII_GOLD_PROFILE"),
    "learned_likpf_gr_sigma_multiplier": float(
        LEARNED_LIKPF_GR_SIGMA_MULTIPLIER
    ),
    "branch_statistics_wells": int(len(PF_SEED_BRANCH_STATS)),
    "rows": int(len(_final)),
    "tvt_mean": float(_values.mean()),
    "tvt_std": float(_values.std()),
    "sha256_submission_csv": _digest,
}
(_final_work / "submission_mainline_6p781_pf_patch.csv").write_bytes(
    _final_path.read_bytes()
)
(_final_work / "submission_mainline_6p781_pf_patch_audit.json").write_text(
    _final_json.dumps(_audit, indent=2, sort_keys=True),
    encoding="utf-8",
)
print("MAINLINE 6.781 PF PATCH FINAL", _audit, flush=True)
