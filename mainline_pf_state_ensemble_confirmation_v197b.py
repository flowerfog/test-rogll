"""Frozen 96-well confirmation of V197 PF-state aggregation.

The implementation is derived at runtime from the audited V197 source so the
PF and learned-core lineage remain byte-for-byte identical.  V97, V98, and all
48 V197 gateway wells are excluded.  Predictions are checkpointed per well.
CPU thread pools are capped at three so this job can run beside GPU-heavy V208.
"""

from pathlib import Path
import os


for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ[variable] = "3"

ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "mainline_pf_state_ensemble_gateway_v197_cell.py"
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

source = SOURCE.read_text(encoding="utf-8")

replacements = {
    'OUT_V197 = OOF_V197 / "mainline_pf_state_ensemble_gateway_v197"':
        'OUT_V197 = OOF_V197 / "mainline_pf_state_ensemble_confirmation_v197b"',
    "N_WELLS_V197 = 48": "N_WELLS_V197 = 96",
    '"experiment": "mainline_pf_state_ensemble_gateway_v197",':
        '"experiment": "mainline_pf_state_ensemble_confirmation_v197b",',
    '"stage": "disjoint_48well_pre_feature_PF_state_aggregation_gateway",':
        '"stage": "disjoint_96well_frozen_pre_feature_PF_state_aggregation_confirmation",',
    'if len(excluded) < 68:': "if len(excluded) < 110:",
    'if len(complete_wells_v197) < 44:': "if len(complete_wells_v197) < 90:",
    '"complete_wells_ge_44": len(complete_wells_v197) >= 44,':
        '"complete_wells_ge_90": len(complete_wells_v197) >= 90,',
    '"positive_well_fraction_ge_0p55": positive_fraction_v197 >= 0.55,':
        '"p95_well_rmse_gain_nonnegative": p95_gain_v197 >= 0.0,',
}
for old, new in replacements.items():
    if old not in source:
        raise RuntimeError(f"V197b source marker missing: {old}")
    source = source.replace(old, new, 1)

# Add all gateway wells to the frozen exclusion set.
exclusion_marker = '''        V98_DIR_V197 / "paired_predictions.parquet",
    )'''
exclusion_replacement = '''        V98_DIR_V197 / "paired_predictions.parquet",
        OOF_V197 / "mainline_pf_state_ensemble_gateway_v197" / "paired_predictions.parquet",
    )'''
if exclusion_marker not in source:
    raise RuntimeError("V197b exclusion marker missing")
source = source.replace(exclusion_marker, exclusion_replacement, 1)

# Cap Numba after the notebook bootstrap has compiled its kernels, and cap any
# sklearn LightGBM wrappers after loading them.
bootstrap_marker = '''_bootstrap_v197()
selected_wells_v197, ranking_source_v197, excluded_wells_v197 = _select_wells_v197()
models_dir_v197, feature_names_v197, models_v197 = _load_models_p5()'''
bootstrap_replacement = '''_bootstrap_v197()
try:
    import numba
    numba.set_num_threads(3)
except Exception:
    pass
selected_wells_v197, ranking_source_v197, excluded_wells_v197 = _select_wells_v197()
models_dir_v197, feature_names_v197, models_v197 = _load_models_p5()
for _model_v197b in models_v197:
    try:
        _model_v197b.set_params(n_jobs=3, num_threads=3)
    except (AttributeError, TypeError, ValueError):
        pass'''
if bootstrap_marker not in source:
    raise RuntimeError("V197b bootstrap marker missing")
source = source.replace(bootstrap_marker, bootstrap_replacement, 1)

# Replace the all-or-nothing prediction loop with one-file-per-well resume.
loop_start = "parts_v197: list[pd.DataFrame] = []"
loop_end = "if failures_v197:"
if loop_start not in source or loop_end not in source:
    raise RuntimeError("V197b loop boundary missing")
prefix, remainder = source.split(loop_start, 1)
_, suffix = remainder.split(loop_end, 1)
resume_loop = '''parts_v197: list[pd.DataFrame] = []
failures_v197: list[dict[str, str]] = []
cache_dir_v197b = OUT_V197 / "well_cache"
cache_dir_v197b.mkdir(parents=True, exist_ok=True)
resumed_v197b = 0
for number_v197, well_v197 in enumerate(selected_wells_v197, 1):
    cache_path_v197b = cache_dir_v197b / f"{well_v197}.parquet"
    try:
        if cache_path_v197b.is_file():
            pair_v197b = pd.read_parquet(cache_path_v197b)
            if set(pair_v197b["variant"].astype(str).unique()) != {"seed1_reference", "state_mean4"}:
                raise RuntimeError(f"{well_v197}: invalid cached variants")
            resumed_v197b += 1
        else:
            reference_v197 = _predict_v197(well_v197, aggregate=False)
            candidate_v197 = _predict_v197(well_v197, aggregate=True)
            keys_v197 = ["well", "id", "md_since", "truth"]
            if not reference_v197[keys_v197].equals(candidate_v197[keys_v197]):
                raise RuntimeError(f"{well_v197}: reference/candidate row alignment differs")
            pair_v197b = pd.concat((reference_v197, candidate_v197), ignore_index=True)
            temporary_v197b = cache_path_v197b.with_suffix(".tmp.parquet")
            pair_v197b.to_parquet(temporary_v197b, index=False, compression="zstd")
            temporary_v197b.replace(cache_path_v197b)
        parts_v197.append(pair_v197b)
    except Exception as exc_v197:
        failures_v197.append({"well": well_v197, "reason": repr(exc_v197)})
    if number_v197 % 4 == 0 or number_v197 == len(selected_wells_v197):
        print(
            f"V197b confirmation {number_v197}/{len(selected_wells_v197)} "
            f"resumed={resumed_v197b} failures={len(failures_v197)} "
            f"elapsed_min={(time.time() - STARTED_V197) / 60.0:.1f}",
            flush=True,
        )

if failures_v197:'''
source = prefix + resume_loop + suffix

# Pooled RMSE is the objective, but catastrophic-well tail behavior must not
# worsen.  This replaces the gateway's majority-well criterion.
p95_marker = '''positive_fraction_v197 = float(np.mean(by_well_v197["gain"] > 0.0))
bootstrap_v197 = _well_bootstrap_v197(by_well_v197)'''
p95_replacement = '''positive_fraction_v197 = float(np.mean(by_well_v197["gain"] > 0.0))
p95_gain_v197 = float(
    by_well_v197["reference_rmse"].quantile(0.95)
    - by_well_v197["candidate_rmse"].quantile(0.95)
)
bootstrap_v197 = _well_bootstrap_v197(by_well_v197)'''
if p95_marker not in source:
    raise RuntimeError("V197b p95 marker missing")
source = source.replace(p95_marker, p95_replacement, 1)

verdict_marker = '''    "positive_well_fraction": positive_fraction_v197,
    "pf_ancc_state_rms_change": state_rms_change_v197,'''
verdict_replacement = '''    "positive_well_fraction": positive_fraction_v197,
    "p95_well_rmse_gain": p95_gain_v197,
    "pf_ancc_state_rms_change": state_rms_change_v197,'''
if verdict_marker not in source:
    raise RuntimeError("V197b verdict marker missing")
source = source.replace(verdict_marker, verdict_replacement, 1)

next_marker = '''"PASS -> test exact state aggregation through the final V69/V77 lineage"
        if strict_pass_v197 else
        "FAIL -> state aggregation is not strong enough; do not build a Kaggle submission"'''
next_replacement = '''"PASS -> run exact state aggregation through the final V69/V77 lineage"
        if strict_pass_v197 else
        "FAIL -> frozen 96-well confirmation rejected state aggregation"'''
if next_marker not in source:
    raise RuntimeError("V197b next marker missing")
source = source.replace(next_marker, next_replacement, 1)

exec(compile(source, str(SOURCE), "exec"), globals())
