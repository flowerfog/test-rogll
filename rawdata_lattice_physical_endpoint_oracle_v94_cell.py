"""V94: cross-evidence oracle for deterministic physical bank selectors.

The learned V70 ranker mixes GR and structural evidence.  This audit selects
from the exact same frozen candidate bank using three deliberately different,
label-free rules: emission only, structure among the best-emission quartile,
and an equal-rank emission/structure compromise.  It then measures whether
any endpoint has enough error independence from V76 to justify training a
consumer.  Truth is used only for metrics and blend oracles.
"""

from __future__ import annotations

import json
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "rawdata_lattice_physical_endpoint_oracle_v94"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

V86_SOURCE = ROOT / "rawdata_lattice_v70_matched_disagreement_guard_v86_cell.py"
V70_PATH = OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70" / "all773_proxy_crossfit_oof.parquet"
V76_WEIGHTS = OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76" / "fixed_ensemble_weights_oof.parquet"

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
FOLDS = (3, 2)
RULES = ("emission_only", "structure_guarded", "balanced_rank")
STRATA = (
    ("near_0_500", -np.inf, 500.0),
    ("mid_500_1000", 500.0, 1000.0),
    ("mid_1000_2000", 1000.0, 2000.0),
    ("far_gt2000", 2000.0, np.inf),
)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _bootstrap_v86():
    source = V86_SOURCE.read_text(encoding="utf-8")
    marker = "for required in (V70_SOURCE, V85_SOURCE, V70_PATH, V76_WEIGHTS):"
    if marker not in source:
        raise RuntimeError(f"Cannot identify V86 execution boundary in {V86_SOURCE}")
    module = types.ModuleType("raw_lattice_v86_v94_definitions")
    module.__file__ = str(V86_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source.split(marker, 1)[0] + "\n", str(V86_SOURCE), "exec"), module.__dict__)
    module.R = module._bootstrap_v70()
    module.H = module._bootstrap_v85_helpers()
    return module


def _rank_low(values):
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(values), np.float64)
    rank[order] = np.arange(len(values), dtype=np.float64)
    return rank / max(len(values) - 1, 1)


def _robust_z(values):
    values = np.asarray(values, np.float64)
    centre = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - centre)))
    return (values - centre) / max(scale, 1e-8)


def _base(fold):
    frame = pd.read_parquet(
        V70_PATH,
        columns=["id", "well", "row_idx", "fold", TRUTH, "md_since", PARENT, COARSE],
        filters=[("fold", "=", int(fold))],
    ).sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    weights = pd.read_parquet(V76_WEIGHTS)
    weights = weights[weights["fold"].eq(fold)][["well", "ensemble_weight"]]
    weight_map = dict(zip(weights["well"].astype(str), weights["ensemble_weight"]))
    frame["dose"] = frame["well"].astype(str).map(weight_map)
    if frame.empty or frame["dose"].isna().any():
        raise RuntimeError(f"Fold{fold}: incomplete V76 alignment")
    parent = frame[PARENT].to_numpy(np.float64)
    frame["v76"] = parent + frame["dose"].to_numpy(np.float64) * (
        frame[COARSE].to_numpy(np.float64) - parent
    )
    return frame


def _feature_indices(names):
    lookup = {name: index for index, name in enumerate(names)}
    required = (
        "gr_raw_cauchy_mean",
        "geom_change_abs_mean",
        "geom_change_abs_max",
        "geom_change_sq_mean",
    )
    missing = [name for name in required if name not in lookup]
    if missing:
        raise RuntimeError(f"Missing physical evidence features: {missing}")
    return {name: lookup[name] for name in required}


def _select(features, indices):
    emission = features[:, indices["gr_raw_cauchy_mean"]].astype(np.float64)
    structure = (
        _robust_z(features[:, indices["geom_change_abs_mean"]])
        + 0.50 * _robust_z(features[:, indices["geom_change_abs_max"]])
        + 0.25 * _robust_z(features[:, indices["geom_change_sq_mean"]])
    )
    emission_rank = _rank_low(emission)
    structure_rank = _rank_low(structure)
    emission_index = int(np.argmin(emission))
    allowed = np.flatnonzero(emission_rank <= 0.25)
    guarded_index = int(allowed[np.argmin(structure[allowed])])
    balanced_index = int(np.argmin(emission_rank + structure_rank))
    return {
        "emission_only": emission_index,
        "structure_guarded": guarded_index,
        "balanced_rank": balanced_index,
    }, {
        "candidate_count": int(len(features)),
        "emission_structure_spearman": float(pd.Series(emission).corr(pd.Series(structure), method="spearman")),
        "guarded_pool": int(len(allowed)),
    }


def _build_fold(fold):
    base = _base(fold)
    wells = sorted(base["well"].astype(str).unique())
    parts, reports = [], []
    for number, well in enumerate(wells, 1):
        local = base[base["well"].astype(str).eq(well)].copy()
        report = {"fold": int(fold), "well": well, "status": "built", "reason": ""}
        try:
            record = M.R.V["_load_record"](well)
            paths, source_mask, arm_rank, arm_score_z, segments, _ = M.R.Q._candidate_union(record, M.R.prior)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                evidence, names = M.R.Q._candidate_evidence(
                    record, M.R.prior, paths, segments, arm_rank, arm_score_z, source_mask
                )
            if list(names) != list(M.R.feature_names):
                raise RuntimeError("feature schema mismatch")
            indices = _feature_indices(names)
            selected, diagnostics = _select(evidence, indices)
            eval_rows = record["eval_rows"]
            truth = record["truth"][eval_rows].astype(np.float64)
            if len(local) != len(eval_rows):
                raise RuntimeError(f"row count {len(local)}/{len(eval_rows)}")
            if np.max(np.abs(local[TRUTH].to_numpy(np.float64) - truth)) > 0.01:
                raise RuntimeError("truth alignment")
            for rule, index in selected.items():
                path = M.R.B["_path_prediction"](
                    record, M.R.prior, paths[index], segments
                )[eval_rows]
                local[f"coarse_{rule}"] = np.asarray(path, np.float32)
            report.update(diagnostics)
            report.update({f"index_{key}": int(value) for key, value in selected.items()})
        except Exception as exc:
            report["status"] = "v70_fallback"
            report["reason"] = repr(exc)
            for rule in RULES:
                local[f"coarse_{rule}"] = local[COARSE].to_numpy(np.float32)
        parts.append(local)
        reports.append(report)
        if number % 10 == 0 or number == len(wells):
            fallback = sum(item["status"] != "built" for item in reports)
            print(f"V94 Fold{fold} endpoints {number}/{len(wells)} fallbacks={fallback}", flush=True)
    output = pd.concat(parts, ignore_index=True)
    return output.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True), reports


def _metric(frame, prediction, endpoint):
    error = np.asarray(prediction, np.float64) - frame[TRUTH].to_numpy(np.float64)
    error2 = error * error
    bywell = pd.DataFrame({"well": frame["well"], "e2": error2})
    bywell = np.sqrt(bywell.groupby("well", sort=False)["e2"].mean().to_numpy())
    distance = frame["md_since"].to_numpy(np.float64)
    result = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(error2.sum()),
        "rmse": float(np.sqrt(error2.mean())),
        "mae": float(np.abs(error).mean()),
        "median_well_rmse": float(np.median(bywell)),
        "p95_well_rmse": float(np.quantile(bywell, 0.95)),
    }
    for name, lower, upper in STRATA:
        mask = (distance > lower) & (distance <= upper)
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return result


def _optimal_mix(base, candidate, truth):
    delta = np.asarray(candidate, np.float64) - np.asarray(base, np.float64)
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(delta, np.asarray(truth, np.float64) - base) / denominator, 0.0, 1.0))


def _detrended_correlation(error_a, error_b, distance):
    a = np.asarray(error_a, np.float64).copy()
    b = np.asarray(error_b, np.float64).copy()
    distance = np.asarray(distance, np.float64)
    within = {}
    for name, lower, upper in STRATA:
        mask = (distance > lower) & (distance <= upper)
        if mask.sum() < 3:
            within[name] = np.nan
            continue
        within[name] = float(np.corrcoef(a[mask], b[mask])[0, 1])
        a[mask] -= float(np.mean(a[mask]))
        b[mask] -= float(np.mean(b[mask]))
    return float(np.corrcoef(a, b)[0, 1]), within


def _audit(frame, rule):
    parent = frame[PARENT].to_numpy(np.float64)
    dose = frame["dose"].to_numpy(np.float64)
    candidate = parent + dose * (
        frame[f"coarse_{rule}"].to_numpy(np.float64) - parent
    )
    baseline = frame["v76"].to_numpy(np.float64)
    truth = frame[TRUTH].to_numpy(np.float64)
    fixed_weight = _optimal_mix(baseline, candidate, truth)
    fixed = baseline + fixed_weight * (candidate - baseline)
    oracle = np.empty(len(frame), np.float64)
    oracle_weights = []
    for _, indices in frame.groupby("well", sort=False).indices.items():
        indices = np.asarray(indices, np.int64)
        weight = _optimal_mix(baseline[indices], candidate[indices], truth[indices])
        oracle[indices] = baseline[indices] + weight * (candidate[indices] - baseline[indices])
        oracle_weights.append(weight)
    error_base = baseline - truth
    error_candidate = candidate - truth
    detrended, within = _detrended_correlation(
        error_base, error_candidate, frame["md_since"].to_numpy(np.float64)
    )
    base_metric = _metric(frame, baseline, "v76")
    candidate_metric = _metric(frame, candidate, rule)
    fixed_metric = _metric(frame, fixed, f"{rule}_fixed_oracle")
    oracle_metric = _metric(frame, oracle, f"{rule}_perwell_oracle")
    return {
        "rule": rule,
        "standalone_rmse": candidate_metric["rmse"],
        "signed_error_correlation": float(np.corrcoef(error_base, error_candidate)[0, 1]),
        "detrended_error_correlation": detrended,
        "within_distance_correlation": within,
        "fixed_oracle_weight_candidate": fixed_weight,
        "fixed_oracle_rmse": fixed_metric["rmse"],
        "fixed_oracle_gain": base_metric["rmse"] - fixed_metric["rmse"],
        "perwell_oracle_rmse": oracle_metric["rmse"],
        "perwell_oracle_gain": base_metric["rmse"] - oracle_metric["rmse"],
        "mean_perwell_oracle_weight_candidate": float(np.mean(oracle_weights)),
        "far_fixed_oracle_gain": base_metric["far_gt2000_rmse"] - fixed_metric["far_gt2000_rmse"],
        "far_perwell_oracle_gain": base_metric["far_gt2000_rmse"] - oracle_metric["far_gt2000_rmse"],
    }, [base_metric, candidate_metric, fixed_metric, oracle_metric]


for required in (V86_SOURCE, V70_PATH, V76_WEIGHTS):
    if not required.is_file():
        raise FileNotFoundError(required)

M = _bootstrap_v86()
all_audits = {}
summary_rows = []
report_rows = []
for fold in FOLDS:
    frame, reports = _build_fold(fold)
    report_rows.extend(reports)
    all_audits[str(fold)] = {}
    for rule in RULES:
        audit, metrics = _audit(frame, rule)
        all_audits[str(fold)][rule] = audit
        for metric in metrics:
            summary_rows.append({"fold": fold, "rule": rule, **metric})

pd.DataFrame(report_rows).to_csv(OUT_DIR / "endpoint_build_report.csv", index=False)
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)

investment = {}
for rule in RULES:
    fold3 = all_audits["3"][rule]
    fold2 = all_audits["2"][rule]
    investment[rule] = {
        "fold3_fixed_oracle_gain": fold3["fixed_oracle_gain"],
        "fold2_fixed_oracle_gain": fold2["fixed_oracle_gain"],
        "fold3_perwell_oracle_gain": fold3["perwell_oracle_gain"],
        "fold2_perwell_oracle_gain": fold2["perwell_oracle_gain"],
        "fold3_detrended_correlation": fold3["detrended_error_correlation"],
        "fold2_detrended_correlation": fold2["detrended_error_correlation"],
        "promising": bool(
            fold3["fixed_oracle_gain"] >= 0.05
            and fold2["fixed_oracle_gain"] >= 0.05
            and fold3["perwell_oracle_gain"] >= 0.20
            and fold2["perwell_oracle_gain"] >= 0.20
            and fold3["detrended_error_correlation"] <= 0.90
            and fold2["detrended_error_correlation"] <= 0.90
        ),
    }
promising = [rule for rule, values in investment.items() if values["promising"]]
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_Fold2_deterministic_physical_endpoint_investment_gate",
    "truth_usage": "metrics and blend oracles only",
    "candidate_generation_uses_truth": False,
    "base": "V69 All773 coarse plus V76 adaptive dose",
    "rules": {
        "emission_only": "minimum gr_raw_cauchy_mean",
        "structure_guarded": "minimum structural roughness inside best-emission quartile",
        "balanced_rank": "minimum equal-rank emission plus structural roughness",
    },
    "audits": all_audits,
    "investment_gate": investment,
    "promising_rules": promising,
    "strict_pass": bool(promising),
    "fold1_touched": False,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> train a low-complexity cross-fit consumer for the promising physical endpoint"
        if promising else "FAIL -> physical evidence-separated selectors add no reachable value; close this bank"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV94 ORACLE AUDIT")
print(pd.DataFrame([
    {"fold": int(fold), **values}
    for fold, rules in all_audits.items()
    for values in rules.values()
]).to_string(index=False), flush=True)
print("\nV94 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
