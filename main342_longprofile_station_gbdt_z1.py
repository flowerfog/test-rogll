"""M342-Z1: add multiscale GR correlation-profile features to station GBDT.

This returns to the point-feature route after X2 established genuine 256-row
observability. Fold4/5 train matched control and profile-augmented residual
models. Fold3 selects a conservative dose and tests incremental value over the
matched control plus a typewell-shuffled feature control. Fold2/1 are untouched.
"""

from pathlib import Path
import json
import sys
import time
import types

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
U1_SOURCE = ROOT / "main342_multicut_controlpoint_endpoint_u1.py"
X1_SOURCE = ROOT / "main342_multiscale_rank_consistency_x1.py"
OUT = ROOT / "oof_exports" / "main342_longprofile_station_gbdt_z1"
MODEL_DIR = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = (4, 5)
VALID_FOLD = 3
SCALES = (64, 128, 256)
OFFSETS = np.arange(-128.0, 128.0 + 1e-6, 4.0, dtype=np.float64)
STATIONS_PER_WELL = 64
DOSES = (0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
MAX_P95_CHANGE = 0.20


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _bootstrap(path, marker, name):
    source = path.read_text(encoding="utf-8")
    if marker not in source:
        raise RuntimeError(f"Execution marker missing from {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


U = _bootstrap(
    U1_SOURCE,
    '\nbase, manifest, base_features, coverage = O["_prepare_inventory"]()',
    "m342_z1_u1_definitions",
)
X = _bootstrap(
    X1_SOURCE,
    "fold3_wells = sorted(",
    "m342_z1_x1_definitions",
)
P, B, O = U["P"], U["B"], U["O"]
P["OUT"] = OUT
P["MODEL_DIR"] = MODEL_DIR
B["OUT_DIR"] = OUT
B["MODEL_DIR"] = MODEL_DIR
H = X["H"]
DATA_ROOT = X["DATA_ROOT"]


PROFILE_FEATURES = []
for scale in SCALES:
    PROFILE_FEATURES.extend([
        f"grprof_winner_{scale}",
        f"grprof_mean_{scale}",
        f"grprof_std_{scale}",
        f"grprof_entropy_{scale}",
        f"grprof_margin_{scale}",
        f"grprof_zero_{scale}",
    ])
PROFILE_FEATURES.extend([
    "grprof_winner_scale_std",
    "grprof_mean_scale_std",
    "grprof_long_short_delta",
])


def _subsample(frame):
    pieces = []
    for _, local in frame.groupby("well", sort=False):
        local = local.sort_values("row_idx")
        if len(local) > STATIONS_PER_WELL:
            take = np.unique(
                np.round(np.linspace(0, len(local) - 1, STATIONS_PER_WELL)).astype(int)
            )
            local = local.iloc[take]
        pieces.append(local)
    return pd.concat(pieces, ignore_index=True)


def _soft_summary(score, valid):
    local = score[valid]
    centre = float(np.median(local))
    scale = float(np.median(np.abs(local - centre)) * 1.4826)
    if scale < 1e-6:
        scale = max(float(np.std(local)), 1e-6)
    z = np.full(len(score), -np.inf, np.float64)
    z[valid] = (local - centre) / scale
    probability = np.zeros(len(score), np.float64)
    probability[valid] = np.exp(np.clip(z[valid] - np.max(z[valid]), -50.0, 0.0))
    probability /= np.sum(probability)
    order = np.argsort(-z, kind="stable")
    mean = float(np.sum(probability * OFFSETS))
    std = float(np.sqrt(np.sum(probability * np.square(OFFSETS - mean))))
    entropy = float(
        -np.sum(probability[valid] * np.log(np.maximum(probability[valid], 1e-12)))
        / np.log(max(int(valid.sum()), 2))
    )
    zero = int(np.argmin(np.abs(OFFSETS)))
    return {
        "winner": float(OFFSETS[order[0]]),
        "mean": mean,
        "std": std,
        "entropy": entropy,
        "margin": float(z[order[0]] - z[order[1]]),
        "zero": float(z[zero]) if valid[zero] else np.nan,
    }


def _well_profiles(well, station_rows, endpoint_rows, typewell_override=None):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    type_name = typewell_override or well
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{type_name}__typewell.csv")
    base_path = (
        pd.to_numeric(horizontal["TVT_input"], errors="coerce")
        .to_numpy(np.float64)
        .copy()
    )
    endpoint_index = endpoint_rows["id"].astype(str).str.rsplit("_", n=1).str[1].astype(int)
    base_path[endpoint_index] = endpoint_rows["k2_parent"].to_numpy(np.float64)
    base_path = H["_fill"](base_path)
    if base_path is None:
        raise RuntimeError(f"{well}: cannot construct K2 path")
    hgr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    tw_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64)
    tw_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64)
    good = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    if good.sum() < 64:
        raise RuntimeError(f"{type_name}: too few typewell rows")
    order = np.argsort(tw_tvt[good])
    tw_tvt, tw_gr = tw_tvt[good][order], tw_gr[good][order]

    records = []
    for item in station_rows.itertuples(index=False):
        centre = int(item.row_idx)
        result = {"id": str(item.id)}
        winners, means = [], []
        for scale in SCALES:
            left = max(0, centre - scale)
            right = min(len(base_path), centre + scale + 1)
            observed = hgr[left:right]
            if len(observed) < scale or np.isfinite(observed).mean() < 0.35:
                summary = {key: np.nan for key in ("winner", "mean", "std", "entropy", "margin", "zero")}
            else:
                observed = H["_fill"](observed)
                candidate_tvt = base_path[left:right][None, :] + OFFSETS[:, None]
                candidate_gr = np.interp(
                    candidate_tvt.ravel(), tw_tvt, tw_gr, left=np.nan, right=np.nan
                ).reshape(len(OFFSETS), -1)
                valid = np.isfinite(candidate_gr).mean(axis=1) >= 0.95
                if observed is None or valid.sum() < 32:
                    summary = {key: np.nan for key in ("winner", "mean", "std", "entropy", "margin", "zero")}
                else:
                    medians = np.nanmedian(candidate_gr, axis=1, keepdims=True)
                    candidate_gr = np.where(np.isfinite(candidate_gr), candidate_gr, medians)
                    score = np.asarray(H["_components"](observed, candidate_gr)["amp"], np.float64)
                    summary = _soft_summary(score, valid)
            for key, value in summary.items():
                result[f"grprof_{key}_{scale}"] = value
            winners.append(summary["winner"])
            means.append(summary["mean"])
        result["grprof_winner_scale_std"] = float(np.nanstd(winners))
        result["grprof_mean_scale_std"] = float(np.nanstd(means))
        result["grprof_long_short_delta"] = float(means[-1] - means[0])
        records.append(result)
    return pd.DataFrame(records)


def _attach_profiles(stations, endpoints, shuffle_map=None, label="aligned"):
    stations = stations.copy()
    endpoints = endpoints.copy()
    stations["well"] = stations["well"].astype(str).str.lower()
    endpoints["well"] = endpoints["well"].astype(str).str.lower()
    if shuffle_map:
        shuffle_map = {
            str(key).lower(): str(value).lower()
            for key, value in shuffle_map.items()
        }
    output, failures = [], []
    endpoint_groups = {well: local for well, local in endpoints.groupby("well", sort=False)}
    groups = list(stations.groupby("well", sort=False))
    for number, (well, local) in enumerate(groups, 1):
        well = str(well).lower()
        try:
            override = shuffle_map.get(well) if shuffle_map else None
            if well not in endpoint_groups:
                raise KeyError(f"{well}: missing endpoint group")
            profile = _well_profiles(well, local, endpoint_groups[well], override)
            merged = local.merge(profile, on="id", how="left", validate="one_to_one")
            output.append(merged)
        except Exception as exc:
            failures.append({"well": str(well), "control": label, "reason": repr(exc)})
        if number % 25 == 0 or number == len(groups):
            print(
                f"M342-Z1 {label} {number}/{len(groups)} failures={len(failures)}",
                flush=True,
            )
    if not output:
        diagnostic = pd.DataFrame(failures)
        diagnostic.to_csv(OUT / f"{label}_profile_failures.csv", index=False)
        raise RuntimeError(
            f"No Z1 profile rows for {label}. Diagnostics:\n"
            + diagnostic.head(12).to_string(index=False)
        )
    return pd.concat(output, ignore_index=True), failures


def _scan(full, correction, label):
    parent = full["k2_parent"].to_numpy(np.float64)
    base_metric = B["_metrics"](full, parent, "k2_parent")
    rows = []
    for dose in DOSES:
        metric = B["_metrics"](full, parent + dose * correction, label)
        rows.append({
            "dose": dose,
            "rmse": metric["rmse"],
            "gain": base_metric["rmse"] - metric["rmse"],
            "far_gain": base_metric["far_gt2000_rmse"] - metric["far_gt2000_rmse"],
            "p95_change": metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
        })
    return base_metric, pd.DataFrame(rows)


base, manifest, base_features, coverage = O["_prepare_inventory"]()
endpoints = P["_endpoint_table"]()
profile_endpoints = endpoints.copy()
profile_endpoints["well"] = (
    profile_endpoints["id"].astype(str).str.rsplit("_", n=1).str[0]
)
control_features = list(dict.fromkeys(base_features + list(P["ENGINEERED"])))
profile_features = control_features + PROFILE_FEATURES
official_manifest = manifest.loc[manifest["tag"].eq("official")].copy()
official_manifest["cut_bin"] = 0
official_manifest["states_in_bin"] = 1

train_wells = sorted(base.loc[base["fold"].isin(TRAIN_FOLDS), "well"].astype(str).unique())
valid_wells = sorted(base.loc[base["fold"].eq(VALID_FOLD), "well"].astype(str).unique())
train, _ = U["_aggregate_stations"](train_wells, manifest, base_features, endpoints)
valid, _ = U["_aggregate_stations"](valid_wells, manifest, base_features, endpoints)
train, valid = _subsample(train), _subsample(valid)
train, failures_a = _attach_profiles(train, profile_endpoints, label="train")
valid, failures_b = _attach_profiles(valid, profile_endpoints, label="aligned")
valid_success = sorted(valid["well"].astype(str).unique())
valid_full = U["_held_full"](valid_success, official_manifest, base, endpoints)

control_model, control_iteration = P["_fit"](
    train, valid, control_features, label="z1_matched_control"
)
profile_model, profile_iteration = P["_fit"](
    train, valid, profile_features, label="z1_profile"
)
control_correction = U["_interpolated_correction"](
    valid, valid_full, control_model, control_features
)
profile_correction = U["_interpolated_correction"](
    valid, valid_full, profile_model, profile_features
)
base_metric, profile_scan = _scan(valid_full, profile_correction, "profile")
_, control_scan = _scan(valid_full, control_correction, "control")
scan = profile_scan.merge(control_scan, on="dose", suffixes=("_profile", "_control"))
scan["increment_vs_control"] = scan["rmse_control"] - scan["rmse_profile"]
safe = scan[
    scan["dose"].gt(0.0)
    & scan["gain_profile"].ge(0.03)
    & scan["far_gain_profile"].ge(0.0)
    & scan["p95_change_profile"].le(MAX_P95_CHANGE)
    & scan["increment_vs_control"].ge(0.02)
]
selected = safe.sort_values("rmse_profile").iloc[0] if not safe.empty else scan.sort_values("rmse_profile").iloc[0]
dose = float(selected["dose"])

centres = {}
for well in valid_wells:
    table = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv", usecols=["TVT"])
    centres[well] = float(pd.to_numeric(table["TVT"], errors="coerce").median())
ordered = sorted(valid_wells, key=lambda well: (centres[well], well))
shuffle_map = dict(zip(ordered, ordered[1:] + ordered[:1]))
shuffled, failures_c = _attach_profiles(
    valid.drop(columns=PROFILE_FEATURES), profile_endpoints, shuffle_map, "typewell_shuffle"
)
common_wells = sorted(
    set(valid["well"].astype(str)).intersection(shuffled["well"].astype(str))
)
valid_common = valid.loc[valid["well"].astype(str).isin(common_wells)].copy()
shuffle_common = shuffled.loc[shuffled["well"].astype(str).isin(common_wells)].copy()
common_full = valid_full.loc[
    valid_full["well"].astype(str).isin(common_wells)
].copy()
aligned_common_correction = U["_interpolated_correction"](
    valid_common, common_full, profile_model, profile_features
)
shuffle_correction = U["_interpolated_correction"](
    shuffle_common, common_full, profile_model, profile_features
)
parent = valid_full["k2_parent"].to_numpy(np.float64)
aligned_metric = B["_metrics"](
    valid_full, parent + dose * profile_correction, "aligned_profile"
)
control_metric = B["_metrics"](
    valid_full, parent + dose * control_correction, "matched_control"
)
shuffle_metric = B["_metrics"](
    common_full,
    common_full["k2_parent"].to_numpy(np.float64) + dose * shuffle_correction,
    "typewell_shuffle",
)
aligned_common_metric = B["_metrics"](
    common_full,
    common_full["k2_parent"].to_numpy(np.float64) + dose * aligned_common_correction,
    "aligned_common",
)
reverse_metric = B["_metrics"](
    valid_full, parent - dose * profile_correction, "reverse_control"
)

gain = float(base_metric["rmse"] - aligned_metric["rmse"])
increment = float(control_metric["rmse"] - aligned_metric["rmse"])
shuffle_loss = float(shuffle_metric["rmse"] - aligned_common_metric["rmse"])
reverse_loss = float(reverse_metric["rmse"] - aligned_metric["rmse"])
checks = {
    "fold3_wells_ge_145": int(aligned_metric["wells"]) >= 145,
    "selected_positive_dose": dose > 0.0,
    "gain_ge_0p03": gain >= 0.03,
    "far_gain_ge_0": base_metric["far_gt2000_rmse"] - aligned_metric["far_gt2000_rmse"] >= 0.0,
    "increment_vs_control_ge_0p02": increment >= 0.02,
    "shuffle_loses_ge_0p02": shuffle_loss >= 0.02,
    "reverse_loses_ge_0p03": reverse_loss >= 0.03,
    "p95_change_le_0p20": aligned_metric["p95_well_rmse"] - base_metric["p95_well_rmse"] <= MAX_P95_CHANGE,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_longprofile_station_gbdt_z1",
    "stage": "fold45_station_training_fold3_profile_increment_gateway",
    "legal_inputs_only": True,
    "protocol": {
        "train_folds": TRAIN_FOLDS,
        "selection_fold": VALID_FOLD,
        "fold2_touched": False,
        "fold1_touched": False,
    },
    "features": {
        "control": len(control_features),
        "profile_added": len(PROFILE_FEATURES),
        "profile_names": PROFILE_FEATURES,
    },
    "models": {
        "control_iteration": control_iteration,
        "profile_iteration": profile_iteration,
        "stations_per_well": STATIONS_PER_WELL,
    },
    "selected": selected.to_dict(),
    "confirmation": {
        "base": base_metric,
        "profile": aligned_metric,
        "matched_control": control_metric,
        "aligned_common": aligned_common_metric,
        "typewell_shuffle": shuffle_metric,
        "reverse": reverse_metric,
        "gain": gain,
        "increment_vs_control": increment,
        "shuffle_loss": shuffle_loss,
        "reverse_loss": reverse_loss,
        "common_shuffle_wells": len(common_wells),
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "failures": {"train": len(failures_a), "aligned": len(failures_b), "shuffle": len(failures_c)},
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> refit on Fold3/4/5 and confirm once on Fold2"
        if strict_pass else
        "FAIL -> long-range correlation profiles add no transferable point-model information; close Z1"
    ),
}

scan.to_csv(OUT / "dose_scan.csv", index=False)
pd.DataFrame([
    base_metric, aligned_metric, control_metric, aligned_common_metric,
    shuffle_metric, reverse_metric,
]).to_csv(
    OUT / "summary.csv", index=False
)
pd.DataFrame(failures_a + failures_b + failures_c).to_csv(OUT / "failures.csv", index=False)
(OUT / "feature_names.json").write_text(json.dumps(profile_features, indent=2), encoding="utf-8")
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-Z1 DOSE SCAN")
print(scan.to_string(index=False), flush=True)
print("\nM342-Z1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
