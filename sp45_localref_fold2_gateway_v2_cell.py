"""Fold-2 paired SP45 replay with the frozen V11 local_050 reference.

Only the typewell GR reference changes.  PF dynamics, 128 deterministic seeds,
500 particles, Gaussian likelihood, all beam configurations, the deterministic
selector, and surface projection are inherited unchanged from the validated
SP45 replay.  If its OOF export is absent, the raw arm is regenerated with the
same deterministic seeds in this cell for an exact paired comparison.
"""

from __future__ import annotations

import ast
import json
import math
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "sp45_localref_fold2_gateway_v2"
OUT_DIR = OOF_ROOT / EXPERIMENT
RECORD_DIR = OUT_DIR / "records"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RECORD_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CANDIDATES = (
    ROOT / "sp45_foldclean_feature_export_v1_cell.py",
    Path.cwd() / "sp45_foldclean_feature_export_v1_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
SP45_DIR = OOF_ROOT / "sp45_foldclean_feature_export_v1"
SP45_OOF = SP45_DIR / "sp45_feature_oof.parquet"
SP45_MANIFEST = SP45_DIR / "sp45_provenance_manifest.json"
HAS_BASELINE_EXPORT = SP45_OOF.exists() and SP45_MANIFEST.exists()
V11_VERDICT = OOF_ROOT / "stride_typewell_reference_audit_v11" / "verdict.json"
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "folds.csv",
)
FOLD_PATH = next((path for path in FOLD_CANDIDATES if path.exists()), None)
for required in (V11_VERDICT,):
    if not required.exists():
        raise FileNotFoundError(required)
if FOLD_PATH is None:
    raise FileNotFoundError("Missing foldclean_full_fleongg_lgb_v0 fold metadata")

VALID_FOLD = 2
PF_SEEDS = 128
PF_PARTICLES = 500
N_JOBS = int(os.environ.get("SP45_LOCALREF_JOBS", str(min(8, os.cpu_count() or 1))))
FORCE = bool(int(os.environ.get("SP45_LOCALREF_FORCE", "0")))
REFERENCE_NAME = "local_050"
REFERENCE_DOSE = 0.50
RAW_COLUMNS = (
    "pf_scale_3",
    "pf_scale_5",
    "pf_scale_8",
    "pf_scale_12",
    "pf_mean",
    "beam_mean",
    "sp45_selector_raw",
    "sp45_selector_projected",
)

PF_SCALES = (3.0, 5.0, 8.0, 12.0)
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}
SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10, 8.0, 64.0, 2),
    (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2),
    (10, 50.0, 400.0, 0),
]


def _find_data_root():
    candidates = (
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        ROOT / "dataset",
        ROOT / "rogii-wellbore-geology-prediction",
        Path.cwd(),
    )
    for path in candidates:
        if (path / "train").exists() and any((path / "train").glob("*__horizontal_well.csv")):
            return path
    for train_dir in ROOT.glob("**/train"):
        if any(train_dir.glob("*__horizontal_well.csv")):
            return train_dir.parent
    raise FileNotFoundError("Cannot locate the ROGII raw data root")


def _embedded_prefix_setup(hw, tw):
    required_hw = {"MD", "Z", "GR", "TVT", "TVT_input"}
    required_tw = {"TVT", "GR"}
    if not required_hw.issubset(hw.columns) or not required_tw.issubset(tw.columns):
        raise RuntimeError(
            f"Missing raw columns: hw={sorted(required_hw - set(hw))}, "
            f"tw={sorted(required_tw - set(tw))}"
        )
    known_mask = hw["TVT_input"].notna().to_numpy()
    eval_mask = hw["TVT_input"].isna().to_numpy() & hw["TVT"].notna().to_numpy()
    known_index = np.flatnonzero(known_mask)
    eval_index = np.flatnonzero(eval_mask)
    if len(known_index) < 5 or len(eval_index) < 50:
        raise RuntimeError(f"known/eval rows too small: {len(known_index)}/{len(eval_index)}")
    if known_index[-1] >= eval_index[0]:
        raise RuntimeError("TVT_input mask is not a prefix mask")

    tw = tw.sort_values("TVT")
    tw_tvt = pd.to_numeric(tw["TVT"], errors="coerce").to_numpy(np.float64)
    tw_gr = pd.to_numeric(tw["GR"], errors="coerce").to_numpy(np.float64)
    finite_tw = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    tw_tvt = tw_tvt[finite_tw]
    tw_gr = tw_gr[finite_tw]
    order = np.argsort(tw_tvt)
    tw_tvt = tw_tvt[order]
    tw_gr = tw_gr[order]
    unique = np.r_[True, np.diff(tw_tvt) > 1e-9]
    tw_tvt = tw_tvt[unique]
    tw_gr = tw_gr[unique]
    if len(tw_tvt) < 32:
        raise RuntimeError(f"only {len(tw_tvt)} finite unique typewell rows")

    gr_all = (
        pd.to_numeric(hw["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(float(np.mean(tw_gr)))
        .to_numpy(np.float64)
    )
    md = pd.to_numeric(hw["MD"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(hw["Z"], errors="coerce").to_numpy(np.float64)
    truth = pd.to_numeric(hw["TVT"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(hw["TVT_input"], errors="coerce").to_numpy(np.float64)
    last = known_index[-1]
    last_tvt = float(tvt_input[last])
    last_z = float(z[last])
    last_md = float(md[last])
    tw_at_known = np.interp(tvt_input[known_index], tw_tvt, tw_gr)
    gr_scale = float(np.clip(np.nanstd(gr_all[known_index] - tw_at_known), 10.0, 60.0))
    tail = known_index[-30:]
    dt = np.diff(tvt_input[tail])
    dz = np.diff(z[tail])
    dm = np.diff(md[tail])
    valid_rate = np.isfinite(dt + dz + dm) & (dm > 0)
    initial_rate = (
        float(np.median((dt[valid_rate] + dz[valid_rate]) / dm[valid_rate]))
        if valid_rate.sum() >= 3
        else 0.0
    )
    return {
        "eval_index": eval_index.astype(np.int32),
        "md_eval": md[eval_index],
        "z_eval": z[eval_index],
        "gr_eval": gr_all[eval_index],
        "truth_eval": truth[eval_index],
        "tw_tvt": tw_tvt,
        "tw_gr": tw_gr,
        "last_tvt": last_tvt,
        "last_z": last_z,
        "last_md": last_md,
        "initial_rate": initial_rate,
        "gr_scale": gr_scale,
        "z_span": float(np.nanmax(z[eval_index]) - np.nanmin(z[eval_index])),
    }


def _embedded_run_pf_seed(payload, seed):
    md_eval = payload["md_eval"]
    z_eval = payload["z_eval"]
    gr_eval = payload["gr_eval"]
    tw_tvt = payload["tw_tvt"]
    tw_gr = payload["tw_gr"]
    previous_md = float(payload["last_md"])
    rng = np.random.default_rng(seed)
    position = float(payload["last_tvt"] + payload["last_z"]) + 4.5 * rng.standard_normal(PF_PARTICLES)
    rate = float(payload["initial_rate"]) + 0.01 * rng.standard_normal(PF_PARTICLES)
    weight = np.full(PF_PARTICLES, 1.0 / PF_PARTICLES, dtype=np.float64)
    result = np.empty(len(md_eval), dtype=np.float32)
    log_likelihood = 0.0
    for index in range(len(md_eval)):
        md_step = max(float(md_eval[index]) - previous_md, 1.0)
        rate = 0.998 * rate + 0.002 * rng.standard_normal(PF_PARTICLES)
        position = position + rate * md_step + 0.005 * rng.standard_normal(PF_PARTICLES)
        tvt_particle = np.clip(
            position - float(z_eval[index]), tw_tvt[0] - 100.0, tw_tvt[-1] + 100.0
        )
        position = tvt_particle + float(z_eval[index])
        expected_gr = np.interp(tvt_particle, tw_tvt, tw_gr)
        residual = (float(gr_eval[index]) - expected_gr) / float(payload["gr_scale"])
        likelihood = np.exp(-0.5 * np.minimum(residual * residual, 600.0))
        likelihood = np.maximum(likelihood, 1e-300)
        log_likelihood += math.log(max(float(np.dot(weight, likelihood)), 1e-300))
        weight *= likelihood
        total = float(weight.sum())
        weight = weight / total if total > 0 else np.full(PF_PARTICLES, 1.0 / PF_PARTICLES)
        effective = 1.0 / float(np.dot(weight, weight))
        if effective < 0.5 * PF_PARTICLES:
            cumulative = np.cumsum(weight)
            start = rng.uniform(0.0, 1.0 / PF_PARTICLES)
            chosen = np.clip(
                np.searchsorted(cumulative, start + np.arange(PF_PARTICLES) / PF_PARTICLES),
                0,
                PF_PARTICLES - 1,
            )
            position = position[chosen] + 0.1 * rng.standard_normal(PF_PARTICLES)
            rate = rate[chosen] + 0.001 * rng.standard_normal(PF_PARTICLES)
            weight = np.full(PF_PARTICLES, 1.0 / PF_PARTICLES)
        result[index] = float(np.dot(weight, position - float(z_eval[index])))
        previous_md = float(md_eval[index])
    return result, log_likelihood


def _embedded_pf_bundle(payload):
    members = []
    likelihoods = np.empty(PF_SEEDS, dtype=np.float64)
    for seed in range(PF_SEEDS):
        prediction, likelihood = _embedded_run_pf_seed(payload, seed)
        members.append(prediction)
        likelihoods[seed] = likelihood
    members = np.stack(members).astype(np.float32)
    shifted = likelihoods - np.max(likelihoods)
    output = {}
    for scale in PF_SCALES:
        weights = np.exp(shifted / scale)
        weights /= weights.sum()
        output[f"pf_scale_{int(scale)}"] = np.sum(
            members * weights[:, None], axis=0, dtype=np.float64
        ).astype(np.float32)
    output["pf_mean"] = members.mean(axis=0).astype(np.float32)
    return output


def _embedded_beam_search(horizontal_gr, typewell_tvt, typewell_gr, last_tvt,
                          beam_size, move_cost, emission_scale, radius):
    count = len(horizontal_gr)
    if radius > 0 and count > max(3, 2 * radius + 1):
        window = min(2 * radius + 1, count if count % 2 else count - 1)
        smoothed = savgol_filter(horizontal_gr, window, min(2, window - 1))
    else:
        smoothed = horizontal_gr.copy()
    start_index = int(np.argmin(np.abs(typewell_tvt - last_tvt)))
    moves = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    penalties = move_cost * np.array([2.0, 1.0, 0.0, 1.0, 2.0])
    beam_index = np.full(beam_size, start_index, dtype=np.int64)
    beam_cost = np.full(beam_size, np.inf, dtype=np.float64)
    beam_cost[0] = 0.0
    active = 1
    result = np.empty(count, dtype=np.float32)
    for step, gr_value in enumerate(smoothed):
        proposed = beam_index[:active, None] + moves[None, :]
        clipped = np.clip(proposed, 0, len(typewell_tvt) - 1)
        valid = (proposed >= 0) & (proposed < len(typewell_tvt))
        emission = (gr_value - typewell_gr[clipped]) ** 2 / emission_scale
        cost = beam_cost[:active, None] + emission + penalties[None, :]
        cost = np.where(valid, cost, np.inf)
        proposed = proposed.ravel()[valid.ravel()]
        cost = cost.ravel()[valid.ravel()]
        order = np.argsort(cost)
        proposed = proposed[order]
        cost = cost[order]
        _, first = np.unique(proposed, return_index=True)
        proposed = proposed[first]
        cost = cost[first]
        kept = min(beam_size, len(proposed))
        top = np.argpartition(cost, kept - 1)[:kept]
        top = top[np.argsort(cost[top])]
        beam_index[:kept] = proposed[top]
        beam_cost[:kept] = cost[top]
        if kept < beam_size:
            beam_index[kept:] = beam_index[kept - 1]
            beam_cost[kept:] = np.inf
        active = kept
        result[step] = typewell_tvt[beam_index[0]]
    return result


def _embedded_beam_bundle(payload):
    stack = np.stack(
        [
            _embedded_beam_search(
                payload["gr_eval"], payload["tw_tvt"], payload["tw_gr"],
                float(payload["last_tvt"]), *config
            )
            for config in BEAM_CONFIGS
        ]
    ).astype(np.float32)
    return {"beam_mean": stack.mean(axis=0).astype(np.float32)}


def _embedded_selector_code(eval_rows, z_span):
    n_bin = int(eval_rows > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right"))
    code = n_bin + 2 * z_bin
    return code, SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)


def _embedded_apply_selector(name, pf, beam_mean, last_tvt):
    parts = name.split("_")
    scale = int(float(parts[2]))
    beam_weight = float(parts[parts.index("beam") + 1]) if "beam" in parts else 0.0
    hold_weight = float(parts[parts.index("hold") + 1]) if "hold" in parts else 0.0
    base = np.asarray(pf[f"pf_scale_{scale}"], dtype=np.float32)
    prediction = (1.0 - beam_weight) * base + beam_weight * beam_mean
    return ((1.0 - hold_weight) * prediction + hold_weight * last_tvt).astype(np.float32)


def _embedded_project_surface(prediction, md_eval, z_eval, last_md, anchor_surface):
    progress = (md_eval - last_md) / max(float(md_eval[-1] - last_md), 1e-6)
    surface_delta = prediction.astype(np.float64) + z_eval - anchor_surface
    if len(progress) < 6:
        return prediction.astype(np.float32)
    coefficient = np.polyfit(progress, surface_delta, 4)
    for _ in range(4):
        residual = surface_delta - np.polyval(coefficient, progress)
        scale = np.median(np.abs(residual)) * 1.4826 + 1e-6
        weights = 1.0 / (1.0 + (residual / (2.0 * scale)) ** 2)
        coefficient = np.polyfit(progress, surface_delta, 4, w=weights)
    fitted = anchor_surface + np.polyval(coefficient, progress) - z_eval
    return (0.25 * prediction + 0.75 * fitted).astype(np.float32)


def _bootstrap(path):
    if path is None:
        return types.SimpleNamespace(
            _prefix_setup=_embedded_prefix_setup,
            _pf_bundle=_embedded_pf_bundle,
            _beam_bundle=_embedded_beam_bundle,
            _selector_code=_embedded_selector_code,
            _apply_selector=_embedded_apply_selector,
            _project_surface=_embedded_project_surface,
            BEAM_CONFIGS=BEAM_CONFIGS,
        )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    name = "sp45_replay_definitions_for_localref_v2"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    module.PF_SEEDS = PF_SEEDS
    module.PF_PARTICLES = PF_PARTICLES
    module.PF_SCALES = (3.0, 5.0, 8.0, 12.0)
    module.SELECTOR_N_EVAL_THRESHOLD = 4840.0
    module.SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
    module.SELECTOR_BIN_VARIANTS = {
        0: "pf_scale_5_hold_0.2",
        1: "pf_scale_3_hold_0.15",
        2: "pf_scale_12_beam_0.2_hold_0.15",
        3: "pf_scale_5_hold_0.15",
        4: "pf_scale_5_beam_0.05_hold_0.05",
        5: "pf_scale_12_beam_0.2_hold_0.05",
    }
    module.SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"
    module.BEAM_CONFIGS = [
        (10, 20.0, 144.0, 2),
        (10, 8.0, 64.0, 2),
        (8, 35.0, 220.0, 1),
        (10, 14.0, 90.0, 5),
        (20, 4.0, 36.0, 3),
        (12, 12.0, 100.0, 3),
        (15, 25.0, 180.0, 2),
        (20, 30.0, 200.0, 2),
        (15, 10.0, 80.0, 4),
        (25, 6.0, 50.0, 3),
        (10, 40.0, 300.0, 1),
        (12, 18.0, 120.0, 5),
        (30, 8.0, 70.0, 2),
        (10, 50.0, 400.0, 0),
    ]
    return module


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _local_reference(hw, tw_tvt, tw_gr, dose=REFERENCE_DOSE):
    known = hw["TVT_input"].notna().to_numpy() & hw["GR"].notna().to_numpy()
    tvt = pd.to_numeric(hw.loc[known, "TVT_input"], errors="coerce").to_numpy(np.float64)
    observed = pd.to_numeric(hw.loc[known, "GR"], errors="coerce").to_numpy(np.float64)
    expected = np.interp(tvt, tw_tvt, tw_gr)
    valid = np.isfinite(tvt) & np.isfinite(observed) & np.isfinite(expected)
    tvt = tvt[valid]
    residual = (observed - expected)[valid]
    if len(tvt) < 32:
        return tw_gr.copy()

    order = np.argsort(tvt, kind="stable")
    work = pd.DataFrame({"tvt": tvt[order], "residual": residual[order]})
    work["bin"] = np.round(work["tvt"] * 2.0) / 2.0
    work = work.groupby("bin", as_index=False)["residual"].median().sort_values("bin")
    window = min(31, len(work))
    if window % 2 == 0:
        window -= 1
    if window >= 5:
        work["residual"] = (
            work["residual"]
            .rolling(window, center=True, min_periods=max(3, window // 3))
            .median()
            .interpolate(limit_direction="both")
        )
    correction = np.interp(
        tw_tvt,
        work["bin"].to_numpy(np.float64),
        work["residual"].to_numpy(np.float64),
    )
    return tw_gr + float(dose) * correction


def _record_path(well):
    return RECORD_DIR / f"{well}.npz"


def _record_valid(path):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            base_valid = (
                int(data["pf_seeds"][0]) == PF_SEEDS
                and int(data["pf_particles"][0]) == PF_PARTICLES
                and str(data["reference"][0]) == REFERENCE_NAME
                and len(data["row_idx"]) >= 50
            )
            if not HAS_BASELINE_EXPORT:
                base_valid = base_valid and all(
                    f"raw_{column}" in data.files for column in RAW_COLUMNS
                )
            return base_valid
    except Exception:
        return False


V = _bootstrap(SOURCE_PATH)
DATA_ROOT = _find_data_root()
v11 = json.loads(V11_VERDICT.read_text(encoding="utf-8"))
if not v11.get("strict_pass", False) or v11["selection"]["selected_recipe"] != REFERENCE_NAME:
    raise RuntimeError("V11 local_050 was not frozen by a passing visible-prefix audit")
if HAS_BASELINE_EXPORT:
    manifest = json.loads(SP45_MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("strict_pass", False):
        raise RuntimeError("SP45 baseline did not pass its full-run and historical-parity guards")
    if not manifest.get("checks", {}).get("full_128seed_500particle_run", False):
        raise RuntimeError("SP45 baseline is not the full 128-seed/500-particle replay")

folds = pd.read_csv(FOLD_PATH, dtype={"well": str})
well_column = next(column for column in ("well", "well_id", "Well") if column in folds)
fold_column = next(column for column in ("fold", "outer_fold", "Fold") if column in folds)
folds = folds[[well_column, fold_column]].copy()
folds.columns = ["well", "fold"]
folds["well"] = folds["well"].astype(str)
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
valid_wells = sorted(folds.loc[folds["fold"].eq(VALID_FOLD), "well"].drop_duplicates())
if len(valid_wells) != 155:
    raise RuntimeError(f"Unexpected Fold2 size: {len(valid_wells)}")


def _process_well(well):
    destination = _record_path(well)
    if not FORCE and _record_valid(destination):
        return {"well": well, "status": "cached"}
    try:
        hw = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
        tw = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
        payload = V._prefix_setup(hw, tw)
        raw_outputs = {}
        if not HAS_BASELINE_EXPORT:
            raw_pf = V._pf_bundle(payload)
            raw_beam = V._beam_bundle(payload)
            raw_selector_code, raw_selector_variant = V._selector_code(
                len(payload["eval_index"]), float(payload["z_span"])
            )
            raw_selector = V._apply_selector(
                raw_selector_variant,
                raw_pf,
                raw_beam["beam_mean"],
                float(payload["last_tvt"]),
            )
            raw_projected = V._project_surface(
                raw_selector,
                payload["md_eval"],
                payload["z_eval"],
                float(payload["last_md"]),
                float(payload["last_tvt"] + payload["last_z"]),
            )
            raw_outputs = {
                "raw_pf_scale_3": raw_pf["pf_scale_3"],
                "raw_pf_scale_5": raw_pf["pf_scale_5"],
                "raw_pf_scale_8": raw_pf["pf_scale_8"],
                "raw_pf_scale_12": raw_pf["pf_scale_12"],
                "raw_pf_mean": raw_pf["pf_mean"],
                "raw_beam_mean": raw_beam["beam_mean"],
                "raw_sp45_selector_raw": raw_selector,
                "raw_sp45_selector_projected": raw_projected,
            }
        # _prefix_setup freezes the raw-reference gamma before this replacement.
        local_payload = dict(payload)
        local_payload["tw_gr"] = _local_reference(
            hw, payload["tw_tvt"], payload["tw_gr"]
        )
        pf = V._pf_bundle(local_payload)
        beam = V._beam_bundle(local_payload)
        selector_code, selector_variant = V._selector_code(
            len(payload["eval_index"]), float(payload["z_span"])
        )
        selector_raw = V._apply_selector(
            selector_variant, pf, beam["beam_mean"], float(payload["last_tvt"])
        )
        selector_projected = V._project_surface(
            selector_raw,
            payload["md_eval"],
            payload["z_eval"],
            float(payload["last_md"]),
            float(payload["last_tvt"] + payload["last_z"]),
        )
        save = {
            "well": np.asarray([well]),
            "row_idx": payload["eval_index"].astype(np.int32),
            "truth": payload["truth_eval"].astype(np.float32),
            "md": payload["md_eval"].astype(np.float32),
            "last_md": np.asarray([payload["last_md"]], np.float32),
            "pf_seeds": np.asarray([PF_SEEDS], np.int16),
            "pf_particles": np.asarray([PF_PARTICLES], np.int16),
            "reference": np.asarray([REFERENCE_NAME]),
            "selector_variant": np.asarray([selector_variant]),
            "sp45_selector_raw": selector_raw.astype(np.float32),
            "sp45_selector_projected": selector_projected.astype(np.float32),
            "beam_mean": beam["beam_mean"].astype(np.float32),
        }
        for column in ("pf_scale_3", "pf_scale_5", "pf_scale_8", "pf_scale_12", "pf_mean"):
            save[column] = np.asarray(pf[column], np.float32)
        for column, values in raw_outputs.items():
            save[column] = np.asarray(values, np.float32)
        np.savez_compressed(destination, **save)
        return {"well": well, "status": "built", "rows": len(payload["eval_index"])}
    except Exception as exc:
        return {"well": well, "status": "failed", "reason": repr(exc)}


print(
    "SP45 LOCALREF CONFIG",
    {
        "fold": VALID_FOLD,
        "wells": len(valid_wells),
        "seeds": PF_SEEDS,
        "particles": PF_PARTICLES,
        "jobs": N_JOBS,
        "reference": REFERENCE_NAME,
        "dose": REFERENCE_DOSE,
        "raw_gamma_held_fixed": True,
        "baseline_source": (
            "validated full SP45 OOF export"
            if HAS_BASELINE_EXPORT
            else "paired raw replay with identical seeds"
        ),
    },
    flush=True,
)

reports = []
batch_size = max(2 * N_JOBS, 1)
for start in range(0, len(valid_wells), batch_size):
    batch = valid_wells[start : start + batch_size]
    reports.extend(Parallel(n_jobs=N_JOBS, backend="loky")(delayed(_process_well)(well) for well in batch))
    done = min(start + len(batch), len(valid_wells))
    failures = sum(report["status"] == "failed" for report in reports)
    print(f"SP45 localref Fold2 {done}/{len(valid_wells)} failures={failures}", flush=True)

build_report = pd.DataFrame(reports)
build_report.to_csv(OUT_DIR / "build_report.csv", index=False)
failures = build_report.loc[build_report["status"].eq("failed")]
if len(failures):
    raise RuntimeError(f"{len(failures)} wells failed: {failures.head(10).to_dict(orient='records')}")

parts = []
for well in valid_wells:
    with np.load(_record_path(well), allow_pickle=False) as data:
        count = len(data["row_idx"])
        rows = data["row_idx"].astype(np.int32)
        frame = pd.DataFrame(
            {
                "id": [f"{well}_{int(row)}" for row in rows],
                "well": well,
                "row_idx": rows,
                "truth": data["truth"].astype(np.float32),
                "md_since": (data["md"] - float(data["last_md"][0])).astype(np.float32),
            }
        )
        for column in RAW_COLUMNS:
            frame[f"local_{column}"] = data[column].astype(np.float32)
            if not HAS_BASELINE_EXPORT:
                frame[column] = data[f"raw_{column}"].astype(np.float32)
        parts.append(frame)
local = pd.concat(parts, ignore_index=True)

if HAS_BASELINE_EXPORT:
    raw = pd.read_parquet(SP45_OOF, columns=["id", "well", "fold"] + list(RAW_COLUMNS))
    raw = raw.loc[raw["fold"].eq(VALID_FOLD)].drop(columns="fold")
    frame = local.merge(raw, on=["id", "well"], how="inner", validate="one_to_one")
    if len(frame) != len(local) or frame["well"].nunique() != 155:
        raise RuntimeError(
            f"Baseline alignment failed: {len(frame)}/{len(local)} rows, "
            f"{frame['well'].nunique()} wells"
        )
else:
    frame = local


def _metrics(column):
    error = frame[column].to_numpy(np.float64) - frame["truth"].to_numpy(np.float64)
    work = pd.DataFrame({"well": frame["well"], "error2": error * error})
    bywell = np.sqrt(work.groupby("well", sort=False)["error2"].mean())
    output = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(bywell.mean()),
        "median_well_rmse": float(bywell.median()),
        "p95_well_rmse": float(bywell.quantile(0.95)),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    for name, low, high in (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    ):
        mask = (distance >= low) & (distance < high)
        output[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return output


def _positive_well_fraction(raw_column, local_column):
    raw_error2 = (frame[raw_column].to_numpy(np.float64) - frame["truth"].to_numpy(np.float64)) ** 2
    local_error2 = (frame[local_column].to_numpy(np.float64) - frame["truth"].to_numpy(np.float64)) ** 2
    work = pd.DataFrame({"well": frame["well"], "raw": raw_error2, "local": local_error2})
    bywell = work.groupby("well", sort=False)[["raw", "local"]].mean()
    return float((bywell["local"] < bywell["raw"]).mean())


summary_rows = []
for raw_column in RAW_COLUMNS:
    local_column = f"local_{raw_column}"
    raw_metrics = _metrics(raw_column)
    local_metrics = _metrics(local_column)
    summary_rows.append(
        {
            "endpoint": raw_column,
            "raw_rmse": raw_metrics["rmse"],
            "local_rmse": local_metrics["rmse"],
            "gain": raw_metrics["rmse"] - local_metrics["rmse"],
            "raw_mean_well_rmse": raw_metrics["mean_well_rmse"],
            "local_mean_well_rmse": local_metrics["mean_well_rmse"],
            "positive_well_fraction": _positive_well_fraction(raw_column, local_column),
            "near_gain": raw_metrics["near_0_500_rmse"] - local_metrics["near_0_500_rmse"],
            "far_gain": raw_metrics["far_gt2000_rmse"] - local_metrics["far_gt2000_rmse"],
        }
    )
summary = pd.DataFrame(summary_rows).sort_values("gain", ascending=False)
summary.to_csv(OUT_DIR / "endpoint_comparison.csv", index=False)

primary = "sp45_selector_projected"
raw_primary = _metrics(primary)
local_primary = _metrics(f"local_{primary}")
primary_row = summary.loc[summary["endpoint"].eq(primary)].iloc[0]
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold2_paired_local050_reference_inside_SP45_likelihood",
    "sample_unit": "complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "validation_fold": VALID_FOLD,
        "reference_recipe": REFERENCE_NAME,
        "recipe_selected_by": "V11 held-out visible-prefix GR only",
        "raw_endpoint": (
            str(SP45_OOF) if HAS_BASELINE_EXPORT else "paired raw replay in this experiment"
        ),
        "truth_usage": "metrics only after local-reference predictions were frozen",
    },
    "engine": {
        "pf_seeds": PF_SEEDS,
        "pf_particles": PF_PARTICLES,
        "pf_likelihood": "original SP45 Gaussian",
        "beam_configs": len(V.BEAM_CONFIGS),
        "raw_gamma_held_fixed": True,
        "selector_and_projection_unchanged": True,
    },
    "comparison": summary.to_dict(orient="records"),
    "primary": {
        "raw": raw_primary,
        "local_050": local_primary,
        "gain": float(primary_row["gain"]),
        "positive_well_fraction": float(primary_row["positive_well_fraction"]),
        "near_gain": float(primary_row["near_gain"]),
        "far_gain": float(primary_row["far_gain"]),
    },
    "gates": {
        "primary_gain_ge_0p10": bool(primary_row["gain"] >= 0.10),
        "positive_wells_ge_0p55": bool(primary_row["positive_well_fraction"] >= 0.55),
        "near_degradation_le_0p10": bool(primary_row["near_gain"] >= -0.10),
        "far_gain_ge_0p10": bool(primary_row["far_gain"] >= 0.10),
    },
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
}
verdict["strict_pass"] = bool(all(verdict["gates"].values()))
verdict["next"] = (
    "PASS -> regenerate all-773 local-reference SP45 features, then refit Track8/mainline descendants"
    if verdict["strict_pass"]
    else "FAIL -> local_050 helps STRIDE but does not transfer through the SP45 PF/beam/selector stack"
)
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nSP45 LOCALREF ENDPOINT COMPARISON")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
