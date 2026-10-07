"""Fold-clean SP45 replay with a feature-rich OOF export.

This is the SP45 engine only.  It intentionally does not train Track8.

For every training well, using its official TVT_input mask, it exports:
* 128 independently seeded PF paths aggregated at scales 3/5/8/12;
* seed mean/std/quantiles and likelihood diagnostics;
* all 14 beam configurations plus their mean/std/median;
* the notebook's deterministic per-well selector output;
* the selector output after the notebook's label-free surface projection.

No full-truth contact/physical override is allowed.  Truth is read only after
all predictions have been frozen, for reporting and later outer-fold models.
Per-well NPZ records make the long CPU job resumable.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
EXPERIMENT = "sp45_foldclean_feature_export_v1"
OUT_DIR = OOF_ROOT / EXPERIMENT
RECORD_DIR = OUT_DIR / "records"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RECORD_DIR.mkdir(parents=True, exist_ok=True)

PF_SEEDS = int(os.environ.get("SP45_SEEDS", "128"))
PF_PARTICLES = int(os.environ.get("SP45_PARTICLES", "500"))
N_JOBS = int(os.environ.get("SP45_JOBS", str(min(8, os.cpu_count() or 1))))
MAX_WELLS = int(os.environ.get("SP45_MAX_WELLS", "0"))
FORCE = bool(int(os.environ.get("SP45_FORCE", "0")))
KEEP_RECORDS = bool(int(os.environ.get("SP45_KEEP_RECORDS", "1")))
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


def _first_existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {label}: {[str(path) for path in candidates]}")


def _find_data_root() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        ROOT / "dataset",
        ROOT / "rogii-wellbore-geology-prediction",
        Path.cwd(),
    ]
    for path in candidates:
        if (path / "train").exists() and any((path / "train").glob("*__horizontal_well.csv")):
            return path
    for train_dir in ROOT.glob("**/train"):
        if any(train_dir.glob("*__horizontal_well.csv")):
            return train_dir.parent
    raise FileNotFoundError("Cannot locate the ROGII raw data root")


DATA_ROOT = _find_data_root()
FOLD_PATH = _first_existing(
    [
        OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
        OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "folds.csv",
    ],
    "fold metadata",
)
POOL_PATH = _first_existing(
    [
        OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
        OOF_ROOT / "candidate_pool_full_merged_oof.csv",
        ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    ],
    "candidate pool",
)


def _normalise_folds(frame: pd.DataFrame) -> pd.DataFrame:
    well_column = next((c for c in ("well", "well_id", "Well") if c in frame), None)
    fold_column = next((c for c in ("fold", "outer_fold", "Fold") if c in frame), None)
    if well_column is None or fold_column is None:
        raise RuntimeError(f"Cannot identify fold columns: {frame.columns.tolist()}")
    output = frame[[well_column, fold_column]].copy()
    output.columns = ["well", "fold"]
    output["well"] = output["well"].astype(str)
    output["fold"] = pd.to_numeric(output["fold"], errors="raise").astype(np.int8)
    output = output.drop_duplicates("well").sort_values("well").reset_index(drop=True)
    if not output["fold"].isin([1, 2, 3, 4, 5]).all():
        raise RuntimeError("Fold metadata contains values outside 1..5")
    return output


folds = _normalise_folds(pd.read_csv(FOLD_PATH))
available_wells = sorted(
    path.name.split("__", 1)[0]
    for path in (DATA_ROOT / "train").glob("*__horizontal_well.csv")
)
wells = [well for well in folds["well"] if well in set(available_wells)]
if MAX_WELLS:
    wells = wells[:MAX_WELLS]
if not MAX_WELLS and len(wells) < 770:
    raise RuntimeError(f"Only {len(wells)} of 773 fold wells have raw files")

fold_by_well = dict(zip(folds["well"], folds["fold"]))


def _prefix_setup(hw: pd.DataFrame, tw: pd.DataFrame) -> dict[str, np.ndarray | float]:
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

    gr_all = pd.to_numeric(hw["GR"], errors="coerce").interpolate(
        limit_direction="both"
    ).fillna(float(np.mean(tw_gr))).to_numpy(np.float64)
    md = pd.to_numeric(hw["MD"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(hw["Z"], errors="coerce").to_numpy(np.float64)
    truth = pd.to_numeric(hw["TVT"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(hw["TVT_input"], errors="coerce").to_numpy(np.float64)

    last = known_index[-1]
    last_tvt = float(tvt_input[last])
    last_z = float(z[last])
    last_md = float(md[last])
    tw_at_known = np.interp(tvt_input[known_index], tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(gr_all[known_index] - tw_at_known), 10.0, 60.0))

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
        "gr_scale": gs,
        "z_span": float(np.nanmax(z[eval_index]) - np.nanmin(z[eval_index])),
    }


def _run_pf_seed(payload: dict, seed: int) -> tuple[np.ndarray, float]:
    md_eval = payload["md_eval"]
    z_eval = payload["z_eval"]
    gr_eval = payload["gr_eval"]
    tw_tvt = payload["tw_tvt"]
    tw_gr = payload["tw_gr"]
    last_tvt = float(payload["last_tvt"])
    last_z = float(payload["last_z"])
    previous_md = float(payload["last_md"])
    gs = float(payload["gr_scale"])

    rng = np.random.default_rng(seed)
    count = PF_PARTICLES
    position = last_tvt + last_z + 4.5 * rng.standard_normal(count)
    rate = float(payload["initial_rate"]) + 0.01 * rng.standard_normal(count)
    weight = np.full(count, 1.0 / count, dtype=np.float64)
    result = np.empty(len(md_eval), dtype=np.float32)
    log_likelihood = 0.0

    for index in range(len(md_eval)):
        md_step = max(float(md_eval[index]) - previous_md, 1.0)
        rate = 0.998 * rate + 0.002 * rng.standard_normal(count)
        position = position + rate * md_step + 0.005 * rng.standard_normal(count)
        tvt_particle = np.clip(
            position - float(z_eval[index]), tw_tvt[0] - 100.0, tw_tvt[-1] + 100.0
        )
        position = tvt_particle + float(z_eval[index])
        expected_gr = np.interp(tvt_particle, tw_tvt, tw_gr)
        residual = (float(gr_eval[index]) - expected_gr) / gs
        likelihood = np.exp(-0.5 * np.minimum(residual * residual, 600.0))
        likelihood = np.maximum(likelihood, 1e-300)
        average_likelihood = float(np.dot(weight, likelihood))
        log_likelihood += math.log(max(average_likelihood, 1e-300))
        weight *= likelihood
        total = float(weight.sum())
        weight = weight / total if total > 0 else np.full(count, 1.0 / count)

        effective = 1.0 / float(np.dot(weight, weight))
        if effective < 0.5 * count:
            cumulative = np.cumsum(weight)
            start = rng.uniform(0.0, 1.0 / count)
            chosen = np.clip(
                np.searchsorted(cumulative, start + np.arange(count) / count),
                0,
                count - 1,
            )
            position = position[chosen] + 0.1 * rng.standard_normal(count)
            rate = rate[chosen] + 0.001 * rng.standard_normal(count)
            weight = np.full(count, 1.0 / count)

        result[index] = float(np.dot(weight, position - float(z_eval[index])))
        previous_md = float(md_eval[index])
    return result, log_likelihood


def _pf_bundle(payload: dict) -> dict[str, np.ndarray | float]:
    predictions = []
    likelihoods = np.empty(PF_SEEDS, dtype=np.float64)
    for seed in range(PF_SEEDS):
        prediction, likelihood = _run_pf_seed(payload, seed)
        predictions.append(prediction)
        likelihoods[seed] = likelihood
    members = np.stack(predictions).astype(np.float32)
    shifted_likelihood = likelihoods - np.max(likelihoods)
    output: dict[str, np.ndarray | float] = {}
    for scale in PF_SCALES:
        weights = np.exp(shifted_likelihood / scale)
        weights /= weights.sum()
        output[f"pf_scale_{int(scale)}"] = np.sum(
            members * weights[:, None], axis=0, dtype=np.float64
        ).astype(np.float32)
    output["pf_mean"] = members.mean(axis=0).astype(np.float32)
    output["pf_std"] = members.std(axis=0).astype(np.float32)
    output["pf_p10"] = np.quantile(members, 0.10, axis=0).astype(np.float32)
    output["pf_p50"] = np.quantile(members, 0.50, axis=0).astype(np.float32)
    output["pf_p90"] = np.quantile(members, 0.90, axis=0).astype(np.float32)
    order = np.sort(likelihoods)
    output["likelihood_max"] = float(order[-1])
    output["likelihood_mean"] = float(np.mean(likelihoods))
    output["likelihood_std"] = float(np.std(likelihoods))
    output["likelihood_top_gap"] = float(order[-1] - order[-2]) if len(order) > 1 else 0.0
    return output


def _beam_search(
    horizontal_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    last_tvt: float,
    beam_size: int,
    move_cost: float,
    emission_scale: float,
    radius: int,
) -> np.ndarray:
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


def _beam_bundle(payload: dict) -> dict[str, np.ndarray]:
    members = []
    for beam_size, move_cost, emission_scale, radius in BEAM_CONFIGS:
        members.append(
            _beam_search(
                payload["gr_eval"],
                payload["tw_tvt"],
                payload["tw_gr"],
                float(payload["last_tvt"]),
                beam_size,
                move_cost,
                emission_scale,
                radius,
            )
        )
    stack = np.stack(members).astype(np.float32)
    output = {f"beam_{index:02d}": path for index, path in enumerate(stack)}
    output["beam_mean"] = stack.mean(axis=0).astype(np.float32)
    output["beam_std"] = stack.std(axis=0).astype(np.float32)
    output["beam_median"] = np.median(stack, axis=0).astype(np.float32)
    return output


def _selector_code(eval_rows: int, z_span: float) -> tuple[int, str]:
    n_bin = int(eval_rows > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right"))
    code = n_bin + 2 * z_bin
    return code, SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)


def _parse_selector(name: str) -> tuple[int, float, float]:
    parts = name.split("_")
    scale = int(float(parts[2]))
    beam_weight = float(parts[parts.index("beam") + 1]) if "beam" in parts else 0.0
    hold_weight = float(parts[parts.index("hold") + 1]) if "hold" in parts else 0.0
    return scale, beam_weight, hold_weight


def _apply_selector(
    name: str, pf: dict[str, np.ndarray | float], beam_mean: np.ndarray, last_tvt: float
) -> np.ndarray:
    scale, beam_weight, hold_weight = _parse_selector(name)
    base = np.asarray(pf[f"pf_scale_{scale}"], dtype=np.float32)
    prediction = (1.0 - beam_weight) * base + beam_weight * beam_mean
    prediction = (1.0 - hold_weight) * prediction + hold_weight * last_tvt
    return prediction.astype(np.float32)


def _project_surface(
    prediction: np.ndarray,
    md_eval: np.ndarray,
    z_eval: np.ndarray,
    last_md: float,
    anchor_surface: float,
) -> np.ndarray:
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


def _record_path(well: str) -> Path:
    return RECORD_DIR / f"{well}.npz"


def _record_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                int(data["pf_seeds"][0]) == PF_SEEDS
                and int(data["pf_particles"][0]) == PF_PARTICLES
                and len(data["row_idx"]) >= 50
                and "beam_13" in data.files
            )
    except Exception:
        return False


def _process_well(well: str) -> dict:
    destination = _record_path(well)
    if not FORCE and _record_valid(destination):
        with np.load(destination, allow_pickle=False) as data:
            return {"well": well, "status": "cached", "rows": int(len(data["row_idx"]))}
    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    try:
        hw = pd.read_csv(horizontal_path)
        tw = pd.read_csv(typewell_path)
        payload = _prefix_setup(hw, tw)
        pf = _pf_bundle(payload)
        beam = _beam_bundle(payload)
        selector_code, selector_variant = _selector_code(
            len(payload["eval_index"]), float(payload["z_span"])
        )
        selector_raw = _apply_selector(
            selector_variant, pf, beam["beam_mean"], float(payload["last_tvt"])
        )
        selector_projected = _project_surface(
            selector_raw,
            payload["md_eval"],
            payload["z_eval"],
            float(payload["last_md"]),
            float(payload["last_tvt"] + payload["last_z"]),
        )
        save = {
            "well": np.asarray([well]),
            "fold": np.asarray([fold_by_well[well]], dtype=np.int8),
            "row_idx": payload["eval_index"].astype(np.int32),
            "md": payload["md_eval"].astype(np.float32),
            "z": payload["z_eval"].astype(np.float32),
            "truth": payload["truth_eval"].astype(np.float32),
            "last_tvt": np.asarray([payload["last_tvt"]], dtype=np.float32),
            "last_md": np.asarray([payload["last_md"]], dtype=np.float32),
            "z_span": np.asarray([payload["z_span"]], dtype=np.float32),
            "gr_scale": np.asarray([payload["gr_scale"]], dtype=np.float32),
            "initial_rate": np.asarray([payload["initial_rate"]], dtype=np.float32),
            "selector_code": np.asarray([selector_code], dtype=np.int8),
            "selector_variant": np.asarray([selector_variant]),
            "selector_raw": selector_raw,
            "selector_projected": selector_projected,
            "pf_seeds": np.asarray([PF_SEEDS], dtype=np.int16),
            "pf_particles": np.asarray([PF_PARTICLES], dtype=np.int16),
        }
        for key, value in pf.items():
            save[key] = np.asarray([value], dtype=np.float32) if np.isscalar(value) else value
        save.update(beam)
        np.savez_compressed(destination, **save)
        return {"well": well, "status": "built", "rows": int(len(payload["eval_index"]))}
    except Exception as exc:
        return {"well": well, "status": "failed", "rows": 0, "reason": repr(exc)}


print(
    "SP45 REPLAY CONFIG",
    {
        "data": str(DATA_ROOT),
        "wells": len(wells),
        "seeds": PF_SEEDS,
        "particles": PF_PARTICLES,
        "beam_configs": len(BEAM_CONFIGS),
        "jobs": N_JOBS,
        "resume_records": str(RECORD_DIR),
    },
    flush=True,
)

free_gb = shutil.disk_usage(ROOT).free / (1024**3)
if free_gb < 1.25:
    raise RuntimeError(f"Only {free_gb:.2f} GB free; free at least 1.25 GB before SP45 export")

reports = []
batch_size = max(N_JOBS * 2, 1)
for start in range(0, len(wells), batch_size):
    batch = wells[start : start + batch_size]
    reports.extend(Parallel(n_jobs=N_JOBS, backend="loky")(delayed(_process_well)(well) for well in batch))
    done = min(start + len(batch), len(wells))
    failures = sum(report["status"] == "failed" for report in reports)
    print(f"SP45 replay {done}/{len(wells)} failures={failures}", flush=True)

build_report = pd.DataFrame(reports)
build_report.to_csv(OUT_DIR / "build_report.csv", index=False)
failures = build_report[build_report["status"].eq("failed")]
if len(failures):
    raise RuntimeError(
        f"{len(failures)} SP45 wells failed: {failures.head(10).to_dict(orient='records')}"
    )


def _load_record_frame(well: str) -> pd.DataFrame:
    with np.load(_record_path(well), allow_pickle=False) as data:
        count = len(data["row_idx"])
        row_index = data["row_idx"].astype(np.int32)
        output = pd.DataFrame(
            {
                "id": [f"{well}_{int(index)}" for index in row_index],
                "well": well,
                "row_idx": row_index,
                "fold": np.full(count, int(data["fold"][0]), dtype=np.int8),
                "truth": data["truth"].astype(np.float32),
                "md": data["md"].astype(np.float32),
                "md_since": (data["md"] - float(data["last_md"][0])).astype(np.float32),
                "last_tvt": np.full(count, float(data["last_tvt"][0]), dtype=np.float32),
                "z_span": np.full(count, float(data["z_span"][0]), dtype=np.float32),
                "gr_scale": np.full(count, float(data["gr_scale"][0]), dtype=np.float32),
                "initial_rate": np.full(count, float(data["initial_rate"][0]), dtype=np.float32),
                "selector_code": np.full(count, int(data["selector_code"][0]), dtype=np.int8),
                "sp45_selector_raw": data["selector_raw"].astype(np.float32),
                "sp45_selector_projected": data["selector_projected"].astype(np.float32),
            }
        )
        array_columns = [
            f"pf_scale_{int(scale)}" for scale in PF_SCALES
        ] + ["pf_mean", "pf_std", "pf_p10", "pf_p50", "pf_p90"]
        array_columns += [f"beam_{index:02d}" for index in range(len(BEAM_CONFIGS))]
        array_columns += ["beam_mean", "beam_std", "beam_median"]
        for column in array_columns:
            output[column] = data[column].astype(np.float32)
        for column in (
            "likelihood_max",
            "likelihood_mean",
            "likelihood_std",
            "likelihood_top_gap",
        ):
            output[column] = np.full(count, float(data[column][0]), dtype=np.float32)
        output["pf_scale_range"] = (
            output[[f"pf_scale_{int(scale)}" for scale in PF_SCALES]].max(axis=1)
            - output[[f"pf_scale_{int(scale)}" for scale in PF_SCALES]].min(axis=1)
        ).astype(np.float32)
        output["beam_range"] = (
            output[[f"beam_{index:02d}" for index in range(len(BEAM_CONFIGS))]].max(axis=1)
            - output[[f"beam_{index:02d}" for index in range(len(BEAM_CONFIGS))]].min(axis=1)
        ).astype(np.float32)
        output["pf_beam_disagreement"] = (
            output["pf_scale_8"] - output["beam_mean"]
        ).astype(np.float32)
        return output


parts = []
for index, well in enumerate(wells, start=1):
    parts.append(_load_record_frame(well))
    if index % 50 == 0 or index == len(wells):
        print(f"SP45 combine {index}/{len(wells)}", flush=True)
oof = pd.concat(parts, ignore_index=True)
del parts
oof = oof.sort_values(["well", "row_idx"]).reset_index(drop=True)

if oof.duplicated("id").any():
    raise RuntimeError("SP45 export has duplicate row ids")
if not np.isfinite(oof.select_dtypes(include=[np.number]).to_numpy(np.float64)).all():
    raise RuntimeError("SP45 export contains non-finite numeric values")

OOF_PATH = OUT_DIR / "sp45_feature_oof.parquet"
oof.to_parquet(OOF_PATH, index=False, compression="zstd")


def _metrics(prediction: np.ndarray) -> dict:
    error = np.asarray(prediction, np.float64) - oof["truth"].to_numpy(np.float64)
    bywell = pd.DataFrame({"well": oof["well"], "error2": error * error}).groupby("well")["error2"].mean()
    return {
        "rows": int(len(error)),
        "wells": int(oof["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(bywell))),
        "p95_well_rmse": float(np.quantile(np.sqrt(bywell), 0.95)),
    }


model_columns = [
    "pf_scale_3",
    "pf_scale_5",
    "pf_scale_8",
    "pf_scale_12",
    "pf_mean",
    "beam_mean",
    "sp45_selector_raw",
    "sp45_selector_projected",
]
summary_rows = [{"variant": column, **_metrics(oof[column])} for column in model_columns]
summary = pd.DataFrame(summary_rows).sort_values("rmse")
summary.to_csv(OUT_DIR / "endpoint_summary.csv", index=False)

pool_header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
compare_columns = [column for column in ("sp45_raw", "sp45") if column in pool_header]
parity = []
if compare_columns:
    pool = pd.read_csv(POOL_PATH, usecols=["id"] + compare_columns)
    comparison = oof[["id", "sp45_selector_raw", "sp45_selector_projected"]].merge(
        pool, on="id", how="left", validate="one_to_one"
    )
    for exported in ("sp45_selector_raw", "sp45_selector_projected"):
        for historical in compare_columns:
            valid = np.isfinite(comparison[historical].to_numpy(float))
            difference = (
                comparison.loc[valid, exported].to_numpy(float)
                - comparison.loc[valid, historical].to_numpy(float)
            )
            parity.append(
                {
                    "exported": exported,
                    "historical": historical,
                    "rows": int(valid.sum()),
                    "coverage": float(valid.mean()),
                    "rmse_difference": float(np.sqrt(np.mean(difference * difference))),
                    "mae_difference": float(np.mean(np.abs(difference))),
                }
            )
parity_report = pd.DataFrame(parity)
parity_report.to_csv(OUT_DIR / "historical_parity.csv", index=False)

full_run = MAX_WELLS == 0 and PF_SEEDS == 128 and PF_PARTICLES == 500
coverage_pass = len(oof) >= 3_760_000 and oof["well"].nunique() >= 770 if full_run else True
parity_best = float(parity_report["rmse_difference"].min()) if len(parity_report) else float("nan")

manifest = {
    "experiment": EXPERIMENT,
    "stage": "foldclean_sp45_multiseed_multibeam_feature_export",
    "sample_unit": "complete physical well",
    "data_root": str(DATA_ROOT),
    "fold_source": str(FOLD_PATH),
    "candidate_pool": str(POOL_PATH),
    "feature_cache": str(OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "features"),
    "truth_usage": "metrics only after all label-free engine outputs are frozen",
    "samewell_physical_contact_override": False,
    "pf": {"seeds": PF_SEEDS, "particles": PF_PARTICLES, "scales": list(PF_SCALES)},
    "beam": {"configs": [list(config) for config in BEAM_CONFIGS], "count": len(BEAM_CONFIGS)},
    "selector": {
        "type": "deterministic per-well rule from the source notebook",
        "n_eval_threshold": SELECTOR_N_EVAL_THRESHOLD,
        "z_span_thresholds": list(SELECTOR_Z_SPAN_THRESHOLDS),
        "variants": SELECTOR_BIN_VARIANTS,
        "global_variant": SELECTOR_GLOBAL_VARIANT,
    },
    "coverage": {"rows": int(len(oof)), "wells": int(oof["well"].nunique())},
    "summary": summary.to_dict(orient="records"),
    "historical_parity": parity_report.to_dict(orient="records"),
    "checks": {
        "full_128seed_500particle_run": full_run,
        "coverage_pass": bool(coverage_pass),
        "historical_endpoint_reproduction_rmse_le_0p25": bool(np.isfinite(parity_best) and parity_best <= 0.25),
    },
    "strict_pass": bool(
        full_run and coverage_pass and np.isfinite(parity_best) and parity_best <= 0.25
    ),
    "oof": str(OOF_PATH),
    "records": str(RECORD_DIR) if KEEP_RECORDS else None,
    "next": (
        "PASS -> train matched Track6/Track8 GBDTs and audit error correlation"
        if full_run and coverage_pass and np.isfinite(parity_best) and parity_best <= 0.25
        else "Do not train Track8 yet; resolve SP45 parity or complete the full replay"
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
}

MANIFEST_PATH = OUT_DIR / "sp45_provenance_manifest.json"
MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if not KEEP_RECORDS:
    for path in RECORD_DIR.glob("*.npz"):
        path.unlink()

print("\nSP45 ENDPOINT SUMMARY")
print(summary.to_string(index=False), flush=True)
if len(parity_report):
    print("\nSP45 HISTORICAL PARITY")
    print(parity_report.to_string(index=False), flush=True)
print("\nSP45 FINAL VERDICT")
print(json.dumps(manifest, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
