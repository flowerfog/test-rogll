"""Append the frozen P3 raw-GR physics endpoint to the final 6.377 submission.

Run this as the final Kaggle cell, after CP8 and Neighbor have written
`/kaggle/working/submission.csv`. Upload P3's `frozen_prior.json` in any Kaggle
dataset. No training labels, OOF predictions, neighbours, CP, or Gold enter the
physics endpoint.
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


WEIGHT = 0.05
SEGMENT_LENGTH_FT = 128.0
ROW_STRIDE = 8
MIN_SEGMENT_ROWS = 4
PERSISTENCE_WEIGHT = 0.1
BEAM_SIZE = 512
DIVERSITY_FT = 4.0
MIN_DECODE_COVERAGE = 0.98

WORKING = Path(os.environ.get("PHYSICS_GR_WORKING", "/kaggle/working"))
SUBMISSION_PATH = WORKING / "submission.csv"
BEFORE_PATH = WORKING / "submission_before_physics_gr_map_w005.csv"
OUTPUT_PATH = WORKING / "submission_physics_gr_map_w005.csv"
AUDIT_PATH = WORKING / "physics_gr_map_w005_audit.json"
PER_WELL_PATH = WORKING / "physics_gr_map_w005_bywell.csv"

ARM = {
    "id": "raw_clip_b512_shape4",
    "beam": BEAM_SIZE,
    "diversity_ft": DIVERSITY_FT,
    "reference": "raw",
    "likelihood": "clipped_gaussian",
    "history_diversity": True,
}


def _find_data_root() -> Path:
    candidates = []
    if os.environ.get("PHYSICS_GR_DATA_ROOT"):
        candidates.append(Path(os.environ["PHYSICS_GR_DATA_ROOT"]))
    candidates.extend([
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path.cwd(),
    ])
    candidates.extend(path.parent for path in Path("/kaggle/input").glob("**/test"))
    for candidate in candidates:
        test_dir = candidate / "test"
        if test_dir.is_dir() and next(test_dir.glob("*__horizontal_well.csv"), None):
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate the competition test directory")


def _find_prior() -> Path:
    names = ("frozen_prior.json", "physics_gr_map_prior_p3.json")
    candidates = []
    if os.environ.get("PHYSICS_GR_PRIOR"):
        candidates.append(Path(os.environ["PHYSICS_GR_PRIOR"]))
    candidates.extend(WORKING / name for name in names)
    for name in names:
        candidates.extend(Path("/kaggle/input").glob(f"**/{name}"))
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise FileNotFoundError(
            "Upload physics_gr_map_fold1_confirmation_p3/frozen_prior.json"
        )
    if len({path.resolve() for path in found}) > 1:
        print("PRIOR CANDIDATES", [str(path) for path in found], flush=True)
    return found[0]


def _interp_nan(x, xp, fp):
    x = np.asarray(x, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    valid = np.isfinite(xp) & np.isfinite(fp)
    if valid.sum() < 2:
        return np.full(x.shape, np.nan, dtype=np.float64)
    xp = xp[valid]
    fp = fp[valid]
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]
    xp, indices = np.unique(xp, return_index=True)
    fp = fp[indices]
    result = np.interp(x, xp, fp)
    result[(x < xp[0]) | (x > xp[-1])] = np.nan
    return result


def _load_record(data_root: Path, well: str):
    horizontal_path = data_root / "test" / f"{well}__horizontal_well.csv"
    typewell_path = data_root / "test" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    required_h = {"MD", "Z", "GR", "TVT_input"}
    required_t = {"TVT", "GR"}
    if not required_h.issubset(horizontal) or not required_t.issubset(typewell):
        raise RuntimeError(f"{well}: missing raw columns")

    horizontal["_original_row"] = np.arange(len(horizontal), dtype=np.int64)
    horizontal = horizontal.sort_values("MD", kind="stable").reset_index(drop=True)
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(np.float64)
    gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    original_rows = horizontal["_original_row"].to_numpy(np.int64)
    known = np.isfinite(tvt_input)
    evaluation = ~known
    if known.sum() < 100 or evaluation.sum() < 100:
        raise RuntimeError(f"{well}: known/evaluation={known.sum()}/{evaluation.sum()}")
    anchor = int(np.flatnonzero(known)[-1])

    type_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64)
    type_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64)
    known_rows = np.flatnonzero(known & np.isfinite(gr))[::4]
    known_reference = _interp_nan(tvt_input[known_rows], type_tvt, type_gr)
    residual = gr[known_rows] - known_reference
    residual = residual[np.isfinite(residual)]
    if len(residual) < 32:
        noise_scale = 15.0
    else:
        centre = np.median(residual)
        noise_scale = 1.4826 * np.median(np.abs(residual - centre))
    noise_scale = float(np.clip(noise_scale, 5.0, 35.0))

    eval_rows = np.flatnonzero(evaluation)
    sampled_rows = eval_rows[::ROW_STRIDE]
    if len(sampled_rows) < 64:
        raise RuntimeError(f"{well}: only {len(sampled_rows)} sampled evaluation rows")
    return {
        "well": well,
        "md": md,
        "z": z,
        "gr": gr,
        "tvt_input": tvt_input,
        "evaluation": evaluation,
        "eval_rows": eval_rows,
        "sampled_rows": sampled_rows,
        "original_rows": original_rows,
        "anchor_md": float(md[anchor]),
        "anchor_surface": float(tvt_input[anchor] + z[anchor]),
        "type_tvt": type_tvt,
        "type_gr": type_gr,
        "noise_scale": noise_scale,
    }


def _segments(record):
    rows = record["sampled_rows"]
    md = record["md"][rows]
    segment_id = np.floor(
        (md - record["anchor_md"]) / SEGMENT_LENGTH_FT
    ).astype(np.int64)
    output = []
    for identifier in np.unique(segment_id):
        selected = rows[segment_id == identifier]
        if len(selected) >= MIN_SEGMENT_ROWS:
            output.append(selected)
    if not output:
        raise RuntimeError(f"{record['well']}: no valid segments")
    return output


def _loss_and_surface(record, rows, start_md, start_surfaces, rates):
    md = record["md"][rows]
    delta = md[None, None, :] - float(start_md)
    surface = start_surfaces[:, None, None] + rates[None, :, None] * delta
    candidate_tvt = surface - record["z"][rows][None, None, :]
    reference = _interp_nan(
        candidate_tvt.ravel(), record["type_tvt"], record["type_gr"]
    ).reshape(candidate_tvt.shape)
    observed = record["gr"][rows][None, None, :]
    valid = np.isfinite(reference) & np.isfinite(observed)
    normalised = (reference - observed) / record["noise_scale"]
    point_loss = np.minimum(normalised**2, 9.0)
    point_loss[~valid] = 0.0
    coverage = np.sum(valid, axis=2)
    loss = np.sum(point_loss, axis=2) / np.maximum(coverage, 1)
    loss[coverage < max(2, int(math.ceil(0.25 * len(rows))))] = 1e6
    end_surface = start_surfaces[:, None] + rates[None, :] * float(md[-1] - start_md)
    return loss, end_surface


def _select_history_diverse(score, end_surface, inherited_history, update_history):
    flat_score = score.ravel()
    flat_surface = end_surface.ravel()
    candidate_history = np.repeat(inherited_history, score.shape[1])
    next_history = flat_surface.copy() if update_history else candidate_history.copy()
    order = np.argsort(flat_score)
    selected = []
    selected_set = set()
    bin_count = {}
    for index in order:
        key = (
            int(np.floor(flat_surface[index] / DIVERSITY_FT)),
            int(np.floor(candidate_history[index] / DIVERSITY_FT)),
        )
        if bin_count.get(key, 0) >= 1:
            continue
        selected.append(int(index))
        selected_set.add(int(index))
        bin_count[key] = 1
        if len(selected) >= BEAM_SIZE:
            break
    if len(selected) < BEAM_SIZE:
        for index in order:
            index = int(index)
            if index not in selected_set:
                selected.append(index)
                if len(selected) >= BEAM_SIZE:
                    break
    keep = np.asarray(selected, dtype=np.int64)
    return keep, next_history[keep]


def _beam_map(record, prior):
    rates = prior["rate_grid"]
    segments = _segments(record)
    scores = np.array([0.0], dtype=np.float64)
    surfaces = np.array([record["anchor_surface"]], dtype=np.float64)
    history_surface = surfaces.copy()
    previous_rate = np.array([-1], dtype=np.int64)
    paths = np.empty((1, 0), dtype=np.int16)
    previous_md = record["anchor_md"]

    for segment_index, rows in enumerate(segments):
        loss, end_surface = _loss_and_surface(
            record, rows, previous_md, surfaces, rates
        )
        candidate_score = scores[:, None] + loss
        previous = np.where(
            previous_rate[:, None] >= 0,
            rates[np.maximum(previous_rate[:, None], 0)],
            prior["rate_centre"],
        )
        transition = (
            (rates[None, :] - previous - prior["change_centre"])
            / prior["change_scale"]
        ) ** 2
        candidate_score += PERSISTENCE_WEIGHT * transition
        update_history = ((segment_index + 1) % 4 == 0)
        keep, history_surface = _select_history_diverse(
            candidate_score, end_surface, history_surface, update_history
        )
        parent = keep // len(rates)
        rate_index = keep % len(rates)
        scores = candidate_score.ravel()[keep]
        surfaces = end_surface.ravel()[keep]
        previous_rate = rate_index.astype(np.int64)
        paths = np.concatenate(
            [paths[parent], rate_index[:, None].astype(np.int16)], axis=1
        )
        previous_md = float(record["md"][rows][-1])
    return paths[int(np.argmin(scores))], segments, float(np.min(scores))


def _path_prediction(record, prior, path, segments):
    rates = prior["rate_grid"]
    prediction = np.full(record["md"].shape, np.nan, dtype=np.float64)
    start_md = record["anchor_md"]
    start_surface = record["anchor_surface"]
    for rows, rate_index in zip(segments, path):
        rate = float(rates[int(rate_index)])
        prediction[rows] = (
            start_surface + rate * (record["md"][rows] - start_md) - record["z"][rows]
        )
        start_surface += rate * float(record["md"][rows][-1] - start_md)
        start_md = float(record["md"][rows][-1])
    valid = np.flatnonzero(np.isfinite(prediction))
    if not len(valid):
        raise RuntimeError(f"{record['well']}: empty path prediction")
    eval_rows = record["eval_rows"]
    surface = _interp_nan(
        record["md"][eval_rows],
        record["md"][valid],
        prediction[valid] + record["z"][valid],
    )
    first_rate = float(rates[int(path[0])])
    last_rate = float(rates[int(path[-1])])
    left = record["md"][eval_rows] < record["md"][valid[0]]
    right = record["md"][eval_rows] > record["md"][valid[-1]]
    first_surface = prediction[valid[0]] + record["z"][valid[0]]
    last_surface = prediction[valid[-1]] + record["z"][valid[-1]]
    surface[left] = first_surface + first_rate * (
        record["md"][eval_rows[left]] - record["md"][valid[0]]
    )
    surface[right] = last_surface + last_rate * (
        record["md"][eval_rows[right]] - record["md"][valid[-1]]
    )
    prediction[eval_rows] = surface - record["z"][eval_rows]
    return prediction


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


started = time.time()
if not SUBMISSION_PATH.is_file():
    raise FileNotFoundError(
        "Run the complete 6.377 CP8+Neighbor notebook first; submission.csv is missing"
    )
base = pd.read_csv(SUBMISSION_PATH)
if list(base.columns) != ["id", "tvt"]:
    raise RuntimeError(f"Unexpected submission columns: {base.columns.tolist()}")
if base["id"].duplicated().any() or not np.isfinite(base["tvt"]).all():
    raise RuntimeError("Invalid upstream submission")
if OUTPUT_PATH.is_file():
    previous = pd.read_csv(OUTPUT_PATH)
    if previous["id"].equals(base["id"]) and np.allclose(
        previous["tvt"], base["tvt"], rtol=0.0, atol=1e-7
    ):
        raise RuntimeError(
            "The physics dose is already present in submission.csv; rerun the upstream 6.377 cells first"
        )

data_root = _find_data_root()
prior_path = _find_prior()
prior = json.loads(prior_path.read_text(encoding="utf-8"))
prior["rate_grid"] = np.asarray(prior["rate_grid"], dtype=np.float64)
required_prior = {"rate_grid", "rate_centre", "change_centre", "change_scale"}
if not required_prior.issubset(prior) or len(prior["rate_grid"]) != 25:
    raise RuntimeError(f"Invalid frozen prior: {prior_path}")

keys = base["id"].str.rsplit("_", n=1, expand=True)
if keys.shape[1] != 2:
    raise RuntimeError("Cannot parse submission ids")
base["well"] = keys[0].astype(str)
base["row_idx"] = pd.to_numeric(keys[1], errors="raise").astype(np.int64)

prediction_by_id = {}
reports = []
failures = []
wells = base["well"].drop_duplicates().tolist()
for number, well in enumerate(wells, 1):
    expected = base.loc[base["well"].eq(well), ["id", "row_idx"]]
    try:
        record = _load_record(data_root, well)
        path, segments, score = _beam_map(record, prior)
        prediction = _path_prediction(record, prior, path, segments)
        eval_rows = record["eval_rows"]
        original_eval_rows = record["original_rows"][eval_rows]
        row_prediction = dict(zip(original_eval_rows.tolist(), prediction[eval_rows].tolist()))
        missing = sorted(set(expected["row_idx"]) - set(row_prediction))
        if missing:
            raise RuntimeError(f"{well}: {len(missing)} submission rows not in raw eval rows")
        values = expected["row_idx"].map(row_prediction).to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{well}: non-finite decoded values")
        prediction_by_id.update(zip(expected["id"], values.tolist()))
        reports.append(
            {
                "well": well,
                "rows": int(len(expected)),
                "segments": int(len(segments)),
                "map_score": score,
                "mean_delta_vs_base": float(
                    np.mean(values - base.loc[expected.index, "tvt"].to_numpy(np.float64))
                ),
            }
        )
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc), "rows": int(len(expected))})
    if number % 12 == 0 or number == len(wells):
        print(
            f"PHYSICS DEPLOY {number}/{len(wells)} success={len(reports)} failures={len(failures)}",
            flush=True,
        )

physics = base["id"].map(prediction_by_id).to_numpy(np.float64)
decoded = np.isfinite(physics)
coverage = float(decoded.mean())
if coverage < MIN_DECODE_COVERAGE:
    raise RuntimeError(f"Decoded row coverage {coverage:.3%} < {MIN_DECODE_COVERAGE:.1%}: {failures[:5]}")

final = base["tvt"].to_numpy(np.float64).copy()
final[decoded] = (1.0 - WEIGHT) * final[decoded] + WEIGHT * physics[decoded]
if not np.isfinite(final).all():
    raise RuntimeError("Non-finite final predictions")

submission = base[["id", "tvt"]].copy()
submission["tvt"] = final
if not submission["id"].equals(base["id"]):
    raise RuntimeError("Submission order changed")

shutil.copy2(SUBMISSION_PATH, BEFORE_PATH)
submission.to_csv(OUTPUT_PATH, index=False)
submission.to_csv(SUBMISSION_PATH, index=False)
pd.DataFrame(reports).to_csv(PER_WELL_PATH, index=False)
audit = {
    "experiment": "physics_gr_map_kaggle_deploy_p4",
    "parent": "exact final 6.377 CP8+Neighbor submission.csv",
    "arm": ARM,
    "prior": str(prior_path),
    "weight": WEIGHT,
    "rows": int(len(base)),
    "wells": int(len(wells)),
    "decoded_rows": int(decoded.sum()),
    "decoded_coverage": coverage,
    "decoded_wells": int(len(reports)),
    "failed_wells": failures,
    "mean_abs_applied_change": float(np.mean(np.abs(final - base["tvt"].to_numpy(np.float64)))),
    "p95_abs_applied_change": float(np.quantile(np.abs(final - base["tvt"].to_numpy(np.float64)), 0.95)),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "output": str(SUBMISSION_PATH),
}
AUDIT_PATH.write_text(json.dumps(_clean(audit), indent=2, allow_nan=False), encoding="utf-8")
print("\nPHYSICS GR MAP DEPLOYMENT")
print(json.dumps(_clean(audit), indent=2), flush=True)
print("saved:", SUBMISSION_PATH, flush=True)
