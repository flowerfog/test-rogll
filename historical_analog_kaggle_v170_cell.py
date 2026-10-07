"""Kaggle inference cell for the frozen V168/V170 metadata analog move.

Run immediately after V69, V77, V116 and V144. Upload the directory produced by
historical_analog_deployment_bundle_v170_cell.py as a Kaggle dataset. The cell
verifies that root submission.csv is exactly V144 before applying the 0.05
move, and preserves both parent and named candidate copies.
"""

from pathlib import Path
import hashlib
import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED_V170K = time.time()
WORK_V170K = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
INPUT_V170K = Path("/kaggle/input")
PARENT_V170K = WORK_V170K / "submission.csv"
PARENT_COPY_V170K = WORK_V170K / "submission_v144_parent_before_v170.csv"
OUTPUT_V170K = WORK_V170K / "submission.csv"
NAMED_V170K = WORK_V170K / "submission_v170_metadata_analog_w0p05.csv"
SUMMARY_V170K = WORK_V170K / "v170_metadata_analog_summary.json"

required_live_v170k = ("sample", "sample_value_column", "data_root", "well_rows")
missing_live_v170k = [name for name in required_live_v170k if name not in globals()]
if missing_live_v170k:
    raise RuntimeError(
        "Run V69, V77, V116 and V144 immediately before V170. Missing: "
        + repr(missing_live_v170k)
    )


def _sha256_v170k(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_bundle_v170k():
    configured = os.environ.get("V170_BUNDLE_DIR")
    candidates = [Path(configured)] if configured else []
    if INPUT_V170K.is_dir():
        candidates.extend(path.parent for path in INPUT_V170K.rglob("manifest.json"))
    for directory in candidates:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(manifest, dict)
            and manifest.get("experiment")
            == "historical_analog_deployment_bundle_v170"
        ):
            return directory, manifest
    raise FileNotFoundError("Cannot locate historical_analog_deployment_bundle_v170")


def _sample_vector_v170k(md, values, anchor, window, nodes, relative, standardise):
    grid = np.linspace(float(md[anchor]) - window, float(md[anchor]), nodes)
    valid = np.isfinite(md) & np.isfinite(values) & (md <= md[anchor] + 1e-6)
    if valid.sum() < 4:
        return np.full(nodes, np.nan)
    order = np.argsort(md[valid], kind="stable")
    x, y = md[valid][order], values[valid][order]
    result = np.interp(grid, x, y, left=y[0], right=y[-1]).astype(np.float64)
    if relative:
        result -= result[-1]
    if standardise:
        result -= np.mean(result)
        scale = float(np.std(result))
        result = result / scale if scale > 1e-6 else np.zeros_like(result)
    return result


def _sample_centered_v170k(x, values, center, half_window, nodes):
    valid = np.isfinite(x) & np.isfinite(values)
    if valid.sum() < 8:
        return np.full(nodes, np.nan, np.float64)
    order = np.argsort(x[valid], kind="stable")
    local_x, local_y = x[valid][order], values[valid][order]
    grid = np.linspace(center - half_window, center + half_window, nodes)
    result = np.interp(grid, local_x, local_y, left=local_y[0], right=local_y[-1])
    result -= np.mean(result)
    scale = float(np.std(result))
    return result / scale if scale > 1e-6 else np.zeros_like(result)


def _sample_future_v170k(md, values, anchor, nodes):
    valid = np.isfinite(md) & np.isfinite(values) & (md >= md[anchor] - 1e-6)
    if valid.sum() < 16 or md[valid][-1] <= md[anchor] + 1.0:
        missing = np.full(nodes, np.nan, np.float64)
        return missing, missing.copy()
    order = np.argsort(md[valid], kind="stable")
    x, y = md[valid][order], values[valid][order]
    grid = np.linspace(float(md[anchor]), float(x[-1]), nodes)
    shape = np.interp(grid, x, y).astype(np.float64)
    shape -= np.median(shape)
    scale = 1.4826 * float(np.median(np.abs(shape)))
    shape /= max(scale, float(np.std(shape)), 1e-6)
    gradient = np.gradient(shape)
    gscale = 1.4826 * float(np.median(np.abs(gradient - np.median(gradient))))
    if gscale > 1e-6:
        gradient /= gscale
    return np.clip(shape, -6, 6), np.clip(gradient, -6, 6)


def _slope_v170k(md, surface, anchor, window):
    valid = (
        np.isfinite(md) & np.isfinite(surface)
        & (md >= md[anchor] - window) & (md <= md[anchor] + 1e-6)
    )
    if valid.sum() < 8:
        return np.nan
    return float(np.polyfit(md[valid] - md[anchor], surface[valid], 1)[0])


def _direction_v170k(xy, anchor):
    vector = xy[-1] - xy[anchor]
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm < 1.0:
        end = min(len(xy) - 1, anchor + max(64, (len(xy) - anchor) // 2))
        vector = xy[end] - xy[anchor]
        norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm < 1.0:
        raise RuntimeError("degenerate trajectory direction")
    return vector / norm


def _robust_z_v170k(values, eligible):
    output = np.full(len(values), np.inf, np.float64)
    local = np.asarray(values, np.float64)[eligible]
    finite = np.isfinite(local)
    median = float(np.median(local[finite])) if finite.any() else 0.0
    scale = 1.4826 * float(np.median(np.abs(local[finite] - median))) if finite.any() else 1.0
    scale = max(scale, float(np.std(local[finite])) if finite.any() else 1.0, 1e-6)
    transformed = (local - median) / scale
    transformed[~finite] = 8.0
    output[eligible] = np.clip(transformed, -4.0, 12.0)
    return output


def _rms_rows_v170k(matrix, vector):
    return np.sqrt(np.nanmean(np.square(matrix - vector[None, :]), axis=1))


def _interp_extrap_v170k(x, y, query):
    output = np.interp(query, x, y)
    if len(x) >= 4:
        tail = x >= max(float(x[-1]) - 384.0, float(x[0]))
        if tail.sum() >= 4:
            slope = float(np.polyfit(x[tail] - x[-1], y[tail], 1)[0])
            slope = float(np.clip(slope, -0.12, 0.12))
            beyond = query > x[-1]
            output[beyond] = y[-1] + slope * (query[beyond] - x[-1])
    return output


def _family_scores_v170k(target, arrays):
    eligible = np.ones(len(arrays["length"]), bool)
    component = {
        "xy": _robust_z_v170k(np.linalg.norm(arrays["anchor_xy"] - target["anchor_xy"], axis=1), eligible),
        "dir": _robust_z_v170k(1.0 - np.clip(arrays["direction"] @ target["direction"], -1, 1), eligible),
        "length": _robust_z_v170k(np.abs(np.log(np.maximum(arrays["length"], 100) / max(target["length"], 100))), eligible),
        "slope": _robust_z_v170k(np.sqrt(np.nanmean(np.square(arrays["slopes"] - target["slopes"][None, :]), axis=1)), eligible),
        "surface": _robust_z_v170k(_rms_rows_v170k(arrays["surface_shape"], target["surface_shape"]), eligible),
        "z": _robust_z_v170k(_rms_rows_v170k(arrays["z_shape"], target["z_shape"]), eligible),
        "gr": _robust_z_v170k(_rms_rows_v170k(arrays["gr_shape"], target["gr_shape"]), eligible),
        "typewell": _robust_z_v170k(_rms_rows_v170k(arrays["typewell_shape"], target["typewell_shape"]), eligible),
        "future_gr": _robust_z_v170k(_rms_rows_v170k(arrays["future_gr_shape"], target["future_gr_shape"]), eligible),
        "future_gr_gradient": _robust_z_v170k(_rms_rows_v170k(arrays["future_gr_gradient"], target["future_gr_gradient"]), eligible),
    }
    return {
        "geometry": component["xy"] + .75 * component["dir"] + .35 * component["length"],
        "surface_no_xy": .5 * component["dir"] + .4 * component["length"] + component["slope"] + component["surface"],
        "typewell": component["typewell"],
        "typewell_context": component["typewell"] + .6 * component["surface"] + .4 * component["gr"] + .3 * component["dir"],
        "future_gr": component["future_gr"] + .35 * component["future_gr_gradient"],
        "future_gr_context": component["future_gr"] + .35 * component["future_gr_gradient"] + .35 * component["surface"] + .25 * component["dir"] + .2 * component["typewell"],
        "geometry_surface": .5 * component["xy"] + .6 * component["dir"] + .35 * component["length"] + component["slope"] + component["surface"],
        "geometry_surface_gr": .4 * component["xy"] + .5 * component["dir"] + .3 * component["length"] + component["slope"] + component["surface"] + .6 * component["gr"],
        "full_context": .4 * component["xy"] + .5 * component["dir"] + .3 * component["length"] + component["slope"] + component["surface"] + .4 * component["gr"] + .35 * component["z"] + .75 * component["typewell"],
    }


def _target_record_v170k(well):
    horizontal = pd.read_csv(data_root / "test" / f"{well}__horizontal_well.csv")
    horizontal["_source_row"] = np.arange(len(horizontal), dtype=np.int64)
    horizontal = horizontal.sort_values("MD", kind="stable").reset_index(drop=True)
    typewell = pd.read_csv(data_root / "test" / f"{well}__typewell.csv")
    typewell = typewell.apply(pd.to_numeric, errors="coerce").dropna(subset=["TVT", "GR"]).sort_values("TVT")
    md = horizontal["MD"].to_numpy(np.float64)
    xy = horizontal[["X", "Y"]].to_numpy(np.float64)
    z = horizontal["Z"].to_numpy(np.float64)
    gr = horizontal["GR"].to_numpy(np.float64)
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input))
    anchor = int(known[-1])
    surface_input = tvt_input + z
    future_gr, future_gradient = _sample_future_v170k(md, gr, anchor, 192)
    return {
        "well": str(well), "md": md, "z": z, "anchor": anchor,
        "source_rows": horizontal["_source_row"].to_numpy(np.int64),
        "anchor_xy": xy[anchor], "anchor_z": float(z[anchor]),
        "anchor_tvt": float(tvt_input[anchor]), "direction": _direction_v170k(xy, anchor),
        "length": float(md[-1] - md[anchor]),
        "slopes": np.asarray([_slope_v170k(md, surface_input, anchor, w) for w in (128., 256., 512.)]),
        "surface_shape": _sample_vector_v170k(md, surface_input, anchor, 512., 24, True, False),
        "z_shape": _sample_vector_v170k(md, z, anchor, 512., 24, True, False),
        "gr_shape": _sample_vector_v170k(md, gr, anchor, 512., 32, False, True),
        "typewell_shape": _sample_centered_v170k(typewell["TVT"].to_numpy(float), typewell["GR"].to_numpy(float), float(tvt_input[anchor]), 512., 128),
        "future_gr_shape": future_gr, "future_gr_gradient": future_gradient,
    }


bundle_dir_v170k, manifest_v170k = _find_bundle_v170k()
deployable_parents_v170k = manifest_v170k["parent_lineage_audit"].get("deployable_parents", [])
if "nested_sequence_coefficient" not in deployable_parents_v170k:
    raise RuntimeError(
        "Deployment blocked before submission.csv was modified: the uploaded "
        "bundle was not validated on V144 nested_sequence_coefficient. Validated parents: "
        f"{deployable_parents_v170k}. Keep the existing V144 submission.csv; "
        "do not bypass this lineage check."
    )
v144_summary_path_v170k = WORK_V170K / "v144_kaggle_summary.json"
if not v144_summary_path_v170k.is_file():
    raise FileNotFoundError("Run V144 immediately before V170")
v144_summary_v170k = json.loads(v144_summary_path_v170k.read_text(encoding="utf-8"))
if _sha256_v170k(PARENT_V170K) != v144_summary_v170k.get("submission_sha256"):
    raise RuntimeError("Root submission.csv is not the V144 output recorded by its summary")

parent_v170k = pd.read_csv(PARENT_V170K)
if not parent_v170k["id"].astype(str).equals(sample["id"].astype(str)):
    raise RuntimeError("Parent/sample ID alignment mismatch")
parent_v170k.to_csv(PARENT_COPY_V170K, index=False)
library_v170k = joblib.load(bundle_dir_v170k / manifest_v170k["reference_library"])
model_v170k = lgb.Booster(model_file=str(bundle_dir_v170k / manifest_v170k["model"]))
reference_wells_v170k = [str(well) for well in library_v170k["wells"]]
reference_records_v170k = library_v170k["records"]
arrays_v170k = library_v170k["arrays"]

analog_by_id_v170k = {}
reports_v170k = []
for number_v170k, well_v170k in enumerate(sorted(well_rows), 1):
    try:
        target_v170k = _target_record_v170k(well_v170k)
        family_scores_v170k = _family_scores_v170k(target_v170k, arrays_v170k)
        references_v170k = {}
        for family_v170k in manifest_v170k["families"]:
            ranking_v170k = np.argsort(family_scores_v170k[family_v170k], kind="stable")[:64]
            for rank_v170k, index_v170k in enumerate(ranking_v170k, 1):
                references_v170k.setdefault(int(index_v170k), []).append(rank_v170k)
        ordered_references_v170k = sorted(
            references_v170k.items(),
            key=lambda item: (min(item[1]), -len(item[1]), float(np.mean(item[1])), item[0]),
        )[:256]
        metadata_v170k = []
        for index_v170k, ranks_v170k in ordered_references_v170k:
            reference_v170k = reference_records_v170k[reference_wells_v170k[index_v170k]]
            metadata_v170k.append([
                float(np.linalg.norm(target_v170k["anchor_xy"] - reference_v170k["anchor_xy"])) / 1000.,
                float(np.dot(target_v170k["direction"], reference_v170k["direction"])),
                float(reference_v170k["length"] / max(target_v170k["length"], 1.0)),
                *np.asarray(reference_v170k["slopes"], float), *target_v170k["slopes"],
                float(min(ranks_v170k)), float(np.mean(ranks_v170k)), float(len(ranks_v170k)),
            ])
        scores_v170k = model_v170k.predict(np.asarray(metadata_v170k, np.float32))
        selected_v170k = ordered_references_v170k[int(np.argmax(scores_v170k))][0]
        reference_v170k = reference_records_v170k[reference_wells_v170k[selected_v170k]]
        eval_rows_v170k = np.flatnonzero(target_v170k["md"] > target_v170k["md"][target_v170k["anchor"]] + 1e-6)
        progress_v170k = target_v170k["md"][eval_rows_v170k] - target_v170k["md"][target_v170k["anchor"]]
        delta_v170k = _interp_extrap_v170k(
            reference_v170k["post_progress"],
            reference_v170k["post_surface_delta"],
            progress_v170k,
        )
        analog_v170k = target_v170k["anchor_tvt"] + delta_v170k - (
            target_v170k["z"][eval_rows_v170k] - target_v170k["anchor_z"]
        )
        for row_v170k, value_v170k in zip(target_v170k["source_rows"][eval_rows_v170k], analog_v170k):
            analog_by_id_v170k[f"{well_v170k}_{int(row_v170k)}"] = float(value_v170k)
        reports_v170k.append({"well": well_v170k, "status": "ok", "reference_well": reference_wells_v170k[selected_v170k]})
    except Exception as exc_v170k:
        reports_v170k.append({"well": well_v170k, "status": "failed", "reason": repr(exc_v170k)})
    if number_v170k % 10 == 0 or number_v170k == len(well_rows):
        print(f"V170 analog inference {number_v170k}/{len(well_rows)}", flush=True)

analog_v170k = sample["id"].astype(str).map(analog_by_id_v170k).to_numpy(np.float64)
available_v170k = np.isfinite(analog_v170k)
if float(available_v170k.mean()) < 0.95:
    report_frame_v170k = pd.DataFrame(reports_v170k)
    report_frame_v170k.to_csv(WORK_V170K / "v170_inference_report.csv", index=False)
    failed_v170k = report_frame_v170k.loc[
        report_frame_v170k["status"].ne("ok")
    ].to_dict("records")
    sample_keys_v170k = sample["id"].astype(str).head(3).tolist()
    generated_keys_v170k = list(analog_by_id_v170k)[:3]
    raise RuntimeError(
        f"V170 analog coverage too low: {available_v170k.mean():.3%}; "
        f"successful_wells={int(report_frame_v170k['status'].eq('ok').sum())}; "
        f"failures={failed_v170k[:5]}; sample_keys={sample_keys_v170k}; "
        f"generated_keys={generated_keys_v170k}"
    )
parent_values_v170k = parent_v170k[sample_value_column].to_numpy(np.float64)
final_v170k = parent_values_v170k.copy()
weight_v170k = float(manifest_v170k["blend_weight"])
final_v170k[available_v170k] += weight_v170k * (
    analog_v170k[available_v170k] - parent_values_v170k[available_v170k]
)
output_v170k = sample.copy()
output_v170k[sample_value_column] = final_v170k
output_v170k.to_csv(OUTPUT_V170K, index=False)
output_v170k.to_csv(NAMED_V170K, index=False)
pd.DataFrame(reports_v170k).to_csv(WORK_V170K / "v170_inference_report.csv", index=False)
summary_v170k = {
    "experiment": "historical_analog_kaggle_v170",
    "parent": "V144 nested_sequence_coefficient",
    "blend_weight": weight_v170k,
    "rows": len(output_v170k),
    "coverage": float(available_v170k.mean()),
    "successful_wells": int(sum(row["status"] == "ok" for row in reports_v170k)),
    "failed_wells": int(sum(row["status"] != "ok" for row in reports_v170k)),
    "parent_sha256": _sha256_v170k(PARENT_COPY_V170K),
    "submission_sha256": _sha256_v170k(OUTPUT_V170K),
    "elapsed_minutes": (time.time() - STARTED_V170K) / 60.0,
    "instruction": "Submit /kaggle/working/submission.csv",
}
SUMMARY_V170K.write_text(json.dumps(summary_v170k, indent=2), encoding="utf-8")
print("\nV170 KAGGLE SUMMARY")
print(json.dumps(summary_v170k, indent=2), flush=True)
