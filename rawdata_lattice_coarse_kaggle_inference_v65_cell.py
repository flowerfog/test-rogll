"""Kaggle inference cell for the V64 all-well coarse lattice ranker.

Run this as the final notebook cell after the incumbent 6.377 pipeline has
written /kaggle/working/submission.csv.  The cell generates a label-free
coarse path for each test well and applies the frozen V63 dose:

    final = 0.90 * incumbent + 0.10 * coarse

No later notebook cell should rewrite the resulting submission.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("V65 requires lightgbm") from exc


STARTED = time.time()
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
RUNTIME = WORK / "v64_coarse_runtime"
RUNTIME.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = WORK / "submission_v64_coarse_w010.csv"
BLEND_WEIGHT = 0.10


def _find_bundle():
    configured = os.environ.get("ROGII_V64_BUNDLE")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    for root in (Path("/kaggle/input"), Path("/root/autodl-tmp/oof_exports"), Path.cwd()):
        if root.exists():
            candidates.extend(root.rglob("bundle.json"))
    matches = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("experiment") == "rawdata_lattice_coarse_final_bundle_v64":
            matches.append((path.stat().st_mtime, path.resolve(), payload))
    if not matches:
        raise FileNotFoundError(
            "Cannot find the V64 deployment bundle under /kaggle/input. "
            "Upload the complete V64 deployment directory."
        )
    _, manifest_path, payload = sorted(matches, reverse=True)[0]
    return manifest_path.parent, payload


def _find_data_root():
    configured = os.environ.get("ROGII_DATA_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
            Path("/root/autodl-tmp"),
            Path.cwd(),
        ]
    )
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(path.parent for path in input_root.rglob("sample_submission.csv"))
    for root in candidates:
        if (
            root.is_dir()
            and (root / "test").is_dir()
            and next((root / "test").glob("*__horizontal_well.csv"), None)
            and (root / "sample_submission.csv").is_file()
        ):
            return root.resolve()
    raise FileNotFoundError("Cannot locate the competition test directory")


def _find_parent_submission(sample):
    configured = os.environ.get("ROGII_PARENT_SUBMISSION")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            WORK / "submission.csv",
            WORK / "final_submission.csv",
            WORK / "submission_6p377.csv",
        ]
    )
    sample_ids = sample["id"].astype(str)
    for path in candidates:
        if not path.is_file() or path.resolve() == OUTPUT_PATH.resolve():
            continue
        frame = pd.read_csv(path)
        if "id" not in frame or len(frame) != len(sample):
            continue
        if frame["id"].astype(str).equals(sample_ids):
            value_column = next((c for c in ("tvt", "TVT") if c in frame), None)
            if value_column is not None:
                return path.resolve(), frame, value_column
    raise FileNotFoundError(
        "Cannot find an aligned incumbent submission. Run the 6.377 pipeline first "
        "or set ROGII_PARENT_SUBMISSION."
    )


def _copy_runtime_sources(bundle_dir, manifest):
    required = [
        "rawdata_segment_lattice_nested_v1_cell.py",
        "rawdata_lattice_beam_survival_v2_cell.py",
        "rawdata_lattice_mode_union_v3_cell.py",
        "rawdata_lattice_coarse_ranker_v4_cell.py",
    ]
    for name in required:
        source = bundle_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, RUNTIME / name)
    model_path = bundle_dir / str(manifest["model"])
    prior_path = bundle_dir / "structural_prior.json"
    features_path = bundle_dir / "feature_names.json"
    for path in (model_path, prior_path, features_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return model_path, prior_path, features_path


def _bootstrap_v4():
    source = RUNTIME / "rawdata_lattice_coarse_ranker_v4_cell.py"
    text = source.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in text:
        raise RuntimeError(f"V4 bootstrap marker missing from {source}")
    old_cwd = Path.cwd()
    try:
        os.chdir(RUNTIME)
        namespace = {"__name__": "v64_kaggle_inference_definitions"}
        exec(compile(text.split(marker, 1)[0], str(source), "exec"), namespace)
    finally:
        os.chdir(old_cwd)
    return namespace


def _load_test_record(data_root, well, row_stride, interp_nan):
    horizontal_path = data_root / "test" / f"{well}__horizontal_well.csv"
    typewell_path = data_root / "test" / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    horizontal["_source_row"] = np.arange(len(horizontal), dtype=np.int64)
    horizontal = horizontal.sort_values("MD", kind="stable").reset_index(drop=True)
    typewell = pd.read_csv(typewell_path).sort_values("TVT", kind="stable")

    md = horizontal["MD"].to_numpy(np.float64)
    z = horizontal["Z"].to_numpy(np.float64)
    gr = horizontal["GR"].to_numpy(np.float64)
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    known = np.isfinite(tvt_input)
    evaluation = ~known
    if known.sum() < 100 or evaluation.sum() < 100:
        raise RuntimeError(f"{well}: known/evaluation={known.sum()}/{evaluation.sum()}")
    anchor = int(np.flatnonzero(known)[-1])
    type_tvt = typewell["TVT"].to_numpy(np.float64)
    type_gr = typewell["GR"].to_numpy(np.float64)

    known_rows = np.flatnonzero(known & np.isfinite(gr))[::4]
    known_reference = interp_nan(tvt_input[known_rows], type_tvt, type_gr)
    residual = gr[known_rows] - known_reference
    residual = residual[np.isfinite(residual)]
    if len(residual) < 32:
        noise_scale = 15.0
    else:
        centre = np.median(residual)
        noise_scale = 1.4826 * np.median(np.abs(residual - centre))
    noise_scale = float(np.clip(noise_scale, 5.0, 35.0))

    eval_rows = np.flatnonzero(evaluation)
    sampled = eval_rows[:: int(row_stride)]
    if len(sampled) < 64:
        raise RuntimeError(f"{well}: only {len(sampled)} sampled evaluation rows")
    return {
        "well": well,
        "md": md,
        "z": z,
        "truth": np.full(len(md), np.nan, dtype=np.float64),
        "gr": gr,
        "tvt_input": tvt_input,
        "evaluation": evaluation,
        "eval_rows": eval_rows,
        "sampled_rows": sampled,
        "anchor_index": anchor,
        "anchor_md": float(md[anchor]),
        "anchor_surface": float(tvt_input[anchor] + z[anchor]),
        "type_tvt": type_tvt,
        "type_gr": type_gr,
        "noise_scale": noise_scale,
        "source_rows": horizontal["_source_row"].to_numpy(np.int64),
    }


bundle_dir, manifest = _find_bundle()
data_root = _find_data_root()
sample = pd.read_csv(data_root / "sample_submission.csv")
if "id" not in sample or sample["id"].duplicated().any():
    raise RuntimeError("Invalid sample_submission.csv")
parent_path, parent, parent_value_column = _find_parent_submission(sample)
model_path, prior_path, features_path = _copy_runtime_sources(bundle_dir, manifest)
Q = _bootstrap_v4()
V = Q["V"]
B = Q["B"]

prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
prior = {
    key: np.asarray(value, np.float64) if key == "rate_grid" else value
    for key, value in prior_payload.items()
}
feature_names_expected = json.loads(features_path.read_text(encoding="utf-8"))
model = lgb.Booster(model_file=str(model_path))
if model.num_feature() != len(feature_names_expected):
    raise RuntimeError(
        f"Model/schema mismatch: {model.num_feature()}/{len(feature_names_expected)}"
    )

test_paths = sorted((data_root / "test").glob("*__horizontal_well.csv"))
wells = [path.name.split("__", 1)[0] for path in test_paths]
coarse_by_id = {}
reports = []
for number, well in enumerate(wells, 1):
    try:
        record = _load_test_record(data_root, well, V["ROW_STRIDE"], V["_interp_nan"])
        paths, source_mask, arm_rank, arm_score_z, segments, _ = Q["_candidate_union"](
            record, prior
        )
        features, feature_names = Q["_candidate_evidence"](
            record, prior, paths, segments, arm_rank, arm_score_z, source_mask
        )
        if feature_names != feature_names_expected:
            raise RuntimeError(f"feature schema changed: {len(feature_names)}")
        score = model.predict(features)
        selected_index = int(np.argmax(score))
        prediction = B["_path_prediction"](
            record, prior, paths[selected_index], segments
        )
        eval_rows = record["eval_rows"]
        source_rows = record["source_rows"][eval_rows]
        values = prediction[eval_rows]
        if not np.isfinite(values).all():
            raise RuntimeError("non-finite selected path prediction")
        for source_row, value in zip(source_rows, values):
            coarse_by_id[f"{well}_{int(source_row)}"] = float(value)
        reports.append(
            {
                "well": well,
                "status": "ok",
                "bank_candidates": int(len(paths)),
                "selected_index": selected_index,
                "selected_score": float(score[selected_index]),
                "rows": int(len(eval_rows)),
            }
        )
    except Exception as exc:
        reports.append({"well": well, "status": "failed", "reason": repr(exc)})
    if number % 10 == 0 or number == len(wells):
        failures = sum(report["status"] != "ok" for report in reports)
        print(
            f"V65 test inference {number}/{len(wells)} failures={failures} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
            flush=True,
        )

report_frame = pd.DataFrame(reports)
report_frame.to_csv(WORK / "v65_coarse_inference_report.csv", index=False)
coarse = sample["id"].astype(str).map(coarse_by_id).to_numpy(np.float64)
available = np.isfinite(coarse)
coverage = float(available.mean())
if coverage < 0.98:
    failed = report_frame.loc[report_frame["status"] != "ok"].to_dict("records")
    raise RuntimeError(
        f"Coarse prediction coverage {coverage:.3%} is below 98%; "
        f"refusing to write a partial submission. Failures: {failed[:10]}"
    )
parent_values = parent[parent_value_column].to_numpy(np.float64)
if not np.isfinite(parent_values).all():
    raise RuntimeError("Incumbent submission contains non-finite predictions")
output_values = parent_values.copy()
output_values[available] = (
    (1.0 - BLEND_WEIGHT) * parent_values[available]
    + BLEND_WEIGHT * coarse[available]
)

sample_value_column = next((column for column in ("tvt", "TVT") if column in sample), None)
if sample_value_column is None:
    raise RuntimeError(f"Cannot identify sample target column: {sample.columns.tolist()}")
output = sample.copy()
output[sample_value_column] = output_values
output.to_csv(OUTPUT_PATH, index=False)
if len(output) != len(sample) or not output["id"].astype(str).equals(sample["id"].astype(str)):
    raise RuntimeError("Output/sample alignment changed")

summary = {
    "experiment": "rawdata_lattice_coarse_kaggle_inference_v65",
    "bundle": str(bundle_dir),
    "parent_submission": str(parent_path),
    "output": str(OUTPUT_PATH),
    "test_wells": len(wells),
    "successful_wells": int((report_frame["status"] == "ok").sum()),
    "failed_wells": int((report_frame["status"] != "ok").sum()),
    "sample_rows": len(sample),
    "coarse_rows": int(available.sum()),
    "coverage": coverage,
    "blend_weight_coarse": BLEND_WEIGHT,
    "changed_rows": int(np.sum(np.abs(output_values - parent_values) > 1e-12)),
    "mean_abs_move_ft": float(np.mean(np.abs(output_values - parent_values))),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "instruction": "Submit submission_v64_coarse_w010.csv; do not run later rewrite cells.",
}
(WORK / "v65_coarse_inference_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print("\nV65 FINAL SUMMARY")
print(json.dumps(summary, indent=2), flush=True)
