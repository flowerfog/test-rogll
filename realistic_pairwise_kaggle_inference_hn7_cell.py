"""Final Kaggle cell for the validated HN3 tournament ensemble.

Run only after the exact 6.377 CP8 + Neighbor notebook has written
/kaggle/working/submission.csv. Upload the HN6 deployment zip/directory as a
Kaggle dataset. The cell preserves that parent once, writes the named HN output,
then copies the validated result to submission.csv for Kaggle submission.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


STARTED = time.time()
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
RUNTIME = WORK / "hn7_runtime"
RUNTIME.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = WORK / "submission_hn7_w0275.csv"
FINAL_SUBMISSION_PATH = WORK / "submission.csv"
PARENT_SNAPSHOT_PATH = WORK / "submission_6p377_before_hn7.csv"
PAIR_BATCH = 65536


def _find_bundle():
    candidates = []
    configured = os.environ.get("ROGII_HN6_BUNDLE")
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
        if payload.get("experiment") == "realistic_pairwise_deployment_bundle_hn6":
            matches.append((path.stat().st_mtime, path.resolve(), payload))
    if not matches:
        archives = []
        for root in (Path("/kaggle/input"), Path.cwd()):
            if root.exists():
                archives.extend(root.rglob("realistic_pairwise_deployment_bundle_hn6.zip"))
        for archive in archives:
            unpacked = WORK / "hn6_bundle_unpacked"
            unpacked.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(archive), str(unpacked))
            manifest_path = unpacked / "bundle.json"
            if not manifest_path.is_file():
                continue
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("experiment") == "realistic_pairwise_deployment_bundle_hn6":
                matches.append((archive.stat().st_mtime, manifest_path.resolve(), payload))
    if not matches:
        raise FileNotFoundError("Upload the complete HN6 deployment directory to Kaggle")
    _, manifest_path, payload = sorted(matches, reverse=True)[0]
    return manifest_path.parent, payload


def _find_data_root():
    candidates = []
    configured = os.environ.get("ROGII_DATA_ROOT")
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
    candidates = []
    configured = os.environ.get("ROGII_PARENT_SUBMISSION")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            PARENT_SNAPSHOT_PATH,
            WORK / "final_submission.csv",
            WORK / "submission_6p377.csv",
            FINAL_SUBMISSION_PATH,
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
            value_column = next((name for name in ("tvt", "TVT") if name in frame), None)
            if value_column is not None:
                return path.resolve(), frame, value_column
    raise FileNotFoundError(
        "Run the exact 6.377 CP8 + Neighbor pipeline first, or set "
        "ROGII_PARENT_SUBMISSION to its aligned CSV"
    )


def _copy_runtime(bundle_dir, manifest):
    for name in manifest["source_files"]:
        source = bundle_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, RUNTIME / name)
    prior_path = bundle_dir / "structural_prior.json"
    features_path = bundle_dir / manifest["selected_features"]
    model_paths = [bundle_dir / name for name in manifest["models"]]
    for path in [prior_path, features_path, *model_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    return prior_path, features_path, model_paths


def _bootstrap_v4():
    source = RUNTIME / "rawdata_lattice_coarse_ranker_v4_cell.py"
    text = source.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in text:
        raise RuntimeError(f"V4 bootstrap marker missing from {source}")
    old_cwd = Path.cwd()
    try:
        os.chdir(RUNTIME)
        namespace = {
            "__name__": "hn7_kaggle_inference_definitions",
            "__file__": str(source),
        }
        exec(compile(text.split(marker, 1)[0], str(source), "exec"), namespace)
    finally:
        os.chdir(old_cwd)
    return namespace


def _load_test_record(data_root, well, row_stride, interp_nan):
    horizontal = pd.read_csv(data_root / "test" / f"{well}__horizontal_well.csv")
    horizontal["_source_row"] = np.arange(len(horizontal), dtype=np.int64)
    horizontal = horizontal.sort_values("MD", kind="stable").reset_index(drop=True)
    typewell = pd.read_csv(
        data_root / "test" / f"{well}__typewell.csv"
    ).sort_values("TVT", kind="stable")

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
    reference = interp_nan(tvt_input[known_rows], type_tvt, type_gr)
    residual = gr[known_rows] - reference
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


def _ensemble_tournament(models, values):
    count = len(values)
    if count == 1:
        return np.ones(1, np.float64)
    sums = np.zeros(count, np.float64)
    seen = np.zeros(count, np.int64)
    left_all, right_all = np.triu_indices(count, 1)
    for start in range(0, len(left_all), PAIR_BATCH):
        left = left_all[start : start + PAIR_BATCH]
        right = right_all[start : start + PAIR_BATCH]
        delta = values[left] - values[right]
        probability = np.mean(
            np.vstack([model.predict(delta) for model in models]), axis=0
        )
        sums[left] += probability
        sums[right] += 1.0 - probability
        seen[left] += 1
        seen[right] += 1
    return sums / np.maximum(seen, 1)


bundle_dir, manifest = _find_bundle()
data_root = _find_data_root()
sample = pd.read_csv(data_root / "sample_submission.csv")
if "id" not in sample or sample["id"].duplicated().any():
    raise RuntimeError("Invalid sample_submission.csv")
parent_path, parent, parent_value_column = _find_parent_submission(sample)
if parent_path == FINAL_SUBMISSION_PATH.resolve() and not PARENT_SNAPSHOT_PATH.exists():
    shutil.copy2(parent_path, PARENT_SNAPSHOT_PATH)
    parent_path = PARENT_SNAPSHOT_PATH.resolve()
prior_path, features_path, model_paths = _copy_runtime(bundle_dir, manifest)
Q = _bootstrap_v4()
V = Q["V"]
B = Q["B"]

prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
prior = {
    key: np.asarray(value, np.float64) if key == "rate_grid" else value
    for key, value in prior_payload.items()
}
selected_features = json.loads(features_path.read_text(encoding="utf-8"))
models = [lgb.Booster(model_file=str(path)) for path in model_paths]
if any(model.num_feature() != len(selected_features) for model in models):
    raise RuntimeError("HN model/selected-feature schema mismatch")

blend_weight = float(manifest["blend_weight_hn"])
if abs(blend_weight - 0.275) > 1e-12:
    raise RuntimeError(f"Unexpected HN blend weight: {blend_weight}")
menu_size = int(manifest["menu_size"])

test_paths = sorted((data_root / "test").glob("*__horizontal_well.csv"))
wells = [path.name.split("__", 1)[0] for path in test_paths]
hn_by_id = {}
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
        feature_index = {name: index for index, name in enumerate(feature_names)}
        missing = [name for name in selected_features if name not in feature_index]
        if missing:
            raise RuntimeError(f"missing selected features: {missing[:10]}")
        columns = np.asarray([feature_index[name] for name in selected_features], np.int64)
        legal_rank = np.min(np.asarray(arm_rank, np.float64), axis=1)
        menu = np.argsort(legal_rank, kind="stable")[: min(menu_size, len(paths))]
        values = np.asarray(features[menu][:, columns], np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=12.0, neginf=-12.0)
        score = _ensemble_tournament(models, values)
        selected_local = int(np.argmax(score))
        selected_index = int(menu[selected_local])
        prediction = B["_path_prediction"](
            record, prior, paths[selected_index], segments
        )
        eval_rows = record["eval_rows"]
        source_rows = record["source_rows"][eval_rows]
        output_values = prediction[eval_rows]
        if not np.isfinite(output_values).all():
            raise RuntimeError("non-finite HN prediction")
        for source_row, value in zip(source_rows, output_values):
            hn_by_id[f"{well}_{int(source_row)}"] = float(value)
        sorted_score = np.sort(score)
        margin = float(sorted_score[-1] - sorted_score[-2]) if len(score) > 1 else 0.0
        reports.append(
            {
                "well": well,
                "status": "ok",
                "bank_candidates": int(len(paths)),
                "menu_candidates": int(len(menu)),
                "selected_index": selected_index,
                "tournament_margin": margin,
                "rows": int(len(eval_rows)),
            }
        )
    except Exception as exc:
        reports.append({"well": well, "status": "failed", "reason": repr(exc)})
    if number % 10 == 0 or number == len(wells):
        failures = sum(report["status"] != "ok" for report in reports)
        print(
            f"HN7 test inference {number}/{len(wells)} failures={failures} "
            f"elapsed_min={(time.time() - STARTED) / 60.0:.1f}",
            flush=True,
        )

report_frame = pd.DataFrame(reports)
report_frame.to_csv(WORK / "hn7_inference_report.csv", index=False)
hn_values = sample["id"].astype(str).map(hn_by_id).to_numpy(np.float64)
available = np.isfinite(hn_values)
coverage = float(available.mean())
if coverage < 0.98:
    failed = report_frame.loc[report_frame["status"] != "ok"].to_dict("records")
    raise RuntimeError(
        f"HN coverage {coverage:.3%} is below 98%; refusing partial output. "
        f"Failures: {failed[:10]}"
    )

parent_values = parent[parent_value_column].to_numpy(np.float64)
if not np.isfinite(parent_values).all():
    raise RuntimeError("Parent submission contains non-finite predictions")
final_values = parent_values.copy()
final_values[available] = (
    (1.0 - blend_weight) * parent_values[available]
    + blend_weight * hn_values[available]
)
sample_value_column = next((name for name in ("tvt", "TVT") if name in sample), None)
if sample_value_column is None:
    raise RuntimeError(f"Cannot identify target column: {sample.columns.tolist()}")
output = sample.copy()
output[sample_value_column] = final_values
output.to_csv(OUTPUT_PATH, index=False)
if len(output) != len(sample) or not output["id"].astype(str).equals(sample["id"].astype(str)):
    raise RuntimeError("Output/sample alignment changed")
shutil.copy2(OUTPUT_PATH, FINAL_SUBMISSION_PATH)

summary = {
    "experiment": "realistic_pairwise_kaggle_inference_hn7",
    "bundle": str(bundle_dir),
    "parent_submission": str(parent_path),
    "output": str(OUTPUT_PATH),
    "final_submission": str(FINAL_SUBMISSION_PATH),
    "test_wells": len(wells),
    "successful_wells": int(report_frame["status"].eq("ok").sum()),
    "failed_wells": int(report_frame["status"].ne("ok").sum()),
    "sample_rows": len(sample),
    "hn_rows": int(available.sum()),
    "coverage": coverage,
    "blend_weight_hn": blend_weight,
    "changed_rows": int(np.sum(np.abs(final_values - parent_values) > 1e-12)),
    "mean_abs_hn_parent_disagreement_ft": float(
        np.mean(np.abs(hn_values[available] - parent_values[available]))
    ),
    "mean_abs_final_move_ft": float(np.mean(np.abs(final_values - parent_values))),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "instruction": "Submit submission.csv; it is an exact copy of submission_hn7_w0275.csv.",
}
(WORK / "hn7_inference_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print("\nHN7 FINAL SUMMARY")
print(json.dumps(summary, indent=2), flush=True)
print(f"\nSubmission ready: {FINAL_SUBMISSION_PATH}", flush=True)
