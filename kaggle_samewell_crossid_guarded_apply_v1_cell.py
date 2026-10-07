"""Final fail-closed Kaggle cell for guarded cross-ID same-well transfer.

Run after every normal prediction/post-processing stage has written the final
mainline candidate to /kaggle/working/submission.csv. The cell snapshots that
file, runs the frozen read-only retrieval audit, and applies train-TVТ transfer
only to candidates passing the frozen full/graded guards.

Any exception restores the snapshot. This is intentionally the final
prediction-changing cell.
"""

from __future__ import annotations

import glob
import hashlib
import json
import runpy
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_APPLY = "kaggle_samewell_crossid_guarded_apply_v1"


def _find_audit_source() -> Path:
    explicit = globals().get("SAMEWELL_AUDIT_SOURCE")
    candidates = [
        Path(explicit) if explicit else None,
        Path.cwd() / "samewell_crossid_retrieval_audit_v0_cell.py",
        Path(__file__).resolve().with_name("samewell_crossid_retrieval_audit_v0_cell.py")
        if "__file__" in globals()
        else None,
    ]
    candidates.extend(
        Path(path)
        for path in glob.glob(
            "/kaggle/input/**/samewell_crossid_retrieval_audit_v0_cell.py",
            recursive=True,
        )
    )
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "samewell_crossid_retrieval_audit_v0_cell.py was not found"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _axis_values(frame: pd.DataFrame, axis_column: str, value_column: str):
    work = pd.DataFrame(
        {
            "axis": pd.to_numeric(frame[axis_column], errors="coerce"),
            "value": pd.to_numeric(frame[value_column], errors="coerce"),
        }
    )
    work = (
        work.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .groupby("axis", sort=True, as_index=False)
        .mean(numeric_only=True)
    )
    if len(work) < 2:
        raise RuntimeError(f"too few finite {axis_column}/{value_column} rows")
    return (
        work["axis"].to_numpy(np.float64),
        work["value"].to_numpy(np.float64),
    )


def _validate_submission(path: Path, sample_path: Path):
    submission = pd.read_csv(path)
    sample = pd.read_csv(sample_path)
    if list(submission.columns) != ["id", "tvt"]:
        raise RuntimeError(f"unexpected submission columns: {submission.columns.tolist()}")
    if len(submission) != len(sample):
        raise RuntimeError(
            f"submission/sample rows differ: {len(submission)}/{len(sample)}"
        )
    if not submission["id"].astype(str).equals(sample["id"].astype(str)):
        raise RuntimeError("submission id order differs from sample_submission.csv")
    values = pd.to_numeric(submission["tvt"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("submission contains non-finite predictions")
    return submission, sample


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


started_apply = time.time()
WORK_APPLY = Path(
    globals().get(
        "SAMEWELL_WORK_DIR",
        "/kaggle/working" if Path("/kaggle/working").exists() else Path.cwd(),
    )
)
WORK_APPLY.mkdir(parents=True, exist_ok=True)
SUBMISSION_APPLY = WORK_APPLY / "submission.csv"
BEFORE_APPLY = WORK_APPLY / "submission_before_samewell_crossid.csv"
FINAL_COPY_APPLY = WORK_APPLY / "submission_samewell_crossid.csv"
AUDIT_OUTPUT_APPLY = WORK_APPLY / "samewell_crossid_retrieval_audit_v0"

if not SUBMISSION_APPLY.exists():
    raise FileNotFoundError(
        f"{SUBMISSION_APPLY} does not exist; run the complete mainline first"
    )
shutil.copy2(SUBMISSION_APPLY, BEFORE_APPLY)
before_hash_apply = _sha256(BEFORE_APPLY)

application_rows = []
failure_apply = None
try:
    audit_source_apply = _find_audit_source()
    audit_apply = runpy.run_path(
        str(audit_source_apply),
        init_globals={"SAMEWELL_OUTPUT_DIR": AUDIT_OUTPUT_APPLY},
    )
    data_root_apply = Path(audit_apply["DATA_ROOT"])
    summary_apply = audit_apply["summary_frame"].copy()
    candidates_apply = audit_apply["candidate_frame"].copy()

    sample_path_apply = data_root_apply / "sample_submission.csv"
    submission_apply, sample_apply = _validate_submission(
        BEFORE_APPLY, sample_path_apply
    )
    submission_apply["id"] = submission_apply["id"].astype(str)
    parsed_apply = submission_apply["id"].str.rsplit("_", n=1, expand=True)
    submission_apply["well"] = parsed_apply[0]
    submission_apply["row_idx"] = pd.to_numeric(
        parsed_apply[1], errors="raise"
    ).astype(int)
    prediction_apply = submission_apply["tvt"].to_numpy(np.float64).copy()

    eligible_apply = summary_apply[
        summary_apply["suggested_action"].isin(["full", "graded"])
    ]
    for summary_row_apply in eligible_apply.itertuples(index=False):
        query_well_apply = str(summary_row_apply.query_well)
        candidate_options_apply = candidates_apply[
            candidates_apply["query_well"].astype(str).eq(query_well_apply)
            & candidates_apply["train_well"].astype(str).eq(
                str(summary_row_apply.top1_train_well)
            )
            & candidates_apply["status"].astype(str).eq("ok")
        ].sort_values("fine_score", ascending=False)
        if candidate_options_apply.empty:
            raise RuntimeError(
                f"missing top candidate details for {query_well_apply}"
            )
        candidate_apply = candidate_options_apply.iloc[0]
        action_apply = str(summary_row_apply.suggested_action)
        weight_apply = 1.0 if action_apply == "full" else 0.85

        # Recheck all frozen deployment gates before changing any row.
        prefix_rmse_apply = float(candidate_apply["prefix_bias_rmse"])
        gr_ncc_apply = float(candidate_apply["gr_smooth21_ncc"])
        coverage_apply = float(candidate_apply["eval_overlap_coverage"])
        if action_apply == "full":
            gate_apply = (
                prefix_rmse_apply <= 0.25
                and gr_ncc_apply >= 0.990
                and coverage_apply >= 0.80
            )
        else:
            gate_apply = (
                prefix_rmse_apply <= 0.75
                and gr_ncc_apply >= 0.970
                and coverage_apply >= 0.50
                and float(summary_row_apply.fine_margin) >= 0.02
            )
        if not gate_apply:
            raise RuntimeError(
                f"frozen gate mismatch for {query_well_apply}/{action_apply}"
            )

        train_well_apply = str(candidate_apply["train_well"])
        train_path_apply = (
            data_root_apply
            / "train"
            / f"{train_well_apply}__horizontal_well.csv"
        )
        test_path_apply = (
            data_root_apply
            / "test"
            / f"{query_well_apply}__horizontal_well.csv"
        )
        train_frame_apply = pd.read_csv(train_path_apply, usecols=["MD", "TVT"])
        test_frame_apply = pd.read_csv(test_path_apply, usecols=["MD"])
        train_md_apply, train_tvt_apply = _axis_values(
            train_frame_apply, "MD", "TVT"
        )
        test_md_apply = pd.to_numeric(
            test_frame_apply["MD"], errors="coerce"
        ).to_numpy(np.float64)
        finite_test_md_apply = test_md_apply[np.isfinite(test_md_apply)]
        if len(finite_test_md_apply) < 2:
            raise RuntimeError(f"too few test MD rows for {query_well_apply}")
        query_md_origin_apply = float(np.min(finite_test_md_apply))
        mapped_md_apply = float(
            candidate_apply["alignment_start_train_md"]
        ) + float(candidate_apply["alignment_signed_scale"]) * (
            test_md_apply - query_md_origin_apply
        )

        positions_apply = submission_apply.index[
            submission_apply["well"].eq(query_well_apply)
        ].to_numpy(int)
        if not len(positions_apply):
            raise RuntimeError(f"submission has no rows for {query_well_apply}")
        row_indices_apply = submission_apply.loc[
            positions_apply, "row_idx"
        ].to_numpy(int)
        if (
            row_indices_apply.min() < 0
            or row_indices_apply.max() >= len(test_frame_apply)
        ):
            raise RuntimeError(f"row index is out of range for {query_well_apply}")
        mapped_eval_apply = mapped_md_apply[row_indices_apply]
        current_apply = prediction_apply[positions_apply].copy()
        in_range_apply = (
            np.isfinite(mapped_eval_apply)
            & (mapped_eval_apply >= train_md_apply[0])
            & (mapped_eval_apply <= train_md_apply[-1])
        )
        if not in_range_apply.any():
            raise RuntimeError(f"no transferable rows for {query_well_apply}")

        prefix_bias_apply = float(candidate_apply["prefix_bias"])
        if not np.isfinite(prefix_bias_apply):
            prefix_bias_apply = 0.0
        transferred_apply = np.full(len(positions_apply), np.nan)
        transferred_apply[in_range_apply] = (
            np.interp(
                mapped_eval_apply[in_range_apply],
                train_md_apply,
                train_tvt_apply,
            )
            + prefix_bias_apply
        )
        updated_apply = current_apply.copy()
        updated_apply[in_range_apply] = (
            weight_apply * transferred_apply[in_range_apply]
            + (1.0 - weight_apply) * current_apply[in_range_apply]
        )

        reanchored_rows_apply = 0
        lower_apply = np.isfinite(mapped_eval_apply) & (
            mapped_eval_apply < train_md_apply[0]
        )
        upper_apply = np.isfinite(mapped_eval_apply) & (
            mapped_eval_apply > train_md_apply[-1]
        )
        if lower_apply.any():
            boundary_position_apply = int(
                np.flatnonzero(in_range_apply)[
                    np.argmin(mapped_eval_apply[in_range_apply])
                ]
            )
            shift_apply = (
                transferred_apply[boundary_position_apply]
                - current_apply[boundary_position_apply]
            )
            updated_apply[lower_apply] = (
                current_apply[lower_apply] + weight_apply * shift_apply
            )
            reanchored_rows_apply += int(lower_apply.sum())
        if upper_apply.any():
            boundary_position_apply = int(
                np.flatnonzero(in_range_apply)[
                    np.argmax(mapped_eval_apply[in_range_apply])
                ]
            )
            shift_apply = (
                transferred_apply[boundary_position_apply]
                - current_apply[boundary_position_apply]
            )
            updated_apply[upper_apply] = (
                current_apply[upper_apply] + weight_apply * shift_apply
            )
            reanchored_rows_apply += int(upper_apply.sum())
        if not np.isfinite(updated_apply).all():
            raise RuntimeError(f"non-finite update for {query_well_apply}")
        prediction_apply[positions_apply] = updated_apply
        application_rows.append(
            {
                "query_well": query_well_apply,
                "train_well": train_well_apply,
                "action": action_apply,
                "weight": weight_apply,
                "exact_id": bool(candidate_apply["exact_id"]),
                "rows": int(len(positions_apply)),
                "in_range_rows": int(in_range_apply.sum()),
                "reanchored_rows": reanchored_rows_apply,
                "mean_abs_move": float(
                    np.mean(np.abs(updated_apply - current_apply))
                ),
                "max_abs_move": float(
                    np.max(np.abs(updated_apply - current_apply))
                ),
                "prefix_bias_rmse": prefix_rmse_apply,
                "gr_smooth21_ncc": gr_ncc_apply,
                "eval_overlap_coverage": coverage_apply,
            }
        )

    final_apply = submission_apply[["id"]].copy()
    final_apply["tvt"] = prediction_apply
    # Restore the exact sample order even after internal grouping columns.
    final_apply = sample_apply[["id"]].merge(final_apply, on="id", how="left")
    if final_apply["tvt"].isna().any() or not np.isfinite(
        final_apply["tvt"].to_numpy(float)
    ).all():
        raise RuntimeError("final same-well candidate contains non-finite values")
    final_apply.to_csv(FINAL_COPY_APPLY, index=False)
    final_apply.to_csv(SUBMISSION_APPLY, index=False)
    _validate_submission(SUBMISSION_APPLY, sample_path_apply)
except Exception as error_apply:
    failure_apply = f"{type(error_apply).__name__}: {error_apply}"
    shutil.copy2(BEFORE_APPLY, SUBMISSION_APPLY)

application_frame_apply = pd.DataFrame(application_rows)
application_frame_apply.to_csv(
    WORK_APPLY / "samewell_crossid_applied_wells.csv", index=False
)
after_hash_apply = _sha256(SUBMISSION_APPLY)
verdict_apply = {
    "experiment": EXPERIMENT_APPLY,
    "status": "applied" if failure_apply is None else "restored_after_failure",
    "failure": failure_apply,
    "before": str(BEFORE_APPLY),
    "final_copy": str(FINAL_COPY_APPLY) if FINAL_COPY_APPLY.exists() else None,
    "submission": str(SUBMISSION_APPLY),
    "before_sha256": before_hash_apply,
    "after_sha256": after_hash_apply,
    "changed": before_hash_apply != after_hash_apply,
    "applied_wells": int(len(application_frame_apply)),
    "cross_id_applied_wells": int(
        (~application_frame_apply["exact_id"]).sum()
    )
    if len(application_frame_apply)
    else 0,
    "applied_rows": int(application_frame_apply["rows"].sum())
    if len(application_frame_apply)
    else 0,
    "mean_abs_move_over_applied_wells": float(
        application_frame_apply["mean_abs_move"].mean()
    )
    if len(application_frame_apply)
    else 0.0,
    "submission_modified_by_this_cell": failure_apply is None
    and before_hash_apply != after_hash_apply,
    "fail_closed": True,
    "elapsed_minutes": (time.time() - started_apply) / 60.0,
}
verdict_apply = _jsonable(verdict_apply)
(WORK_APPLY / "samewell_crossid_apply_verdict.json").write_text(
    json.dumps(verdict_apply, indent=2), encoding="utf-8"
)
print("\nSAME-WELL APPLIED WELLS")
print(
    application_frame_apply.to_string(index=False)
    if len(application_frame_apply)
    else "<none>"
)
print("\nSAME-WELL FINAL VERDICT")
print(json.dumps(verdict_apply, indent=2))
print("final Kaggle file:", SUBMISSION_APPLY)

