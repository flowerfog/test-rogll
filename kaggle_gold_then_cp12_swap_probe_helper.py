"""Helpers for an exact Gold -> CP12 order probe in the original notebook.

Run this helper cell once to define the functions. After Gold has produced
submission.csv, call prepare_gold_then_cp12_probe(), run the original CP12 cell
unchanged, then call finalize_gold_then_cp12_probe(). Do not rerun Gold after
the CP12 cell.
"""

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd


SWAP_WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
SWAP_STATE = SWAP_WORK / "gold_then_cp12_probe_state.json"
SWAP_ORIGINAL_BASE = SWAP_WORK / "submission_before_gold_original_for_swap.csv"
SWAP_GOLD_INPUT = SWAP_WORK / "submission_gold_before_cp12.csv"
SWAP_OUTPUT = SWAP_WORK / "submission_gold_then_cp12_a07_c3.csv"
SWAP_AUDIT = SWAP_WORK / "gold_then_cp12_probe_audit.json"


def _swap_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _swap_read(path):
    frame = pd.read_csv(path)
    if not {"id", "tvt"}.issubset(frame.columns):
        raise RuntimeError(f"bad submission columns at {path}: {frame.columns.tolist()}")
    output = frame[["id", "tvt"]].copy()
    output["id"] = output["id"].astype(str)
    output["tvt"] = pd.to_numeric(output["tvt"], errors="coerce")
    if output["id"].duplicated().any() or not np.isfinite(output["tvt"]).all():
        raise RuntimeError(f"invalid submission: {path}")
    return output


def _swap_align(reference, candidate):
    joined = reference[["id", "tvt"]].rename(columns={"tvt": "reference"}).merge(
        candidate[["id", "tvt"]].rename(columns={"tvt": "candidate"}),
        on="id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError(joined["_merge"].value_counts().to_dict())
    joined = joined.drop(columns="_merge")
    joined = reference[["id"]].merge(joined, on="id", how="left", validate="one_to_one")
    return joined


def _swap_move_stats(reference, candidate):
    joined = _swap_align(reference, candidate)
    move = joined["candidate"].to_numpy(float) - joined["reference"].to_numpy(float)
    return {
        "rows": int(len(joined)),
        "mean_abs_move": float(np.mean(np.abs(move))),
        "rms_move": float(np.sqrt(np.mean(move * move))),
        "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
        "max_abs_move": float(np.max(np.abs(move))),
        "changed_rows": int(np.sum(np.abs(move) > 1e-9)),
    }


def prepare_gold_then_cp12_probe():
    """Freeze Gold-only and temporarily route it into the unchanged CP12 cell."""
    before_path = SWAP_WORK / "submission_before_gold.csv"
    current_path = SWAP_WORK / "submission.csv"
    if not before_path.exists():
        raise FileNotFoundError(before_path)
    if not current_path.exists():
        raise FileNotFoundError(current_path)

    before = _swap_read(before_path)
    gold = _swap_read(current_path)
    gold_vs_base = _swap_move_stats(before, gold)
    if gold_vs_base["changed_rows"] == 0:
        raise RuntimeError(
            "Current submission.csv is identical to submission_before_gold.csv. "
            "Run Gold-only first, with CP12 skipped."
        )

    # Preserve both endpoints before temporarily replacing the filename that
    # the original CP12 cell reads.
    before.to_csv(SWAP_ORIGINAL_BASE, index=False)
    gold.to_csv(SWAP_GOLD_INPUT, index=False)
    gold.to_csv(before_path, index=False)

    state = {
        "status": "prepared",
        "original_before_gold": str(SWAP_ORIGINAL_BASE),
        "gold_before_cp12": str(SWAP_GOLD_INPUT),
        "original_before_gold_sha256": _swap_sha256(SWAP_ORIGINAL_BASE),
        "gold_before_cp12_sha256": _swap_sha256(SWAP_GOLD_INPUT),
        "gold_vs_original_base": gold_vs_base,
        "next": (
            "Run the original CP12 cell unchanged now. Do not run Gold after it. "
            "Then call finalize_gold_then_cp12_probe()."
        ),
    }
    SWAP_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print("GOLD -> CP12 PROBE PREPARED")
    print(json.dumps(state, indent=2))
    return state


def restore_original_before_gold():
    """Emergency restore if the CP12 cell fails after prepare()."""
    if not SWAP_ORIGINAL_BASE.exists():
        raise FileNotFoundError(SWAP_ORIGINAL_BASE)
    restored = _swap_read(SWAP_ORIGINAL_BASE)
    restored.to_csv(SWAP_WORK / "submission_before_gold.csv", index=False)
    print("restored:", SWAP_WORK / "submission_before_gold.csv")


def finalize_gold_then_cp12_probe():
    """Archive the swapped endpoint, restore lineage files, and audit movement."""
    current_path = SWAP_WORK / "submission.csv"
    if not SWAP_STATE.exists():
        raise FileNotFoundError(SWAP_STATE)
    if not SWAP_GOLD_INPUT.exists() or not SWAP_ORIGINAL_BASE.exists():
        raise RuntimeError("swap endpoints are incomplete; rerun prepare()")
    if not current_path.exists():
        raise FileNotFoundError(current_path)

    state = json.loads(SWAP_STATE.read_text(encoding="utf-8"))
    if state.get("status") != "prepared":
        raise RuntimeError(f"unexpected swap state: {state}")

    # The original cell leaves these globals behind. This prevents accidentally
    # scoring an old cap=6 or another CP profile under the swapped filename.
    alpha = float(globals().get("CP_ALPHA", np.nan))
    cap = float(globals().get("CP_CAP", np.nan))
    ctrl_n = int(globals().get("CP_CTRL_N", -1))
    if not np.isclose(alpha, 0.7) or not np.isclose(cap, 3.0) or ctrl_n != 12:
        raise RuntimeError(
            f"wrong CP12 configuration: alpha={alpha}, cap={cap}, ctrl_n={ctrl_n}"
        )

    gold = _swap_read(SWAP_GOLD_INPUT)
    swapped = _swap_read(current_path)
    cp_stats = _swap_move_stats(gold, swapped)
    if cp_stats["changed_rows"] == 0:
        raise RuntimeError("CP12 did not change the Gold endpoint")

    swapped.to_csv(SWAP_OUTPUT, index=False)
    swapped.to_csv(current_path, index=False)
    restore_original_before_gold()

    audit = {
        "order": "Gold -> CP12",
        "cp_alpha": alpha,
        "cp_cap": cap,
        "cp_control_points": ctrl_n,
        "original_before_gold": str(SWAP_ORIGINAL_BASE),
        "gold_input": str(SWAP_GOLD_INPUT),
        "swapped_output": str(SWAP_OUTPUT),
        "original_before_gold_sha256": _swap_sha256(SWAP_ORIGINAL_BASE),
        "gold_input_sha256": _swap_sha256(SWAP_GOLD_INPUT),
        "swapped_output_sha256": _swap_sha256(SWAP_OUTPUT),
        "gold_vs_original_base": state["gold_vs_original_base"],
        "cp12_move_on_gold": cp_stats,
        "final_submission": str(current_path),
        "status": "ready_to_submit",
    }
    SWAP_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    SWAP_STATE.write_text(
        json.dumps({**state, "status": "finalized", "audit": str(SWAP_AUDIT)}, indent=2),
        encoding="utf-8",
    )
    print("GOLD -> CP12 PROBE READY")
    print(json.dumps(audit, indent=2))
    print("candidate:", SWAP_OUTPUT)
    print("final Kaggle file:", current_path)
    return audit


print("Gold -> CP12 helper loaded.")
print("After Gold-only finishes, call: prepare_gold_then_cp12_probe()")
