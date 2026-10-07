"""Resume the P2a Fold1 diagnostic after a borderline Fold2 resource gate.

Run this in the same live kernel immediately after
``learned_interval_sgraph_p2a_onefold_v0_cell.py`` stopped at its Fold2 gate.
The selected epoch and all normalisation statistics are reused unchanged.

The original 0.10-ft Fold2 gate is not relaxed or relabelled as a pass.  This
resume merely spends the small remaining cost when the honest Fold2 gain is at
least 0.05 ft, then records Fold1 and causal-control diagnostics.  Fold1 is not
used for any tuning.
"""

from pathlib import Path
import json


REQUIRED_RESUME_P2A = {
    "best_epoch_p2a",
    "selected_tune_gain_p2a",
    "TUNE_PATH_GAIN_GATE_P2A",
    "OUT_DIR_P2A",
    "_json_p2a",
}
missing_resume_p2a = sorted(REQUIRED_RESUME_P2A.difference(globals()))
if missing_resume_p2a:
    raise RuntimeError(
        "Run this in the live kernel stopped by the P2a Fold2 gate; missing: "
        + repr(missing_resume_p2a)
    )

HARD_STOP_GAIN_P2A = 0.05
if int(best_epoch_p2a) <= 0 or float(selected_tune_gain_p2a) < HARD_STOP_GAIN_P2A:
    raise RuntimeError(
        "P2a is below the borderline continuation floor: "
        + repr(
            {
                "selected_epoch": int(best_epoch_p2a),
                "selected_gain": float(selected_tune_gain_p2a),
                "required_gain": HARD_STOP_GAIN_P2A,
            }
        )
    )

SOURCE_NAME_P2A = "learned_interval_sgraph_p2a_onefold_v0_cell.py"
marker_p2a = "# Reinitialise and train the untouched-Fold1 model for the frozen epoch count."
source_candidates_p2a = (
    Path("/root/autodl-tmp") / SOURCE_NAME_P2A,
    Path.cwd() / SOURCE_NAME_P2A,
)
source_path_p2a = next(
    (path for path in source_candidates_p2a if path.exists()), None
)
source_label_p2a = None
source_p2a = None
if source_path_p2a is not None:
    source_p2a = source_path_p2a.read_text(encoding="utf-8")
    source_label_p2a = str(source_path_p2a)
else:
    # Notebook users commonly paste the original cell rather than upload the
    # .py file.  IPython retains that exact source in the current kernel.
    shell_p2a = globals().get("get_ipython", lambda: None)()
    history_p2a = (
        getattr(getattr(shell_p2a, "history_manager", None), "input_hist_raw", [])
        if shell_p2a is not None
        else []
    )
    for history_index_p2a in range(len(history_p2a) - 1, 0, -1):
        candidate_source_p2a = history_p2a[history_index_p2a]
        if (
            marker_p2a in candidate_source_p2a
            and 'outer_phase_p2a = _phase_p2a("outer2345"' in candidate_source_p2a
            and "final_model_p2a = _new_model_p2a(" in candidate_source_p2a
            and "mode_outputs_p2a = {}" in candidate_source_p2a
            and "REQUIRED_RESUME_P2A" not in candidate_source_p2a
        ):
            source_p2a = candidate_source_p2a
            source_label_p2a = f"<ipython-cell-{history_index_p2a}>"
            break

if source_p2a is None:
    raise FileNotFoundError(
        "Cannot recover the original P2a cell from disk or current IPython "
        "history. Upload learned_interval_sgraph_p2a_onefold_v0_cell.py to "
        "/root/autodl-tmp, without restarting this kernel."
    )

if marker_p2a not in source_p2a:
    raise RuntimeError(f"Cannot find the frozen Fold1 marker in {source_label_p2a}")
tail_p2a = marker_p2a + source_p2a.split(marker_p2a, 1)[1]

# Preserve the original resource gate in the final strict classification.
gate_marker_p2a = "gates_p2a = {\n"
if gate_marker_p2a not in tail_p2a:
    raise RuntimeError("Cannot inject the pre-registered Fold2 gate")
tail_p2a = tail_p2a.replace(
    gate_marker_p2a,
    gate_marker_p2a
    + '    "fold2_pre_registered_gain": selected_tune_gain_p2a >= TUNE_PATH_GAIN_GATE_P2A,\n',
    1,
)

print(
    "P2A BORDERLINE RESUME",
    {
        "selected_epoch": int(best_epoch_p2a),
        "fold2_gain": float(selected_tune_gain_p2a),
        "pre_registered_gate": float(TUNE_PATH_GAIN_GATE_P2A),
        "classification": "borderline diagnostic, not a Fold2 pass",
    },
    flush=True,
)
exec(compile(tail_p2a, source_label_p2a + "::<fold1_resume>", "exec"), globals())

fold1_only_gates_p2a = {
    key: bool(value)
    for key, value in verdict_p2a["gates"].items()
    if key != "fold2_pre_registered_gain"
}
fold1_mechanism_pass_p2a = bool(all(fold1_only_gates_p2a.values()))
verdict_p2a["borderline_resume"] = {
    "hard_stop_gain": HARD_STOP_GAIN_P2A,
    "fold2_pre_registered_pass": False,
    "fold1_used_for_tuning": False,
    "fold1_mechanism_pass": fold1_mechanism_pass_p2a,
}
if fold1_mechanism_pass_p2a:
    verdict_p2a["next"] = (
        "HOLD -> Fold1 mechanism passed but Fold2 missed its original gate; "
        "repeat one independent outer fold before end-to-end training"
    )
else:
    verdict_p2a["next"] = (
        "FAIL -> Fold1 mechanism/controls also failed; close learned long-edge residual"
    )
verdict_p2a["strict_pass"] = False
_json_p2a(OUT_DIR_P2A / "verdict.json", verdict_p2a)

print("\nP2A BORDERLINE CLASSIFICATION")
print(json.dumps(verdict_p2a["borderline_resume"], indent=2), flush=True)
print("next:", verdict_p2a["next"], flush=True)
