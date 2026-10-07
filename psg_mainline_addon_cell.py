"""Final-cell PSG candidate runner and guarded mainline blend for Kaggle.

Prerequisites
-------------
1. The CP12 -> Gold -> full-progress geometric Neighbor mainline has already
   written ``/kaggle/working/submission.csv``.
2. ``kaggle_notebook.py`` (the standalone PSG training/inference script) is
   attached as a Kaggle dataset, or its exact path is assigned to
   ``PSG_SCRIPT_OVERRIDE`` below.

The PSG script runs in an isolated namespace. Its ``submission.csv`` is saved
as a candidate, then the exact upstream mainline is restored before blending.
The default output is deliberately conservative because PSG's measured
standalone Fold-1 RMSE is materially worse than the current mainline.
"""

from __future__ import annotations

import json
import os
import runpy
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "psg_guarded_mainline_addon_v1"
WORK = Path("/kaggle/working")
UPSTREAM = WORK / "submission.csv"
UPSTREAM_SNAPSHOT = WORK / "submission_before_psg.csv"
PSG_RAW = WORK / "submission_psg_raw.csv"

# Kaggle dataset containing the standalone PSG training/inference script.
PSG_SCRIPT_OVERRIDE = (
    "/kaggle/input/datasets/fogflower/psg-submission/"
    "kaggle_psg_submission_notebook.py"
)

# ``safe`` becomes submission.csv. ``probe`` and ``legacy`` are exported for
# diagnostics or separate submissions; neither silently replaces the mainline.
ACTIVE_PROFILE = "safe"

PROFILES = {
    "safe": {
        "coverage_weights": ((0.40, 0.15), (0.60, 0.10), (0.75, 0.04), (1.01, 0.0)),
        "taper_start_ft": 500.0,
        "taper_full_ft": 1500.0,
        "raw_disagreement_cap_ft": 40.0,
        "applied_move_cap_ft": 3.0,
    },
    "probe": {
        "coverage_weights": ((0.40, 0.25), (0.60, 0.15), (0.75, 0.07), (1.01, 0.0)),
        "taper_start_ft": 250.0,
        "taper_full_ft": 1250.0,
        "raw_disagreement_cap_ft": 40.0,
        "applied_move_cap_ft": 4.0,
    },
    # Reproduces the weights proposed in the original add-on. It is retained
    # only as an explicit probe because it has no validation against the final
    # 6.781 mainline and can move low-coverage rows by almost 16 ft.
    "legacy": {
        "coverage_weights": ((0.40, 0.40), (0.60, 0.25), (0.75, 0.10), (1.01, 0.04)),
        "taper_start_ft": 0.0,
        "taper_full_ft": 0.0,
        "raw_disagreement_cap_ft": 40.0,
        "applied_move_cap_ft": 16.0,
    },
}


def _submission_columns(frame: pd.DataFrame) -> tuple[str, str]:
    id_candidates = [column for column in frame.columns if column.lower() == "id"]
    if len(id_candidates) != 1:
        raise RuntimeError(f"Expected one id column, found {frame.columns.tolist()}")
    id_column = id_candidates[0]
    targets = [column for column in frame.columns if column != id_column]
    if len(targets) != 1:
        raise RuntimeError(f"Expected one prediction column, found {targets}")
    return id_column, targets[0]


def _read_submission(path: Path, label: str) -> tuple[pd.DataFrame, str, str]:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    frame = pd.read_csv(path)
    id_column, target_column = _submission_columns(frame)
    if frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise RuntimeError(f"{label} has missing or duplicate IDs")
    values = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label} contains non-finite predictions")
    return frame, id_column, target_column


def _parse_ids(ids: pd.Series) -> pd.DataFrame:
    parts = ids.astype(str).str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2:
        raise RuntimeError("Submission IDs must end in '_<horizontal-row-index>'")
    rows = pd.to_numeric(parts[1], errors="raise").astype(int)
    return pd.DataFrame({"well": parts[0].astype(str), "row": rows}, index=ids.index)


def _find_psg_script() -> Path:
    if PSG_SCRIPT_OVERRIDE is not None:
        path = Path(PSG_SCRIPT_OVERRIDE)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = []
    working_copy = WORK / "kaggle_notebook.py"
    if working_copy.exists():
        candidates.append(working_copy)
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(input_root.rglob("kaggle_notebook.py"))
    candidates = sorted({path.resolve() for path in candidates})
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one PSG kaggle_notebook.py. Found: "
            + repr([str(path) for path in candidates])
            + ". Set PSG_SCRIPT_OVERRIDE to the intended file."
        )
    return candidates[0]


def _run_psg_candidate(script: Path) -> dict:
    old_cwd = Path.cwd()
    namespace = None
    try:
        os.chdir(WORK)
        namespace = runpy.run_path(str(script), run_name="__psg_addon__")
        generated, _, _ = _read_submission(UPSTREAM, "PSG-generated submission")
        generated.to_csv(PSG_RAW, index=False)
    finally:
        os.chdir(old_cwd)
        # The upstream vector is restored even when PSG training/inference fails.
        shutil.copy2(UPSTREAM_SNAPSHOT, UPSTREAM)
    if namespace is None:
        raise RuntimeError("PSG script returned no namespace")
    return namespace


def _coverage_weight(coverage: float, rules) -> float:
    for upper, weight in rules:
        if coverage < upper:
            return float(weight)
    return 0.0


def _taper(distance_ft: np.ndarray, start_ft: float, full_ft: float) -> np.ndarray:
    if full_ft <= start_ft:
        return np.ones_like(distance_ft, dtype=float)
    return np.clip((distance_ft - start_ft) / (full_ft - start_ft), 0.0, 1.0)


def _well_arrays(namespace: dict, well: str) -> tuple[np.ndarray, np.ndarray, int]:
    cache = namespace.get("CACHE", {})
    if well not in cache:
        raise KeyError(f"PSG CACHE does not contain test well {well}")
    record = cache[well]
    gr = np.asarray(record["GR"], dtype=float)
    md = np.asarray(record["MD"], dtype=float)
    prediction_start = int(record["real_ps"])
    if len(gr) != len(md) or not 0 < prediction_start < len(md):
        raise RuntimeError(
            f"Invalid PSG record for {well}: n={len(md)}, ps={prediction_start}"
        )
    return gr, md, prediction_start


def _build_profile(
    profile_name: str,
    config: dict,
    base: pd.DataFrame,
    psg_values: np.ndarray,
    parsed_ids: pd.DataFrame,
    namespace: dict,
    id_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base_values = base[target_column].to_numpy(float)
    output = base_values.copy()
    diagnostics = []

    for well, positions_index in parsed_ids.groupby("well", sort=False).groups.items():
        positions = np.asarray(list(positions_index), dtype=int)
        horizontal_rows = parsed_ids.loc[positions, "row"].to_numpy(int)
        gr, md, prediction_start = _well_arrays(namespace, str(well))
        if horizontal_rows.min() < prediction_start or horizontal_rows.max() >= len(md):
            raise RuntimeError(
                f"Submission rows for {well} are outside its evaluation interval"
            )

        coverage = float(np.isfinite(gr[horizontal_rows]).mean())
        well_weight = _coverage_weight(coverage, config["coverage_weights"])
        distance = md[horizontal_rows] - md[prediction_start - 1]
        taper = _taper(
            distance,
            float(config["taper_start_ft"]),
            float(config["taper_full_ft"]),
        )
        raw_delta = psg_values[positions] - base_values[positions]
        eligible = (
            np.isfinite(raw_delta)
            & (np.abs(raw_delta) < float(config["raw_disagreement_cap_ft"]))
        )
        move = np.zeros(len(positions), dtype=float)
        proposed = well_weight * taper * raw_delta
        move[eligible] = np.clip(
            proposed[eligible],
            -float(config["applied_move_cap_ft"]),
            float(config["applied_move_cap_ft"]),
        )
        output[positions] += move

        abs_move = np.abs(move)
        diagnostics.append(
            {
                "profile": profile_name,
                "well": str(well),
                "rows": int(len(positions)),
                "gr_coverage": coverage,
                "well_weight": well_weight,
                "eligible_rows": int(eligible.sum()),
                "changed_rows": int((abs_move > 1e-12).sum()),
                "mean_abs_move": float(abs_move.mean()),
                "p95_abs_move": float(np.quantile(abs_move, 0.95)),
                "max_abs_move": float(abs_move.max()),
                "mean_raw_disagreement": float(np.mean(np.abs(raw_delta))),
            }
        )

    if not np.isfinite(output).all():
        raise RuntimeError(f"{profile_name} produced non-finite predictions")
    move = output - base_values
    frame = base[[id_column]].copy()
    frame[target_column] = output
    diagnostic_frame = pd.DataFrame(diagnostics)
    summary = {
        "profile": profile_name,
        "rows": int(len(frame)),
        "wells": int(parsed_ids["well"].nunique()),
        "changed_wells": int(
            diagnostic_frame.loc[diagnostic_frame["changed_rows"] > 0, "well"].nunique()
        ),
        "changed_rows": int((np.abs(move) > 1e-12).sum()),
        "mean_abs_move": float(np.mean(np.abs(move))),
        "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
        "max_abs_move": float(np.max(np.abs(move))),
    }
    return frame, diagnostic_frame, summary


started = time.time()
if ACTIVE_PROFILE not in PROFILES:
    raise KeyError(f"Unknown ACTIVE_PROFILE={ACTIVE_PROFILE!r}")

base, id_column, target_column = _read_submission(UPSTREAM, "upstream mainline")
base.to_csv(UPSTREAM_SNAPSHOT, index=False)
script_path = _find_psg_script()
print("PSG script:", script_path)
print("Saved exact upstream:", UPSTREAM_SNAPSHOT)

psg_namespace = _run_psg_candidate(script_path)
psg, psg_id_column, psg_target_column = _read_submission(PSG_RAW, "raw PSG candidate")

if set(psg[psg_id_column].astype(str)) != set(base[id_column].astype(str)):
    raise RuntimeError("PSG and mainline submissions do not contain the same IDs")
psg_lookup_frame = psg.copy()
psg_lookup_frame[psg_id_column] = psg_lookup_frame[psg_id_column].astype(str)
psg_lookup = psg_lookup_frame.set_index(psg_id_column)[psg_target_column]
psg_values = base[id_column].astype(str).map(psg_lookup).to_numpy(float)
if not np.isfinite(psg_values).all():
    raise RuntimeError("PSG candidate could not be aligned exactly to the mainline IDs")

parsed_ids = _parse_ids(base[id_column])
profile_summaries = {}
diagnostic_frames = []
profile_outputs = {}
for profile_name, profile_config in PROFILES.items():
    profile_frame, diagnostic_frame, profile_summary = _build_profile(
        profile_name,
        profile_config,
        base,
        psg_values,
        parsed_ids,
        psg_namespace,
        id_column,
        target_column,
    )
    output_path = WORK / f"submission_psg_{profile_name}.csv"
    profile_frame.to_csv(output_path, index=False)
    profile_outputs[profile_name] = output_path
    profile_summaries[profile_name] = profile_summary
    diagnostic_frames.append(diagnostic_frame)

active_output = profile_outputs[ACTIVE_PROFILE]
shutil.copy2(active_output, UPSTREAM)
pd.concat(diagnostic_frames, ignore_index=True).to_csv(
    WORK / "psg_blend_diagnostics.csv", index=False
)

verdict = {
    "experiment": EXPERIMENT,
    "upstream": str(UPSTREAM_SNAPSHOT),
    "psg_script": str(script_path),
    "psg_raw": str(PSG_RAW),
    "active_profile": ACTIVE_PROFILE,
    "profiles": PROFILES,
    "observed": profile_summaries,
    "final": str(UPSTREAM),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "interpretation": (
        "LB probe only. The coverage gate has not been validated against the "
        "final geometric-neighbor OOF chain."
    ),
}
(WORK / "psg_addon_verdict.json").write_text(
    json.dumps(verdict, indent=2), encoding="utf-8"
)
print("PSG MAINLINE ADD-ON COMPLETE")
print(json.dumps(verdict, indent=2))
print("Final Kaggle file:", UPSTREAM)
