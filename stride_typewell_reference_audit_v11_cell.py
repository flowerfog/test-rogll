"""Known-prefix audit and Fold-5 confirmation of STRIDE reference terms.

The writeup says the typewell curve is nudged/blended toward the lateral's
known-section GR.  Our full affine remapping has the opposite ablation sign.
Select a reference recipe without hidden truth by fitting on the early 70% of
each visible prefix and scoring the held-out prefix tail.  Then decode only
Fold 5, the fold that invalidated V9 fusion, with the selected recipe and its
outer-train prior.  Cauchy gamma is held fixed to isolate the reference term.
"""

from __future__ import annotations

import ast
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "stride_typewell_reference_audit_v11"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
V9_DIR = OOF_ROOT / "stride_sp45_fivefold_oof_v9"
V9_FOLD5 = V9_DIR / "folds" / "fold5_stride_raw.npz"
V9_PRIOR5 = V9_DIR / "folds" / "fold5_prior.json"
V3_VERDICT = OOF_ROOT / "stride_figure1_posterior_parity_v3" / "verdict.json"
V5_VERDICT = OOF_ROOT / "stride_sequence_scale_repair_v5" / "verdict.json"
SOURCE_CANDIDATES = (
    ROOT / "stride_fullsequence_beam_v4_cell.py",
    Path.cwd() / "stride_fullsequence_beam_v4_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError("Upload the updated stride_fullsequence_beam_v4_cell.py")
for required in (V9_FOLD5, V9_PRIOR5, V3_VERDICT, V5_VERDICT):
    if not required.exists():
        raise FileNotFoundError(required)

RECIPES = (
    ("raw", "raw", 0.0),
    ("shift_025", "shift", 0.25),
    ("shift_050", "shift", 0.50),
    ("shift_100", "shift", 1.00),
    ("affine_025", "affine", 0.25),
    ("affine_050", "affine", 0.50),
    ("affine_100", "affine", 1.00),
    ("local_025", "local", 0.25),
    ("local_050", "local", 0.50),
    ("local_100", "local", 1.00),
)


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "v3" in names:
                break
        body.append(node)
    name = "stride_v4_repaired_definitions_for_v11"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
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


def _reference(record, kind, dose, known_rows):
    raw = record["tw_gr_raw"].astype(np.float64)
    if kind == "raw" or dose == 0.0:
        return raw.copy()
    tvt = record["tvt_input"][known_rows]
    observed = record["gr"][known_rows]
    expected = np.interp(tvt, record["tw_tvt"], raw)
    valid = np.isfinite(tvt) & np.isfinite(observed) & np.isfinite(expected)
    tvt = tvt[valid]
    observed = observed[valid]
    expected = expected[valid]
    if len(tvt) < 32:
        return raw.copy()
    if kind == "shift":
        correction = float(np.median(observed - expected))
        target = raw + correction
    elif kind == "affine":
        gain, intercept, _, _ = V.V._robust_affine(expected, observed)
        target = gain * raw + intercept
    elif kind == "local":
        order = np.argsort(tvt, kind="stable")
        work = pd.DataFrame(
            {"tvt": tvt[order], "residual": (observed - expected)[order]}
        )
        work["bin"] = np.round(work["tvt"] * 2.0) / 2.0
        work = work.groupby("bin", as_index=False)["residual"].median().sort_values("bin")
        window = min(31, len(work))
        if window % 2 == 0:
            window -= 1
        if window >= 5:
            work["residual"] = work["residual"].rolling(
                window, center=True, min_periods=max(3, window // 3)
            ).median().interpolate(limit_direction="both")
        correction = np.interp(
            record["tw_tvt"],
            work["bin"].to_numpy(np.float64),
            work["residual"].to_numpy(np.float64),
        )
        target = raw + correction
    else:
        raise ValueError(kind)
    return raw + float(dose) * (target - raw)


def _path_metrics(rows):
    frame = pd.DataFrame(rows)
    total_rows = int(frame["rows"].sum())
    total_sse = float(frame["sse"].sum())
    well_rmse = np.sqrt(frame["sse"] / frame["rows"])
    return {
        "rows": total_rows,
        "wells": int(frame["well"].nunique()),
        "exact_sse": total_sse,
        "rmse": float(np.sqrt(total_sse / total_rows)),
        "mean_well_rmse": float(well_rmse.mean()),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
    }


V = _bootstrap(SOURCE_PATH)
v3 = json.loads(V3_VERDICT.read_text(encoding="utf-8"))
v5 = json.loads(V5_VERDICT.read_text(encoding="utf-8"))
recipe = v5["selected_recipe"]
temperature = float(v3["selected"]["emission_temperature"])
prior5 = json.loads(V9_PRIOR5.read_text(encoding="utf-8"))

folds = pd.read_csv(V.V.FOLD_PATH, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
records = {}
for number, well in enumerate(sorted(folds["well"]), start=1):
    records[well] = V.load_record(well)
    if number % 100 == 0 or number == len(folds):
        print(f"V11 load records {number}/{len(folds)}", flush=True)

prefix_rows = []
for well, record in records.items():
    known = record["known"]
    split = int(np.floor(0.70 * len(known)))
    split = min(max(split, 64), len(known) - 32)
    fit_rows = known[:split]
    valid_rows = known[split:]
    if len(fit_rows) < 64 or len(valid_rows) < 32:
        continue
    raw_expected = np.interp(
        record["tvt_input"][valid_rows], record["tw_tvt"], record["tw_gr_raw"]
    )
    raw_residual = record["gr"][valid_rows] - raw_expected
    for name, kind, dose in RECIPES:
        reference = _reference(record, kind, dose, fit_rows)
        expected = np.interp(
            record["tvt_input"][valid_rows], record["tw_tvt"], reference
        )
        residual = record["gr"][valid_rows] - expected
        scale = max(float(record["gamma"]), 1e-6)
        prefix_rows.append(
            {
                "well": well,
                "recipe": name,
                "rows": len(valid_rows),
                "sse": float(np.dot(residual, residual)),
                "absolute_error": float(np.abs(residual).sum()),
                "cauchy_nll": float(np.log1p(np.minimum((residual / scale) ** 2, 1e4)).sum()),
                "raw_sse": float(np.dot(raw_residual, raw_residual)),
            }
        )
prefix = pd.DataFrame(prefix_rows)
prefix.to_csv(OUT_DIR / "prefix_tail_bywell.csv", index=False)
summary_rows = []
raw_bywell = prefix.loc[prefix["recipe"].eq("raw")].set_index("well")
for name, _, _ in RECIPES:
    local = prefix.loc[prefix["recipe"].eq(name)].set_index("well")
    rows = int(local["rows"].sum())
    summary_rows.append(
        {
            "recipe": name,
            "rows": rows,
            "wells": len(local),
            "gr_rmse": float(np.sqrt(local["sse"].sum() / rows)),
            "gr_mae": float(local["absolute_error"].sum() / rows),
            "cauchy_nll_per_row": float(local["cauchy_nll"].sum() / rows),
            "positive_well_fraction": float((local["sse"] < raw_bywell.loc[local.index, "sse"]).mean()),
        }
    )
summary = pd.DataFrame(summary_rows).sort_values(
    ["cauchy_nll_per_row", "gr_rmse"], ignore_index=True
)
summary.to_csv(OUT_DIR / "prefix_tail_summary.csv", index=False)
selected_name = str(summary.iloc[0]["recipe"])
selected_kind, selected_dose = next(
    (kind, dose) for name, kind, dose in RECIPES if name == selected_name
)

with np.load(V9_FOLD5, allow_pickle=False) as data:
    artifact_wells = data["wells"].astype(str)
    artifact_codes = data["well_code"].astype(np.int64)
    artifact_rows = data["row_idx"].astype(np.int64)
    artifact_prediction = data["stride_raw"].astype(np.float64)
old_rows = []
for code, well in enumerate(artifact_wells):
    mask = artifact_codes == code
    truth = records[well]["truth"][artifact_rows[mask]]
    error = artifact_prediction[mask] - truth
    old_rows.append(
        {"well": well, "rows": int(mask.sum()), "sse": float(np.dot(error, error))}
    )
old_metrics = _path_metrics(old_rows)

started = time.time()
V.BEAM_SIZE = 48
new_rows = []
for number, well in enumerate(artifact_wells, start=1):
    record = records[well]
    calibrated = dict(record)
    calibrated["tw_gr_raw"] = _reference(
        record, selected_kind, selected_dose, record["known"]
    )
    prediction, _, _ = V._decode(
        calibrated,
        prior5,
        temperature,
        "raw_reference",
        effective_spacing_ft=float(recipe["effective_spacing_ft"]),
        persistence_weight=float(recipe["persistence_weight"]),
        lambda_override=float(recipe["lambda_previous_dip"]),
    )
    evaluation = record["evaluation"]
    error = prediction - record["truth"][evaluation]
    new_rows.append(
        {"well": well, "rows": len(evaluation), "sse": float(np.dot(error, error))}
    )
    if number % 25 == 0 or number == len(artifact_wells):
        print(f"V11 Fold5 {selected_name} {number}/{len(artifact_wells)}", flush=True)
new_metrics = _path_metrics(new_rows)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "known_prefix_reference_selection_to_Fold5_confirmation",
    "selection": {
        "uses_hidden_truth": False,
        "fit_prefix_fraction": 0.70,
        "score_prefix_fraction": 0.30,
        "gamma_held_fixed": True,
        "recipes": [name for name, _, _ in RECIPES],
        "selected_recipe": selected_name,
        "selected_kind": selected_kind,
        "selected_dose": selected_dose,
        "summary": summary.to_dict(orient="records"),
    },
    "fold5": {
        "v9_raw": old_metrics,
        "selected_reference": new_metrics,
        "gain": float(old_metrics["rmse"] - new_metrics["rmse"]),
    },
    "gates": {
        "prefix_selected_not_raw": selected_name != "raw",
        "selected_prefix_positive_wells_ge_0p55": bool(
            summary.iloc[0]["positive_well_fraction"] >= 0.55
        ),
        "fold5_gain_ge_0p30": bool(old_metrics["rmse"] - new_metrics["rmse"] >= 0.30),
    },
    "elapsed_minutes": (time.time() - started) / 60.0,
}
verdict["strict_pass"] = bool(all(verdict["gates"].values()))
verdict["next"] = (
    "PASS -> replace the V9 reference term and regenerate five-fold STRIDE"
    if verdict["strict_pass"]
    else "FAIL -> the disclosed typewell reference value is not reproduced by these legal calibration families"
)
_write_json(OUT_DIR / "verdict.json", verdict)

print("\nV11 PREFIX REFERENCE SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
