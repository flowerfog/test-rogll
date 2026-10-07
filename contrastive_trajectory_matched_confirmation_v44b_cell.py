"""Matched confirmation of the only positive V44 learned-energy arm."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
EXPERIMENT = "contrastive_trajectory_matched_confirmation_v44b"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
V44_DIR = OOF_ROOT / "contrastive_trajectory_bank_reranker_v44"
V42_CHECKPOINT = OOF_ROOT / "contrastive_interval_embedding_v42" / "models" / "best_interval_encoder.pt"


def _find_source():
    candidates = [
        Path(os.environ.get("ROGII_V44_SOURCE", "")),
        ROOT / "contrastive_trajectory_bank_reranker_v44_cell.py",
        Path("contrastive_trajectory_bank_reranker_v44_cell.py"),
    ]
    for path in candidates:
        if str(path) and path.is_file():
            return path.resolve()
    raise FileNotFoundError("Upload contrastive_trajectory_bank_reranker_v44_cell.py")


def _bootstrap(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STARTED"
            for target in node.targets
        ):
            break
        body.append(node)
    name = "contrastive_trajectory_v44_definitions_for_v44b"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


V44_SOURCE = _find_source()
V = _bootstrap(V44_SOURCE)
SEED = V.SEED
FIXED_ARM = "raw_clip_b512_shape4"
FIXED_DOSE = 0.10
CONTROLS = V.CONTROLS


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
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


STARTED = time.time()
discovery_path = V44_DIR / "fold3_discovery_scan.csv"
for required in (V42_CHECKPOINT, discovery_path):
    if not required.exists():
        raise FileNotFoundError(required)

# Reconstruct the secondary selection rule from discovery only: maximize the
# gain over the same arm's dose-zero baseline, while requiring baseline <15 ft.
discovery = pd.read_csv(discovery_path)
eligible = discovery.loc[(discovery["dose"] >= 0) & (discovery["rmse"] < 200)].copy()
baselines = eligible.loc[eligible["dose"].eq(0), ["arm", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
matched = eligible.merge(baselines, on="arm", how="inner")
matched["matched_gain"] = matched["baseline_rmse"] - matched["rmse"]
matched = matched.loc[(matched["dose"] > 0) & (matched["baseline_rmse"] <= 15.0)].sort_values("matched_gain", ascending=False)
if matched.empty:
    raise RuntimeError("No eligible positive-dose matched recipe in V44 discovery")
selected = matched.iloc[0]
if str(selected["arm"]) != FIXED_ARM or not np.isclose(float(selected["dose"]), FIXED_DOSE):
    raise RuntimeError(f"Unexpected matched discovery winner: {selected.to_dict()}")
print("V44B FROZEN MATCHED RECIPE", selected.to_dict(), flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(V42_CHECKPOINT, map_location=device, weights_only=False)
model = V.V42.CrossModalIntervalEncoder().to(device)
model.load_state_dict(checkpoint["model"])
model.eval()
folds, fold_path = V.V42._load_folds()
folds["well"] = folds["well"].astype(str)
prior_wells = folds.loc[folds["fold"].isin(V.PRIOR_FOLDS), "well"].tolist()
fold3_wells = folds.loc[folds["fold"].eq(V.SELECTION_FOLD), "well"].tolist()
fold2_wells = folds.loc[folds["fold"].eq(V.APPLICATION_FOLD), "well"].tolist()
all_wells = sorted(set(prior_wells + fold3_wells + fold2_wells))
V.V42._ensure_records(all_wells)
V.V42._ensure_packs(all_wells)
records42, packs42, records3, failures = {}, {}, {}, []
for number, well in enumerate(all_wells, 1):
    try:
        records42[well] = V.V42._load_record(well)
        packs42[well] = V.V42._load_pack(well)
        records3[well] = V.V3.V["_load_record"](well)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(all_wells):
        print(f"V44b load {number}/{len(all_wells)} failures={len(failures)}", flush=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)
available = set(records42) & set(packs42) & set(records3)
prior_wells = [well for well in prior_wells if well in available]
fold3_wells = [well for well in fold3_wells if well in available]
fold2_wells = [well for well in fold2_wells if well in available]
prior = V.V3.V["_fit_prior"](
    [records3[well] for well in prior_wells], V.V3.SEGMENT_LENGTH_FT
)
arm_map = {arm["id"]: arm for arm in V.V3.ARMS}
arm = arm_map[FIXED_ARM]
fold3_order = sorted(
    fold3_wells,
    key=lambda well: hashlib.sha256(f"v44-discovery:{well}".encode("ascii")).digest(),
)
confirmation_wells = fold3_order[V.DISCOVERY_WELLS:]

confirmation_all = V._evaluate(
    confirmation_wells, records3, records42, packs42, prior, [arm], model,
    device, "fold3_confirmation_matched",
)
confirmation = V._keep_recipe(confirmation_all, FIXED_ARM, (0.0, FIXED_DOSE, -1.0))
confirmation_summary = V._summary(confirmation)

ordered2 = sorted(fold2_wells)
donors = {well: ordered2[(index + 1) % len(ordered2)] for index, well in enumerate(ordered2)}
fold2_parts = []
for control in ("full",) + CONTROLS:
    detail = V._evaluate(
        fold2_wells, records3, records42, packs42, prior, [arm], model, device,
        "fold2_application_matched", control,
        donors if control == "donor_exact" else None,
    )
    doses = (0.0, FIXED_DOSE, -1.0) if control == "full" else (FIXED_DOSE,)
    fold2_parts.append(V._keep_recipe(detail, FIXED_ARM, doses))
fold2 = pd.concat(fold2_parts, ignore_index=True)
fold2_summary = V._summary(fold2)
confirmation_summary.to_csv(OUT_DIR / "fold3_confirmation.csv", index=False)
fold2_summary.to_csv(OUT_DIR / "fold2_application_controls.csv", index=False)


def _pick(frame, dose, control="full"):
    rows = frame.loc[frame["dose"].eq(dose) & frame["control"].eq(control)]
    if len(rows) != 1:
        raise RuntimeError(f"Cannot select dose={dose}, control={control}: {len(rows)}")
    return rows.iloc[0].to_dict()


fold3_base = _pick(confirmation_summary, 0.0)
fold3_selected = _pick(confirmation_summary, FIXED_DOSE)
fold2_base = _pick(fold2_summary, 0.0)
fold2_selected = _pick(fold2_summary, FIXED_DOSE)
fold2_oracle = _pick(fold2_summary, -1.0)
controls = {name: _pick(fold2_summary, FIXED_DOSE, name) for name in CONTROLS}
fold3_gain = fold3_base["rmse"] - fold3_selected["rmse"]
fold2_gain = fold2_base["rmse"] - fold2_selected["rmse"]
fold2_far_gain = fold2_base["far_gt2000_rmse"] - fold2_selected["far_gt2000_rmse"]
control_gain = {name: fold2_base["rmse"] - row["rmse"] for name, row in controls.items()}
gates = {
    "fold3_gain_ge_0p3": bool(fold3_gain >= 0.3),
    "fold2_gain_ge_0p3": bool(fold2_gain >= 0.3),
    "fold2_far_gain_positive": bool(fold2_far_gain > 0),
    "fold2_oracle_rmse_le_6": bool(fold2_oracle["rmse"] <= 6.0),
    "state_shuffle_weakens": bool(control_gain["state_shuffle"] < fold2_gain),
    "window_shuffle_weakens": bool(control_gain["window_shuffle"] < fold2_gain),
    "donor_weakens": bool(control_gain["donor_exact"] < fold2_gain),
    "shift_weakens": bool(control_gain["shift_exact_64"] < fold2_gain),
    "no_gr_weakens": bool(control_gain["no_gr"] < fold2_gain),
}
strict_pass = bool(all(gates.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "V44_discovery_matched_gain_Fold3_and_Fold2_confirmation",
    "selection_rule": "maximum within-arm gain, baseline RMSE <=15, positive dose",
    "frozen_recipe": {"arm": FIXED_ARM, "dose": FIXED_DOSE},
    "fold3": {"baseline": fold3_base, "selected": fold3_selected, "gain": fold3_gain},
    "fold2": {
        "baseline": fold2_base, "selected": fold2_selected, "oracle": fold2_oracle,
        "gain": fold2_gain, "far_gain": fold2_far_gain,
    },
    "controls": controls,
    "gates": gates,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> cross-fit this trajectory endpoint and measure correlation with 6.377"
        if strict_pass else
        "FAIL -> close V42 interval-identity family"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV44B FOLD3 CONFIRMATION")
print(confirmation_summary.to_string(index=False))
print("\nV44B FOLD2 APPLICATION AND CONTROLS")
print(fold2_summary.to_string(index=False))
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
