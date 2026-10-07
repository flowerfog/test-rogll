"""Rerank complete V3 trajectory hypotheses with the frozen V42 emission."""

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
EXPERIMENT = "contrastive_trajectory_bank_reranker_v44"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
V42_DIR = OOF_ROOT / "contrastive_interval_embedding_v42"
V3_DIR = OOF_ROOT / "rawdata_lattice_mode_union_v3"
V42_CHECKPOINT = V42_DIR / "models" / "best_interval_encoder.pt"
V3_VERDICT = V3_DIR / "verdict.json"


def _find_source(env, names):
    candidates = []
    configured = os.environ.get(env, "").strip()
    if configured:
        candidates.append(Path(configured))
    for name in names:
        candidates.extend((ROOT / name, Path(name)))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Missing source for {env}: {names}")


def _bootstrap_before_assignment(path, assignment):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == assignment
            for target in node.targets
        ):
            break
        body.append(node)
    name = f"{path.stem}_definitions_for_v44"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


V42_SOURCE = _find_source("ROGII_V42_SOURCE", ("contrastive_interval_embedding_v42_cell.py",))
V3_SOURCE = _find_source("ROGII_V3_SOURCE", ("rawdata_lattice_mode_union_v3_cell.py",))
V42 = _bootstrap_before_assignment(V42_SOURCE, "STARTED")
V3 = _bootstrap_before_assignment(V3_SOURCE, "started")

SEED = 20260727
PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DISCOVERY_WELLS = 48
DOSES = (0.0, 0.10, 0.25, 0.50, 1.0, 2.0)
CONTROLS = ("state_shuffle", "window_shuffle", "donor_exact", "shift_exact_64", "no_gr")


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


def _robust_z(values):
    values = np.asarray(values, np.float64)
    centre = float(np.median(values))
    low, high = np.quantile(values, (0.1, 0.9))
    scale = max(float(high - low), float(np.std(values)) * 0.5, 1e-6)
    return np.clip((values - centre) / scale, -8.0, 8.0)


def _trajectory_matrix(record, prior, paths, segments, grid_md):
    rates = np.asarray(prior["rate_grid"], np.float64)
    path_rates = rates[np.asarray(paths, np.int64)]
    end_md = np.asarray([record["md"][rows][-1] for rows in segments], np.float64)
    start_md = np.r_[float(record["anchor_md"]), end_md[:-1]]
    lengths = end_md - start_md
    increments = path_rates * lengths[None, :]
    start_surface = float(record["anchor_surface"]) + np.c_[
        np.zeros(len(paths), np.float64), np.cumsum(increments, axis=1)[:, :-1]
    ]
    segment = np.searchsorted(end_md, grid_md, side="left")
    segment = np.clip(segment, 0, len(segments) - 1)
    column = np.arange(len(grid_md))
    surface = (
        start_surface[:, segment]
        + path_rates[:, segment] * (grid_md[None, :] - start_md[segment][None, :])
    )
    z_grid = np.interp(grid_md, record["md"], record["z"])
    return surface - z_grid[None, :]


def _trajectory_learned_score(path_tvt, pack, learned_profile):
    offsets = path_tvt - float(pack.anchor_tvt)
    result = np.empty(offsets.shape, np.float64)
    for column in range(offsets.shape[1]):
        result[:, column] = np.interp(
            offsets[:, column],
            V42.STATE_OFFSETS,
            learned_profile[:, column],
            left=-6.0,
            right=-6.0,
        )
    distance = pack.grid_md - float(pack.anchor_md)
    usable = distance > 0.0
    if not usable.any():
        raise RuntimeError("No post-anchor V42 grid columns")
    # Equal-weight blocks prevent dense short-distance sampling from dominating.
    block = np.floor(distance[usable] / 256.0).astype(np.int32)
    block_scores = []
    for value in np.unique(block):
        block_scores.append(np.mean(result[:, usable][:, block == value], axis=1))
    return np.mean(np.stack(block_scores, axis=1), axis=1)


@torch.no_grad()
def _profile(model, record, pack, device, control="full", donor=None):
    override = None
    profile_control = "full"
    if control == "donor_exact":
        override = donor.typewell
    elif control == "shift_exact_64":
        override = V42._shift_typewell(record.typewell, 64.0)
    elif control in ("window_shuffle", "no_gr"):
        profile_control = control
    score = V42._learned_profile(
        model, record, pack, device, override, profile_control
    ).astype(np.float64)
    if control == "state_shuffle":
        order = np.random.default_rng(SEED + 441).permutation(len(V42.STATE_OFFSETS))
        score = score[order]
    return score


def _well_candidates(well, record, pack, prior, arms, learned_profile, include_oracle=True):
    candidates = []
    for arm in arms:
        raw_score, paths, segments = V3._beam_bank_v3(record, prior, arm)
        path_tvt_grid = _trajectory_matrix(record, prior, paths, segments, pack.grid_md)
        learned = _trajectory_learned_score(path_tvt_grid, pack, learned_profile)
        handcrafted = _robust_z(raw_score)
        learned = _robust_z(learned)
        for dose in DOSES:
            index = int(np.argmin(handcrafted - float(dose) * learned))
            prediction = V3.B["_path_prediction"](record, prior, paths[index], segments)
            candidates.append(
                {
                    "arm": arm["id"],
                    "dose": float(dose),
                    "prediction": prediction[record["eval_rows"]].astype(np.float64),
                    "selected_path": index,
                    "learned_z": float(learned[index]),
                    "handcrafted_z": float(handcrafted[index]),
                    "bank_paths": int(len(paths)),
                }
            )
        if include_oracle:
            truth = record["truth"][record["eval_rows"]]
            oracle_sse = np.inf
            oracle_prediction = None
            for index, path in enumerate(paths):
                prediction = V3.B["_path_prediction"](record, prior, path, segments)[record["eval_rows"]]
                sse = float(np.sum((prediction - truth) ** 2))
                if sse < oracle_sse:
                    oracle_sse, oracle_prediction = sse, prediction.copy()
            candidates.append(
                {
                    "arm": arm["id"], "dose": -1.0, "prediction": oracle_prediction,
                    "selected_path": -1, "learned_z": np.nan, "handcrafted_z": np.nan,
                    "bank_paths": int(len(paths)),
                }
            )
    return candidates


def _rows_for_candidates(record, candidates, stage, control):
    truth = record["truth"][record["eval_rows"]]
    distance = record["md"][record["eval_rows"]] - float(record["anchor_md"])
    far = distance > 2000.0
    rows = []
    for candidate in candidates:
        error = candidate["prediction"] - truth
        rows.append(
            {
                "stage": stage,
                "control": control,
                "well": record["well"],
                "arm": candidate["arm"],
                "dose": candidate["dose"],
                "rows": int(len(error)),
                "sse": float(np.dot(error, error)),
                "sae": float(np.sum(np.abs(error))),
                "far_rows": int(far.sum()),
                "far_sse": float(np.dot(error[far], error[far])) if far.any() else 0.0,
                "selected_path": candidate["selected_path"],
                "learned_z": candidate["learned_z"],
                "handcrafted_z": candidate["handcrafted_z"],
                "bank_paths": candidate["bank_paths"],
            }
        )
    return rows


def _evaluate(wells, records3, records42, packs42, prior, arms, model, device, stage, control="full", donors=None):
    output = []
    for number, well in enumerate(wells, 1):
        donor = None if donors is None else records42[donors[well]]
        learned = _profile(model, records42[well], packs42[well], device, control, donor)
        candidates = _well_candidates(
            well, records3[well], packs42[well], prior, arms, learned,
            include_oracle=(control == "full"),
        )
        output.extend(_rows_for_candidates(records3[well], candidates, stage, control))
        if number % 12 == 0 or number == len(wells):
            print(f"V44 {stage} {control} {number}/{len(wells)}", flush=True)
    return pd.DataFrame(output)


def _summary(frame):
    rows = []
    for (stage, control, arm, dose), group in frame.groupby(
        ["stage", "control", "arm", "dose"], dropna=False
    ):
        count = int(group["rows"].sum())
        far_rows = int(group["far_rows"].sum())
        well_rmse = np.sqrt(group["sse"] / group["rows"])
        rows.append(
            {
                "stage": stage, "control": control, "arm": arm, "dose": float(dose),
                "rows": count, "wells": int(group["well"].nunique()),
                "exact_sse": float(group["sse"].sum()),
                "rmse": math.sqrt(float(group["sse"].sum()) / count),
                "mae": float(group["sae"].sum() / count),
                "median_well_rmse": float(np.median(well_rmse)),
                "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
                "far_gt2000_rmse": math.sqrt(float(group["far_sse"].sum()) / max(far_rows, 1)),
                "mean_selected_path": float(group.loc[group["selected_path"] >= 0, "selected_path"].mean()),
            }
        )
    return pd.DataFrame(rows)


STARTED = time.time()
for required in (V42_CHECKPOINT, V3_VERDICT):
    if not required.exists():
        raise FileNotFoundError(required)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(V42_CHECKPOINT, map_location=device, weights_only=False)
model = V42.CrossModalIntervalEncoder().to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

v3_verdict = json.loads(V3_VERDICT.read_text(encoding="utf-8"))
selected_arm_ids = list(v3_verdict["selected_arms"])
arm_map = {arm["id"]: arm for arm in V3.ARMS}
arms = [arm_map[name] for name in selected_arm_ids]
folds, fold_path = V42._load_folds()
folds["well"] = folds["well"].astype(str)
prior_wells = folds.loc[folds["fold"].isin(PRIOR_FOLDS), "well"].tolist()
fold3_wells = folds.loc[folds["fold"].eq(SELECTION_FOLD), "well"].tolist()
fold2_wells = folds.loc[folds["fold"].eq(APPLICATION_FOLD), "well"].tolist()
all_wells = sorted(set(prior_wells + fold3_wells + fold2_wells))
V42._ensure_records(all_wells)
V42._ensure_packs(all_wells)
records42, packs42, records3, failures = {}, {}, {}, []
for number, well in enumerate(all_wells, 1):
    try:
        records42[well] = V42._load_record(well)
        packs42[well] = V42._load_pack(well)
        records3[well] = V3.V["_load_record"](well)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(all_wells):
        print(f"V44 load {number}/{len(all_wells)} failures={len(failures)}", flush=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)
eligible = set(records42) & set(packs42) & set(records3)
prior_wells = [well for well in prior_wells if well in eligible]
fold3_wells = [well for well in fold3_wells if well in eligible]
fold2_wells = [well for well in fold2_wells if well in eligible]
prior = V3.V["_fit_prior"]([records3[well] for well in prior_wells], V3.SEGMENT_LENGTH_FT)

order = sorted(
    fold3_wells,
    key=lambda well: hashlib.sha256(f"v44-discovery:{well}".encode("ascii")).digest(),
)
discovery_wells = order[:DISCOVERY_WELLS]
confirmation_wells = order[DISCOVERY_WELLS:]
discovery_detail = _evaluate(
    discovery_wells, records3, records42, packs42, prior, arms, model, device,
    "fold3_discovery",
)
discovery_summary = _summary(discovery_detail).sort_values("rmse")
deployable = discovery_summary.loc[discovery_summary["dose"] >= 0].copy()
selected = deployable.iloc[0].to_dict()
selected_arm = str(selected["arm"])
selected_dose = float(selected["dose"])

def _keep_recipe(frame, arm, doses):
    return frame.loc[frame["arm"].eq(arm) & frame["dose"].isin(doses)].copy()

confirmation_all = _evaluate(
    confirmation_wells, records3, records42, packs42, prior,
    [arm_map[selected_arm]], model, device, "fold3_confirmation",
)
confirmation_detail = _keep_recipe(confirmation_all, selected_arm, (0.0, selected_dose, -1.0))
confirmation_summary = _summary(confirmation_detail)

ordered2 = sorted(fold2_wells)
donors = {well: ordered2[(index + 1) % len(ordered2)] for index, well in enumerate(ordered2)}
fold2_frames = []
for control in ("full",) + CONTROLS:
    detail = _evaluate(
        fold2_wells, records3, records42, packs42, prior,
        [arm_map[selected_arm]], model, device, "fold2_application", control,
        donors if control == "donor_exact" else None,
    )
    doses = (0.0, selected_dose, -1.0) if control == "full" else (selected_dose,)
    fold2_frames.append(_keep_recipe(detail, selected_arm, doses))
fold2_detail = pd.concat(fold2_frames, ignore_index=True)
fold2_summary = _summary(fold2_detail)

discovery_summary.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
confirmation_summary.to_csv(OUT_DIR / "fold3_confirmation.csv", index=False)
fold2_summary.to_csv(OUT_DIR / "fold2_application_controls.csv", index=False)

def _pick(frame, dose, control="full"):
    rows = frame.loc[frame["dose"].eq(dose) & frame["control"].eq(control)]
    if len(rows) != 1:
        raise RuntimeError(f"Cannot select dose={dose}, control={control}: {len(rows)}")
    return rows.iloc[0].to_dict()

fold3_base = _pick(confirmation_summary, 0.0)
fold3_selected = _pick(confirmation_summary, selected_dose)
fold2_base = _pick(fold2_summary, 0.0)
fold2_selected = _pick(fold2_summary, selected_dose)
fold2_oracle = _pick(fold2_summary, -1.0)
controls = {name: _pick(fold2_summary, selected_dose, name) for name in CONTROLS}
fold3_gain = fold3_base["rmse"] - fold3_selected["rmse"]
fold2_gain = fold2_base["rmse"] - fold2_selected["rmse"]
fold2_far_gain = fold2_base["far_gt2000_rmse"] - fold2_selected["far_gt2000_rmse"]
control_gain = {name: fold2_base["rmse"] - row["rmse"] for name, row in controls.items()}
gates = {
    "selected_dose_nonzero": bool(selected_dose > 0),
    "fold3_gain_ge_0p5": bool(fold3_gain >= 0.5),
    "fold2_gain_ge_0p5": bool(fold2_gain >= 0.5),
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
    "stage": "Fold3_complete_trajectory_energy_selection_Fold2_confirmation",
    "sample_unit": "one complete physical well trajectory",
    "raw_inputs_only": True,
    "historical_prediction_dependencies": [],
    "checkpoint": str(V42_CHECKPOINT),
    "selected_v3_arms": selected_arm_ids,
    "selected_recipe": {"arm": selected_arm, "dose": selected_dose},
    "coverage": {
        "prior_wells": len(prior_wells), "fold3_wells": len(fold3_wells),
        "fold2_wells": len(fold2_wells), "record_failures": len(failures),
    },
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
        "PASS -> cross-fit trajectory endpoint and audit correlation with the 6.377 incumbent"
        if strict_pass else
        "FAIL -> learned interval identity cannot rank coherent complete paths; close this family"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV44 DISCOVERY SCAN")
print(discovery_summary.to_string(index=False))
print("\nV44 FOLD3 CONFIRMATION")
print(confirmation_summary.to_string(index=False))
print("\nV44 FOLD2 APPLICATION AND CONTROLS")
print(fold2_summary.to_string(index=False))
print("\nFINAL VERDICT", json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
