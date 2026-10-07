"""Non-rigid complete-well GR observability gateway.

Run after fullwell_truthshape_alias_observability_fold2_v0 in the same kernel,
or upload that source file beside this one. The v0 model proved that complete-
well GR resolves most rigid aliases. This extension asks whether the same
energy representation can identify the correct path among global shifts,
linear drift, quadratic drift, and smooth local bows.

Validation truth creates the frozen candidate shapes but is never an input to
the model. Candidate order is permuted, warp parameters are hidden, Fold 1 is
not read, and the model uses no neighbour/PF/strong/CP12/Gold evidence. This is
an observability gateway, not a deployable CV score.
"""

from __future__ import annotations

from pathlib import Path
import ast
import gc
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reuse validated v0 definitions without rerunning v0 training.
# ---------------------------------------------------------------------------


REQUIRED_V0 = {
    "DATA_ROOT",
    "OOF_ROOT",
    "SEQUENCE_LENGTH",
    "MIN_CANDIDATE_COVERAGE",
    "MIN_TRUE_COVERAGE",
    "MIN_AVAILABLE_CANDIDATES",
    "CompleteWellAliasEnergy",
    "_numeric",
    "_interp_finite",
    "_robust_location_scale",
    "_fit_affine",
    "_read_well",
    "_load_folds",
    "_seed_everything",
    "_json_default",
}


def _bootstrap_v0():
    if not REQUIRED_V0.difference(globals()):
        print("V1 bootstrap: using live v0 definitions", flush=True)
        return
    candidates = (
        Path("/root/autodl-tmp/fullwell_truthshape_alias_observability_fold2_v0_cell.py"),
        Path.cwd() / "fullwell_truthshape_alias_observability_fold2_v0_cell.py",
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            "Run v0 in this kernel or upload "
            "fullwell_truthshape_alias_observability_fold2_v0_cell.py"
        )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "folds"
            for target in node.targets
        ):
            break
        definitions.append(node)
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(source_path),
            "exec",
        ),
        globals(),
    )
    missing = sorted(REQUIRED_V0.difference(globals()))
    if missing:
        raise RuntimeError("v0 bootstrap missing definitions: " + repr(missing))
    print("V1 bootstrap source:", source_path, flush=True)


_bootstrap_v0()


# ---------------------------------------------------------------------------
# Frozen extension configuration
# ---------------------------------------------------------------------------


EXPERIMENT_V1 = "fullwell_nonrigid_warp_observability_fold2_v1"
OUT_DIR_V1 = Path(OOF_ROOT) / EXPERIMENT_V1
CACHE_DIR_V1 = OUT_DIR_V1 / "panels"
MODEL_DIR_V1 = OUT_DIR_V1 / "models"
for directory in (OUT_DIR_V1, CACHE_DIR_V1, MODEL_DIR_V1):
    directory.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS_V1 = (3, 4, 5)
VALID_FOLD_V1 = 2
SEED_V1 = 20260721
MAX_EPOCHS_V1 = 12
MIN_EPOCHS_V1 = 4
PATIENCE_V1 = 4
LEARNING_RATE_V1 = 1.2e-4
MIN_LEARNING_RATE_V1 = 2.0e-5
WEIGHT_DECAY_V1 = 3.0e-5
GRAD_ACCUM_WELLS_V1 = 4
GRADIENT_CLIP_V1 = 4.0
HARD_MARGIN_V1 = 0.75
LOSS_HARD_MARGIN_V1 = 0.20

GATE_PATH_RMSE_V1 = 5.5
GATE_EXACT_TOP1_V1 = 0.80
GATE_TOP5_V1 = 0.95
GATE_HARD12_WIN_V1 = 0.90
GATE_NO_GR_DROP_V1 = 0.50
GATE_SHUFFLED_DROP_V1 = 0.70


def _warp_templates_v1(progress):
    progress = np.asarray(progress, np.float32)
    templates = [np.zeros_like(progress)]
    names = ["truth"]

    for amplitude in (4, 8, 12, 16, 24, 32, 48, 64, 96):
        for sign in (-1.0, 1.0):
            templates.append(np.full_like(progress, sign * amplitude))
            names.append(f"global_{int(sign * amplitude):+d}")

    for amplitude in (8, 12, 16, 24, 32, 48, 64):
        for sign in (-1.0, 1.0):
            templates.append(sign * amplitude * progress)
            names.append(f"linear_{int(sign * amplitude):+d}")

    for amplitude in (8, 12, 16, 24):
        for sign in (-1.0, 1.0):
            templates.append(sign * amplitude * np.square(progress))
            names.append(f"quadratic_{int(sign * amplitude):+d}")

    for amplitude in (8, 12, 16, 24):
        for sign in (-1.0, 1.0):
            templates.append(sign * amplitude * np.sin(np.pi * progress))
            names.append(f"bow_{int(sign * amplitude):+d}")

    output = np.stack(templates, axis=0).astype(np.float32)
    if output.shape[0] != 49:
        raise RuntimeError(f"expected 49 warp templates, got {output.shape}")
    return output, np.asarray(names)


WARP_NAMES_V1 = _warp_templates_v1(np.linspace(0, 1, SEQUENCE_LENGTH))[1]
N_CANDIDATES_V1 = len(WARP_NAMES_V1)
ZERO_INDEX_V1 = 0


def _panel_path_v1(well):
    return CACHE_DIR_V1 / f"{well}.npz"


def _build_warp_panel_v1(well):
    horizontal, typewell = _read_well(well)
    md = _numeric(horizontal, "MD")
    horizontal_gr = _numeric(horizontal, "GR")
    truth_tvt = _numeric(horizontal, "TVT")
    tvt_input = _numeric(horizontal, "TVT_input")
    typewell_tvt = _numeric(typewell, "TVT")
    typewell_gr = _numeric(typewell, "GR")

    known = np.flatnonzero(np.isfinite(tvt_input))
    hidden = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth_tvt))
    if len(known) < 128 or len(hidden) < 128 or hidden[0] != len(known):
        raise RuntimeError(
            f"invalid official mask known={len(known)} hidden={len(hidden)}"
        )

    tw_prefix = _interp_finite(typewell_tvt, typewell_gr, tvt_input[known])
    slope, intercept, pairs, calibration_rmse = _fit_affine(
        tw_prefix, horizontal_gr[known]
    )
    calibrated_typewell = slope * typewell_gr + intercept
    centre, scale = _robust_location_scale(horizontal_gr[known])

    sequence_md = np.linspace(md[hidden[0]], md[hidden[-1]], SEQUENCE_LENGTH)
    query = _interp_finite(md, horizontal_gr, sequence_md)
    truth_sequence = _interp_finite(md, truth_tvt, sequence_md)
    progress = (sequence_md - sequence_md[0]) / max(
        sequence_md[-1] - sequence_md[0], 1.0
    )
    corrections, names = _warp_templates_v1(progress)

    query_valid = np.isfinite(query)
    query = np.clip((query - centre) / scale, -8.0, 8.0)
    candidate_tvt = truth_sequence[None, :] + corrections
    reference = _interp_finite(
        typewell_tvt,
        calibrated_typewell,
        candidate_tvt.reshape(-1),
    ).reshape(N_CANDIDATES_V1, SEQUENCE_LENGTH)
    valid = np.isfinite(reference) & query_valid[None, :]
    reference = np.clip((reference - centre) / scale, -8.0, 8.0)
    query = np.where(query_valid, query, 0.0)
    reference = np.where(valid, reference, 0.0)
    coverage = valid.mean(axis=1)
    available = coverage >= MIN_CANDIDATE_COVERAGE

    if float(coverage[ZERO_INDEX_V1]) < MIN_TRUE_COVERAGE:
        raise RuntimeError(
            f"true candidate coverage={float(coverage[ZERO_INDEX_V1]):.3f}"
        )
    if int(available.sum()) < MIN_AVAILABLE_CANDIDATES:
        raise RuntimeError(f"only {int(available.sum())} available candidates")

    # Exact row-weighted candidate errors are computed on native eval rows.
    native_progress = (md[hidden] - md[hidden[0]]) / max(
        md[hidden[-1]] - md[hidden[0]], 1.0
    )
    native_corrections, _ = _warp_templates_v1(native_progress)
    candidate_mse = np.mean(native_corrections.astype(np.float64) ** 2, axis=1)

    return {
        "well": str(well),
        "query": query.astype(np.float32),
        "reference": reference.astype(np.float32),
        "valid": valid.astype(bool),
        "correction": corrections.astype(np.float32),
        "candidate_mse": candidate_mse.astype(np.float32),
        "coverage": coverage.astype(np.float32),
        "eval_rows": int(len(hidden)),
        "calibration_pairs": int(pairs),
        "calibration_rmse": float(calibration_rmse),
        "names": names,
    }


def _save_panel_v1(panel):
    np.savez_compressed(
        _panel_path_v1(panel["well"]),
        well=np.asarray(panel["well"]),
        query=panel["query"].astype(np.float16),
        reference=panel["reference"].astype(np.float16),
        valid=panel["valid"].astype(np.uint8),
        correction=panel["correction"].astype(np.float16),
        candidate_mse=panel["candidate_mse"].astype(np.float32),
        coverage=panel["coverage"].astype(np.float32),
        eval_rows=np.asarray([panel["eval_rows"]], np.int32),
        names=panel["names"],
        cache_version=np.asarray([1], np.int16),
    )


def _load_panel_v1(well):
    with np.load(_panel_path_v1(well), allow_pickle=False) as data:
        return {
            "well": str(data["well"].item()),
            "query": data["query"].astype(np.float32),
            "reference": data["reference"].astype(np.float32),
            "valid": data["valid"].astype(bool),
            "correction": data["correction"].astype(np.float32),
            "candidate_mse": data["candidate_mse"].astype(np.float32),
            "coverage": data["coverage"].astype(np.float32),
            "eval_rows": int(data["eval_rows"][0]),
            "names": data["names"].astype(str),
        }


folds_v1 = _load_folds()
train_wells_v1 = folds_v1.loc[
    folds_v1["fold"].isin(TRAIN_FOLDS_V1), "well"
].tolist()
valid_wells_v1 = folds_v1.loc[
    folds_v1["fold"].eq(VALID_FOLD_V1), "well"
].tolist()
needed_wells_v1 = train_wells_v1 + valid_wells_v1
if len(train_wells_v1) < 450 or len(valid_wells_v1) < 150:
    raise RuntimeError(
        f"unexpected folds train={len(train_wells_v1)} valid={len(valid_wells_v1)}"
    )

started_v1 = time.time()
build_rows_v1 = []
for index, well in enumerate(needed_wells_v1, start=1):
    try:
        if not _panel_path_v1(well).exists():
            _save_panel_v1(_build_warp_panel_v1(well))
        panel = _load_panel_v1(well)
        build_rows_v1.append(
            {
                "well": well,
                "status": "ok",
                "eval_rows": panel["eval_rows"],
                "true_coverage": float(panel["coverage"][ZERO_INDEX_V1]),
                "available_candidates": int(
                    (panel["coverage"] >= MIN_CANDIDATE_COVERAGE).sum()
                ),
            }
        )
    except Exception as error:
        build_rows_v1.append(
            {"well": well, "status": "failed", "reason": repr(error)}
        )
    if index % 50 == 0 or index == len(needed_wells_v1):
        print(f"build nonrigid panels {index}/{len(needed_wells_v1)}", flush=True)

build_report_v1 = pd.DataFrame(build_rows_v1)
build_report_v1.to_csv(OUT_DIR_V1 / "panel_build_report.csv", index=False)
failed_v1 = build_report_v1[build_report_v1["status"].eq("failed")]
if len(failed_v1):
    raise RuntimeError(
        f"{len(failed_v1)} panel builds failed: "
        + repr(failed_v1.head(10).to_dict(orient="records"))
    )

guards_v1 = {
    "train_wells": len(train_wells_v1),
    "valid_wells": len(valid_wells_v1),
    "candidates": N_CANDIDATES_V1,
    "sequence_length": SEQUENCE_LENGTH,
    "minimum_true_coverage": float(build_report_v1["true_coverage"].min()),
    "minimum_available_candidates": int(
        build_report_v1["available_candidates"].min()
    ),
    "fold1_touched": False,
}
print("NONRIGID PANEL GUARDS", guards_v1, flush=True)


def _tensor_panel_v1(panel, device, control="full"):
    query = panel["query"].copy()
    reference = panel["reference"].copy()
    valid = panel["valid"].copy()
    if control == "no_gr":
        query.fill(0.0)
    elif control == "shuffled_alignment":
        reference = np.roll(reference, shift=7, axis=0)
        valid = np.roll(valid, shift=7, axis=0)
    elif control != "full":
        raise ValueError(control)
    return (
        torch.from_numpy(query).to(device),
        torch.from_numpy(reference).to(device),
        torch.from_numpy(valid.astype(np.float32)).to(device),
    )


def _loss_v1(model, panel, device):
    query, reference, valid = _tensor_panel_v1(panel, device)
    permutation = torch.randperm(N_CANDIDATES_V1, device=device)
    reference = reference[permutation]
    valid = valid[permutation]
    candidate_mse = torch.from_numpy(panel["candidate_mse"]).to(device)[permutation]
    original_index = torch.arange(N_CANDIDATES_V1, device=device)[permutation]
    target = torch.nonzero(
        original_index == ZERO_INDEX_V1, as_tuple=False
    ).flatten()[0]
    logits = model(query, reference, valid)
    ce = F.cross_entropy(logits[None], target[None])
    hard = (candidate_mse >= 12.0**2) & (logits > -1e3)
    if hard.any():
        hard_margin = F.relu(HARD_MARGIN_V1 - logits[target] + logits[hard].amax())
    else:
        hard_margin = logits[target] * 0.0
    loss = ce + LOSS_HARD_MARGIN_V1 * hard_margin
    return loss, {
        "ce": float(ce.detach().cpu()),
        "hard_margin": float(hard_margin.detach().cpu()),
    }


@torch.no_grad()
def _scores_v1(model, panel, device, control="full"):
    model.eval()
    query, reference, valid = _tensor_panel_v1(panel, device, control)
    return model(query, reference, valid).detach().cpu().numpy()


def _evaluate_v1(model, wells, device, control="full"):
    rows = []
    total_sse = 0.0
    total_posterior_sse = 0.0
    total_rows = 0
    for well in wells:
        panel = _load_panel_v1(well)
        scores = _scores_v1(model, panel, device, control)
        probability = np.exp(scores - np.max(scores))
        probability /= probability.sum()
        selected = int(np.argmax(scores))
        ranked = np.argsort(-scores)
        rank_truth = int(np.flatnonzero(ranked == ZERO_INDEX_V1)[0]) + 1
        posterior_correction = np.sum(
            probability[:, None] * panel["correction"], axis=0
        )
        selected_mse = float(panel["candidate_mse"][selected])
        posterior_mse = float(np.mean(posterior_correction.astype(np.float64) ** 2))
        rows_count = int(panel["eval_rows"])
        total_sse += rows_count * selected_mse
        total_posterior_sse += rows_count * posterior_mse
        total_rows += rows_count
        hard = panel["candidate_mse"] >= 12.0**2
        hard &= scores > -1e3
        rows.append(
            {
                "well": well,
                "eval_rows": rows_count,
                "selected": selected,
                "selected_name": panel["names"][selected],
                "selected_rmse": math.sqrt(selected_mse),
                "posterior_rmse": math.sqrt(posterior_mse),
                "truth_rank": rank_truth,
                "exact_top1": float(selected == ZERO_INDEX_V1),
                "top5_contains_truth": float(rank_truth <= 5),
                "hard12_win": float(
                    not hard.any() or scores[ZERO_INDEX_V1] > np.max(scores[hard])
                ),
                "entropy": float(
                    -np.sum(probability * np.log(probability + 1e-12))
                ),
            }
        )
    frame = pd.DataFrame(rows)
    weights = frame["eval_rows"].to_numpy(np.float64)
    metrics = {
        "rows": int(total_rows),
        "wells": len(frame),
        "path_rmse": float(math.sqrt(total_sse / total_rows)),
        "posterior_path_rmse": float(
            math.sqrt(total_posterior_sse / total_rows)
        ),
        "exact_top1": float(np.average(frame["exact_top1"], weights=weights)),
        "top5_contains_truth": float(
            np.average(frame["top5_contains_truth"], weights=weights)
        ),
        "median_truth_rank": float(frame["truth_rank"].median()),
        "p90_truth_rank": float(frame["truth_rank"].quantile(0.90)),
        "hard12_win_rate": float(
            np.average(frame["hard12_win"], weights=weights)
        ),
        "mean_entropy": float(np.average(frame["entropy"], weights=weights)),
    }
    return metrics, frame


def _train_epoch_v1(model, wells, optimizer, scaler, device, epoch):
    model.train()
    order = list(wells)
    random.Random(SEED_V1 + epoch).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    sums = {"loss": 0.0, "ce": 0.0, "hard_margin": 0.0}
    steps = 0
    for index, well in enumerate(order, start=1):
        panel = _load_panel_v1(well)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            loss, parts = _loss_v1(model, panel, device)
            scaled = loss / GRAD_ACCUM_WELLS_V1
        scaler.scale(scaled).backward()
        if index % GRAD_ACCUM_WELLS_V1 == 0 or index == len(order):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_V1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        sums["loss"] += float(loss.detach().cpu())
        for key, value in parts.items():
            sums[key] += value
        steps += 1
    return {key: value / max(steps, 1) for key, value in sums.items()}


device_v1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device_v1.type != "cuda":
    raise RuntimeError("CUDA is required")
_seed_everything(SEED_V1)

checkpoint_v0 = (
    Path(OOF_ROOT)
    / "fullwell_truthshape_alias_observability_fold2_v0"
    / "models"
    / "best_fold2_alias_observability.pt"
)
if not checkpoint_v0.exists():
    raise FileNotFoundError(checkpoint_v0)
payload_v0 = torch.load(checkpoint_v0, map_location="cpu", weights_only=False)
model_v1 = CompleteWellAliasEnergy().to(device_v1)
model_v1.load_state_dict(payload_v0["model"], strict=True)
parameters_v1 = int(sum(parameter.numel() for parameter in model_v1.parameters()))

optimizer_v1 = torch.optim.AdamW(
    model_v1.parameters(), lr=LEARNING_RATE_V1, weight_decay=WEIGHT_DECAY_V1
)
scheduler_v1 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_v1, T_max=MAX_EPOCHS_V1, eta_min=MIN_LEARNING_RATE_V1
)
scaler_v1 = torch.amp.GradScaler("cuda", enabled=True)

history_v1 = []
best_v1 = {"rmse": np.inf, "epoch": 0, "state": None}
bad_v1 = 0
for epoch in range(1, MAX_EPOCHS_V1 + 1):
    training = _train_epoch_v1(
        model_v1,
        train_wells_v1,
        optimizer_v1,
        scaler_v1,
        device_v1,
        epoch,
    )
    metrics, _ = _evaluate_v1(model_v1, valid_wells_v1, device_v1)
    report = {
        "epoch": epoch,
        **{f"train_{key}": value for key, value in training.items()},
        **{f"valid_{key}": value for key, value in metrics.items()},
        "learning_rate": optimizer_v1.param_groups[0]["lr"],
        "elapsed_minutes": (time.time() - started_v1) / 60.0,
    }
    history_v1.append(report)
    print("NONRIGID OBSERVABILITY EPOCH", report, flush=True)
    if metrics["path_rmse"] < best_v1["rmse"] - 1e-6:
        best_v1 = {
            "rmse": metrics["path_rmse"],
            "epoch": epoch,
            "state": {
                key: value.detach().cpu().clone()
                for key, value in model_v1.state_dict().items()
            },
        }
        bad_v1 = 0
    else:
        bad_v1 += 1
    scheduler_v1.step()
    if epoch >= MIN_EPOCHS_V1 and bad_v1 >= PATIENCE_V1:
        print("nonrigid early stopping", flush=True)
        break

if best_v1["state"] is None:
    raise RuntimeError("No nonrigid checkpoint")
model_v1.load_state_dict(best_v1["state"])
checkpoint_v1 = MODEL_DIR_V1 / "best_fold2_nonrigid_observability.pt"
torch.save(
    {
        "experiment": EXPERIMENT_V1,
        "epoch": best_v1["epoch"],
        "model": best_v1["state"],
        "warp_names": WARP_NAMES_V1,
        "parameters": parameters_v1,
    },
    checkpoint_v1,
)
pd.DataFrame(history_v1).to_csv(
    OUT_DIR_V1 / "training_history.csv", index=False
)

full_v1, full_wells_v1 = _evaluate_v1(
    model_v1, valid_wells_v1, device_v1, "full"
)
no_gr_v1, no_gr_wells_v1 = _evaluate_v1(
    model_v1, valid_wells_v1, device_v1, "no_gr"
)
shuffle_v1, shuffle_wells_v1 = _evaluate_v1(
    model_v1, valid_wells_v1, device_v1, "shuffled_alignment"
)
full_wells_v1.to_csv(OUT_DIR_V1 / "fold2_full_wells.csv", index=False)
no_gr_wells_v1.to_csv(OUT_DIR_V1 / "fold2_no_gr_wells.csv", index=False)
shuffle_wells_v1.to_csv(
    OUT_DIR_V1 / "fold2_shuffled_alignment_wells.csv", index=False
)

controls_v1 = {
    "no_gr_exact_top1_drop": full_v1["exact_top1"] - no_gr_v1["exact_top1"],
    "shuffled_exact_top1_drop": (
        full_v1["exact_top1"] - shuffle_v1["exact_top1"]
    ),
    "no_gr_path_degradation": no_gr_v1["path_rmse"] - full_v1["path_rmse"],
    "shuffled_path_degradation": (
        shuffle_v1["path_rmse"] - full_v1["path_rmse"]
    ),
}

strict_pass_v1 = bool(
    full_v1["path_rmse"] <= GATE_PATH_RMSE_V1
    and full_v1["exact_top1"] >= GATE_EXACT_TOP1_V1
    and full_v1["top5_contains_truth"] >= GATE_TOP5_V1
    and full_v1["hard12_win_rate"] >= GATE_HARD12_WIN_V1
    and controls_v1["no_gr_exact_top1_drop"] >= GATE_NO_GR_DROP_V1
    and controls_v1["shuffled_exact_top1_drop"] >= GATE_SHUFFLED_DROP_V1
)

verdict_v1 = {
    "experiment": EXPERIMENT_V1,
    "stage": "fold2_complete_well_nonrigid_shape_observability",
    "sample_unit": "one complete physical well",
    "deployable": False,
    "truth_usage": "truth defines frozen translated/warped candidates only",
    "uses_neighbor_pf_strong_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS_V1),
        "validation_fold": VALID_FOLD_V1,
        "fold1_touched": False,
        "warm_start": checkpoint_v0,
        "selected_epoch": best_v1["epoch"],
        "candidate_order_randomised": True,
        "warp_parameters_are_model_inputs": False,
    },
    "parameters": parameters_v1,
    "guards": guards_v1,
    "full": full_v1,
    "posterior_warning": (
        "Posterior path is diagnostic only because the candidate family is "
        "approximately symmetric; promotion uses hard path RMSE and ranks."
    ),
    "no_gr": no_gr_v1,
    "shuffled_alignment": shuffle_v1,
    "controls": controls_v1,
    "gates": {
        "path_rmse_le": GATE_PATH_RMSE_V1,
        "exact_top1_ge": GATE_EXACT_TOP1_V1,
        "top5_contains_truth_ge": GATE_TOP5_V1,
        "hard12_win_rate_ge": GATE_HARD12_WIN_V1,
        "no_gr_exact_top1_drop_ge": GATE_NO_GR_DROP_V1,
        "shuffled_exact_top1_drop_ge": GATE_SHUFFLED_DROP_V1,
    },
    "strict_pass": strict_pass_v1,
    "checkpoint": checkpoint_v1,
    "elapsed_minutes": (time.time() - started_v1) / 60.0,
    "next": (
        "PASS -> transfer this energy encoder to a label-free complete-well mode bank"
        if strict_pass_v1
        else "FAIL -> GR energy resolves rigid aliases but not deployable path-shape variation"
    ),
}
(OUT_DIR_V1 / "verdict.json").write_text(
    json.dumps(verdict_v1, indent=2, default=_json_default), encoding="utf-8"
)

print("FULL", json.dumps(full_v1, indent=2), flush=True)
print("CONTROLS", json.dumps(controls_v1, indent=2), flush=True)
print("FINAL VERDICT", json.dumps(verdict_v1, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR_V1, flush=True)

del model_v1
gc.collect()
torch.cuda.empty_cache()
