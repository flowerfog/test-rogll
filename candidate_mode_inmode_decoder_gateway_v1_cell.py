"""Fold-2 gateway for a hierarchical full-well candidate decoder.

The zero-training mode audit established that a frozen unary retriever retains
an excellent path in its top eight *modes* (4.87-ft oracle), but that replacing
a mode by a single medoid destroys material local shape detail.  This script
tests only the missing first layer: a GR-conditioned decoder which selects a
precise candidate *within* a retrieved mode.

The global retriever is frozen.  Train wells use label-free top-eight global
modes plus deterministic random modes; target TVT is used only in the local
ranking loss.  Fold 2 is a same-fold gateway and Fold 1 is never read.

This does not yet claim a deployable 5.5-ft model.  It authorises the second
set-context mode scorer only if local decoding retains the Fold-2 top-eight
candidate ceiling.
"""

from __future__ import annotations

from pathlib import Path
import gc
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Restore only the definitions from the prior mode audit and its V0/V1 chain.
# ---------------------------------------------------------------------------


SOURCE_CANDIDATES_LOCAL = (
    Path("/root/autodl-tmp/candidate_mode_posterior_audit_v0_cell.py"),
    Path.cwd() / "candidate_mode_posterior_audit_v0_cell.py",
)
SOURCE_PATH_LOCAL = next(
    (path for path in SOURCE_CANDIDATES_LOCAL if path.exists()), None
)
if SOURCE_PATH_LOCAL is None:
    raise FileNotFoundError(
        "Upload candidate_mode_posterior_audit_v0_cell.py plus its V0/V1 "
        "dependencies to /root/autodl-tmp before running this gateway."
    )
source_local = SOURCE_PATH_LOCAL.read_text(encoding="utf-8")
marker_local = "# Frozen Fold-2 application."
if marker_local not in source_local:
    raise RuntimeError(f"Cannot locate audit execution marker in {SOURCE_PATH_LOCAL}")
exec(
    compile(source_local.split(marker_local, 1)[0], str(SOURCE_PATH_LOCAL), "exec"),
    globals(),
)

REQUIRED_LOCAL = {
    "OOF_ROOT",
    "MODEL_DIR_V1",
    "VALID_FOLD",
    "TRAIN_FOLDS",
    "_load_folds",
    "_render_full_bank_v1",
    "_score_arrays_v1",
    "_cluster_modes_mode",
    "_mode_scores_mode",
    "_summarize_rows_v1",
    "FullBankUnaryEnergyV1",
    "_seed_everything",
}
missing_local = sorted(REQUIRED_LOCAL.difference(globals()))
if missing_local:
    raise RuntimeError("Mode-audit bootstrap missing: " + repr(missing_local))
print("IN-MODE bootstrap:", SOURCE_PATH_LOCAL, flush=True)


# ---------------------------------------------------------------------------
# Frozen configuration.  Fold 2 does not select these settings.
# ---------------------------------------------------------------------------


EXPERIMENT_LOCAL = "candidate_mode_inmode_decoder_gateway_v1"
OUT_DIR_LOCAL = Path(OOF_ROOT) / EXPERIMENT_LOCAL
TRAIN_CACHE_LOCAL = OUT_DIR_LOCAL / "train_panels"
VALID_CACHE_LOCAL = OUT_DIR_LOCAL / "fold2_panels"
MODEL_DIR_LOCAL = OUT_DIR_LOCAL / "models"
for directory_local in (OUT_DIR_LOCAL, TRAIN_CACHE_LOCAL, VALID_CACHE_LOCAL, MODEL_DIR_LOCAL):
    directory_local.mkdir(parents=True, exist_ok=True)

SEED_LOCAL = 20260731
MODE_COUNT_LOCAL = 96
TOP_GLOBAL_MODES_LOCAL = 8
RANDOM_TRAIN_MODES_LOCAL = 4
MAX_MEMBERS_PER_MODE_LOCAL = 128
MAX_EPOCHS_LOCAL = 8
MIN_EPOCHS_LOCAL = 3
PATIENCE_LOCAL = 2
LEARNING_RATE_LOCAL = 1.0e-4
MIN_LEARNING_RATE_LOCAL = 2.0e-5
WEIGHT_DECAY_LOCAL = 4.0e-5
GRAD_ACCUM_WELLS_LOCAL = 2
GRADIENT_CLIP_LOCAL = 4.0
SOFT_TEMPERATURE_LOCAL = 1.25
HARD_GAP_LOCAL = 3.0
LOCAL_MARGIN_LOCAL = 0.50

# These are gateway criteria, not validation claims.  The top-eight member
# oracle from the prior audit is 4.873 ft, so 5.2 leaves little headroom.
GATE_LOCAL_CHOICE_ORACLE_LOCAL = 5.20
GATE_HIERARCHICAL_MAP_LOCAL = 8.00
GATE_LOCAL_ORACLE_REGRET_LOCAL = 0.35
GATE_NO_GR_DEGRADATION_LOCAL = 0.50


def _json_default_local(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json_local(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default_local), encoding="utf-8"
    )


def _panel_path_local(directory, well):
    return directory / f"{well}.npz"


def _active_modes_local(bank, global_scores, well):
    """Cluster only deployable candidates and score modes without truth."""
    deployable = np.flatnonzero(np.isfinite(bank["candidate_mse"]))
    if len(deployable) < MODE_COUNT_LOCAL:
        raise RuntimeError(f"only {len(deployable)} deployable candidates for {well}")
    local_labels, local_members, local_medoids, sizes = _cluster_modes_mode(
        bank["path_delta"][deployable], MODE_COUNT_LOCAL, well
    )
    members = [deployable[indices].astype(np.int32) for indices in local_members]
    medoids = deployable[local_medoids].astype(np.int32)
    scores = _mode_scores_mode(global_scores, members)["logmeanexp"]
    order = np.argsort(-scores, kind="mergesort").astype(np.int16)
    return members, medoids, sizes.astype(np.int32), scores.astype(np.float32), order


def _truncate_members_local(indices, bank, global_scores, medoid, limit):
    """Cap pathological 500-member aliases with only label-free choices."""
    if len(indices) <= limit:
        return np.asarray(indices, np.int32)
    selected = []
    seen = set()

    def add(values):
        for value in values:
            value = int(value)
            if value in seen:
                continue
            seen.add(value)
            selected.append(value)
            if len(selected) >= limit:
                return

    add([medoid])
    ordered_global = indices[np.argsort(-global_scores[indices], kind="mergesort")]
    add(ordered_global[: min(48, len(ordered_global))])
    # V1's heuristic has its most plausible paths at lower rank values.
    ordered_heuristic = indices[np.argsort(bank["heuristic"][indices], kind="mergesort")]
    add(ordered_heuristic[: min(32, len(ordered_heuristic))])
    add(indices[np.linspace(0, len(indices) - 1, num=limit, dtype=np.int32)])
    return np.asarray(selected[:limit], np.int32)


def _choose_train_modes_local(order, well):
    selected = [int(value) for value in order[:TOP_GLOBAL_MODES_LOCAL]]
    remaining = np.asarray(
        [value for value in range(MODE_COUNT_LOCAL) if value not in set(selected)],
        np.int16,
    )
    rng = np.random.default_rng(SEED_LOCAL + int(str(well), 16) % 1000003)
    if len(remaining):
        extra = rng.choice(
            remaining,
            size=min(RANDOM_TRAIN_MODES_LOCAL, len(remaining)),
            replace=False,
        )
        selected.extend(int(value) for value in extra)
    return np.asarray(selected, np.int16)


def _pack_panel_local(bank, global_scores, well, train):
    members, medoids, sizes, mode_scores, order = _active_modes_local(
        bank, global_scores, well
    )
    modes = _choose_train_modes_local(order, well) if train else order[:TOP_GLOBAL_MODES_LOCAL]
    candidate_indices = []
    starts = [0]
    for mode in modes:
        indices = _truncate_members_local(
            members[int(mode)],
            bank,
            global_scores,
            medoids[int(mode)],
            MAX_MEMBERS_PER_MODE_LOCAL,
        )
        candidate_indices.append(indices)
        starts.append(starts[-1] + len(indices))
    candidate_indices = np.concatenate(candidate_indices).astype(np.int32)
    starts = np.asarray(starts, np.int32)
    selected_scores = mode_scores[modes].astype(np.float32)
    return {
        "well": str(well),
        "query": bank["query"].astype(np.float32),
        "reference": bank["reference"][candidate_indices].astype(np.float32),
        "valid": bank["valid"][candidate_indices].astype(bool),
        "path_delta": bank["path_delta"][candidate_indices].astype(np.float32),
        "heuristic": bank["heuristic"][candidate_indices].astype(np.float32),
        "candidate_mse": bank["candidate_mse"][candidate_indices].astype(np.float32),
        "candidate_mae": bank["candidate_mae"][candidate_indices].astype(np.float32),
        "strata_mse": bank["strata_mse"][candidate_indices].astype(np.float32),
        "strata_counts": bank["strata_counts"].astype(np.int32),
        "candidate_indices": candidate_indices,
        "starts": starts,
        "mode_ids": modes.astype(np.int16),
        "mode_scores": selected_scores,
        "mode_sizes": sizes[modes].astype(np.int32),
        "eval_rows": int(bank["eval_rows"]),
        "full_mode_member_oracle_mse": np.asarray(
            [bank["candidate_mse"][members[int(mode)]].min() for mode in modes],
            np.float32,
        ),
    }


def _save_panel_local(path, panel):
    np.savez_compressed(
        path,
        well=np.asarray(panel["well"]),
        query=panel["query"].astype(np.float16),
        reference=panel["reference"].astype(np.float16),
        valid=panel["valid"].astype(np.uint8),
        path_delta=panel["path_delta"].astype(np.float16),
        heuristic=panel["heuristic"].astype(np.float16),
        candidate_mse=panel["candidate_mse"].astype(np.float32),
        candidate_mae=panel["candidate_mae"].astype(np.float32),
        strata_mse=panel["strata_mse"].astype(np.float32),
        strata_counts=panel["strata_counts"].astype(np.int32),
        candidate_indices=panel["candidate_indices"].astype(np.int32),
        starts=panel["starts"].astype(np.int32),
        mode_ids=panel["mode_ids"].astype(np.int16),
        mode_scores=panel["mode_scores"].astype(np.float32),
        mode_sizes=panel["mode_sizes"].astype(np.int32),
        eval_rows=np.asarray([panel["eval_rows"]], np.int32),
        full_mode_member_oracle_mse=panel["full_mode_member_oracle_mse"].astype(np.float32),
        cache_version=np.asarray([1], np.int16),
    )


def _load_panel_local(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "well": str(data["well"].item()),
            "query": data["query"].astype(np.float32),
            "reference": data["reference"].astype(np.float32),
            "valid": data["valid"].astype(bool),
            "path_delta": data["path_delta"].astype(np.float32),
            "heuristic": data["heuristic"].astype(np.float32),
            "candidate_mse": data["candidate_mse"].astype(np.float32),
            "candidate_mae": data["candidate_mae"].astype(np.float32),
            "strata_mse": data["strata_mse"].astype(np.float32),
            "strata_counts": data["strata_counts"].astype(np.int32),
            "candidate_indices": data["candidate_indices"].astype(np.int32),
            "starts": data["starts"].astype(np.int32),
            "mode_ids": data["mode_ids"].astype(np.int16),
            "mode_scores": data["mode_scores"].astype(np.float32),
            "mode_sizes": data["mode_sizes"].astype(np.int32),
            "eval_rows": int(data["eval_rows"][0]),
            "full_mode_member_oracle_mse": data[
                "full_mode_member_oracle_mse"
            ].astype(np.float32),
        }


def _tensor_local(panel, device, control="full"):
    query = panel["query"].copy()
    reference = panel["reference"].copy()
    valid = panel["valid"].astype(np.float32).copy()
    heuristic = panel["heuristic"].copy()
    if control == "no_gr":
        query.fill(0.0)
        heuristic.fill(0.5)
    elif control == "shuffled_alignment":
        shift = max(1, len(reference) // 3)
        indices = (np.arange(len(reference)) + shift) % len(reference)
        reference = reference[indices]
        valid = valid[indices]
        heuristic = heuristic[indices]
    elif control != "full":
        raise ValueError(control)
    return _tensor_arrays_v1(
        query,
        reference,
        valid,
        panel["path_delta"],
        heuristic,
        device,
    )


def _local_loss(model, panel, device):
    scores = model(*_tensor_local(panel, device))
    mse = torch.from_numpy(panel["candidate_mse"]).to(device)
    starts = panel["starts"]
    losses = []
    hard_losses = []
    soft_losses = []
    margins = []
    for group in range(len(starts) - 1):
        left, right = int(starts[group]), int(starts[group + 1])
        values = scores[left:right]
        group_mse = mse[left:right]
        rmse = torch.sqrt(group_mse.clamp_min(0.0) + 1e-8)
        target = torch.argmin(group_mse)
        hard_ce = F.cross_entropy(values[None], target[None])
        soft_target = torch.softmax(
            -(rmse - rmse.min()) / SOFT_TEMPERATURE_LOCAL, dim=0
        )
        soft_ce = -(soft_target * F.log_softmax(values, dim=0)).sum()
        hard = rmse >= rmse[target] + HARD_GAP_LOCAL
        margin = (
            F.relu(LOCAL_MARGIN_LOCAL - values[target] + values[hard].amax())
            if hard.any()
            else values[target] * 0.0
        )
        losses.append(0.50 * hard_ce + 0.50 * soft_ce + 0.15 * margin)
        hard_losses.append(hard_ce)
        soft_losses.append(soft_ce)
        margins.append(margin)
    loss = torch.stack(losses).mean()
    return loss, {
        "hard_ce": float(torch.stack(hard_losses).mean().detach().cpu()),
        "soft_ce": float(torch.stack(soft_losses).mean().detach().cpu()),
        "margin": float(torch.stack(margins).mean().detach().cpu()),
        "groups": int(len(losses)),
    }


def _row_local(panel, index, variant, **extra):
    row = {
        "well": panel["well"],
        "variant": variant,
        "rows": int(panel["eval_rows"]),
        "mse": float(panel["candidate_mse"][index]),
        "mae": float(panel["candidate_mae"][index]),
    }
    for stratum, name in enumerate(STRATA_NAMES):
        row[f"{name}_rows"] = int(panel["strata_counts"][stratum])
        row[f"{name}_mse"] = float(panel["strata_mse"][index, stratum])
    row.update(extra)
    return row


def _hierarchical_choices_local(scores, panel):
    starts = panel["starts"]
    local_indices = []
    conditional = np.full(len(scores), -np.inf, np.float64)
    for group in range(len(starts) - 1):
        left, right = int(starts[group]), int(starts[group + 1])
        local = scores[left:right].astype(np.float64)
        local_indices.append(left + int(np.argmax(local)))
        maximum = float(local.max())
        normalizer = maximum + math.log(float(np.exp(local - maximum).sum()))
        conditional[left:right] = local - normalizer + float(panel["mode_scores"][group])
    return np.asarray(local_indices, np.int32), int(np.argmax(conditional))


@torch.no_grad()
def _evaluate_local(model, wells, device, control="full"):
    rows = {
        "global_top1_local": [],
        "hierarchical_map": [],
        "local_choice_oracle": [],
        "top8_member_oracle": [],
    }
    local_regrets = []
    local_hits = []
    for well in wells:
        panel = _load_panel_local(_panel_path_local(VALID_CACHE_LOCAL, well))
        model.eval()
        scores = model(*_tensor_local(panel, device, control)).detach().cpu().numpy()
        chosen, hierarchical = _hierarchical_choices_local(scores, panel)
        rows["global_top1_local"].append(
            _row_local(panel, int(chosen[0]), "global_top1_local")
        )
        rows["hierarchical_map"].append(
            _row_local(panel, hierarchical, "hierarchical_map")
        )
        oracle_chosen = int(chosen[np.argmin(panel["candidate_mse"][chosen])])
        rows["local_choice_oracle"].append(
            _row_local(panel, oracle_chosen, "local_choice_oracle")
        )
        member_oracle = int(np.argmin(panel["candidate_mse"]))
        rows["top8_member_oracle"].append(
            _row_local(panel, member_oracle, "top8_member_oracle")
        )
        starts = panel["starts"]
        for group in range(len(starts) - 1):
            left, right = int(starts[group]), int(starts[group + 1])
            chosen_index = int(chosen[group])
            oracle_mse = float(panel["full_mode_member_oracle_mse"][group])
            chosen_mse = float(panel["candidate_mse"][chosen_index])
            local_regrets.append(max(0.0, chosen_mse - oracle_mse))
            local_hits.append(chosen_mse <= oracle_mse + 1.0)
    reports = {}
    frames = {}
    for name, values in rows.items():
        frame = pd.DataFrame(values)
        frames[name] = frame
        reports[name] = _summarize_rows_v1(frame)
    diagnostics = {
        "mean_local_mse_regret": float(np.mean(local_regrets)),
        "median_local_mse_regret": float(np.median(local_regrets)),
        "within_1ft_mode_oracle_fraction": float(np.mean(local_hits)),
    }
    return reports, frames, diagnostics


def _train_epoch_local(model, wells, optimizer, scaler, device, epoch):
    model.train()
    order = list(wells)
    random.Random(SEED_LOCAL + epoch).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "hard_ce": 0.0, "soft_ce": 0.0, "margin": 0.0, "groups": 0}
    steps = 0
    for number, well in enumerate(order, start=1):
        panel = _load_panel_local(_panel_path_local(TRAIN_CACHE_LOCAL, well))
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            loss, parts = _local_loss(model, panel, device)
            scaled = loss / GRAD_ACCUM_WELLS_LOCAL
        scaler.scale(scaled).backward()
        if number % GRAD_ACCUM_WELLS_LOCAL == 0 or number == len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_LOCAL)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        totals["loss"] += float(loss.detach().cpu())
        for key in ("hard_ce", "soft_ce", "margin", "groups"):
            totals[key] += parts[key]
        if number % 100 == 0 or number == len(order):
            print(
                f"INMODE epoch={epoch} wells={number}/{len(order)} "
                f"loss={totals['loss'] / number:.4f}",
                flush=True,
            )
    output = {key: value / len(order) for key, value in totals.items()}
    output["optimizer_steps"] = steps
    return output


# ---------------------------------------------------------------------------
# Build immutable mode panels from label-free global retrieval decisions.
# ---------------------------------------------------------------------------


_seed_everything(SEED_LOCAL)
folds_local = _load_folds()
train_wells_local = sorted(
    folds_local.loc[folds_local["fold"].isin(TRAIN_FOLDS), "well"].astype(str).tolist()
)
valid_wells_local = sorted(
    folds_local.loc[folds_local["fold"].eq(VALID_FOLD), "well"].astype(str).tolist()
)
if len(train_wells_local) < 450 or len(valid_wells_local) < 150:
    raise RuntimeError(f"Unexpected folds {len(train_wells_local)}/{len(valid_wells_local)}")

global_checkpoint_path_local = Path(MODEL_DIR_V1) / "best_fold2_unary_retriever.pt"
if not global_checkpoint_path_local.exists():
    raise FileNotFoundError(global_checkpoint_path_local)
device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_checkpoint_local = torch.load(
    global_checkpoint_path_local, map_location=device_local, weights_only=False
)
global_model_local = FullBankUnaryEnergyV1().to(device_local)
global_model_local.load_state_dict(global_checkpoint_local["model"])
global_model_local.eval()


def _build_one_panel_local(well, directory, train):
    destination = _panel_path_local(directory, well)
    if destination.exists():
        panel = _load_panel_local(destination)
        return {
            "well": well,
            "status": "cached",
            "candidates": int(len(panel["candidate_mse"])),
            "groups": int(len(panel["starts"]) - 1),
        }
    bank = _render_full_bank_v1(well)
    global_scores = _score_arrays_v1(
        global_model_local,
        bank["query"],
        bank["reference"],
        bank["valid"],
        bank["path_delta"],
        bank["heuristic"],
        device_local,
        control="full",
    )
    panel = _pack_panel_local(bank, global_scores, well, train=train)
    _save_panel_local(destination, panel)
    report = {
        "well": well,
        "status": "ok",
        "candidates": int(len(panel["candidate_mse"])),
        "groups": int(len(panel["starts"]) - 1),
        "top8_member_oracle_rmse": float(np.sqrt(panel["candidate_mse"].min())),
    }
    del bank, global_scores, panel
    gc.collect()
    return report


started_local = time.time()
build_rows_local = []
for label_local, wells_local, directory_local, is_train_local in (
    ("train", train_wells_local, TRAIN_CACHE_LOCAL, True),
    (f"fold{VALID_FOLD}", valid_wells_local, VALID_CACHE_LOCAL, False),
):
    for number_local, well_local in enumerate(wells_local, start=1):
        try:
            build_rows_local.append(
                {"subset": label_local, **_build_one_panel_local(well_local, directory_local, is_train_local)}
            )
        except Exception as error:
            build_rows_local.append(
                {"subset": label_local, "well": well_local, "status": "failed", "reason": repr(error)}
            )
        if number_local % 25 == 0 or number_local == len(wells_local):
            print(
                f"build in-mode {label_local} {number_local}/{len(wells_local)} "
                f"elapsed={(time.time() - started_local) / 60.0:.1f}m",
                flush=True,
            )
build_frame_local = pd.DataFrame(build_rows_local)
build_frame_local.to_csv(OUT_DIR_LOCAL / "panel_build_report.csv", index=False)
failed_local = build_frame_local[build_frame_local["status"].eq("failed")]
if len(failed_local):
    raise RuntimeError(
        f"{len(failed_local)} mode panel builds failed: "
        + repr(failed_local.head(10).to_dict(orient="records"))
    )

guards_local = {
    "train_wells": len(train_wells_local),
    "valid_wells": len(valid_wells_local),
    "train_candidates_median": float(
        build_frame_local.loc[build_frame_local["subset"].eq("train"), "candidates"].median()
    ),
    "valid_candidates_median": float(
        build_frame_local.loc[build_frame_local["subset"].ne("train"), "candidates"].median()
    ),
    "fold1_touched": False,
}
print("IN-MODE PANEL GUARDS", json.dumps(guards_local, indent=2), flush=True)


# ---------------------------------------------------------------------------
# Train the local conditional decoder, warm-started only from the frozen V1
# energy encoder.  The Fold-2 epoch choice is a gateway diagnostic, not OOF.
# ---------------------------------------------------------------------------


local_model = FullBankUnaryEnergyV1().to(device_local)
local_model.load_state_dict(global_checkpoint_local["model"])
parameters_local = int(sum(parameter.numel() for parameter in local_model.parameters()))
optimizer_local = torch.optim.AdamW(
    local_model.parameters(), lr=LEARNING_RATE_LOCAL, weight_decay=WEIGHT_DECAY_LOCAL
)
scheduler_local = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_local, T_max=MAX_EPOCHS_LOCAL, eta_min=MIN_LEARNING_RATE_LOCAL
)
scaler_local = torch.amp.GradScaler("cuda", enabled=device_local.type == "cuda")
checkpoint_path_local = MODEL_DIR_LOCAL / "best_fold2_inmode_decoder.pt"
history_local = []
best_local = None
bad_epochs_local = 0

for epoch_local in range(1, MAX_EPOCHS_LOCAL + 1):
    train_metrics_local = _train_epoch_local(
        local_model, train_wells_local, optimizer_local, scaler_local, device_local, epoch_local
    )
    valid_reports_local, _, valid_diag_local = _evaluate_local(
        local_model, valid_wells_local, device_local
    )
    selection_metric_local = valid_reports_local["local_choice_oracle"]["rmse"]
    report_local = {
        "epoch": epoch_local,
        **{f"train_{key}": value for key, value in train_metrics_local.items()},
        **{
            f"valid_{name}_{key}": value
            for name, values in valid_reports_local.items()
            for key, value in values.items()
        },
        **{f"valid_{key}": value for key, value in valid_diag_local.items()},
        "learning_rate": float(optimizer_local.param_groups[0]["lr"]),
        "elapsed_minutes": (time.time() - started_local) / 60.0,
    }
    history_local.append(report_local)
    print("IN-MODE EPOCH", report_local, flush=True)
    if best_local is None or selection_metric_local < best_local["selection_rmse"] - 1e-4:
        best_local = {"epoch": epoch_local, "selection_rmse": selection_metric_local}
        torch.save(
            {
                "model": local_model.state_dict(),
                "epoch": epoch_local,
                "selection_metric": selection_metric_local,
                "global_checkpoint": str(global_checkpoint_path_local),
            },
            checkpoint_path_local,
        )
        bad_epochs_local = 0
    else:
        bad_epochs_local += 1
    scheduler_local.step()
    if epoch_local >= MIN_EPOCHS_LOCAL and bad_epochs_local >= PATIENCE_LOCAL:
        print("IN-MODE early stopping", flush=True)
        break

pd.DataFrame(history_local).to_csv(OUT_DIR_LOCAL / "training_history.csv", index=False)
best_checkpoint_local = torch.load(
    checkpoint_path_local, map_location=device_local, weights_only=False
)
local_model.load_state_dict(best_checkpoint_local["model"])
reports_local, frames_local, diagnostics_local = _evaluate_local(
    local_model, valid_wells_local, device_local
)
for name_local, frame_local in frames_local.items():
    frame_local.to_csv(OUT_DIR_LOCAL / f"fold{VALID_FOLD}_{name_local}_bywell.csv", index=False)

no_gr_reports_local, _, _ = _evaluate_local(
    local_model, valid_wells_local, device_local, control="no_gr"
)
shuffle_reports_local, _, _ = _evaluate_local(
    local_model, valid_wells_local, device_local, control="shuffled_alignment"
)
controls_local = {
    "no_gr_hierarchical_map_degradation": float(
        no_gr_reports_local["hierarchical_map"]["rmse"]
        - reports_local["hierarchical_map"]["rmse"]
    ),
    "shuffled_hierarchical_map_degradation": float(
        shuffle_reports_local["hierarchical_map"]["rmse"]
        - reports_local["hierarchical_map"]["rmse"]
    ),
    "no_gr_local_choice_oracle_degradation": float(
        no_gr_reports_local["local_choice_oracle"]["rmse"]
        - reports_local["local_choice_oracle"]["rmse"]
    ),
}

oracle_regret_local = float(
    reports_local["local_choice_oracle"]["rmse"]
    - reports_local["top8_member_oracle"]["rmse"]
)
checks_local = {
    "local_choice_oracle": bool(
        reports_local["local_choice_oracle"]["rmse"] <= GATE_LOCAL_CHOICE_ORACLE_LOCAL
    ),
    "local_oracle_regret": bool(oracle_regret_local <= GATE_LOCAL_ORACLE_REGRET_LOCAL),
    "hierarchical_map": bool(
        reports_local["hierarchical_map"]["rmse"] <= GATE_HIERARCHICAL_MAP_LOCAL
    ),
    "no_gr": bool(
        controls_local["no_gr_hierarchical_map_degradation"] >= GATE_NO_GR_DEGRADATION_LOCAL
    ),
}
strict_pass_local = bool(
    checks_local["local_choice_oracle"]
    and checks_local["local_oracle_regret"]
    and checks_local["no_gr"]
)
verdict_local = {
    "experiment": EXPERIMENT_LOCAL,
    "stage": "fold2_inmode_local_decoder_gateway",
    "sample_unit": "complete physical well, conditionally decoded within a mode",
    "truth_usage": "outer-train local ranking labels and Fold-2 metrics only",
    "uses_neighbor_spatial_strong_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
        "global_retriever_checkpoint": str(global_checkpoint_path_local),
        "global_retriever_epoch": int(global_checkpoint_local["epoch"]),
        "selected_local_epoch": int(best_checkpoint_local["epoch"]),
        "mode_count": MODE_COUNT_LOCAL,
        "top_global_modes": TOP_GLOBAL_MODES_LOCAL,
    },
    "parameters": parameters_local,
    "guards": guards_local,
    "full": reports_local,
    "diagnostics": diagnostics_local,
    "oracle_regret_rmse": oracle_regret_local,
    "no_gr": no_gr_reports_local,
    "shuffled_alignment": shuffle_reports_local,
    "controls": controls_local,
    "gates": {
        "local_choice_oracle_rmse_le": GATE_LOCAL_CHOICE_ORACLE_LOCAL,
        "local_oracle_regret_rmse_le": GATE_LOCAL_ORACLE_REGRET_LOCAL,
        "hierarchical_map_rmse_le": GATE_HIERARCHICAL_MAP_LOCAL,
        "no_gr_hierarchical_map_degradation_ge": GATE_NO_GR_DEGRADATION_LOCAL,
    },
    "checks": checks_local,
    "strict_pass": strict_pass_local,
    "checkpoint": str(checkpoint_path_local),
    "elapsed_minutes": (time.time() - started_local) / 60.0,
}
if strict_pass_local:
    verdict_local["next"] = (
        "PASS -> build inner-OOF global retrieval features and train the second "
        "set-context scorer over exactly eight locally decoded modes."
    )
elif checks_local["local_choice_oracle"]:
    verdict_local["next"] = (
        "PARTIAL -> local decoding preserves the useful paths, but global mode "
        "selection remains unresolved; do not submit or touch Fold 1."
    )
else:
    verdict_local["next"] = (
        "FAIL -> GR energy cannot resolve local shape even after mode conditioning; "
        "close this candidate bank."
    )
_write_json_local(OUT_DIR_LOCAL / "verdict.json", verdict_local)
print("IN-MODE FINAL VERDICT")
print(json.dumps(verdict_local, indent=2, default=_json_default_local), flush=True)
print("saved:", OUT_DIR_LOCAL, flush=True)
