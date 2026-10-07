"""Full-bank unary complete-well GR retriever.

Run after candidate_conditioned_fullwell_gateway_v0 has produced its template
bank and Fold-2 panels. V0 proved that the 3216-path bank has a 3.91-ft oracle
but its fixed heuristic loses too many modes before the 192-path reranker.

This follow-up trains a candidate-independent complete-well energy function on
folds 3/4/5. Training panels contain supervised positives, alias negatives,
and random negatives. At validation every one of the 3216 label-free paths is
rendered and scored in chunks; no heuristic pre-pruning controls survival.

Fold 1 is never loaded. No neighbour, spatial strong, CP12, Gold, or LB signal
is consumed.
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
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Restore only V0 definitions. The V0 execution tail is not run.
# ---------------------------------------------------------------------------


SOURCE_CANDIDATES = (
    Path("/root/autodl-tmp/candidate_conditioned_fullwell_gateway_v0_cell.py"),
    Path.cwd() / "candidate_conditioned_fullwell_gateway_v0_cell.py",
)
SOURCE_PATH = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError(
        "Upload candidate_conditioned_fullwell_gateway_v0_cell.py to "
        "/root/autodl-tmp before running V1."
    )
source = SOURCE_PATH.read_text(encoding="utf-8")
marker = "# Execute the nested gateway"
if marker not in source:
    raise RuntimeError(f"Cannot find V0 definition marker in {SOURCE_PATH}")
exec(compile(source.split(marker, 1)[0], str(SOURCE_PATH), "exec"), globals())

REQUIRED = {
    "DATA_ROOT",
    "OOF_ROOT",
    "OUT_DIR",
    "PANEL_DIR",
    "BASE_MENU",
    "TEMPLATE_GRID",
    "SEQUENCE_LENGTH",
    "MIN_REFERENCE_COVERAGE",
    "TRAIN_FOLDS",
    "VALID_FOLD",
    "STRATA_NAMES",
    "_load_folds",
    "_well_payload",
    "_prepare_typewell",
    "_fit_affine",
    "_interp_finite",
    "_robust_location_scale",
    "_candidate_errors",
    "_heuristic_scores",
    "_percentile_rank",
    "_load_panel",
    "_seed_everything",
    "_write_json",
}
missing = sorted(REQUIRED.difference(globals()))
if missing:
    raise RuntimeError("V0 bootstrap missing: " + repr(missing))
print("V1 bootstrap:", SOURCE_PATH, flush=True)


# ---------------------------------------------------------------------------
# V1 configuration
# ---------------------------------------------------------------------------


V0_OUT_DIR = Path(OUT_DIR)
EXPERIMENT_V1 = "candidate_conditioned_fullbank_retriever_v1"
OUT_DIR_V1 = Path(OOF_ROOT) / EXPERIMENT_V1
SAMPLE_DIR_V1 = OUT_DIR_V1 / "train_samples"
MODEL_DIR_V1 = OUT_DIR_V1 / "models"
for directory in (OUT_DIR_V1, SAMPLE_DIR_V1, MODEL_DIR_V1):
    directory.mkdir(parents=True, exist_ok=True)

TEMPLATE_PATH_V1 = V0_OUT_DIR / "outer_train_template_bank.npz"
if not TEMPLATE_PATH_V1.exists():
    raise FileNotFoundError(TEMPLATE_PATH_V1)
with np.load(TEMPLATE_PATH_V1, allow_pickle=False) as data:
    TEMPLATE_NAMES_V1 = data["names"].astype(str)
    TEMPLATES_V1 = data["templates"].astype(np.float32)
    TEMPLATE_GRID_V1 = data["grid"].astype(np.float32)
if not np.array_equal(TEMPLATE_GRID_V1, TEMPLATE_GRID):
    raise RuntimeError("Template grid differs from the V0 source")

SEED_V1 = 20260726
TRAIN_SAMPLE_CANDIDATES_V1 = 160
INFERENCE_CHUNK_V1 = 256
RETRIEVAL_K_V1 = 192

MAX_EPOCHS_V1 = 10
MIN_EPOCHS_V1 = 4
PATIENCE_V1 = 3
LEARNING_RATE_V1 = 2.5e-4
MIN_LEARNING_RATE_V1 = 3.0e-5
WEIGHT_DECAY_V1 = 4.0e-5
GRAD_ACCUM_WELLS_V1 = 4
GRADIENT_CLIP_V1 = 4.0

EMBEDDING_V1 = 40
HIDDEN_V1 = 96
SOFT_TEMPERATURE_FT_V1 = 1.5
HARD_GAP_FT_V1 = 4.0
LOGIT_MARGIN_V1 = 0.75

GATE_RETRIEVED_ORACLE_V1 = 4.50
GATE_MODEL_RMSE_V1 = 8.00
GATE_MODEL_MEDIAN_V1 = 6.50
GATE_MODEL_FAR_V1 = 10.00
GATE_CONTROL_DEGRADATION_V1 = 1.00


def _json_default_v1(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _sample_path_v1(well):
    return SAMPLE_DIR_V1 / f"{well}.npz"


# ---------------------------------------------------------------------------
# Render the full 3216-path bank for one well
# ---------------------------------------------------------------------------


def _render_full_bank_v1(well):
    payload = _well_payload(well)
    typewell_tvt, typewell_gr = _prepare_typewell(payload["typewell"])
    known = payload["known"]
    tw_prefix = _interp_finite(
        typewell_tvt, typewell_gr, payload["tvt_input"][known]
    )
    slope, intercept, pairs, calibration_rmse = _fit_affine(
        tw_prefix, payload["gr"][known]
    )
    calibrated_typewell = slope * typewell_gr + intercept
    centre, scale = _robust_location_scale(payload["gr"][known])

    sequence_md = np.linspace(
        payload["eval_md"][0], payload["eval_md"][-1], SEQUENCE_LENGTH
    )
    sequence_progress = np.interp(
        sequence_md, payload["eval_md"], payload["progress"]
    )
    query = _interp_finite(payload["md"], payload["gr"], sequence_md)
    query_valid = np.isfinite(query)
    query = np.clip((query - centre) / scale, -8.0, 8.0)
    query = np.where(query_valid, query, 0.0).astype(np.float32)

    base_sequence = np.stack(
        [
            np.interp(sequence_md, payload["eval_md"], base)
            for base in payload["bases"]
        ]
    ).astype(np.float32)
    template_sequence = np.stack(
        [
            np.interp(sequence_progress, TEMPLATE_GRID_V1, row)
            for row in TEMPLATES_V1
        ]
    ).astype(np.float32)
    paths = (
        base_sequence[:, None, :] + template_sequence[None, :, :]
    ).reshape(-1, SEQUENCE_LENGTH)
    valid = (
        (paths >= typewell_tvt[0])
        & (paths <= typewell_tvt[-1])
        & query_valid[None, :]
    )
    reference = np.interp(
        paths.reshape(-1), typewell_tvt, calibrated_typewell
    ).reshape(paths.shape)
    reference = np.clip((reference - centre) / scale, -8.0, 8.0)
    reference = np.where(valid, reference, 0.0).astype(np.float32)
    heuristic, _ = _heuristic_scores(query, reference, valid)
    heuristic = _percentile_rank(heuristic).astype(np.float32)

    mse, mae, strata_mse, strata_counts, base_ids, template_ids = _candidate_errors(
        payload, TEMPLATES_V1
    )
    coverage = valid.mean(axis=1)
    deployable = coverage >= MIN_REFERENCE_COVERAGE
    mse = np.where(deployable, mse, np.inf).astype(np.float32)
    candidate_ids = np.column_stack([base_ids, template_ids]).astype(np.int16)
    return {
        "well": str(well),
        "query": query,
        "reference": reference,
        "valid": valid,
        "path_delta": (paths - payload["anchor_tvt"]).astype(np.float32),
        "heuristic": heuristic,
        "candidate_mse": mse,
        "candidate_mae": mae,
        "strata_mse": strata_mse,
        "strata_counts": strata_counts,
        "candidate_ids": candidate_ids,
        "eval_rows": int(len(payload["truth"])),
        "deployable": deployable,
        "calibration_pairs": int(pairs),
        "calibration_rmse": float(calibration_rmse),
    }


def _select_train_candidates_v1(bank, well):
    mse = bank["candidate_mse"]
    rmse = np.sqrt(mse)
    heuristic = bank["heuristic"]
    deployable = np.flatnonzero(np.isfinite(mse))
    if len(deployable) < TRAIN_SAMPLE_CANDIDATES_V1:
        raise RuntimeError(f"only {len(deployable)} deployable candidates")
    selected = []
    seen = set()

    def add(values):
        for value in values:
            value = int(value)
            if value in seen or not np.isfinite(mse[value]):
                continue
            seen.add(value)
            selected.append(value)

    # Exact positives and near-positive shape alternatives.
    add(np.argsort(mse, kind="mergesort")[:24])
    # Candidates the old label-free score finds convincing, including its
    # dangerous wrong aliases.
    add(np.argsort(heuristic, kind="mergesort")[:40])
    hard = deployable[(rmse[deployable] >= 8.0) & (rmse[deployable] <= 40.0)]
    add(hard[np.argsort(heuristic[hard], kind="mergesort")[:36]])
    # Every base family contributes its own supervised best candidate.
    for base_index in range(len(BASE_MENU)):
        members = np.flatnonzero(
            (bank["candidate_ids"][:, 0] == base_index) & np.isfinite(mse)
        )
        if len(members):
            add([members[np.argmin(mse[members])]])

    rng = np.random.default_rng(SEED_V1 + int(str(well), 16) % 1000003)
    for low, high in ((0, 5), (5, 8), (8, 12), (12, 20), (20, 40), (40, np.inf)):
        members = deployable[(rmse[deployable] >= low) & (rmse[deployable] < high)]
        if len(members):
            add(rng.choice(members, size=min(12, len(members)), replace=False))
    add(rng.permutation(deployable))
    selected = np.asarray(selected[:TRAIN_SAMPLE_CANDIDATES_V1], np.int32)
    if len(selected) != TRAIN_SAMPLE_CANDIDATES_V1:
        raise RuntimeError(f"selected only {len(selected)} candidates")
    return selected


def _build_train_sample_v1(well):
    bank = _render_full_bank_v1(well)
    selected = _select_train_candidates_v1(bank, well)
    np.savez_compressed(
        _sample_path_v1(well),
        well=np.asarray(str(well)),
        query=bank["query"].astype(np.float16),
        reference=bank["reference"][selected].astype(np.float16),
        valid=bank["valid"][selected].astype(np.uint8),
        path_delta=bank["path_delta"][selected].astype(np.float16),
        heuristic=bank["heuristic"][selected].astype(np.float32),
        candidate_mse=bank["candidate_mse"][selected].astype(np.float32),
        candidate_ids=bank["candidate_ids"][selected].astype(np.int16),
        eval_rows=np.asarray([bank["eval_rows"]], np.int32),
        cache_version=np.asarray([1], np.int16),
    )
    return {
        "well": well,
        "status": "ok",
        "best_rmse": float(np.sqrt(np.min(bank["candidate_mse"][selected]))),
        "worst_rmse": float(np.sqrt(np.max(bank["candidate_mse"][selected]))),
        "positive_under5": int((np.sqrt(bank["candidate_mse"][selected]) <= 5).sum()),
    }


def _load_train_sample_v1(well):
    with np.load(_sample_path_v1(well), allow_pickle=False) as data:
        return {
            "well": str(data["well"].item()),
            "query": data["query"].astype(np.float32),
            "reference": data["reference"].astype(np.float32),
            "valid": data["valid"].astype(bool),
            "path_delta": data["path_delta"].astype(np.float32),
            "heuristic": data["heuristic"].astype(np.float32),
            "candidate_mse": data["candidate_mse"].astype(np.float32),
        }


# ---------------------------------------------------------------------------
# Candidate-independent complete-well energy
# ---------------------------------------------------------------------------


class ResidualBlockV1(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(
            channels, channels, 5, padding=2 * dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )

    def forward(self, values):
        residual = self.conv1(F.gelu(self.norm1(values)))
        residual = self.conv2(F.gelu(self.norm2(residual)))
        return values + residual


class FullBankUnaryEnergyV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(11, EMBEDDING_V1, 7, padding=3)
        self.blocks = nn.ModuleList(
            [ResidualBlockV1(EMBEDDING_V1, d) for d in (1, 2, 4, 8, 16, 32)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(EMBEDDING_V1 * 3 + 10),
            nn.Linear(EMBEDDING_V1 * 3 + 10, HIDDEN_V1),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(HIDDEN_V1, 1),
        )

    def forward(self, query, reference, valid, path_delta, heuristic):
        candidates, length = reference.shape
        q = query[None, :].expand(candidates, -1)
        difference = (q - reference) * valid
        absolute = difference.abs()
        q_derivative = torch.diff(q, dim=1, prepend=q[:, :1])
        r_derivative = torch.diff(reference, dim=1, prepend=reference[:, :1])
        path = torch.clamp(path_delta / 96.0, -8.0, 8.0)
        path_derivative = torch.diff(path, dim=1, prepend=path[:, :1]) * 16.0
        progress = torch.linspace(
            0.0, 1.0, length, device=query.device, dtype=query.dtype
        )[None, :].expand(candidates, -1)
        heuristic_sequence = heuristic[:, None].expand(-1, length)
        inputs = torch.stack(
            [
                q,
                reference,
                difference,
                absolute,
                q_derivative,
                r_derivative,
                path,
                path_derivative,
                valid,
                progress,
                heuristic_sequence,
            ],
            dim=1,
        )
        hidden = self.stem(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        mask = valid[:, None, :]
        count = mask.sum(dim=2).clamp_min(1.0)
        mean = (hidden * mask).sum(dim=2) / count
        variance = ((hidden - mean[:, :, None]) ** 2 * mask).sum(dim=2) / count
        maximum = hidden.masked_fill(mask <= 0, -1e4).amax(dim=2)
        valid_count = valid.sum(dim=1).clamp_min(1.0)
        early = progress <= 0.33
        late = progress >= 0.67
        summaries = torch.stack(
            [
                (absolute * valid).sum(dim=1) / valid_count,
                torch.sqrt((difference.square() * valid).sum(dim=1) / valid_count + 1e-8),
                (difference * valid).sum(dim=1) / valid_count,
                valid.mean(dim=1),
                (absolute * valid * early).sum(dim=1)
                / (valid * early).sum(dim=1).clamp_min(1.0),
                (absolute * valid * late).sum(dim=1)
                / (valid * late).sum(dim=1).clamp_min(1.0),
                path.abs().mean(dim=1),
                path_derivative.abs().mean(dim=1),
                heuristic,
                absolute.amax(dim=1),
            ],
            dim=1,
        )
        descriptor = torch.cat(
            [mean, torch.sqrt(variance + 1e-8), maximum, summaries], dim=1
        )
        scores = self.head(descriptor)[:, 0]
        available = valid.mean(dim=1) >= MIN_REFERENCE_COVERAGE
        return scores.masked_fill(~available, -1e4)


def _tensor_arrays_v1(query, reference, valid, path_delta, heuristic, device):
    return (
        torch.from_numpy(np.asarray(query, np.float32)).to(device),
        torch.from_numpy(np.asarray(reference, np.float32)).to(device),
        torch.from_numpy(np.asarray(valid, np.float32)).to(device),
        torch.from_numpy(np.asarray(path_delta, np.float32)).to(device),
        torch.from_numpy(np.asarray(heuristic, np.float32)).to(device),
    )


def _loss_v1(model, panel, device):
    permutation = np.random.permutation(len(panel["candidate_mse"]))
    tensors = _tensor_arrays_v1(
        panel["query"],
        panel["reference"][permutation],
        panel["valid"][permutation],
        panel["path_delta"][permutation],
        panel["heuristic"][permutation],
        device,
    )
    mse = torch.from_numpy(panel["candidate_mse"][permutation]).to(device)
    rmse = torch.sqrt(mse.clamp_min(0.0) + 1e-8)
    target = torch.argmin(mse)
    scores = model(*tensors)
    hard_ce = F.cross_entropy(scores[None], target[None])
    soft_target = torch.softmax(
        -(rmse - rmse.min()) / SOFT_TEMPERATURE_FT_V1, dim=0
    )
    soft_ce = -(soft_target * F.log_softmax(scores, dim=0)).sum()
    hard = (rmse >= rmse[target] + HARD_GAP_FT_V1) & (scores > -1e3)
    margin = (
        F.relu(LOGIT_MARGIN_V1 - scores[target] + scores[hard].amax())
        if hard.any()
        else scores[target] * 0.0
    )
    return 0.55 * hard_ce + 0.45 * soft_ce + 0.20 * margin, {
        "hard_ce": float(hard_ce.detach().cpu()),
        "soft_ce": float(soft_ce.detach().cpu()),
        "margin": float(margin.detach().cpu()),
    }


@torch.no_grad()
def _score_arrays_v1(
    model, query, reference, valid, path_delta, heuristic, device, control="full"
):
    model.eval()
    output = []
    for left in range(0, len(reference), INFERENCE_CHUNK_V1):
        right = min(left + INFERENCE_CHUNK_V1, len(reference))
        chunk_query = query.copy()
        chunk_reference = reference[left:right]
        chunk_valid = valid[left:right]
        chunk_heuristic = heuristic[left:right].copy()
        if control == "no_gr":
            chunk_query.fill(0.0)
            chunk_heuristic.fill(0.5)
        elif control == "shuffled_alignment":
            indices = (np.arange(left, right) + 37) % len(reference)
            chunk_reference = reference[indices]
            chunk_valid = valid[indices]
            chunk_heuristic = heuristic[indices]
        elif control != "full":
            raise ValueError(control)
        tensors = _tensor_arrays_v1(
            chunk_query,
            chunk_reference,
            chunk_valid,
            path_delta[left:right],
            chunk_heuristic,
            device,
        )
        output.append(model(*tensors).detach().cpu().numpy())
    return np.concatenate(output)


def _score_v0_panel_v1(model, panel, device, control="full"):
    return _score_arrays_v1(
        model,
        panel["query"],
        panel["reference"],
        panel["valid"],
        panel["path_delta"],
        panel["heuristic"],
        device,
        control,
    )


def _summarize_rows_v1(frame, prefix=""):
    rows = int(frame["rows"].sum())
    report = {
        "rows": rows,
        "wells": int(len(frame)),
        "exact_sse": float((frame["mse"] * frame["rows"]).sum()),
        "rmse": float(math.sqrt((frame["mse"] * frame["rows"]).sum() / rows)),
        "mae": float((frame["mae"] * frame["rows"]).sum() / rows),
        "median_well_rmse": float(np.sqrt(frame["mse"]).median()),
        "p75_well_rmse": float(np.sqrt(frame["mse"]).quantile(0.75)),
        "p95_well_rmse": float(np.sqrt(frame["mse"]).quantile(0.95)),
        "max_well_rmse": float(np.sqrt(frame["mse"]).max()),
    }
    for name in STRATA_NAMES:
        count_column = f"{name}_rows"
        mse_column = f"{name}_mse"
        count = int(frame[count_column].sum())
        report[f"{name}_rmse"] = (
            float(math.sqrt((frame[mse_column] * frame[count_column]).sum() / count))
            if count
            else np.nan
        )
    return report


def _row_for_index_v1(well, bank, index, mode):
    row = {
        "well": well,
        "mode": mode,
        "selected_index": int(index),
        "rows": int(bank["eval_rows"]),
        "mse": float(bank["candidate_mse"][index]),
        "mae": float(bank["candidate_mae"][index]),
    }
    for stratum, name in enumerate(STRATA_NAMES):
        row[f"{name}_rows"] = int(bank["strata_counts"][stratum])
        row[f"{name}_mse"] = float(bank["strata_mse"][index, stratum])
    return row


def _evaluate_pruned_v1(model, wells, device):
    rows = []
    for well in wells:
        panel = _load_panel(well)
        scores = _score_v0_panel_v1(model, panel, device)
        index = int(np.argmax(scores))
        row = {
            "well": well,
            "rows": panel["eval_rows"],
            "mse": float(panel["candidate_mse"][index]),
            "mae": float(panel["candidate_mae"][index]),
        }
        for stratum, name in enumerate(STRATA_NAMES):
            row[f"{name}_rows"] = int(panel["strata_counts"][stratum])
            row[f"{name}_mse"] = float(panel["strata_mse"][index, stratum])
        rows.append(row)
    frame = pd.DataFrame(rows)
    return _summarize_rows_v1(frame), frame


def _evaluate_full_bank_v1(model, wells, device):
    rows = {mode: [] for mode in ("model", "no_gr", "shuffled_alignment", "retrieved_oracle")}
    started = time.time()
    for number, well in enumerate(wells, start=1):
        bank = _render_full_bank_v1(well)
        scores = {}
        for control in ("full", "no_gr", "shuffled_alignment"):
            scores[control] = _score_arrays_v1(
                model,
                bank["query"],
                bank["reference"],
                bank["valid"],
                bank["path_delta"],
                bank["heuristic"],
                device,
                control,
            )
        full_index = int(np.argmax(scores["full"]))
        rows["model"].append(_row_for_index_v1(well, bank, full_index, "model"))
        for control in ("no_gr", "shuffled_alignment"):
            index = int(np.argmax(scores[control]))
            rows[control].append(_row_for_index_v1(well, bank, index, control))
        retrieved = np.argsort(-scores["full"], kind="mergesort")[:RETRIEVAL_K_V1]
        oracle_index = int(retrieved[np.argmin(bank["candidate_mse"][retrieved])])
        rows["retrieved_oracle"].append(
            _row_for_index_v1(well, bank, oracle_index, "retrieved_oracle")
        )
        del bank, scores
        gc.collect()
        if number % 10 == 0 or number == len(wells):
            print(
                f"FULLBANK fold2 {number}/{len(wells)} "
                f"elapsed={(time.time() - started) / 60.0:.1f}m",
                flush=True,
            )
    reports = {}
    frames = {}
    for mode, values in rows.items():
        frame = pd.DataFrame(values)
        frames[mode] = frame
        reports[mode] = _summarize_rows_v1(frame)
        frame.to_csv(OUT_DIR_V1 / f"fold2_{mode}_bywell.csv", index=False)
    return reports, frames


def _train_epoch_v1(model, wells, optimizer, scaler, device, epoch):
    model.train()
    order = list(wells)
    random.Random(SEED_V1 + epoch).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "hard_ce": 0.0, "soft_ce": 0.0, "margin": 0.0}
    steps = 0
    for index, well in enumerate(order, start=1):
        panel = _load_train_sample_v1(well)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            loss, parts = _loss_v1(model, panel, device)
            scaled_loss = loss / GRAD_ACCUM_WELLS_V1
        scaler.scale(scaled_loss).backward()
        if index % GRAD_ACCUM_WELLS_V1 == 0 or index == len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_V1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        totals["loss"] += float(loss.detach().cpu())
        for key, value in parts.items():
            totals[key] += value
        if index % 100 == 0 or index == len(order):
            print(
                f"UNARY epoch={epoch} wells={index}/{len(order)} "
                f"loss={totals['loss'] / index:.4f}",
                flush=True,
            )
    output = {key: value / len(order) for key, value in totals.items()}
    output["optimizer_steps"] = steps
    return output


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


_seed_everything(SEED_V1)
folds_v1 = _load_folds()
train_wells_v1 = sorted(
    folds_v1.loc[folds_v1["fold"].isin(TRAIN_FOLDS), "well"].astype(str).tolist()
)
valid_wells_v1 = sorted(
    folds_v1.loc[folds_v1["fold"].eq(VALID_FOLD), "well"].astype(str).tolist()
)
if len(train_wells_v1) < 450 or len(valid_wells_v1) < 150:
    raise RuntimeError(
        f"Unexpected folds {len(train_wells_v1)}/{len(valid_wells_v1)}"
    )
for well in valid_wells_v1:
    if not (PANEL_DIR / f"{well}.npz").exists():
        raise FileNotFoundError(
            PANEL_DIR / f"{well}.npz",
            "Run V0 through its ORACLE GATE first.",
        )

started_v1 = time.time()
build_rows_v1 = []
for number, well in enumerate(train_wells_v1, start=1):
    try:
        if not _sample_path_v1(well).exists():
            report = _build_train_sample_v1(well)
        else:
            panel = _load_train_sample_v1(well)
            report = {
                "well": well,
                "status": "cached",
                "best_rmse": float(np.sqrt(panel["candidate_mse"].min())),
                "worst_rmse": float(np.sqrt(panel["candidate_mse"].max())),
                "positive_under5": int((np.sqrt(panel["candidate_mse"]) <= 5).sum()),
            }
        build_rows_v1.append(report)
    except Exception as error:
        build_rows_v1.append({"well": well, "status": "failed", "reason": repr(error)})
    if number % 25 == 0 or number == len(train_wells_v1):
        print(
            f"build unary samples {number}/{len(train_wells_v1)} "
            f"elapsed={(time.time() - started_v1) / 60.0:.1f}m",
            flush=True,
        )
build_report_v1 = pd.DataFrame(build_rows_v1)
build_report_v1.to_csv(OUT_DIR_V1 / "train_sample_build_report.csv", index=False)
failed_v1 = build_report_v1[build_report_v1["status"].eq("failed")]
if len(failed_v1):
    raise RuntimeError(
        f"{len(failed_v1)} train samples failed: "
        + repr(failed_v1.head(10).to_dict(orient="records"))
    )

device_v1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_v1 = FullBankUnaryEnergyV1().to(device_v1)
parameters_v1 = int(sum(parameter.numel() for parameter in model_v1.parameters()))
optimizer_v1 = torch.optim.AdamW(
    model_v1.parameters(), lr=LEARNING_RATE_V1, weight_decay=WEIGHT_DECAY_V1
)
scheduler_v1 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_v1, T_max=MAX_EPOCHS_V1, eta_min=MIN_LEARNING_RATE_V1
)
scaler_v1 = torch.amp.GradScaler("cuda", enabled=device_v1.type == "cuda")
checkpoint_path_v1 = MODEL_DIR_V1 / "best_fold2_unary_retriever.pt"
history_v1 = []
best_v1 = None
bad_epochs_v1 = 0

for epoch in range(1, MAX_EPOCHS_V1 + 1):
    training = _train_epoch_v1(
        model_v1, train_wells_v1, optimizer_v1, scaler_v1, device_v1, epoch
    )
    validation, _ = _evaluate_pruned_v1(model_v1, valid_wells_v1, device_v1)
    report = {
        "epoch": epoch,
        **{f"train_{key}": value for key, value in training.items()},
        **{f"valid_pruned_{key}": value for key, value in validation.items()},
        "learning_rate": float(optimizer_v1.param_groups[0]["lr"]),
        "elapsed_minutes": (time.time() - started_v1) / 60.0,
    }
    history_v1.append(report)
    print("UNARY EPOCH", report, flush=True)
    if best_v1 is None or validation["rmse"] < best_v1["rmse"] - 1e-4:
        best_v1 = {"epoch": epoch, **validation}
        torch.save(
            {
                "model": model_v1.state_dict(),
                "epoch": epoch,
                "pruned_metrics": validation,
                "template_names": TEMPLATE_NAMES_V1,
                "base_menu": BASE_MENU,
            },
            checkpoint_path_v1,
        )
        bad_epochs_v1 = 0
    else:
        bad_epochs_v1 += 1
    scheduler_v1.step()
    if epoch >= MIN_EPOCHS_V1 and bad_epochs_v1 >= PATIENCE_V1:
        print("UNARY early stopping", flush=True)
        break

pd.DataFrame(history_v1).to_csv(OUT_DIR_V1 / "training_history.csv", index=False)
checkpoint_v1 = torch.load(checkpoint_path_v1, map_location=device_v1, weights_only=False)
model_v1.load_state_dict(checkpoint_v1["model"])

reports_v1, frames_v1 = _evaluate_full_bank_v1(
    model_v1, valid_wells_v1, device_v1
)
controls_v1 = {
    "no_gr_rmse_degradation": float(
        reports_v1["no_gr"]["rmse"] - reports_v1["model"]["rmse"]
    ),
    "shuffled_alignment_rmse_degradation": float(
        reports_v1["shuffled_alignment"]["rmse"]
        - reports_v1["model"]["rmse"]
    ),
}
verdict_v1 = {
    "experiment": EXPERIMENT_V1,
    "stage": "fold2_fullbank_unary_retrieval_gateway",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor": False,
    "uses_spatial_strong_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS),
        "validation_fold": VALID_FOLD,
        "fold1_touched": False,
        "selected_epoch": int(checkpoint_v1["epoch"]),
    },
    "parameters": parameters_v1,
    "bank": {
        "base_paths": len(BASE_MENU),
        "templates": len(TEMPLATES_V1),
        "full_candidates": int(len(BASE_MENU) * len(TEMPLATES_V1)),
        "train_sample_candidates": TRAIN_SAMPLE_CANDIDATES_V1,
        "retrieval_k": RETRIEVAL_K_V1,
    },
    "model": reports_v1["model"],
    "retrieved_top192_oracle": reports_v1["retrieved_oracle"],
    "no_gr": reports_v1["no_gr"],
    "shuffled_alignment": reports_v1["shuffled_alignment"],
    "controls": controls_v1,
    "gates": {
        "retrieved_top192_oracle_rmse_le": GATE_RETRIEVED_ORACLE_V1,
        "model_rmse_le": GATE_MODEL_RMSE_V1,
        "model_median_well_rmse_le": GATE_MODEL_MEDIAN_V1,
        "model_far_gt2000_rmse_le": GATE_MODEL_FAR_V1,
        "each_control_degradation_ge": GATE_CONTROL_DEGRADATION_V1,
        "target_5p5_rmse_le": 5.5,
    },
    "retrieval_pass": bool(
        reports_v1["retrieved_oracle"]["rmse"] <= GATE_RETRIEVED_ORACLE_V1
    ),
    "target_5p5_pass": bool(reports_v1["model"]["rmse"] <= 5.5),
    "strict_pass": bool(
        reports_v1["retrieved_oracle"]["rmse"] <= GATE_RETRIEVED_ORACLE_V1
        and reports_v1["model"]["rmse"] <= GATE_MODEL_RMSE_V1
        and reports_v1["model"]["median_well_rmse"] <= GATE_MODEL_MEDIAN_V1
        and reports_v1["model"]["far_gt2000_rmse"] <= GATE_MODEL_FAR_V1
        and controls_v1["no_gr_rmse_degradation"] >= GATE_CONTROL_DEGRADATION_V1
        and controls_v1["shuffled_alignment_rmse_degradation"]
        >= GATE_CONTROL_DEGRADATION_V1
    ),
    "checkpoint": str(checkpoint_path_v1),
    "elapsed_minutes": (time.time() - started_v1) / 60.0,
    "free_gb": float(__import__("shutil").disk_usage(OUT_DIR_V1).free / (1024**3)),
}
if verdict_v1["retrieval_pass"] and not verdict_v1["strict_pass"]:
    verdict_v1["next"] = (
        "RETRIEVAL PASSES -> train a set-context reranker only on learned top-192"
    )
elif verdict_v1["strict_pass"]:
    verdict_v1["next"] = "PASS -> untouched Fold1 confirmation"
else:
    verdict_v1["next"] = (
        "FAIL -> full-well unary energy cannot retain the useful modes"
    )
_write_json(OUT_DIR_V1 / "verdict.json", verdict_v1)
print("FULLBANK FINAL VERDICT", json.dumps(verdict_v1, indent=2), flush=True)
print("saved:", OUT_DIR_V1, flush=True)
