"""S4: calibrated synthetic pretraining for the complete-panel axial RAFT model.

Run immediately after ``tvt_corrnet_axial_raft_onefold_v0_cell.py`` in the
same AutoDL kernel.  This is a fixed-budget A/B experiment:

* ``scratch`` and ``synthetic_pretrained`` share the exact initial weights,
  real-panel order, optimiser, schedule, and number of real fine-tuning epochs.
* Synthetic panels are generated online.  Nothing large is written to disk.
* The horizontal toe GR is rendered from the target typewell at a real
  correction path and receives an orthogonal residual block transferred from
  another outer-train panel.  The visible prefix remains untouched.
* Half of the synthetic examples preserve the target path; half transplant a
  real correction path from another outer-train state.  No validation well,
  neighbour well, strong/CP12/Gold prediction, or LB feedback is used.

The frozen residual recipe is the one that passed the earlier nested domain
audit: orthogonal residual transfer with scale 1.0.
"""

from pathlib import Path
import copy
import gc
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


S4_REQUIRED_GLOBALS = (
    "OOF_ROOT_RAFT",
    "OUT_DIR_RAFT",
    "STATE_OFFSETS_RAFT",
    "PREFIX_COLUMNS_RAFT",
    "BATCH_SIZE_RAFT",
    "LEARNING_RATE_RAFT",
    "MIN_LEARNING_RATE_RAFT",
    "WEIGHT_DECAY_RAFT",
    "device_raft",
    "train_store_raft",
    "valid_store_raft",
    "AxialRAFTTVT",
    "RAFTPanelDataset",
    "_model_forward",
    "_trajectory_loss",
    "_evaluate_raft",
    "_seed_raft",
)
s4_missing = [name for name in S4_REQUIRED_GLOBALS if name not in globals()]
if s4_missing:
    raise RuntimeError(
        "Run tvt_corrnet_axial_raft_onefold_v0_cell.py first; missing: "
        + repr(s4_missing)
    )


S4_OUT_DIR = Path(OOF_ROOT_RAFT) / "tvt_corrnet_synthetic_pretrain_gateway_v0"
S4_MODEL_DIR = S4_OUT_DIR / "models"
for s4_directory in (S4_OUT_DIR, S4_MODEL_DIR):
    s4_directory.mkdir(parents=True, exist_ok=True)

S4_SEED = 20260721
S4_PRETRAIN_SEED = 20260723
S4_PRETRAIN_EPOCHS = 4
S4_SYNTHETIC_MULTIPLIER = 2
S4_REAL_EPOCHS = 8
S4_PATH_TRANSPLANT_PROBABILITY = 0.50
S4_RESIDUAL_SCALE = 1.0
S4_MIN_REFERENCE_COVERAGE = 0.85
S4_BATCH_SIZE = int(BATCH_SIZE_RAFT)
S4_SYNTHETIC_GAIN_GATE = 0.30
S4_LINEAGE_TOLERANCE = 0.30


def _s4_cpu_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _s4_amp_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=(device_raft.type == "cuda"))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=(device_raft.type == "cuda"))


def _s4_sample_surface(surface, validity, correction):
    """Linearly sample a [state, column] panel at one correction per column."""
    surface = np.asarray(surface, np.float32)
    validity = np.asarray(validity, np.float32)
    correction = np.asarray(correction, np.float32)
    states = np.asarray(STATE_OFFSETS_RAFT, np.float32)
    step = float(np.median(np.diff(states)))
    position = (correction - float(states[0])) / step
    inside = (position >= 0.0) & (position <= len(states) - 1)
    position = np.clip(position, 0.0, len(states) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(states) - 1)
    fraction = position - lower
    columns = np.arange(surface.shape[1], dtype=np.int64)
    sampled = (
        (1.0 - fraction) * surface[lower, columns]
        + fraction * surface[upper, columns]
    )
    valid = (
        inside
        & (validity[lower, columns] > 0.5)
        & (validity[upper, columns] > 0.5)
    )
    sampled = sampled.astype(np.float32)
    sampled[~valid] = 0.0
    return sampled, valid.astype(np.float32)


def _s4_fill(values, valid):
    values = np.asarray(values, np.float32).copy()
    valid = np.asarray(valid, bool)
    positions = np.arange(len(values), dtype=np.float32)
    if int(valid.sum()) >= 2:
        values = np.interp(positions, positions[valid], values[valid]).astype(np.float32)
    elif int(valid.sum()) == 1:
        values[:] = float(values[valid][0])
    else:
        values[:] = 0.0
    return values


class S4ResidualBank:
    """Cache the calibrated donor residual decomposition for real train panels."""

    def __init__(self, store):
        self.rows = []
        for record in store.records:
            truth = np.asarray(record["truth_residual"], np.float32)
            reference, reference_valid = _s4_sample_surface(
                record["typewell_gr"], record["typewell_valid"], truth
            )
            horizontal = np.asarray(record["horizontal_gr"], np.float32)
            horizontal_valid = np.asarray(record["horizontal_valid"], np.float32) > 0.5
            valid = horizontal_valid & (reference_valid > 0.5)
            residual = horizontal - reference
            filled = _s4_fill(residual, valid)
            if int(valid.sum()) >= 12 and float(np.var(reference[valid])) > 1e-6:
                x = reference[valid].astype(np.float64)
                y = filled[valid].astype(np.float64)
                denominator = float(np.dot(x - x.mean(), x - x.mean()))
                beta = float(np.dot(x - x.mean(), y - y.mean()) / denominator)
                beta = float(np.clip(beta, -1.5, 1.5))
                intercept = float(np.mean(y) - beta * np.mean(x))
            else:
                beta = 0.0
                intercept = float(np.mean(filled[valid])) if np.any(valid) else 0.0
            orthogonal = filled - (intercept + beta * reference)
            self.rows.append(
                {
                    "truth": truth,
                    "reference": reference,
                    "reference_valid": reference_valid,
                    "horizontal_valid": horizontal_valid.astype(np.float32),
                    "orthogonal": orthogonal.astype(np.float32),
                    "beta": beta,
                    "intercept": intercept,
                }
            )


class S4SyntheticPanelDataset(Dataset):
    """Deterministic online full-panel generator using outer-train records only."""

    def __init__(self, store, residual_bank, multiplier, seed):
        self.store = store
        self.bank = residual_bank
        self.multiplier = int(multiplier)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.store.records) * self.multiplier

    def _rng(self, index):
        return np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + int(index) * 104_729
        )

    def _render(self, index):
        rng = self._rng(index)
        target_index = int(index) % len(self.store.records)
        target = self.store.records[target_index]
        donor_index = int(rng.integers(0, len(self.store.records) - 1))
        if donor_index >= target_index:
            donor_index += 1
        donor = self.bank.rows[donor_index]

        target_truth = np.asarray(target["truth_residual"], np.float32)
        correction = target_truth.copy()
        transplanted = bool(rng.random() < S4_PATH_TRANSPLANT_PROBABILITY)
        if transplanted:
            path_index = int(rng.integers(0, len(self.store.records) - 1))
            if path_index >= target_index:
                path_index += 1
            candidate = np.asarray(
                self.store.records[path_index]["truth_residual"], np.float32
            ).copy()
            eval_columns = np.flatnonzero(
                np.asarray(target["eval_mask"], np.float32) > 0.5
            )
            if len(eval_columns):
                first = int(eval_columns[0])
                candidate += float(target_truth[first] - candidate[first])
            candidate[:PREFIX_COLUMNS_RAFT] = 0.0
            candidate = np.clip(
                candidate,
                float(STATE_OFFSETS_RAFT[1]),
                float(STATE_OFFSETS_RAFT[-2]),
            ).astype(np.float32)
            _, candidate_valid = _s4_sample_surface(
                target["typewell_gr"], target["typewell_valid"], candidate
            )
            eval_mask = np.asarray(target["eval_mask"], np.float32) > 0.5
            coverage = float(candidate_valid[eval_mask].mean())
            if coverage >= S4_MIN_REFERENCE_COVERAGE:
                correction = candidate
            else:
                transplanted = False

        reference, reference_valid = _s4_sample_surface(
            target["typewell_gr"], target["typewell_valid"], correction
        )
        transferred = (
            float(donor["intercept"])
            + float(donor["beta"]) * reference
            + np.asarray(donor["orthogonal"], np.float32)
        )
        synthetic = reference + S4_RESIDUAL_SCALE * transferred
        synthetic = np.clip(synthetic, -8.0, 8.0).astype(np.float32)
        synthetic_valid = (
            (reference_valid > 0.5)
            & (np.asarray(donor["horizontal_valid"], np.float32) > 0.5)
        ).astype(np.float32)
        synthetic[synthetic_valid < 0.5] = 0.0

        # Deployment exposes the real heel.  Keep it and synthesize only the toe.
        horizontal = synthetic
        horizontal_valid = synthetic_valid
        horizontal[:PREFIX_COLUMNS_RAFT] = np.asarray(
            target["horizontal_gr"], np.float32
        )[:PREFIX_COLUMNS_RAFT]
        horizontal_valid[:PREFIX_COLUMNS_RAFT] = np.asarray(
            target["horizontal_valid"], np.float32
        )[:PREFIX_COLUMNS_RAFT]
        correction[:PREFIX_COLUMNS_RAFT] = 0.0
        eval_mask = np.asarray(target["eval_mask"], np.float32)

        metadata = {
            "transplanted": transplanted,
            "reference_coverage": float(
                np.sum((reference_valid > 0.5) * eval_mask)
                / max(float(eval_mask.sum()), 1.0)
            ),
            "horizontal_coverage": float(
                np.sum((horizontal_valid > 0.5) * eval_mask)
                / max(float(eval_mask.sum()), 1.0)
            ),
            "correction_p95": float(
                np.quantile(np.abs(correction[eval_mask > 0.5]), 0.95)
            ),
        }
        return target, horizontal, horizontal_valid, correction, metadata

    def audit(self, index):
        return self._render(index)[-1]

    def __getitem__(self, index):
        target, horizontal, horizontal_valid, correction, _ = self._render(index)
        return {
            "horizontal": torch.from_numpy(horizontal.copy()),
            "horizontal_valid": torch.from_numpy(horizontal_valid.copy()),
            "typewell": torch.from_numpy(
                np.asarray(target["typewell_gr"], np.float32).copy()
            ),
            "typewell_valid": torch.from_numpy(
                np.asarray(target["typewell_valid"], np.float32).copy()
            ),
            "heel": torch.from_numpy(
                np.asarray(target["heel_reference_gr"], np.float32).copy()
            ),
            "heel_valid": torch.from_numpy(
                np.asarray(target["heel_reference_valid"], np.float32).copy()
            ),
            "geometry": torch.from_numpy(
                np.asarray(target["geometry"], np.float32).copy()
            ),
            "truth_residual": torch.from_numpy(correction.copy()),
            "eval_mask": torch.from_numpy(
                np.asarray(target["eval_mask"], np.float32).copy()
            ),
            "column_md": torch.from_numpy(
                np.asarray(target["column_md"], np.float32).copy()
            ),
            "column_z": torch.from_numpy(
                np.asarray(target["column_z"], np.float32).copy()
            ),
            "panel_weight": torch.tensor(
                float(target["panel_weight"]), dtype=torch.float32
            ),
        }


def _s4_loader(dataset, seed, shuffle=True):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=S4_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device_raft.type == "cuda"),
        generator=generator,
    )


def _s4_train_epoch(model, loader, optimizer, scaler):
    model.train()
    rows = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device_raft.type,
            dtype=torch.float16,
            enabled=(device_raft.type == "cuda"),
        ):
            predictions = _model_forward(model, batch)
        loss, components = _trajectory_loss(
            [prediction.float() for prediction in predictions],
            batch["truth_residual"].to(device_raft, non_blocking=True),
            batch["eval_mask"].to(device_raft, non_blocking=True),
            batch["column_md"].to(device_raft, non_blocking=True),
            batch["column_z"].to(device_raft, non_blocking=True),
            batch["panel_weight"].to(device_raft, non_blocking=True),
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        rows.append(
            {
                "loss": float(loss.detach().cpu()),
                **{
                    name: float(value.detach().cpu())
                    for name, value in components.items()
                },
            }
        )
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in rows[0]
    }


def _s4_pretrain(initial_state, geometry_channels, synthetic_dataset):
    _seed_raft(S4_PRETRAIN_SEED)
    model = AxialRAFTTVT(geometry_channels).to(device_raft)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_RAFT,
        weight_decay=WEIGHT_DECAY_RAFT,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=S4_PRETRAIN_EPOCHS,
        eta_min=MIN_LEARNING_RATE_RAFT,
    )
    scaler = _s4_amp_scaler()
    history = []
    started = time.time()
    for epoch in range(1, S4_PRETRAIN_EPOCHS + 1):
        synthetic_dataset.set_epoch(epoch)
        loader = _s4_loader(
            synthetic_dataset,
            S4_PRETRAIN_SEED + epoch * 10_007,
            shuffle=True,
        )
        train_metrics = _s4_train_epoch(model, loader, optimizer, scaler)
        scheduler.step()
        row = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("S4 SYNTHETIC PRETRAIN", row, flush=True)
    pd.DataFrame(history).to_csv(S4_OUT_DIR / "pretrain_history.csv", index=False)
    state = _s4_cpu_state(model)
    del model, optimizer, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state


def _s4_finetune(arm, start_state, geometry_channels):
    # Reset all stochastic state so both arms see the same panel order/dropout stream.
    _seed_raft(S4_SEED)
    model = AxialRAFTTVT(geometry_channels).to(device_raft)
    model.load_state_dict(start_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE_RAFT,
        weight_decay=WEIGHT_DECAY_RAFT,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=S4_REAL_EPOCHS,
        eta_min=MIN_LEARNING_RATE_RAFT,
    )
    scaler = _s4_amp_scaler()
    dataset = RAFTPanelDataset(train_store_raft, mode="full")
    loader = _s4_loader(dataset, S4_SEED + 91, shuffle=True)
    best_rmse = np.inf
    best_state = None
    best_epoch = 0
    history = []
    started = time.time()
    for epoch in range(1, S4_REAL_EPOCHS + 1):
        train_metrics = _s4_train_epoch(model, loader, optimizer, scaler)
        scheduler.step()
        valid_metrics, _, _ = _evaluate_raft(
            model, valid_store_raft, mode="full", return_rows=False
        )
        row = {
            "arm": arm,
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"valid_{name}": value for name, value in valid_metrics.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("S4 REAL FINETUNE", row, flush=True)
        if float(valid_metrics["rmse"]) < best_rmse - 1e-4:
            best_rmse = float(valid_metrics["rmse"])
            best_epoch = epoch
            best_state = _s4_cpu_state(model)

    if best_state is None:
        raise RuntimeError(f"{arm}: no finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    metrics, by_well, _ = _evaluate_raft(
        model, valid_store_raft, mode="full", return_rows=False
    )
    checkpoint_path = S4_MODEL_DIR / f"{arm}.pt"
    torch.save(
        {
            "model": best_state,
            "arm": arm,
            "epoch": best_epoch,
            "metrics": metrics,
            "geometry_channels": geometry_channels,
            "seed": S4_SEED,
            "real_epochs": S4_REAL_EPOCHS,
        },
        checkpoint_path,
    )
    by_well.to_csv(S4_OUT_DIR / f"{arm}_bywell.csv", index=False)
    pd.DataFrame(history).to_csv(
        S4_OUT_DIR / f"{arm}_finetune_history.csv", index=False
    )
    result = {
        "arm": arm,
        "best_epoch": int(best_epoch),
        **metrics,
        "runtime_minutes": (time.time() - started) / 60.0,
        "checkpoint": str(checkpoint_path),
    }
    return model, result


print(
    {
        "experiment": "tvt_corrnet_synthetic_pretrain_gateway_v0",
        "device": str(device_raft),
        "train_panels": len(train_store_raft.records),
        "valid_wells": len(valid_store_raft.records),
        "synthetic_examples_per_epoch": (
            len(train_store_raft.records) * S4_SYNTHETIC_MULTIPLIER
        ),
        "pretrain_epochs": S4_PRETRAIN_EPOCHS,
        "real_epochs_per_arm": S4_REAL_EPOCHS,
        "generator": {
            "mode": "orthogonal_residual_transfer",
            "scale": S4_RESIDUAL_SCALE,
            "path_transplant_probability": S4_PATH_TRANSPLANT_PROBABILITY,
        },
    },
    flush=True,
)

s4_started = time.time()
s4_bank = S4ResidualBank(train_store_raft)
s4_synthetic = S4SyntheticPanelDataset(
    train_store_raft,
    s4_bank,
    multiplier=S4_SYNTHETIC_MULTIPLIER,
    seed=S4_PRETRAIN_SEED,
)
s4_audit_count = min(512, len(s4_synthetic))
s4_audit = pd.DataFrame([s4_synthetic.audit(i) for i in range(s4_audit_count)])
s4_audit.to_csv(S4_OUT_DIR / "synthetic_guard_samples.csv", index=False)
s4_guards = {
    "samples": int(len(s4_audit)),
    "transplanted_fraction": float(s4_audit["transplanted"].mean()),
    "mean_reference_coverage": float(s4_audit["reference_coverage"].mean()),
    "p05_reference_coverage": float(s4_audit["reference_coverage"].quantile(0.05)),
    "mean_horizontal_coverage": float(s4_audit["horizontal_coverage"].mean()),
    "median_correction_p95": float(s4_audit["correction_p95"].median()),
    "p95_correction_p95": float(s4_audit["correction_p95"].quantile(0.95)),
}
print("SYNTHETIC GUARDS", s4_guards, flush=True)
if (
    s4_guards["mean_reference_coverage"] < 0.90
    or s4_guards["p05_reference_coverage"] < 0.75
    or s4_guards["mean_horizontal_coverage"] < 0.85
):
    raise RuntimeError(f"synthetic full-panel guard failed: {s4_guards}")

s4_geometry_channels = int(train_store_raft.records[0]["geometry"].shape[0])
_seed_raft(S4_SEED)
s4_initial_model = AxialRAFTTVT(s4_geometry_channels).to(device_raft)
s4_initial_state = _s4_cpu_state(s4_initial_model)
del s4_initial_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

s4_pretrained_state = _s4_pretrain(
    s4_initial_state, s4_geometry_channels, s4_synthetic
)

s4_models = {}
s4_results = []
for s4_arm, s4_state in (
    ("scratch", s4_initial_state),
    ("synthetic_pretrained", s4_pretrained_state),
):
    s4_model, s4_result = _s4_finetune(
        s4_arm, s4_state, s4_geometry_channels
    )
    s4_models[s4_arm] = s4_model
    s4_results.append(s4_result)

s4_scores = pd.DataFrame(s4_results).sort_values("rmse").reset_index(drop=True)
s4_scores.to_csv(S4_OUT_DIR / "arm_scores.csv", index=False)
s4_scratch_rmse = float(
    s4_scores.loc[s4_scores["arm"] == "scratch", "rmse"].iloc[0]
)
s4_pretrained_rmse = float(
    s4_scores.loc[s4_scores["arm"] == "synthetic_pretrained", "rmse"].iloc[0]
)
s4_gain = s4_scratch_rmse - s4_pretrained_rmse
s4_winner = str(s4_scores.iloc[0]["arm"])
s4_winner_model = s4_models[s4_winner]

s4_intervention_rows = []
for s4_mode in ("full", "no_gr", "shuffled_typewell"):
    s4_metrics, _, s4_rows = _evaluate_raft(
        s4_winner_model,
        valid_store_raft,
        mode=s4_mode,
        return_rows=(s4_mode == "full"),
    )
    s4_intervention_rows.append({"mode": s4_mode, **s4_metrics})
    if s4_mode == "full":
        s4_rows.to_pickle(S4_OUT_DIR / "winner_fold1_predictions.pkl")
s4_interventions = pd.DataFrame(s4_intervention_rows)
s4_full_rmse = float(
    s4_interventions.loc[s4_interventions["mode"] == "full", "rmse"].iloc[0]
)
s4_interventions["degradation_vs_full"] = s4_interventions["rmse"] - s4_full_rmse
s4_interventions.to_csv(S4_OUT_DIR / "winner_interventions.csv", index=False)
s4_no_gr_degradation = float(
    s4_interventions.loc[
        s4_interventions["mode"] == "no_gr", "degradation_vs_full"
    ].iloc[0]
)
s4_shuffle_degradation = float(
    s4_interventions.loc[
        s4_interventions["mode"] == "shuffled_typewell", "degradation_vs_full"
    ].iloc[0]
)

s4_historical_rmse = 9.68718285607606
s4_prior_verdict_path = Path(OUT_DIR_RAFT) / "verdict.json"
if s4_prior_verdict_path.exists():
    try:
        with open(s4_prior_verdict_path, "r", encoding="utf-8") as s4_handle:
            s4_prior_verdict = json.load(s4_handle)
        s4_historical_rmse = float(s4_prior_verdict["result"]["rmse"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass

s4_lineage_pass = bool(
    abs(s4_scratch_rmse - s4_historical_rmse) <= S4_LINEAGE_TOLERANCE
)
s4_pretraining_pass = bool(s4_gain >= S4_SYNTHETIC_GAIN_GATE)
s4_architecture_pass = bool(s4_full_rmse <= 7.0)
s4_target_pass = bool(s4_full_rmse <= 5.5)
s4_causal_pass = bool(
    s4_no_gr_degradation >= 0.30 and s4_shuffle_degradation >= 0.30
)
s4_strict_pass = bool(
    s4_lineage_pass
    and s4_pretraining_pass
    and s4_architecture_pass
    and s4_causal_pass
)
if s4_architecture_pass and s4_causal_pass:
    s4_next = "PASS -> run a second seed, then expand S4 to five folds"
elif s4_pretraining_pass:
    s4_next = (
        "SYNTHETIC SIGNAL ONLY -> retain pretraining recipe for future complete-well "
        "architectures; current S3 backbone still fails; proceed to S5"
    )
else:
    s4_next = (
        "FAIL -> calibrated synthetic pretraining does not rescue the S3 backbone; "
        "proceed to S5 whole-path listwise scoring"
    )

s4_verdict = {
    "experiment": "tvt_corrnet_synthetic_pretrain_gateway_v0",
    "sample_unit": "complete panel/well state",
    "standalone": True,
    "generator": {
        "mode": "orthogonal_residual_transfer",
        "residual_scale": S4_RESIDUAL_SCALE,
        "path_transplant_probability": S4_PATH_TRANSPLANT_PROBABILITY,
        "pretrain_epochs": S4_PRETRAIN_EPOCHS,
        "synthetic_multiplier": S4_SYNTHETIC_MULTIPLIER,
    },
    "real_training_budget_per_arm": S4_REAL_EPOCHS,
    "guards": s4_guards,
    "historical_s3_rmse": s4_historical_rmse,
    "scratch_rmse": s4_scratch_rmse,
    "synthetic_pretrained_rmse": s4_pretrained_rmse,
    "synthetic_gain": s4_gain,
    "winner": s4_winner,
    "winner_rmse": s4_full_rmse,
    "no_gr_degradation": s4_no_gr_degradation,
    "shuffled_typewell_degradation": s4_shuffle_degradation,
    "lineage_pass": s4_lineage_pass,
    "pretraining_pass": s4_pretraining_pass,
    "architecture_pass": s4_architecture_pass,
    "target_5p5_pass": s4_target_pass,
    "causal_pass": s4_causal_pass,
    "strict_pass": s4_strict_pass,
    "next": s4_next,
    "elapsed_minutes": (time.time() - s4_started) / 60.0,
}
with open(S4_OUT_DIR / "verdict.json", "w", encoding="utf-8") as s4_handle:
    json.dump(s4_verdict, s4_handle, indent=2)

print("\nARM SCORES")
print(s4_scores.to_string(index=False))
print("\nWINNER INTERVENTIONS")
print(s4_interventions.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(s4_verdict, indent=2))
print("saved:", S4_OUT_DIR)

for s4_name, s4_model in list(s4_models.items()):
    if s4_model is not s4_winner_model:
        del s4_models[s4_name]
del s4_bank, s4_synthetic, s4_pretrained_state, s4_initial_state
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
