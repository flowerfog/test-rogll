"""Refit the frozen V144 coefficient model on all 773 wells and export it.

It can run immediately after V144 in the same kernel or recover independently
from V144's saved OOF artifacts after a kernel restart. Hyperparameters and
consumption are unchanged.
"""

from pathlib import Path
import hashlib
import json
import shutil
import sys
import types

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT_V144B = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OUT_V144B = ROOT_V144B / "final_consensus_sequence_coefficient_bundle_v144"
OUT_V144B.mkdir(parents=True, exist_ok=True)

required_v144b = (
    "Q", "sequence_raw", "metadata_raw", "target_raw", "FIXED_EPOCHS", "SEED",
    "FROZEN_V126_RECIPE", "well_order",
)
missing_v144b = [name for name in required_v144b if name not in globals()]
if missing_v144b:
    print(
        "V144 BUNDLE RECOVERY",
        {"missing_live_globals": missing_v144b, "source": "saved V144 OOF artifacts"},
        flush=True,
    )
    source_v144b = ROOT_V144B / "final_consensus_sequence_coefficient_v126_cell.py"
    oof_dir_v144b = (
        ROOT_V144B / "oof_exports" / "final_consensus_sequence_coefficient_full5_v144"
    )
    oof_path_v144b = oof_dir_v144b / "crossfit_oof.parquet"
    coefficient_path_v144b = oof_dir_v144b / "crossfit_coefficients.csv"
    for path_v144b in (source_v144b, oof_path_v144b, coefficient_path_v144b):
        if not path_v144b.is_file():
            raise FileNotFoundError(path_v144b)

    source_text_v144b = source_v144b.read_text(encoding="utf-8")
    marker_v144b = "for required in (V110_SOURCE, BASE_PATH, TARGET_PATH):"
    if marker_v144b not in source_text_v144b:
        raise RuntimeError(f"V126 definition marker missing from {source_v144b}")
    module_v144b = types.ModuleType("v144_bundle_recovered_v126_definitions")
    module_v144b.__file__ = str(source_v144b)
    sys.modules[module_v144b.__name__] = module_v144b
    exec(
        compile(
            source_text_v144b.split(marker_v144b, 1)[0] + "\n",
            str(source_v144b),
            "exec",
        ),
        module_v144b.__dict__,
    )
    Q = module_v144b
    R_v144b = Q._bootstrap_v110()
    N_v144b = R_v144b._bootstrap_v81()
    V_v144b = N_v144b.V

    frame_v144b = pd.read_parquet(oof_path_v144b)
    frame_v144b["well"] = frame_v144b["well"].astype(str)
    required_columns_v144b = {
        "well", "row_idx", Q.TRUTH, "md_since", Q.BASE, "exact_v76",
        Q.PARENT, Q.COARSE,
    }
    missing_columns_v144b = sorted(required_columns_v144b.difference(frame_v144b.columns))
    if missing_columns_v144b:
        raise RuntimeError(f"Saved V144 OOF missing columns: {missing_columns_v144b}")
    frame_v144b = frame_v144b.sort_values(
        ["well", "row_idx"], kind="stable"
    ).reset_index(drop=True)

    coefficient_v144b = pd.read_csv(coefficient_path_v144b, dtype={"well": str})
    true_columns_v144b = [f"true_{name}" for name in Q.TARGET_COLUMNS]
    missing_targets_v144b = sorted(
        {"well", *true_columns_v144b}.difference(coefficient_v144b.columns)
    )
    if missing_targets_v144b:
        raise RuntimeError(
            f"Saved V144 coefficients missing columns: {missing_targets_v144b}"
        )
    coefficient_v144b["well"] = coefficient_v144b["well"].astype(str)
    target_lookup_v144b = coefficient_v144b.set_index("well")[true_columns_v144b]

    well_order = sorted(frame_v144b["well"].unique())
    sequence_parts_v144b = []
    metadata_parts_v144b = []
    target_parts_v144b = []
    for number_v144b, well_v144b in enumerate(well_order, 1):
        local_v144b = frame_v144b[frame_v144b["well"].eq(well_v144b)]
        sequence_v144b, metadata_v144b = Q._sequence_features(
            local_v144b, Q._load_record_relaxed(V_v144b, well_v144b)
        )
        sequence_parts_v144b.append(sequence_v144b)
        metadata_parts_v144b.append(metadata_v144b)
        if well_v144b not in target_lookup_v144b.index:
            raise RuntimeError(f"Missing saved coefficient target for {well_v144b}")
        target_parts_v144b.append(
            target_lookup_v144b.loc[well_v144b].to_numpy(np.float32)
        )
        if number_v144b % 100 == 0 or number_v144b == len(well_order):
            print(
                f"V144 bundle recovery features {number_v144b}/{len(well_order)}",
                flush=True,
            )

    sequence_raw = np.stack(sequence_parts_v144b).astype(np.float32)
    metadata_raw = np.stack(metadata_parts_v144b).astype(np.float32)
    target_raw = np.stack(target_parts_v144b).astype(np.float32)
    FIXED_EPOCHS = 3
    SEED = 20260801
    FROZEN_V126_RECIPE = ("tail_affine", 1.0, 4.0)
if len(well_order) != 773:
    raise RuntimeError(f"Expected 773 training wells, found {len(well_order)}")
if tuple(FROZEN_V126_RECIPE) != ("tail_affine", 1.0, 4.0):
    raise RuntimeError(f"Unexpected V144 recipe: {FROZEN_V126_RECIPE}")

sequence_mean_v144b = np.mean(sequence_raw, axis=(0, 2), keepdims=True)
sequence_std_v144b = np.maximum(
    np.std(sequence_raw, axis=(0, 2), keepdims=True), 1e-3
)
metadata_mean_v144b = np.mean(metadata_raw, axis=0, keepdims=True)
metadata_std_v144b = np.maximum(np.std(metadata_raw, axis=0, keepdims=True), 1e-3)
target_mean_v144b = np.mean(target_raw, axis=0, keepdims=True)
target_std_v144b = np.maximum(np.std(target_raw, axis=0, keepdims=True), 0.25)
target_low_v144b = np.quantile(target_raw, 0.01, axis=0)
target_high_v144b = np.quantile(target_raw, 0.99, axis=0)

x_sequence_v144b = (
    (sequence_raw - sequence_mean_v144b) / sequence_std_v144b
).astype(np.float32)
x_metadata_v144b = (
    (metadata_raw - metadata_mean_v144b) / metadata_std_v144b
).astype(np.float32)
y_v144b = np.clip(
    (target_raw - target_mean_v144b) / target_std_v144b, -6.0, 6.0
).astype(np.float32)

Q._seed_everything(int(SEED))
device_v144b = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_v144b = Q.SequenceCoefficientEncoder(
    x_sequence_v144b.shape[1], x_metadata_v144b.shape[1]
).to(device_v144b)
optimizer_v144b = torch.optim.AdamW(
    model_v144b.parameters(), lr=Q.LEARNING_RATE, weight_decay=Q.WEIGHT_DECAY
)
criterion_v144b = nn.SmoothL1Loss(beta=0.75)
generator_v144b = torch.Generator().manual_seed(int(SEED))
loader_v144b = DataLoader(
    TensorDataset(
        torch.from_numpy(x_sequence_v144b),
        torch.from_numpy(x_metadata_v144b),
        torch.from_numpy(y_v144b),
    ),
    batch_size=Q.BATCH_SIZE,
    shuffle=True,
    generator=generator_v144b,
)
history_v144b = []
for epoch_v144b in range(1, int(FIXED_EPOCHS) + 1):
    model_v144b.train()
    losses_v144b = []
    for sequence_v144b, metadata_v144b, target_v144b in loader_v144b:
        optimizer_v144b.zero_grad(set_to_none=True)
        prediction_v144b = model_v144b(
            sequence_v144b.to(device_v144b), metadata_v144b.to(device_v144b)
        )
        loss_v144b = criterion_v144b(prediction_v144b, target_v144b.to(device_v144b))
        loss_v144b.backward()
        torch.nn.utils.clip_grad_norm_(model_v144b.parameters(), 2.0)
        optimizer_v144b.step()
        losses_v144b.append(float(loss_v144b.detach().cpu()))
    row_v144b = {"epoch": epoch_v144b, "loss": float(np.mean(losses_v144b))}
    history_v144b.append(row_v144b)
    print("V144 BUNDLE EPOCH", row_v144b, flush=True)

checkpoint_v144b = OUT_V144B / "v144_all773.pt"
torch.save(
    {
        "state_dict": {
            key: value.detach().cpu() for key, value in model_v144b.state_dict().items()
        },
        "sequence_mean": sequence_mean_v144b.astype(np.float32),
        "sequence_std": sequence_std_v144b.astype(np.float32),
        "metadata_mean": metadata_mean_v144b.astype(np.float32),
        "metadata_std": metadata_std_v144b.astype(np.float32),
        "target_mean": target_mean_v144b.astype(np.float32),
        "target_std": target_std_v144b.astype(np.float32),
        "target_low": target_low_v144b.astype(np.float32),
        "target_high": target_high_v144b.astype(np.float32),
        "target_columns": tuple(Q.TARGET_COLUMNS),
        "sequence_channels": int(sequence_raw.shape[1]),
        "metadata_features": int(metadata_raw.shape[1]),
        "nodes": int(sequence_raw.shape[2]),
        "epochs": int(FIXED_EPOCHS),
        "seed": int(SEED),
    },
    checkpoint_v144b,
)


def _sha256_v144b(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


manifest_v144b = {
    "experiment": "final_consensus_sequence_coefficient_bundle_v144",
    "training": "all 773 wells",
    "source_validation": "final_consensus_sequence_coefficient_full5_v144",
    "checkpoint": checkpoint_v144b.name,
    "checkpoint_sha256": _sha256_v144b(checkpoint_v144b),
    "recipe": {"profile": "tail_affine", "dose": 1.0, "cap_ft": 4.0},
    "freeze_ft": float(Q.FREEZE_FT),
    "ramp_ft": float(Q.RAMP_FT),
    "nodes": int(sequence_raw.shape[2]),
    "sequence_channels": int(sequence_raw.shape[1]),
    "metadata_features": int(metadata_raw.shape[1]),
    "epochs": int(FIXED_EPOCHS),
    "seed": int(SEED),
    "training_wells": len(well_order),
    "history": history_v144b,
}
(OUT_V144B / "manifest.json").write_text(
    json.dumps(manifest_v144b, indent=2), encoding="utf-8"
)
archive_v144b = shutil.make_archive(
    str(ROOT_V144B / "final_consensus_sequence_coefficient_bundle_v144"),
    "zip",
    ROOT_V144B,
    OUT_V144B.name,
)
print("V144 DEPLOYMENT BUNDLE")
print(json.dumps(manifest_v144b, indent=2), flush=True)
print("UPLOAD:", archive_v144b, flush=True)
