"""SEGN-R3 query-causal absolute matching gateway.

R2's apparent anchor recovery survived no-GR and therefore came from a
reference-only shortcut.  R3 removes that shortcut by construction:

* the encoder receives GR value and derivatives only, never the validity bit;
* every convolution/projection has no bias;
* normalization layers have no affine offset;
* the output consists only of bilinear/cosine query-reference interactions.

Consequently an all-zero horizontal GR query produces a uniform posterior for
every typewell.  Training/evaluation otherwise reuse R2's identical nested
anchor-jitter protocol.  Fold 1 and Fold 2 are never loaded.
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SOURCE_CANDIDATES_R3C = (
    Path("/root/autodl-tmp/segn_r2_absolute_matching_jitter_gateway_v0_cell.py"),
    Path.cwd() / "segn_r2_absolute_matching_jitter_gateway_v0_cell.py",
)
SOURCE_PATH_R3C = next(
    (path for path in SOURCE_CANDIDATES_R3C if path.exists()), None
)
if SOURCE_PATH_R3C is None:
    raise FileNotFoundError(
        "Upload segn_r2_absolute_matching_jitter_gateway_v0_cell.py to "
        "/root/autodl-tmp before running SEGN-R3."
    )

source_r3c = SOURCE_PATH_R3C.read_text(encoding="utf-8")
marker_r3c = "\nstarted_r2 = time.time()"
if marker_r3c not in source_r3c:
    raise RuntimeError(f"Cannot find gateway marker in {SOURCE_PATH_R3C}")
exec(
    compile(source_r3c.split(marker_r3c, 1)[0], str(SOURCE_PATH_R3C), "exec"),
    globals(),
)


EXPERIMENT_R3C = "segn_r3_querycausal_absolute_matcher_fold3_v0"
OUT_DIR_R2 = OOF_ROOT / EXPERIMENT_R3C
MODEL_DIR_R2 = OUT_DIR_R2 / "models"
OUT_DIR_R2.mkdir(parents=True, exist_ok=True)
MODEL_DIR_R2.mkdir(parents=True, exist_ok=True)

CAUSAL_CHANNELS_R3C = 32
CAUSAL_HEADS_R3C = 4
CAUSAL_HEAD_DIM_R3C = 8


class CausalResidualR3(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            7,
            padding=3 * dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(4, channels, affine=False)

    def forward(self, values):
        update = self.norm(self.pointwise(self.depthwise(values)))
        return values + F.gelu(update)


class CausalGREncoderR3(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv1d(3, CAUSAL_CHANNELS_R3C, 9, padding=4, bias=False)
        self.norm = nn.GroupNorm(4, CAUSAL_CHANNELS_R3C, affine=False)
        self.blocks = nn.Sequential(
            CausalResidualR3(CAUSAL_CHANNELS_R3C, 1),
            CausalResidualR3(CAUSAL_CHANNELS_R3C, 4),
            CausalResidualR3(CAUSAL_CHANNELS_R3C, 16),
            CausalResidualR3(CAUSAL_CHANNELS_R3C, 32),
        )

    def forward(self, values):
        # Ignore the validity channel.  _chunk_payload already zeroes value and
        # derivatives for the matched no-GR control.
        values = values[:, (0, 2, 3), :]
        encoded = F.gelu(self.norm(self.input(values)))
        return self.blocks(encoded)


class QueryCausalMatcherR3(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CausalGREncoderR3()
        self.query_bilinear = nn.Conv1d(
            CAUSAL_CHANNELS_R3C, CAUSAL_HEAD_DIM_R3C, 1, bias=False
        )
        self.reference_bilinear = nn.Conv1d(
            CAUSAL_CHANNELS_R3C, CAUSAL_HEAD_DIM_R3C, 1, bias=False
        )
        self.bilinear_matrix = nn.Parameter(torch.eye(CAUSAL_HEAD_DIM_R3C))
        self.query_heads = nn.Conv1d(
            CAUSAL_CHANNELS_R3C,
            CAUSAL_HEADS_R3C * CAUSAL_HEAD_DIM_R3C,
            1,
            bias=False,
        )
        self.reference_heads = nn.Conv1d(
            CAUSAL_CHANNELS_R3C,
            CAUSAL_HEADS_R3C * CAUSAL_HEAD_DIM_R3C,
            1,
            bias=False,
        )
        self.head_weights = nn.Parameter(torch.zeros(CAUSAL_HEADS_R3C))
        self.log_scales = nn.Parameter(torch.log(torch.tensor([2.0, 4.0])))
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _normalise(logits, candidate_valid):
        valid = candidate_valid[:, None]
        values = logits.masked_fill(~valid, 0.0)
        count = valid.sum().clamp_min(1).to(logits.dtype)
        mean = values.sum(dim=0) / count
        centred = (values - mean[None]) * valid
        variance = centred.square().sum(dim=0) / count
        output = centred / torch.sqrt(variance + 1.0e-4)[None]
        return output.masked_fill(~valid, -1.0e4)

    def forward(
        self,
        horizontal,
        typewell,
        node_positions,
        candidate_positions,
        geometry,
    ):
        del geometry
        horizontal_embedding = self.encoder(horizontal)[0]
        typewell_embedding = self.encoder(typewell)[0]
        query = horizontal_embedding[:, node_positions]
        reference, axis_valid = _sample_axis(typewell_embedding, candidate_positions)
        reference_raw, raw_axis_valid = _sample_axis(typewell[0], candidate_positions)
        candidate_valid = axis_valid & raw_axis_valid & (reference_raw[1] >= 0.5)

        query_b = self.query_bilinear(query[None])[0]
        reference_b = self.reference_bilinear(reference[None])[0]
        bilinear = torch.einsum(
            "dt,ds->st", self.bilinear_matrix @ query_b, reference_b
        ) / math.sqrt(CAUSAL_HEAD_DIM_R3C)

        query_h = self.query_heads(query[None])[0].reshape(
            CAUSAL_HEADS_R3C, CAUSAL_HEAD_DIM_R3C, -1
        )
        reference_h = self.reference_heads(reference[None])[0].reshape(
            CAUSAL_HEADS_R3C, CAUSAL_HEAD_DIM_R3C, -1
        )
        query_h = F.normalize(query_h, dim=1, eps=1.0e-5)
        reference_h = F.normalize(reference_h, dim=1, eps=1.0e-5)
        heads = torch.einsum("hdt,hds->hst", query_h, reference_h)
        cosine = (
            F.softmax(self.head_weights, dim=0)[:, None, None] * heads
        ).sum(dim=0)

        maps = torch.stack(
            [
                self._normalise(bilinear, candidate_valid),
                self._normalise(cosine, candidate_valid),
            ],
            dim=0,
        )
        maps = maps * torch.exp(self.log_scales).clamp(0.25, 20.0)[:, None, None]
        weight = torch.sigmoid(self.mix_logit)
        logits = weight * maps[0] + (1.0 - weight) * maps[1]
        logits = logits.masked_fill(~candidate_valid[:, None], -1.0e4)
        return logits, weight


def _query_zero_guard_r3(model, device):
    model.eval()
    horizontal = torch.zeros(1, 4, 768, device=device)
    horizontal[:, 1] = 1.0
    typewell = torch.randn(1, 4, 385, device=device)
    typewell[:, 1] = 1.0
    nodes = torch.arange(128, 144, device=device)
    candidates = torch.as_tensor(STATE_OFFSETS, device=device)
    geometry = torch.zeros(8, 16, device=device)
    with torch.no_grad():
        logits, _ = model(horizontal, typewell, nodes, candidates, geometry)
    valid = logits > -1.0e3
    per_column_std = []
    for column in range(logits.shape[1]):
        per_column_std.append(float(logits[:, column][valid[:, column]].std()))
    return {
        "maximum_no_gr_logit_std": float(max(per_column_std)),
        "finite": bool(torch.isfinite(logits).all()),
    }


started_r3c = time.time()
_seed_everything(SEED)
if REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for SEGN-R3")
device_r3c = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folds_r3c, fold_path_r3c = _load_folds()
train_wells_r3c = folds_r3c.loc[
    folds_r3c["fold"].isin(TRAIN_FOLDS_R2), "well"
].tolist()
fold3_wells_r3c = folds_r3c.loc[
    folds_r3c["fold"].eq(VALID_FOLD_R2), "well"
].tolist()
if len(train_wells_r3c) != 309 or len(fold3_wells_r3c) != 154:
    raise RuntimeError(
        f"Unexpected Fold45/Fold3 sizes: {len(train_wells_r3c)}/{len(fold3_wells_r3c)}"
    )
records_r3c = {
    record.well: record
    for record in _load_records(
        sorted(set(train_wells_r3c + fold3_wells_r3c)), "r3_querycausal_fold345"
    )
}
train_records_r3c = [records_r3c[well] for well in train_wells_r3c]
fold3_records_r3c = [records_r3c[well] for well in fold3_wells_r3c]

order_r3c = np.argsort([record.eval_rows for record in fold3_records_r3c])
selection_positions_r3c = np.unique(
    np.rint(np.linspace(0, len(order_r3c) - 1, SELECTION_WELLS_R2)).astype(int)
)
selection_indices_r3c = set(order_r3c[selection_positions_r3c].tolist())
selection_records_r3c = [
    record
    for index, record in enumerate(fold3_records_r3c)
    if index in selection_indices_r3c
]
confirmation_records_r3c = [
    record
    for index, record in enumerate(fold3_records_r3c)
    if index not in selection_indices_r3c
]

model_r3c = QueryCausalMatcherR3().to(device_r3c)
query_zero_guard_r3c = _query_zero_guard_r3(model_r3c, device_r3c)
if (
    not query_zero_guard_r3c["finite"]
    or query_zero_guard_r3c["maximum_no_gr_logit_std"] > 1.0e-6
):
    raise RuntimeError(f"Query-zero invariance failed: {query_zero_guard_r3c}")

print(
    "SEGN-R3 CONFIG",
    {
        "train_wells": len(train_records_r3c),
        "selection_wells": len(selection_records_r3c),
        "confirmation_wells": len(confirmation_records_r3c),
        "parameters": sum(p.numel() for p in model_r3c.parameters()),
        "query_zero_guard": query_zero_guard_r3c,
        "jitters_ft": JITTERS_R2,
        "fold2_touched": False,
        "fold1_touched": False,
    },
    flush=True,
)

model_r3c, best_epoch_r3c, best_objective_r3c = _fit_r2(
    model_r3c, train_records_r3c, selection_records_r3c, device_r3c
)

confirmation_full_r3c = _offset_suite_r2(
    model_r3c, confirmation_records_r3c, device_r3c, control="full"
)
confirmation_no_gr_r3c = _offset_suite_r2(
    model_r3c, confirmation_records_r3c, device_r3c, control="no_gr"
)
confirmation_shuffled_r3c = _offset_suite_r2(
    model_r3c, confirmation_records_r3c, device_r3c, control="shuffled_typewell"
)

for label_r3c, payload_r3c in (
    ("full", confirmation_full_r3c),
    ("no_gr", confirmation_no_gr_r3c),
    ("shuffled_typewell", confirmation_shuffled_r3c),
):
    pd.DataFrame(payload_r3c["offsets"]).to_csv(
        OUT_DIR_R2 / f"confirmation_offsets_{label_r3c}.csv", index=False
    )

checks_r3c = {
    "query_zero_logit_std_le_1e_6": (
        query_zero_guard_r3c["maximum_no_gr_logit_std"] <= 1.0e-6
    ),
    "zero_offset_rmse_le_8": confirmation_full_r3c["zero"]["rmse"] <= 8.0,
    "jittered_rmse_le_10": confirmation_full_r3c["jittered_rmse"] <= 10.0,
    "recovery_slope_ge_0p5": confirmation_full_r3c["recovery_slope"] >= 0.50,
    "no_gr_jittered_degradation_ge_1": (
        confirmation_no_gr_r3c["jittered_rmse"]
        - confirmation_full_r3c["jittered_rmse"]
        >= 1.0
    ),
    "shuffle_jittered_degradation_ge_1": (
        confirmation_shuffled_r3c["jittered_rmse"]
        - confirmation_full_r3c["jittered_rmse"]
        >= 1.0
    ),
    "no_gr_recovery_drop_ge_0p2": (
        confirmation_full_r3c["recovery_slope"]
        - confirmation_no_gr_r3c["recovery_slope"]
        >= 0.20
    ),
}
strict_pass_r3c = bool(all(checks_r3c.values()))

verdict_r3c = {
    "experiment": EXPERIMENT_R3C,
    "stage": "fold3_query_causal_absolute_gr_matching_gateway",
    "sample_unit": "anchor-jitter chunks nested within complete physical wells",
    "deployable": False,
    "truth_usage": "training labels and controlled diagnostic anchor jitter",
    "model_inputs": "horizontal GR and assigned calibrated typewell only",
    "anchor_jitter_is_model_input": False,
    "query_zero_uniform_by_construction": True,
    "geometry_prior_gate_used_in_output": False,
    "uses_pf_sp45_strong_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "train_folds": list(TRAIN_FOLDS_R2),
        "selection_fold": VALID_FOLD_R2,
        "selection_wells": len(selection_records_r3c),
        "confirmation_wells": len(confirmation_records_r3c),
        "fold2_touched": False,
        "fold1_touched": False,
        "selected_epoch": best_epoch_r3c,
        "selection_objective": best_objective_r3c,
    },
    "query_zero_guard": query_zero_guard_r3c,
    "confirmation": {
        "full": confirmation_full_r3c,
        "no_gr": confirmation_no_gr_r3c,
        "shuffled_typewell": confirmation_shuffled_r3c,
    },
    "checks": checks_r3c,
    "strict_pass": strict_pass_r3c,
    "checkpoint": str(MODEL_DIR_R2 / "best_absolute_matcher.pt"),
    "elapsed_minutes": (time.time() - started_r3c) / 60.0,
    "next": (
        "PASS -> fuse this causal absolute emission with the R1 prior"
        if strict_pass_r3c
        else "FAIL -> close local bilinear/cosine SEGN matching and change evidence family"
    ),
}
_write_json(OUT_DIR_R2 / "verdict.json", verdict_r3c)
print("SEGN-R3 FINAL VERDICT", json.dumps(verdict_r3c, indent=2), flush=True)
print("saved:", OUT_DIR_R2, flush=True)

if not strict_pass_r3c:
    raise RuntimeError(json.dumps(verdict_r3c, indent=2))

