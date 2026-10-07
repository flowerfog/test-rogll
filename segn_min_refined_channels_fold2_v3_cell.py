"""SEGN-min v3: strong independently supervised cost-map channels.

V2 proved that online scheduled sampling and posterior-mean readout work, but
its direct bilinear/cosine maps failed the evidence gate: teacher-anchor RMSE
was high and no-GR was not worse. V3 keeps the validated recursive protocol
and changes only the evidence model:

* bilinear and multi-head-cosine channels each receive their own state-by-time
  axial refinement network;
* each channel is directly supervised by an auxiliary soft-target CE;
* a learned per-step soft gate combines the two refined maps;
* Fold-3 teacher-anchor and causal controls must pass before Fold 2 is touched.

Prerequisites
-------------
Keep these files in /root/autodl-tmp (or the current directory):
* segn_chunked_costmap_rebased_fold2_gateway_v1_cell.py
* segn_min_scheduled_gate_fold2_v2_cell.py

The V2 definitions are bootstrapped, then its guarded nested execution tail is
run with the V3 model and loss. No V1/V2 checkpoint is loaded.
"""

from __future__ import annotations

from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


V2_SOURCE_CANDIDATES = (
    Path("/root/autodl-tmp/segn_min_scheduled_gate_fold2_v2_cell.py"),
    Path.cwd() / "segn_min_scheduled_gate_fold2_v2_cell.py",
)
V2_SOURCE = next((path for path in V2_SOURCE_CANDIDATES if path.exists()), None)
if V2_SOURCE is None:
    raise FileNotFoundError(
        "Upload segn_min_scheduled_gate_fold2_v2_cell.py to /root/autodl-tmp."
    )

v2_source = V2_SOURCE.read_text(encoding="utf-8")
main_marker = "\nstarted = time.time()\nif REQUIRE_CUDA"
if main_marker not in v2_source:
    raise RuntimeError(f"Cannot find execution marker in {V2_SOURCE}")
v2_definitions, v2_tail = v2_source.split(main_marker, 1)
exec(compile(v2_definitions, str(V2_SOURCE), "exec"), globals())


EXPERIMENT = "segn_min_refined_channels_fold2_v3"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

AUXILIARY_CHANNEL_CE = 0.30
GATE_HIDDEN = 24
CHANNEL_BASE = 24


class RefinedCostChannel(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(input_channels, CHANNEL_BASE, 1),
            nn.GroupNorm(4, CHANNEL_BASE),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            AxialBlock(CHANNEL_BASE, 1),
            AxialBlock(CHANNEL_BASE, 2),
            AxialBlock(CHANNEL_BASE, 4),
            AxialBlock(CHANNEL_BASE, 8),
        )
        self.output = nn.Conv2d(CHANNEL_BASE, 1, 1)

    def forward(self, volume):
        return self.output(self.blocks(self.input(volume))).squeeze(0).squeeze(0)


class SEGNMinCostMapNet(nn.Module):
    """Two full cost-map channels plus an explicit evidence-quality gate."""

    def __init__(self):
        super().__init__()
        self.encoder = MultiScaleGREncoder()

        self.query_bilinear = nn.Conv1d(EMBEDDING, HEAD_DIM, 1)
        self.reference_bilinear = nn.Conv1d(EMBEDDING, HEAD_DIM, 1)
        self.bilinear_matrix = nn.Parameter(torch.eye(HEAD_DIM))

        self.query_heads = nn.Conv1d(EMBEDDING, HEADS * HEAD_DIM, 1)
        self.reference_heads = nn.Conv1d(EMBEDDING, HEADS * HEAD_DIM, 1)

        # Common context contains value/d1 mismatch, validity, coordinates,
        # legal geometry/recursion channels, and the recursive anchor offset.
        common_channels = 2 + 3 + 2 + 8 + 1
        self.bilinear_refiner = RefinedCostChannel(1 + common_channels)
        self.cosine_refiner = RefinedCostChannel(HEADS + common_channels)

        self.gate = nn.Sequential(
            nn.Linear(12, GATE_HIDDEN),
            nn.GELU(),
            nn.Linear(GATE_HIDDEN, 12),
            nn.GELU(),
            nn.Linear(12, 2),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.channel_log_scale = nn.Parameter(torch.zeros(2))

    @staticmethod
    def _distribution_features(logits, states):
        probability = F.softmax(logits, dim=0)
        entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=0)
        entropy = entropy / math.log(max(logits.shape[0], 2))
        sharpness = 1.0 - entropy
        confidence = probability.max(dim=0).values
        expected = (probability * states[:, None]).sum(dim=0)
        return sharpness, confidence, expected

    def forward(
        self,
        horizontal,
        typewell,
        node_positions,
        candidate_positions,
        geometry,
        channel_control="full",
        return_aux=False,
    ):
        horizontal_embedding = self.encoder(horizontal)[0]
        typewell_embedding = self.encoder(typewell)[0]
        query = horizontal_embedding[:, node_positions]
        reference, axis_valid = _sample_axis(typewell_embedding, candidate_positions)

        query_a = self.query_bilinear(query[None])[0]
        reference_a = self.reference_bilinear(reference[None])[0]
        transformed_query = self.bilinear_matrix @ query_a
        bilinear = torch.einsum("dt,ds->st", transformed_query, reference_a)
        bilinear = bilinear[None] / math.sqrt(HEAD_DIM)

        query_b = self.query_heads(query[None])[0].reshape(HEADS, HEAD_DIM, -1)
        reference_b = self.reference_heads(reference[None])[0].reshape(
            HEADS, HEAD_DIM, -1
        )
        query_b = F.normalize(query_b, dim=1, eps=1e-5)
        reference_b = F.normalize(reference_b, dim=1, eps=1e-5)
        cosine = torch.einsum("hdt,hds->hst", query_b, reference_b)

        query_raw = horizontal[0, :, node_positions]
        reference_raw, raw_axis_valid = _sample_axis(typewell[0], candidate_positions)
        value_difference = -torch.abs(
            reference_raw[0, :, None] - query_raw[0, None, :]
        )[None]
        d1_difference = -torch.abs(
            reference_raw[2, :, None] - query_raw[2, None, :]
        )[None]
        query_valid = query_raw[1][None, None, :].expand(
            1, len(candidate_positions), -1
        )
        reference_valid = reference_raw[1][None, :, None].expand(
            1, -1, query.shape[-1]
        )
        joint_valid = query_valid * reference_valid
        candidate_valid = axis_valid & raw_axis_valid & (reference_raw[1] >= 0.5)
        valid_volume = candidate_valid[:, None]

        state_coordinate = (
            torch.from_numpy(STATE_OFFSETS).to(query.device, query.dtype)[None, :, None]
            / STATE_HALF_WIDTH_FT
        ).expand(1, -1, query.shape[-1])
        chunk_coordinate = torch.linspace(
            0.0, 1.0, query.shape[-1], device=query.device, dtype=query.dtype
        )[None, None, :].expand(1, len(candidate_positions), -1)
        geometry_volume = geometry[None, :, None, :].expand(
            1, -1, len(candidate_positions), -1
        )[0]
        anchor_offset = candidate_positions[len(candidate_positions) // 2]
        anchor_volume = (
            anchor_offset.reshape(1, 1, 1)
            .expand(1, len(candidate_positions), query.shape[-1])
            / REFERENCE_HALF_WIDTH_FT
        )
        common = torch.cat(
            [
                value_difference,
                d1_difference,
                query_valid,
                reference_valid,
                joint_valid,
                state_coordinate,
                chunk_coordinate,
                geometry_volume,
                anchor_volume,
            ],
            dim=0,
        )[None]

        logits_a = self.bilinear_refiner(
            torch.cat([bilinear[None], common], dim=1)
        )
        logits_b = self.cosine_refiner(
            torch.cat([cosine[None], common], dim=1)
        )
        logits_a = logits_a.masked_fill(~valid_volume, -1.0e4)
        logits_b = logits_b.masked_fill(~valid_volume, -1.0e4)

        states = torch.from_numpy(STATE_OFFSETS).to(query.device, query.dtype)
        sharp_a, conf_a, expected_a = self._distribution_features(logits_a, states)
        sharp_b, conf_b, expected_b = self._distribution_features(logits_b, states)
        provisional = 0.5 * (logits_a + logits_b)
        provisional_conf = F.softmax(provisional, dim=0).max(dim=0).values
        disagreement = torch.abs(expected_a - expected_b) / STATE_HALF_WIDTH_FT
        query_coverage = query_raw[1].clamp(0.0, 1.0)
        reference_coverage = candidate_valid.float().mean().expand_as(query_coverage)
        gate_features = torch.stack(
            [
                sharp_a,
                sharp_b,
                conf_a,
                conf_b,
                provisional_conf,
                disagreement,
                geometry[1].clamp(-1.0, 1.0),
                query_coverage,
                reference_coverage,
                geometry[3].clamp(-2.0, 2.0),
                geometry[6].clamp(-4.0, 4.0),
                geometry[7].clamp(-4.0, 4.0),
            ],
            dim=1,
        )
        gate_weights = F.softmax(self.gate(gate_features), dim=1).transpose(0, 1)
        channel_logits = torch.stack([logits_a, logits_b], dim=0)
        channel_logits = channel_logits * torch.exp(self.channel_log_scale).clamp(
            0.5, 10.0
        )[:, None, None]

        if channel_control == "no_bilinear":
            gate_weights = torch.stack(
                [torch.zeros_like(gate_weights[0]), torch.ones_like(gate_weights[1])]
            )
        elif channel_control == "no_cosine":
            gate_weights = torch.stack(
                [torch.ones_like(gate_weights[0]), torch.zeros_like(gate_weights[1])]
            )
        logits = (gate_weights[:, None, :] * channel_logits).sum(dim=0)
        logits = logits.masked_fill(~valid_volume, -1.0e4)
        if not return_aux:
            return logits
        return logits, {
            "gate_weights": gate_weights,
            "channel_logits": channel_logits,
            "channel_expected": torch.stack([expected_a, expected_b]),
        }


def _soft_ce(logits, target_distribution):
    return -(
        target_distribution * F.log_softmax(logits, dim=0)
    ).sum(dim=0).mean()


def _loss_min(model, payload, device):
    tensors = _tensor_payload(payload, device)
    logits, model_aux = model(*tensors, return_aux=True)
    target = torch.from_numpy(payload["target_state"]).to(device)
    states = torch.from_numpy(STATE_OFFSETS).to(device)[:, None]
    target_distribution = torch.exp(
        -0.5 * ((states - target[None]) / TARGET_SIGMA_FT_MIN) ** 2
    )
    target_distribution = target_distribution / target_distribution.sum(
        dim=0, keepdim=True
    ).clamp_min(1e-8)

    ce = _soft_ce(logits, target_distribution)
    ce_a = _soft_ce(model_aux["channel_logits"][0], target_distribution)
    ce_b = _soft_ce(model_aux["channel_logits"][1], target_distribution)
    probability = F.softmax(logits, dim=0)
    expected = (probability * states).sum(dim=0)
    mean_loss = F.smooth_l1_loss(expected / 8.0, target / 8.0)
    auxiliary_ce = 0.5 * (ce_a + ce_b)
    total = ce + AUXILIARY_CHANNEL_CE * auxiliary_ce + LOSS_MEAN * mean_loss
    return total, {
        "ce": float(ce.detach()),
        "mean": float(mean_loss.detach()),
        "ce_bilinear": float(ce_a.detach()),
        "ce_cosine": float(ce_b.detach()),
        "expected": expected.detach(),
        "gate_weights": model_aux["gate_weights"].detach(),
    }


# Execute V2's nested protocol with the V3 model/loss and its strengthened
# Fold-3 evidence and causal guards.
tail = "\nstarted = time.time()\nif REQUIRE_CUDA" + v2_tail
exec(compile(tail, str(V2_SOURCE) + "::<v3_execution>", "exec"), globals())

