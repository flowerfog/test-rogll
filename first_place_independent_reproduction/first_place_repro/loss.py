from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .inferred_defaults import LossDefaults


class AlignmentLoss(nn.Module):
    def __init__(self, defaults: LossDefaults = LossDefaults()):
        super().__init__()
        self.defaults = defaults

    def forward(
        self,
        logits: torch.Tensor,
        target_probability: torch.Tensor,
        target_offset: torch.Tensor,
        target_mask: torch.Tensor,
        state_offsets: torch.Tensor,
        gr_gap: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if logits.shape != target_probability.shape:
            raise ValueError((logits.shape, target_probability.shape))
        mask = target_mask.bool()
        if not mask.any():
            raise ValueError("batch has no supervised bins")
        log_probability = F.log_softmax(logits.float(), dim=-1)
        probability = log_probability.exp()
        cross_entropy = -(target_probability.float() * log_probability).sum(dim=-1)[mask].mean()
        expected = (probability * state_offsets.float()[:, None, :]).sum(dim=-1)
        expected_normalized = (
            expected - self.defaults.target_mean_ft
        ) / self.defaults.target_std_ft
        target_normalized = (
            target_offset.float() - self.defaults.target_mean_ft
        ) / self.defaults.target_std_ft
        path_huber = F.huber_loss(
            expected_normalized[mask],
            target_normalized[mask],
            delta=self.defaults.huber_delta_ft,
        )
        gr_penalty = (probability * gr_gap.float()).sum(dim=-1)[mask].mean()
        total = (
            self.defaults.cross_entropy_weight * cross_entropy
            + self.defaults.path_huber_weight * path_huber
            + self.defaults.gr_penalty_weight * gr_penalty
        )
        return total, {
            "cross_entropy": cross_entropy.detach(),
            "path_huber": path_huber.detach(),
            "gr_penalty": gr_penalty.detach(),
            "expected": expected.detach(),
        }
