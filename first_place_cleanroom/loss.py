from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import TRAIN


def decode_expected(logits: torch.Tensor, state_tvt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probability = logits.float().softmax(dim=1)
    if state_tvt.ndim == 1:
        state_tvt = state_tvt[None].expand(logits.shape[0], -1)
    expected = (probability * state_tvt[:, :, None]).sum(dim=1)
    return probability, expected


def weighted_huber_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    state_tvt: torch.Tensor,
    truth_tvt: torch.Tensor,
    gr_gap: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
    truth_tvt = torch.nan_to_num(truth_tvt.float(), nan=0.0, posinf=0.0, neginf=0.0)
    gr_gap = torch.nan_to_num(gr_gap.float(), nan=0.0, posinf=0.0, neginf=0.0)
    probability, expected = decode_expected(logits, state_tvt)
    valid = mask.bool()
    denominator = valid.sum().clamp_min(1)
    ce_column = -(target * logits.float().log_softmax(dim=1)).sum(dim=1)
    ce = (ce_column * valid).sum() / denominator
    if valid.any():
        huber = F.huber_loss(expected[valid], truth_tvt[valid], delta=1.0, reduction="mean")
        gr = ((probability * gr_gap).sum(dim=1) * valid).sum() / denominator
    else:
        # Some wells leave the disclosed +/-100 ft state grid entirely.
        # They have no supervised alignment column for this crop.
        huber = expected.sum() * 0.0
        gr = probability.sum() * 0.0
    total = TRAIN.alignment_weight * ce + TRAIN.regression_weight * huber + TRAIN.gr_penalty_weight * gr
    return total, {"alignment": ce.detach(), "regression": huber.detach(), "gr_penalty": gr.detach()}
