import math

import torch

from first_place_repro.train import source_member_seeds


def test_cosine_schedule_is_stepped_per_optimizer_update():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    steps_per_epoch = math.ceil(39 / 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=300 * steps_per_epoch, eta_min=5e-7
    )
    initial = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < initial
    assert scheduler.T_max == 300 * steps_per_epoch


def test_source_member_seed_protocol():
    assert source_member_seeds(7, 0, 1) == (0, 7, 8, 9)
    assert source_member_seeds(7, 0, 5) == (4, 11, 4008, 4009)
    assert source_member_seeds(7, 1, 1) == (5, 12, 5008, 5009)
