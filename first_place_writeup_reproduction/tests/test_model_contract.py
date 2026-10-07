import tempfile
from pathlib import Path

import torch
from torch import nn

from writeup_repro.model import AlignmentUNet


def test_average_pool_downsampling_and_checkpoint_parity():
    model = AlignmentUNet(17, pretrained=False).eval()
    strided_convolutions = [
        name
        for name, layer in model.encoder.named_modules()
        if isinstance(layer, nn.Conv2d) and max(layer.stride) > 1
    ]
    average_pools = [
        name for name, layer in model.encoder.named_modules() if isinstance(layer, nn.AvgPool2d)
    ]
    assert strided_convolutions == []
    assert len(average_pools) == 4

    values = torch.randn(1, 17, 64, 64)
    with torch.no_grad():
        expected = model(values)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        torch.save(model.state_dict(), path)
        restored = AlignmentUNet(17, pretrained=False).eval()
        restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
        with torch.no_grad():
            actual = restored(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
