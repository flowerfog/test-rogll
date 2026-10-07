from __future__ import annotations

from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F


def bn(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels, eps=1e-4, momentum=.1, track_running_stats=True)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = bn(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1), nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1),
        )
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(self.norm(self.conv_dw(x)))
        return x + y * self.gamma[None, :, None, None]


class ConvNeXtSmallBN(nn.Module):
    """ConvNeXt-small with the disclosed BN and non-parametric downsampling changes."""
    depths = (3, 3, 27, 3)
    dims = (96, 192, 384, 768)

    def __init__(self, in_channels: int):
        super().__init__()
        # Disclosed configuration: stem_stride=(2, 4), stem_down_mode="pool".
        # The channel projection is parametric; spatial reduction is not.
        self.stem = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 4), stride=(2, 4)),
            nn.Conv2d(in_channels, 96, 3, padding=1),
            bn(96),
        )
        self.stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for index, (depth, dim) in enumerate(zip(self.depths, self.dims)):
            self.stages.append(nn.Sequential(*(ConvNeXtBlock(dim) for _ in range(depth))))
            if index < 3:
                # Disclosed updown_mode_2d="non-param": projection and pooling
                # are separate so no learned kernel performs spatial reduction.
                self.downsample.append(nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    bn(dim),
                    nn.Conv2d(dim, self.dims[index + 1], 1),
                ))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        output = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            output.append(x)
            if index < len(self.downsample):
                x = self.downsample[index](x)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), bn(channels), nn.SiLU(),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(channels, channels, 3, padding=1), bn(channels), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.project = nn.Conv2d(in_channels + skip_channels, out_channels, 1)
        self.block = ResidualBlock(out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(self.project(torch.cat((x, skip), dim=1)))


class AlignmentUNet(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.0,
                 layout: str = "state_horizontal"):
        super().__init__()
        if layout not in {"state_horizontal", "horizontal_state"}:
            raise ValueError(f"unknown network layout: {layout}")
        self.layout = layout
        self.encoder = ConvNeXtSmallBN(in_channels)
        # Public configuration: emb_dim=32, mults=(1,2,4,4), one block per stage.
        self.lateral = nn.ModuleList([nn.Conv2d(a, b, 1) for a, b in zip((96, 192, 384, 768), (32, 64, 128, 128))])
        self.decode2 = DecoderStage(128, 128, 128, dropout)
        self.decode1 = DecoderStage(128, 64, 64, dropout)
        self.decode0 = DecoderStage(64, 32, 32, dropout)
        self.head = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), bn(64), nn.SiLU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The loss/decoder contract remains (batch, state, horizontal). Only
        # the encoder's spatial presentation is switched for the audit.
        transposed = self.layout == "horizontal_state"
        if transposed:
            x = x.transpose(-2, -1)
        size = x.shape[-2:]
        features = [project(value) for project, value in zip(self.lateral, self.encoder(x))]
        y = self.decode2(features[3], features[2])
        y = self.decode1(y, features[1])
        y = self.decode0(y, features[0])
        output = F.interpolate(self.head(y), size=size, mode="bilinear", align_corners=False)[:, 0]
        return output.transpose(-2, -1) if transposed else output


def load_encoder_checkpoint(model: AlignmentUNet, path: Path) -> dict[str, object]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        source = load_file(str(path), device="cpu")
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        source = payload.get("model", payload)
    destination = model.encoder.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    for raw_key, value in source.items():
        key = raw_key.removeprefix("model.").removeprefix("encoder.")
        key = key.replace("stages_", "stages.").replace("stem_", "stem.")
        downsample_layers = re.match(r"downsample_layers\.(\d+)\.(.+)", key)
        if downsample_layers:
            layer = int(downsample_layers.group(1))
            key = (f"stem.{downsample_layers.group(2)}" if layer == 0 else
                   f"downsample.{layer - 1}.{downsample_layers.group(2)}")
        key = re.sub(r"stages\.(\d+)\.blocks\.(\d+)", r"stages.\1.\2", key)
        match = re.match(r"stages\.(\d+)\.downsample\.(.+)", key)
        if match:
            key = f"downsample.{int(match.group(1))}.{match.group(2)}"
        key = key.replace(".mlp.fc1.", ".mlp.0.").replace(".mlp.fc2.", ".mlp.2.")
        if key not in destination:
            continue
        expected = destination[key]
        if value.ndim == 2 and expected.ndim == 4:
            value = value[:, :, None, None]
        if value.shape == expected.shape:
            loaded[key] = value
    missing, unexpected = model.encoder.load_state_dict(loaded, strict=False)
    buffer_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    parameter_keys = [key for key in destination if not key.endswith(buffer_suffixes)]
    parameter_loaded = sum(key in loaded for key in parameter_keys)
    parameter_coverage = parameter_loaded / max(1, len(parameter_keys))
    # ImageNet checkpoints do not carry BN running statistics; the model initializes them.
    if parameter_coverage < .75:
        raise RuntimeError({"message": "encoder parameter checkpoint coverage too low",
                            "parameter_coverage": parameter_coverage,
                            "loaded_parameters": parameter_loaded,
                            "expected_parameters": len(parameter_keys),
                            "loaded_total": len(loaded), "expected_total": len(destination),
                            "missing": list(missing)[:30]})
    return {"parameter_coverage": parameter_coverage, "loaded_parameters": parameter_loaded,
            "loaded_total": len(loaded), "expected_total": len(destination),
            "missing_bn_or_other": list(missing), "unexpected": list(unexpected)}
