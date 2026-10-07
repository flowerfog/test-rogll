from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = "0719_convnext_small_stage_unet_v2"
CONTRACT_VERSION = "0719_contract_v11_pageable_activation_offload"


class ChannelLastFP32BatchNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-4, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm2d(features, eps=eps, momentum=momentum, track_running_stats=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.permute(0, 3, 1, 2)
        x = self.bn(x.float()).to(dtype=dtype)
        return x.permute(0, 2, 3, 1)


class ChannelFirstFP32BatchNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-4, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm2d(features, eps=eps, momentum=momentum, track_running_stats=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return self.bn(x.float()).to(dtype=dtype)


def _norm_features(module: nn.Module) -> int:
    shape = getattr(module, "normalized_shape", None)
    if shape is not None:
        return int(shape[-1] if isinstance(shape, (tuple, list)) else shape)
    return int(getattr(module, "num_features"))


def replace_convnext_norms(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        class_name = child.__class__.__name__.lower()
        if "layernorm" in class_name:
            features = _norm_features(child)
            # timm LayerNorm2d receives NCHW; torch LayerNorm receives NHWC.
            replacement = (
                ChannelFirstFP32BatchNorm(features)
                if class_name == "layernorm2d"
                else ChannelLastFP32BatchNorm(features)
            )
            source_weight = getattr(child, "weight", None)
            source_bias = getattr(child, "bias", None)
            with torch.no_grad():
                if source_weight is not None:
                    replacement.bn.weight.copy_(source_weight.detach().float().reshape(-1))
                if source_bias is not None:
                    replacement.bn.bias.copy_(source_bias.detach().float().reshape(-1))
            setattr(module, name, replacement)
        else:
            replace_convnext_norms(child)


def load_backbone_checkpoint(backbone: nn.Module, path: str) -> dict[str, float | int]:
    source_path = Path(path)
    if source_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        source = load_file(str(source_path), device="cpu")
    else:
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        source = payload.get("state_dict", payload.get("model", payload))
    source = {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in source.items()
    }
    target = backbone.state_dict()
    loaded = {key: value for key, value in source.items() if key in target and value.shape == target[key].shape}
    backbone.load_state_dict(loaded, strict=False)
    stage_keys = [key for key, value in target.items() if key.startswith("stages.") and value.is_floating_point()]
    covered = sum(target[key].numel() for key in stage_keys if key in loaded)
    total = sum(target[key].numel() for key in stage_keys)
    report = {
        "loaded": len(loaded),
        "stage_parameter_coverage": covered / max(total, 1),
    }
    if report["stage_parameter_coverage"] < 0.95:
        raise RuntimeError({"message": "ConvNeXt stage checkpoint coverage too low", **report})
    print("ENCODER CHECKPOINT", report, flush=True)
    return report


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(dropout)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.drop(self.bn2(self.conv2(x)))
        return self.act(x + residual)


class ConcatenateUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, dropout: float):
        super().__init__()
        self.project = nn.Conv2d(in_channels + skip_channels, skip_channels, 1, bias=False)
        self.block = ResidualBlock(skip_channels, skip_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(self.project(torch.cat([x, skip], dim=1)))


class AddUpBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.block = ResidualBlock(channels, channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(x + skip)


class AlignmentUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        encoder_name: str = "convnext_small.in12k_ft_in1k_384",
        pretrained: bool = True,
        encoder_checkpoint: str | None = None,
        dropout: float = 0.05,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise RuntimeError("timm is required for the disclosed ConvNeXt backbone") from error
        backbone = timm.create_model(
            encoder_name,
            pretrained=pretrained and not encoder_checkpoint,
            num_classes=0,
        )
        if encoder_checkpoint:
            load_backbone_checkpoint(backbone, encoder_checkpoint)
        if not hasattr(backbone, "stages") or len(backbone.stages) != 4:
            raise RuntimeError("ConvNeXt backbone does not expose four stages")
        self.backbone_stages = nn.ModuleList(list(backbone.stages))
        replace_convnext_norms(self.backbone_stages)

        stage_channels = (96, 192, 384, 768)
        embedding = 32
        self.stem = ResidualBlock(in_channels, embedding, dropout)
        self.stem_down = nn.AvgPool2d((2, 4), stride=(2, 4), ceil_mode=True)
        self.to_backbone = nn.Conv2d(embedding, stage_channels[0], 1, bias=True)
        self.up_blocks = nn.ModuleList(
            ConcatenateUpBlock(source, skip, dropout)
            for source, skip in zip(reversed(stage_channels[1:]), reversed(stage_channels[:-1]))
        )
        self.to_full_resolution = nn.Conv2d(stage_channels[0], embedding, 1, bias=False)
        self.full_up = AddUpBlock(embedding, dropout)
        self.out = ResidualBlock(embedding, embedding, dropout)
        self.head = nn.Conv2d(embedding, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem_skip = self.stem(x)
        x = self.to_backbone(self.stem_down(stem_skip))
        skips = []
        for index, stage in enumerate(self.backbone_stages):
            x = stage(x)
            if index < 3:
                skips.append(x)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip)
        x = self.to_full_resolution(x)
        x = self.full_up(x, stem_skip)
        return self.head(self.out(x))[:, 0]


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            incoming = source[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(incoming, alpha=1.0 - self.decay)
            else:
                value.copy_(incoming)
