from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = "convnext_small_bn_avgpool_unet"


class FlexibleFP32BatchNorm(nn.Module):
    """BatchNorm replacement that accepts ConvNeXt NCHW or NHWC tensors."""

    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.features = features
        self.norm = nn.BatchNorm2d(features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if x.ndim == 2:
            output = self.norm(x.float()[:, :, None, None])[:, :, 0, 0]
        elif x.ndim == 4 and x.shape[1] == self.features:
            output = self.norm(x.float())
        elif x.ndim == 4 and x.shape[-1] == self.features:
            output = self.norm(x.float().permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:
            raise ValueError(f"cannot apply {self.features}-channel BN to {tuple(x.shape)}")
        return output.to(dtype)


def _pooled_projection_from_strided_conv(conv: nn.Conv2d) -> nn.Sequential:
    if conv.kernel_size != conv.stride or conv.kernel_size[0] != conv.kernel_size[1]:
        raise ValueError(f"cannot convert downsampling convolution {conv}")
    factor = conv.stride[0]
    projection = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=1,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        projection.weight.copy_(conv.weight.sum(dim=(-1, -2), keepdim=True))
        if conv.bias is not None:
            projection.bias.copy_(conv.bias)
    return nn.Sequential(nn.AvgPool2d(factor, stride=factor), projection)


def replace_convnext_downsampling_with_average_pool(module: nn.Module) -> list[str]:
    replaced = []
    stem = getattr(module, "stem_0", None)
    if isinstance(stem, nn.Conv2d) and stem.stride[0] > 1:
        module.stem_0 = _pooled_projection_from_strided_conv(stem)
        replaced.append("stem_0")
    for index in range(1, 4):
        stage = getattr(module, f"stages_{index}", None)
        if stage is None or not isinstance(stage.downsample, nn.Sequential):
            raise RuntimeError(f"missing ConvNeXt stage {index} downsample")
        layers = list(stage.downsample.children())
        if len(layers) != 2 or not isinstance(layers[1], nn.Conv2d):
            raise RuntimeError(f"unexpected ConvNeXt stage {index} downsample: {stage.downsample}")
        stage.downsample = nn.Sequential(layers[0], _pooled_projection_from_strided_conv(layers[1]))
        replaced.append(f"stages_{index}.downsample")
    return replaced


def replace_layer_norm_with_batch_norm(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm) or child.__class__.__name__.lower().startswith("layernorm"):
            shape = getattr(child, "normalized_shape", None)
            features = int(shape[-1] if isinstance(shape, (tuple, list)) else shape)
            replacement = FlexibleFP32BatchNorm(features, eps=float(getattr(child, "eps", 1e-5)))
            if getattr(child, "weight", None) is not None and child.weight.numel() == features:
                replacement.norm.weight.data.copy_(child.weight.detach().float().reshape(-1))
            if getattr(child, "bias", None) is not None and child.bias.numel() == features:
                replacement.norm.bias.data.copy_(child.bias.detach().float().reshape(-1))
            setattr(module, name, replacement)
        else:
            replace_layer_norm_with_batch_norm(child)
    return module


def load_encoder_checkpoint(module: nn.Module, path: str) -> dict[str, float | int]:
    source_path = Path(path)
    if source_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        source = load_file(str(source_path), device="cpu")
    else:
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        source = payload.get("state_dict", payload.get("model", payload))
    source = {key.removeprefix("module.").removeprefix("model."): value for key, value in source.items()}

    def canonical(key: str) -> str:
        # timm's features_only wrapper flattens only the top-level stage names.
        if key.startswith("stem_"):
            key = "stem." + key[len("stem_") :]
        if key.startswith("stages_"):
            key = "stages." + key[len("stages_") :]
        return key

    canonical_source = {canonical(key): value for key, value in source.items()}
    target = module.state_dict()
    loaded = {}
    adapted = []
    for key, value in target.items():
        incoming = canonical_source.get(canonical(key))
        if incoming is not None and incoming.shape == value.shape:
            loaded[key] = incoming
            continue
        if incoming is not None and value.ndim == 4 and incoming.ndim == 4:
            if value.shape[0] == incoming.shape[0] and value.shape[2:] == incoming.shape[2:]:
                repeat = math.ceil(value.shape[1] / incoming.shape[1])
                expanded = incoming.repeat(1, repeat, 1, 1)[:, : value.shape[1]]
                loaded[key] = expanded * (incoming.shape[1] / value.shape[1])
                adapted.append(key)
    module.load_state_dict(loaded, strict=False)
    trainable = {key: value for key, value in target.items() if value.is_floating_point()}
    covered = sum(trainable[key].numel() for key in loaded if key in trainable)
    total = sum(value.numel() for value in trainable.values())
    coverage = covered / max(total, 1)
    report = {"loaded": len(loaded), "expected": len(target), "parameter_coverage": coverage, "adapted": len(adapted)}
    if coverage < 0.85:
        raise RuntimeError({"message": "encoder checkpoint coverage too low", **report})
    print("ENCODER CHECKPOINT", report, flush=True)
    return report


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class ResidualUpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )
        self.residual = nn.Sequential(ResidualBlock(output_channels), ResidualBlock(output_channels))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.residual(self.project(torch.cat([x, skip], dim=1)))


class AlignmentUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        encoder_name: str = "convnext_small.in12k_ft_in1k_384",
        pretrained: bool = True,
        encoder_checkpoint: str | None = None,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise RuntimeError(
                "The disclosed timm ConvNeXt backbone is required. Install timm; no fallback is used."
            ) from error
        try:
            self.encoder = timm.create_model(
                encoder_name,
                pretrained=pretrained and not encoder_checkpoint,
                features_only=True,
                in_chans=in_channels,
                out_indices=(0, 1, 2, 3),
            )
        except Exception as error:
            raise RuntimeError(f"cannot create disclosed encoder {encoder_name!r}") from error
        if encoder_checkpoint:
            load_encoder_checkpoint(self.encoder, encoder_checkpoint)
        replaced_downsampling = replace_convnext_downsampling_with_average_pool(self.encoder)
        if len(replaced_downsampling) != 4:
            raise RuntimeError(f"expected four average-pool downsamplers, got {replaced_downsampling}")
        replace_layer_norm_with_batch_norm(self.encoder)
        channels = list(self.encoder.feature_info.channels())
        if len(channels) != 4:
            raise RuntimeError(f"expected four ConvNeXt stages, got {channels}")
        stem_channels = 48
        self.full_resolution_stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
            ResidualBlock(stem_channels),
        )
        decoder_channels = (384, 192, 96)
        self.up3 = ResidualUpBlock(channels[3], channels[2], decoder_channels[0])
        self.up2 = ResidualUpBlock(decoder_channels[0], channels[1], decoder_channels[1])
        self.up1 = ResidualUpBlock(decoder_channels[1], channels[0], decoder_channels[2])
        self.final = ResidualUpBlock(decoder_channels[2], stem_channels, 64)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        stem = self.full_resolution_stem(x)
        features = self.encoder(x)
        y = self.up3(features[3], features[2])
        y = self.up2(y, features[1])
        y = self.up1(y, features[0])
        y = self.final(y, stem)
        return self.head(y)[:, 0, : size[0], : size[1]]


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
