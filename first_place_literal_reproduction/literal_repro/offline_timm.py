from __future__ import annotations

from pathlib import Path


def install(checkpoint: str | Path) -> None:
    import timm
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = load_file(str(checkpoint), device="cpu")
    original = timm.create_model

    def create_model(model_name, *args, **kwargs):
        pretrained = bool(kwargs.pop("pretrained", False))
        if not pretrained:
            return original(model_name, *args, pretrained=False, **kwargs)
        # The supplied safetensors file is the disclosed ConvNeXt-small
        # ImageNet initialization. Construct the same timm graph offline and
        # load all shared encoder parameters before the author converts norms.
        local_name = str(model_name).removeprefix("hf_hub:timm/").removeprefix("timm/")
        try:
            model = original(local_name, *args, pretrained=False, **kwargs)
        except Exception:
            model = original("convnext_small", *args, pretrained=False, **kwargs)
        result = model.load_state_dict(state, strict=False)
        expected = {name for name in model.state_dict() if name.startswith(("stem.", "stages."))}
        loaded = expected - set(result.missing_keys)
        coverage = len(loaded) / max(len(expected), 1)
        report = {
            "checkpoint": str(checkpoint), "loaded": len(loaded),
            "expected": len(expected), "coverage": coverage,
            "missing": [name for name in result.missing_keys if name in expected],
        }
        print("OFFLINE TIMM", report, flush=True)
        if coverage < 0.95:
            raise RuntimeError({"message": "offline ConvNeXt coverage too low", **report})
        return model

    timm.create_model = create_model
