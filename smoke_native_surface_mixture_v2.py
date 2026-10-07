from pathlib import Path
import ast
import sys
import types

import torch


path = Path(__file__).with_name("native_tvt_mixture_surfacefeatures_fold2_v2r1_cell.py")
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
body = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "STARTED_DCE"
        for target in node.targets
    ):
        break
    body.append(node)
module = types.ModuleType("native_surface_mixture_v2_smoke")
module.__file__ = str(path)
sys.modules[module.__name__] = module
exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)

model = module.SurfaceMixtureCostNetDCE()
length = 512
reference_length = len(module.REFERENCE_OFFSETS)
horizontal = torch.randn(1, 4, length)
horizontal[:, 1] = 1.0
condition = torch.randn(1, 10, length) * 0.05
condition[:, 0, :160] = 1.0
condition[:, 0, 160:] = 0.0
condition[:, 1] = 0.0
condition[:, 4] = torch.linspace(0.0, 0.15, length)
typewell = torch.randn(1, 4, reference_length)
heel = torch.randn(1, 4, reference_length)
typewell[:, 1] = 1.0
heel[:, 1] = 1.0
logits, path_prediction, diagnostics = model(horizontal, condition, typewell, heel)
assert logits.shape == (1, module.N_STATES, length // module.HORIZONTAL_STRIDE)
assert path_prediction.shape == (1, length // module.HORIZONTAL_STRIDE)
assert torch.isfinite(logits).all()
assert torch.isfinite(path_prediction).all()
(logits.square().mean() + path_prediction.square().mean()).backward()
print(
    {
        "logits": list(logits.shape),
        "path": list(path_prediction.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "finite": True,
        "dose": float(diagnostics["dose"].detach()),
    }
)
