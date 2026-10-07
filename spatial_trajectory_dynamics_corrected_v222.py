"""V222: corrected, seed-matched rerun of V221 trajectory dynamics.

Corrections only:
1. Keep the exact V217 LightGBM seed for a valid feature-only A/B.
2. Do not let the score-suffix filter reject ``likpf_scale_*`` channels.
"""

from pathlib import Path


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = ROOT / "spatial_trajectory_dynamics_gbdt_v221.py"
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

source = SOURCE.read_text(encoding="utf-8")

old_seed = "source = source.replace('SEED = 217', 'SEED = 221', 1)"
new_seed = "source = source.replace('SEED = 217', 'SEED = 217', 1)"
if source.count(old_seed) != 1:
    raise RuntimeError("V222 could not locate the V221 seed change")
source = source.replace(old_seed, new_seed, 1)

old_filter = '''    blocked = ("rmse", "_std", "dist", "bias", "trust", "score", "_sc", "unc", "entropy")
    if any(token in lowered for token in blocked):
        return False
'''
new_filter = '''    blocked = ("rmse", "_std", "dist", "bias", "trust", "score", "unc", "entropy")
    if any(token in lowered for token in blocked) or lowered.endswith("_sc"):
        return False
'''
if source.count(old_filter) != 1:
    raise RuntimeError("V222 could not locate the V221 channel filter")
source = source.replace(old_filter, new_filter, 1)

old_guard = '''trajectory_channels = trajectory_channels[:36]
if len(trajectory_channels) < 16:
    raise RuntimeError(f"Only {len(trajectory_channels)} trajectory channels found")
'''
new_guard = '''trajectory_channels = trajectory_channels[:36]
_likpf_scale_channels = [name for name in trajectory_channels if name.lower().startswith("likpf_scale_")]
if len(trajectory_channels) < 16 or len(_likpf_scale_channels) < 4:
    raise RuntimeError({
        "trajectory_channels": trajectory_channels,
        "likpf_scale_channels": _likpf_scale_channels,
    })
'''
if source.count(old_guard) != 1:
    raise RuntimeError("V222 could not locate the V221 channel guard")
source = source.replace(old_guard, new_guard, 1)

source = source.replace(
    "spatial_trajectory_dynamics_gbdt_v221",
    "spatial_trajectory_dynamics_corrected_v222",
)
source = source.replace("V221", "V222").replace("v221", "v222")

exec(compile(source, str(SOURCE), "exec"), globals())
