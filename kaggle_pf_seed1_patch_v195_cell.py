"""Freeze the mainline PF feature build to the V98-confirmed seed-1 recipe.

Run this cell after the notebook's final ``build_well`` definition and before
the cell that calls ``main()``.  It matches the seeding convention used by
V97/V98: CRC32(well id) + 1_000_003 * seed, modulo 2^31-1.
"""

from pathlib import Path
import zlib

import numpy as np
from numba import njit


PF_RECIPE_SEED_V195 = 1
PF_SEED_MODULUS_V195 = 2_147_483_647

if "build_well" not in globals() or not callable(build_well):
    raise RuntimeError(
        "Run V195 after the final build_well definition and before main()"
    )
if globals().get("PF_SEED1_PATCH_INSTALLED_V195", False):
    raise RuntimeError("V195 seed-1 patch was already installed")


@njit(cache=False)
def _seed_numba_v195(seed):
    np.random.seed(seed)


def _well_id_v195(horizontal_path):
    name = Path(str(horizontal_path)).name
    suffix = "__horizontal_well.csv"
    if not name.endswith(suffix):
        raise RuntimeError(f"Unexpected horizontal-well path: {horizontal_path}")
    well = name[: -len(suffix)]
    if not well:
        raise RuntimeError(f"Cannot parse well id from {horizontal_path}")
    return well


def _stable_seed_v195(well, recipe_seed=PF_RECIPE_SEED_V195):
    token = zlib.crc32(str(well).encode("ascii")) & 0x7FFFFFFF
    return int((token + 1_000_003 * int(recipe_seed)) % PF_SEED_MODULUS_V195)


_build_well_unseeded_v195 = build_well
PF_SEED1_CALLS_V195 = []


def build_well(hw_path, tw_path, is_train, *args, **kwargs):
    well = _well_id_v195(hw_path)
    seed = _stable_seed_v195(well)
    _seed_numba_v195(seed)
    PF_SEED1_CALLS_V195.append(
        {"well": well, "seed": seed, "is_train": bool(is_train)}
    )
    return _build_well_unseeded_v195(
        hw_path, tw_path, is_train, *args, **kwargs
    )


PF_SEED1_PATCH_INSTALLED_V195 = True
print(
    "V195 deterministic PF patch installed",
    {
        "recipe_seed": PF_RECIPE_SEED_V195,
        "placement": "after final build_well definition, before main()",
        "seeding_rule": "crc32(well)+1000003*seed mod 2147483647",
    },
    flush=True,
)
