"""V210 deployment patch for the confirmed V209b 1200-particle PF state.

Run this file in the Kaggle notebook after ``build_well``, ``run_pf_ancc``,
and ``run_pf_z`` have been defined, but immediately before ``main()`` is
called.  It reproduces V209b exactly: seed 1 is fixed per well, ANCC and Z
consume one paired random stream, and only their particle counts change from
600 to 1200.  All later mainline, CP8, Neighbor, V69, and V77 code is left
unchanged.
"""

from __future__ import annotations

import zlib
import threading
from pathlib import Path

import numpy as np
from numba import njit


V210_REQUIRED = {
    "build_well", "run_pf_ancc", "run_pf_z", "ANCC_N", "PF_N",
}
V210_MISSING = sorted(V210_REQUIRED.difference(globals()))
if V210_MISSING:
    raise RuntimeError(
        "Run the mainline definition cells before V210. Missing: "
        + repr(V210_MISSING)
    )

V210_REFERENCE_SEED = 1
V210_MULTIPLIER = 2
V210_DEFAULT_ANCC_N = int(ANCC_N)
V210_DEFAULT_PF_N = int(PF_N)
V210_ANCC_N = V210_DEFAULT_ANCC_N * V210_MULTIPLIER
V210_PF_N = V210_DEFAULT_PF_N * V210_MULTIPLIER

if V210_DEFAULT_ANCC_N != 600 or V210_DEFAULT_PF_N != 600:
    raise RuntimeError(
        "V210 was validated only for 600 -> 1200 particles; found "
        f"ANCC_N={V210_DEFAULT_ANCC_N}, PF_N={V210_DEFAULT_PF_N}"
    )

# Preserve the true originals when a fixed V210 patch is reloaded after an
# earlier V210 wrapper failed in the same notebook kernel.
if "V210_ORIGINAL_BUILD_WELL" not in globals():
    V210_ORIGINAL_BUILD_WELL = build_well
    V210_ORIGINAL_RUN_PF_ANCC = run_pf_ancc
    V210_ORIGINAL_RUN_PF_Z = run_pf_z
V210_THREAD_STATE = threading.local()
V210_COUNT_LOCK = threading.Lock()
V210_BUILD_CALLS = 0
V210_ANCC_CALLS = 0
V210_Z_CALLS = 0


@njit(cache=True)
def _seed_numba_v210(seed):
    np.random.seed(seed)


def _stable_seed_v210(well: str) -> int:
    token = zlib.crc32(str(well).encode("ascii")) & 0x7FFFFFFF
    return int((token + 1000003 * V210_REFERENCE_SEED) % 2147483647)


def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    global V210_ANCC_CALLS
    active_well = getattr(V210_THREAD_STATE, "well", None)
    if active_well is None:
        raise RuntimeError("V210 ANCC called outside the patched build_well")
    if getattr(V210_THREAD_STATE, "cached_z", None) is not None:
        raise RuntimeError(f"{active_well}: stale V210 Z state")
    _seed_numba_v210(_stable_seed_v210(active_well))
    ancc = V210_ORIGINAL_RUN_PF_ANCC(hw, tw_tvt, tw_gr, N=V210_ANCC_N)
    V210_THREAD_STATE.cached_z = V210_ORIGINAL_RUN_PF_Z(
        hw, tw_tvt, tw_gr, N=V210_PF_N
    )
    with V210_COUNT_LOCK:
        V210_ANCC_CALLS += 1
    return ancc


def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    global V210_Z_CALLS
    active_well = getattr(V210_THREAD_STATE, "well", None)
    cached_z = getattr(V210_THREAD_STATE, "cached_z", None)
    if cached_z is None:
        raise RuntimeError(
            f"{active_well}: V210 Z requested before its paired ANCC state"
        )
    V210_THREAD_STATE.cached_z = None
    with V210_COUNT_LOCK:
        V210_Z_CALLS += 1
    return cached_z


def build_well(horizontal_path, typewell_path, *args, **kwargs):
    global V210_BUILD_CALLS
    active_well = getattr(V210_THREAD_STATE, "well", None)
    if active_well is not None:
        raise RuntimeError(f"Nested V210 build_well call for {active_well}")
    V210_THREAD_STATE.well = Path(horizontal_path).name.split("__", 1)[0]
    V210_THREAD_STATE.cached_z = None
    try:
        output = V210_ORIGINAL_BUILD_WELL(
            horizontal_path, typewell_path, *args, **kwargs
        )
        if getattr(V210_THREAD_STATE, "cached_z", None) is not None:
            raise RuntimeError(
                f"{V210_THREAD_STATE.well}: paired Z state was not consumed"
            )
        with V210_COUNT_LOCK:
            V210_BUILD_CALLS += 1
        return output
    finally:
        V210_THREAD_STATE.well = None
        V210_THREAD_STATE.cached_z = None


def assert_v210_complete():
    counts = {
        "build_well": int(V210_BUILD_CALLS),
        "ancc": int(V210_ANCC_CALLS),
        "z": int(V210_Z_CALLS),
    }
    if not (V210_BUILD_CALLS > 0 and V210_BUILD_CALLS == V210_ANCC_CALLS == V210_Z_CALLS):
        raise RuntimeError(f"Incomplete V210 call lineage: {counts}")
    print("V210 CALL AUDIT", counts, flush=True)
    return counts


print(
    "V210 PATCH ACTIVE",
    {
        "reference_seed": V210_REFERENCE_SEED,
        "ancc_particles": V210_ANCC_N,
        "z_particles": V210_PF_N,
        "paired_random_stream": True,
        "validated_confirmation_wells": 192,
        "validated_core_gain": 0.07688682416404902,
    },
    flush=True,
)
