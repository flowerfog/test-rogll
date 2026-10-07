"""V182 T0 bootstrap: run as the first Kaggle notebook cell."""

import os


SEED_V182 = 20260803
THREADS_V182 = 1

# These must be set before numpy/numba/torch initialize their worker pools.
os.environ["PYTHONHASHSEED"] = str(SEED_V182)
os.environ["OMP_NUM_THREADS"] = str(THREADS_V182)
os.environ["MKL_NUM_THREADS"] = str(THREADS_V182)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_V182)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS_V182)
os.environ["NUMBA_NUM_THREADS"] = str(THREADS_V182)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import numpy as np

random.seed(SEED_V182)
np.random.seed(SEED_V182)

try:
    import numba
    from numba import njit

    numba.set_num_threads(THREADS_V182)

    @njit(cache=False)
    def _seed_numba_v182(seed):
        np.random.seed(seed)

    _seed_numba_v182(SEED_V182)
except Exception as exc:
    print("V182 numba setup unavailable:", repr(exc), flush=True)

try:
    import torch

    torch.manual_seed(SEED_V182)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_V182)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
except Exception as exc:
    print("V182 torch setup unavailable:", repr(exc), flush=True)

DETERMINISM_CONTRACT_V182 = {
    "seed": SEED_V182,
    "threads": THREADS_V182,
    "pythonhashseed_effective_after_process_restart": False,
    "numpy_seeded": True,
    "numba_seeded_inside_njit": "_seed_numba_v182" in globals(),
    "torch_deterministic": "torch" in globals(),
}
print("V182 DETERMINISM CONTRACT", DETERMINISM_CONTRACT_V182, flush=True)

