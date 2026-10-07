# DO NOT APPLY TO THE 6.781 MAINLINE

This draft was mapped against `ultimate-pf-config-strategy-a-reproducible-score.ipynb`,
not against the exact notebook that produced the 6.781 mainline submission. It is
retained only as a reference inventory until the 6.781 source notebook is supplied.
Its cell numbers, upstream blend assumptions, and overwrite order must not be pasted
into the production mainline.

# Hidden-aware PF edits for the 6.56/6.594 reference notebook

This patch targets `ultimate-pf-config-strategy-a-reproducible-score.ipynb`.
Apply the edits in the order below. The three visible test wells are smoke tests
only; they must not be used to disable a conditional module for the hidden run.

## Final pipeline

```text
SP45 = 0.30 * Ridge + 0.70 * selector
SP45 projected = cubic projection of SP45
learned = FLE learned-trajectory branch, including its own 128-seed lik-PF
base blend = 0.60 * projected SP45 + 0.40 * learned
then guarded same-well contact
then balanced visible-prefix calibration
then model-package correction with its per-run p95 guard
then PF seed-branch hedge
then final audit
```

Do not add the A24 `+1 ft` example-well continuation. It is not a hidden-well
inference rule and has no 6.594 score attribution.

## Edit 1: profile cell

Keep this assignment at the start of the profile cell, before `_profile` is
constructed:

```python
SUBMISSION_PROFILE = "vp_balanced_modelpkg_005"
```

Keep these effective settings:

```python
SP45_RIDGE_MODEL_WEIGHT = 0.30
SP45_SELECTOR_WEIGHT = 0.70
SP45_PROJECTION_DEGREE = 3
SP45_BLEND_WEIGHT = 0.60

RUN_GUARDED_OVERLAP_OVERRIDE = True
RUN_VISIBLE_PREFIX_CALIBRATION = True
VISIBLE_PREFIX_PROFILE = "balanced"
VISIBLE_PREFIX_FINAL_SELECTION = "profile"

RUN_MODEL_PACKAGE_CORRECTION = True
MODEL_PACKAGE_GATED_MAX_WEIGHT = 0.00425
MODEL_PACKAGE_GATED_SCALE = 6.0
MODEL_PACKAGE_DIFF_P95_DISABLE = 25.0

LEARNED_LIKPF_GR_SIGMA_MULTIPLIER = 1.30
RUN_PF_SEED_BRANCH_HEDGE = True
```

The model-package p95 guard must remain dynamic. A p95 of `26.701` on the three
visible wells does not establish what the hidden run will do.

## Edit 2: learned/FLE `lik_pf`

This is the `lik_pf` in the learned-trajectory section, not the SP45 selector
`run_particle_filter`.

Replace:

```python
gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10., 60.)) * 1.3
```

with the explicit configurable form:

```python
gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10.0, 60.0))
gs *= float(globals().get("LEARNED_LIKPF_GR_SIGMA_MULTIPLIER", 1.30))
```

This changes only the FLE learned branch's lik-PF. Do not multiply `gs` inside
the SP45 selector PF unless running a separately identified ablation.

## Edit 3: SP45 PF function

Use the notebook's existing `run_pf_lik_ensemble_scales(..., branch_stats=None)`
implementation. It must return exactly the same `pf_scale_3/5/8/12` and
`pf_mean` as before while optionally filling the supplied dictionary.

Do not add a second wrapper that reruns all 128 seeds. The per-well inference
call must look like this:

```python
PF_SEED_BRANCH_STATS = {}

# Inside the existing test-well loop, after tw_ref is assigned:
try:
    _seed_branch = {}
    pf_by_scale = run_pf_lik_ensemble_scales(
        hw_te,
        tw_ref,
        scales=SELECTOR_SCALES,
        n_particles=int(SP45_SELECTOR_N_PARTICLES),
        n_seeds=int(globals().get("SELECTOR_PF_SEEDS", SP45_SELECTOR_N_SEEDS)),
        branch_stats=_seed_branch,
    )
    if _seed_branch:
        PF_SEED_BRANCH_STATS[str(wid)] = _seed_branch
    tvt_pf = pf_by_scale["pf_scale_8"]
except Exception as exc:
    print(f"  PF failed: {exc}")
    PF_SEED_BRANCH_STATS.pop(str(wid), None)
    last_known = hw_te["TVT_input"].dropna()
    last_value = float(last_known.iloc[-1]) if len(last_known) else 0.0
    tvt_pf = hw_te["TVT_input"].fillna(last_value).to_numpy(dtype=float)
    pf_by_scale = {
        f"pf_scale_{float(scale):g}": tvt_pf.copy()
        for scale in SELECTOR_SCALES
    }

# Beam is independent of PF success and must always be assigned.
try:
    tvt_beam = run_beam_ensemble(hw_te, tw_ref)
except Exception as exc:
    print(f"  Beam failed: {exc}")
    tvt_beam = tvt_pf.copy()

last_known = hw_te["TVT_input"].dropna()
last_known_tvt = (
    float(last_known.iloc[-1])
    if len(last_known)
    else float(np.nanmean(tvt_pf))
)
tvt_selector, bimodal_info = apply_selector_variant(
    selector_variant,
    pf_by_scale,
    tvt_beam,
    last_known_tvt,
    hw=hw_te,
    tw=tw_ref,
    return_info=True,
)
```

Delete the old PF/Beam/selector block when inserting this one. Otherwise the PF
will run twice and its runtime will roughly double.

## Edit 4: preserve the base blend

The final blend cell must still read the two distinct files:

```python
_INPUT_FILES = {
    "learned": _WORK / "submission.csv",              # FLE branch
    "sp45": _WORK / "sp45_projection_submission.csv", # projected SP45
}
_SELECTED_SP45_WEIGHT = 0.60
```

and calculate:

```python
tvt = 0.60 * tvt_sp45 + 0.40 * tvt_learned
```

Do not replace `learned` with another SP45 output. That would remove the
cross-engine diversity the blend is intended to retain.

## Edit 5: conditional stages and execution order

Run the existing cells in this exact order:

```text
1. SP45 inference and cubic projection
2. FLE learned inference (`lik_pf` uses multiplier 1.30)
3. 0.60 SP45 + 0.40 learned blend
4. guarded same-well contact override
5. balanced visible-prefix calibration
6. model-package correction and its dynamic p95 guard
7. PF seed-branch hedge
8. final submission audit
```

Do not disable stage 5 because `applied_wells=0` on the three visible wells.
Do not pre-disable stage 6 because its visible-well p95 exceeded 25. Both
decisions are recomputed from the active test population during the hidden run.

## Edit 6: branch hedge switch

At the beginning of the existing branch-hedge cell, add:

```python
if not bool(globals().get("RUN_PF_SEED_BRANCH_HEDGE", True)):
    print("PF seed-branch hedge disabled")
else:
    # Indent the existing branch-hedge body under this branch.
    ...
```

Keep the established production parameters:

```python
_BH_STRENGTH = 0.60
_BH_MIN_MASS = 0.25
_BH_SEP_LOW = 4.0
_BH_SEP_HIGH = 40.0
_BH_CAP = 2.0
_BH_SKIP_EXISTING = False
```

The hedge runs after visible-prefix and model-package processing. It therefore
changes the final active route rather than only the pre-Gold SP45 path.

## Optional audit cell

Place this function after the profile cell. Call `snapshot_submission(...)`
after each stage while testing the visible notebook. These snapshots verify
cell placement, but their three-well activation counts are not hidden-set
evidence.

```python
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

_STAGE_WORK = Path("/kaggle/working")


def snapshot_submission(label):
    source = _STAGE_WORK / "submission.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    if list(frame.columns) != ["id", "tvt"]:
        raise RuntimeError(f"{label}: bad columns {list(frame.columns)}")
    values = pd.to_numeric(frame["tvt"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label}: non-finite TVT")
    target = _STAGE_WORK / f"submission_stage_{label}.csv"
    frame.to_csv(target, index=False)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(
        "STAGE SNAPSHOT",
        {
            "label": label,
            "rows": len(frame),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "sha256": digest,
            "path": str(target),
        },
        flush=True,
    )
```

Suggested calls:

```python
snapshot_submission("learned")
snapshot_submission("sp45_learned_blend")
snapshot_submission("contact")
snapshot_submission("visible_prefix")
snapshot_submission("model_package")
snapshot_submission("branch_hedge_final")
```

Only call a snapshot after the cell named by its label has actually run.

## Final invariants

Before submission, verify:

```python
assert SUBMISSION_PROFILE == "vp_balanced_modelpkg_005"
assert abs(SP45_RIDGE_MODEL_WEIGHT - 0.30) < 1e-12
assert abs(SP45_SELECTOR_WEIGHT - 0.70) < 1e-12
assert SP45_PROJECTION_DEGREE == 3
assert abs(SP45_BLEND_WEIGHT - 0.60) < 1e-12
assert VISIBLE_PREFIX_PROFILE == "balanced"
assert VISIBLE_PREFIX_FINAL_SELECTION == "profile"
assert RUN_VISIBLE_PREFIX_CALIBRATION
assert RUN_MODEL_PACKAGE_CORRECTION
assert abs(LEARNED_LIKPF_GR_SIGMA_MULTIPLIER - 1.30) < 1e-12
assert bool(globals().get("RUN_PF_SEED_BRANCH_HEDGE", True))
assert "PF_SEED_BRANCH_STATS" in globals()
```
