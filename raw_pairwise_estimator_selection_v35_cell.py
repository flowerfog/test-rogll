"""Select MAP versus posterior mean after V34 evidence-dose calibration.

V34 selected alpha on Fold 3 using posterior mean, but its learned evidence
substantially improved the Fold 2 MAP path while posterior averaging crossed
incompatible modes.  This read-only audit selects the estimator on Fold 3 at
the already-frozen alpha and then reports the untouched Fold 2 result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "raw_pairwise_estimator_selection_v35"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports" if ROOT.exists() else Path.cwd() / "output"
V34_DIR = OOF_ROOT / "raw_pairwise_emission_calibration_v34"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


for required in (
    V34_DIR / "verdict.json",
    V34_DIR / "fold3_alpha_scan.csv",
    V34_DIR / "fold2_summary.csv",
):
    if not required.exists():
        raise FileNotFoundError(required)

v34 = json.loads((V34_DIR / "verdict.json").read_text(encoding="utf-8"))
alpha = float(v34["selected_alpha"])
fold3 = pd.read_csv(V34_DIR / "fold3_alpha_scan.csv")
fold2 = pd.read_csv(V34_DIR / "fold2_summary.csv")
estimators = ("map", "posterior_mean")

selection = fold3[
    np.isclose(fold3["alpha"], alpha)
    & fold3["estimator"].isin(estimators)
].copy()
if len(selection) != 2:
    raise RuntimeError(f"Expected two Fold3 estimator rows at alpha={alpha}: {selection}")
selection = selection.sort_values(["rmse", "far_gt2000_rmse", "estimator"])
selected_estimator = str(selection.iloc[0]["estimator"])

base3 = fold3[
    np.isclose(fold3["alpha"], 0.0)
    & fold3["estimator"].eq("posterior_mean")
].iloc[0]
chosen3 = selection.iloc[0]
base2 = fold2[
    np.isclose(fold2["alpha"], 0.0)
    & fold2["estimator"].eq("posterior_mean")
].iloc[0]
chosen2_rows = fold2[
    np.isclose(fold2["alpha"], alpha)
    & fold2["estimator"].eq(selected_estimator)
]
if len(chosen2_rows) != 1:
    raise RuntimeError(
        f"Fold2 lacks estimator={selected_estimator} at frozen alpha={alpha}"
    )
chosen2 = chosen2_rows.iloc[0]

comparison = pd.DataFrame([
    {"fold": 3, "role": "baseline", **base3.to_dict()},
    {"fold": 3, "role": "selected", **chosen3.to_dict()},
    {"fold": 2, "role": "baseline", **base2.to_dict()},
    {"fold": 2, "role": "selected", **chosen2.to_dict()},
])
comparison.to_csv(OUT_DIR / "matched_comparison.csv", index=False)
print("V35 MATCHED COMPARISON")
print(comparison.to_string(index=False), flush=True)

gates = {
    "fold3_gain_positive": bool(chosen3["rmse"] < base3["rmse"]),
    "fold2_gain_positive": bool(chosen2["rmse"] < base2["rmse"]),
    "fold2_far_gain_positive": bool(
        chosen2["far_gt2000_rmse"] < base2["far_gt2000_rmse"]
    ),
    "fold2_p95_gain_positive": bool(
        chosen2["p95_well_rmse"] < base2["p95_well_rmse"]
    ),
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold3_estimator_selection_at_frozen_alpha_Fold2_read_only_confirmation",
    "read_only": True,
    "raw_inputs_only": True,
    "selected_alpha_from_v34": alpha,
    "selected_estimator": selected_estimator,
    "fold3": {
        "baseline_rmse": float(base3["rmse"]),
        "selected_rmse": float(chosen3["rmse"]),
        "gain": float(base3["rmse"] - chosen3["rmse"]),
    },
    "fold2": {
        "baseline_rmse": float(base2["rmse"]),
        "selected_rmse": float(chosen2["rmse"]),
        "gain": float(base2["rmse"] - chosen2["rmse"]),
        "baseline_far_rmse": float(base2["far_gt2000_rmse"]),
        "selected_far_rmse": float(chosen2["far_gt2000_rmse"]),
        "baseline_p95": float(base2["p95_well_rmse"]),
        "selected_p95": float(chosen2["p95_well_rmse"]),
    },
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "next": (
        "PASS -> retain modal decoding and train a sequence model to discriminate the remaining cycle skips"
        if all(gates.values())
        else "FAIL -> estimator reversal does not transfer; proceed to a trainable sequence encoder"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print(f"saved: {OUT_DIR}", flush=True)
