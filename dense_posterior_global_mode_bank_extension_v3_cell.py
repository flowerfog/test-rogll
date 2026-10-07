"""Final saturation audit for the whole-well posterior mode bank.

Run in the same live kernel after
``dense_posterior_global_mode_bank_audit_v2_cell.py``.

The B32 -> B64 oracle improved from 7.07 to 6.00 ft, so the original B64 gate
was not a valid saturation test.  This zero-training extension regenerates the
same deterministic bank with 24 suppression rounds (up to 193 paths including
the anchor) and measures B64/96/128/160/192 whole-well oracle ceilings.

Truth is never used to generate a path.  It is used only to measure the frozen
bank after generation.  No scorer is fitted here.
"""

import json
import shutil
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


REQUIRED_EXTENSION = {
    "OOF_ROOT",
    "fold1_volumes_dense",
    "STATE_OFFSETS",
    "MODE_CONFIGS",
    "_bank_for_well_mode_bank",
    "_path_payload_d0",
    "_summarise_paths_d0",
    "_write_json",
    "_json_default",
    "verdict_mode_bank",
}
missing_extension = sorted(REQUIRED_EXTENSION.difference(globals()))
if missing_extension:
    raise RuntimeError(
        "Run dense_posterior_global_mode_bank_audit_v2_cell.py in this live "
        "kernel first; missing: " + repr(missing_extension)
    )


EXPERIMENT_EXTENSION = "dense_posterior_global_mode_bank_extension_v3"
OUT_DIR_EXTENSION = OOF_ROOT / EXPERIMENT_EXTENSION
OUT_DIR_EXTENSION.mkdir(parents=True, exist_ok=True)

SUPPRESSION_ROUNDS_ORIGINAL_EXTENSION = int(SUPPRESSION_ROUNDS)
SUPPRESSION_ROUNDS = 24
EXTENDED_BUDGETS = (64, 96, 128, 160, 192)

STRONG_ORACLE_GATE_EXTENSION = 5.0
MINIMUM_ORACLE_GATE_EXTENSION = 5.5
P95_GATE_EXTENSION = 10.0
SATURATION_GAIN_GATE_EXTENSION = 0.20
B64_REPRODUCTION_TOLERANCE = 1e-5


def _payload_for_mode_extension(volume, state_path):
    offset = STATE_OFFSETS[state_path].astype(np.float64)
    return _path_payload_d0(volume, offset)


started_extension = time.time()
payloads_by_budget_extension = {
    budget: [] for budget in EXTENDED_BUDGETS
}
well_rows_extension = []
bank_rows_extension = []

for well in tqdm(
    sorted(fold1_volumes_dense), desc="extend complete-well posterior modes"
):
    volume = fold1_volumes_dense[well]
    candidates = _bank_for_well_mode_bank(volume)
    payloads = []
    sse = []
    rmse = []

    for candidate_index, candidate in enumerate(candidates):
        payload = _payload_for_mode_extension(volume, candidate["state_path"])
        truth = payload["truth"].astype(np.float64)
        prediction = payload["prediction"].astype(np.float64)
        error = truth - prediction
        candidate_sse = float(np.sum(error**2))
        candidate_rmse = float(np.sqrt(np.mean(error**2)))
        payloads.append(payload)
        sse.append(candidate_sse)
        rmse.append(candidate_rmse)
        bank_rows_extension.append(
            {
                "well": well,
                "candidate_index": candidate_index,
                "source_mode": candidate["source_mode"],
                "transition_lambda": candidate["transition_lambda"],
                "round": candidate["round"],
                "common_energy": candidate["common_energy"],
                "truth_rmse_diagnostic": candidate_rmse,
            }
        )

    sse = np.asarray(sse, np.float64)
    rmse = np.asarray(rmse, np.float64)
    row = {
        "well": well,
        "unique_modes": len(candidates),
    }
    for budget in EXTENDED_BUDGETS:
        stop = min(int(budget), len(candidates))
        selected = int(np.argmin(sse[:stop]))
        payloads_by_budget_extension[budget].append(payloads[selected])
        row[f"b{budget}_selected_index"] = selected
        row[f"b{budget}_rmse"] = float(rmse[selected])
    well_rows_extension.append(row)

bank_frame_extension = pd.DataFrame(bank_rows_extension)
well_frame_extension = pd.DataFrame(well_rows_extension)
bank_frame_extension.to_csv(
    OUT_DIR_EXTENSION / "fold1_extended_global_mode_bank.csv", index=False
)
well_frame_extension.to_csv(
    OUT_DIR_EXTENSION / "fold1_extended_global_mode_bywell.csv", index=False
)

curve_rows_extension = []
curve_payload_extension = {}
for budget in EXTENDED_BUDGETS:
    metrics, bywell, payload = _summarise_paths_d0(
        payloads_by_budget_extension[budget]
    )
    metrics["budget"] = int(budget)
    curve_rows_extension.append(metrics)
    curve_payload_extension[budget] = payload
    bywell.to_csv(
        OUT_DIR_EXTENSION / f"fold1_extended_mode_oracle_b{budget}_bywell.csv",
        index=False,
    )

curve_extension = pd.DataFrame(curve_rows_extension).sort_values("budget")
curve_extension["gain_from_previous"] = -curve_extension["rmse"].diff()
curve_extension.to_csv(
    OUT_DIR_EXTENSION / "fold1_extended_global_mode_oracle_curve.csv",
    index=False,
)

original_b64 = next(
    row for row in verdict_mode_bank["oracle_curve"] if int(row["budget"]) == 64
)
extended_b64 = next(
    row for row in curve_rows_extension if int(row["budget"]) == 64
)
b64_reproduction_error = abs(
    float(original_b64["rmse"]) - float(extended_b64["rmse"])
)
# A few wells produced fewer than 64 distinct paths in the eight-round run.
# Their extended B64 bank can include a later-round replacement, so a small
# improvement is legitimate.  A degradation still indicates non-determinism.
if float(extended_b64["rmse"]) > float(original_b64["rmse"]) + B64_REPRODUCTION_TOLERANCE:
    raise RuntimeError(
        "The extended bank degraded the frozen B64 oracle: "
        f"old={original_b64['rmse']:.9f}, new={extended_b64['rmse']:.9f}"
    )

best_row_extension = min(curve_rows_extension, key=lambda row: row["rmse"])
last_gain_extension = float(
    curve_extension.iloc[-2]["rmse"] - curve_extension.iloc[-1]["rmse"]
)
strong_oracle_pass_extension = bool(
    best_row_extension["rmse"] <= STRONG_ORACLE_GATE_EXTENSION
    and best_row_extension["p95_well_rmse"] <= P95_GATE_EXTENSION
)
minimum_oracle_pass_extension = bool(
    best_row_extension["rmse"] <= MINIMUM_ORACLE_GATE_EXTENSION
)
saturated_extension = bool(
    last_gain_extension < SATURATION_GAIN_GATE_EXTENSION
)

if strong_oracle_pass_extension:
    next_extension = (
        "EXTENDED MENU PASS -> one nested whole-well scorer with abstention is "
        "authorised; do not alter the candidate generator"
    )
elif minimum_oracle_pass_extension:
    next_extension = (
        "MARGINAL MENU -> selector has no error budget; inspect tail overlap with "
        "the safe anchor before deciding whether to train"
    )
elif saturated_extension:
    next_extension = (
        "SATURATED MENU FAIL -> close this matcher posterior completely"
    )
else:
    next_extension = (
        "UNSATURATED BUT ABOVE GATE -> stop at B192 anyway; the menu is too large "
        "and the common-energy selector is already anti-predictive"
    )

verdict_extension = {
    "experiment": EXPERIMENT_EXTENSION,
    "stage": "final_complete_well_mode_bank_saturation_audit",
    "source": str(OOF_ROOT / "dense_posterior_global_mode_bank_audit_v2"),
    "generation": {
        "suppression_rounds": SUPPRESSION_ROUNDS,
        "configs": len(MODE_CONFIGS),
        "maximum_modes_plus_anchor": SUPPRESSION_ROUNDS * len(MODE_CONFIGS) + 1,
        "median_unique_modes": float(well_frame_extension["unique_modes"].median()),
        "uses_truth_during_generation": False,
    },
    "b64_reproduction_error": b64_reproduction_error,
    "oracle_curve": curve_rows_extension,
    "best_oracle": best_row_extension,
    "last_budget_gain": last_gain_extension,
    "gates": {
        "strong_oracle_rmse_le": STRONG_ORACLE_GATE_EXTENSION,
        "minimum_oracle_rmse_le": MINIMUM_ORACLE_GATE_EXTENSION,
        "p95_le": P95_GATE_EXTENSION,
        "saturation_gain_lt": SATURATION_GAIN_GATE_EXTENSION,
    },
    "strong_oracle_pass": strong_oracle_pass_extension,
    "minimum_oracle_pass": minimum_oracle_pass_extension,
    "saturated": saturated_extension,
    "strict_pass": False,
    "next": next_extension,
    "elapsed_minutes": (time.time() - started_extension) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
}
_write_json(OUT_DIR_EXTENSION / "verdict.json", verdict_extension)

np.savez_compressed(
    OUT_DIR_EXTENSION / "fold1_extended_mode_oracles.npz",
    well=curve_payload_extension[192]["well"],
    truth=curve_payload_extension[192]["truth"],
    distance=curve_payload_extension[192]["distance"],
    b64=curve_payload_extension[64]["prediction"],
    b96=curve_payload_extension[96]["prediction"],
    b128=curve_payload_extension[128]["prediction"],
    b160=curve_payload_extension[160]["prediction"],
    b192=curve_payload_extension[192]["prediction"],
)

SUPPRESSION_ROUNDS = SUPPRESSION_ROUNDS_ORIGINAL_EXTENSION

print("\nEXTENDED GLOBAL MODE ORACLE CURVE")
print(curve_extension.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(
    json.dumps(
        verdict_extension,
        indent=2,
        ensure_ascii=False,
        default=_json_default,
    ),
    flush=True,
)
print("saved:", OUT_DIR_EXTENSION, flush=True)
