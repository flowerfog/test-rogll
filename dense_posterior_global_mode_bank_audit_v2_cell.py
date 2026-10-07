"""Whole-well global-mode proposal audit for the dense matcher posterior.

Run in the same live kernel after:

1. prefix_contrastive_matcher_adaptation_gateway_v1_cell.py
2. dense_matcher_posterior_path_ceiling_audit_v0_cell.py
3. dense_posterior_oracle_tail_decomposition_v1_cell.py

The local K64 menu has a 4.10-ft unconstrained oracle but a 6.08-ft connected
oracle.  This cell asks the remaining narrow question: can the frozen posterior
produce a small bank of complete, physically coherent modes whose whole-well
oracle reaches 5.x?  Candidate generation is label-free.  Fold-1 truth is used
only after the bank is frozen to measure its ceiling.

No model is trained and no local matcher parameters are changed.
"""

import json
import math
import shutil
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


REQUIRED_MODE_BANK = {
    "OOF_ROOT",
    "fold1_volumes_dense",
    "selected_mode_dense",
    "selected_lambda_dense",
    "outer_prior_dense",
    "STATE_OFFSETS",
    "N_STATES",
    "ZERO_STATE_INDEX_DENSE",
    "_emission_d0",
    "_shift_slices_d0",
    "_path_payload_d0",
    "_summarise_paths_d0",
    "_write_json",
    "_json_default",
    "verdict_tail",
}
missing_mode_bank = sorted(REQUIRED_MODE_BANK.difference(globals()))
if missing_mode_bank:
    raise RuntimeError(
        "Run the dense posterior audit and tail decomposition in this live "
        "kernel first; missing: " + repr(missing_mode_bank)
    )


EXPERIMENT_MODE_BANK = "dense_posterior_global_mode_bank_audit_v2"
OUT_DIR_MODE_BANK = OOF_ROOT / EXPERIMENT_MODE_BANK
OUT_DIR_MODE_BANK.mkdir(parents=True, exist_ok=True)

MODE_CONFIGS = (
    ("adapted", 1.00),
    ("adapted", 0.30),
    ("adapted", 3.00),
    ("adapted", 0.10),
    ("frozen", 1.00),
    ("frozen", 0.30),
    ("frozen", 3.00),
    ("frozen", 0.10),
)
SUPPRESSION_ROUNDS = 8
SUPPRESSION_SIGMA_FT = 4.0
SUPPRESSION_QUANTILE = 0.75
BUDGETS = (1, 2, 4, 8, 16, 32, 64)

MODE64_ORACLE_GATE = 5.0
MODE32_ORACLE_GATE = 5.5
MODE64_P95_GATE = 12.0
MIN_UNIQUE_MODES_GATE = 16


def _rmse_mode_bank(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def _decode_mode_bank(volume, emission, prior, transition_lambda):
    emission = np.asarray(emission, np.float64)
    nodes, states = emission.shape
    dp = np.full(states, -np.inf, np.float64)
    dp[ZERO_STATE_INDEX_DENSE] = 0.0
    back = np.full((nodes, states), -1, np.int16)

    for node in range(1, nodes):
        delta_md = float(volume.node_md[node] - volume.node_md[node - 1])
        delta_z = float(volume.node_z[node] - volume.node_z[node - 1])
        expected_delta_tvt = float(prior.slope_centre) * delta_md - delta_z
        centre_shift = int(round(expected_delta_tvt))
        new = np.full(states, -np.inf, np.float64)
        new_back = np.full(states, -1, np.int16)
        for shift in range(
            centre_shift - int(prior.residual_cap),
            centre_shift + int(prior.residual_cap) + 1,
        ):
            if abs(shift) >= states:
                continue
            source, target = _shift_slices_d0(shift, states)
            residual_s = (
                shift + delta_z - float(prior.slope_centre) * delta_md
            )
            penalty = float(transition_lambda) * (
                residual_s / float(prior.residual_scale)
            ) ** 2
            candidate = dp[source] - penalty + emission[node, target]
            better = candidate > new[target]
            if np.any(better):
                target_index = np.arange(states)[target][better]
                source_index = target_index - shift
                new[target_index] = candidate[better]
                new_back[target_index] = source_index.astype(np.int16)
        new[~volume.state_valid] = -np.inf
        if not np.isfinite(new).any():
            return None
        dp = new
        back[node] = new_back

    state_path = np.empty(nodes, np.int16)
    state_path[-1] = int(np.argmax(dp))
    for node in range(nodes - 1, 0, -1):
        previous = int(back[node, state_path[node]])
        if previous < 0:
            return None
        state_path[node - 1] = previous
    if int(state_path[0]) != int(ZERO_STATE_INDEX_DENSE):
        return None
    return state_path, float(np.max(dp))


def _suppression_strength_mode_bank(base_emission, state_valid):
    valid = base_emission[:, state_valid]
    top = np.max(valid, axis=1)
    reference = np.quantile(valid, SUPPRESSION_QUANTILE, axis=1)
    strength = np.maximum(top - reference, 0.25)
    return strength.astype(np.float64)


def _suppress_path_mode_bank(emission, state_path, row_strength):
    offsets = STATE_OFFSETS.astype(np.float64)
    selected = STATE_OFFSETS[state_path].astype(np.float64)
    distance = offsets[None, :] - selected[:, None]
    kernel = np.exp(-0.5 * (distance / SUPPRESSION_SIGMA_FT) ** 2)
    emission -= row_strength[:, None] * kernel
    return emission


def _common_energy_mode_bank(volume, state_path, prior):
    emission = _emission_d0(volume, selected_mode_dense)
    node = np.arange(len(state_path))
    emission_term = float(np.mean(emission[node[1:], state_path[1:]]))
    offset = STATE_OFFSETS[state_path].astype(np.float64)
    delta_md = np.diff(volume.node_md.astype(np.float64))
    delta_z = np.diff(volume.node_z.astype(np.float64))
    residual = (
        np.diff(offset) + delta_z - float(prior.slope_centre) * delta_md
    )
    transition_term = float(
        np.mean((residual / float(prior.residual_scale)) ** 2)
    )
    return emission_term - float(selected_lambda_dense) * transition_term


def _row_arrays_mode_bank(volume, offset_path):
    payload = _path_payload_d0(volume, offset_path)
    return (
        payload,
        payload["truth"].astype(np.float64),
        payload["prediction"].astype(np.float64),
    )


def _bank_for_well_mode_bank(volume):
    work = {}
    strength = {}
    for mode, transition_lambda in MODE_CONFIGS:
        key = (mode, float(transition_lambda))
        base = _emission_d0(volume, mode).astype(np.float64)
        work[key] = base.copy()
        strength[key] = _suppression_strength_mode_bank(base, volume.state_valid)

    candidates = []
    signatures = set()

    anchor_state = np.full(
        len(volume.node_md), ZERO_STATE_INDEX_DENSE, dtype=np.int16
    )
    anchor_signature = anchor_state.tobytes()
    signatures.add(anchor_signature)
    candidates.append(
        {
            "source_mode": "anchor",
            "transition_lambda": 0.0,
            "round": -1,
            "state_path": anchor_state,
            "decoder_score": np.nan,
            "common_energy": _common_energy_mode_bank(
                volume, anchor_state, outer_prior_dense
            ),
        }
    )

    for round_index in range(SUPPRESSION_ROUNDS):
        for mode, transition_lambda in MODE_CONFIGS:
            key = (mode, float(transition_lambda))
            decoded = _decode_mode_bank(
                volume,
                work[key],
                outer_prior_dense,
                transition_lambda,
            )
            if decoded is None:
                continue
            state_path, decoder_score = decoded
            signature = state_path.tobytes()
            if signature not in signatures:
                signatures.add(signature)
                candidates.append(
                    {
                        "source_mode": mode,
                        "transition_lambda": float(transition_lambda),
                        "round": int(round_index),
                        "state_path": state_path.copy(),
                        "decoder_score": float(decoder_score),
                        "common_energy": _common_energy_mode_bank(
                            volume, state_path, outer_prior_dense
                        ),
                    }
                )
            work[key] = _suppress_path_mode_bank(
                work[key], state_path, strength[key]
            )
    return candidates


def _metrics_bywell_mode_bank(payload, prefix):
    frame = pd.DataFrame(
        {
            "well": payload["well"].astype(str),
            "truth": payload["truth"].astype(np.float64),
            "prediction": payload["prediction"].astype(np.float64),
            "distance": payload["distance"].astype(np.float64),
        }
    )
    rows = []
    for well, group in frame.groupby("well", sort=False):
        error = group["truth"].to_numpy() - group["prediction"].to_numpy()
        far = group["distance"].to_numpy() > 2000.0
        rows.append(
            {
                "well": well,
                f"{prefix}_rows": len(group),
                f"{prefix}_sse": float(np.sum(error**2)),
                f"{prefix}_rmse": float(np.sqrt(np.mean(error**2))),
                f"{prefix}_far_rmse": (
                    float(np.sqrt(np.mean(error[far] ** 2))) if np.any(far) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


started_mode_bank = time.time()
oracle_payloads_by_budget = {budget: [] for budget in BUDGETS}
score_selected_payloads = []
bank_rows = []
well_rows = []

for well in tqdm(sorted(fold1_volumes_dense), desc="global posterior mode banks"):
    volume = fold1_volumes_dense[well]
    candidates = _bank_for_well_mode_bank(volume)
    candidate_payload = []
    candidate_sse = []
    candidate_rmse = []

    for candidate_index, candidate in enumerate(candidates):
        offset = STATE_OFFSETS[candidate["state_path"]].astype(np.float64)
        payload, truth, prediction = _row_arrays_mode_bank(volume, offset)
        error = truth - prediction
        sse = float(np.sum(error**2))
        rmse = float(np.sqrt(np.mean(error**2)))
        candidate_payload.append(payload)
        candidate_sse.append(sse)
        candidate_rmse.append(rmse)
        bank_rows.append(
            {
                "well": well,
                "candidate_index": candidate_index,
                "source_mode": candidate["source_mode"],
                "transition_lambda": candidate["transition_lambda"],
                "round": candidate["round"],
                "decoder_score": candidate["decoder_score"],
                "common_energy": candidate["common_energy"],
                "truth_rmse_diagnostic": rmse,
                "mean_offset": float(np.mean(offset)),
                "end_offset": float(offset[-1]),
                "offset_span": float(np.max(offset) - np.min(offset)),
            }
        )

    candidate_sse = np.asarray(candidate_sse, np.float64)
    common_energy = np.asarray(
        [candidate["common_energy"] for candidate in candidates], np.float64
    )
    for budget in BUDGETS:
        stop = min(int(budget), len(candidates))
        selected = int(np.argmin(candidate_sse[:stop]))
        oracle_payloads_by_budget[budget].append(candidate_payload[selected])

    score_selected = int(np.argmax(common_energy))
    score_selected_payloads.append(candidate_payload[score_selected])
    oracle64_stop = min(64, len(candidates))
    oracle64_index = int(np.argmin(candidate_sse[:oracle64_stop]))
    well_rows.append(
        {
            "well": well,
            "unique_modes": len(candidates),
            "oracle64_index": oracle64_index,
            "oracle64_rmse": candidate_rmse[oracle64_index],
            "score_selected_index": score_selected,
            "score_selected_rmse": candidate_rmse[score_selected],
            "oracle64_source_mode": candidates[oracle64_index]["source_mode"],
            "oracle64_transition_lambda": candidates[oracle64_index][
                "transition_lambda"
            ],
            "oracle64_round": candidates[oracle64_index]["round"],
        }
    )

bank_frame = pd.DataFrame(bank_rows)
well_frame = pd.DataFrame(well_rows)
bank_frame.to_csv(OUT_DIR_MODE_BANK / "fold1_global_mode_bank.csv", index=False)

budget_rows = []
budget_payload = {}
for budget in BUDGETS:
    metrics, bywell, payload = _summarise_paths_d0(
        oracle_payloads_by_budget[budget]
    )
    metrics["budget"] = int(budget)
    budget_rows.append(metrics)
    budget_payload[budget] = payload
    bywell.to_csv(
        OUT_DIR_MODE_BANK / f"fold1_mode_oracle_b{budget}_bywell.csv",
        index=False,
    )

budget_report = pd.DataFrame(budget_rows).sort_values("budget")
budget_report.to_csv(
    OUT_DIR_MODE_BANK / "fold1_global_mode_oracle_curve.csv", index=False
)
score_metrics, score_bywell, score_payload = _summarise_paths_d0(
    score_selected_payloads
)
score_bywell.to_csv(
    OUT_DIR_MODE_BANK / "fold1_common_energy_selector_bywell.csv", index=False
)

oracle64_bywell = _metrics_bywell_mode_bank(budget_payload[64], "mode64_oracle")
score_bywell_detailed = _metrics_bywell_mode_bank(
    score_payload, "common_energy_selector"
)
well_frame = well_frame.merge(
    oracle64_bywell, on="well", how="left", validate="one_to_one"
).merge(
    score_bywell_detailed, on="well", how="left", validate="one_to_one"
)
well_frame = well_frame.sort_values("mode64_oracle_sse", ascending=False)
well_frame.to_csv(OUT_DIR_MODE_BANK / "fold1_global_mode_bywell.csv", index=False)

mode64_metrics = next(row for row in budget_rows if row["budget"] == 64)
mode32_metrics = next(row for row in budget_rows if row["budget"] == 32)
median_unique_modes = float(well_frame["unique_modes"].median())

gates_mode_bank = {
    "mode32_oracle": mode32_metrics["rmse"] <= MODE32_ORACLE_GATE,
    "mode64_oracle": mode64_metrics["rmse"] <= MODE64_ORACLE_GATE,
    "mode64_p95": mode64_metrics["p95_well_rmse"] <= MODE64_P95_GATE,
    "mode_diversity": median_unique_modes >= MIN_UNIQUE_MODES_GATE,
}
menu_pass_mode_bank = bool(
    gates_mode_bank["mode64_oracle"]
    and gates_mode_bank["mode64_p95"]
    and gates_mode_bank["mode_diversity"]
)

verdict_mode_bank = {
    "experiment": EXPERIMENT_MODE_BANK,
    "stage": "label_free_complete_well_global_mode_proposal_ceiling",
    "source": str(OOF_ROOT / "dense_posterior_oracle_tail_decomposition_v1"),
    "candidate_generation": {
        "configs": [
            {"mode": mode, "transition_lambda": transition_lambda}
            for mode, transition_lambda in MODE_CONFIGS
        ],
        "suppression_rounds": SUPPRESSION_ROUNDS,
        "suppression_sigma_ft": SUPPRESSION_SIGMA_FT,
        "maximum_generated_modes_plus_anchor": (
            len(MODE_CONFIGS) * SUPPRESSION_ROUNDS + 1
        ),
        "median_unique_modes": median_unique_modes,
        "uses_truth_during_generation": False,
    },
    "oracle_curve": budget_rows,
    "common_energy_selector": score_metrics,
    "references": {
        "local_k64_unconstrained_oracle_rmse": verdict_tail[
            "unconstrained_topk_oracles"
        ][-1]["rmse"],
        "local_k64_connected_oracle_rmse": verdict_tail[
            "k64_historical_connected_oracle"
        ]["rmse"],
    },
    "gates": gates_mode_bank,
    "menu_pass": menu_pass_mode_bank,
    "strict_pass": False,
    "next": (
        "MODE MENU PASS -> fit only a nested whole-well mode scorer with abstention"
        if menu_pass_mode_bank
        else (
            "MODE MENU FAIL -> close this matcher posterior completely; its local "
            "recall cannot be organised into a deployable complete-well menu"
        )
    ),
    "elapsed_minutes": (time.time() - started_mode_bank) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
}
_write_json(OUT_DIR_MODE_BANK / "verdict.json", verdict_mode_bank)

np.savez_compressed(
    OUT_DIR_MODE_BANK / "fold1_mode64_oracle_and_selector.npz",
    well=budget_payload[64]["well"],
    truth=budget_payload[64]["truth"],
    distance=budget_payload[64]["distance"],
    mode64_oracle=budget_payload[64]["prediction"],
    common_energy_selector=score_payload["prediction"],
)

print("\nGLOBAL MODE ORACLE CURVE")
print(budget_report.to_string(index=False), flush=True)
print("\nCOMMON-ENERGY SELECTOR")
print(json.dumps(score_metrics, indent=2, default=_json_default), flush=True)
print("\nMODE SOURCE COUNTS")
print(
    bank_frame.groupby(["source_mode", "transition_lambda", "round"])
    .size()
    .rename("paths")
    .reset_index()
    .to_string(index=False),
    flush=True,
)
print("\nFINAL VERDICT")
print(
    json.dumps(
        verdict_mode_bank,
        indent=2,
        ensure_ascii=False,
        default=_json_default,
    ),
    flush=True,
)
print("saved:", OUT_DIR_MODE_BANK, flush=True)
