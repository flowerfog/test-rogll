"""Zero-training integrity and tail audit for the dense matcher posterior.

Run this in the same live kernel immediately after
``dense_matcher_posterior_path_ceiling_audit_v0_cell.py``.

The previous audit established that a 64-state node menu usually contains a
near-truth state, while its physically connected trajectory oracle is still
6.08 ft pooled.  This cell separates four possible causes:

1. representation or oracle implementation failure;
2. local menu misses;
3. alias branches that are locally good but cannot form one continuous path;
4. a small number of wells dominating pooled SSE.

No model is fitted and Fold 1 remains untouched by parameter selection.
"""

from pathlib import Path
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


REQUIRED_TAIL = {
    "OOF_ROOT",
    "fold1_volumes_dense",
    "selected_mode_dense",
    "outer_prior_dense",
    "oracle_rows_dense",
    "oracle_payloads_dense",
    "anchor_payload_dense",
    "STATE_OFFSETS",
    "N_STATES",
    "ZERO_STATE_INDEX_DENSE",
    "_emission_d0",
    "_path_payload_d0",
    "_summarise_paths_d0",
    "_write_json",
    "_json_default",
}
missing_tail = sorted(REQUIRED_TAIL.difference(globals()))
if missing_tail:
    raise RuntimeError(
        "Run dense_matcher_posterior_path_ceiling_audit_v0_cell.py in this "
        "live kernel first; missing: " + repr(missing_tail)
    )


EXPERIMENT_TAIL = "dense_posterior_oracle_tail_decomposition_v1"
OUT_DIR_TAIL = OOF_ROOT / EXPERIMENT_TAIL
OUT_DIR_TAIL.mkdir(parents=True, exist_ok=True)

TOPK_TAIL = (8, 16, 32, 64)
CAP_GRID_TAIL = (8, 12, 16, 24, 32, 48, 80, 128, 256)
FULL_REPRESENTATION_RMSE_GATE = 0.75
FULL_REPRESENTATION_P95_GATE = 1.50
UNCONSTRAINED_K64_GATE = 5.0
CONNECTED_K64_GATE = 5.0


def _rmse_tail(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def _valid_candidates_tail(volume, mode, topk):
    logits = _emission_d0(volume, mode)
    candidates = []
    valid_index = np.flatnonzero(volume.state_valid)
    for node in range(len(logits)):
        if topk >= len(valid_index):
            index = valid_index.copy()
        else:
            local = np.argpartition(logits[node, valid_index], -topk)[-topk:]
            index = valid_index[local]
        candidates.append(np.asarray(index, np.int16))
    candidates[0] = np.asarray([ZERO_STATE_INDEX_DENSE], np.int16)
    return candidates, logits


def _nearest_node_path_tail(volume, mode, topk):
    candidates, logits = _valid_candidates_tail(volume, mode, topk)
    path = np.empty(len(candidates), np.float64)
    nearest_error = np.empty(len(candidates), np.float64)
    truth_rank = np.empty(len(candidates), np.int32)
    for node, index in enumerate(candidates):
        offsets = STATE_OFFSETS[index].astype(np.float64)
        best = int(np.argmin(np.abs(offsets - volume.truth_offset[node])))
        path[node] = offsets[best]
        nearest_error[node] = abs(path[node] - volume.truth_offset[node])

        truth_state = int(
            np.clip(
                np.rint(volume.truth_offset[node] - float(STATE_OFFSETS[0])),
                0,
                N_STATES - 1,
            )
        )
        truth_rank[node] = 1 + int(np.sum(logits[node] > logits[node, truth_state]))
    path[0] = 0.0
    nearest_error[0] = abs(float(volume.truth_offset[0]))
    return path, nearest_error, truth_rank, candidates


def _transition_residual_tail(volume, path, prior):
    delta_md = np.diff(volume.node_md.astype(np.float64))
    delta_z = np.diff(volume.node_z.astype(np.float64))
    delta_offset = np.diff(np.asarray(path, np.float64))
    return delta_offset + delta_z - float(prior.slope_centre) * delta_md


def _sparse_oracle_tail(volume, mode, prior, topk, residual_cap):
    candidates, _ = _valid_candidates_tail(volume, mode, topk)
    back = [None]
    previous = candidates[0]
    previous_cost = np.zeros(len(previous), np.float64)

    for node in range(1, len(candidates)):
        current = candidates[node]
        previous_offset = STATE_OFFSETS[previous].astype(np.float64)
        current_offset = STATE_OFFSETS[current].astype(np.float64)
        delta_md = float(volume.node_md[node] - volume.node_md[node - 1])
        delta_z = float(volume.node_z[node] - volume.node_z[node - 1])
        residual = (
            current_offset[:, None]
            - previous_offset[None, :]
            + delta_z
            - float(prior.slope_centre) * delta_md
        )
        legal = np.abs(residual) <= float(residual_cap) + 1e-9
        unary = (current_offset - float(volume.truth_offset[node])) ** 2
        transition = 0.01 * (residual / float(prior.residual_scale)) ** 2
        cost = previous_cost[None, :] + transition + unary[:, None]
        cost[~legal] = np.inf
        predecessor = np.argmin(cost, axis=1)
        current_cost = cost[np.arange(len(current)), predecessor]
        if not np.isfinite(current_cost).any():
            return None
        back.append(predecessor.astype(np.int16))
        previous = current
        previous_cost = current_cost

    selected = np.empty(len(candidates), np.int16)
    position = int(np.argmin(previous_cost))
    selected[-1] = candidates[-1][position]
    for node in range(len(candidates) - 1, 0, -1):
        position = int(back[node][position])
        selected[node - 1] = candidates[node - 1][position]
    if int(selected[0]) != int(ZERO_STATE_INDEX_DENSE):
        raise RuntimeError(f"{volume.well}: sparse oracle lost its anchor")
    return STATE_OFFSETS[selected].astype(np.float64)


def _max_false_run_tail(hit):
    miss = ~np.asarray(hit, bool)
    longest = 0
    current = 0
    for value in miss:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def _bywell_from_payload_tail(payload, prefix):
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


def _concentration_tail(bywell, sse_column, rows_column):
    ordered = bywell.sort_values(sse_column, ascending=False).reset_index(drop=True)
    total_sse = float(ordered[sse_column].sum())
    output = []
    for removed in (0, 1, 3, 5, 10, 15, 20, 25, 40):
        remaining = ordered.iloc[removed:]
        remaining_sse = float(remaining[sse_column].sum())
        remaining_rows = int(remaining[rows_column].sum())
        output.append(
            {
                "removed_worst_wells": removed,
                "remaining_wells": len(remaining),
                "remaining_rmse": (
                    float(np.sqrt(remaining_sse / remaining_rows))
                    if remaining_rows
                    else np.nan
                ),
                "removed_share_of_sse": (
                    1.0 - remaining_sse / total_sse if total_sse > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(output)


started_tail = time.time()
mode_tail = selected_mode_dense
prior_tail = outer_prior_dense

unconstrained_payloads = {topk: [] for topk in TOPK_TAIL}
full_payloads = []
first_connected_payloads = []
diagnostic_rows = []

for well in tqdm(sorted(fold1_volumes_dense), desc="decompose dense posterior tail"):
    volume = fold1_volumes_dense[well]
    per_topk = {}
    for topk in TOPK_TAIL:
        path, nearest_error, truth_rank, candidates = _nearest_node_path_tail(
            volume, mode_tail, topk
        )
        unconstrained_payloads[topk].append(_path_payload_d0(volume, path))
        per_topk[topk] = (path, nearest_error, truth_rank, candidates)

    full_path, full_error, _, _ = _nearest_node_path_tail(
        volume, mode_tail, N_STATES
    )
    full_payloads.append(_path_payload_d0(volume, full_path))

    first_cap = None
    connected_path = None
    for cap in CAP_GRID_TAIL:
        connected_path = _sparse_oracle_tail(
            volume, mode_tail, prior_tail, 64, cap
        )
        if connected_path is not None:
            first_cap = cap
            break
    if connected_path is None:
        raise RuntimeError(f"{well}: K64 graph is disconnected even at cap 256")
    first_connected_payloads.append(_path_payload_d0(volume, connected_path))

    k64_path, k64_error, k64_rank, _ = per_topk[64]
    k64_residual = np.abs(_transition_residual_tail(volume, k64_path, prior_tail))
    full_residual = np.abs(_transition_residual_tail(volume, full_path, prior_tail))
    hit5 = k64_error <= 5.0
    diagnostic_rows.append(
        {
            "well": well,
            "nodes": len(volume.node_md),
            "span_ft": float(volume.node_md[-1] - volume.node_md[0]),
            "axis_coverage": float(np.mean(full_error <= 0.51)),
            "full_node_error_p95": float(np.quantile(full_error, 0.95)),
            "full_transition_residual_p95": float(np.quantile(full_residual, 0.95)),
            "full_transition_residual_max": float(np.max(full_residual)),
            "k64_within1": float(np.mean(k64_error <= 1.0)),
            "k64_within3": float(np.mean(k64_error <= 3.0)),
            "k64_within5": float(np.mean(hit5)),
            "k64_max_consecutive_miss_nodes": _max_false_run_tail(hit5),
            "k64_node_error_p50": float(np.median(k64_error)),
            "k64_node_error_p95": float(np.quantile(k64_error, 0.95)),
            "truth_rank_p50": float(np.median(k64_rank)),
            "truth_rank_p90": float(np.quantile(k64_rank, 0.90)),
            "nearest_k64_transition_residual_p95": float(np.quantile(k64_residual, 0.95)),
            "nearest_k64_transition_residual_max": float(np.max(k64_residual)),
            "first_connected_cap_ft": int(first_cap),
        }
    )

diagnostics_tail = pd.DataFrame(diagnostic_rows)
diagnostics_tail.to_csv(OUT_DIR_TAIL / "fold1_well_connectivity_diagnostics.csv", index=False)

unconstrained_metrics = []
unconstrained_bywell = {}
for topk in TOPK_TAIL:
    metrics, _, payload = _summarise_paths_d0(unconstrained_payloads[topk])
    metrics["topk"] = topk
    unconstrained_metrics.append(metrics)
    unconstrained_bywell[topk] = _bywell_from_payload_tail(
        payload, f"k{topk}_unconstrained"
    )

full_metrics, _, full_payload = _summarise_paths_d0(full_payloads)
full_bywell = _bywell_from_payload_tail(full_payload, "full_nearest")
connected_metrics, _, connected_payload = _summarise_paths_d0(first_connected_payloads)
connected_bywell = _bywell_from_payload_tail(
    connected_payload, "k64_first_connected"
)
historical_k64_bywell = _bywell_from_payload_tail(
    oracle_payloads_dense[64], "k64_historical_connected"
)
anchor_bywell_tail = _bywell_from_payload_tail(anchor_payload_dense, "anchor")

bywell_tail = diagnostics_tail
for frame in (
    anchor_bywell_tail,
    full_bywell,
    unconstrained_bywell[8],
    unconstrained_bywell[16],
    unconstrained_bywell[32],
    unconstrained_bywell[64],
    connected_bywell,
    historical_k64_bywell,
):
    bywell_tail = bywell_tail.merge(frame, on="well", how="left", validate="one_to_one")

bywell_tail["historical_k64_sse_share"] = (
    bywell_tail["k64_historical_connected_sse"]
    / bywell_tail["k64_historical_connected_sse"].sum()
)
bywell_tail = bywell_tail.sort_values(
    "k64_historical_connected_sse", ascending=False
).reset_index(drop=True)
bywell_tail.to_csv(OUT_DIR_TAIL / "fold1_oracle_tail_bywell.csv", index=False)
bywell_tail.head(40).to_csv(OUT_DIR_TAIL / "fold1_worst40_oracle_wells.csv", index=False)

unconstrained_report = pd.DataFrame(unconstrained_metrics).sort_values("topk")
unconstrained_report.to_csv(
    OUT_DIR_TAIL / "fold1_unconstrained_topk_oracles.csv", index=False
)
concentration_report = _concentration_tail(
    bywell_tail,
    "k64_historical_connected_sse",
    "k64_historical_connected_rows",
)
concentration_report.to_csv(
    OUT_DIR_TAIL / "fold1_k64_tail_concentration.csv", index=False
)

cap_distribution = (
    diagnostics_tail.groupby("first_connected_cap_ft")
    .agg(wells=("well", "size"), median_k64_coverage=("k64_within5", "median"))
    .reset_index()
)
cap_distribution["fraction"] = cap_distribution["wells"] / len(diagnostics_tail)
cap_distribution.to_csv(
    OUT_DIR_TAIL / "fold1_k64_first_connected_cap_distribution.csv", index=False
)

historical_k64_metrics = next(
    row for row in oracle_rows_dense if int(row["topk"]) == 64
)
unconstrained_k64_metrics = next(
    row for row in unconstrained_metrics if int(row["topk"]) == 64
)
integrity_pass = bool(
    full_metrics["rmse"] <= FULL_REPRESENTATION_RMSE_GATE
    and full_metrics["p95_well_rmse"] <= FULL_REPRESENTATION_P95_GATE
)
unconstrained_menu_pass = bool(
    unconstrained_k64_metrics["rmse"] <= UNCONSTRAINED_K64_GATE
)
connected_menu_pass = bool(
    historical_k64_metrics["rmse"] <= CONNECTED_K64_GATE
)

if not integrity_pass:
    bottleneck = "representation_or_oracle_integrity"
    next_step = "FAIL INTEGRITY -> fix the state/oracle implementation before inference"
elif unconstrained_menu_pass and not connected_menu_pass:
    bottleneck = "alias_branch_connectivity"
    next_step = (
        "LOCAL MENU PASS / CONNECTIVITY FAIL -> only a branch-consistent global "
        "mode proposal audit is justified; do not train another local matcher"
    )
elif not unconstrained_menu_pass:
    bottleneck = "local_menu_recall"
    next_step = (
        "LOCAL MENU FAIL -> close this posterior and do not train a CRF on it"
    )
else:
    bottleneck = "mixed_tail_and_selector"
    next_step = "HOLD -> inspect worst40 before authorising any path model"

verdict_tail = {
    "experiment": EXPERIMENT_TAIL,
    "stage": "zero_training_oracle_integrity_connectivity_and_tail_decomposition",
    "source": str(OOF_ROOT / "dense_matcher_posterior_path_ceiling_audit_v0"),
    "mode": mode_tail,
    "wells": len(fold1_volumes_dense),
    "full_state_nearest_oracle": full_metrics,
    "unconstrained_topk_oracles": unconstrained_metrics,
    "k64_first_connected_oracle": connected_metrics,
    "k64_historical_connected_oracle": historical_k64_metrics,
    "first_connected_cap_distribution": cap_distribution.to_dict(orient="records"),
    "tail_concentration": concentration_report.to_dict(orient="records"),
    "gates": {
        "full_state_rmse_le": FULL_REPRESENTATION_RMSE_GATE,
        "full_state_p95_le": FULL_REPRESENTATION_P95_GATE,
        "unconstrained_k64_rmse_le": UNCONSTRAINED_K64_GATE,
        "connected_k64_rmse_le": CONNECTED_K64_GATE,
    },
    "integrity_pass": integrity_pass,
    "unconstrained_menu_pass": unconstrained_menu_pass,
    "connected_menu_pass": connected_menu_pass,
    "bottleneck": bottleneck,
    "strict_pass": bool(integrity_pass and connected_menu_pass),
    "next": next_step,
    "elapsed_minutes": (time.time() - started_tail) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
}
_write_json(OUT_DIR_TAIL / "verdict.json", verdict_tail)

print("\nUNCONSTRAINED TOPK ORACLES")
print(unconstrained_report.to_string(index=False), flush=True)
print("\nFULL-STATE REPRESENTATION ORACLE")
print(json.dumps(full_metrics, indent=2, default=_json_default), flush=True)
print("\nK64 FIRST-CONNECTED CAP DISTRIBUTION")
print(cap_distribution.to_string(index=False), flush=True)
print("\nK64 HISTORICAL TAIL CONCENTRATION")
print(concentration_report.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict_tail, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR_TAIL, flush=True)
