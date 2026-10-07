"""Zero-shot application of the complete-well GR energy to a label-free bank.

Run in a kernel containing:
1. the v1 non-rigid observability definitions/checkpoint; and
2. the frozen Fold-1 dense posterior volumes and v3 mode-bank generator.

The energy network was trained on folds 3/4/5 and selected on fold 2. This
cell applies it without fitting to the label-free Fold-1 mode bank. Candidate
generation uses no truth. Truth is read only after selection for reporting.

The checkpoint saw 49-candidate sets. A deterministic anchor-calibrated
tournament therefore reduces a bank of up to 193 paths to a final set of 49:
each qualifying group contains the same anchor plus 48 candidates; candidates
are compared by energy relative to that anchor; group winners enter one final
49-way forward. No validation truth participates in this reduction.
"""

from __future__ import annotations

from pathlib import Path
import ast
import gc
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
import torch


DENSE_LIVE_NAMES_V2 = {
    "MODE_CONFIGS",
    "ZERO_STATE_INDEX_DENSE",
    "_bank_for_well_mode_bank",
    "_emission_d0",
    "_path_payload_d0",
    "_shift_slices_d0",
    "fold1_volumes_dense",
    "outer_prior_dense",
    "selected_lambda_dense",
    "selected_mode_dense",
}


def _assigned_names_v2(node):
    names = set()
    targets = []
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.add(child.id)
    return names


def _source_path_v2(filename):
    candidates = (
        Path("/root/autodl-tmp") / filename,
        Path.cwd() / filename,
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Upload {filename} to /root/autodl-tmp for dense-state recovery"
        )
    return path


def _exec_source_until_v2(filename, stop_assignment):
    path = _source_path_v2(filename)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = []
    for node in tree.body:
        if stop_assignment in _assigned_names_v2(node):
            break
        nodes.append(node)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
        globals(),
    )
    return path


def _restore_dense_live_v2():
    if not DENSE_LIVE_NAMES_V2.difference(globals()):
        return
    print("RESTORE: rebuilding dense live state from persisted artifacts", flush=True)

    # This executes record/sample construction but stops before epoch tuning.
    v0_path = _exec_source_until_v2(
        "real_correspondence_multiscale_matcher_gateway_v0_cell.py",
        "selected_epoch",
    )

    prefix_checkpoint = (
        Path(OOF_ROOT)
        / "prefix_contrastive_matcher_adaptation_gateway_v1"
        / "models"
        / "fold1_prefix_contrastive_matcher.pt"
    )
    if not prefix_checkpoint.exists():
        raise FileNotFoundError(prefix_checkpoint)
    prefix_payload = torch.load(
        prefix_checkpoint, map_location="cpu", weights_only=False
    )
    # The prefix source validates the v0-selected epoch at import time. The
    # persisted final checkpoint contains that frozen value.
    globals()["selected_epoch"] = int(prefix_payload["epochs"])

    # Load prefix-adaptation function definitions without executing training.
    prefix_path = _exec_source_until_v2(
        "prefix_contrastive_matcher_adaptation_gateway_v1_cell.py",
        "started_v1",
    )
    globals()["mixed_epochs_v1"] = int(prefix_payload["epochs"])
    globals()["selected_lambda_v1"] = float(prefix_payload["ridge_lambda"])
    globals()["selected_dose_v1"] = float(prefix_payload["adapter_dose"])

    restored_prefix_queries = {}
    for index, well in enumerate(sorted(samples), start=1):
        restored_prefix_queries[well] = _legal_prefix_query_v1(
            well, samples[well].state_valid
        )
        if index % 100 == 0 or index == len(samples):
            print(f"RESTORE prefix queries {index}/{len(samples)}", flush=True)
    globals()["prefix_queries_v1"] = restored_prefix_queries

    restored_matcher = RealCorrespondenceMatcher().to(device)
    restored_matcher.load_state_dict(prefix_payload["model"], strict=True)
    restored_matcher.eval()
    globals()["final_model_v1"] = restored_matcher

    # Dense definitions are loaded without running Fold2 tuning or Fold1 audit.
    dense_path = _exec_source_until_v2(
        "dense_matcher_posterior_path_ceiling_audit_v0_cell.py",
        "started_dense",
    )
    grid_path = Path(OUT_DIR_DENSE) / "fold2_viterbi_grid.csv"
    if not grid_path.exists():
        raise FileNotFoundError(grid_path)
    restored_grid = pd.read_csv(grid_path).sort_values(
        ["rmse", "far_gt2000_rmse", "p95_well_rmse"]
    )
    selected = restored_grid.iloc[0]
    globals()["selected_mode_dense"] = str(selected["mode"])
    globals()["selected_lambda_dense"] = float(selected["transition_lambda"])
    globals()["outer_prior_dense"] = _fit_transition_prior_d0(outer_train_wells)
    restored_volumes, restored_report = _build_volumes_d0(
        restored_matcher, test_wells, "fold1_restore"
    )
    globals()["fold1_volumes_dense"] = restored_volumes
    restored_report.to_csv(
        Path(OUT_DIR_DENSE) / "fold1_volume_restore_report.csv", index=False
    )

    # Load only deterministic mode-bank configuration and function definitions.
    mode_path = _source_path_v2("dense_posterior_global_mode_bank_audit_v2_cell.py")
    mode_tree = ast.parse(
        mode_path.read_text(encoding="utf-8"), filename=str(mode_path)
    )
    mode_assignments = {
        "MODE_CONFIGS",
        "SUPPRESSION_ROUNDS",
        "SUPPRESSION_SIGMA_FT",
        "SUPPRESSION_QUANTILE",
    }
    mode_nodes = []
    for node in mode_tree.body:
        if isinstance(node, ast.FunctionDef):
            mode_nodes.append(node)
        elif _assigned_names_v2(node).intersection(mode_assignments):
            mode_nodes.append(node)
    exec(
        compile(
            ast.Module(body=mode_nodes, type_ignores=[]),
            str(mode_path),
            "exec",
        ),
        globals(),
    )
    globals()["SUPPRESSION_ROUNDS"] = 24
    print(
        "RESTORE COMPLETE",
        {
            "v0_source": str(v0_path),
            "prefix_source": str(prefix_path),
            "dense_source": str(dense_path),
            "mode_source": str(mode_path),
            "volumes": len(restored_volumes),
            "selected_mode": selected_mode_dense,
            "selected_lambda": selected_lambda_dense,
        },
        flush=True,
    )


if DENSE_LIVE_NAMES_V2.difference(globals()):
    _restore_dense_live_v2()


REQUIRED_LIVE_V2 = {
    "DATA_ROOT",
    "OOF_ROOT",
    "SEQUENCE_LENGTH",
    "MIN_CANDIDATE_COVERAGE",
    "CompleteWellAliasEnergy",
    "_numeric",
    "_interp_finite",
    "_robust_location_scale",
    "_fit_affine",
    "_read_well",
    "_json_default",
    "fold1_volumes_dense",
    "STATE_OFFSETS",
    "ZERO_STATE_INDEX_DENSE",
    "selected_mode_dense",
    "selected_lambda_dense",
    "outer_prior_dense",
    "_emission_d0",
    "_shift_slices_d0",
    "_bank_for_well_mode_bank",
    "_path_payload_d0",
    "MODE_CONFIGS",
}
missing_live_v2 = sorted(REQUIRED_LIVE_V2.difference(globals()))
if missing_live_v2:
    raise RuntimeError(
        "Missing live prerequisites: "
        + repr(missing_live_v2)
        + ". Run the prefix matcher -> dense posterior -> global mode-bank "
        "chain, then v0/v1 observability, in this kernel."
    )


EXPERIMENT_V2 = "fullwell_energy_densemode_fold1_application_v2"
OUT_DIR_V2 = Path(OOF_ROOT) / EXPERIMENT_V2
OUT_DIR_V2.mkdir(parents=True, exist_ok=True)

SEED_V2 = 20260722
MODE_BUDGET_V2 = 192
GROUP_SIZE_V2 = 49
GROUP_NONANCHOR_V2 = GROUP_SIZE_V2 - 1
WINNERS_PER_GROUP_V2 = 12
FINAL_SIZE_V2 = 49
SUPPRESSION_ROUNDS = 24

GATE_RMSE_V2 = 8.0
TARGET_RMSE_V2 = 5.5
GATE_GAIN_VS_COMMON_V2 = 0.50
GATE_POSITIVE_WELLS_V2 = 0.55
GATE_NO_GR_DEGRADATION_V2 = 1.0
GATE_SHUFFLED_DEGRADATION_V2 = 1.0


def _rmse_v2(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def _calibrated_well_v2(well):
    horizontal, typewell = _read_well(well)
    md = _numeric(horizontal, "MD")
    gr = _numeric(horizontal, "GR")
    truth = _numeric(horizontal, "TVT")
    tvt_input = _numeric(horizontal, "TVT_input")
    tw_tvt = _numeric(typewell, "TVT")
    tw_gr = _numeric(typewell, "GR")
    known = np.flatnonzero(np.isfinite(tvt_input))
    hidden = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
    if len(known) < 128 or len(hidden) < 128 or hidden[0] != len(known):
        raise RuntimeError(
            f"{well}: invalid official mask known={len(known)} hidden={len(hidden)}"
        )
    tw_prefix = _interp_finite(tw_tvt, tw_gr, tvt_input[known])
    slope, intercept, pairs, calibration_rmse = _fit_affine(
        tw_prefix, gr[known]
    )
    calibrated_tw = slope * tw_gr + intercept
    centre, scale = _robust_location_scale(gr[known])
    return {
        "horizontal": horizontal,
        "md": md,
        "gr": gr,
        "truth": truth,
        "known": known,
        "hidden": hidden,
        "tw_tvt": tw_tvt,
        "tw_gr": calibrated_tw,
        "centre": centre,
        "scale": scale,
        "calibration_pairs": pairs,
        "calibration_rmse": calibration_rmse,
    }


def _candidate_bank_v2(well, volume):
    candidates = _bank_for_well_mode_bank(volume)
    candidates = candidates[: min(MODE_BUDGET_V2, len(candidates))]
    paths = []
    metadata = []
    truth = None
    distance = None
    for index, candidate in enumerate(candidates):
        offset = STATE_OFFSETS[candidate["state_path"]].astype(np.float64)
        payload = _path_payload_d0(volume, offset)
        if truth is None:
            truth = payload["truth"].astype(np.float64)
            distance = payload["distance"].astype(np.float64)
        prediction = payload["prediction"].astype(np.float64)
        if len(prediction) != len(truth):
            raise RuntimeError(f"{well}: inconsistent candidate rows")
        paths.append(prediction)
        metadata.append(
            {
                "candidate_index": index,
                "source_mode": candidate["source_mode"],
                "transition_lambda": float(candidate["transition_lambda"]),
                "round": int(candidate["round"]),
                "common_energy": float(candidate["common_energy"]),
            }
        )
    if len(paths) < 32:
        raise RuntimeError(f"{well}: only {len(paths)} mode-bank paths")
    return (
        np.stack(paths).astype(np.float32),
        pd.DataFrame(metadata),
        truth.astype(np.float32),
        distance.astype(np.float32),
    )


def _alignment_panel_v2(well_data, paths):
    hidden = well_data["hidden"]
    md = well_data["md"]
    eval_md = md[hidden]
    if paths.shape[1] != len(eval_md):
        raise RuntimeError(
            f"candidate rows {paths.shape[1]} != official eval rows {len(eval_md)}"
        )
    sequence_md = np.linspace(eval_md[0], eval_md[-1], SEQUENCE_LENGTH)
    query = _interp_finite(md, well_data["gr"], sequence_md)
    query_valid = np.isfinite(query)
    query = np.clip(
        (query - well_data["centre"]) / well_data["scale"], -8.0, 8.0
    )
    sequence_paths = np.stack(
        [_interp_finite(eval_md, path, sequence_md) for path in paths], axis=0
    )
    reference = _interp_finite(
        well_data["tw_tvt"],
        well_data["tw_gr"],
        sequence_paths.reshape(-1),
    ).reshape(len(paths), SEQUENCE_LENGTH)
    valid = np.isfinite(reference) & query_valid[None, :]
    reference = np.clip(
        (reference - well_data["centre"]) / well_data["scale"], -8.0, 8.0
    )
    query = np.where(query_valid, query, 0.0).astype(np.float32)
    reference = np.where(valid, reference, 0.0).astype(np.float32)
    coverage = valid.mean(axis=1).astype(np.float32)
    return query, reference, valid, coverage


@torch.no_grad()
def _forward_indices_v2(
    model,
    query,
    reference,
    valid,
    indices,
    device,
    control="full",
):
    indices = np.asarray(indices, np.int64)
    q = query.copy()
    r = reference[indices].copy()
    v = valid[indices].copy()
    if control == "no_gr":
        q.fill(0.0)
    elif control == "shuffled_alignment":
        r = np.roll(r, shift=7, axis=0)
        v = np.roll(v, shift=7, axis=0)
    elif control != "full":
        raise ValueError(control)
    q_t = torch.from_numpy(q).to(device)
    r_t = torch.from_numpy(r).to(device)
    v_t = torch.from_numpy(v.astype(np.float32)).to(device)
    model.eval()
    return model(q_t, r_t, v_t).detach().cpu().numpy()


@torch.no_grad()
def _tournament_v2(model, query, reference, valid, device, control="full"):
    n_candidates = len(reference)
    if n_candidates <= FINAL_SIZE_V2:
        final_indices = np.arange(n_candidates, dtype=np.int64)
        final_logits = _forward_indices_v2(
            model, query, reference, valid, final_indices, device, control
        )
        return final_indices, final_logits, []

    nonanchor = np.arange(1, n_candidates, dtype=np.int64)
    groups = [
        nonanchor[start : start + GROUP_NONANCHOR_V2]
        for start in range(0, len(nonanchor), GROUP_NONANCHOR_V2)
    ]
    promoted = []
    group_rows = []
    for group_id, group in enumerate(groups):
        indices = np.r_[0, group]
        logits = _forward_indices_v2(
            model, query, reference, valid, indices, device, control
        )
        relative = logits - logits[0]
        order = np.argsort(-relative[1:])
        winners = group[order[: min(WINNERS_PER_GROUP_V2, len(group))]]
        promoted.extend(winners.tolist())
        for local, candidate in enumerate(indices):
            group_rows.append(
                {
                    "group": group_id,
                    "candidate": int(candidate),
                    "relative_anchor_energy": float(relative[local]),
                    "promoted": bool(candidate in set(winners.tolist())),
                }
            )

    promoted = np.asarray(list(dict.fromkeys(promoted)), np.int64)
    if len(promoted) > FINAL_SIZE_V2 - 1:
        # Cross-group energies are anchor calibrated, so this truncation remains
        # label-free and comparable across groups.
        relative_by_candidate = {
            int(row["candidate"]): float(row["relative_anchor_energy"])
            for row in group_rows
            if int(row["candidate"]) != 0
        }
        promoted = np.asarray(
            sorted(
                promoted,
                key=lambda candidate: relative_by_candidate[int(candidate)],
                reverse=True,
            )[: FINAL_SIZE_V2 - 1],
            np.int64,
        )
    final_indices = np.r_[0, promoted]
    final_logits = _forward_indices_v2(
        model, query, reference, valid, final_indices, device, control
    )
    return final_indices, final_logits, group_rows


def _subset_rmse_v2(truth, prediction, distance, low, high):
    mask = np.ones(len(truth), bool)
    if low is not None:
        mask &= distance > low
    if high is not None:
        mask &= distance <= high
    return _rmse_v2(truth[mask], prediction[mask]) if mask.any() else np.nan


def _metrics_v2(payload):
    truth = payload["truth"].astype(np.float64)
    prediction = payload["prediction"].astype(np.float64)
    distance = payload["distance"].astype(np.float64)
    wells = payload["well"].astype(str)
    bywell = []
    for well in np.unique(wells):
        mask = wells == well
        bywell.append(
            {
                "well": well,
                "rmse": _rmse_v2(truth[mask], prediction[mask]),
            }
        )
    bywell = pd.DataFrame(bywell)
    return {
        "rows": len(truth),
        "wells": len(bywell),
        "rmse": _rmse_v2(truth, prediction),
        "mae": float(np.mean(np.abs(truth - prediction))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
        "near_0_500_rmse": _subset_rmse_v2(truth, prediction, distance, None, 500),
        "mid_500_1000_rmse": _subset_rmse_v2(truth, prediction, distance, 500, 1000),
        "mid_1000_2000_rmse": _subset_rmse_v2(truth, prediction, distance, 1000, 2000),
        "far_gt2000_rmse": _subset_rmse_v2(truth, prediction, distance, 2000, None),
    }, bywell


device_v2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device_v2.type != "cuda":
    raise RuntimeError("CUDA is required")

checkpoint_v1 = (
    Path(OOF_ROOT)
    / "fullwell_nonrigid_warp_observability_fold2_v1"
    / "models"
    / "best_fold2_nonrigid_observability.pt"
)
if not checkpoint_v1.exists():
    raise FileNotFoundError(checkpoint_v1)
checkpoint_payload_v2 = torch.load(
    checkpoint_v1, map_location="cpu", weights_only=False
)
model_v2 = CompleteWellAliasEnergy().to(device_v2)
model_v2.load_state_dict(checkpoint_payload_v2["model"], strict=True)
model_v2.eval()

started_v2 = time.time()
controls_v2 = ("full", "no_gr", "shuffled_alignment")
selected_payloads_v2 = {control: [] for control in controls_v2}
posterior_payloads_v2 = {control: [] for control in controls_v2}
oracle_payloads_v2 = []
common_payloads_v2 = []
well_rows_v2 = []
group_rows_v2 = []

for well_index, well in enumerate(sorted(fold1_volumes_dense), start=1):
    volume = fold1_volumes_dense[well]
    well_data = _calibrated_well_v2(well)
    paths, metadata, truth, distance = _candidate_bank_v2(well, volume)
    query, reference, valid, coverage = _alignment_panel_v2(well_data, paths)

    common_index = int(metadata["common_energy"].to_numpy().argmax())
    row = {
        "well": well,
        "candidates": len(paths),
        "available_candidates": int(
            (coverage >= MIN_CANDIDATE_COVERAGE).sum()
        ),
        "common_index": common_index,
    }

    final_indices_by_control = {}
    for control in controls_v2:
        final_indices, final_logits, group_rows = _tournament_v2(
            model_v2,
            query,
            reference,
            valid,
            device_v2,
            control,
        )
        selected_local = int(np.argmax(final_logits))
        selected_index = int(final_indices[selected_local])
        probability = np.exp(final_logits - np.max(final_logits))
        probability /= probability.sum()
        posterior_path = np.sum(
            probability[:, None] * paths[final_indices], axis=0
        )
        selected_payloads_v2[control].append(
            {
                "well": np.full(len(truth), well, dtype="U16"),
                "truth": truth,
                "prediction": paths[selected_index],
                "distance": distance,
            }
        )
        posterior_payloads_v2[control].append(
            {
                "well": np.full(len(truth), well, dtype="U16"),
                "truth": truth,
                "prediction": posterior_path.astype(np.float32),
                "distance": distance,
            }
        )
        row[f"{control}_selected_index"] = selected_index
        row[f"{control}_final_candidates"] = len(final_indices)
        row[f"{control}_entropy"] = float(
            -np.sum(probability * np.log(probability + 1e-12))
        )
        final_indices_by_control[control] = final_indices.copy()
        for group_row in group_rows:
            group_rows_v2.append(
                {"well": well, "control": control, **group_row}
            )

    # Selection is now frozen. Truth enters only below this boundary.
    candidate_sse = np.sum(
        (paths.astype(np.float64) - truth[None, :].astype(np.float64)) ** 2,
        axis=1,
    )
    oracle_index = int(np.argmin(candidate_sse))
    row["oracle_index"] = oracle_index
    row["oracle_rmse"] = math.sqrt(candidate_sse[oracle_index] / len(truth))
    row["common_rmse"] = math.sqrt(candidate_sse[common_index] / len(truth))
    oracle_payloads_v2.append(
        {
            "well": np.full(len(truth), well, dtype="U16"),
            "truth": truth,
            "prediction": paths[oracle_index],
            "distance": distance,
        }
    )
    common_payloads_v2.append(
        {
            "well": np.full(len(truth), well, dtype="U16"),
            "truth": truth,
            "prediction": paths[common_index],
            "distance": distance,
        }
    )
    for control in controls_v2:
        selected_index = int(row[f"{control}_selected_index"])
        posterior_path = posterior_payloads_v2[control][-1]["prediction"]
        row[f"{control}_selected_rmse"] = math.sqrt(
            candidate_sse[selected_index] / len(truth)
        )
        row[f"{control}_posterior_rmse"] = _rmse_v2(truth, posterior_path)
        row[f"{control}_oracle_promoted"] = bool(
            oracle_index in set(final_indices_by_control[control].tolist())
        )
    well_rows_v2.append(row)
    if well_index % 20 == 0 or well_index == len(fold1_volumes_dense):
        print(
            f"score dense modes {well_index}/{len(fold1_volumes_dense)}",
            flush=True,
        )


def _merge_payloads_v2(payloads):
    return {
        key: np.concatenate([payload[key] for payload in payloads])
        for key in ("well", "truth", "prediction", "distance")
    }


well_frame_v2 = pd.DataFrame(well_rows_v2)
well_frame_v2.to_csv(OUT_DIR_V2 / "fold1_bywell.csv", index=False)
pd.DataFrame(group_rows_v2).to_csv(
    OUT_DIR_V2 / "tournament_groups.csv", index=False
)

merged_selected_v2 = {
    control: _merge_payloads_v2(payloads)
    for control, payloads in selected_payloads_v2.items()
}
merged_posterior_v2 = {
    control: _merge_payloads_v2(payloads)
    for control, payloads in posterior_payloads_v2.items()
}
merged_oracle_v2 = _merge_payloads_v2(oracle_payloads_v2)
merged_common_v2 = _merge_payloads_v2(common_payloads_v2)

metrics_selected_v2 = {}
metrics_posterior_v2 = {}
bywell_selected_v2 = {}
for control in controls_v2:
    metrics_selected_v2[control], bywell_selected_v2[control] = _metrics_v2(
        merged_selected_v2[control]
    )
    metrics_posterior_v2[control], _ = _metrics_v2(
        merged_posterior_v2[control]
    )
metrics_oracle_v2, bywell_oracle_v2 = _metrics_v2(merged_oracle_v2)
metrics_common_v2, bywell_common_v2 = _metrics_v2(merged_common_v2)

full_bywell_v2 = bywell_selected_v2["full"].merge(
    bywell_common_v2, on="well", suffixes=("_full", "_common")
)
positive_fraction_v2 = float(
    np.mean(full_bywell_v2["rmse_full"] < full_bywell_v2["rmse_common"])
)
gain_vs_common_v2 = float(
    metrics_common_v2["rmse"] - metrics_selected_v2["full"]["rmse"]
)
no_gr_degradation_v2 = float(
    metrics_selected_v2["no_gr"]["rmse"]
    - metrics_selected_v2["full"]["rmse"]
)
shuffled_degradation_v2 = float(
    metrics_selected_v2["shuffled_alignment"]["rmse"]
    - metrics_selected_v2["full"]["rmse"]
)

strict_pass_v2 = bool(
    metrics_selected_v2["full"]["rmse"] <= GATE_RMSE_V2
    and gain_vs_common_v2 >= GATE_GAIN_VS_COMMON_V2
    and positive_fraction_v2 >= GATE_POSITIVE_WELLS_V2
    and no_gr_degradation_v2 >= GATE_NO_GR_DEGRADATION_V2
    and shuffled_degradation_v2 >= GATE_SHUFFLED_DEGRADATION_V2
)
target_pass_v2 = bool(
    metrics_selected_v2["full"]["rmse"] <= TARGET_RMSE_V2
)

np.savez_compressed(
    OUT_DIR_V2 / "fold1_predictions.npz",
    well=merged_selected_v2["full"]["well"],
    truth=merged_selected_v2["full"]["truth"],
    distance=merged_selected_v2["full"]["distance"],
    selected=merged_selected_v2["full"]["prediction"],
    posterior=merged_posterior_v2["full"]["prediction"],
    common=merged_common_v2["prediction"],
    oracle=merged_oracle_v2["prediction"],
)

verdict_v2 = {
    "experiment": EXPERIMENT_V2,
    "stage": "zero_shot_fold1_label_free_dense_mode_selection",
    "sample_unit": "one complete physical well",
    "candidate_generation_uses_truth": False,
    "model_training_folds": [3, 4, 5],
    "model_selection_fold": 2,
    "application_fold": 1,
    "uses_neighbor_pf_strong_cp12_gold": False,
    "checkpoint": checkpoint_v1,
    "candidate_bank": {
        "budget": MODE_BUDGET_V2,
        "suppression_rounds": SUPPRESSION_ROUNDS,
        "configs": len(MODE_CONFIGS),
        "tournament_group_size": GROUP_SIZE_V2,
        "winners_per_group": WINNERS_PER_GROUP_V2,
        "final_size": FINAL_SIZE_V2,
    },
    "selected": metrics_selected_v2,
    "posterior": metrics_posterior_v2,
    "common_energy_baseline": metrics_common_v2,
    "oracle": metrics_oracle_v2,
    "diagnostics": {
        "gain_vs_common": gain_vs_common_v2,
        "positive_well_fraction_vs_common": positive_fraction_v2,
        "no_gr_degradation": no_gr_degradation_v2,
        "shuffled_alignment_degradation": shuffled_degradation_v2,
        "oracle_promoted_fraction": float(
            well_frame_v2["full_oracle_promoted"].mean()
        ),
    },
    "gates": {
        "rmse_le": GATE_RMSE_V2,
        "target_rmse_le": TARGET_RMSE_V2,
        "gain_vs_common_ge": GATE_GAIN_VS_COMMON_V2,
        "positive_wells_ge": GATE_POSITIVE_WELLS_V2,
        "each_control_degradation_ge": min(
            GATE_NO_GR_DEGRADATION_V2, GATE_SHUFFLED_DEGRADATION_V2
        ),
    },
    "strict_pass": strict_pass_v2,
    "target_5p5_pass": target_pass_v2,
    "elapsed_minutes": (time.time() - started_v2) / 60.0,
    "free_gb": shutil.disk_usage(OUT_DIR_V2).free / 2**30,
    "next": (
        "PASS -> freeze scorer and build five-fold label-free mode-bank CV"
        if strict_pass_v2
        else "FAIL -> synthetic-warp energy does not transfer to generated path modes"
    ),
}
(OUT_DIR_V2 / "verdict.json").write_text(
    json.dumps(verdict_v2, indent=2, default=_json_default), encoding="utf-8"
)

print("SELECTED", json.dumps(metrics_selected_v2, indent=2), flush=True)
print("POSTERIOR", json.dumps(metrics_posterior_v2, indent=2), flush=True)
print("ORACLE", json.dumps(metrics_oracle_v2, indent=2), flush=True)
print("FINAL VERDICT", json.dumps(verdict_v2, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR_V2, flush=True)

del model_v2
gc.collect()
torch.cuda.empty_cache()
