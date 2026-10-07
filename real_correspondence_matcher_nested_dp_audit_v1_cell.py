"""Nested path-integration audit for the real-correspondence matcher.

Run in the same live kernel immediately after
``real_correspondence_multiscale_matcher_gateway_v0_cell.py``.  The cell:

* retrains a Fold-2 decoder-tuning matcher on Folds 3-5 only;
* selects emission mode and dose on untouched Fold 2;
* applies the frozen decoder to the already-trained Folds 2-5 matcher;
* evaluates Fold 1 once on every official evaluation row.

No neighbour, CP12, Gold, validation truth routing, or learned trajectory head
is used.  The decoder is a first-order Viterbi path with its transition prior
estimated from outer-train truth slopes.
"""

from pathlib import Path
import gc
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch


REQUIRED_NDP = {
    "samples",
    "model",
    "device",
    "core_wells",
    "tune_wells",
    "outer_train_wells",
    "test_wells",
    "selected_epoch",
    "_new_training_objects",
    "_train_epoch",
    "_to_tensors",
    "_load_raw_record",
    "STATE_OFFSETS",
    "N_STATES",
    "STATE_HALF_WIDTH_FT",
    "OOF_ROOT",
    "SEED",
    "_rmse",
    "_json_default",
}
missing_ndp = sorted(REQUIRED_NDP.difference(globals()))
if missing_ndp:
    raise RuntimeError(
        "Run this directly after the matcher gateway; missing globals: "
        + repr(missing_ndp)
    )


EXPERIMENT_NDP = "real_correspondence_matcher_nested_dp_audit_v1"
OUT_DIR_NDP = OOF_ROOT / EXPERIMENT_NDP
MODEL_DIR_NDP = OUT_DIR_NDP / "models"
for directory in (OUT_DIR_NDP, MODEL_DIR_NDP):
    directory.mkdir(parents=True, exist_ok=True)

EMISSION_MODES_NDP = (
    "raw",
    "half_prior_removed",
    "likelihood_ratio",
)
EMISSION_WEIGHTS_NDP = (0.25, 0.50, 1.0, 2.0, 4.0)
TRANSITION_SIGMA_FLOOR_FT = 0.75
MAX_TRANSITION_CAP = 16
PATH_PASS_RMSE = 8.0
PATH_PASS_MEDIAN_WELL = 5.5
PATH_PASS_P95_WELL = 15.0


def _json_ndp(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _train_core_matcher_ndp():
    epochs = int(selected_epoch)
    tune_model, optimizer, scheduler, scaler = _new_training_objects(
        device, epochs, SEED + 2200
    )
    rng = np.random.default_rng(SEED + 2200)
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        metrics = _train_epoch(
            tune_model,
            samples,
            core_wells,
            device,
            optimizer,
            scaler,
            rng,
            epoch,
            "nested_dp_core",
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            **metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(row)
        print("NESTED-DP CORE", row, flush=True)
    checkpoint = MODEL_DIR_NDP / "fold2_tuning_matcher.pt"
    torch.save(
        {
            "model": {
                key: value.detach().cpu()
                for key, value in tune_model.state_dict().items()
            },
            "train_folds": [3, 4, 5],
            "epochs": epochs,
        },
        checkpoint,
    )
    pd.DataFrame(history).to_csv(
        OUT_DIR_NDP / "fold2_matcher_training_history.csv", index=False
    )
    del optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tune_model, checkpoint


@torch.no_grad()
def _score_wells_ndp(matcher, wells):
    matcher.eval()
    result = {}
    for index, well in enumerate(wells, 1):
        sample = samples[well]
        query_set = sample.evaluation
        if query_set is None:
            raise RuntimeError(f"{well}: missing evaluation bank")
        query, reference, state_valid, _, _, _ = _to_tensors(
            sample, query_set, device, no_gr=False
        )
        no_gr_query = query.clone()
        no_gr_query[:, :, 0] = 0.0
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            full = matcher(query, reference, state_valid)
            no_gr = matcher(no_gr_query, reference, state_valid)
        raw = _load_raw_record(well)
        sample_md = raw.md[query_set.sample_rows].astype(np.float32)
        result[well] = {
            "full": full.float().cpu().numpy().astype(np.float32),
            "no_gr": no_gr.float().cpu().numpy().astype(np.float32),
            "state_valid": sample.state_valid.astype(bool),
            "sample_md": sample_md,
            "sample_labels": query_set.labels.astype(np.float32),
            "anchor_md": float(raw.md[raw.anchor_index]),
            "anchor_tvt": float(raw.anchor_tvt),
            "eval_md": raw.md[raw.eval_index].astype(np.float32),
            "eval_truth": raw.truth[raw.eval_index].astype(np.float32),
        }
        if index % 50 == 0 or index == len(wells):
            print(f"score path emissions {index}/{len(wells)}", flush=True)
    return result


def _fit_transition_prior_ndp(wells):
    slopes = []
    changes = []
    for well in wells:
        query_set = samples[well].train
        raw = _load_raw_record(well)
        md = raw.md[query_set.sample_rows].astype(np.float64)
        labels = query_set.labels.astype(np.float64)
        delta_md = np.diff(md)
        delta_state = np.diff(labels)
        valid = np.isfinite(delta_md) & np.isfinite(delta_state) & (delta_md > 0)
        slopes.extend((delta_state[valid] / delta_md[valid]).tolist())
        changes.extend(np.abs(delta_state[valid]).tolist())
    slopes = np.asarray(slopes, np.float64)
    changes = np.asarray(changes, np.float64)
    centre = float(np.median(slopes))
    scale = float(1.4826 * np.median(np.abs(slopes - centre)))
    scale = max(scale, 1e-4)
    max_jump = int(
        np.clip(np.ceil(np.quantile(changes, 0.999)) + 2, 4, MAX_TRANSITION_CAP)
    )
    return {
        "slope_centre": centre,
        "slope_scale": scale,
        "max_jump": max_jump,
        "slope_count": len(slopes),
        "change_p99": float(np.quantile(changes, 0.99)),
        "change_p999": float(np.quantile(changes, 0.999)),
    }


def _robust_emission_ndp(full, no_gr, mode):
    if mode == "raw":
        score = full
    elif mode == "half_prior_removed":
        score = full - 0.5 * no_gr
    elif mode == "likelihood_ratio":
        score = full - no_gr
    else:
        raise ValueError(mode)
    median = np.median(score, axis=1, keepdims=True)
    q25 = np.quantile(score, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(score, 0.75, axis=1, keepdims=True)
    scale = np.maximum(q75 - q25, 0.10)
    return np.clip((score - median) / scale, -12.0, 12.0).astype(np.float32)


def _stack_decoder_inputs_ndp(score_store, wells, mode):
    lengths = {len(score_store[well]["sample_md"]) for well in wells}
    if len(lengths) != 1:
        raise RuntimeError(f"Decoder expects one sampled length; found {lengths}")
    emissions = np.stack(
        [
            _robust_emission_ndp(
                score_store[well]["full"], score_store[well]["no_gr"], mode
            )
            for well in wells
        ]
    )
    state_valid = np.stack([score_store[well]["state_valid"] for well in wells])
    sample_md = np.stack([score_store[well]["sample_md"] for well in wells])
    anchor_md = np.asarray([score_store[well]["anchor_md"] for well in wells])
    return emissions, state_valid, sample_md, anchor_md


def _viterbi_batch_ndp(emissions, state_valid, sample_md, anchor_md, prior, weight):
    emission = torch.from_numpy(emissions).to(device)
    valid = torch.from_numpy(state_valid).to(device)
    md = torch.from_numpy(sample_md.astype(np.float32)).to(device)
    anchor = torch.from_numpy(anchor_md.astype(np.float32)).to(device)
    states = torch.from_numpy(np.asarray(STATE_OFFSETS, np.float32)).to(device)
    batch, columns, n_states = emission.shape
    max_jump = int(prior["max_jump"])
    slope_centre = float(prior["slope_centre"])
    slope_scale = float(prior["slope_scale"])

    first_md = (md[:, 0] - anchor).clamp_min(1.0)
    first_expected = slope_centre * first_md[:, None]
    first_sigma = torch.clamp(
        slope_scale * first_md[:, None], min=TRANSITION_SIGMA_FLOOR_FT
    )
    dp = float(weight) * emission[:, 0]
    dp = dp - 0.5 * ((states[None] - first_expected) / first_sigma) ** 2
    dp = dp.masked_fill(~valid, -1e8)
    back = torch.zeros(
        (batch, columns, n_states), dtype=torch.int8, device=device
    )

    negative = torch.full((batch, n_states), -1e8, device=device)
    for column in range(1, columns):
        delta_md = (md[:, column] - md[:, column - 1]).clamp_min(1.0)
        expected = slope_centre * delta_md
        sigma = torch.clamp(
            slope_scale * delta_md, min=TRANSITION_SIGMA_FLOOR_FT
        )
        candidates = []
        shifts = list(range(-max_jump, max_jump + 1))
        for shift in shifts:
            shifted = negative.clone()
            if shift >= 0:
                shifted[:, shift:] = dp[:, : n_states - shift]
            else:
                shifted[:, : n_states + shift] = dp[:, -shift:]
            penalty = 0.5 * ((float(shift) - expected) / sigma) ** 2
            candidates.append(shifted - penalty[:, None])
        candidate = torch.stack(candidates, dim=0)
        best_value, best_index = candidate.max(dim=0)
        chosen_shift = (
            best_index.to(torch.int16) - max_jump
        ).to(torch.int8)
        dp = best_value + float(weight) * emission[:, column]
        dp = dp.masked_fill(~valid, -1e8)
        back[:, column] = chosen_shift

    state_index = dp.argmax(dim=1)
    path_index = torch.empty(
        (batch, columns), dtype=torch.long, device=device
    )
    path_index[:, -1] = state_index
    rows = torch.arange(batch, device=device)
    for column in range(columns - 1, 0, -1):
        shift = back[rows, column, state_index].long()
        state_index = (state_index - shift).clamp(0, n_states - 1)
        path_index[:, column - 1] = state_index
    return states[path_index].cpu().numpy().astype(np.float32)


def _path_metrics_ndp(paths, score_store, wells):
    truth_all = []
    prediction_all = []
    bywell = []
    for path, well in zip(paths, wells):
        item = score_store[well]
        prediction_offset = np.interp(
            item["eval_md"],
            np.r_[item["anchor_md"], item["sample_md"]],
            np.r_[0.0, path],
        )
        prediction = item["anchor_tvt"] + prediction_offset
        truth = item["eval_truth"].astype(np.float64)
        truth_all.append(truth)
        prediction_all.append(prediction)
        bywell.append(
            {
                "well": well,
                "rows": len(truth),
                "rmse": _rmse(truth, prediction),
                "mae": float(np.mean(np.abs(truth - prediction))),
            }
        )
    truth = np.concatenate(truth_all)
    prediction = np.concatenate(prediction_all)
    bywell = pd.DataFrame(bywell)
    return {
        "rows": len(truth),
        "wells": len(wells),
        "rmse": _rmse(truth, prediction),
        "mae": float(np.mean(np.abs(truth - prediction))),
        "median_well_rmse": float(bywell["rmse"].median()),
        "p75_well_rmse": float(bywell["rmse"].quantile(0.75)),
        "p95_well_rmse": float(bywell["rmse"].quantile(0.95)),
        "max_well_rmse": float(bywell["rmse"].max()),
    }, bywell, truth.astype(np.float32), prediction.astype(np.float32)


def _anchor_metrics_ndp(score_store, wells):
    truth = []
    prediction = []
    for well in wells:
        item = score_store[well]
        truth.append(item["eval_truth"])
        prediction.append(
            np.full(len(item["eval_truth"]), item["anchor_tvt"], np.float32)
        )
    truth = np.concatenate(truth)
    prediction = np.concatenate(prediction)
    return {"rmse": _rmse(truth, prediction), "mae": float(np.mean(np.abs(truth - prediction)))}


# ---------------------------------------------------------------------------
# Execute nested decoder selection and untouched Fold-1 application
# ---------------------------------------------------------------------------

started_ndp = time.time()
final_model_ndp = model
transition_prior_ndp = _fit_transition_prior_ndp(core_wells)
print("TRANSITION PRIOR", transition_prior_ndp, flush=True)

tune_model_ndp, tune_checkpoint_ndp = _train_core_matcher_ndp()
tune_scores_ndp = _score_wells_ndp(tune_model_ndp, tune_wells)
tune_anchor_ndp = _anchor_metrics_ndp(tune_scores_ndp, tune_wells)
grid_rows_ndp = []
best_ndp = None
for mode_ndp in EMISSION_MODES_NDP:
    emission_ndp, valid_ndp, md_ndp, anchor_md_ndp = _stack_decoder_inputs_ndp(
        tune_scores_ndp, tune_wells, mode_ndp
    )
    for weight_ndp in EMISSION_WEIGHTS_NDP:
        paths_ndp = _viterbi_batch_ndp(
            emission_ndp,
            valid_ndp,
            md_ndp,
            anchor_md_ndp,
            transition_prior_ndp,
            weight_ndp,
        )
        metrics_ndp, _, _, _ = _path_metrics_ndp(
            paths_ndp, tune_scores_ndp, tune_wells
        )
        row_ndp = {
            "mode": mode_ndp,
            "emission_weight": weight_ndp,
            **metrics_ndp,
        }
        grid_rows_ndp.append(row_ndp)
        print("FOLD2 DECODER", row_ndp, flush=True)
        objective_ndp = (
            metrics_ndp["rmse"],
            metrics_ndp["p95_well_rmse"],
            metrics_ndp["median_well_rmse"],
        )
        if best_ndp is None or objective_ndp < best_ndp["objective"]:
            best_ndp = {
                "objective": objective_ndp,
                "mode": mode_ndp,
                "emission_weight": float(weight_ndp),
                "metrics": metrics_ndp,
            }

grid_ndp = pd.DataFrame(grid_rows_ndp).sort_values("rmse")
grid_ndp.to_csv(OUT_DIR_NDP / "fold2_decoder_grid.csv", index=False)
print("FROZEN DECODER", best_ndp, flush=True)

test_scores_ndp = _score_wells_ndp(final_model_ndp, test_wells)
test_anchor_ndp = _anchor_metrics_ndp(test_scores_ndp, test_wells)
test_emission_ndp, test_valid_ndp, test_md_ndp, test_anchor_md_ndp = _stack_decoder_inputs_ndp(
    test_scores_ndp, test_wells, best_ndp["mode"]
)
test_paths_ndp = _viterbi_batch_ndp(
    test_emission_ndp,
    test_valid_ndp,
    test_md_ndp,
    test_anchor_md_ndp,
    transition_prior_ndp,
    best_ndp["emission_weight"],
)
test_metrics_ndp, test_bywell_ndp, test_truth_ndp, test_prediction_ndp = _path_metrics_ndp(
    test_paths_ndp, test_scores_ndp, test_wells
)
test_bywell_ndp.to_csv(OUT_DIR_NDP / "fold1_bywell.csv", index=False)
np.savez_compressed(
    OUT_DIR_NDP / "fold1_oof.npz",
    truth=test_truth_ndp,
    prediction=test_prediction_ndp,
    well=np.concatenate(
        [
            np.full(len(test_scores_ndp[well]["eval_truth"]), well, dtype="U16")
            for well in test_wells
        ]
    ),
)

strict_pass_ndp = bool(
    test_metrics_ndp["rmse"] <= PATH_PASS_RMSE
    and test_metrics_ndp["median_well_rmse"] <= PATH_PASS_MEDIAN_WELL
    and test_metrics_ndp["p95_well_rmse"] <= PATH_PASS_P95_WELL
)
verdict_ndp = {
    "experiment": EXPERIMENT_NDP,
    "stage": "nested_zero_learning_path_integration",
    "matcher_epoch": int(selected_epoch),
    "decoder_selection": {
        "train_folds": [3, 4, 5],
        "validation_fold": 2,
        "selected_mode": best_ndp["mode"],
        "selected_emission_weight": best_ndp["emission_weight"],
        "fold2_metrics": best_ndp["metrics"],
        "fold2_anchor": tune_anchor_ndp,
    },
    "fold1_application": {
        "matcher_train_folds": [2, 3, 4, 5],
        "metrics": test_metrics_ndp,
        "anchor": test_anchor_ndp,
        "gain_vs_anchor": test_anchor_ndp["rmse"] - test_metrics_ndp["rmse"],
    },
    "transition_prior": transition_prior_ndp,
    "gates": {
        "rmse_le": PATH_PASS_RMSE,
        "median_well_rmse_le": PATH_PASS_MEDIAN_WELL,
        "p95_well_rmse_le": PATH_PASS_P95_WELL,
    },
    "strict_pass": strict_pass_ndp,
    "tuning_matcher_checkpoint": str(tune_checkpoint_ndp),
    "elapsed_minutes": (time.time() - started_ndp) / 60.0,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 1024**3,
    "next": (
        "PASS -> use the frozen matcher likelihood in per-well PCA/B-spline registration"
        if strict_pass_ndp
        else (
            "HOLD -> train an explicit query-contrast matcher before registration"
            if test_metrics_ndp["rmse"] <= 10.0
            else "FAIL -> close this matcher representation before registration"
        )
    ),
}
_json_ndp(OUT_DIR_NDP / "verdict.json", verdict_ndp)

print("\nFOLD2 DECODER GRID")
print(grid_ndp.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict_ndp, indent=2, ensure_ascii=False, default=_json_default), flush=True)
print("saved:", OUT_DIR_NDP, flush=True)

