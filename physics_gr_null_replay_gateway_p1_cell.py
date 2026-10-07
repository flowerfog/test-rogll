"""Conservative GR physics endpoint with a visible-prefix null-model gate.

One frozen raw-data beam arm supplies the GR-steered path.  Two zero-GR null
paths (constant TVT and constant structural surface) are retained explicitly.
Pseudo-PS cuts inside the visible prefix decide (a) which null is safer and
(b) whether GR steering has earned activation.  Truth never enters a per-well
decision.  Fold 3 selects one global margin/dose; Fold 2 confirms it once.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "physics_gr_null_replay_gateway_p1"
ROOT = Path("/root/autodl-tmp") if os.name != "nt" else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT if os.name != "nt" else Path.cwd() / "output" / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
V3_SOURCE = ROOT / "rawdata_lattice_mode_union_v3_cell.py"
BASE_PATH = OOF_ROOT / "frontier_6p377_cp8_neighbor_dose_replay_v33" / "cp8_neighbor_replay_oof.parquet"
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"

PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
DISCOVERY_WELLS = 24
REPLAY_BACKOFFS_FT = (512.0, 1024.0)
MARGINS_FT = (0.0, 0.5, 1.0, 2.0, 4.0)
DOSES = (0.25, 0.5, 0.75, 1.0)
MIN_REPLAY_ROWS = 128
BLEND_WEIGHTS = np.arange(0.0, 0.5001, 0.025)
started = time.time()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload):
    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False), encoding="utf-8")


if not V3_SOURCE.is_file():
    raise FileNotFoundError(
        f"Upload rawdata_lattice_mode_union_v3_cell.py to {V3_SOURCE}; its V1/V2 dependencies must also be present"
    )
source = V3_SOURCE.read_text(encoding="utf-8")
marker = "# --- V3 EXECUTION ---"
if marker not in source:
    raise RuntimeError("V3 execution marker not found")
N = {"__name__": "physics_p1_v3_definitions", "__file__": str(V3_SOURCE)}
exec(compile(source.split(marker, 1)[0], str(V3_SOURCE), "exec"), N)
B = N["B"]
V = N["V"]
DATA_ROOT = N["DATA_ROOT"]

ARM = {
    "id": "raw_clip_b512_shape4",
    "beam": 512,
    "diversity_ft": 4.0,
    "reference": "raw",
    "likelihood": "clipped_gaussian",
    "history_diversity": True,
}


def pseudo_record(record, cut):
    cut = int(cut)
    rows = np.flatnonzero(
        (np.arange(len(record["md"])) > cut)
        & (np.arange(len(record["md"])) <= record["anchor_index"])
        & np.isfinite(record["tvt_input"])
        & np.isfinite(record["md"])
        & np.isfinite(record["z"])
    )
    if len(rows) < MIN_REPLAY_ROWS:
        raise RuntimeError(f"only {len(rows)} pseudo-evaluation rows")
    output = dict(record)
    output["anchor_index"] = cut
    output["anchor_md"] = float(record["md"][cut])
    output["anchor_surface"] = float(record["tvt_input"][cut] + record["z"][cut])
    output["eval_rows"] = rows
    output["sampled_rows"] = rows[:: int(V["ROW_STRIDE"])]
    output["evaluation"] = np.isin(np.arange(len(record["md"])), rows)
    return output


def null_paths(record, rows, anchor_index=None):
    anchor = record["anchor_index"] if anchor_index is None else int(anchor_index)
    anchor_tvt = float(record["tvt_input"][anchor])
    anchor_surface = anchor_tvt + float(record["z"][anchor])
    return {
        "tvt_flat": np.full(len(rows), anchor_tvt, np.float64),
        "surface_flat": anchor_surface - record["z"][rows],
    }


def decode_map(record, prior):
    scores, paths, segments = N["_beam_bank_v3"](record, prior, ARM)
    path = paths[int(np.argmin(scores))]
    prediction = B["_path_prediction"](record, prior, path, segments)
    return prediction, int(len(paths)), float(np.min(scores))


def build_payload(record, prior):
    eval_rows = record["eval_rows"]
    gr_prediction, bank_paths, map_score = decode_map(record, prior)
    actual_nulls = null_paths(record, eval_rows)
    replay = {"tvt_flat": [], "surface_flat": [], "gr_map": []}
    successful = 0
    anchor_md = float(record["anchor_md"])
    known = np.flatnonzero(
        np.isfinite(record["tvt_input"])
        & np.isfinite(record["md"])
        & np.isfinite(record["z"])
    )
    for backoff in REPLAY_BACKOFFS_FT:
        eligible = known[record["md"][known] <= anchor_md - backoff]
        if len(eligible) < 128:
            continue
        cut = int(eligible[-1])
        try:
            pseudo = pseudo_record(record, cut)
            prediction, _, _ = decode_map(pseudo, prior)
            rows = pseudo["eval_rows"]
            truth = record["tvt_input"][rows]
            nulls = null_paths(record, rows, cut)
            replay["gr_map"].append(float(np.sqrt(np.mean((prediction[rows] - truth) ** 2))))
            for name in ("tvt_flat", "surface_flat"):
                replay[name].append(float(np.sqrt(np.mean((nulls[name] - truth) ** 2))))
            successful += 1
        except Exception:
            continue
    if not successful:
        replay = {"tvt_flat": [0.0], "surface_flat": [1e6], "gr_map": [1e6]}
    replay_mean = {name: float(np.mean(values)) for name, values in replay.items()}
    null_name = min(("tvt_flat", "surface_flat"), key=replay_mean.get)
    return {
        "well": record["well"],
        "rows": eval_rows,
        "truth": record["truth"][eval_rows],
        "distance": record["md"][eval_rows] - record["anchor_md"],
        "null_name": null_name,
        "null": actual_nulls[null_name],
        "gr": gr_prediction[eval_rows],
        "replay_null_rmse": replay_mean[null_name],
        "replay_gr_rmse": replay_mean["gr_map"],
        "replay_gain": replay_mean[null_name] - replay_mean["gr_map"],
        "replay_payloads": successful,
        "bank_paths": bank_paths,
        "map_score": map_score,
    }


def assemble(payloads, margin, dose, shuffle_gate=False, seed=20260727):
    gate_gains = np.asarray([payload["replay_gain"] for payload in payloads], np.float64)
    if shuffle_gate:
        gate_gains = np.random.default_rng(seed).permutation(gate_gains)
    frames = []
    for payload, gate_gain in zip(payloads, gate_gains):
        active = gate_gain > float(margin)
        prediction = payload["null"] + (float(dose) if active else 0.0) * (payload["gr"] - payload["null"])
        rows = payload["rows"]
        well = payload["well"]
        frames.append(
            pd.DataFrame(
                {
                    "id": [f"{well}_{int(row)}" for row in rows],
                    "well": well,
                    "row_idx": rows.astype(np.int32),
                    "tvt_true": payload["truth"].astype(np.float32),
                    "md_since": payload["distance"].astype(np.float32),
                    "prediction": prediction.astype(np.float32),
                    "null": payload["null"].astype(np.float32),
                    "gr_map": payload["gr"].astype(np.float32),
                    "active": active,
                    "replay_gain": float(payload["replay_gain"]),
                    "gate_gain": float(gate_gain),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def metrics(frame, column="prediction"):
    error = frame[column].to_numpy(np.float64) - frame["tvt_true"].to_numpy(np.float64)
    bywell = pd.DataFrame({"well": frame["well"], "e2": error * error}).groupby("well")["e2"].mean()
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    return {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(bywell))),
        "p95_well_rmse": float(np.quantile(np.sqrt(bywell), 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else None,
    }


wells = [path.name.split("__", 1)[0] for path in sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))]
if not FOLD_PATH.is_file():
    raise FileNotFoundError(f"Exact 6.377 fold metadata is required: {FOLD_PATH}")
folds = pd.read_csv(FOLD_PATH)
well_column = next((column for column in ("well", "well_id", "pair_id") if column in folds), None)
fold_column = next((column for column in ("fold", "outer_fold") if column in folds), None)
if well_column is None or fold_column is None:
    raise RuntimeError(f"Cannot identify well/fold columns in {FOLD_PATH}: {folds.columns.tolist()}")
folds = folds[[well_column, fold_column]].rename(columns={well_column: "well", fold_column: "fold"})
folds["well"] = folds["well"].astype(str)
folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
if folds["well"].duplicated().any():
    raise RuntimeError("Duplicate wells in exact fold metadata")
fold_by_well = dict(zip(folds["well"], folds["fold"]))
missing_fold_wells = sorted(set(wells).difference(fold_by_well))
if missing_fold_wells:
    raise RuntimeError(f"Raw wells missing exact fold assignment: {missing_fold_wells[:20]}")
needed = [well for well in wells if fold_by_well[well] in (2, 3, 4, 5)]
records = {}
failures = []
for number, well in enumerate(needed, 1):
    try:
        records[well] = V["_load_record"](well)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(needed):
        print(f"P1 load {number}/{len(needed)} success={len(records)} failures={len(failures)}", flush=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)

prior_records = [records[w] for w in needed if w in records and fold_by_well[w] in PRIOR_FOLDS]
prior = V["_fit_prior"](prior_records, float(N["SEGMENT_LENGTH_FT"]))
fold3_records = [records[w] for w in needed if w in records and fold_by_well[w] == SELECTION_FOLD]
fold2_records = [records[w] for w in needed if w in records and fold_by_well[w] == APPLICATION_FOLD]
ordered_fold3 = sorted(
    fold3_records,
    key=lambda record: hashlib.sha256(f"physics-p1:{record['well']}".encode("ascii")).digest(),
)
discovery_records = ordered_fold3[:DISCOVERY_WELLS]
confirmation_records = ordered_fold3[DISCOVERY_WELLS:]


def decode_records(local_records, label):
    output = []
    for number, record in enumerate(local_records, 1):
        try:
            output.append(build_payload(record, prior))
        except Exception as exc:
            failures.append({"well": record["well"], "stage": label, "reason": repr(exc)})
        if number % 12 == 0 or number == len(local_records):
            print(f"P1 {label} {number}/{len(local_records)} success={len(output)}", flush=True)
    return output


discovery_payloads = decode_records(discovery_records, "fold3_discovery")
scan = []
for margin in MARGINS_FT:
    for dose in DOSES:
        frame = assemble(discovery_payloads, margin, dose)
        result = metrics(frame)
        null_result = metrics(frame, "null")
        scan.append(
            {
                "margin_ft": margin,
                "dose": dose,
                **result,
                "gain_vs_null": null_result["rmse"] - result["rmse"],
                "activation_fraction": float(frame.groupby("well")["active"].first().mean()),
            }
        )
scan = pd.DataFrame(scan).sort_values(["rmse", "margin_ft", "dose"], kind="stable")
selected = scan.iloc[0]
margin = float(selected["margin_ft"])
dose = float(selected["dose"])
print("\nP1 DISCOVERY SCAN")
print(scan[["margin_ft", "dose", "rmse", "far_gt2000_rmse", "gain_vs_null", "activation_fraction"]].to_string(index=False), flush=True)
print("P1 SELECTED", {"margin_ft": margin, "dose": dose}, flush=True)

confirmation_payloads = decode_records(confirmation_records, "fold3_confirmation")
application_payloads = decode_records(fold2_records, "fold2_application")
confirmation = assemble(confirmation_payloads, margin, dose)
application = assemble(application_payloads, margin, dose)
confirmation_shuffle = assemble(confirmation_payloads, margin, dose, shuffle_gate=True, seed=20260728)
application_shuffle = assemble(application_payloads, margin, dose, shuffle_gate=True, seed=20260729)
confirmation["fold"] = SELECTION_FOLD
application["fold"] = APPLICATION_FOLD
endpoint = pd.concat([confirmation, application], ignore_index=True)

summary = {
    "discovery": metrics(assemble(discovery_payloads, margin, dose)),
    "confirmation_null": metrics(confirmation, "null"),
    "confirmation_gr_map": metrics(confirmation, "gr_map"),
    "confirmation": metrics(confirmation),
    "confirmation_gate_shuffle": metrics(confirmation_shuffle),
    "application_null": metrics(application, "null"),
    "application_gr_map": metrics(application, "gr_map"),
    "application": metrics(application),
    "application_gate_shuffle": metrics(application_shuffle),
    "confirmation_activation_fraction": float(confirmation.groupby("well")["active"].first().mean()),
    "application_activation_fraction": float(application.groupby("well")["active"].first().mean()),
}

base_audit = None
if BASE_PATH.is_file():
    base_frame = pd.read_parquet(BASE_PATH, columns=["id", "well", "fold", "tvt_true", "md_since", "current_dose_0p20"])
    aligned = base_frame.merge(
        endpoint[["id", "prediction", "fold"]].rename(columns={"fold": "physics_fold"}),
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if not aligned["fold"].eq(aligned["physics_fold"]).all():
        mismatch = aligned.loc[~aligned["fold"].eq(aligned["physics_fold"]), ["well", "fold", "physics_fold"]].head(20)
        raise RuntimeError(f"Physics/base fold lineage mismatch:\n{mismatch}")
    base = aligned["current_dose_0p20"].to_numpy(np.float64)
    physics = aligned["prediction"].to_numpy(np.float64)
    truth = aligned["tvt_true"].to_numpy(np.float64)
    error_base = base - truth
    error_physics = physics - truth
    rho = float(np.corrcoef(error_base, error_physics)[0, 1])
    selection_mask = aligned["fold"].eq(SELECTION_FOLD).to_numpy()
    application_mask = aligned["fold"].eq(APPLICATION_FOLD).to_numpy()
    blend_rows = []
    for weight in BLEND_WEIGHTS:
        blend = (1.0 - weight) * base + weight * physics
        rmse = float(np.sqrt(np.mean((blend[selection_mask] - truth[selection_mask]) ** 2)))
        blend_rows.append({"physics_weight": float(weight), "fold3_rmse": rmse})
    blend_scan = pd.DataFrame(blend_rows).sort_values("fold3_rmse", kind="stable")
    weight = float(blend_scan.iloc[0]["physics_weight"])
    blend = (1.0 - weight) * base + weight * physics
    base_fold2_rmse = float(np.sqrt(np.mean(error_base[application_mask] ** 2)))
    blend_fold2_rmse = float(np.sqrt(np.mean((blend[application_mask] - truth[application_mask]) ** 2)))
    base_audit = {
        "aligned_rows": int(len(aligned)),
        "aligned_wells": int(aligned["well"].nunique()),
        "signed_error_rho": rho,
        "selected_weight_on_fold3": weight,
        "fold2_base_rmse": base_fold2_rmse,
        "fold2_blend_rmse": blend_fold2_rmse,
        "fold2_blend_gain": base_fold2_rmse - blend_fold2_rmse,
    }
    blend_scan.to_csv(OUT_DIR / "fold3_blend_scan.csv", index=False)

endpoint.to_parquet(OUT_DIR / "physics_gr_null_replay_oof.parquet", index=False, compression="zstd")
scan.to_csv(OUT_DIR / "discovery_scan.csv", index=False)
pd.DataFrame(
    [
        {
            "well": payload["well"],
            "fold": fold_by_well[payload["well"]],
            "null_name": payload["null_name"],
            "replay_null_rmse": payload["replay_null_rmse"],
            "replay_gr_rmse": payload["replay_gr_rmse"],
            "replay_gain": payload["replay_gain"],
            "replay_payloads": payload["replay_payloads"],
            "bank_paths": payload["bank_paths"],
        }
        for payload in discovery_payloads + confirmation_payloads + application_payloads
    ]
).to_csv(OUT_DIR / "well_replay_diagnostics.csv", index=False)

confirmation_gain = summary["confirmation_null"]["rmse"] - summary["confirmation"]["rmse"]
application_gain = summary["application_null"]["rmse"] - summary["application"]["rmse"]
confirmation_shuffle_loss = summary["confirmation_gate_shuffle"]["rmse"] - summary["confirmation"]["rmse"]
application_shuffle_loss = summary["application_gate_shuffle"]["rmse"] - summary["application"]["rmse"]
checks = {
    "confirmation_gain_vs_null_ge_0p30": confirmation_gain >= 0.30,
    "application_gain_vs_null_ge_0p30": application_gain >= 0.30,
    "confirmation_rmse_le_14": summary["confirmation"]["rmse"] <= 14.0,
    "application_rmse_le_14": summary["application"]["rmse"] <= 14.0,
    "application_activation_between_0p10_and_0p90": 0.10 <= summary["application_activation_fraction"] <= 0.90,
    "confirmation_gate_shuffle_loses_ge_0p10": confirmation_shuffle_loss >= 0.10,
    "application_gate_shuffle_loses_ge_0p10": application_shuffle_loss >= 0.10,
    "confirmation_p95_not_worse_than_null_by_0p20": summary["confirmation"]["p95_well_rmse"] <= summary["confirmation_null"]["p95_well_rmse"] + 0.20,
    "application_p95_not_worse_than_null_by_0p20": summary["application"]["p95_well_rmse"] <= summary["application_null"]["p95_well_rmse"] + 0.20,
}
if base_audit is not None:
    checks.update(
        {
            "error_rho_le_0p75": base_audit["signed_error_rho"] <= 0.75,
            "fold2_blend_gain_ge_0p05": base_audit["fold2_blend_gain"] >= 0.05,
            "nonzero_blend_weight": base_audit["selected_weight_on_fold3"] >= 0.025,
        }
    )
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": EXPERIMENT,
    "stage": "raw_gr_beam_with_visible_prefix_null_model_activation",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "prior_folds": PRIOR_FOLDS,
        "selection_fold": SELECTION_FOLD,
        "selection_wells": DISCOVERY_WELLS,
        "confirmation_wells": len(confirmation_payloads),
        "application_fold": APPLICATION_FOLD,
        "fold1_touched": False,
        "fold_source": FOLD_PATH,
    },
    "arm": ARM,
    "replay_backoffs_ft": REPLAY_BACKOFFS_FT,
    "selected": {"margin_ft": margin, "dose": dose},
    "summary": summary,
    "confirmation_gain_vs_null": confirmation_gain,
    "application_gain_vs_null": application_gain,
    "confirmation_gate_shuffle_loss": confirmation_shuffle_loss,
    "application_gate_shuffle_loss": application_shuffle_loss,
    "base_6p377_audit": base_audit,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> retain as the physics endpoint and package one frozen low-dose blend"
        if strict_pass
        else "FAIL -> visible-prefix GR replay cannot protect flat wells; close this physics assimilation family"
    ),
}
write_json(OUT_DIR / "result.json", verdict)
print("\nPHYSICS-P1 SUMMARY")
print(pd.DataFrame([{ "split": key, **value } for key, value in summary.items() if isinstance(value, dict)]).to_string(index=False), flush=True)
print("\nPHYSICS-P1 FINAL VERDICT")
print(json.dumps(clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
