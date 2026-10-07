"""Put heavy-tailed dip transitions inside V3 beam expansion.

V50 found a real but small post-hoc Student-t transition effect.  Because its
bank was pruned under the old Gaussian transition, this final attribution test
regenerates every bank with the candidate transition active before pruning.
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


def _bootstrap_v3():
    paths = [
        Path.cwd() / "rawdata_lattice_mode_union_v3_cell.py",
        Path("/root/autodl-tmp/rawdata_lattice_mode_union_v3_cell.py"),
    ]
    source = next((path for path in paths if path.is_file()), None)
    if source is None:
        raise FileNotFoundError("Upload the V1/V2/V3 raw lattice cells")
    text = source.read_text(encoding="utf-8")
    marker = "# --- V3 EXECUTION ---"
    if marker not in text:
        raise RuntimeError(f"V3 bootstrap marker missing from {source}")
    namespace = {"__name__": "raw_lattice_v3_definitions"}
    exec(compile(text.split(marker, 1)[0], str(source), "exec"), namespace)
    return namespace, source


B, V3_SOURCE = _bootstrap_v3()
V = B["V"]
V2 = B["B"]
DATA_ROOT = Path(B["DATA_ROOT"])
EXPERIMENT = "dip_fault_beam_expansion_attribution_v51"
if os.name != "nt" and Path("/root/autodl-tmp").exists():
    OUT_DIR = Path("/root/autodl-tmp/oof_exports") / EXPERIMENT
else:
    OUT_DIR = Path.cwd() / "output" / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRIOR_FOLDS = (4, 5)
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
DISCOVERY_WELLS = 24
SEGMENT_LENGTH_FT = 128.0
RANDOM_STATE = 20260727
ARM = {
    "id": "raw_clip_b512_shape4",
    "beam": 512,
    "diversity_ft": 4.0,
    "reference": "raw",
    "likelihood": "clipped_gaussian",
    "history_diversity": True,
}
BASE = {"id": "gaussian_w0.1", "family": "gaussian", "weight": 0.1}
DISCOVERY_CONFIGS = [
    BASE,
    {"id": "student_s2_w1", "family": "student", "scale": 2.0, "weight": 1.0},
    {"id": "student_s4_w0.3", "family": "student", "scale": 4.0, "weight": 0.3},
    {"id": "student_s4_w1", "family": "student", "scale": 4.0, "weight": 1.0},
    {"id": "student_s8_w1", "family": "student", "scale": 8.0, "weight": 1.0},
    {
        "id": "mixture_p0.03_w1",
        "family": "mixture",
        "fault_probability": 0.03,
        "fault_scale": 4.0,
        "weight": 1.0,
    },
]
CONFIG_BY_ID = {config["id"]: config for config in DISCOVERY_CONFIGS}


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def _point_penalty(z, config):
    if config["family"] == "gaussian":
        point = z**2
    elif config["family"] == "student":
        point = np.log1p((z / float(config["scale"])) ** 2)
    elif config["family"] == "mixture":
        probability = float(config["fault_probability"])
        scale = float(config["fault_scale"])
        normal_log = math.log1p(-probability) - 0.5 * z**2
        fault_log = (
            math.log(probability) - math.log(scale) - np.log1p((z / scale) ** 2)
        )
        point = -np.logaddexp(normal_log, fault_log)
    else:
        raise ValueError(config)
    return float(config["weight"]) * point


def _seed(well, segment_index, config_id):
    digest = hashlib.sha256(
        f"v51:{well}:{segment_index}:{config_id}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _beam_bank(record, prior, config, shuffled=False):
    rates = np.asarray(prior["rate_grid"], np.float64)
    segments = V["_segments"](record, SEGMENT_LENGTH_FT)
    reference_gr = V2["_reference_for_record"](record, ARM["reference"])
    scores = np.array([0.0], np.float64)
    surfaces = np.array([record["anchor_surface"]], np.float64)
    history_surface = surfaces.copy()
    previous_rate = np.array([-1], np.int64)
    paths = np.empty((1, 0), np.int16)
    previous_md = record["anchor_md"]

    for segment_index, rows in enumerate(segments):
        loss, end_surface = V2["_loss_and_surface"](
            record,
            rows,
            previous_md,
            surfaces,
            rates,
            reference_gr,
            ARM["likelihood"],
        )
        previous = np.where(
            previous_rate[:, None] >= 0,
            rates[np.maximum(previous_rate[:, None], 0)],
            prior["rate_centre"],
        )
        z = (
            rates[None, :] - previous - prior["change_centre"]
        ) / max(float(prior["change_scale"]), 1e-8)
        transition = _point_penalty(z, config)
        if shuffled:
            rng = np.random.default_rng(_seed(record["well"], segment_index, config["id"]))
            transition = transition[:, rng.permutation(len(rates))]
        candidate_score = scores[:, None] + loss + transition

        if ARM["history_diversity"]:
            keep, next_history = B["_select_history_diverse"](
                candidate_score,
                end_surface,
                history_surface,
                int(ARM["beam"]),
                float(ARM["diversity_ft"]),
                update_history=((segment_index + 1) % 4 == 0),
            )
        else:
            keep = V2["_select_diverse"](
                candidate_score,
                end_surface,
                int(ARM["beam"]),
                float(ARM["diversity_ft"]),
            )
            inherited = np.repeat(history_surface, len(rates))
            next_history = inherited[keep]

        parent = keep // len(rates)
        rate_index = keep % len(rates)
        scores = candidate_score.ravel()[keep]
        surfaces = end_surface.ravel()[keep]
        history_surface = next_history
        previous_rate = rate_index.astype(np.int64)
        paths = np.concatenate(
            [paths[parent], rate_index[:, None].astype(np.int16)], axis=1
        )
        previous_md = float(record["md"][rows][-1])
    return scores, paths, segments


def _one(record, prior, config, shuffled=False):
    scores, paths, segments = _beam_bank(record, prior, config, shuffled)
    eval_rows = record["eval_rows"]
    truth = np.asarray(record["truth"][eval_rows], np.float64)
    distance = np.asarray(record["md"][eval_rows] - record["anchor_md"], np.float64)
    far = distance > 2000.0
    sse, far_sse = [], []
    for path in paths:
        prediction = np.asarray(
            V2["_path_prediction"](record, prior, path, segments)[eval_rows],
            np.float64,
        )
        error = prediction - truth
        sse.append(float(np.sum(error**2)))
        far_sse.append(float(np.sum(error[far] ** 2)))
    selected = int(np.argmin(scores))
    return {
        "well": str(record["well"]),
        "rows": int(len(truth)),
        "far_rows": int(far.sum()),
        "sse": float(sse[selected]),
        "far_sse": float(far_sse[selected]),
        "oracle_sse": float(np.min(sse)),
        "selected": selected,
        "paths": int(len(paths)),
    }


def _evaluate(records, prior, configs, label="aligned", shuffled=False):
    summaries, details = [], []
    for config in configs:
        output, failures = [], []
        for number, record in enumerate(records, 1):
            try:
                output.append(_one(record, prior, config, shuffled))
            except Exception as exc:
                failures.append({"well": str(record["well"]), "reason": repr(exc)})
            if number % 12 == 0 or number == len(records):
                print(
                    f"V51 {label} {config['id']} {number}/{len(records)} "
                    f"failures={len(failures)}",
                    flush=True,
                )
        frame = pd.DataFrame(output)
        pd.DataFrame(failures).to_csv(
            OUT_DIR / f"{label}_{config['id']}_failures.csv", index=False
        )
        if frame.empty:
            raise RuntimeError(f"No successful wells for {label}/{config['id']}")
        total_rows = int(frame["rows"].sum())
        far_rows = int(frame["far_rows"].sum())
        summaries.append(
            {
                "control": label,
                "config": config["id"],
                "rows": total_rows,
                "wells": int(len(frame)),
                "rmse": math.sqrt(float(frame["sse"].sum()) / total_rows),
                "far_gt2000_rmse": math.sqrt(
                    float(frame["far_sse"].sum()) / far_rows
                ) if far_rows else np.nan,
                "bank_oracle_rmse": math.sqrt(
                    float(frame["oracle_sse"].sum()) / total_rows
                ),
            }
        )
        frame.insert(0, "control", label)
        frame.insert(1, "config", config["id"])
        details.append(frame)
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def _select(scan):
    base = scan.loc[scan["config"].eq(BASE["id"])].iloc[0]
    candidates = scan.loc[~scan["config"].eq(BASE["id"])].copy()
    candidates["gain"] = float(base["rmse"]) - candidates["rmse"]
    candidates["far_gain"] = float(base["far_gt2000_rmse"]) - candidates["far_gt2000_rmse"]
    candidates["oracle_gain"] = float(base["bank_oracle_rmse"]) - candidates["bank_oracle_rmse"]
    candidates["objective"] = (
        candidates["gain"] + 0.5 * candidates["far_gain"] + 0.25 * candidates["oracle_gain"]
    )
    return str(candidates.sort_values("objective", ascending=False).iloc[0]["config"])


started = time.time()
wells = [
    path.name.split("__", 1)[0]
    for path in sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
]
folds = V["_stable_folds"](wells)
fold_by_well = dict(zip(folds["well"].astype(str), folds["fold"].astype(int)))
needed = [well for well in wells if fold_by_well.get(well) in (2, 3, 4, 5)]
records, failures = {}, []
for number, well in enumerate(needed, 1):
    try:
        records[well] = V["_load_record"](well)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(needed):
        print(f"V51 load {number}/{len(needed)} failures={len(failures)}", flush=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)

prior_records = [records[w] for w in needed if w in records and fold_by_well[w] in PRIOR_FOLDS]
fold3 = [records[w] for w in needed if w in records and fold_by_well[w] == SELECTION_FOLD]
fold2 = [records[w] for w in needed if w in records and fold_by_well[w] == APPLICATION_FOLD]
prior = V["_fit_prior"](prior_records, SEGMENT_LENGTH_FT)
fold3 = sorted(
    fold3,
    key=lambda record: hashlib.sha256(
        f"dip-fault-v51:{record['well']}".encode("ascii")
    ).digest(),
)
discovery = fold3[:DISCOVERY_WELLS]
confirmation_records = fold3[DISCOVERY_WELLS:]

discovery_scan, discovery_details = _evaluate(discovery, prior, DISCOVERY_CONFIGS)
selected_id = _select(discovery_scan)
selected = CONFIG_BY_ID[selected_id]
discovery_scan.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
discovery_details.to_csv(OUT_DIR / "fold3_discovery_details.csv", index=False)
print("\nV51 DISCOVERY")
print(discovery_scan.sort_values("rmse").to_string(index=False), flush=True)
print("V51 SELECTED", selected, flush=True)

confirmation_scan, confirmation_details = _evaluate(
    confirmation_records, prior, [BASE, selected]
)
shuffle_scan, shuffle_details = _evaluate(
    confirmation_records, prior, [selected], label="transition_state_shuffle", shuffled=True
)
pd.concat([confirmation_scan, shuffle_scan], ignore_index=True).to_csv(
    OUT_DIR / "fold3_confirmation_scan.csv", index=False
)
pd.concat([confirmation_details, shuffle_details], ignore_index=True).to_csv(
    OUT_DIR / "fold3_confirmation_details.csv", index=False
)
base_row = confirmation_scan.loc[confirmation_scan["config"].eq(BASE["id"])].iloc[0]
full_row = confirmation_scan.loc[confirmation_scan["config"].eq(selected_id)].iloc[0]
control_row = shuffle_scan.iloc[0]
confirmation = {
    "base_rmse": float(base_row["rmse"]),
    "rmse": float(full_row["rmse"]),
    "gain": float(base_row["rmse"] - full_row["rmse"]),
    "base_far_rmse": float(base_row["far_gt2000_rmse"]),
    "far_rmse": float(full_row["far_gt2000_rmse"]),
    "far_gain": float(base_row["far_gt2000_rmse"] - full_row["far_gt2000_rmse"]),
    "base_oracle_rmse": float(base_row["bank_oracle_rmse"]),
    "oracle_rmse": float(full_row["bank_oracle_rmse"]),
    "oracle_gain": float(base_row["bank_oracle_rmse"] - full_row["bank_oracle_rmse"]),
    "shuffle_rmse": float(control_row["rmse"]),
    "shuffle_loss": float(control_row["rmse"] - full_row["rmse"]),
}
checks = {
    "gain_ge_0p20": confirmation["gain"] >= 0.20,
    "far_gain_ge_0p30": confirmation["far_gain"] >= 0.30,
    "shuffle_loses_ge_0p20": confirmation["shuffle_loss"] >= 0.20,
    "oracle_not_worse_by_gt_0p10": confirmation["oracle_gain"] >= -0.10,
}
mechanism_pass = bool(all(checks.values()))

application = None
if mechanism_pass:
    application_scan, application_details = _evaluate(fold2, prior, [BASE, selected])
    application_shuffle, application_shuffle_details = _evaluate(
        fold2, prior, [selected], label="transition_state_shuffle", shuffled=True
    )
    pd.concat([application_scan, application_shuffle], ignore_index=True).to_csv(
        OUT_DIR / "fold2_application_scan.csv", index=False
    )
    pd.concat([application_details, application_shuffle_details], ignore_index=True).to_csv(
        OUT_DIR / "fold2_application_details.csv", index=False
    )
    base2 = application_scan.loc[application_scan["config"].eq(BASE["id"])].iloc[0]
    full2 = application_scan.loc[application_scan["config"].eq(selected_id)].iloc[0]
    shuffled2 = application_shuffle.iloc[0]
    application = {
        "base_rmse": float(base2["rmse"]),
        "rmse": float(full2["rmse"]),
        "gain": float(base2["rmse"] - full2["rmse"]),
        "far_gain": float(base2["far_gt2000_rmse"] - full2["far_gt2000_rmse"]),
        "shuffle_loss": float(shuffled2["rmse"] - full2["rmse"]),
    }

strict_pass = bool(
    mechanism_pass
    and application is not None
    and application["gain"] >= 0.15
    and application["far_gain"] >= 0.20
    and application["shuffle_loss"] >= 0.15
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "heavy_tail_transition_inside_beam_expansion",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery),
        "confirmation_wells": len(confirmation_records),
        "application_fold": APPLICATION_FOLD,
        "fold1_touched": False,
    },
    "source": V3_SOURCE,
    "arm": ARM,
    "selected_config": selected,
    "confirmation": confirmation,
    "checks": checks,
    "mechanism_pass": mechanism_pass,
    "application": application,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> retain as a distinct lattice arm and test union oracle/scorer"
        if strict_pass
        else "FAIL -> heavy-tail dip/fault transition is causally real but not a useful alias selector; close branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV51 CONFIRMATION")
print(pd.concat([confirmation_scan, shuffle_scan], ignore_index=True).to_string(index=False))
print("\nV51 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
