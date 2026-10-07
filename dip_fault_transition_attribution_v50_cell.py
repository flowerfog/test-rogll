"""Heavy-tailed dip/fault transition attribution on a frozen V3 path bank.

V49 showed that major formation boundaries reject generic shifted paths but do
not distinguish the aliases actually selected by the lattice.  This cell tests
the other independent mechanism highlighted by the literature: an explicit
dip state with a smooth background process and sparse heavy-tailed fault jumps.

The existing V3 score contains a 0.1-weight Gaussian dip-change penalty.  We
remove that term exactly for every retained path, then rescore the same bank
with Gaussian, Student-t and Gaussian/Cauchy-mixture transition families.  A
within-well permutation of transition penalties is the matched causal control.
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
    candidates = [
        Path.cwd() / "rawdata_lattice_mode_union_v3_cell.py",
        Path("/root/autodl-tmp/rawdata_lattice_mode_union_v3_cell.py"),
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            "Upload rawdata_lattice_mode_union_v3_cell.py and its V1/V2 dependencies"
        )
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
EXPERIMENT = "dip_fault_transition_attribution_v50"
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
OLD_GAUSSIAN_WEIGHT = 0.1
PRIMARY_ARM = {
    "id": "raw_clip_b512_shape4",
    "beam": 512,
    "diversity_ft": 4.0,
    "reference": "raw",
    "likelihood": "clipped_gaussian",
    "history_diversity": True,
}
RANDOM_STATE = 20260727


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )


def _configs():
    output = [
        {"id": "emission_only", "family": "none", "weight": 0.0},
        {"id": "original_gaussian_w0.1", "family": "gaussian", "weight": 0.1},
    ]
    for weight in (0.03, 0.3, 1.0):
        output.append({"id": f"gaussian_w{weight:g}", "family": "gaussian", "weight": weight})
    for scale in (0.5, 1.0, 2.0, 4.0):
        for weight in (0.1, 0.3, 1.0):
            output.append(
                {
                    "id": f"student_s{scale:g}_w{weight:g}",
                    "family": "student",
                    "scale": scale,
                    "weight": weight,
                }
            )
    for fault_probability in (0.01, 0.03, 0.10):
        for weight in (0.1, 0.3, 1.0):
            output.append(
                {
                    "id": f"mixture_p{fault_probability:g}_w{weight:g}",
                    "family": "mixture",
                    "fault_probability": fault_probability,
                    "fault_scale": 4.0,
                    "weight": weight,
                }
            )
    return output


CONFIGS = _configs()
CONFIG_BY_ID = {config["id"]: config for config in CONFIGS}


def _transition_z(paths, prior):
    rates = np.asarray(prior["rate_grid"], np.float64)
    selected_rates = rates[np.asarray(paths, np.int64)]
    previous = np.empty_like(selected_rates)
    previous[:, 0] = float(prior["rate_centre"])
    previous[:, 1:] = selected_rates[:, :-1]
    return (
        selected_rates - previous - float(prior["change_centre"])
    ) / max(float(prior["change_scale"]), 1e-8)


def _transition_penalty(z, config):
    family = config["family"]
    if family == "none":
        return np.zeros(z.shape[0], np.float64)
    if family == "gaussian":
        point = z**2
    elif family == "student":
        scale = float(config["scale"])
        point = np.log1p((z / scale) ** 2)
    elif family == "mixture":
        probability = float(config["fault_probability"])
        fault_scale = float(config["fault_scale"])
        normal_log = math.log1p(-probability) - 0.5 * z**2
        fault_log = (
            math.log(probability)
            - math.log(fault_scale)
            - np.log1p((z / fault_scale) ** 2)
        )
        point = -np.logaddexp(normal_log, fault_log)
    else:
        raise ValueError(config)
    return float(config["weight"]) * np.sum(point, axis=1)


def _payload(record, prior):
    scores, paths, segments = B["_beam_bank_v3"](record, prior, PRIMARY_ARM)
    eval_rows = record["eval_rows"]
    truth = np.asarray(record["truth"][eval_rows], np.float64)
    distance = np.asarray(record["md"][eval_rows] - record["anchor_md"], np.float64)
    predictions = np.asarray(
        [
            V2["_path_prediction"](record, prior, path, segments)[eval_rows]
            for path in paths
        ],
        np.float64,
    )
    error = predictions - truth[None, :]
    sse = np.sum(error**2, axis=1)
    far = distance > 2000.0
    far_sse = np.sum(error[:, far] ** 2, axis=1) if np.any(far) else np.zeros(len(paths))
    z = _transition_z(paths, prior)
    old_penalty = OLD_GAUSSIAN_WEIGHT * np.sum(z**2, axis=1)
    emission_score = np.asarray(scores, np.float64) - old_penalty
    reconstructed_score = emission_score + old_penalty
    reconstruction_gap = float(
        np.max(np.abs(reconstructed_score - np.asarray(scores, np.float64)))
    )
    if reconstruction_gap > 1e-8 or int(np.argmin(reconstructed_score)) != 0:
        raise RuntimeError(
            f"{record['well']}: V3 transition-score reconstruction failed "
            f"(gap={reconstruction_gap:.3g}, argmin={np.argmin(reconstructed_score)})"
        )
    return {
        "well": str(record["well"]),
        "rows": int(len(truth)),
        "far_rows": int(far.sum()),
        "sse": sse,
        "far_sse": far_sse,
        "emission_score": emission_score,
        "z": z,
    }


def _build_payloads(records, prior, label):
    output, failures = [], []
    for number, record in enumerate(records, 1):
        try:
            output.append(_payload(record, prior))
        except Exception as exc:
            failures.append({"well": str(record["well"]), "reason": repr(exc)})
        if number % 12 == 0 or number == len(records):
            print(f"V50 {label} banks {number}/{len(records)} failures={len(failures)}", flush=True)
    pd.DataFrame(failures).to_csv(OUT_DIR / f"{label}_failures.csv", index=False)
    return output


def _evaluate(payloads, configs, shuffled=False, seed=RANDOM_STATE, label="aligned"):
    rng = np.random.default_rng(seed)
    rows, choices = [], []
    for config in configs:
        total_rows = total_far_rows = changed = 0
        total_sse = total_far_sse = oracle_sse = 0.0
        for payload in payloads:
            penalty = _transition_penalty(payload["z"], config)
            if shuffled and len(penalty) > 1:
                penalty = penalty[rng.permutation(len(penalty))]
            objective = payload["emission_score"] + penalty
            selected = int(np.argmin(objective))
            emission_selected = int(np.argmin(payload["emission_score"]))
            changed += int(selected != emission_selected)
            total_rows += payload["rows"]
            total_far_rows += payload["far_rows"]
            total_sse += float(payload["sse"][selected])
            total_far_sse += float(payload["far_sse"][selected])
            oracle_sse += float(np.min(payload["sse"]))
            choices.append(
                {
                    "control": label,
                    "config": config["id"],
                    "well": payload["well"],
                    "selected_path": selected,
                    "emission_selected_path": emission_selected,
                    "selected_sse": float(payload["sse"][selected]),
                    "oracle_sse": float(np.min(payload["sse"])),
                }
            )
        rows.append(
            {
                "control": label,
                "config": config["id"],
                "rows": int(total_rows),
                "wells": int(len(payloads)),
                "rmse": math.sqrt(total_sse / total_rows),
                "far_gt2000_rmse": (
                    math.sqrt(total_far_sse / total_far_rows) if total_far_rows else np.nan
                ),
                "bank_oracle_rmse": math.sqrt(oracle_sse / total_rows),
                "changed_wells_vs_emission": int(changed),
                "changed_fraction_vs_emission": float(changed / max(len(payloads), 1)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(choices)


def _select_config(scan):
    baseline = scan.loc[scan["config"].eq("original_gaussian_w0.1")].iloc[0]
    candidates = scan.loc[~scan["config"].eq("original_gaussian_w0.1")].copy()
    candidates["gain"] = float(baseline["rmse"]) - candidates["rmse"]
    candidates["far_gain"] = float(baseline["far_gt2000_rmse"]) - candidates["far_gt2000_rmse"]
    candidates["objective"] = candidates["gain"] + 0.5 * candidates["far_gain"]
    return str(candidates.sort_values(["objective", "gain"], ascending=False).iloc[0]["config"])


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
        print(f"V50 load {number}/{len(needed)} failures={len(failures)}", flush=True)
pd.DataFrame(failures).to_csv(OUT_DIR / "record_failures.csv", index=False)

prior_records = [
    records[well]
    for well in needed
    if well in records and fold_by_well[well] in PRIOR_FOLDS
]
fold3_records = [
    records[well]
    for well in needed
    if well in records and fold_by_well[well] == SELECTION_FOLD
]
fold2_records = [
    records[well]
    for well in needed
    if well in records and fold_by_well[well] == APPLICATION_FOLD
]
prior = V["_fit_prior"](prior_records, SEGMENT_LENGTH_FT)
fold3_records = sorted(
    fold3_records,
    key=lambda record: hashlib.sha256(
        f"dip-fault-v50:{record['well']}".encode("ascii")
    ).digest(),
)
discovery_records = fold3_records[:DISCOVERY_WELLS]
confirmation_records = fold3_records[DISCOVERY_WELLS:]

discovery_payloads = _build_payloads(discovery_records, prior, "fold3_discovery")
discovery_scan, discovery_choices = _evaluate(discovery_payloads, CONFIGS)
selected_id = _select_config(discovery_scan)
selected_config = CONFIG_BY_ID[selected_id]
discovery_scan.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
discovery_choices.to_csv(OUT_DIR / "fold3_discovery_choices.csv", index=False)
print("\nV50 DISCOVERY SCAN")
print(discovery_scan.sort_values("rmse").head(20).to_string(index=False), flush=True)
print("V50 SELECTED", selected_config, flush=True)

confirmation_payloads = _build_payloads(confirmation_records, prior, "fold3_confirmation")
confirmation_scan, confirmation_choices = _evaluate(
    confirmation_payloads,
    [CONFIG_BY_ID["original_gaussian_w0.1"], selected_config],
    label="aligned",
)
shuffle_scan, shuffle_choices = _evaluate(
    confirmation_payloads,
    [selected_config],
    shuffled=True,
    seed=RANDOM_STATE + 3,
    label="transition_path_shuffle",
)
pd.concat([confirmation_scan, shuffle_scan], ignore_index=True).to_csv(
    OUT_DIR / "fold3_confirmation_scan.csv", index=False
)
pd.concat([confirmation_choices, shuffle_choices], ignore_index=True).to_csv(
    OUT_DIR / "fold3_confirmation_choices.csv", index=False
)

base_row = confirmation_scan.loc[
    confirmation_scan["config"].eq("original_gaussian_w0.1")
].iloc[0]
full_row = confirmation_scan.loc[confirmation_scan["config"].eq(selected_id)].iloc[0]
shuffle_row = shuffle_scan.iloc[0]
confirmation = {
    "base_rmse": float(base_row["rmse"]),
    "rmse": float(full_row["rmse"]),
    "gain": float(base_row["rmse"] - full_row["rmse"]),
    "base_far_rmse": float(base_row["far_gt2000_rmse"]),
    "far_rmse": float(full_row["far_gt2000_rmse"]),
    "far_gain": float(base_row["far_gt2000_rmse"] - full_row["far_gt2000_rmse"]),
    "shuffle_rmse": float(shuffle_row["rmse"]),
    "shuffle_loss": float(shuffle_row["rmse"] - full_row["rmse"]),
    "changed_wells": int(full_row["changed_wells_vs_emission"]),
    "changed_fraction": float(full_row["changed_fraction_vs_emission"]),
    "bank_oracle_rmse": float(full_row["bank_oracle_rmse"]),
}
checks = {
    "gain_ge_0p30": confirmation["gain"] >= 0.30,
    "far_gain_ge_0p40": confirmation["far_gain"] >= 0.40,
    "shuffle_loses_ge_0p20": confirmation["shuffle_loss"] >= 0.20,
    "changed_fraction_ge_0p05": confirmation["changed_fraction"] >= 0.05,
}
mechanism_pass = bool(all(checks.values()))

application = None
if mechanism_pass:
    fold2_payloads = _build_payloads(fold2_records, prior, "fold2_application")
    fold2_scan, fold2_choices = _evaluate(
        fold2_payloads,
        [CONFIG_BY_ID["original_gaussian_w0.1"], selected_config],
        label="aligned",
    )
    fold2_shuffle, fold2_shuffle_choices = _evaluate(
        fold2_payloads,
        [selected_config],
        shuffled=True,
        seed=RANDOM_STATE + 2,
        label="transition_path_shuffle",
    )
    pd.concat([fold2_scan, fold2_shuffle], ignore_index=True).to_csv(
        OUT_DIR / "fold2_application_scan.csv", index=False
    )
    pd.concat([fold2_choices, fold2_shuffle_choices], ignore_index=True).to_csv(
        OUT_DIR / "fold2_application_choices.csv", index=False
    )
    fold2_base = fold2_scan.loc[
        fold2_scan["config"].eq("original_gaussian_w0.1")
    ].iloc[0]
    fold2_full = fold2_scan.loc[fold2_scan["config"].eq(selected_id)].iloc[0]
    fold2_control = fold2_shuffle.iloc[0]
    application = {
        "base_rmse": float(fold2_base["rmse"]),
        "rmse": float(fold2_full["rmse"]),
        "gain": float(fold2_base["rmse"] - fold2_full["rmse"]),
        "base_far_rmse": float(fold2_base["far_gt2000_rmse"]),
        "far_rmse": float(fold2_full["far_gt2000_rmse"]),
        "far_gain": float(fold2_base["far_gt2000_rmse"] - fold2_full["far_gt2000_rmse"]),
        "shuffle_rmse": float(fold2_control["rmse"]),
        "shuffle_loss": float(fold2_control["rmse"] - fold2_full["rmse"]),
    }

strict_pass = bool(
    mechanism_pass
    and application is not None
    and application["gain"] >= 0.20
    and application["far_gain"] >= 0.25
    and application["shuffle_loss"] >= 0.15
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "frozen_lattice_dip_fault_transition_attribution",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "uses_neighbor_spatial_cp12_gold": False,
    "protocol": {
        "prior_folds": list(PRIOR_FOLDS),
        "selection_fold": SELECTION_FOLD,
        "selection_wells": len(discovery_payloads),
        "confirmation_wells": len(confirmation_payloads),
        "application_fold": APPLICATION_FOLD,
        "fold1_touched": False,
    },
    "source": str(V3_SOURCE),
    "arm": PRIMARY_ARM,
    "selected_config": selected_config,
    "confirmation": confirmation,
    "checks": checks,
    "mechanism_pass": mechanism_pass,
    "application": application,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> put the fault-mixture transition inside beam expansion and retrain no network"
        if strict_pass
        else "FAIL -> close transition-family rescoring; retained-bank likelihood, not transition shape, controls the wrong mode"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV50 CONFIRMATION")
print(pd.concat([confirmation_scan, shuffle_scan], ignore_index=True).to_string(index=False), flush=True)
print("\nV50 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
