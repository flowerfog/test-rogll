"""Formation-zone attribution on a frozen raw-data lattice bank.

This is the first experiment after the V48 endpoint/risk gate failed.  It tests
the genuinely new lever identified by the literature review: use typewell
formation intervals to eliminate parallel GR aliases *inside* a multimodal
path bank.  It does not clip an already selected prediction.

Fold 4/5 fit the existing transition prior.  A deterministic 24-well subset of
Fold 3 selects one boundary rule, the remaining Fold-3 wells confirm it, and
Fold 2 is evaluated only after confirmation passes.  Fold 1 remains untouched.
Truth is used for experiment selection/evaluation only, never as model input.
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
            "Upload rawdata_lattice_mode_union_v3_cell.py and its V1/V2 "
            "dependencies to /root/autodl-tmp"
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
EXPERIMENT = "formation_zone_lattice_attribution_v49"
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
PRIMARY_ARM = {
    "id": "raw_clip_b512_shape4",
    "beam": 512,
    "diversity_ft": 4.0,
    "reference": "raw",
    "likelihood": "clipped_gaussian",
    "history_diversity": True,
}
MAJOR_FORMATIONS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
ALIAS_OFFSETS_FT = (-48.0, -24.0, -12.0, 12.0, 24.0, 48.0)
UNCERTAINTIES_FT = (0.0, 4.0, 8.0, 12.0, 16.0)
SOFT_WEIGHTS = (0.03, 0.1, 0.3, 1.0, 3.0)
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


def _formation_corridor(well, anchor_tvt):
    path = DATA_ROOT / "train" / f"{well}__typewell.csv"
    typewell = pd.read_csv(path, usecols=["TVT", "Geology"])
    geology = typewell["Geology"].astype("string").str.strip()
    tvt = pd.to_numeric(typewell["TVT"], errors="coerce")
    tops = []
    for name in MAJOR_FORMATIONS:
        values = tvt[geology.eq(name)].dropna()
        if len(values):
            tops.append((float(values.min()), name))
    tops.sort()
    if len(tops) < 2 or not np.isfinite(anchor_tvt):
        return None
    values = np.asarray([item[0] for item in tops], np.float64)
    index = int(np.searchsorted(values, anchor_tvt, side="right") - 1)
    if index < 0 or index >= len(tops) - 1:
        return None
    low, low_name = tops[index]
    high, high_name = tops[index + 1]
    return {
        "zone": f"{low_name}-{high_name}",
        "low": float(low),
        "high": float(high),
        "width": float(high - low),
        "low_margin": float(anchor_tvt - low),
        "high_margin": float(high - anchor_tvt),
    }


def _outside_distance(predictions, low, high, uncertainty):
    low = float(low) - float(uncertainty)
    high = float(high) + float(uncertainty)
    return np.maximum(low - predictions, 0.0) + np.maximum(predictions - high, 0.0)


def _path_payload(record, prior, corridor):
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
    if predictions.ndim != 2 or predictions.shape[1] != len(truth):
        raise RuntimeError(f"{record['well']}: invalid path bank {predictions.shape}")
    if not np.isfinite(predictions).all() or not np.isfinite(scores).all():
        raise RuntimeError(f"{record['well']}: non-finite bank")

    raw = np.asarray(scores, np.float64)
    scale = max(float(np.quantile(raw, 0.90) - np.min(raw)), 1e-6)
    score01 = (raw - np.min(raw)) / scale
    error = predictions - truth[None, :]
    sse = np.sum(error**2, axis=1)
    far = distance > 2000.0
    far_sse = np.sum(error[:, far] ** 2, axis=1) if np.any(far) else np.zeros(len(paths))
    return {
        "well": str(record["well"]),
        "rows": int(len(truth)),
        "far_rows": int(far.sum()),
        "truth": truth,
        "predictions": predictions,
        "score01": score01,
        "sse": sse,
        "far_sse": far_sse,
        "corridor": corridor,
    }


def _choose(payload, config, corridor):
    predictions = payload["predictions"]
    outside = _outside_distance(
        predictions,
        corridor["low"],
        corridor["high"],
        config["uncertainty_ft"],
    )
    mean_outside = np.mean(outside, axis=1)
    max_outside = np.max(outside, axis=1)
    if config["mode"] == "none":
        return int(np.argmin(payload["score01"]))
    if config["mode"] == "hard":
        legal = max_outside <= 1e-9
        if np.any(legal):
            indices = np.flatnonzero(legal)
            return int(indices[np.argmin(payload["score01"][indices])])
        # A hard rule must still return a path. Prefer least violation, then GR.
        objective = max_outside / 24.0 + 1e-3 * payload["score01"]
        return int(np.argmin(objective))
    if config["mode"] == "soft":
        penalty = np.mean((outside / 24.0) ** 2, axis=1)
        objective = payload["score01"] + float(config["weight"]) * penalty
        return int(np.argmin(objective))
    raise ValueError(config)


def _oracle_indices(payload, config, corridor):
    if config["mode"] != "hard":
        return np.arange(len(payload["sse"]), dtype=np.int64)
    outside = _outside_distance(
        payload["predictions"],
        corridor["low"],
        corridor["high"],
        config["uncertainty_ft"],
    )
    legal = np.max(outside, axis=1) <= 1e-9
    if np.any(legal):
        return np.flatnonzero(legal)
    minimum_violation = np.min(np.max(outside, axis=1))
    return np.flatnonzero(np.max(outside, axis=1) <= minimum_violation + 1e-9)


def _configs():
    output = [{"id": "unconstrained", "mode": "none", "uncertainty_ft": 0.0, "weight": 0.0}]
    for uncertainty in UNCERTAINTIES_FT:
        output.append(
            {
                "id": f"hard_u{uncertainty:g}",
                "mode": "hard",
                "uncertainty_ft": float(uncertainty),
                "weight": 0.0,
            }
        )
        for weight in SOFT_WEIGHTS:
            output.append(
                {
                    "id": f"soft_u{uncertainty:g}_w{weight:g}",
                    "mode": "soft",
                    "uncertainty_ft": float(uncertainty),
                    "weight": float(weight),
                }
            )
    return output


CONFIGS = _configs()
CONFIG_BY_ID = {config["id"]: config for config in CONFIGS}


def _evaluate(payloads, configs, corridor_override=None, label="aligned"):
    rows = []
    choices = []
    for config in configs:
        total_rows = total_far_rows = 0
        total_sse = total_far_sse = 0.0
        oracle_sse = 0.0
        for payload in payloads:
            corridor = (
                corridor_override[payload["well"]]
                if corridor_override is not None
                else payload["corridor"]
            )
            selected = _choose(payload, config, corridor)
            total_rows += payload["rows"]
            total_far_rows += payload["far_rows"]
            total_sse += float(payload["sse"][selected])
            total_far_sse += float(payload["far_sse"][selected])
            oracle_indices = _oracle_indices(payload, config, corridor)
            oracle_sse += float(np.min(payload["sse"][oracle_indices]))
            choices.append(
                {
                    "control": label,
                    "config": config["id"],
                    "well": payload["well"],
                    "selected_path": selected,
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
                    math.sqrt(total_far_sse / total_far_rows)
                    if total_far_rows
                    else np.nan
                ),
                "bank_oracle_rmse": math.sqrt(oracle_sse / total_rows),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(choices)


def _alias_audit(payloads, corridor_override=None, label="aligned"):
    rows = []
    for payload in payloads:
        corridor = (
            corridor_override[payload["well"]]
            if corridor_override is not None
            else payload["corridor"]
        )
        truth = payload["truth"]
        for uncertainty in UNCERTAINTIES_FT:
            truth_legal = bool(
                np.max(
                    _outside_distance(
                        truth[None, :], corridor["low"], corridor["high"], uncertainty
                    )
                )
                <= 1e-9
            )
            row = {
                "control": label,
                "well": payload["well"],
                "zone": corridor["zone"],
                "uncertainty_ft": float(uncertainty),
                "truth_legal": truth_legal,
            }
            for offset in ALIAS_OFFSETS_FT:
                rejected = bool(
                    np.max(
                        _outside_distance(
                            (truth + offset)[None, :],
                            corridor["low"],
                            corridor["high"],
                            uncertainty,
                        )
                    )
                    > 1e-9
                )
                row[f"alias_{offset:+g}_rejected"] = rejected
            rows.append(row)
    return pd.DataFrame(rows)


def _shuffle_corridors(payloads):
    metadata = pd.DataFrame(
        [
            {
                "well": payload["well"],
                "zone": payload["corridor"]["zone"],
                "anchor": float(payload["corridor"]["low"] + payload["corridor"]["low_margin"]),
                "low_margin": payload["corridor"]["low_margin"],
                "high_margin": payload["corridor"]["high_margin"],
            }
            for payload in payloads
        ]
    )
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = {}
    for zone, group in metadata.groupby("zone", sort=True):
        positions = group.index.to_numpy()
        donors = positions.copy()
        rng.shuffle(donors)
        for receiver, donor in zip(positions, donors):
            row = metadata.loc[receiver]
            source = metadata.loc[donor]
            anchor = float(row["anchor"])
            shuffled[str(row["well"])] = {
                "zone": str(zone),
                "low": anchor - float(source["low_margin"]),
                "high": anchor + float(source["high_margin"]),
                "width": float(source["low_margin"] + source["high_margin"]),
                "low_margin": float(source["low_margin"]),
                "high_margin": float(source["high_margin"]),
            }
    return shuffled


def _build_payloads(records, prior, label):
    payloads = []
    failures = []
    for number, record in enumerate(records, 1):
        try:
            anchor_tvt = float(record["tvt_input"][record["anchor_index"]])
            corridor = _formation_corridor(record["well"], anchor_tvt)
            if corridor is None:
                raise RuntimeError("formation corridor unavailable")
            payloads.append(_path_payload(record, prior, corridor))
        except Exception as exc:
            failures.append({"well": str(record["well"]), "reason": repr(exc)})
        if number % 12 == 0 or number == len(records):
            print(f"V49 {label} banks {number}/{len(records)} failures={len(failures)}", flush=True)
    pd.DataFrame(failures).to_csv(OUT_DIR / f"{label}_failures.csv", index=False)
    return payloads


def _select_config(scan):
    base = scan.loc[scan["config"].eq("unconstrained")].iloc[0]
    candidates = scan.loc[~scan["config"].eq("unconstrained")].copy()
    candidates["gain"] = float(base["rmse"]) - candidates["rmse"]
    candidates["far_gain"] = float(base["far_gt2000_rmse"]) - candidates["far_gt2000_rmse"]
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
records = {}
failures = []
for number, well in enumerate(needed, 1):
    try:
        records[well] = V["_load_record"](well)
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(needed):
        print(f"V49 load {number}/{len(needed)} failures={len(failures)}", flush=True)
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
        f"formation-v49:{record['well']}".encode("ascii")
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
print("\nV49 DISCOVERY SCAN")
print(discovery_scan.sort_values("rmse").head(20).to_string(index=False), flush=True)
print("V49 SELECTED", selected_config, flush=True)

confirmation_payloads = _build_payloads(confirmation_records, prior, "fold3_confirmation")
confirmation_shuffle = _shuffle_corridors(confirmation_payloads)
confirmation_scan, confirmation_choices = _evaluate(
    confirmation_payloads,
    [CONFIG_BY_ID["unconstrained"], selected_config],
    label="aligned",
)
shuffle_scan, shuffle_choices = _evaluate(
    confirmation_payloads,
    [selected_config],
    corridor_override=confirmation_shuffle,
    label="matched_margin_shuffle",
)
alias_aligned = _alias_audit(confirmation_payloads, label="aligned")
alias_shuffle = _alias_audit(
    confirmation_payloads,
    corridor_override=confirmation_shuffle,
    label="matched_margin_shuffle",
)
confirmation_all = pd.concat([confirmation_scan, shuffle_scan], ignore_index=True)
confirmation_all.to_csv(OUT_DIR / "fold3_confirmation_scan.csv", index=False)
pd.concat([confirmation_choices, shuffle_choices], ignore_index=True).to_csv(
    OUT_DIR / "fold3_confirmation_choices.csv", index=False
)
pd.concat([alias_aligned, alias_shuffle], ignore_index=True).to_csv(
    OUT_DIR / "fold3_alias_attribution.csv", index=False
)

base_row = confirmation_scan.loc[confirmation_scan["config"].eq("unconstrained")].iloc[0]
aligned_row = confirmation_scan.loc[confirmation_scan["config"].eq(selected_id)].iloc[0]
shuffle_row = shuffle_scan.iloc[0]
uncertainty = float(selected_config["uncertainty_ft"])
alias_local = alias_aligned.loc[alias_aligned["uncertainty_ft"].eq(uncertainty)]
alias_control = alias_shuffle.loc[alias_shuffle["uncertainty_ft"].eq(uncertainty)]
retained = alias_local.loc[alias_local["truth_legal"]]
retained_control = alias_control.loc[alias_control["truth_legal"]]
alias_columns_24 = ["alias_-24_rejected", "alias_+24_rejected"]

confirmation = {
    "base_rmse": float(base_row["rmse"]),
    "rmse": float(aligned_row["rmse"]),
    "gain": float(base_row["rmse"] - aligned_row["rmse"]),
    "base_far_rmse": float(base_row["far_gt2000_rmse"]),
    "far_rmse": float(aligned_row["far_gt2000_rmse"]),
    "far_gain": float(base_row["far_gt2000_rmse"] - aligned_row["far_gt2000_rmse"]),
    "shuffle_rmse": float(shuffle_row["rmse"]),
    "aligned_increment_vs_shuffle": float(shuffle_row["rmse"] - aligned_row["rmse"]),
    "bank_oracle_rmse": float(aligned_row["bank_oracle_rmse"]),
    "truth_retention": float(alias_local["truth_legal"].mean()),
    "alias24_any_rejected_given_truth_retained": float(
        retained[alias_columns_24].any(axis=1).mean()
    ) if len(retained) else 0.0,
    "shuffle_alias24_any_rejected_given_truth_retained": float(
        retained_control[alias_columns_24].any(axis=1).mean()
    ) if len(retained_control) else 0.0,
}
checks = {
    "confirmation_gain_ge_0p30": confirmation["gain"] >= 0.30,
    "confirmation_far_gain_ge_0p40": confirmation["far_gain"] >= 0.40,
    "aligned_increment_vs_shuffle_ge_0p15": confirmation["aligned_increment_vs_shuffle"] >= 0.15,
    "truth_retention_ge_0p90": confirmation["truth_retention"] >= 0.90,
    "alias24_any_rejected_ge_0p25": confirmation["alias24_any_rejected_given_truth_retained"] >= 0.25,
}
mechanism_pass = bool(all(checks.values()))

application = None
if mechanism_pass:
    fold2_payloads = _build_payloads(fold2_records, prior, "fold2_application")
    fold2_shuffle = _shuffle_corridors(fold2_payloads)
    fold2_scan, fold2_choices = _evaluate(
        fold2_payloads,
        [CONFIG_BY_ID["unconstrained"], selected_config],
        label="aligned",
    )
    fold2_control, fold2_control_choices = _evaluate(
        fold2_payloads,
        [selected_config],
        corridor_override=fold2_shuffle,
        label="matched_margin_shuffle",
    )
    pd.concat([fold2_scan, fold2_control], ignore_index=True).to_csv(
        OUT_DIR / "fold2_application_scan.csv", index=False
    )
    pd.concat([fold2_choices, fold2_control_choices], ignore_index=True).to_csv(
        OUT_DIR / "fold2_application_choices.csv", index=False
    )
    fold2_base = fold2_scan.loc[fold2_scan["config"].eq("unconstrained")].iloc[0]
    fold2_full = fold2_scan.loc[fold2_scan["config"].eq(selected_id)].iloc[0]
    fold2_shuffled = fold2_control.iloc[0]
    application = {
        "base_rmse": float(fold2_base["rmse"]),
        "rmse": float(fold2_full["rmse"]),
        "gain": float(fold2_base["rmse"] - fold2_full["rmse"]),
        "base_far_rmse": float(fold2_base["far_gt2000_rmse"]),
        "far_rmse": float(fold2_full["far_gt2000_rmse"]),
        "far_gain": float(fold2_base["far_gt2000_rmse"] - fold2_full["far_gt2000_rmse"]),
        "shuffle_rmse": float(fold2_shuffled["rmse"]),
        "aligned_increment_vs_shuffle": float(fold2_shuffled["rmse"] - fold2_full["rmse"]),
    }

strict_pass = bool(
    mechanism_pass
    and application is not None
    and application["gain"] >= 0.20
    and application["far_gain"] >= 0.25
    and application["aligned_increment_vs_shuffle"] >= 0.10
)
verdict = {
    "experiment": EXPERIMENT,
    "stage": "formation_zone_constraint_frozen_lattice_attribution",
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
        "PASS -> move the selected boundary penalty inside beam expansion, then add dip/fault state"
        if strict_pass
        else "FAIL -> formation intervals do not provide well-specific alias discrimination; close boundary branch"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV49 CONFIRMATION")
print(confirmation_all.to_string(index=False), flush=True)
print("\nV49 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
