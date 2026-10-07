"""Complete-seed-trajectory mode-bank oracle audit for the 128-seed lik-PF.

This is a zero-training representation audit. It reuses the seed trajectories
cached by ``likpf_blockposterior_globaldp_v0_cell.py`` and deliberately keeps
each seed's complete path intact. Fold 3 selects the smallest useful mode-bank
budget; Fold 2 is confirmation; Fold 1 remains untouched.

The matched control independently permutes seed identity inside each 128-ft
block. It preserves every block's marginal set of PF paths while destroying
the cross-block identity that this experiment claims is useful.
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


EXPERIMENT = "likpf_fulltrajectory_mode_oracle_audit_v1"
SOURCE_EXPERIMENT = "likpf_blockposterior_globaldp_v0"

ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").exists() else Path.cwd()
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
if not DATA_ROOT.exists():
    DATA_ROOT = Path.cwd()
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").exists() or ROOT.name == "autodl-tmp" else Path.cwd() / "output"
SOURCE_DIR = OOF_ROOT / SOURCE_EXPERIMENT
CACHE_DIR = SOURCE_DIR / "pf_cache"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260721  # Must match the source experiment's fold construction.
SELECTION_FOLD = 3
APPLICATION_FOLD = 2
UNTOUCHED_FOLD = 1
BLOCK_FT = 128.0
WEIGHT_TEMPERATURE = 5.0
SIGNATURE_POINTS = 129
MODE_BUDGETS = (2, 4, 6, 8, 12, 16, 24, 32)
FAMILIES = ("mean", "medoid", "union")
SELECTION_TOLERANCE_FT = 0.10
WELL_LIMIT = int(os.environ.get("ROGII_MODE_ORACLE_WELL_LIMIT", "0"))


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _stable_folds(wells):
    scored = []
    for well in wells:
        digest = hashlib.sha256(f"{SEED}:{well}".encode("ascii")).digest()
        scored.append((int.from_bytes(digest[:8], "little"), well))
    ordered = [well for _, well in sorted(scored)]
    fold = np.empty(len(ordered), dtype=np.int64)
    for number, indices in enumerate(np.array_split(np.arange(len(ordered)), 5), start=1):
        fold[indices] = number
    return pd.DataFrame({"well": ordered, "fold": fold})


def _softmax(values, temperature=WEIGHT_TEMPERATURE):
    values = np.asarray(values, np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), 1.0 / len(values), np.float64)
    floor = float(np.min(values[finite])) - 1000.0
    values = np.where(finite, values, floor)
    scaled = (values - np.max(values)) / float(temperature)
    weight = np.exp(np.clip(scaled, -700.0, 0.0))
    return weight / max(float(weight.sum()), 1e-300)


def _cache_path(well):
    return CACHE_DIR / f"{well}.npz"


def _load_cache(well):
    path = _cache_path(well)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run {SOURCE_EXPERIMENT}_cell.py first."
        )
    with np.load(path, allow_pickle=False) as data:
        required = {
            "md",
            "z",
            "truth",
            "center",
            "seed_delta",
            "total_loglik",
            "anchor_md",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(f"{well}: cache missing {missing}")
        return {key: data[key] for key in data.files}


def _pairwise_path_rmse(signature):
    signature = np.asarray(signature, np.float64)
    gram = signature @ signature.T / max(signature.shape[1], 1)
    norm = np.diag(gram)
    squared = norm[:, None] + norm[None, :] - 2.0 * gram
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared)


def _medoid_order(distance, weight, maximum):
    maximum = min(int(maximum), len(weight))
    first = int(np.argmax(weight))
    selected = [first]
    minimum_distance = distance[:, first].copy()
    maximum_weight = max(float(np.max(weight)), 1e-300)
    while len(selected) < maximum:
        novelty_weight = 0.35 + 0.65 * np.sqrt(weight / maximum_weight)
        score = minimum_distance * novelty_weight
        score[np.asarray(selected, dtype=int)] = -np.inf
        candidate = int(np.argmax(score))
        if not np.isfinite(score[candidate]) or minimum_distance[candidate] < 1e-4:
            break
        selected.append(candidate)
        minimum_distance = np.minimum(minimum_distance, distance[:, candidate])
    return np.asarray(selected, dtype=np.int64)


def _prepare_paths(cache, well, control):
    center = cache["center"].astype(np.float64)
    delta = cache["seed_delta"].astype(np.float64)
    total_loglik = cache["total_loglik"].astype(np.float64)
    if delta.ndim != 2 or delta.shape[1] != len(center):
        raise RuntimeError(f"{well}: invalid seed trajectory shape {delta.shape}")

    if control == "block_shuffled":
        md = cache["md"].astype(np.float64)
        anchor_md = float(cache["anchor_md"][0])
        block_id = np.floor((md - anchor_md) / BLOCK_FT).astype(np.int64)
        shuffled = np.empty_like(delta)
        rng_seed = int.from_bytes(
            hashlib.sha256(f"mode-shuffle:{well}".encode("ascii")).digest()[:8],
            "little",
        )
        rng = np.random.default_rng(rng_seed)
        for block in np.unique(block_id):
            rows = np.flatnonzero(block_id == block)
            permutation = rng.permutation(delta.shape[0])
            shuffled[:, rows] = delta[permutation][:, rows]
        delta = shuffled
    elif control != "coherent":
        raise ValueError(control)

    prediction = center[None, :] + delta
    sample_index = np.unique(
        np.rint(np.linspace(0, len(center) - 1, min(SIGNATURE_POINTS, len(center)))).astype(int)
    )
    signature = delta[:, sample_index]
    distance = _pairwise_path_rmse(signature)
    weight = _softmax(total_loglik)
    order = _medoid_order(distance, weight, max(MODE_BUDGETS))
    return {
        "prediction": prediction,
        "distance": distance,
        "weight": weight,
        "order": order,
        "center": center,
    }


def _mode_library(prepared, requested_modes):
    prediction = prepared["prediction"]
    distance = prepared["distance"]
    weight = prepared["weight"]
    medoids = prepared["order"][: min(int(requested_modes), len(prepared["order"]))].copy()
    if not len(medoids):
        raise RuntimeError("no trajectory medoids")

    # Two deterministic PAM-style update passes are enough at 128 seeds.
    for _ in range(2):
        assignment = np.argmin(distance[:, medoids], axis=1)
        assignment[medoids] = np.arange(len(medoids))
        updated = medoids.copy()
        for cluster in range(len(medoids)):
            members = np.flatnonzero(assignment == cluster)
            if not len(members):
                continue
            local_weight = weight[members]
            local_weight = local_weight / max(float(local_weight.sum()), 1e-300)
            objective = distance[np.ix_(members, members)] @ local_weight
            updated[cluster] = members[int(np.argmin(objective))]
        medoids = updated

    assignment = np.argmin(distance[:, medoids], axis=1)
    assignment[medoids] = np.arange(len(medoids))
    support = np.bincount(assignment, minlength=len(medoids)).astype(np.int64)
    mass = np.bincount(assignment, weights=weight, minlength=len(medoids)).astype(np.float64)
    mass /= max(float(mass.sum()), 1e-300)

    means = []
    for cluster in range(len(medoids)):
        members = np.flatnonzero(assignment == cluster)
        local_weight = weight[members]
        local_weight /= max(float(local_weight.sum()), 1e-300)
        means.append(np.sum(local_weight[:, None] * prediction[members], axis=0))
    means = np.asarray(means, np.float64)
    medoid_paths = prediction[medoids]
    effective_modes = float(
        np.exp(-np.sum(mass * np.log(np.maximum(mass, 1e-300))))
    )
    return {
        "mean": means,
        "medoid": medoid_paths,
        "union": np.concatenate([means, medoid_paths], axis=0),
        "top_mass_mean": means[int(np.argmax(mass))],
        "modes": int(len(medoids)),
        "support": support,
        "mass": mass,
        "effective_modes": effective_modes,
    }


def _error_payload(well, variant, prediction, truth, md, anchor_md, **extra):
    prediction = np.asarray(prediction, np.float64)
    truth = np.asarray(truth, np.float64)
    error = prediction - truth
    distance = md - float(anchor_md)
    payload = {
        "well": str(well),
        "variant": str(variant),
        "rows": int(len(error)),
        "sse": float(np.sum(error**2)),
        "sae": float(np.sum(np.abs(error))),
    }
    strata = {
        "near_0_500": (distance >= 0.0) & (distance <= 500.0),
        "mid_500_1000": (distance > 500.0) & (distance <= 1000.0),
        "mid_1000_2000": (distance > 1000.0) & (distance <= 2000.0),
        "far_gt2000": distance > 2000.0,
    }
    for label, mask in strata.items():
        payload[f"{label}_rows"] = int(mask.sum())
        payload[f"{label}_sse"] = float(np.sum(error[mask] ** 2))
    payload.update(extra)
    return payload


def _oracle_candidate(candidates, truth):
    candidates = np.asarray(candidates, np.float64)
    candidate_mse = np.mean((candidates - truth[None, :]) ** 2, axis=1)
    selected = int(np.argmin(candidate_mse))
    return candidates[selected], selected, float(candidate_mse[selected])


def _evaluate_wells(wells, budgets, controls, label):
    rows = []
    reproduction = []
    for number, well in enumerate(wells, start=1):
        cache = _load_cache(well)
        truth = cache["truth"].astype(np.float64)
        md = cache["md"].astype(np.float64)
        anchor_md = float(cache["anchor_md"][0])

        coherent = _prepare_paths(cache, well, "coherent")
        weighted_reproduction = np.sum(
            coherent["weight"][:, None] * coherent["prediction"], axis=0
        )
        reproduction.append(
            math.sqrt(float(np.mean((weighted_reproduction - coherent["center"]) ** 2)))
        )

        seed_oracle, seed_index, _ = _oracle_candidate(coherent["prediction"], truth)
        rows.append(
            _error_payload(
                well,
                "pf_weighted",
                coherent["center"],
                truth,
                md,
                anchor_md,
                control="coherent",
                requested_modes=0,
                effective_modes=0.0,
                candidate_count=1,
                selected_candidate=-1,
                selected_support=128,
                top_mode_mass=1.0,
            )
        )
        rows.append(
            _error_payload(
                well,
                "all_seed_oracle",
                seed_oracle,
                truth,
                md,
                anchor_md,
                control="coherent",
                requested_modes=128,
                effective_modes=128.0,
                candidate_count=128,
                selected_candidate=seed_index,
                selected_support=1,
                top_mode_mass=float(np.max(coherent["weight"])),
            )
        )

        for control in controls:
            prepared = coherent if control == "coherent" else _prepare_paths(cache, well, control)
            for budget in budgets:
                library = _mode_library(prepared, budget)
                common = {
                    "control": control,
                    "requested_modes": int(budget),
                    "effective_modes": float(library["effective_modes"]),
                    "top_mode_mass": float(np.max(library["mass"])),
                    "minimum_mode_support": int(np.min(library["support"])),
                }
                rows.append(
                    _error_payload(
                        well,
                        "top_mass_mean",
                        library["top_mass_mean"],
                        truth,
                        md,
                        anchor_md,
                        candidate_count=1,
                        selected_candidate=int(np.argmax(library["mass"])),
                        selected_support=int(library["support"][np.argmax(library["mass"])]),
                        **common,
                    )
                )
                for family in FAMILIES:
                    candidates = library[family]
                    prediction, selected, _ = _oracle_candidate(candidates, truth)
                    mode_index = selected % library["modes"]
                    rows.append(
                        _error_payload(
                            well,
                            f"mode_{family}_oracle",
                            prediction,
                            truth,
                            md,
                            anchor_md,
                            candidate_count=int(len(candidates)),
                            selected_candidate=int(selected),
                            selected_support=int(library["support"][mode_index]),
                            **common,
                        )
                    )

        if number % 25 == 0 or number == len(wells):
            print(f"MODE ORACLE {label} {number}/{len(wells)}", flush=True)
    return pd.DataFrame(rows), reproduction


def _summary(frame):
    output = []
    group_columns = ["control", "requested_modes", "variant"]
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        control, requested_modes, variant = keys
        count = int(group["rows"].sum())
        sse = float(group["sse"].sum())
        well_rmse = np.sqrt(group["sse"].to_numpy(np.float64) / group["rows"].to_numpy(np.float64))
        row = {
            "control": control,
            "requested_modes": int(requested_modes),
            "variant": variant,
            "candidate_count": float(group["candidate_count"].mean()),
            "rows": count,
            "wells": int(group["well"].nunique()),
            "exact_sse": sse,
            "rmse": math.sqrt(sse / count),
            "mae": float(group["sae"].sum() / count),
            "median_well_rmse": float(np.median(well_rmse)),
            "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
            "max_well_rmse": float(np.max(well_rmse)),
            "mean_effective_modes": float(group["effective_modes"].mean()),
            "mean_top_mode_mass": float(group["top_mode_mass"].mean()),
            "minimum_selected_support": int(group["selected_support"].min()),
        }
        for label in ("near_0_500", "mid_500_1000", "mid_1000_2000", "far_gt2000"):
            rows = int(group[f"{label}_rows"].sum())
            row[f"{label}_rmse"] = (
                math.sqrt(float(group[f"{label}_sse"].sum()) / rows)
                if rows
                else np.nan
            )
        output.append(row)
    return pd.DataFrame(output)


def _one(summary, variant, control="coherent", requested_modes=None):
    mask = summary["variant"].eq(variant) & summary["control"].eq(control)
    if requested_modes is not None:
        mask &= summary["requested_modes"].eq(int(requested_modes))
    selected = summary.loc[mask]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one summary for {variant}/{control}/{requested_modes}, got {len(selected)}"
        )
    return selected.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

started = time.time()
if not CACHE_DIR.exists():
    raise FileNotFoundError(
        f"Missing {CACHE_DIR}. Run {SOURCE_EXPERIMENT}_cell.py first."
    )

all_wells = [
    path.name.split("__", 1)[0]
    for path in sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
]
folds = _stable_folds(all_wells)
fold_by_well = dict(zip(folds["well"], folds["fold"]))
selection_wells = [well for well in all_wells if fold_by_well[well] == SELECTION_FOLD]
application_wells = [well for well in all_wells if fold_by_well[well] == APPLICATION_FOLD]
if WELL_LIMIT:
    selection_wells = selection_wells[:WELL_LIMIT]
    application_wells = application_wells[:WELL_LIMIT]

missing_cache = [well for well in selection_wells + application_wells if not _cache_path(well).exists()]
if missing_cache:
    raise RuntimeError(
        f"Missing {len(missing_cache)} PF caches, first={missing_cache[:10]}. "
        f"Run {SOURCE_EXPERIMENT}_cell.py first."
    )

print(
    "FULL-TRAJECTORY MODE CONFIG",
    {
        "source": str(SOURCE_DIR),
        "selection_wells": len(selection_wells),
        "application_wells": len(application_wells),
        "mode_budgets": MODE_BUDGETS,
        "signature_points": SIGNATURE_POINTS,
        "well_limit": WELL_LIMIT,
    },
    flush=True,
)

# Fold 3 prices candidate coverage and freezes the smallest near-optimal bank.
selection_detail, selection_reproduction = _evaluate_wells(
    selection_wells,
    MODE_BUDGETS,
    ("coherent",),
    "fold3_selection",
)
selection_summary = _summary(selection_detail)
selection_curve = selection_summary[
    selection_summary["variant"].isin(
        ["mode_mean_oracle", "mode_medoid_oracle", "mode_union_oracle"]
    )
].copy()
best_selection_rmse = float(selection_curve["rmse"].min())
eligible = selection_curve[
    selection_curve["rmse"] <= best_selection_rmse + SELECTION_TOLERANCE_FT
].copy()
family_priority = {
    "mode_mean_oracle": 0,
    "mode_medoid_oracle": 1,
    "mode_union_oracle": 2,
}
eligible["family_priority"] = eligible["variant"].map(family_priority)
eligible = eligible.sort_values(
    ["candidate_count", "requested_modes", "rmse", "family_priority"]
)
selected_row = eligible.iloc[0]
selected_budget = int(selected_row["requested_modes"])
selected_variant = str(selected_row["variant"])

selection_detail.to_csv(OUT_DIR / "fold3_mode_detail.csv", index=False)
selection_summary.sort_values("rmse").to_csv(OUT_DIR / "fold3_mode_summary.csv", index=False)

# Fold 2 confirms the frozen bank and the trajectory-identity control.
application_detail, application_reproduction = _evaluate_wells(
    application_wells,
    (selected_budget,),
    ("coherent", "block_shuffled"),
    "fold2_confirmation",
)
application_summary = _summary(application_detail)
application_detail.to_csv(OUT_DIR / "fold2_mode_detail.csv", index=False)
application_summary.sort_values("rmse").to_csv(OUT_DIR / "fold2_mode_summary.csv", index=False)
folds.to_csv(OUT_DIR / "folds.csv", index=False)

selection_base = _one(selection_summary, "pf_weighted", requested_modes=0)
selection_mode = _one(
    selection_summary,
    selected_variant,
    requested_modes=selected_budget,
)
application_base = _one(application_summary, "pf_weighted", requested_modes=0)
application_seed_oracle = _one(application_summary, "all_seed_oracle", requested_modes=128)
application_mode = _one(
    application_summary,
    selected_variant,
    control="coherent",
    requested_modes=selected_budget,
)
application_shuffle = _one(
    application_summary,
    selected_variant,
    control="block_shuffled",
    requested_modes=selected_budget,
)
application_top_mass = _one(
    application_summary,
    "top_mass_mean",
    control="coherent",
    requested_modes=selected_budget,
)

selection_gain = float(selection_base["rmse"] - selection_mode["rmse"])
application_gain = float(application_base["rmse"] - application_mode["rmse"])
available_seed_gain = float(application_base["rmse"] - application_seed_oracle["rmse"])
retained_seed_gain = (
    application_gain / available_seed_gain
    if available_seed_gain > 1e-9
    else float("nan")
)
shuffle_degradation = float(application_shuffle["rmse"] - application_mode["rmse"])
maximum_reproduction_error = float(
    max(selection_reproduction + application_reproduction)
)

checks = {
    "cache_reproduction_rmse_le_0p05": maximum_reproduction_error <= 0.05,
    "selection_mode_gain_ge_1ft": selection_gain >= 1.0,
    "confirmation_mode_gain_ge_1ft": application_gain >= 1.0,
    "confirmation_retains_ge_70pct_seed_oracle_gain": retained_seed_gain >= 0.70,
    "coherent_beats_block_shuffle_ge_0p25": shuffle_degradation >= 0.25,
    "confirmation_mode_p95_le_15": application_mode["p95_well_rmse"] <= 15.0,
    "selected_candidate_count_le_24": application_mode["candidate_count"] <= 24.0,
    "selected_oracle_never_single_seed_supported": application_mode["minimum_selected_support"] >= 2,
}
strict_pass = bool(all(checks.values())) and WELL_LIMIT == 0

verdict = {
    "experiment": EXPERIMENT,
    "stage": "nested_complete_seed_trajectory_mode_bank_oracle",
    "sample_unit": "one complete physical well",
    "legal_inputs_only": True,
    "truth_usage": "oracle pricing only; truth is never used to construct candidates",
    "uses_neighbor_spatial_cp12_gold": False,
    "optimizer_steps": 0,
    "source": str(SOURCE_DIR),
    "protocol": {
        "selection_fold": SELECTION_FOLD,
        "application_fold": APPLICATION_FOLD,
        "untouched_fold": UNTOUCHED_FOLD,
        "fold1_touched": False,
    },
    "bank": {
        "seed_paths": 128,
        "signature_points": SIGNATURE_POINTS,
        "mode_budgets_scanned": list(MODE_BUDGETS),
        "selection_tolerance_ft": SELECTION_TOLERANCE_FT,
        "selected_budget": selected_budget,
        "selected_variant": selected_variant,
        "matched_control": "independent seed permutation inside every 128-ft block",
    },
    "guards": {
        "selection_wells": len(selection_wells),
        "application_wells": len(application_wells),
        "maximum_cache_reproduction_rmse": maximum_reproduction_error,
        "well_limit": WELL_LIMIT,
    },
    "fold3_selection": {
        "base": selection_base,
        "selected_mode_oracle": selection_mode,
        "gain": selection_gain,
        "best_scanned_rmse": best_selection_rmse,
    },
    "fold2_confirmation": {
        "pf_weighted": application_base,
        "all_seed_oracle": application_seed_oracle,
        "selected_mode_oracle": application_mode,
        "block_shuffled_mode_oracle": application_shuffle,
        "top_mass_mode": application_top_mass,
        "gain_vs_pf_weighted": application_gain,
        "available_seed_oracle_gain": available_seed_gain,
        "retained_seed_oracle_gain_fraction": retained_seed_gain,
        "block_shuffle_degradation": shuffle_degradation,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> freeze this complete-path bank and train a fold-clean complete-well listwise mode scorer"
        if strict_pass
        else "FAIL -> coherent 128-seed trajectories do not provide a stable selectable mode bank"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}

_write_json(OUT_DIR / "verdict.json", verdict)
print("\nFOLD3 MODE ORACLE CURVE")
print(
    selection_curve.sort_values(["candidate_count", "rmse"])[
        [
            "requested_modes",
            "variant",
            "candidate_count",
            "rmse",
            "median_well_rmse",
            "p95_well_rmse",
            "far_gt2000_rmse",
            "mean_effective_modes",
            "minimum_selected_support",
        ]
    ].to_string(index=False),
    flush=True,
)
print("\nFOLD2 MODE CONFIRMATION")
print(application_summary.sort_values("rmse").to_string(index=False), flush=True)
print("\nFULL-TRAJECTORY MODE FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
