"""V99: can upstream PF disagreement predict useful independent coarse dose?

This is a read-only investment gate using the already-computed V97 and V98
paired seed predictions. V97's 24 wells are discovery and V98's disjoint
48 wells are confirmation. No PF inference or model fitting is performed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "mainline_pf_disagreement_coarse_gate_v99"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V97_PATH = (
    OOF_ROOT / "mainline_pf_repeatability_v97"
    / "paired_seed_predictions.parquet"
)
V98_PATH = (
    OOF_ROOT / "mainline_pf_seed1_confirmation_v98"
    / "paired_predictions.parquet"
)
V70_PATH = (
    OOF_ROOT / "rawdata_lattice_coarse_all773_crossfit_v70"
    / "all773_proxy_crossfit_oof.parquet"
)
V76_WEIGHT_PATH = (
    OOF_ROOT / "rawdata_lattice_coarse_adaptive_dose_bundle_v76"
    / "fixed_ensemble_weights_oof.parquet"
)

TRUTH = "tvt_true"
PARENT = "incumbent_6p377_replay"
COARSE = "coarse_crossfit"
PRIMARY_FEATURE = "prediction_pair_rms"


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _paired_rows(path, split):
    frame = pd.read_parquet(path)
    required = {
        "well", "id", "md_since", "truth", "seed",
        "prediction", "pf_ancc", "pf_z",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing {sorted(missing)}")
    frame = frame.loc[frame["seed"].isin([0, 1]), list(required)].copy()
    frame["well"] = frame["well"].astype(str)
    frame["id"] = frame["id"].astype(str)
    if frame.duplicated(["id", "seed"]).any():
        raise RuntimeError(f"Duplicate id/seed rows in {path}")
    wide = frame.pivot(
        index=["well", "id", "md_since", "truth"],
        columns="seed",
        values=["prediction", "pf_ancc", "pf_z"],
    )
    needed = [(name, seed) for name in ("prediction", "pf_ancc", "pf_z") for seed in (0, 1)]
    wide = wide.dropna(subset=needed).reset_index()
    wide.columns = [
        column if isinstance(column, str) else (
            column[0] if column[1] == "" else f"{column[0]}_seed{column[1]}"
        )
        for column in wide.columns
    ]
    wide["split"] = split
    return wide


def _slope(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if len(x) < 3 or np.std(x) < 1e-8:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _disagreement_features(rows):
    local = rows.copy()
    for name in ("prediction", "pf_ancc", "pf_z"):
        local[f"{name}_difference"] = (
            local[f"{name}_seed1"].to_numpy(np.float64)
            - local[f"{name}_seed0"].to_numpy(np.float64)
        )
    output = []
    for (split, well), group in local.groupby(["split", "well"], sort=False):
        distance = group["md_since"].to_numpy(np.float64)
        pred_difference = group["prediction_difference"].to_numpy(np.float64)
        far = distance > 2000.0
        output.append(
            {
                "split": split,
                "well": well,
                "rows": len(group),
                "mean_md_since": float(np.mean(distance)),
                "far_fraction": float(np.mean(far)),
                "prediction_pair_rms": float(np.sqrt(np.mean(pred_difference ** 2))),
                "prediction_pair_mae": float(np.mean(np.abs(pred_difference))),
                "prediction_pair_p95": float(np.quantile(np.abs(pred_difference), 0.95)),
                "prediction_pair_far_rms": (
                    float(np.sqrt(np.mean(pred_difference[far] ** 2)))
                    if far.any() else np.nan
                ),
                "prediction_pair_growth": _slope(distance, np.abs(pred_difference)),
                "pf_ancc_pair_rms": float(
                    np.sqrt(np.mean(group["pf_ancc_difference"].to_numpy(np.float64) ** 2))
                ),
                "pf_z_pair_rms": float(
                    np.sqrt(np.mean(group["pf_z_difference"].to_numpy(np.float64) ** 2))
                ),
            }
        )
    return pd.DataFrame(output)


def _target_table(wells):
    requested_wells = {str(well).strip().lower() for well in wells}
    frame = pd.read_parquet(
        V70_PATH,
        columns=["well", "fold", TRUTH, PARENT, COARSE],
    )
    frame["well"] = frame["well"].astype(str).str.strip().str.lower()
    frame = frame.loc[frame["well"].isin(requested_wells)].copy()
    found_wells = set(frame["well"].unique())
    missing_wells = sorted(requested_wells.difference(found_wells))
    print(
        "V99 TARGET COVERAGE",
        {
            "requested_wells": len(requested_wells),
            "found_wells": len(found_wells),
            "missing_wells": missing_wells[:10],
        },
        flush=True,
    )
    if missing_wells:
        pd.DataFrame({"well": missing_wells}).to_csv(
            OUT_DIR / "wells_missing_v70.csv", index=False
        )
    weights = pd.read_parquet(V76_WEIGHT_PATH)
    weights["well"] = weights["well"].astype(str).str.strip().str.lower()
    weight_map = dict(zip(weights["well"], weights["ensemble_weight"]))
    frame["v76_weight"] = frame["well"].map(weight_map)
    if frame["v76_weight"].isna().any():
        raise RuntimeError("Missing V76 weights in V99")
    output = []
    for well, group in frame.groupby("well", sort=False):
        truth = group[TRUTH].to_numpy(np.float64)
        parent = group[PARENT].to_numpy(np.float64)
        coarse = group[COARSE].to_numpy(np.float64)
        move = coarse - parent
        denominator = float(np.dot(move, move))
        if denominator <= 1e-8:
            oracle_weight = 0.0
            identifiable = False
        else:
            oracle_weight = float(
                np.clip(np.dot(truth - parent, move) / denominator, 0.0, 1.0)
            )
            identifiable = True
        v76_weight = float(group["v76_weight"].iloc[0])
        v76 = parent + v76_weight * move
        oracle = parent + oracle_weight * move
        parent_rmse = float(np.sqrt(np.mean((parent - truth) ** 2)))
        coarse_rmse = float(np.sqrt(np.mean((coarse - truth) ** 2)))
        v76_rmse = float(np.sqrt(np.mean((v76 - truth) ** 2)))
        oracle_rmse = float(np.sqrt(np.mean((oracle - truth) ** 2)))
        output.append(
            {
                "well": well,
                "fold": int(group["fold"].iloc[0]),
                "identifiable": identifiable,
                "v76_weight": v76_weight,
                "oracle_weight": oracle_weight,
                "parent_rmse": parent_rmse,
                "coarse_rmse": coarse_rmse,
                "v76_rmse": v76_rmse,
                "oracle_rmse": oracle_rmse,
                "coarse_gain_vs_parent": parent_rmse - coarse_rmse,
                "oracle_gain_vs_v76": v76_rmse - oracle_rmse,
                "move_rms": float(np.sqrt(np.mean(move ** 2))),
            }
        )
    return pd.DataFrame(output)


def _rho(frame, left, right):
    local = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(local) < 12 or local[left].nunique() < 3 or local[right].nunique() < 3:
        return np.nan
    return float(local[left].corr(local[right], method="spearman"))


def _associations(frame, split):
    local = frame.loc[frame["split"].eq(split) & frame["identifiable"]].copy()
    features = (
        "prediction_pair_rms",
        "prediction_pair_mae",
        "prediction_pair_p95",
        "prediction_pair_far_rms",
        "prediction_pair_growth",
        "pf_ancc_pair_rms",
        "pf_z_pair_rms",
    )
    targets = (
        "oracle_weight",
        "coarse_gain_vs_parent",
        "oracle_gain_vs_v76",
        "parent_rmse",
    )
    rows = []
    for feature in features:
        for target in targets:
            rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "target": target,
                    "wells": len(local),
                    "spearman": _rho(local, feature, target),
                }
            )
    return pd.DataFrame(rows)


for required in (V97_PATH, V98_PATH, V70_PATH, V76_WEIGHT_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

row_frame = pd.concat(
    [
        _paired_rows(V97_PATH, "discovery"),
        _paired_rows(V98_PATH, "confirmation"),
    ],
    ignore_index=True,
)
row_frame["well"] = row_frame["well"].astype(str).str.strip().str.lower()
features = _disagreement_features(row_frame)
targets = _target_table(set(features["well"]))
panel = features.merge(targets, on="well", how="inner", validate="one_to_one")
matched_counts = panel.groupby("split")["well"].nunique().to_dict()
print(
    "V99 MATCHED COVERAGE",
    {
        "feature_wells": int(features["well"].nunique()),
        "matched_wells": int(panel["well"].nunique()),
        "by_split": matched_counts,
    },
    flush=True,
)
if matched_counts.get("discovery", 0) < 16 or matched_counts.get("confirmation", 0) < 30:
    raise RuntimeError(f"Insufficient matched V99 coverage: {matched_counts}")

association = pd.concat(
    [_associations(panel, "discovery"), _associations(panel, "confirmation")],
    ignore_index=True,
)
lookup = {
    (row.split, row.feature, row.target): float(row.spearman)
    for row in association.itertuples()
}
discovery_weight_rho = lookup[("discovery", PRIMARY_FEATURE, "oracle_weight")]
confirmation_weight_rho = lookup[("confirmation", PRIMARY_FEATURE, "oracle_weight")]
confirmation_gain_rho = lookup[
    ("confirmation", PRIMARY_FEATURE, "coarse_gain_vs_parent")
]

# Matched negative control: rotate disagreement among wells within each split.
rolled_rows = []
for split, local in panel.groupby("split", sort=False):
    local = local.sort_values("well").copy()
    local["rolled_disagreement"] = np.roll(
        local[PRIMARY_FEATURE].to_numpy(np.float64), 1
    )
    rolled_rows.append(local)
rolled = pd.concat(rolled_rows, ignore_index=True)
rolled_confirmation_rho = _rho(
    rolled.loc[rolled["split"].eq("confirmation")],
    "rolled_disagreement",
    "oracle_weight",
)

checks = {
    "discovery_wells_ge_16": matched_counts.get("discovery", 0) >= 16,
    "confirmation_wells_ge_30": matched_counts.get("confirmation", 0) >= 30,
    "discovery_weight_rho_ge_0p20": discovery_weight_rho >= 0.20,
    "confirmation_weight_rho_ge_0p15": confirmation_weight_rho >= 0.15,
    "confirmation_coarse_gain_rho_ge_0p10": confirmation_gain_rho >= 0.10,
    "direction_consistent": discovery_weight_rho * confirmation_weight_rho > 0.0,
    "aligned_exceeds_rolled_by_0p10": (
        confirmation_weight_rho - rolled_confirmation_rho >= 0.10
    ),
}
strict_pass = bool(all(checks.values()))

verdict = {
    "experiment": "mainline_pf_disagreement_coarse_gate_v99",
    "stage": "existing_paired_seed_disagreement_observability_gate",
    "selection_or_tuning_count": 0,
    "primary_feature": PRIMARY_FEATURE,
    "hypothesis": "larger upstream PF disagreement implies larger useful independent coarse dose",
    "coverage": {
        "rows": len(row_frame),
        "feature_wells": features["well"].nunique(),
        "matched_wells": panel["well"].nunique(),
        "missing_v70_wells": features["well"].nunique() - panel["well"].nunique(),
        "discovery_wells": int(matched_counts.get("discovery", 0)),
        "confirmation_wells": int(matched_counts.get("confirmation", 0)),
    },
    "primary_association": {
        "discovery_oracle_weight_spearman": discovery_weight_rho,
        "confirmation_oracle_weight_spearman": confirmation_weight_rho,
        "confirmation_coarse_gain_spearman": confirmation_gain_rho,
        "rolled_confirmation_oracle_weight_spearman": rolled_confirmation_rho,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> generate fold-clean second-seed disagreement and add it to V76 dose features"
        if strict_pass
        else "FAIL -> PF randomness is neither a useful estimator nor a useful risk signal; close branch"
    ),
}

panel.to_csv(OUT_DIR / "well_panel.csv", index=False)
association.to_csv(OUT_DIR / "associations.csv", index=False)
(OUT_DIR / "result.json").write_text(
    json.dumps(_json_value(verdict), indent=2), encoding="utf-8"
)

print("V99 PRIMARY PANEL")
print(
    panel[
        [
            "split", "well", PRIMARY_FEATURE, "oracle_weight",
            "coarse_gain_vs_parent", "oracle_gain_vs_v76",
        ]
    ].to_string(index=False)
)
print("\nV99 ASSOCIATIONS")
print(association.to_string(index=False))
print("\nV99 FINAL VERDICT")
print(json.dumps(_json_value(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
