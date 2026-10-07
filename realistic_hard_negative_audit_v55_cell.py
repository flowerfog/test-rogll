"""Audit realistic hard negatives in the frozen V3/V4 lattice bank.

This is the pre-training gate suggested by the competition discussion.  A
positive and a negative are both real paths emitted by the same beam arm.  A
negative is hard only when its legal generator rank is as good as (or better
than) the positive while its complete-well truth RMSE is materially worse.

Fold 3 is deterministically split into 24 discovery wells and the remaining
wells for confirmation.  Fold 2 is reported only after the mining recipe is
frozen.  Folds 4/5 are exported for the subsequent pairwise scorer.
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


ROOT = Path("/root/autodl-tmp/oof_exports")
V4_DIR = ROOT / "rawdata_lattice_coarse_ranker_v4"
PANEL_DIR = V4_DIR / "panels"
OUT_DIR = ROOT / "realistic_hard_negative_audit_v55"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = OUT_DIR.name
DISCOVERY_WELLS = 24
POSITIVE_REGRET_FT = 1.0
MAX_POSITIVES_PER_WELL = 4
MAX_NEGATIVES_PER_POSITIVE = 8
SEED = 20260727

REGRET_MARGINS_FT = (4.0, 6.0, 8.0, 12.0)
RANK_SLACKS = (0.0, 0.02, 0.05, 0.10)
TOP_RANK_CAPS = (0.05, 0.10, 0.20)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def _ensure_panels():
    panel_count = len(list(PANEL_DIR.glob("*.npz")))
    feature_path = V4_DIR / "feature_names.json"
    if panel_count >= 617 and feature_path.exists():
        print(f"V55 bootstrap: reusing {panel_count} frozen V4 panels", flush=True)
        return

    candidates = [
        Path.cwd() / "rawdata_lattice_coarse_ranker_v4_cell.py",
        Path("/root/autodl-tmp/rawdata_lattice_coarse_ranker_v4_cell.py"),
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "V4 panels are incomplete. Upload rawdata_segment_lattice_nested_v1_cell.py, "
            "rawdata_lattice_beam_survival_v2_cell.py, "
            "rawdata_lattice_mode_union_v3_cell.py and "
            "rawdata_lattice_coarse_ranker_v4_cell.py to /root/autodl-tmp."
        )
    print("V55 bootstrap: rebuilding frozen V4 panels", flush=True)
    namespace = {"__name__": "raw_lattice_v4_rebuild"}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    panel_count = len(list(PANEL_DIR.glob("*.npz")))
    if panel_count < 617 or not feature_path.exists():
        raise RuntimeError(f"V4 rebuild incomplete: panels={panel_count}/617")


def _fold_table():
    candidates = [
        ROOT / "rawdata_lattice_mode_union_v3" / "folds.csv",
        ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError("Cannot locate fold metadata")
    table = pd.read_csv(source)
    well_column = next(
        column for column in ("well", "Well_ID", "well_id") if column in table
    )
    fold_column = next(column for column in ("fold", "Fold") if column in table)
    table = table[[well_column, fold_column]].rename(
        columns={well_column: "well", fold_column: "fold"}
    )
    table["well"] = table["well"].astype(str)
    eligible = {path.stem for path in PANEL_DIR.glob("*.npz")}
    return (
        table[table["well"].isin(eligible)]
        .drop_duplicates("well")
        .sort_values("well")
        .reset_index(drop=True)
    )


def _load_panel(well):
    with np.load(PANEL_DIR / f"{well}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _feature_normalizer(features):
    values = np.asarray(features[:, EVIDENCE_COLUMNS], np.float64)
    median = np.median(values, axis=0)
    q25 = np.quantile(values, 0.25, axis=0)
    q75 = np.quantile(values, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-3)
    return np.clip((values - median) / scale, -12.0, 12.0)


def _shared_rank_gap(panel, positive_index):
    source_mask = panel["source_mask"].astype(np.uint16)
    arm_rank = panel["arm_rank"].astype(np.float64)
    positive_mask = int(source_mask[positive_index])
    shared = (source_mask & positive_mask) != 0
    gap = np.full(len(source_mask), np.inf, np.float64)
    candidate_shared_rank = np.full(len(source_mask), np.inf, np.float64)
    positive_shared_rank = np.full(len(source_mask), np.inf, np.float64)
    for arm in range(arm_rank.shape[1]):
        bit = 1 << arm
        member = shared & ((source_mask & bit) != 0) & ((positive_mask & bit) != 0)
        if not member.any():
            continue
        candidate_shared_rank[member] = np.minimum(
            candidate_shared_rank[member], arm_rank[member, arm]
        )
        positive_shared_rank[member] = np.minimum(
            positive_shared_rank[member], arm_rank[positive_index, arm]
        )
        gap[member] = np.minimum(
            gap[member], arm_rank[member, arm] - arm_rank[positive_index, arm]
        )
    return shared, gap, candidate_shared_rank, positive_shared_rank


def _mine_panel(well, fold, panel, config, include_pairs=False):
    rmse = np.sqrt(np.maximum(panel["mse"].astype(np.float64), 0.0))
    rows = int(panel["rows"][0])
    oracle = int(np.argmin(rmse))
    oracle_rmse = float(rmse[oracle])
    legal_rank = np.min(panel["arm_rank"].astype(np.float64), axis=1)
    legal_choice = int(np.argmin(legal_rank))
    positive_pool = np.flatnonzero(rmse <= oracle_rmse + POSITIVE_REGRET_FT)
    positive_order = positive_pool[np.lexsort((rmse[positive_pool], legal_rank[positive_pool]))]
    positives = list(positive_order[:MAX_POSITIVES_PER_WELL])
    if oracle not in positives:
        positives = [oracle] + positives[: MAX_POSITIVES_PER_WELL - 1]

    normalized = _feature_normalizer(panel["features"])
    pair_rows = []
    unique_negatives = set()
    for positive in positives:
        shared, rank_gap, candidate_rank, positive_rank = _shared_rank_gap(panel, positive)
        rank_ceiling = np.maximum(
            float(config["top_rank_cap"]), positive_rank + float(config["rank_slack"])
        )
        eligible = (
            shared
            & (rank_gap <= float(config["rank_slack"]) + 1e-12)
            & (candidate_rank <= rank_ceiling + 1e-12)
            & (rmse >= rmse[positive] + float(config["regret_margin_ft"]))
        )
        eligible[positive] = False
        negative_indices = np.flatnonzero(eligible)
        if not len(negative_indices):
            continue
        distance = np.sqrt(
            np.mean(
                (normalized[negative_indices] - normalized[positive][None, :]) ** 2,
                axis=1,
            )
        )
        order = np.lexsort(
            (
                -rmse[negative_indices],
                np.abs(rank_gap[negative_indices]),
                distance,
            )
        )[:MAX_NEGATIVES_PER_POSITIVE]
        for local_index in order:
            negative = int(negative_indices[local_index])
            unique_negatives.add(negative)
            pair_rows.append(
                {
                    "well": well,
                    "fold": int(fold),
                    "positive_index": int(positive),
                    "negative_index": negative,
                    "positive_rmse": float(rmse[positive]),
                    "negative_rmse": float(rmse[negative]),
                    "regret_ft": float(rmse[negative] - rmse[positive]),
                    "positive_legal_rank": float(positive_rank[negative]),
                    "negative_legal_rank": float(candidate_rank[negative]),
                    "rank_gap": float(rank_gap[negative]),
                    "negative_outranks_positive": bool(rank_gap[negative] < 0.0),
                    "feature_rms_distance": float(distance[local_index]),
                    "shared_source_mask": int(
                        int(panel["source_mask"][positive])
                        & int(panel["source_mask"][negative])
                    ),
                }
            )

    summary = {
        "well": well,
        "fold": int(fold),
        "rows": rows,
        "candidates": int(len(rmse)),
        "oracle_sse": float(panel["sse"][oracle]),
        "oracle_rmse": oracle_rmse,
        "legal_sse": float(panel["sse"][legal_choice]),
        "legal_rmse": float(rmse[legal_choice]),
        "positive_candidates": int(len(positives)),
        "hard_pairs": int(len(pair_rows)),
        "unique_hard_negatives": int(len(unique_negatives)),
    }
    if pair_rows:
        pairs = pd.DataFrame(pair_rows)
        summary.update(
            {
                "median_regret_ft": float(pairs["regret_ft"].median()),
                "median_rank_gap": float(pairs["rank_gap"].median()),
                "negative_outrank_rate": float(
                    pairs["negative_outranks_positive"].mean()
                ),
                "median_feature_rms_distance": float(
                    pairs["feature_rms_distance"].median()
                ),
            }
        )
    else:
        summary.update(
            {
                "median_regret_ft": np.nan,
                "median_rank_gap": np.nan,
                "negative_outrank_rate": np.nan,
                "median_feature_rms_distance": np.nan,
            }
        )
    return summary, pair_rows if include_pairs else []


def _aggregate(summaries, pairs, label, config):
    wells = pd.DataFrame(summaries)
    pair_frame = pd.DataFrame(pairs)
    total_rows = int(wells["rows"].sum())
    paired = wells["unique_hard_negatives"] >= 4
    payload = {
        "stage": label,
        **config,
        "wells": int(len(wells)),
        "rows": total_rows,
        "bank_oracle_rmse": math.sqrt(float(wells["oracle_sse"].sum()) / total_rows),
        "legal_map_rmse": math.sqrt(float(wells["legal_sse"].sum()) / total_rows),
        "paired_wells_fraction_ge4": float(paired.mean()),
        "paired_wells_fraction_ge1": float((wells["unique_hard_negatives"] >= 1).mean()),
        "median_unique_hard_negatives": float(wells["unique_hard_negatives"].median()),
        "p25_unique_hard_negatives": float(wells["unique_hard_negatives"].quantile(0.25)),
        "mean_candidates": float(wells["candidates"].mean()),
    }
    if len(pair_frame):
        payload.update(
            {
                "pairs": int(len(pair_frame)),
                "median_regret_ft": float(pair_frame["regret_ft"].median()),
                "p25_regret_ft": float(pair_frame["regret_ft"].quantile(0.25)),
                "median_rank_gap": float(pair_frame["rank_gap"].median()),
                "negative_outranks_positive_rate": float(
                    pair_frame["negative_outranks_positive"].mean()
                ),
                "median_feature_rms_distance": float(
                    pair_frame["feature_rms_distance"].median()
                ),
                "all_pairs_share_generator": bool(
                    (pair_frame["shared_source_mask"] > 0).all()
                ),
            }
        )
    else:
        payload.update(
            {
                "pairs": 0,
                "median_regret_ft": np.nan,
                "p25_regret_ft": np.nan,
                "median_rank_gap": np.nan,
                "negative_outranks_positive_rate": np.nan,
                "median_feature_rms_distance": np.nan,
                "all_pairs_share_generator": False,
            }
        )
    return payload, wells, pair_frame


def _evaluate_wells(wells, fold_by_well, config, label, include_pairs=True):
    summaries = []
    pairs = []
    for number, well in enumerate(wells, start=1):
        summary, current_pairs = _mine_panel(
            well, fold_by_well[well], _load_panel(well), config, include_pairs
        )
        summaries.append(summary)
        pairs.extend(current_pairs)
        if number % 50 == 0 or number == len(wells):
            print(f"V55 {label} {number}/{len(wells)}", flush=True)
    return _aggregate(summaries, pairs, label, config)


started = time.time()
_ensure_panels()
FEATURE_NAMES = json.loads((V4_DIR / "feature_names.json").read_text(encoding="utf-8"))
EVIDENCE_COLUMNS = np.asarray(
    [index for index, name in enumerate(FEATURE_NAMES) if not name.startswith("gen_")],
    dtype=np.int64,
)
if not len(EVIDENCE_COLUMNS):
    raise RuntimeError("No non-generator evidence columns found")

folds = _fold_table()
if len(folds) < 617:
    raise RuntimeError(f"Only {len(folds)} fold-assigned V4 panels")
fold_by_well = dict(zip(folds["well"], folds["fold"]))
fold3_wells = folds.loc[folds["fold"].eq(3), "well"].tolist()
fold3_wells = sorted(
    fold3_wells,
    key=lambda well: hashlib.sha256(f"hardnegative-v55:{well}".encode("ascii")).digest(),
)
discovery_wells = fold3_wells[:DISCOVERY_WELLS]
confirmation_wells = fold3_wells[DISCOVERY_WELLS:]
fold2_wells = folds.loc[folds["fold"].eq(2), "well"].tolist()

discovery_panels = {well: _load_panel(well) for well in discovery_wells}
scan_rows = []
for regret_margin in REGRET_MARGINS_FT:
    for rank_slack in RANK_SLACKS:
        for top_rank_cap in TOP_RANK_CAPS:
            config = {
                "regret_margin_ft": regret_margin,
                "rank_slack": rank_slack,
                "top_rank_cap": top_rank_cap,
            }
            summaries = []
            pairs = []
            for well in discovery_wells:
                summary, current = _mine_panel(
                    well, 3, discovery_panels[well], config, include_pairs=True
                )
                summaries.append(summary)
                pairs.extend(current)
            aggregate, _, _ = _aggregate(summaries, pairs, "fold3_discovery", config)
            coverage = aggregate["paired_wells_fraction_ge4"]
            count_factor = min(aggregate["median_unique_hard_negatives"], 8.0) / 8.0
            aggregate["selection_score"] = (
                coverage * count_factor * aggregate["median_regret_ft"]
                - 10.0 * rank_slack
                - 2.0 * top_rank_cap
            )
            scan_rows.append(aggregate)

scan = pd.DataFrame(scan_rows)
viable = scan[
    (scan["paired_wells_fraction_ge4"] >= 0.75)
    & (scan["median_unique_hard_negatives"] >= 4)
]
if len(viable):
    selected_row = viable.sort_values(
        ["selection_score", "regret_margin_ft"], ascending=[False, False]
    ).iloc[0]
else:
    selected_row = scan.sort_values(
        ["paired_wells_fraction_ge4", "median_unique_hard_negatives", "median_regret_ft"],
        ascending=False,
    ).iloc[0]
selected = {
    "regret_margin_ft": float(selected_row["regret_margin_ft"]),
    "rank_slack": float(selected_row["rank_slack"]),
    "top_rank_cap": float(selected_row["top_rank_cap"]),
}
scan.to_csv(OUT_DIR / "fold3_discovery_scan.csv", index=False)
print("\nV55 DISCOVERY TOP")
print(
    scan.sort_values("selection_score", ascending=False)
    .head(15)[
        [
            "regret_margin_ft",
            "rank_slack",
            "top_rank_cap",
            "paired_wells_fraction_ge4",
            "median_unique_hard_negatives",
            "negative_outranks_positive_rate",
            "median_regret_ft",
            "selection_score",
        ]
    ]
    .to_string(index=False),
    flush=True,
)
print("V55 SELECTED", selected, flush=True)

discovery, discovery_bywell, discovery_pairs = _evaluate_wells(
    discovery_wells, fold_by_well, selected, "fold3_discovery"
)
confirmation, confirmation_bywell, confirmation_pairs = _evaluate_wells(
    confirmation_wells, fold_by_well, selected, "fold3_confirmation"
)
application, application_bywell, application_pairs = _evaluate_wells(
    fold2_wells, fold_by_well, selected, "fold2_application"
)

all_wells = folds["well"].tolist()
all_summary, all_bywell, all_pairs = _evaluate_wells(
    all_wells, fold_by_well, selected, "fold2345_export"
)
all_bywell.to_parquet(OUT_DIR / "hard_negative_bywell.parquet", index=False)
all_pairs.to_parquet(OUT_DIR / "hard_negative_pairs.parquet", index=False)
pd.concat(
    [
        discovery_bywell.assign(stage="fold3_discovery"),
        confirmation_bywell.assign(stage="fold3_confirmation"),
        application_bywell.assign(stage="fold2_application"),
    ],
    ignore_index=True,
).to_csv(OUT_DIR / "audit_bywell.csv", index=False)

checks = {
    "confirmation_bank_oracle_rmse_le_5p8": confirmation["bank_oracle_rmse"] <= 5.8,
    "confirmation_paired_wells_ge_0p75": confirmation["paired_wells_fraction_ge4"] >= 0.75,
    "confirmation_median_negatives_ge_4": confirmation["median_unique_hard_negatives"] >= 4,
    "confirmation_negative_outrank_rate_ge_0p25": confirmation[
        "negative_outranks_positive_rate"
    ]
    >= 0.25,
    "fold2_paired_wells_ge_0p65": application["paired_wells_fraction_ge4"] >= 0.65,
    "all_pairs_share_generator": bool(
        confirmation["all_pairs_share_generator"] and application["all_pairs_share_generator"]
    ),
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "real_bank_same_arm_score_matched_hard_negative_audit",
    "sample_unit": "positive/negative complete paths emitted by the same legal beam arm",
    "truth_usage": "candidate labels and audit only; no validation feature uses truth",
    "protocol": {
        "train_export_folds": [2, 3, 4, 5],
        "selection_fold": 3,
        "selection_wells": len(discovery_wells),
        "confirmation_wells": len(confirmation_wells),
        "application_fold": 2,
        "fold1_touched": False,
    },
    "positive_definition": {
        "real_bank_paths_within_ft_of_oracle": POSITIVE_REGRET_FT,
        "maximum_per_well": MAX_POSITIVES_PER_WELL,
    },
    "negative_definition": {
        "same_generator_required": True,
        "maximum_per_positive": MAX_NEGATIVES_PER_POSITIVE,
        "selection_prefers_nearest_existing_evidence_profile": True,
    },
    "selected_config": selected,
    "discovery": discovery,
    "confirmation": confirmation,
    "fold2_application": application,
    "full_export": all_summary,
    "features_available_for_next_scorer": len(FEATURE_NAMES),
    "non_generator_evidence_features": int(len(EVIDENCE_COLUMNS)),
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> train V56 symmetric pairwise scorer; require full evidence to beat generator-only artifact control"
        if all(checks.values())
        else "FAIL -> the frozen bank does not contain enough realistic matched negatives; expand candidate generation before training"
    ),
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\nV55 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
