"""Inventory the artifacts needed for an author-like SP45 -> Track8 endpoint.

Run this as one AutoDL cell before rebuilding SP45.  It does no training and
does not use validation truth for selection.  Its purpose is to distinguish:

* SP45: a multi-seed PF + multi-configuration beam + per-well selector;
* Track8: the Track6-style GBDT, with SP45 outputs/features replacing Track6's
  self-built PF block while retaining the same common correlation/spatial block.

The inventory deliberately treats an endpoint-only SP45 artifact as
insufficient for a feature-rich Track8 reproduction.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "sp45_track8_parity_inventory_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _first_existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {label}: {[str(path) for path in candidates]}")


FOLD_PATH = _first_existing(
    [
        OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
        OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "folds.csv",
    ],
    "fold metadata",
)
FEATURE_ROOT = OOF_ROOT / "foldclean_full_fleongg_lgb_v0"
FEATURE_DIR = FEATURE_ROOT / "features"
if not FEATURE_DIR.exists():
    raise FileNotFoundError(f"Missing fold-clean feature cache: {FEATURE_DIR}")

POOL_PATH = _first_existing(
    [
        OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
        OOF_ROOT / "candidate_pool_full_merged_oof.csv",
        ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
        ROOT / "candidate_pool_full_merged_oof.csv",
    ],
    "candidate pool",
)


def _normalise_folds(frame: pd.DataFrame) -> pd.DataFrame:
    well_column = next(
        (column for column in ("well", "well_id", "Well") if column in frame.columns),
        None,
    )
    fold_column = next(
        (column for column in ("fold", "outer_fold", "Fold") if column in frame.columns),
        None,
    )
    if well_column is None or fold_column is None:
        raise RuntimeError(f"Cannot identify fold columns: {frame.columns.tolist()}")
    output = frame[[well_column, fold_column]].copy()
    output.columns = ["well", "fold"]
    output["well"] = output["well"].astype(str)
    output["fold"] = pd.to_numeric(output["fold"], errors="raise").astype(int)
    output = output.drop_duplicates("well")
    if not output["fold"].isin([1, 2, 3, 4, 5]).all():
        raise RuntimeError("Fold metadata contains values outside 1..5")
    return output


folds = _normalise_folds(pd.read_csv(FOLD_PATH))


def _load_feature_schema() -> pd.DataFrame:
    listed = FEATURE_ROOT / "all_features.csv"
    if listed.exists():
        table = pd.read_csv(listed)
        if "feature" in table.columns:
            return pd.DataFrame(columns=table["feature"].astype(str).tolist())
        if len(table.columns):
            return pd.DataFrame(columns=table.iloc[:, 0].astype(str).tolist())

    for well in folds["well"]:
        for suffix in ("sample", "full"):
            path = FEATURE_DIR / f"{well}__{suffix}.pkl"
            if path.exists():
                return pd.read_pickle(path).iloc[:0].copy()
    raise RuntimeError("No feature schema or feature pickle was found")


schema = _load_feature_schema()
feature_columns = list(map(str, schema.columns))


def _matches_any(column: str, patterns: tuple[str, ...]) -> bool:
    lower = column.lower()
    return any(re.search(pattern, lower) for pattern in patterns)


pf_scale_columns = [
    column
    for column in feature_columns
    if re.search(r"(^|_)(lik)?pf_scale_(3|5|8|12)(\.0)?($|_)", column.lower())
]
pf_path_columns = [
    column
    for column in feature_columns
    if _matches_any(
        column,
        (
            r"^pf_ancc$",
            r"^pf_z$",
            r"^likpf_",
            r"^pf_scale_",
            r"^pf_mean$",
        ),
    )
]
beam_columns = [
    column
    for column in feature_columns
    if column.lower().startswith("beam_")
]
beam_member_columns = [
    column
    for column in beam_columns
    if not _matches_any(column, (r"mean", r"med", r"std", r"range", r"vs_"))
]
uncertainty_columns = [
    column
    for column in feature_columns
    if _matches_any(
        column,
        (
            r"std",
            r"entropy",
            r"margin",
            r"spread",
            r"range",
            r"quantile",
            r"^pf_mean$",
        ),
    )
    and _matches_any(column, (r"pf", r"beam", r"lik"))
]
selector_columns = [
    column for column in feature_columns if "selector" in column.lower()
]
seed_member_columns = [
    column
    for column in feature_columns
    if _matches_any(column, (r"pf_seed_", r"seed_[0-9]+", r"pf_run_", r"member_[0-9]+"))
]

correlation_columns = [
    column
    for column in feature_columns
    if _matches_any(
        column,
        (
            r"corr",
            r"xcorr",
            r"ncc",
            r"tdbc",
            r"cost",
            r"misfit",
            r"grdiff",
        ),
    )
]
formation_neighbor_columns = [
    column
    for column in feature_columns
    if _matches_any(
        column,
        (
            r"formation",
            r"^form",
            r"^frm",
            r"^tvtf",
            r"neighbor",
            r"nearest",
            r"spatial",
            r"^bw",
            r"dense",
        ),
    )
]


pool_header = pd.read_csv(POOL_PATH, nrows=0).columns.astype(str).tolist()
pool_sp45_columns = [column for column in pool_header if column.lower().startswith("sp45")]
if not pool_sp45_columns:
    raise RuntimeError(f"Candidate pool has no SP45 endpoint: {pool_header}")

pool_key_columns = [
    column for column in ("id", "well", "row_idx") if column in pool_header
]
if "id" not in pool_key_columns and not {"well", "row_idx"}.issubset(pool_key_columns):
    raise RuntimeError(f"Candidate pool has no usable row keys: {pool_header}")

pool = pd.read_csv(POOL_PATH, usecols=list(dict.fromkeys(pool_key_columns + pool_sp45_columns)))
if "well" not in pool:
    pool["well"] = pool["id"].astype(str).str[:8]
pool["well"] = pool["well"].astype(str)

endpoint_reports = []
for column in pool_sp45_columns:
    values = pd.to_numeric(pool[column], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(values)
    endpoint_reports.append(
        {
            "column": column,
            "rows": int(len(values)),
            "wells": int(pool.loc[finite, "well"].nunique()),
            "finite_fraction": float(finite.mean()),
            "std": float(np.nanstd(values)),
        }
    )
endpoint_report = pd.DataFrame(endpoint_reports)
endpoint_report.to_csv(OUT_DIR / "sp45_endpoint_inventory.csv", index=False)


sampled_wells = []
for fold, group in folds.groupby("fold", sort=True):
    available = [
        well
        for well in group["well"].astype(str)
        if (FEATURE_DIR / f"{well}__sample.pkl").exists()
    ]
    sampled_wells.extend(available[:5])

candidate_feature_columns = list(
    dict.fromkeys(
        pf_path_columns
        + beam_columns
        + uncertainty_columns
        + correlation_columns
        + formation_neighbor_columns
    )
)
feature_parts = []
for well in sampled_wells:
    path = FEATURE_DIR / f"{well}__sample.pkl"
    frame = pd.read_pickle(path)
    present = [column for column in candidate_feature_columns if column in frame.columns]
    if not present:
        continue
    if len(frame) > 1024:
        rows = np.linspace(0, len(frame) - 1, 1024).round().astype(int)
        frame = frame.iloc[rows]
    part = frame[present].copy()
    part["__well"] = well
    feature_parts.append(part)

feature_reports = []
if feature_parts:
    sampled = pd.concat(feature_parts, ignore_index=True, sort=False)
    for column in candidate_feature_columns:
        if column not in sampled:
            continue
        values = pd.to_numeric(sampled[column], errors="coerce").to_numpy(np.float64)
        finite = np.isfinite(values)
        feature_reports.append(
            {
                "feature": column,
                "sample_rows": int(len(values)),
                "finite_fraction": float(finite.mean()),
                "std": float(np.nanstd(values)),
                "nonconstant": bool(np.nanstd(values) > 1e-8),
            }
        )
feature_report = pd.DataFrame(feature_reports)
feature_report.to_csv(OUT_DIR / "feature_inventory.csv", index=False)

usable = set(
    feature_report.loc[
        feature_report.get("finite_fraction", pd.Series(dtype=float)).ge(0.95)
        & feature_report.get("nonconstant", pd.Series(dtype=bool)),
        "feature",
    ].astype(str)
) if len(feature_report) else set()

usable_pf_scales = [column for column in pf_scale_columns if column in usable]
usable_beam_members = [column for column in beam_member_columns if column in usable]
usable_uncertainty = [column for column in uncertainty_columns if column in usable]
usable_correlation = [column for column in correlation_columns if column in usable]
usable_formation_neighbor = [
    column for column in formation_neighbor_columns if column in usable
]

endpoint_ready = bool(
    len(endpoint_report)
    and endpoint_report["finite_fraction"].max() >= 0.995
    and endpoint_report["wells"].max() >= 770
)
aggregate_engine_ready = bool(
    len(usable_pf_scales) >= 4
    and len(usable_beam_members) >= 7
    and len(usable_uncertainty) >= 2
)
common_track8_ready = bool(
    len(usable_correlation) >= 8 and len(usable_formation_neighbor) >= 8
)

# Schema alone cannot prove that the cached PF/beam features came from the same
# SP45 execution as the endpoint.  A matched provenance manifest is mandatory
# before calling a feature-rich reproduction exact.
manifest_candidates = list(OOF_ROOT.glob("**/*sp45*manifest*.json"))
manifest_candidates += list(OOF_ROOT.glob("**/*sp45*verdict*.json"))
matched_manifest = None
for path in manifest_candidates:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    text = json.dumps(payload).lower()
    # An unrelated SP45 experiment is not provenance.  The manifest must name
    # both artifacts whose row-level lineage we are trying to establish.
    if (
        all(token in text for token in ("128", "beam", "selector"))
        and str(FEATURE_DIR).lower() in text
        and str(POOL_PATH).lower() in text
    ):
        matched_manifest = path
        break

verdict = {
    "experiment": "sp45_track8_parity_inventory_v0",
    "stage": "zero_training_artifact_and_provenance_inventory",
    "definitions": {
        "sp45": "multi-seed PF + multi-config beam + per-well selector",
        "track8": "Track6-like GBDT using SP45 engine outputs/features in place of Track6 self-PF",
    },
    "inputs": {
        "folds": str(FOLD_PATH),
        "feature_cache": str(FEATURE_DIR),
        "candidate_pool": str(POOL_PATH),
    },
    "coverage": {
        "fold_wells": int(len(folds)),
        "pool_rows": int(len(pool)),
        "pool_wells": int(pool["well"].nunique()),
        "sampled_cache_wells": int(len(sampled_wells)),
    },
    "inventory": {
        "sp45_endpoint_columns": pool_sp45_columns,
        "usable_pf_scale_columns": usable_pf_scales,
        "usable_beam_member_columns": usable_beam_members,
        "usable_uncertainty_columns": usable_uncertainty,
        "raw_seed_member_columns": seed_member_columns,
        "selector_columns_in_cache": selector_columns,
        "usable_multiscale_correlation_count": len(usable_correlation),
        "usable_formation_neighbor_count": len(usable_formation_neighbor),
        "matched_provenance_manifest": str(matched_manifest) if matched_manifest else None,
    },
    "checks": {
        "endpoint_ready": endpoint_ready,
        "aggregate_engine_features_ready": aggregate_engine_ready,
        "common_track8_features_ready": common_track8_ready,
        "matched_sp45_provenance_ready": matched_manifest is not None,
    },
    "conclusion": (
        "READY_FOR_ENDPOINT-ONLY TRACK8 A/B; rebuild/export matched SP45 internals before feature-rich Track8"
        if endpoint_ready and common_track8_ready and matched_manifest is None
        else "READY_FOR MATCHED TRACK8 A/B"
        if endpoint_ready and aggregate_engine_ready and common_track8_ready and matched_manifest is not None
        else "REBUILD SP45 OOF BUNDLE BEFORE TRACK8"
    ),
    "next": (
        "Export fold-clean PF scales, beam members, dispersion and selector output from one SP45 run; then train paired Track6/Track8 with an identical common block and trainer."
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}

(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("SP45 ENDPOINT INVENTORY")
print(endpoint_report.to_string(index=False), flush=True)
print("\nSP45/TRACK8 PARITY VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
