"""Build a compact, deterministic research subset from the full handoff bundle.

The primary sample contains the 40 highest-error wells plus 80 wells balanced
over fold and error severity. Neighbor reference wells required by the primary
sample are added only to the neighbor/typewell sections, not to every artifact.
"""

from pathlib import Path
import json
import shutil
import time

import numpy as np
import pandas as pd


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
SOURCE = OOF_ROOT / "research_handoff_bundle_v1"
OUTPUT = OOF_ROOT / "research_handoff_subset_v1_n120"
PRIMARY_WELLS = 120
TOP_ERROR_WELLS = 40
ERROR_BINS = 8
RANDOM_SEED = 20260716

PER_WELL_SECTIONS = (
    "candidate_paths",
    "local_evidence/trajectory_predictions",
    "local_evidence/pf_seed_evidence",
    "local_evidence/learned_matcher_samples",
    "local_evidence/pf_dense",
    "local_evidence/learned_dense",
    "zero_learning_dp/paths",
)


def _directory_size(path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_per_well(relative, wells):
    source_dir = SOURCE / relative
    output_dir = OUTPUT / relative
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    if not source_dir.exists():
        return copied
    for well in sorted(wells):
        candidates = list(source_dir.glob(f"{well}.*"))
        candidates += list(source_dir.glob(f"{well}_*"))
        seen = set()
        for source in candidates:
            if source in seen or not source.is_file():
                continue
            seen.add(source)
            _copy_file(source, output_dir / source.name)
            copied += 1
    return copied


def _well_metrics():
    trajectory_dir = SOURCE / "local_evidence" / "trajectory_predictions"
    curve_dir = SOURCE / "neighbor" / "reference_curves"
    rows = []
    for path in sorted(trajectory_dir.glob("*.npz")):
        well = path.stem
        curve_path = curve_dir / f"{well}.npz"
        if not curve_path.exists():
            continue
        with np.load(path, allow_pickle=False) as trajectory:
            row_idx = trajectory["row_idx"].astype(int)
            fold = int(np.ravel(trajectory["fold"])[0])
            prediction_name = "cp12" if "cp12" in trajectory.files else "strong"
            prediction = trajectory[prediction_name].astype(float)
        with np.load(curve_path, allow_pickle=False) as curve:
            truth = curve["tvt"].astype(float)[row_idx]
        valid = np.isfinite(truth) & np.isfinite(prediction)
        if not valid.any():
            continue
        error = prediction[valid] - truth[valid]
        rows.append(
            {
                "well": well,
                "fold": fold,
                "rows": int(valid.sum()),
                "reference_prediction": prediction_name,
                "rmse": float(np.sqrt(np.mean(error**2))),
                "sse": float(np.sum(error**2)),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) < PRIMARY_WELLS:
        raise RuntimeError(f"Only {len(frame)} complete wells in the full bundle")
    return frame


def _select_primary(metrics):
    metrics = metrics.copy()
    metrics["error_bin"] = pd.qcut(
        metrics["rmse"].rank(method="first"), ERROR_BINS, labels=False
    ).astype(int)
    top = metrics.nlargest(TOP_ERROR_WELLS, "sse")
    selected = set(top["well"].astype(str))
    rng = np.random.default_rng(RANDOM_SEED)

    # Two wells per fold x severity bin gives 80 representative wells.
    for fold in sorted(metrics["fold"].unique()):
        for error_bin in range(ERROR_BINS):
            candidates = metrics.loc[
                (metrics["fold"] == fold)
                & (metrics["error_bin"] == error_bin)
                & ~metrics["well"].isin(selected),
                "well",
            ].astype(str).to_numpy()
            if len(candidates):
                take = min(2, len(candidates), PRIMARY_WELLS - len(selected))
                selected.update(rng.choice(candidates, size=take, replace=False).tolist())
            if len(selected) >= PRIMARY_WELLS:
                break
        if len(selected) >= PRIMARY_WELLS:
            break

    if len(selected) < PRIMARY_WELLS:
        candidates = metrics.loc[~metrics["well"].isin(selected), "well"].astype(str).to_numpy()
        selected.update(
            rng.choice(candidates, size=PRIMARY_WELLS - len(selected), replace=False).tolist()
        )
    return set(sorted(selected)[:PRIMARY_WELLS]), metrics


def _subset_table(source, selected, known_wells):
    frame = pd.read_csv(source, dtype=str)
    if frame.empty:
        return frame, set()
    well_columns = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in ("well", "neighbor", "reference"))
    ]
    target_columns = [
        column
        for column in well_columns
        if "target" in column.lower() or column.lower() in {"well", "wid"}
    ]
    match_columns = target_columns or well_columns
    if not match_columns:
        return frame.iloc[0:0].copy(), set()
    mask = np.zeros(len(frame), dtype=bool)
    for column in match_columns:
        mask |= frame[column].astype(str).isin(selected).to_numpy()
    subset = frame.loc[mask].copy()
    dependencies = set()
    for column in well_columns:
        dependencies.update(
            value for value in subset[column].astype(str) if value in known_wells
        )
    return subset, dependencies


if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

started = time.time()
OUTPUT.mkdir(parents=True, exist_ok=True)
metrics = _well_metrics()
primary, metrics = _select_primary(metrics)
known_wells = set(metrics["well"].astype(str))

neighbor_dir = SOURCE / "neighbor"
neighbor_dependencies = set(primary)
filtered_tables = {}
for name in ("selected_neighbors.csv", "neighbor_shortlist_diagnostics.csv"):
    source = neighbor_dir / name
    if source.exists():
        subset, dependencies = _subset_table(source, primary, known_wells)
        neighbor_dependencies.update(dependencies)
        filtered_tables[name] = subset

selection = metrics.copy()
selection["primary_selected"] = selection["well"].isin(primary)
selection["neighbor_dependency"] = selection["well"].isin(neighbor_dependencies)
selection = selection.sort_values(
    ["primary_selected", "sse"], ascending=[False, False]
)
(OUTPUT / "metadata").mkdir(parents=True, exist_ok=True)
selection.to_csv(OUTPUT / "metadata" / "well_selection.csv", index=False)

counts = {}
for relative in PER_WELL_SECTIONS:
    counts[relative] = _copy_per_well(relative, primary)

# Neighbor reference curves must include any source wells used by selected targets.
counts["neighbor/reference_curves"] = _copy_per_well(
    "neighbor/reference_curves", neighbor_dependencies
)
for name, frame in filtered_tables.items():
    destination = OUTPUT / "neighbor" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)

# Keep small aggregate reports, but omit full row-level neighbor_transfer_oof.npz.
for relative in (
    "neighbor/neighbor_transfer_scan.csv",
    "neighbor/neighbor_transfer_folds.csv",
    "neighbor/best_by_base.csv",
    "neighbor/verdict.json",
    "zero_learning_dp/diagnostics.csv",
    "zero_learning_dp/standalone_scores.csv",
    "zero_learning_dp/blend_scores.csv",
    "zero_learning_dp/verdict.json",
    "schema.json",
):
    source = SOURCE / relative
    if source.exists():
        _copy_file(source, OUTPUT / relative)

# Fold map for primary wells and neighbor dependencies.
fold_source = SOURCE / "metadata" / "fold_clean_well_map.csv"
if fold_source.exists():
    fold = pd.read_csv(fold_source, dtype={"well": str})
    fold = fold[fold["well"].isin(primary | neighbor_dependencies)]
    fold.to_csv(OUTPUT / "metadata" / "fold_clean_well_map.csv", index=False)

# Copy only typewell groups referenced by the selected/dependency wells.
typewell_map_path = SOURCE / "typewell" / "well_to_typewell_group.csv"
typewell_groups = set()
if typewell_map_path.exists():
    typewell_map = pd.read_csv(typewell_map_path, dtype=str)
    typewell_map = typewell_map[
        typewell_map["well"].isin(primary | neighbor_dependencies)
    ].copy()
    destination = OUTPUT / "typewell" / "well_to_typewell_group.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    typewell_map.to_csv(destination, index=False)
    typewell_groups = set(typewell_map["typewell_group"].astype(str))
    for group in sorted(typewell_groups):
        source = SOURCE / "typewell" / "groups" / f"{group}.npz"
        if source.exists():
            _copy_file(source, OUTPUT / "typewell" / "groups" / source.name)

manifest = {
    "bundle": "research_handoff_subset_v1",
    "source": str(SOURCE),
    "output": str(OUTPUT),
    "selection": {
        "primary_wells": len(primary),
        "forced_top_error_wells": TOP_ERROR_WELLS,
        "balanced_wells": len(primary) - TOP_ERROR_WELLS,
        "folds": selection.loc[selection["primary_selected"], "fold"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "neighbor_dependency_wells": len(neighbor_dependencies - primary),
        "typewell_groups": len(typewell_groups),
        "seed": RANDOM_SEED,
    },
    "copied_files_by_section": counts,
    "notes": {
        "pf_and_learned_dense": (
            "Automatically copied when per-well files are present under "
            "local_evidence/pf_dense or local_evidence/learned_dense."
        ),
        "neighbor_transfer_oof": "Omitted because it is full row-level data.",
    },
    "size_mb": _directory_size(OUTPUT) / 2**20,
    "elapsed_seconds": time.time() - started,
}
(OUTPUT / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
)

print("SUBSET MANIFEST")
print(json.dumps(manifest, indent=2))
print("saved:", OUTPUT)
