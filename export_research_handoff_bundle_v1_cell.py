"""Export the surviving trajectory, local-evidence, spatial, and DP artifacts.

The bundle is self-describing and uses one compressed file per well so it can
be resumed, inspected, or uploaded in parts. Candidate matrices are stored as
0.01-ft quantised corrections relative to `strong`; this preserves the full
candidate TVT matrix while avoiding several gigabytes of repeated base values.
"""

from pathlib import Path
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd


DATA_ROOT_CANDIDATES = (
    Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    Path(r"E:\rogii-wellbore-geology-prediction"),
)
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
OUT_DIR = OOF_ROOT / "research_handoff_bundle_v1"
QUANTISATION_FT = 0.01

POOL_CANDIDATES = (
    OOF_ROOT / "candidate_pool_full_merged_oof_honest_fle.csv",
    OOF_ROOT / "candidate_pool_full_merged_oof.csv",
)
PATH_BANK_DIR = OOF_ROOT / "gr_path_bank_oracle_v0" / "banks"
FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "gr_path_bank_oracle_v0" / "well_metadata.csv",
)
NEIGHBOR_DIR = OOF_ROOT / "neighbor_fullcurve_transfer_v0"
ZERO_DP_DIR = OOF_ROOT / "tvt_drift_zero_learning_dp_v0"


def _find_data_root():
    for root in DATA_ROOT_CANDIDATES:
        if (root / "train").exists() and next(
            (root / "train").glob("*__horizontal_well.csv"), None
        ):
            return root
    raise FileNotFoundError("cannot locate competition train directory")


def _first_existing(paths):
    return next((path for path in paths if path.exists()), None)


def _source_family(name):
    name = str(name)
    if name == "zero":
        return "zero"
    if name.startswith("line_"):
        return "linear_endpoint_quantile"
    if name.startswith("raw_"):
        return "raw_shape_kmeans"
    if name.startswith("lenhalf_"):
        return "length_normalised_kmeans"
    if name == "pca_mean":
        return "pca_mean"
    if name.startswith("pca_"):
        return "pca_component"
    return "other"


def _save_frame_npz(path, frame):
    arrays = {}
    for column in frame.columns:
        values = frame[column].to_numpy()
        if values.dtype == object:
            values = values.astype(str)
        arrays[column] = values
    np.savez_compressed(path, **arrays)


DATA_ROOT = _find_data_root()
TRAIN_DIR = DATA_ROOT / "train"
POOL_PATH = _first_existing(POOL_CANDIDATES)
FOLD_PATH = _first_existing(FOLD_CANDIDATES)
if POOL_PATH is None:
    raise FileNotFoundError("candidate pool not found")
if FOLD_PATH is None:
    raise FileNotFoundError("fold metadata not found")

directories = {
    "candidate_paths": OUT_DIR / "candidate_paths",
    "local_predictions": OUT_DIR / "local_evidence" / "trajectory_predictions",
    "pf_seed_evidence": OUT_DIR / "local_evidence" / "pf_seed_evidence",
    "learned_local": OUT_DIR / "local_evidence" / "learned_matcher_samples",
    "neighbor_curves": OUT_DIR / "neighbor" / "reference_curves",
    "typewell_groups": OUT_DIR / "typewell" / "groups",
    "zero_dp": OUT_DIR / "zero_learning_dp" / "paths",
    "metadata": OUT_DIR / "metadata",
}
for directory in directories.values():
    directory.mkdir(parents=True, exist_ok=True)

started = time.time()
manifest = {
    "bundle": "research_handoff_bundle_v1",
    "data_root": str(DATA_ROOT),
    "oof_root": str(OOF_ROOT),
    "pool_source": str(POOL_PATH),
    "fold_source": str(FOLD_PATH),
    "sections": {},
    "limitations": {
        "pf_dense_column_likelihood": (
            "Not persisted by the original PF runs. Export includes seed paths "
            "and seed-level log likelihood where those caches survive."
        ),
        "learned_dense_score_volume": (
            "Not persisted. Export includes sampled-column local best TVT and "
            "peak margin from the fold-clean learned-emission DP cache."
        ),
        "zero_dp_forward_backward_margin": (
            "Not persisted. Export includes all decoded paths plus well-level "
            "mean emission sharpness from diagnostics."
        ),
    },
}

print(
    {
        "experiment": "export_research_handoff_bundle_v1",
        "data_root": str(DATA_ROOT),
        "pool": str(POOL_PATH),
        "output": str(OUT_DIR),
        "free_gb": shutil.disk_usage(OOF_ROOT).free / 2**30,
    },
    flush=True,
)

# ---------------------------------------------------------------------------
# Common fold map and row-level trajectory predictions.
# ---------------------------------------------------------------------------
fold_frame = pd.read_csv(FOLD_PATH, dtype={"well": str})
if not {"well", "fold"}.issubset(fold_frame.columns):
    raise RuntimeError("fold metadata must contain well/fold")
fold_frame["well"] = fold_frame["well"].astype(str)
fold_map = fold_frame.set_index("well")["fold"].astype(int).to_dict()
fold_frame.to_csv(directories["metadata"] / "fold_clean_well_map.csv", index=False)

header = pd.read_csv(POOL_PATH, nrows=0).columns.tolist()
identity = ["well", "row_idx"] if {"well", "row_idx"}.issubset(header) else ["id"]
prediction_columns = [
    column
    for column in (
        "strong",
        "strong_bias",
        "cp12",
        "pf_ancc",
        "pf_z",
        "likpf_scale_3",
        "likpf_scale_5",
        "likpf_scale_8",
        "likpf_scale_12",
        "likpf_mean",
    )
    if column in header
]
pool = pd.read_csv(
    POOL_PATH,
    usecols=identity + prediction_columns,
    dtype={"id": str, "well": str},
)
if "well" not in pool:
    split = pool["id"].astype(str).str.rsplit("_", n=1, expand=True)
    pool["well"] = split[0].astype(str)
    pool["row_idx"] = split[1].astype(np.int32)
    pool = pool.drop(columns="id")
pool["well"] = pool["well"].astype(str)
pool["row_idx"] = pool["row_idx"].astype(np.int32)
pool["fold"] = pool["well"].map(fold_map).astype(int)
pool_groups = {
    str(well): frame.sort_values("row_idx").reset_index(drop=True)
    for well, frame in pool.groupby("well", sort=False)
}

for number, (well, frame) in enumerate(pool_groups.items(), 1):
    arrays = {
        "row_idx": frame["row_idx"].to_numpy(np.int32),
        "fold": np.asarray([int(frame["fold"].iloc[0])], np.int8),
    }
    for column in prediction_columns:
        arrays[column] = frame[column].to_numpy(np.float32)
    np.savez_compressed(directories["local_predictions"] / f"{well}.npz", **arrays)
    if number % 100 == 0 or number == len(pool_groups):
        print(f"trajectory predictions {number}/{len(pool_groups)}", flush=True)
manifest["sections"]["trajectory_predictions"] = {
    "wells": len(pool_groups),
    "columns": prediction_columns,
    "meaning": "Row-level OOF trajectory predictions; likpf_* are paths, not dense likelihood surfaces.",
}

# ---------------------------------------------------------------------------
# 1. Fold-clean candidate path matrices.
# ---------------------------------------------------------------------------
candidate_exported = 0
candidate_failures = []
bank_cache = {}
for number, (well, frame) in enumerate(pool_groups.items(), 1):
    output_path = directories["candidate_paths"] / f"{well}.npz"
    if output_path.exists():
        candidate_exported += 1
        continue
    try:
        fold = int(frame["fold"].iloc[0])
        if fold not in bank_cache:
            bank_path = PATH_BANK_DIR / f"fold{fold}_global.npz"
            if not bank_path.exists():
                raise FileNotFoundError(bank_path)
            with np.load(bank_path, allow_pickle=False) as source:
                bank_cache[fold] = {key: source[key] for key in source.files}
        bank = bank_cache[fold]
        horizontal = pd.read_csv(
            TRAIN_DIR / f"{well}__horizontal_well.csv", usecols=["MD"]
        )
        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
        rows = frame["row_idx"].to_numpy(int)
        anchor = int(rows.min() - 1)
        eval_span = float(max(md[rows.max()] - md[anchor], 1.0))
        progress = np.clip((md[rows] - md[anchor]) / eval_span, 0.0, 1.0)
        templates = bank["templates"].astype(np.float32)
        alphas = bank["length_alphas"].astype(float)
        names = bank["names"].astype(str)
        progress_grid = bank["progress_grid"].astype(float)
        median_length = float(bank["median_length"][0])
        pointwise_cap = float(bank["pointwise_cap"][0])
        scale = max(eval_span / max(median_length, 1.0), 0.1) ** alphas
        corrections = np.stack(
            [np.interp(progress, progress_grid, template) for template in templates]
        )
        corrections *= scale[:, None]
        corrections = np.clip(corrections, -pointwise_cap, pointwise_cap)
        correction_q01 = np.rint(corrections / QUANTISATION_FT).astype(np.int16)
        source_family = np.asarray([_source_family(name) for name in names], dtype=str)
        np.savez_compressed(
            output_path,
            row_idx=rows.astype(np.int32),
            fold=np.asarray([fold], np.int8),
            strong=frame["strong"].to_numpy(np.float32),
            correction_q01=correction_q01,
            quantisation_ft=np.asarray([QUANTISATION_FT], np.float32),
            candidate_names=names,
            source_family=source_family,
            eval_span_ft=np.asarray([eval_span], np.float32),
            source_bank=np.asarray([f"fold{fold}_global"], dtype=str),
        )
        candidate_exported += 1
    except Exception as exc:
        candidate_failures.append({"well": well, "reason": repr(exc)})
    if number % 50 == 0 or number == len(pool_groups):
        print(
            f"candidate matrices {number}/{len(pool_groups)} "
            f"ok={candidate_exported} failures={len(candidate_failures)}",
            flush=True,
        )
pd.DataFrame(candidate_failures).to_csv(
    directories["metadata"] / "candidate_export_failures.csv", index=False
)
manifest["sections"]["candidate_paths"] = {
    "wells": candidate_exported,
    "failures": len(candidate_failures),
    "shape": "correction_q01[n_candidates, n_eval_rows]",
    "reconstruction": "candidate_tvt = strong[None, :] + 0.01 * correction_q01",
    "source_labels": "candidate_names and source_family",
    "fold_clean": True,
}

# ---------------------------------------------------------------------------
# 2a. Surviving PF seed evidence caches.
# ---------------------------------------------------------------------------
pf_directories = []
pf_directories.extend(path for path in OOF_ROOT.glob("pf_seed_cache_*") if path.is_dir())
pf_directories.extend(path for path in OOF_ROOT.glob("pf_multimodal_input*/pf_cache") if path.is_dir())
pf_inventory = []
pf_exported_wells = set()
for directory in pf_directories:
    files = sorted(directory.glob("*.npz"))
    pf_inventory.append({"source": str(directory), "files": len(files)})
    for source_path in files:
        well = source_path.stem
        output_path = directories["pf_seed_evidence"] / f"{well}.npz"
        if output_path.exists():
            pf_exported_wells.add(well)
            continue
        try:
            with np.load(source_path, allow_pickle=False) as source:
                keys = set(source.files)
                arrays = {}
                for key in (
                    "eval_pos",
                    "pred_arr",
                    "log_lik",
                    "features",
                    "names",
                    "n_seeds",
                    "n_particles",
                ):
                    if key in keys:
                        arrays[key] = source[key]
                if not arrays:
                    continue
                arrays["source_path"] = np.asarray([str(source_path)], dtype=str)
                np.savez_compressed(output_path, **arrays)
                pf_exported_wells.add(well)
        except Exception:
            continue
pd.DataFrame(pf_inventory).to_csv(
    directories["metadata"] / "pf_cache_inventory.csv", index=False
)
manifest["sections"]["pf_seed_evidence"] = {
    "wells": len(pf_exported_wells),
    "sources": pf_inventory,
    "available": "seed trajectories/global log_lik or compact PF multimodal features",
}

# ---------------------------------------------------------------------------
# 2b. Learned matcher sampled-column local scores.
# ---------------------------------------------------------------------------
learned_candidates = []
for well_dir in OOF_ROOT.glob("learned_emission_dp*/wells"):
    sample_files = list(well_dir.glob("*_samples.pkl"))
    if sample_files:
        learned_candidates.append((len(sample_files), well_dir, sample_files))
learned_source = None
learned_exported = 0
if learned_candidates:
    _, learned_source, sample_files = max(learned_candidates, key=lambda item: item[0])
    for number, source_path in enumerate(sorted(sample_files), 1):
        well = source_path.stem.removesuffix("_samples")
        output_path = directories["learned_local"] / f"{well}.npz"
        if not output_path.exists():
            frame = pd.read_pickle(source_path)
            _save_frame_npz(output_path, frame)
        learned_exported += 1
        if number % 100 == 0 or number == len(sample_files):
            print(f"learned local scores {number}/{len(sample_files)}", flush=True)
manifest["sections"]["learned_matcher_samples"] = {
    "wells": learned_exported,
    "source": str(learned_source) if learned_source else None,
    "fields": "Includes local_best_tvt/local_peak_margin when present.",
}

# ---------------------------------------------------------------------------
# 3 and 4. Neighbor reference curves and content-derived typewell groups.
# ---------------------------------------------------------------------------
typewell_rows = []
group_representatives = {}
neighbor_curve_count = 0
for number, well in enumerate(sorted(pool_groups), 1):
    horizontal_path = TRAIN_DIR / f"{well}__horizontal_well.csv"
    typewell_path = TRAIN_DIR / f"{well}__typewell.csv"
    horizontal = pd.read_csv(
        horizontal_path, usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"]
    )
    xy = horizontal[["X", "Y"]].to_numpy(np.float32)
    z = horizontal["Z"].to_numpy(np.float32)
    tvt = horizontal["TVT"].to_numpy(np.float32)
    visible = horizontal["TVT_input"].notna().to_numpy()
    anchor = int(np.flatnonzero(visible)[-1])
    direction = xy[-1].astype(float) - xy[anchor].astype(float)
    norm = max(float(np.linalg.norm(direction)), 1e-8)
    direction = (direction / norm).astype(np.float32)
    np.savez_compressed(
        directories["neighbor_curves"] / f"{well}.npz",
        md=horizontal["MD"].to_numpy(np.float32),
        x=xy[:, 0],
        y=xy[:, 1],
        z=z,
        tvt=tvt,
        structure_s=(tvt + z).astype(np.float32),
        tvt_input=horizontal["TVT_input"].to_numpy(np.float32),
        anchor_row=np.asarray([anchor], np.int32),
        direction=direction,
    )
    neighbor_curve_count += 1

    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    clean = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna()
    clean = clean.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
    rounded = np.round(clean[["TVT", "GR"]].to_numpy(np.float64), 6)
    digest = hashlib.sha256(rounded.tobytes()).hexdigest()[:16]
    typewell_rows.append(
        {
            "well": well,
            "typewell_group": digest,
            "rows": len(clean),
            "tvt_min": float(clean["TVT"].min()),
            "tvt_max": float(clean["TVT"].max()),
        }
    )
    if digest not in group_representatives:
        group_representatives[digest] = well
        np.savez_compressed(
            directories["typewell_groups"] / f"{digest}.npz",
            tvt=clean["TVT"].to_numpy(np.float32),
            gr=clean["GR"].to_numpy(np.float32),
            representative_well=np.asarray([well], dtype=str),
        )
    if number % 100 == 0 or number == len(pool_groups):
        print(f"neighbor/typewell curves {number}/{len(pool_groups)}", flush=True)

typewell_map = pd.DataFrame(typewell_rows)
typewell_map.to_csv(OUT_DIR / "typewell" / "well_to_typewell_group.csv", index=False)
typewell_summary = (
    typewell_map.groupby("typewell_group", as_index=False)
    .agg(wells=("well", "size"), representative_well=("well", "first"))
    .sort_values("wells", ascending=False)
)
typewell_summary.to_csv(OUT_DIR / "typewell" / "typewell_group_summary.csv", index=False)

neighbor_files_copied = []
for name in (
    "selected_neighbors.csv",
    "neighbor_shortlist_diagnostics.csv",
    "neighbor_transfer_oof.npz",
    "neighbor_transfer_scan.csv",
    "neighbor_transfer_folds.csv",
    "best_by_base.csv",
    "verdict.json",
):
    source = NEIGHBOR_DIR / name
    if source.exists():
        destination = OUT_DIR / "neighbor" / name
        shutil.copy2(source, destination)
        neighbor_files_copied.append(name)
manifest["sections"]["neighbor"] = {
    "reference_curves": neighbor_curve_count,
    "selection_files": neighbor_files_copied,
    "curve_fields": "MD/X/Y/Z/TVT/S/TVT_input/anchor/direction",
}
manifest["sections"]["typewell_groups"] = {
    "wells": len(typewell_map),
    "groups": len(typewell_summary),
    "grouping": "SHA256 of TVT/GR curve rounded to 1e-6 after duplicate-TVT averaging",
}

# ---------------------------------------------------------------------------
# 6. Existing zero-learning DP paths and available well-level margins.
# ---------------------------------------------------------------------------
zero_dp_exported = 0
zero_dp_source = ZERO_DP_DIR / "wells"
if zero_dp_source.exists():
    zero_dp_files = sorted(zero_dp_source.glob("*.pkl"))
    for number, source_path in enumerate(zero_dp_files, 1):
        frame = pd.read_pickle(source_path)
        output_path = directories["zero_dp"] / f"{source_path.stem}.npz"
        if not output_path.exists():
            _save_frame_npz(output_path, frame)
        zero_dp_exported += 1
        if number % 100 == 0 or number == len(zero_dp_files):
            print(f"zero DP paths {number}", flush=True)
for name in ("diagnostics.csv", "standalone_scores.csv", "blend_scores.csv", "verdict.json"):
    source = ZERO_DP_DIR / name
    if source.exists():
        shutil.copy2(source, OUT_DIR / "zero_learning_dp" / name)
manifest["sections"]["zero_learning_dp"] = {
    "wells": zero_dp_exported,
    "available": "decoded row-level paths and well-level mean sharpness",
    "dense_column_margins": False,
}

# ---------------------------------------------------------------------------
# Schemas and final manifest.
# ---------------------------------------------------------------------------
schema = {
    "candidate_paths": {
        "row_idx": "int32 [R]",
        "strong": "float32 [R]",
        "correction_q01": "int16 [N,R]",
        "candidate_tvt_reconstruction": "strong[None,:] + correction_q01 * quantisation_ft",
        "candidate_names": "string [N]",
        "source_family": "string [N]",
    },
    "neighbor_reference_curves": {
        "md/x/y/z/tvt/structure_s/tvt_input": "float32 [rows]",
        "anchor_row": "int32 [1]",
        "direction": "float32 [2]",
    },
}
manifest["elapsed_minutes"] = (time.time() - started) / 60.0
manifest["free_gb_after"] = shutil.disk_usage(OOF_ROOT).free / 2**30
(OUT_DIR / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
(OUT_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
)

print("\nEXPORT COMPLETE")
print(json.dumps(manifest, indent=2))
print("saved:", OUT_DIR)
