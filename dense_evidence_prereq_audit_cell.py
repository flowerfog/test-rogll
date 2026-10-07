"""Audit whether PF and learned-matcher dense evidence can be reconstructed."""

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OUT_DIR = OOF_ROOT / "dense_evidence_prereq_audit_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V0_DIR = OOF_ROOT / "learned_gr_matcher_v0_win65_rows128_offsets4to64"
V1_DIR = OOF_ROOT / "learned_gr_matcher_v1_global_ncc_hardneg27"
V0_MODELS = {
    f"v0_model_fold{fold}": V0_DIR / "models" / f"matcher_fold{fold}.pt"
    for fold in range(1, 6)
}
V1_MODELS = {
    f"v1_model_fold{fold}": V1_DIR / "models" / f"matcher_fold{fold}.pt"
    for fold in range(1, 6)
}
LEARNED_REQUIRED = {
    "horizontal_windows": V0_DIR / "horizontal_windows.npy",
    "sample_metadata": V0_DIR / "sample_metadata.csv",
    "v0_build_info": V0_DIR / "build_info.json",
    "v1_typewell_windows": V1_DIR / "typewell_candidate_windows.npy",
    "v1_candidate_centres": V1_DIR / "candidate_centres.npy",
    **V0_MODELS,
    **V1_MODELS,
}


def _size(path):
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


learned_rows = []
for name, path in LEARNED_REQUIRED.items():
    learned_rows.append(
        {
            "artifact": name,
            "path": str(path),
            "exists": path.exists(),
            "size_mb": _size(path) / 2**20,
        }
    )
learned = pd.DataFrame(learned_rows)

sample_count = None
sample_wells = None
if LEARNED_REQUIRED["sample_metadata"].exists():
    metadata = pd.read_csv(LEARNED_REQUIRED["sample_metadata"], usecols=["well"])
    sample_count = len(metadata)
    sample_wells = metadata["well"].astype(str).nunique()

pf_directories = []
for pattern in ("pf_seed_cache_*", "pf_multimodal_input*/pf_cache"):
    for path in OOF_ROOT.glob(pattern):
        if path.is_dir():
            files = list(path.glob("*.npz"))
            key_counts = {}
            for source in files[: min(len(files), 20)]:
                try:
                    with np.load(source, allow_pickle=False) as data:
                        signature = ",".join(sorted(data.files))
                    key_counts[signature] = key_counts.get(signature, 0) + 1
                except Exception:
                    key_counts["<unreadable>"] = key_counts.get("<unreadable>", 0) + 1
            pf_directories.append(
                {
                    "path": str(path),
                    "files": len(files),
                    "size_mb": _size(path) / 2**20,
                    "sample_key_signatures": json.dumps(key_counts, sort_keys=True),
                }
            )
pf = pd.DataFrame(pf_directories)

horizontal_files = list((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
eval_rows = 0
if horizontal_files:
    for path in horizontal_files:
        frame = pd.read_csv(path, usecols=["TVT_input"])
        eval_rows += int(frame["TVT_input"].isna().sum())

# Compact export estimates. These are deliberately not estimates for retaining
# every seed x particle x row state, which is much larger.
topk = 8
learned_topk_bytes = eval_rows * topk * (4 + 2)  # TVT float32 + score float16
pf_bins = 65
pf_binned_bytes = eval_rows * pf_bins * 2  # uint16 posterior mass
pf_particle_bytes = eval_rows * 200 * 2 * 2  # state+weight float16, one aggregate run

run_pf_available = "run_particle_filter" in globals()
pf_internals = sorted(
    name
    for name in globals()
    if "particle" in name.lower() or "pf_lik" in name.lower()
)

learned_windows_ready = bool(
    LEARNED_REQUIRED["horizontal_windows"].exists()
    and LEARNED_REQUIRED["sample_metadata"].exists()
)
learned_v0_models_ready = all(path.exists() for path in V0_MODELS.values())
learned_v1_models_ready = all(path.exists() for path in V1_MODELS.values())
learned_replay_ready = bool(learned_windows_ready and learned_v1_models_ready)
learned_retrain_ready = bool(
    (DATA_ROOT / "train").exists()
    and learned_windows_ready
    and learned_v0_models_ready
)
pf_cache_wells = int(pf["files"].max()) if len(pf) else 0
pf_replay_ready = bool(pf_cache_wells >= 700)
pf_rerun_ready = bool(run_pf_available and len(horizontal_files) >= 700)

verdict = {
    "data_root": str(DATA_ROOT),
    "train_wells": len(horizontal_files),
    "eval_rows": eval_rows,
    "free_gb": shutil.disk_usage(OOF_ROOT).free / 2**30,
    "learned": {
        "sample_count": sample_count,
        "sample_wells": sample_wells,
        "windows_ready": learned_windows_ready,
        "v0_models_ready": learned_v0_models_ready,
        "v1_models_ready": learned_v1_models_ready,
        "replay_ready": learned_replay_ready,
        "retrain_ready": learned_retrain_ready,
        "missing": learned.loc[~learned["exists"], "artifact"].tolist(),
        "recommended_export": "top-k TVT/score per eval row; dense sampled grid optional",
    },
    "pf": {
        "cache_max_wells": pf_cache_wells,
        "run_particle_filter_available_in_kernel": run_pf_available,
        "detected_pf_globals": pf_internals,
        "replay_ready": pf_replay_ready,
        "rerun_ready": pf_rerun_ready,
        "recommended_export": "65-bin aggregated posterior per row, not all seed-particle states",
    },
    "estimated_outputs_gb": {
        "learned_top8": learned_topk_bytes / 2**30,
        "pf_65bin_uint16": pf_binned_bytes / 2**30,
        "pf_particle_state_weight_one_run": pf_particle_bytes / 2**30,
    },
    "next": (
        "replay surviving artifacts"
        if learned_replay_ready or pf_replay_ready
        else "rebuild learned matcher and/or run instrumented PF"
    ),
}

learned.to_csv(OUT_DIR / "learned_artifacts.csv", index=False)
pf.to_csv(OUT_DIR / "pf_cache_directories.csv", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("LEARNED ARTIFACTS")
print(learned.to_string(index=False))
print("\nPF CACHE DIRECTORIES")
print(pf.to_string(index=False) if len(pf) else "<none>")
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
