"""Disk cleanup before building the full strict-spatial deployment bundle.

Run once with EXECUTE=False. Review the printed DELETE NOW table, then set
EXECUTE=True and rerun. Required K5 feature-bank and strict-spatial CV assets
are protected by explicit completeness checks.
"""

from pathlib import Path
import shutil

import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF = ROOT / "oof_exports"
ARCHIVE = OOF / "_archived_reports"

EXECUTE = False
EMPTY_TRASH = True

BANK = OOF / "multicut_fleongg_matched_k3_gateway_v1"
K5 = OOF / "multicut_fleongg_matched_k5_extension_v1"
STRICT_CV = OOF / "multicut_k5_strict_outer_spatial_v1"
FULL_RUN = OOF / "multicut_k5_full_spatial_lgb_bundle_v1"
FULL_BUNDLE = FULL_RUN / "strict_spatial_full_lgb_bundle_v1.joblib"

K5_TAGS = ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4")


def _size_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            pass
    return total


def _assert_safe_target(path):
    path = Path(path).resolve()
    allowed_parents = {OOF.resolve(), (ROOT / ".Trash-0").resolve()}
    if path == (ROOT / ".Trash-0").resolve():
        return
    if path.parent == OOF.resolve():
        return
    if OOF.resolve() in path.parents:
        return
    raise RuntimeError(f"Refusing path outside managed cleanup roots: {path}")


def _archive_top_level_reports(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return
    destination = ARCHIVE / directory.name
    destination.mkdir(parents=True, exist_ok=True)
    for item in directory.iterdir():
        if not item.is_file():
            continue
        if item.suffix.lower() not in {".csv", ".json", ".txt"}:
            continue
        if item.stat().st_size > 5 * 2**20:
            continue
        shutil.copy2(item, destination / item.name)


def _remove(path, archive_reports=False):
    path = Path(path)
    if not path.exists():
        return
    _assert_safe_target(path)
    if archive_reports:
        _archive_top_level_reports(path)
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


# ---------------------------------------------------------------------------
# Dependency guards.
# ---------------------------------------------------------------------------
required_bank_files = [
    BANK / "well_metadata.csv",
    BANK / "cut_manifest.csv",
    BANK / "matched_k3_gateway_features.csv",
    K5 / "new_cut_manifest.csv",
    BANK / "features",
]
missing_bank_files = [str(path) for path in required_bank_files if not path.exists()]
if missing_bank_files:
    raise RuntimeError("Active K5 bank is incomplete: " + repr(missing_bank_files))

meta = pd.read_csv(BANK / "well_metadata.csv", dtype={"well": str})
feature_dir = BANK / "features"
missing_states = []
for wid in meta["well"].astype(str):
    for tag in K5_TAGS:
        path = feature_dir / f"{wid}__{tag}__sample.pkl"
        if not path.exists():
            missing_states.append(str(path))
    official_full = feature_dir / f"{wid}__official_full.pkl"
    if not official_full.exists():
        missing_states.append(str(official_full))
if missing_states:
    raise RuntimeError(
        f"Do not delete donors: active K5 bank misses {len(missing_states)} files; "
        f"first={missing_states[:10]}"
    )

required_cv = [
    STRICT_CV / "all_features.csv",
    STRICT_CV / "strict_outer_spatial_oof.pkl",
    *[
        STRICT_CV / "models" / f"spatial_lgb_fold{fold}.joblib"
        for fold in range(1, 6)
    ],
]
missing_cv = [str(path) for path in required_cv if not path.exists()]
if missing_cv:
    raise RuntimeError("Strict-spatial CV essentials missing: " + repr(missing_cv))

print(
    {
        "active_wells": int(meta["well"].nunique()),
        "active_k5_states": int(meta["well"].nunique()) * len(K5_TAGS),
        "strict_cv_models": 5,
        "full_bundle_exists": FULL_BUNDLE.exists(),
        "execute": EXECUTE,
    }
)


# ---------------------------------------------------------------------------
# Safe now: no path below is read by the full-bundle training cell.
# ---------------------------------------------------------------------------
delete_now = [
    # Biggest new caches. Keep STRICT_CV/models, all_features.csv and OOF.
    STRICT_CV / "raw_spatial",
    STRICT_CV / "spatial_states",

    # Model-family branches have been rejected; preserve only tiny reports.
    OOF / "multicut_k5_strict_spatial_three_model_v1",
    OOF / "multicut_fleongg_matched_k5_three_model_v1",

    # K3/K5 training models are no longer read; feature bank/manifests remain.
    BANK / "models",
    K5 / "models",

    # Donor feature run is redundant because the active bank was verified.
    OOF / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75",

    # Previously rejected large branches. Missing paths are simply ignored.
    OOF / "learned_emission_dp_v0_fullgrid_step4_stride8",
    OOF / "learned_emission_dp_v1_baseband64_step4_stride8",
    OOF / "learned_emission_dp_v2_matcher_v1_fullgrid_step4_stride8",
    OOF / "learned_emission_dp_v3_matcher_v1_baseprior_step4_stride8",
    OOF / "learned_emission_baseprior_dp_v0",
    OOF / "learned_emission_dp_v0",
    OOF / "learned_emission_dp_v1",
    OOF / "learned_emission_dp_v2",
    OOF / "learned_emission_dp_v3",
    OOF / "learned_gr_matcher_v0_win65_rows128_offsets4to64",
    OOF / "learned_gr_matcher_v1_global_ncc_hardneg27",
    OOF / "pf_multimodal_input_ablation_v0_s16_p200",
    OOF / "forward_ncc_profile_ablation_v0",
    OOF / "anchor_aware_tcn_v0_cp12_matcher33_stride8",
    OOF / "repaired_likpf_spatial_triage_v1",
    OOF / "aux_future_deltas_v0_k5_h025_050_075_100",
    OOF / "aux_future_calibration_rescore_v0",
]

if EMPTY_TRASH:
    delete_now.insert(0, ROOT / ".Trash-0")

rows = []
for path in delete_now:
    size = _size_bytes(path)
    if size:
        rows.append(
            {
                "path": str(path),
                "size_gb": size / 2**30,
                "action": "DELETE NOW",
            }
        )
delete_table = pd.DataFrame(rows)
if len(delete_table):
    delete_table = delete_table.sort_values("size_gb", ascending=False).reset_index(drop=True)
else:
    delete_table = pd.DataFrame(columns=["path", "size_gb", "action"])

print("\nDELETE NOW")
display(delete_table)
print("reclaimable_gb:", float(delete_table["size_gb"].sum()))


# These become safe only after the full bundle has been created and loaded.
after_bundle = [
    FULL_RUN / "raw_spatial",
    FULL_RUN / "spatial_states",
]
after_rows = [
    {"path": str(path), "size_gb": _size_bytes(path) / 2**30}
    for path in after_bundle
    if path.exists()
]
print("\nDELETE AFTER BUNDLE SUCCESS")
display(pd.DataFrame(after_rows))

if not EXECUTE:
    print("\nDRY RUN ONLY. Set EXECUTE=True and rerun to delete DELETE NOW paths.")
else:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for path in delete_now:
        if not Path(path).exists():
            continue
        print("deleting:", path)
        _remove(path, archive_reports=(Path(path).parent == OOF))
    print("\ncleanup complete")
    usage = shutil.disk_usage(ROOT)
    print(
        {
            "total_gb": usage.total / 2**30,
            "used_gb": usage.used / 2**30,
            "free_gb": usage.free / 2**30,
        }
    )
