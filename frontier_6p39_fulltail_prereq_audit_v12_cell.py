"""Audit prerequisites for an honest CP8 full-tail OOF replay on AutoDL.

This cell does not score model-package predictions. It determines whether the
uploaded package contains fold-clean models or persisted OOF predictions; a
deployment-only all-well model is not an honest source for 773-well OOF.
"""

from pathlib import Path
import json

import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "frontier_6p39_fulltail_prereq_audit_v12"
OUT_DIR.mkdir(parents=True, exist_ok=True)

required = {
    "cp8_oof": OOF_ROOT
    / "frontier_6p39_cp8_nocp12_matched_v9"
    / "cp8_nocp12_oof.parquet",
    "midpoint_stats": OOF_ROOT
    / "pf_bimodal_midpoint_hedge_oof_v0"
    / "bimodal_well_statistics.csv",
    "fold_metadata": OOF_ROOT
    / "foldclean_full_fleongg_lgb_v0"
    / "well_metadata.csv",
}

required_report = []
for name, path in required.items():
    required_report.append(
        {
            "name": name,
            "path": str(path),
            "exists": bool(path.exists()),
            "size_mb": float(path.stat().st_size / 2**20) if path.exists() else None,
        }
    )

manifest_paths = sorted(ROOT.glob("**/metadata/model_package_manifest.json"))
packages = []
file_rows = []
for manifest_path in manifest_paths:
    package_root = manifest_path.parents[1]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        error = None
    except Exception as exc:
        manifest = {}
        error = repr(exc)

    entries = manifest.get("models", manifest.get("model_entries", []))
    if not isinstance(entries, list):
        entries = []
    entry_fold_fields = []
    for number, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        fold_payload = {
            key: value
            for key, value in entry.items()
            if any(token in key.lower() for token in ("fold", "split", "train", "valid", "oof"))
        }
        if fold_payload:
            entry_fold_fields.append({"entry": number, **fold_payload})

    package_files = [path for path in package_root.rglob("*") if path.is_file()]
    oof_like = [
        path
        for path in package_files
        if any(token in path.name.lower() for token in ("oof", "fold", "split"))
    ]
    for path in oof_like:
        file_rows.append(
            {
                "package_root": str(package_root),
                "relative_path": str(path.relative_to(package_root)),
                "size_mb": float(path.stat().st_size / 2**20),
            }
        )

    explicit_oof_keys = {
        key: value
        for key, value in manifest.items()
        if any(token in key.lower() for token in ("oof", "fold", "split"))
    }
    fold_evidence = bool(entry_fold_fields or explicit_oof_keys or oof_like)
    packages.append(
        {
            "package_root": str(package_root),
            "manifest": str(manifest_path),
            "manifest_error": error,
            "model_entries": int(len(entries)),
            "model_types": sorted(
                {
                    str(entry.get("model_type", "unknown"))
                    for entry in entries
                    if isinstance(entry, dict)
                }
            ),
            "entry_fold_fields": entry_fold_fields,
            "manifest_fold_oof_fields": explicit_oof_keys,
            "oof_or_fold_named_files": int(len(oof_like)),
            "fold_clean_evidence_found": fold_evidence,
            "honest_oof_status": (
                "candidate_fold_artifacts_found_review_required"
                if fold_evidence
                else "deployment_only_no_honest_773_oof"
            ),
        }
    )

required_frame = pd.DataFrame(required_report)
files_frame = pd.DataFrame(file_rows)
required_frame.to_csv(OUT_DIR / "required_inputs.csv", index=False)
files_frame.to_csv(OUT_DIR / "package_oof_fold_files.csv", index=False)

report = {
    "experiment": "frontier_6p39_fulltail_prereq_audit_v12",
    "required_inputs": required_report,
    "model_packages": packages,
    "model_package_count": int(len(packages)),
    "all_required_inputs_present": bool(all(row["exists"] for row in required_report)),
    "honest_model_package_candidate_count": int(
        sum(package["fold_clean_evidence_found"] for package in packages)
    ),
    "next": (
        "review fold artifacts, then build matched package OOF replay"
        if any(package["fold_clean_evidence_found"] for package in packages)
        else "exclude model-package from OOF metric; reconstruct visible-prefix and midpoint only"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

print("FULLTAIL REQUIRED INPUTS")
print(required_frame.to_string(index=False), flush=True)
print("\nMODEL PACKAGE AUDIT")
print(json.dumps(report, indent=2), flush=True)
if len(files_frame):
    print("\nOOF/FOLD-LIKE PACKAGE FILES")
    print(files_frame.to_string(index=False), flush=True)
print("saved:", OUT_DIR, flush=True)
