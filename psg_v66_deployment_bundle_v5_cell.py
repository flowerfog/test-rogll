"""Build the frozen PSG/V66 deployment bundle after PSG-V4 passed.

Run on AutoDL after the PSG five-fold and full-stack LightGBM artifacts exist.
The output directory can be uploaded directly as one Kaggle Dataset.
"""

from pathlib import Path
import hashlib
import json
import shutil


ROOT = Path("/root/autodl-tmp/oof_exports")
PSG_DIR = ROOT / "psg_foldclean_fivefold_oof_v0"
LGB_DIR = ROOT / "psg_foldclean_fullstack_lgb_gateway_v1"
V3_DIR = ROOT / "psg_v66_exact_delta_gateway_v3"
V4_DIR = ROOT / "psg_v66_fixed_dose_robustness_v4"
OUT_DIR = Path("/root/autodl-tmp/psg_v66_delta_deployment_v5")

RUNTIME_CANDIDATES = (
    Path("/root/autodl-tmp/psg_infer_cell.py"),
    Path("psg_infer_cell.py"),
)
FROZEN_DOSE = 0.20


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


runtime_source = next((path for path in RUNTIME_CANDIDATES if path.is_file()), None)
if runtime_source is None:
    raise FileNotFoundError(
        "Upload psg_infer_cell.py to /root/autodl-tmp before building the bundle"
    )

required = [
    LGB_DIR / "base_features.csv",
    LGB_DIR / "psg_features.csv",
    LGB_DIR / "verdict.json",
    V3_DIR / "verdict.json",
    V4_DIR / "verdict.json",
]
for fold in range(1, 6):
    required.extend(
        [
            PSG_DIR / f"psg_oof_fold{fold}.pt",
            LGB_DIR / "models" / f"baseline_fold{fold}.joblib",
            LGB_DIR / "models" / f"psg_features_fold{fold}.joblib",
        ]
    )
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing deployment inputs: " + repr(missing))

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
for name in ("psg_models", "lgb_models", "metadata", "runtime"):
    (OUT_DIR / name).mkdir(parents=True, exist_ok=True)

copies = []
for fold in range(1, 6):
    copies.extend(
        [
            (
                PSG_DIR / f"psg_oof_fold{fold}.pt",
                OUT_DIR / "psg_models" / f"psg_fold{fold}.pt",
            ),
            (
                LGB_DIR / "models" / f"baseline_fold{fold}.joblib",
                OUT_DIR / "lgb_models" / f"baseline_fold{fold}.joblib",
            ),
            (
                LGB_DIR / "models" / f"psg_features_fold{fold}.joblib",
                OUT_DIR / "lgb_models" / f"psg_features_fold{fold}.joblib",
            ),
        ]
    )

copies.extend(
    [
        (LGB_DIR / "base_features.csv", OUT_DIR / "metadata" / "base_features.csv"),
        (LGB_DIR / "psg_features.csv", OUT_DIR / "metadata" / "psg_features.csv"),
        (LGB_DIR / "verdict.json", OUT_DIR / "metadata" / "feature_gateway.json"),
        (V3_DIR / "verdict.json", OUT_DIR / "metadata" / "v66_delta_gateway.json"),
        (V4_DIR / "verdict.json", OUT_DIR / "metadata" / "fixed_dose_robustness.json"),
        (runtime_source, OUT_DIR / "runtime" / "psg_infer_cell.py"),
    ]
)
for source, destination in copies:
    shutil.copy2(source, destination)

config = {
    "experiment": "psg_v66_delta_deployment_v5",
    "parent": "exact complete 6.377 CP8 + Neighbor submission",
    "prediction_rule": "parent + 0.20 * (psg_feature_lgb - baseline_lgb)",
    "dose": FROZEN_DOSE,
    "psg_aggregation": "mean of five checkpoint predictions",
    "lgb_aggregation": "mean of five fold-model deltas before common postprocess",
    "placement": "after every 6.377 mainline overlay and immediately before serialization",
    "prohibited_rule": "do not blend the raw PSG trajectory directly into the parent",
    "validation_status": "retrospective five-fold OOF; one LB validation candidate",
}
(OUT_DIR / "deployment_config.json").write_text(
    json.dumps(config, indent=2), encoding="utf-8"
)

manifest = []
for path in sorted(path for path in OUT_DIR.rglob("*") if path.is_file()):
    manifest.append(
        {
            "path": str(path.relative_to(OUT_DIR)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    )
(OUT_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

total_bytes = sum(item["bytes"] for item in manifest)
print(
    "PSG-V5 DEPLOYMENT BUNDLE",
    {
        "path": str(OUT_DIR),
        "files": len(manifest) + 1,
        "size_mb": total_bytes / 1024**2,
        "dose": FROZEN_DOSE,
    },
    flush=True,
)
archive = shutil.make_archive(
    "/root/autodl-tmp/psg_v66_delta_deployment_v5",
    "zip",
    "/root/autodl-tmp",
    OUT_DIR.name,
)
print("Upload either the directory or this archive as one Kaggle Dataset:")
print(archive)
