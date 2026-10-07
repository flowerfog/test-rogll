"""Build a compact deployment bundle for the frozen PSG feature correction."""

from pathlib import Path
import hashlib
import json
import shutil


ROOT = Path("/root/autodl-tmp/oof_exports")
PSG_DIR = ROOT / "psg_foldclean_fivefold_oof_v0"
LGB_DIR = ROOT / "psg_foldclean_fullstack_lgb_gateway_v1"
AUDIT_DIR = ROOT / "psg_feature_delta_survival_audit_v2"
OUT_DIR = Path("/root/autodl-tmp/psg_feature_delta_deployment_v0")

FROZEN_DOSE = 0.40
FROZEN_REFERENCE = "cp12_neighbor"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


required = []
for fold in range(1, 6):
    required.extend(
        [
            PSG_DIR / f"psg_oof_fold{fold}.pt",
            LGB_DIR / "models" / f"baseline_fold{fold}.joblib",
            LGB_DIR / "models" / f"psg_features_fold{fold}.joblib",
        ]
    )
required.extend(
    [
        LGB_DIR / "base_features.csv",
        LGB_DIR / "psg_features.csv",
        LGB_DIR / "verdict.json",
        AUDIT_DIR / "verdict.json",
        AUDIT_DIR / "summary.csv",
        AUDIT_DIR / "fold_report.csv",
    ]
)
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Missing deployment inputs: " + repr(missing))

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
(OUT_DIR / "psg_models").mkdir(parents=True)
(OUT_DIR / "lgb_models").mkdir(parents=True)
(OUT_DIR / "metadata").mkdir(parents=True)

copied = []
for fold in range(1, 6):
    mappings = [
        (PSG_DIR / f"psg_oof_fold{fold}.pt", OUT_DIR / "psg_models" / f"psg_fold{fold}.pt"),
        (
            LGB_DIR / "models" / f"baseline_fold{fold}.joblib",
            OUT_DIR / "lgb_models" / f"baseline_fold{fold}.joblib",
        ),
        (
            LGB_DIR / "models" / f"psg_features_fold{fold}.joblib",
            OUT_DIR / "lgb_models" / f"psg_features_fold{fold}.joblib",
        ),
    ]
    for source, destination in mappings:
        shutil.copy2(source, destination)
        copied.append(destination)

metadata_mappings = [
    (LGB_DIR / "base_features.csv", OUT_DIR / "metadata" / "base_features.csv"),
    (LGB_DIR / "psg_features.csv", OUT_DIR / "metadata" / "psg_features.csv"),
    (LGB_DIR / "verdict.json", OUT_DIR / "metadata" / "feature_gateway_verdict.json"),
    (AUDIT_DIR / "verdict.json", OUT_DIR / "metadata" / "delta_survival_verdict.json"),
    (AUDIT_DIR / "summary.csv", OUT_DIR / "metadata" / "delta_survival_summary.csv"),
    (AUDIT_DIR / "fold_report.csv", OUT_DIR / "metadata" / "delta_survival_folds.csv"),
]
for source, destination in metadata_mappings:
    shutil.copy2(source, destination)
    copied.append(destination)

config = {
    "experiment": "psg_feature_delta_deployment_v0",
    "psg_checkpoints": 5,
    "baseline_lgb_models": 5,
    "psg_feature_lgb_models": 5,
    "prediction_rule": "mainline_tvt + dose * (psg_feature_tvt - baseline_feature_tvt)",
    "dose": FROZEN_DOSE,
    "validation_reference": FROZEN_REFERENCE,
    "placement": "after the frozen mainline/neighbor path; before final submission serialization",
    "aggregation": {
        "psg": "mean of five PSG checkpoint predictions",
        "baseline_lgb": "mean of five fold-model deltas, then common postprocess",
        "psg_feature_lgb": "mean of five fold-model deltas, then common postprocess",
    },
    "guardrails": {
        "clip_increment_ft": None,
        "test_time_training_or_optimization": 0,
        "feature_order_from_csv": True,
    },
}
(OUT_DIR / "deployment_config.json").write_text(
    json.dumps(config, indent=2), encoding="utf-8"
)
copied.append(OUT_DIR / "deployment_config.json")

manifest = []
for path in sorted(copied):
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

total_bytes = sum(path.stat().st_size for path in OUT_DIR.rglob("*") if path.is_file())
print(
    "PSG DELTA DEPLOYMENT BUNDLE",
    {
        "path": str(OUT_DIR),
        "files": len(manifest) + 1,
        "size_mb": total_bytes / 1024**2,
        "dose": FROZEN_DOSE,
        "reference": FROZEN_REFERENCE,
    },
    flush=True,
)
print("Do not zip on the training disk; upload this directory as an AutoDL/Kaggle dataset.")
