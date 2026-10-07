"""V212: read-only lineage audit of the deployed mainline booster package."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import joblib
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OUT = ROOT / "oof_exports" / "mainline_model_package_lineage_audit_v212"
OUT.mkdir(parents=True, exist_ok=True)


def _find_models() -> Path:
    candidates = []
    for root in (
        ROOT / "rogii-claude-models-pub",
        ROOT,
        Path("/kaggle/input"),
        Path.cwd(),
    ):
        if root.is_dir():
            candidates.extend(root.rglob("features.json"))
    matches = [path.parent for path in candidates if list(path.parent.glob("lgb*.pkl"))]
    if not matches:
        raise FileNotFoundError("Cannot locate features.json with lgb*.pkl")
    matches = sorted(set(path.resolve() for path in matches), key=lambda path: ("rogii-claude-models-pub" not in str(path), str(path)))
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold_token(name: str) -> int | None:
    patterns = (
        r"(?:^|[_-])fold[_-]?([0-5])(?:[_\-.]|$)",
        r"(?:^|[_-])f([0-5])(?:[_\-.]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, name.lower())
        if match:
            return int(match.group(1))
    return None


models_dir = _find_models()
feature_names = json.loads((models_dir / "features.json").read_text(encoding="utf-8"))
rows = []
for path in sorted(models_dir.glob("lgb*.pkl")):
    model = joblib.load(path)
    params = model.get_params() if hasattr(model, "get_params") else {}
    booster = getattr(model, "booster_", None)
    rows.append(
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "class": type(model).__name__,
            "fold_token": _fold_token(path.name),
            "n_features": int(getattr(model, "n_features_in_", -1)),
            "best_iteration": int(getattr(model, "best_iteration_", -1) or -1),
            "num_trees": int(booster.num_trees()) if booster is not None else -1,
            "random_state": str(params.get("random_state", params.get("seed"))),
            "objective": str(params.get("objective")),
        }
    )

inventory = pd.DataFrame(rows)
fold_tokens = inventory["fold_token"].dropna().astype(int) if not inventory.empty else pd.Series(dtype=int)
unique_tokens = sorted(fold_tokens.unique().tolist())
token_counts = fold_tokens.value_counts().sort_index().to_dict()
mapping_by_filename = bool(
    len(inventory) > 0
    and inventory["fold_token"].notna().all()
    and len(unique_tokens) == 5
    and len(set(token_counts.values())) == 1
)

sidecars = []
for path in sorted(models_dir.iterdir()):
    if path.is_file() and path.suffix.lower() in {".json", ".csv", ".txt", ".md"}:
        sidecars.append({"file": path.name, "bytes": path.stat().st_size})

if len(inventory) == 3 and not mapping_by_filename:
    package_kind = "three_deployment_boosters_without_outer_fold_mapping"
elif mapping_by_filename:
    package_kind = "fivefold_models_recoverable_from_filenames"
else:
    package_kind = "ambiguous_model_package"

verdict = {
    "experiment": "mainline_model_package_lineage_audit_v212",
    "stage": "read_only_deployment_booster_fold_lineage_audit",
    "models_dir": str(models_dir),
    "models": int(len(inventory)),
    "features": int(len(feature_names)),
    "package_kind": package_kind,
    "fold_tokens": unique_tokens,
    "fold_token_counts": {str(key): int(value) for key, value in token_counts.items()},
    "recoverable_fold_mapping": int(mapping_by_filename),
    "sidecars": sidecars,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "Recover held-out predictions with the matching fold models"
        if mapping_by_filename
        else "Retrain an honest outer fivefold mainline; deployed boosters cannot produce OOF"
    ),
}
inventory.to_csv(OUT / "models.csv", index=False)
(OUT / "result.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("V212 MODEL INVENTORY")
print(inventory.to_string(index=False), flush=True)
print("\nV212 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT, flush=True)
