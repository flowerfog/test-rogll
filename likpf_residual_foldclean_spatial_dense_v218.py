"""V218: 3x training-density confirmation of the frozen V217 representation.

This launcher deliberately reuses the audited V217 implementation.  It changes
only the training-row sampler: every training well contributes up to 3072 rows
from its full cache instead of the frozen 1024-row sample file.  Validation
remains every official hidden row and all feature/target/readout choices stay
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = ROOT / "likpf_residual_foldclean_spatial_v217.py"
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

source = SOURCE.read_text(encoding="utf-8")
old_loader = '''def _load(wells, full):
    return pd.concat([_load_one(str(well), full) for well in wells], ignore_index=True, sort=False)
'''
new_loader = '''def _load(wells, full):
    frames = []
    for well in wells:
        # Validation remains full-row. Training is deterministically sampled
        # from the full cache so this experiment changes density only.
        frame = _load_one(str(well), True)
        if not full and len(frame) > 3072:
            positions = np.unique(
                np.round(np.linspace(0, len(frame) - 1, 3072)).astype(int)
            )
            frame = frame.iloc[positions].copy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)
'''
if source.count(old_loader) != 1:
    raise RuntimeError("V218 could not locate the audited V217 loader")
source = source.replace(old_loader, new_loader, 1)
source = source.replace('SEED = 217', 'SEED = 218', 1)
source = source.replace('likpf_residual_foldclean_spatial_v217', 'likpf_residual_foldclean_spatial_dense_v218')
source = source.replace('V217', 'V218')
source = source.replace('v217', 'v218')

exec(compile(source, str(SOURCE), "exec"), globals())

# Add an exact density-only comparison against the frozen V217 predictions.
oof_root = ROOT / "oof_exports"
v217_path = oof_root / "likpf_residual_foldclean_spatial_v217" / "oof.parquet"
v218_path = oof_root / "likpf_residual_foldclean_spatial_dense_v218" / "oof.parquet"
for path in (v217_path, v218_path):
    if not path.is_file():
        raise FileNotFoundError(path)

v217 = pd.read_parquet(v217_path)[["id", "fold", "tvt_true", "candidate"]].rename(
    columns={"candidate": "v217"}
)
v218 = pd.read_parquet(v218_path)[["id", "candidate"]].rename(columns={"candidate": "v218"})
matched = v217.merge(v218, on="id", how="left", validate="one_to_one")
if len(matched) != 3_783_989 or matched["v218"].isna().any():
    raise RuntimeError("Incomplete V217/V218 alignment")


def _rmse(group, column):
    error = group[column].to_numpy(float) - group["tvt_true"].to_numpy(float)
    return float(np.sqrt(np.mean(error * error)))


rows = []
for fold, group in matched.groupby("fold", sort=True):
    row = {
        "fold": int(fold),
        "v217_rmse": _rmse(group, "v217"),
        "v218_rmse": _rmse(group, "v218"),
    }
    row["density_gain"] = row["v217_rmse"] - row["v218_rmse"]
    rows.append(row)
v217_rmse = _rmse(matched, "v217")
v218_rmse = _rmse(matched, "v218")
audit = {
    "experiment": "likpf_residual_foldclean_spatial_dense_v218_density_audit",
    "changed_variable": "training rows per well 1024 -> min(3072, complete well)",
    "validation": "unchanged complete 3,783,989 OOF rows",
    "v217_rmse": v217_rmse,
    "v218_rmse": v218_rmse,
    "density_gain": v217_rmse - v218_rmse,
    "folds": rows,
    "checks": {
        "density_gain_ge_0p10": v217_rmse - v218_rmse >= 0.10,
        "positive_folds_ge_4": sum(row["density_gain"] > 0 for row in rows) >= 4,
        "minimum_fold_gain_ge_minus0p05": min(row["density_gain"] for row in rows) >= -0.05,
    },
}
audit["strict_pass"] = bool(all(audit["checks"].values()))
out = oof_root / "likpf_residual_foldclean_spatial_dense_v218"
pd.DataFrame(rows).to_csv(out / "density_comparison.csv", index=False)
(out / "density_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

print("\nV218 DENSITY FOLDS")
print(pd.DataFrame(rows).to_string(index=False))
print("\nV218 DENSITY VERDICT")
print(json.dumps(audit, indent=2))
