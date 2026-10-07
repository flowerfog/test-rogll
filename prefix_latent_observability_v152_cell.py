"""V152: analysis-only audit of visible-prefix observability."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "prefix_latent_observability_v152"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"
BASES = (
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet",
    OOF_ROOT / "frontier_6p39_full_oof_cp_increment_v18" / "full_oof_cp_endpoints.parquet",
)
BASE = next((path for path in BASES if path.is_file()), None)
FOLD_PATH = OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
if BASE is None or not FOLD_PATH.is_file():
    raise FileNotFoundError((BASES, FOLD_PATH))

frame = pd.read_parquet(BASE)
BASE_COLUMN = "incumbent_6p377_replay" if "incumbent_6p377_replay" in frame.columns else "full_cp8_ordered"
required = {"well", "row_idx", "fold", "tvt_true", "md_since", BASE_COLUMN}
missing = sorted(required.difference(frame.columns))
if missing:
    raise RuntimeError(f"Missing OOF columns: {missing}")
frame = frame[list(required)].copy()
frame["well"] = frame["well"].astype(str)
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
folds = pd.read_csv(FOLD_PATH, dtype={"well": str})
folds["well"] = folds["well"].astype(str)
fold_map = dict(zip(folds["well"], folds["fold"].astype(int)))


def _slope(x, y):
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[keep], np.asarray(y)[keep]
    if len(x) < 8 or np.ptp(x) < 32.0:
        return np.nan
    return float(np.polyfit(x - x[-1], y, 1)[0])


def _one_well(well, local):
    path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    raw = pd.read_csv(path)
    required_raw = {"MD", "TVT_input", "GR", "X", "Y", "Z"}
    if not required_raw.issubset(raw.columns):
        raise RuntimeError(f"{well}: missing raw prefix columns")
    raw = raw[list(required_raw)].apply(pd.to_numeric, errors="coerce")
    md = raw["MD"].to_numpy(float)
    tvt = raw["TVT_input"].to_numpy(float)
    known = np.flatnonzero(np.isfinite(md) & np.isfinite(tvt))
    if len(known) < 32:
        raise RuntimeError(f"{well}: too few visible TVT rows")
    anchor = float(md[known[-1]])
    out = {"well": well, "fold": int(fold_map[well]), "visible_rows": int(len(known))}
    for width in (128, 256, 512, 1024):
        keep = np.isfinite(md) & np.isfinite(tvt) & (md >= anchor - width) & (md <= anchor)
        out[f"prefix_slope_{width}"] = _slope(md[keep], tvt[keep])
        out[f"prefix_gr_slope_{width}"] = _slope(md[keep], raw["GR"].to_numpy(float)[keep])
        out[f"prefix_z_slope_{width}"] = _slope(md[keep], raw["Z"].to_numpy(float)[keep])
    future = local[local["md_since"] > 1000.0].sort_values("row_idx", kind="stable")
    distance = future["md_since"].to_numpy(float)
    residual = future["tvt_true"].to_numpy(float) - future[BASE_COLUMN].to_numpy(float)
    out["future_residual_mean"] = float(np.mean(residual))
    out["future_residual_slope"] = _slope(distance, residual)
    far = distance > 2000.0
    out["future_far_residual_mean"] = float(np.mean(residual[far])) if far.any() else np.nan
    out["future_far_residual_slope"] = _slope(distance[far], residual[far]) if far.any() else np.nan
    return out


rows, failures = [], []
for number, (well, local) in enumerate(frame.groupby("well", sort=True), 1):
    try:
        rows.append(_one_well(well, local))
    except Exception as exc:
        failures.append({"well": well, "reason": repr(exc)})
    if number % 100 == 0 or number == frame["well"].nunique():
        print(f"V152 audit {number}/{frame['well'].nunique()} failures={len(failures)}", flush=True)
audit = pd.DataFrame(rows)
audit.to_csv(OUT_DIR / "audit_by_well.csv", index=False)
if failures:
    pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)


def _assoc(part, feature, target):
    valid = part[[feature, target]].dropna()
    if len(valid) < 12:
        return {"n": int(len(valid)), "pearson": np.nan, "spearman": np.nan, "sign_accuracy": np.nan}
    x, y = valid[feature].to_numpy(float), valid[target].to_numpy(float)
    return {
        "n": int(len(valid)),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(pd.Series(x).rank().corr(pd.Series(y).rank())),
        "sign_accuracy": float(np.mean(np.sign(x) == np.sign(y))),
    }


targets = ("future_residual_mean", "future_residual_slope", "future_far_residual_mean", "future_far_residual_slope")
features = [f"prefix_slope_{width}" for width in (128, 256, 512, 1024)]
association_rows = []
for fold in (0, 1, 2, 3, 4, 5):
    part = audit if fold == 0 else audit[audit["fold"].eq(fold)]
    split = "all" if fold == 0 else f"fold{fold}"
    for feature in features:
        for target in targets:
            association_rows.append({"split": split, "feature": feature, "target": target, **_assoc(part, feature, target)})
associations = pd.DataFrame(association_rows)
associations.to_csv(OUT_DIR / "associations.csv", index=False)
all_assoc = associations[(associations["split"] == "all") & (associations["target"] == "future_residual_mean")].copy()
best = all_assoc.sort_values("spearman", key=lambda s: s.abs(), ascending=False).iloc[0]
verdict = {
    "experiment": "prefix_latent_observability_v152",
    "base": str(BASE),
    "wells": int(len(audit)),
    "failures": int(len(failures)),
    "best_prefix_signal": best.to_dict(),
    "gates": {
        "coverage_exact": len(audit) == 773,
        "best_abs_spearman_ge_0p20": bool(abs(float(best["spearman"])) >= 0.20),
        "best_sign_accuracy_ge_0p60": bool(float(best["sign_accuracy"]) >= 0.60),
    },
    "next": "PASS -> build online latent calibration" if abs(float(best["spearman"])) >= 0.20 and float(best["sign_accuracy"]) >= 0.60 else "FAIL -> prefix state cannot identify future residual direction",
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, allow_nan=False), encoding="utf-8")
print("V152 ASSOCIATIONS TOP")
print(all_assoc.sort_values("spearman", key=lambda s: s.abs(), ascending=False).head(20).to_string(index=False))
print("V152 FINAL VERDICT")
print(json.dumps(verdict, indent=2, allow_nan=False))
