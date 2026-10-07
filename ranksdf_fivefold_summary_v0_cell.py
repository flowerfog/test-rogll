"""Summarise the preregistered RankSDF blend after all five folds finish.

This consumes only each fold's frozen cp12_w100_cap6 prediction. Diagnostic
single-fold grid winners are deliberately ignored.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
FOLDS = (1, 2, 3, 4, 5)
PRIMARY_WEIGHT = 0.10
PRIMARY_CAP_FT = 6.0

PASS_OVERALL_GAIN = 0.10
PASS_MIN_FOLD_GAIN = 0.0
PASS_POSITIVE_FOLDS = 5


def _rmse(y, prediction):
    y = np.asarray(y, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - y) ** 2)))


def _cosine(left, right):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / max(denominator, 1e-12))


fold_rows = []
arrays = {name: [] for name in ("truth", "strong", "cp12", "ranksdf", "primary")}

for fold in FOLDS:
    run_dir = OOF_ROOT / f"ranksdf_onefold_blend_rescore_v0_fold{fold}"
    prediction_path = run_dir / f"fold{fold}_predictions.npz"
    verdict_path = run_dir / "verdict.json"
    for path in (prediction_path, verdict_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing fold {fold} artifact: {path}. Train and rescore that "
                "fold with fixed epoch=10 and cp12_w100_cap6 first."
            )

    with np.load(prediction_path, allow_pickle=False) as data:
        current = {name: data[name].astype(np.float64) for name in arrays}
    with verdict_path.open("r", encoding="utf-8") as handle:
        verdict = json.load(handle)

    if int(verdict.get("valid_fold", -1)) != fold:
        raise RuntimeError(f"Fold lineage mismatch in {verdict_path}")
    primary = verdict.get("primary_variant", {})
    if (
        abs(float(primary.get("weight_ranksdf", -1.0)) - PRIMARY_WEIGHT) > 1e-9
        or abs(float(primary.get("applied_cap_ft", -1.0)) - PRIMARY_CAP_FT) > 1e-9
    ):
        raise RuntimeError(f"Fold {fold} does not contain frozen cp12_w100_cap6")

    truth = current["truth"]
    cp12 = current["cp12"]
    ranksdf = current["ranksdf"]
    prediction = current["primary"]
    move = prediction - cp12
    fold_rows.append(
        {
            "fold": fold,
            "rows": len(truth),
            "ranksdf_rmse": _rmse(truth, ranksdf),
            "cp12_rmse": _rmse(truth, cp12),
            "primary_rmse": _rmse(truth, prediction),
            "gain_vs_cp12": _rmse(truth, cp12) - _rmse(truth, prediction),
            "error_cosine_vs_cp12": _cosine(ranksdf - truth, cp12 - truth),
            "mean_abs_move": float(np.mean(np.abs(move))),
            "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
        }
    )
    for name, values in current.items():
        arrays[name].append(values)

fold_report = pd.DataFrame(fold_rows)
full = {name: np.concatenate(parts) for name, parts in arrays.items()}
truth = full["truth"]
cp12 = full["cp12"]
ranksdf = full["ranksdf"]
primary = full["primary"]
move = primary - cp12

overall = {
    "rows": len(truth),
    "folds": len(FOLDS),
    "ranksdf_rmse": _rmse(truth, ranksdf),
    "cp12_rmse": _rmse(truth, cp12),
    "primary_rmse": _rmse(truth, primary),
    "gain_vs_cp12": _rmse(truth, cp12) - _rmse(truth, primary),
    "mean_fold_gain": float(fold_report["gain_vs_cp12"].mean()),
    "min_fold_gain": float(fold_report["gain_vs_cp12"].min()),
    "positive_folds": int((fold_report["gain_vs_cp12"] > 0).sum()),
    "error_cosine_vs_cp12": _cosine(ranksdf - truth, cp12 - truth),
    "mean_abs_move": float(np.mean(np.abs(move))),
    "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
    "max_abs_move": float(np.max(np.abs(move))),
}
strict_pass = bool(
    overall["gain_vs_cp12"] >= PASS_OVERALL_GAIN
    and overall["min_fold_gain"] >= PASS_MIN_FOLD_GAIN
    and overall["positive_folds"] >= PASS_POSITIVE_FOLDS
)
result = {
    **overall,
    "frozen_recipe": {
        "base": "cp12",
        "weight_ranksdf": PRIMARY_WEIGHT,
        "applied_cap_ft": PRIMARY_CAP_FT,
        "checkpoint_epoch": 10,
    },
    "gates": {
        "overall_gain_min": PASS_OVERALL_GAIN,
        "min_fold_gain": PASS_MIN_FOLD_GAIN,
        "positive_folds": PASS_POSITIVE_FOLDS,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> build full training/deployment bundle and test downstream CP12/Gold order"
        if strict_pass
        else "FAIL -> stop RankSDF v0; do not submit"
    ),
}

out_dir = OOF_ROOT / "ranksdf_fivefold_summary_v0"
out_dir.mkdir(parents=True, exist_ok=True)
fold_report.to_csv(out_dir / "fold_scores.csv", index=False)
with (out_dir / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)

print("FOLDS")
print(fold_report.to_string(index=False))
print("\nOVERALL")
print(json.dumps(result, indent=2))
print("saved:", out_dir)
