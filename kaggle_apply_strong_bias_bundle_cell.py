"""Apply the rebuilt Huber bias bundle to an explicitly frozen strong curve.

Before running this cell, the old SP45/fleongg notebook must have produced the
projected 0.55/0.45 strong curve, saved as submission_strong.csv. Do not point
this cell at a CP12, Gold, or strict-spatial submission.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


STRONG_SUBMISSION_OVERRIDE = None
BUNDLE_OVERRIDE = None
WRITE_AS_FINAL_SUBMISSION = True


def _find_data_root():
    candidates = []
    cfg = globals().get("CFG")
    if cfg is not None:
        for attr in ("DATA", "dataset_path"):
            value = getattr(cfg, attr, None)
            if value:
                candidates.append(Path(value))
    candidates.extend(
        [
            Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
            Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        ]
    )
    for root in candidates:
        if (root / "train").is_dir() and (root / "test").is_dir():
            return root
    raise FileNotFoundError("Cannot locate competition data root")


def _find_bundle():
    if BUNDLE_OVERRIDE:
        path = Path(BUNDLE_OVERRIDE)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    matches = list(Path("/kaggle/input").rglob("strong_bias_huber_bundle_v1.joblib"))
    if not matches:
        raise FileNotFoundError(
            "Upload strong_bias_huber_bundle_v1.joblib or set BUNDLE_OVERRIDE"
        )
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _safe_stats(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_p10": float(np.quantile(values, 0.10)),
        f"{prefix}_p50": float(np.quantile(values, 0.50)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
    }


def _robust_slope(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 8 or np.max(x) <= np.min(x):
        return 0.0
    x = x - x[-1]
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def _build_well_features(wells, data_root):
    rows = []
    for index, wid in enumerate(wells, 1):
        try:
            hw = pd.read_csv(data_root / "test" / f"{wid}__horizontal_well.csv")
            tw = pd.read_csv(data_root / "test" / f"{wid}__typewell.csv")
            tvt_input = hw["TVT_input"].to_numpy(float)
            known = np.flatnonzero(np.isfinite(tvt_input))
            evaluation = np.flatnonzero(~np.isfinite(tvt_input))
            row = {"well": str(wid)}
            if len(known) == 0 or len(evaluation) == 0:
                rows.append(row)
                continue

            before_eval = known[known < evaluation[0]]
            last_index = int(before_eval[-1]) if len(before_eval) else int(known[-1])
            last_tvt = float(tvt_input[last_index])
            md = hw["MD"].to_numpy(float)
            z = hw["Z"].to_numpy(float)
            gr = hw["GR"].to_numpy(float) if "GR" in hw else np.full(len(hw), np.nan)

            row["known_len"] = float(len(known))
            row["eval_len"] = float(len(evaluation))
            row["last_tvt"] = last_tvt
            row["last_z"] = float(z[last_index])
            row["last_u"] = float(last_tvt + z[last_index])
            row["eval_md_len"] = float(md[evaluation[-1]] - md[evaluation[0]])
            row["eval_md_since"] = float(md[evaluation[-1]] - md[last_index])
            row["eval_z_delta"] = float(z[evaluation[-1]] - z[last_index])
            row["eval_z_span"] = float(np.nanmax(z[evaluation]) - np.nanmin(z[evaluation]))
            row["eval_gr_cov"] = float(np.isfinite(gr[evaluation]).mean())
            row["known_gr_cov"] = float(np.isfinite(gr[known]).mean())

            tail50 = known[-min(50, len(known)) :]
            tail200 = known[-min(200, len(known)) :]
            row["slope_tvt_tail50"] = _robust_slope(md[tail50], tvt_input[tail50])
            row["slope_tvt_tail200"] = _robust_slope(md[tail200], tvt_input[tail200])
            row["slope_u_tail50"] = _robust_slope(md[tail50], tvt_input[tail50] + z[tail50])
            row["slope_u_tail200"] = _robust_slope(md[tail200], tvt_input[tail200] + z[tail200])
            row["slope_z_eval"] = _robust_slope(md[evaluation], z[evaluation])

            for values, prefix in (
                (gr[known], "known_gr"),
                (gr[tail200], "tail200_gr"),
                (gr[evaluation], "eval_gr"),
                (z[evaluation] - z[last_index], "eval_dz"),
            ):
                row.update(_safe_stats(values, prefix))

            if {"TVT", "GR"}.issubset(tw.columns):
                tw_clean = tw[["TVT", "GR"]].dropna().sort_values("TVT")
                if len(tw_clean):
                    tw_tvt = tw_clean["TVT"].to_numpy(float)
                    tw_gr = tw_clean["GR"].to_numpy(float)
                    tw_range = max(float(np.ptp(tw_tvt)), 1e-6)
                    row["tw_tvt_min"] = float(np.min(tw_tvt))
                    row["tw_tvt_max"] = float(np.max(tw_tvt))
                    row["tw_tvt_range"] = float(np.ptp(tw_tvt))
                    row["last_tvt_rel_tw"] = float((last_tvt - np.min(tw_tvt)) / tw_range)
                    for offset in (-120, -60, -30, 0, 30, 60, 120):
                        row[f"tw_gr_at_{offset}"] = float(np.interp(last_tvt + offset, tw_tvt, tw_gr))
            rows.append(row)
        except Exception as exc:
            print("bias feature fallback", wid, repr(exc))
            rows.append({"well": str(wid)})
        if index % 50 == 0:
            print(f"test bias features {index}/{len(wells)}")
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)


WORK = Path("/kaggle/working")
DATA_ROOT = _find_data_root()
BUNDLE_PATH = _find_bundle()
STRONG_PATH = (
    Path(STRONG_SUBMISSION_OVERRIDE)
    if STRONG_SUBMISSION_OVERRIDE
    else WORK / "submission_strong.csv"
)
if not STRONG_PATH.exists():
    raise FileNotFoundError(
        "submission_strong.csv is missing. Freeze it immediately after the old "
        "SP45/fleongg 0.55/0.45 final-projection cell; do not use submission.csv blindly."
    )

bundle = joblib.load(BUNDLE_PATH)
if bundle.get("version") != "strong_bias_huber_bundle_v1":
    raise RuntimeError(f"Wrong bundle version: {bundle.get('version')}")
oof_summary = bundle.get("oof_summary", {})
if not (
    oof_summary.get("strict_pass", False)
    or oof_summary.get("performance_pass", False)
):
    raise RuntimeError("Bias bundle failed its standalone OOF performance gate")
if not oof_summary.get("lineage_match", False):
    print(
        "WARNING: validated rebuilt Huber variant, not a row-exact reproduction "
        "of historical pool.strong_bias"
    )

strong = pd.read_csv(STRONG_PATH, usecols=["id", "tvt"])
strong["id"] = strong["id"].astype(str)
if strong["id"].duplicated().any() or not np.isfinite(strong["tvt"]).all():
    raise RuntimeError("Invalid strong submission")
strong["well"] = strong["id"].str.rsplit("_", n=1).str[0]
wells = strong["well"].drop_duplicates().tolist()

test_features = _build_well_features(wells, DATA_ROOT)
feature_columns = list(bundle["feature_columns"])
for column in feature_columns:
    if column not in test_features:
        test_features[column] = 0.0

raw_bias = bundle["model"].predict(test_features[feature_columns].to_numpy(float))
move = np.clip(
    float(bundle["bias_alpha"]) * raw_bias,
    -float(bundle["bias_cap"]),
    float(bundle["bias_cap"]),
)
test_features["bias_pred_raw"] = raw_bias
test_features["bias_move"] = move
move_map = dict(zip(test_features["well"], test_features["bias_move"]))

output = strong.copy()
output["bias_move"] = output["well"].map(move_map)
if output["bias_move"].isna().any():
    raise RuntimeError("Missing bias move for one or more test wells")
output["tvt"] = output["tvt"].astype(float) + output["bias_move"].astype(float)

candidate_path = WORK / "submission_strong_bias.csv"
output[["id", "tvt"]].to_csv(candidate_path, index=False)
strong[["id", "tvt"]].to_csv(WORK / "submission_before_strong_bias.csv", index=False)
test_features.to_csv(WORK / "strong_bias_test_well_diagnostics.csv", index=False)
if WRITE_AS_FINAL_SUBMISSION:
    output[["id", "tvt"]].to_csv(WORK / "submission.csv", index=False)

audit = {
    "strong_source": str(STRONG_PATH),
    "bundle": str(BUNDLE_PATH),
    "rows": int(len(output)),
    "wells": int(len(wells)),
    "mean_abs_move": float(np.mean(np.abs(move))),
    "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
    "max_abs_move": float(np.max(np.abs(move))),
    "candidate": str(candidate_path),
}
(WORK / "strong_bias_test_audit.json").write_text(json.dumps(audit, indent=2))
print("STRONG_BIAS COMPLETE")
print(json.dumps(audit, indent=2))
