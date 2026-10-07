"""M342-Y1b: read-only repair of the Y1 discovery safety constraint.

Y1 required activation between 10% and 40% in its final checks but omitted that
constraint while selecting the discovery candidate. Y1b applies the declared
constraint to the already-saved discovery rows, freezes the resulting feature
and threshold, and evaluates the already-saved confirmation rows once.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
Y1_DIR = ROOT / "oof_exports" / "main342_prefix_match_reliability_y1"
SOURCE = Y1_DIR / "well_reliability.csv"
OUT = ROOT / "oof_exports" / "main342_prefix_match_reliability_y1b"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = (
    "prefix_top8_fraction",
    "prefix_top32_fraction",
    "prefix_reciprocal_rank",
    "prefix_zero_margin",
)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _pooled(local, active):
    active = np.asarray(active, bool)
    rows = int(local["rows"].sum())
    far_rows = int(local["far_rows"].sum())
    base_sse = float(local["base_sse"].sum())
    candidate_sse = float(
        np.where(active, local["candidate_sse"], local["base_sse"]).sum()
    )
    base_far_sse = float(local["base_far_sse"].sum())
    candidate_far_sse = float(
        np.where(active, local["candidate_far_sse"], local["base_far_sse"]).sum()
    )
    base_rmse = float(np.sqrt(base_sse / rows))
    rmse = float(np.sqrt(candidate_sse / rows))
    base_far_rmse = float(np.sqrt(base_far_sse / far_rows))
    far_rmse = float(np.sqrt(candidate_far_sse / far_rows))
    return {
        "wells": int(len(local)),
        "rows": rows,
        "active_wells": int(active.sum()),
        "active_fraction": float(active.mean()),
        "base_rmse": base_rmse,
        "rmse": rmse,
        "gain": base_rmse - rmse,
        "base_far_rmse": base_far_rmse,
        "far_rmse": far_rmse,
        "far_gain": base_far_rmse - far_rmse,
        "positive_active_fraction": (
            float(np.mean(local.loc[active, "well_gain"].to_numpy(np.float64) > 0.0))
            if active.any() else 0.0
        ),
    }


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
detail = pd.read_csv(SOURCE)
discovery = detail.loc[detail["stage"].eq("discovery")].copy()
confirmation = detail.loc[detail["stage"].eq("confirmation")].copy()
if len(discovery) < 20 or len(confirmation) < 120:
    raise RuntimeError(
        f"Incomplete Y1 artifact: discovery={len(discovery)} confirmation={len(confirmation)}"
    )

scan_rows = []
for feature in FEATURES:
    threshold = float(discovery[feature].quantile(0.75))
    active = discovery[feature].to_numpy(np.float64) >= threshold
    report = _pooled(discovery, active)
    scan_rows.append({
        "feature": feature,
        "threshold": threshold,
        "spearman_gain": float(
            discovery[feature].corr(discovery["well_gain"], method="spearman")
        ),
        **report,
        "selection_safe": bool(
            0.10 <= report["active_fraction"] <= 0.40
            and report["far_gain"] >= 0.0
            and report["positive_active_fraction"] >= 0.55
        ),
    })
scan = pd.DataFrame(scan_rows).sort_values(
    ["selection_safe", "gain", "spearman_gain"],
    ascending=[False, False, False],
)
safe = scan.loc[scan["selection_safe"]]
if safe.empty:
    selected = None
    confirmation_report = None
    confirmation_spearman = None
else:
    selected = safe.iloc[0]
    feature = str(selected["feature"])
    threshold = float(selected["threshold"])
    active = confirmation[feature].to_numpy(np.float64) >= threshold
    confirmation_report = _pooled(confirmation, active)
    confirmation_spearman = float(
        confirmation[feature].corr(confirmation["well_gain"], method="spearman")
    )
    confirmation_report["spearman_gain"] = confirmation_spearman

checks = {
    "discovery_has_safe_candidate": selected is not None,
    "confirmation_wells_ge_120": confirmation_report is not None and confirmation_report["wells"] >= 120,
    "active_fraction_between_0p10_and_0p40": (
        confirmation_report is not None
        and 0.10 <= confirmation_report["active_fraction"] <= 0.40
    ),
    "confirmation_gain_ge_0p03": (
        confirmation_report is not None and confirmation_report["gain"] >= 0.03
    ),
    "confirmation_far_gain_ge_0": (
        confirmation_report is not None and confirmation_report["far_gain"] >= 0.0
    ),
    "positive_active_fraction_ge_0p55": (
        confirmation_report is not None
        and confirmation_report["positive_active_fraction"] >= 0.55
    ),
    "confirmation_spearman_ge_0p15": (
        confirmation_spearman is not None and confirmation_spearman >= 0.15
    ),
}
strict_pass = bool(all(checks.values()))

verdict = {
    "experiment": "main342_prefix_match_reliability_y1b",
    "stage": "read_only_repair_of_y1_discovery_activation_constraint",
    "selection_or_tuning_count": 0,
    "source": str(SOURCE),
    "repair": (
        "apply Y1's declared 10%-40% activation, nonnegative far gain, and "
        "55% active-well win-rate constraints before discovery selection"
    ),
    "selected": None if selected is None else selected.to_dict(),
    "confirmation": confirmation_report,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> run matched typewell-shuffle attribution"
        if strict_pass else
        "FAIL -> prefix reliability cannot safely gate X3; close Y1/Y1b"
    ),
}

scan.to_csv(OUT / "corrected_discovery_scan.csv", index=False)
(OUT / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2), encoding="utf-8"
)
print("\nM342-Y1b CORRECTED DISCOVERY")
print(scan.to_string(index=False), flush=True)
print("\nM342-Y1b FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
