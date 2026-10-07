"""M342-Y1: can visible-prefix match quality gate long-window corrections?

The known TVT prefix supplies a legal, same-well calibration target. For each
well, Y1 measures where the true zero offset ranks under 256-row GR matching,
then asks whether that reliability predicts the frozen X3 hidden-path gain.
The first 24 Fold3 wells select one reliability statistic and its upper-quartile
threshold; the remaining Fold3 wells provide a single confirmation.
"""

from pathlib import Path
import json
import sys
import time
import types

import numpy as np
import pandas as pd


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
X3_SOURCE = ROOT / "main342_longwindow_path_residual_x3.py"
OUT = ROOT / "oof_exports" / "main342_prefix_match_reliability_y1"
OUT.mkdir(parents=True, exist_ok=True)

FOLD = 3
DISCOVERY_WELLS = 24
HALF_WINDOW = 256
PREFIX_STATIONS = 32
X3_TRANSITION_SCALE_FT = 16.0
X3_DOSE = 0.05
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


def _bootstrap_x3():
    source = X3_SOURCE.read_text(encoding="utf-8")
    marker = "fold_wells = sorted("
    if marker not in source:
        raise RuntimeError("X3 execution marker not found")
    name = "m342_y1_x3_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(X3_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(X3_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


V = _bootstrap_x3()
frame = V["frame"]
OFFSETS = V["OFFSETS"]
ZERO_STATE = int(np.argmin(np.abs(OFFSETS)))
DATA_ROOT = V["DATA_ROOT"]
METRICS = V["METRICS"]
H = V["X"]["H"]


def _prefix_reliability(well):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
    tvt = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    hgr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    tw_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64)
    tw_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64)
    good = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    if good.sum() < 64:
        raise RuntimeError(f"{well}: too few typewell rows")
    order = np.argsort(tw_tvt[good])
    tw_tvt, tw_gr = tw_tvt[good][order], tw_gr[good][order]

    known = np.flatnonzero(np.isfinite(tvt))
    if len(known) < 2 * HALF_WINDOW + 1:
        raise RuntimeError(f"{well}: prefix shorter than one complete 256 window")
    eligible = known[
        (known >= known.min() + HALF_WINDOW)
        & (known <= known.max() - HALF_WINDOW)
    ]
    positions = H["_sample_positions"](eligible, PREFIX_STATIONS)
    ranks, reciprocal, margins = [], [], []
    for centre in positions:
        left, right = int(centre) - HALF_WINDOW, int(centre) + HALF_WINDOW + 1
        observed = hgr[left:right]
        local_tvt = tvt[left:right]
        if (
            np.isfinite(observed).mean() < 0.35
            or not np.isfinite(local_tvt).all()
        ):
            continue
        observed = H["_fill"](observed)
        if observed is None:
            continue
        candidate_tvt = local_tvt[None, :] + OFFSETS[:, None]
        candidate_gr = np.interp(
            candidate_tvt.ravel(), tw_tvt, tw_gr, left=np.nan, right=np.nan
        ).reshape(len(OFFSETS), -1)
        valid = np.isfinite(candidate_gr).mean(axis=1) >= 0.95
        if not valid[ZERO_STATE] or valid.sum() < 64:
            continue
        medians = np.nanmedian(candidate_gr, axis=1, keepdims=True)
        candidate_gr = np.where(np.isfinite(candidate_gr), candidate_gr, medians)
        score = np.asarray(H["_components"](observed, candidate_gr)["amp"], np.float64)
        score[~valid] = -np.inf
        order_states = np.argsort(-score, kind="stable")
        rank = int(np.flatnonzero(order_states == ZERO_STATE)[0])
        ranks.append(rank)
        reciprocal.append(1.0 / (rank + 1.0))
        margins.append(float(score[ZERO_STATE] - np.max(np.delete(score, ZERO_STATE))))
    if len(ranks) < 8:
        raise RuntimeError(f"{well}: only {len(ranks)} valid prefix stations")
    ranks = np.asarray(ranks, np.int64)
    return {
        "prefix_stations": len(ranks),
        "prefix_top8_fraction": float(np.mean(ranks < 8)),
        "prefix_top32_fraction": float(np.mean(ranks < 32)),
        "prefix_reciprocal_rank": float(np.mean(reciprocal)),
        "prefix_zero_margin": float(np.mean(margins)),
    }


def _evaluate_wells(wells, label):
    rows, failures = [], []
    for number, well in enumerate(wells, 1):
        local = frame.loc[frame["well"].eq(well)].sort_values("row_idx").copy()
        try:
            reliability = _prefix_reliability(well)
            positions, emission = V["_station_emissions"](local)
            offsets = V["_decode"](emission, X3_TRANSITION_SCALE_FT)
            correction = V["_full_correction"](local, positions, offsets)
            parent = local["k2_parent"].to_numpy(np.float64)
            truth = local["tvt_true"].to_numpy(np.float64)
            candidate = parent + X3_DOSE * correction
            base_error = parent - truth
            candidate_error = candidate - truth
            far = local["md_since"].to_numpy(np.float64) > 2000.0
            rows.append({
                "stage": label,
                "well": well,
                "rows": len(local),
                "base_sse": float(np.sum(np.square(base_error))),
                "candidate_sse": float(np.sum(np.square(candidate_error))),
                "base_far_sse": float(np.sum(np.square(base_error[far]))),
                "candidate_far_sse": float(np.sum(np.square(candidate_error[far]))),
                "far_rows": int(far.sum()),
                "well_gain": float(
                    np.sqrt(np.mean(np.square(base_error)))
                    - np.sqrt(np.mean(np.square(candidate_error)))
                ),
                **reliability,
            })
        except Exception as exc:
            failures.append({"stage": label, "well": well, "reason": repr(exc)})
        if number % 25 == 0 or number == len(wells):
            print(
                f"M342-Y1 {label} {number}/{len(wells)} rows={len(rows)} "
                f"failures={len(failures)}", flush=True
            )
    return pd.DataFrame(rows), failures


def _pooled(local, active):
    active = np.asarray(active, bool)
    rows = int(local["rows"].sum())
    far_rows = int(local["far_rows"].sum())
    base_sse = float(local["base_sse"].sum())
    selected_sse = float(np.where(active, local["candidate_sse"], local["base_sse"]).sum())
    base_far = float(local["base_far_sse"].sum())
    selected_far = float(
        np.where(active, local["candidate_far_sse"], local["base_far_sse"]).sum()
    )
    return {
        "wells": int(len(local)),
        "rows": rows,
        "active_wells": int(active.sum()),
        "active_fraction": float(active.mean()),
        "base_rmse": float(np.sqrt(base_sse / rows)),
        "rmse": float(np.sqrt(selected_sse / rows)),
        "gain": float(np.sqrt(base_sse / rows) - np.sqrt(selected_sse / rows)),
        "base_far_rmse": float(np.sqrt(base_far / far_rows)),
        "far_rmse": float(np.sqrt(selected_far / far_rows)),
        "far_gain": float(np.sqrt(base_far / far_rows) - np.sqrt(selected_far / far_rows)),
        "positive_active_fraction": float(
            np.mean(local.loc[active, "well_gain"].to_numpy() > 0.0)
        ) if active.any() else 0.0,
    }


fold_wells = sorted(frame.loc[frame["fold"].eq(FOLD), "well"].unique())
discovery_wells = fold_wells[:DISCOVERY_WELLS]
confirmation_wells = fold_wells[DISCOVERY_WELLS:]
discovery, failures_a = _evaluate_wells(discovery_wells, "discovery")
if len(discovery) < 20:
    raise RuntimeError(f"Only {len(discovery)} Y1 discovery wells")

feature_rows = []
for feature in FEATURES:
    correlation = float(discovery[feature].corr(discovery["well_gain"], method="spearman"))
    threshold = float(discovery[feature].quantile(0.75))
    report = _pooled(discovery, discovery[feature].to_numpy() >= threshold)
    feature_rows.append({
        "feature": feature,
        "spearman_gain": correlation,
        "threshold": threshold,
        **report,
    })
feature_scan = pd.DataFrame(feature_rows).sort_values(
    ["gain", "spearman_gain"], ascending=[False, False]
)
selected = feature_scan.iloc[0]
selected_feature = str(selected["feature"])
threshold = float(selected["threshold"])

confirmation, failures_b = _evaluate_wells(confirmation_wells, "confirmation")
active = confirmation[selected_feature].to_numpy(np.float64) >= threshold
report = _pooled(confirmation, active)
confirmation_correlation = float(
    confirmation[selected_feature].corr(confirmation["well_gain"], method="spearman")
)
report["spearman_gain"] = confirmation_correlation

checks = {
    "discovery_wells_ge_20": len(discovery) >= 20,
    "confirmation_wells_ge_120": len(confirmation) >= 120,
    "active_fraction_between_0p10_and_0p40": 0.10 <= report["active_fraction"] <= 0.40,
    "confirmation_gain_ge_0p03": report["gain"] >= 0.03,
    "confirmation_far_gain_ge_0": report["far_gain"] >= 0.0,
    "positive_active_fraction_ge_0p55": report["positive_active_fraction"] >= 0.55,
    "confirmation_spearman_ge_0p15": confirmation_correlation >= 0.15,
}
strict_pass = bool(all(checks.values()))

verdict = {
    "experiment": "main342_prefix_match_reliability_y1",
    "stage": "visible_prefix_match_reliability_gateway_for_frozen_x3",
    "legal_inputs_only": True,
    "known_prefix_truth_used_only_for_reliability": True,
    "protocol": {
        "selection_wells": len(discovery_wells),
        "confirmation_wells_requested": len(confirmation_wells),
        "x3_transition_scale_ft": X3_TRANSITION_SCALE_FT,
        "x3_dose": X3_DOSE,
        "fold2_touched": False,
        "fold1_touched": False,
    },
    "selected": selected.to_dict(),
    "confirmation": report,
    "checks": checks,
    "strict_pass": strict_pass,
    "failures": {"discovery": len(failures_a), "confirmation": len(failures_b)},
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> add matched typewell-shuffle attribution before Fold2"
        if strict_pass else
        "FAIL -> visible-prefix match quality cannot gate hidden GR corrections; close Y1"
    ),
}

feature_scan.to_csv(OUT / "discovery_feature_scan.csv", index=False)
pd.concat([discovery, confirmation], ignore_index=True).to_csv(
    OUT / "well_reliability.csv", index=False
)
pd.DataFrame(failures_a + failures_b).to_csv(OUT / "failures.csv", index=False)
(OUT / "verdict.json").write_text(
    json.dumps(_clean(verdict), indent=2), encoding="utf-8"
)
print("\nM342-Y1 DISCOVERY")
print(feature_scan.to_string(index=False), flush=True)
print("\nM342-Y1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
