"""Low-cost G3a prefix-signal audit for revised execution plan V2.

This is a zero-training rescore of the deployment-legal prefix meta-adapter.
The pre-registered primary signal is improvement of the adapter objective on
earlier pseudo-PS support columns.  The action outcome is row-normalised toe
DeltaSSE of the already saved adapted prediction versus the frozen v4 model.

The saved adapted prediction is the optimistic same-fold dose chosen by the
historical audit.  Therefore a failure closes G3a; a pass only authorises a
new frozen T1 experiment and is not itself a promotion result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


STARTED = time.time()
EXPERIMENT = "revised_v2_g3a_prefix_signal_audit_v0"
SEED = 20260718
BOOTSTRAPS = 4000

SPEARMAN_GATE = 0.25
AUC_GATE = 0.60
COVERAGE_GATE = 0.70
WRONG_SIGNAL_ABS_SPEARMAN_MAX = 0.10


def _first_existing(paths):
    return next((Path(path) for path in paths if Path(path).exists()), None)


OOF_ROOT = _first_existing(
    [
        "/root/autodl-tmp/oof_exports",
        "/kaggle/working/oof_exports",
        Path.cwd() / "oof_exports",
    ]
)
if OOF_ROOT is None:
    raise FileNotFoundError("Cannot locate oof_exports")

V4_DIR = OOF_ROOT / "tvt_corrnet_heel_conditioned_onefold_v4"
META_DIR = OOF_ROOT / "tvt_corrnet_prefix_meta_adapter_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

ADAPTER_PATH = META_DIR / "adapter_bywell.csv"
ADAPTED_ROWS_PATH = META_DIR / "best_predictions.pkl"
FROZEN_ROWS_PATH = _first_existing(
    [
        V4_DIR / "fold1_predictions.pkl",
        V4_DIR / "intervention_full.pkl",
    ]
)
SCORES_PATH = META_DIR / "scores.csv"
for path in (ADAPTER_PATH, ADAPTED_ROWS_PATH, FROZEN_ROWS_PATH, SCORES_PATH):
    if path is None or not Path(path).exists():
        raise FileNotFoundError(path)


def _safe_auc(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    valid = np.isfinite(score)
    if valid.sum() < 10 or len(np.unique(y[valid])) < 2:
        return np.nan
    return float(roc_auc_score(y[valid], score[valid]))


def _safe_spearman(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return np.nan
    value = spearmanr(np.asarray(x)[valid], np.asarray(y)[valid]).statistic
    return float(value) if np.isfinite(value) else np.nan


adapter = pd.read_csv(ADAPTER_PATH, dtype={"well": str})
adapted = pd.read_pickle(ADAPTED_ROWS_PATH).copy()
frozen = pd.read_pickle(FROZEN_ROWS_PATH).copy()
for frame in (adapted, frozen):
    frame["well"] = frame["well"].astype(str)
    frame["row_idx"] = frame["row_idx"].astype(np.int32)

required_adapted = {"well", "row_idx", "truth", "prediction"}
required_frozen = {"well", "row_idx", "truth", "prediction"}
if not required_adapted.issubset(adapted.columns):
    raise RuntimeError(f"Adapted rows missing {required_adapted - set(adapted.columns)}")
if not required_frozen.issubset(frozen.columns):
    raise RuntimeError(f"Frozen rows missing {required_frozen - set(frozen.columns)}")

rows = frozen[["well", "row_idx", "truth", "prediction"]].rename(
    columns={"truth": "truth_frozen", "prediction": "frozen_prediction"}
).merge(
    adapted[["well", "row_idx", "truth", "prediction"]].rename(
        columns={"truth": "truth_adapted", "prediction": "adapted_prediction"}
    ),
    on=["well", "row_idx"],
    how="inner",
    validate="one_to_one",
)
if len(rows) != len(frozen) or len(rows) != len(adapted):
    raise RuntimeError(
        f"Row lineage mismatch: merged={len(rows)} frozen={len(frozen)} adapted={len(adapted)}"
    )
truth_difference = float(
    np.max(np.abs(rows["truth_frozen"] - rows["truth_adapted"]))
)
if truth_difference > 1e-4:
    raise RuntimeError(f"Truth mismatch {truth_difference}")

truth = rows["truth_frozen"].to_numpy(np.float64)
frozen_prediction = rows["frozen_prediction"].to_numpy(np.float64)
adapted_prediction = rows["adapted_prediction"].to_numpy(np.float64)
rows["frozen_se"] = (frozen_prediction - truth) ** 2
rows["adapted_se"] = (adapted_prediction - truth) ** 2
rows["frozen_error"] = frozen_prediction - truth
rows["adapted_error"] = adapted_prediction - truth

bywell = (
    rows.groupby("well", as_index=False)
    .agg(
        rows=("row_idx", "size"),
        frozen_sse=("frozen_se", "sum"),
        adapted_sse=("adapted_se", "sum"),
        frozen_bias=("frozen_error", "mean"),
        adapted_bias=("adapted_error", "mean"),
    )
)
bywell["delta_sse"] = bywell["frozen_sse"] - bywell["adapted_sse"]
bywell["delta_mse"] = bywell["delta_sse"] / bywell["rows"]
bywell["frozen_rmse"] = np.sqrt(bywell["frozen_sse"] / bywell["rows"])
bywell["adapted_rmse"] = np.sqrt(bywell["adapted_sse"] / bywell["rows"])
bywell["rmse_gain"] = bywell["frozen_rmse"] - bywell["adapted_rmse"]
bywell["action_win"] = (bywell["delta_sse"] > 0).astype(np.int8)

adapter["inner_loss_improvement"] = adapter["loss_before"] - adapter["loss_after"]
theta_columns = [
    "theta_log_scale",
    "theta_linear",
    "theta_quadratic",
    "theta_progress_linear",
]
adapter["theta_norm"] = np.sqrt(
    np.sum(adapter[theta_columns].fillna(0.0).to_numpy(np.float64) ** 2, axis=1)
)
bywell = bywell.merge(adapter, on="well", how="left", validate="one_to_one")
bywell["eligible"] = (
    bywell["support_panels"].fillna(0).to_numpy(float) > 0
)
eligible = bywell["eligible"].to_numpy(bool)
coverage = float(eligible.mean())

# Primary sign is frozen before looking at toe outcomes: more support loss
# improvement should imply a more useful target-well adapter.
primary_signal = bywell["inner_loss_improvement"].to_numpy(np.float64)
outcome = bywell["delta_mse"].to_numpy(np.float64)
win = bywell["action_win"].to_numpy(np.int8)
primary_spearman = _safe_spearman(primary_signal[eligible], outcome[eligible])
primary_auc = _safe_auc(win[eligible], primary_signal[eligible])

# Deterministic wrong-well control: rotate support evidence over eligible wells
# while leaving each well's toe outcome fixed.
eligible_positions = np.flatnonzero(eligible)
wrong_signal = np.full(len(bywell), np.nan, np.float64)
if len(eligible_positions):
    shift = max(1, len(eligible_positions) // 3)
    wrong_signal[eligible_positions] = np.roll(
        primary_signal[eligible_positions], shift
    )
wrong_spearman = _safe_spearman(wrong_signal[eligible], outcome[eligible])
wrong_auc = _safe_auc(win[eligible], wrong_signal[eligible])


def _bootstrap_metrics(signal, selected):
    signal = np.asarray(signal, float)[selected]
    selected_outcome = outcome[selected]
    selected_win = win[selected]
    rng = np.random.default_rng(SEED)
    records = []
    for _ in range(BOOTSTRAPS):
        sample = rng.integers(0, len(signal), len(signal))
        records.append(
            (
                _safe_spearman(signal[sample], selected_outcome[sample]),
                _safe_auc(selected_win[sample], signal[sample]),
            )
        )
    frame = pd.DataFrame(records, columns=["spearman", "auc"])
    return {
        "spearman_p05": float(frame["spearman"].quantile(0.05)),
        "spearman_p50": float(frame["spearman"].quantile(0.50)),
        "spearman_p95": float(frame["spearman"].quantile(0.95)),
        "auc_p05": float(frame["auc"].quantile(0.05)),
        "auc_p50": float(frame["auc"].quantile(0.50)),
        "auc_p95": float(frame["auc"].quantile(0.95)),
    }, frame


primary_bootstrap, primary_bootstrap_frame = _bootstrap_metrics(
    primary_signal, eligible
)
wrong_bootstrap, wrong_bootstrap_frame = _bootstrap_metrics(wrong_signal, eligible)
primary_bootstrap_frame.assign(control="real_prefix").to_csv(
    OUT_DIR / "bootstrap_real.csv", index=False
)
wrong_bootstrap_frame.assign(control="wrong_well_prefix").to_csv(
    OUT_DIR / "bootstrap_wrong.csv", index=False
)

signal_rows = []
signal_specs = [
    ("inner_loss_improvement", 1.0),
    ("loss_before", 1.0),
    ("theta_norm", 1.0),
    ("theta_log_scale", 1.0),
    ("theta_linear", 1.0),
    ("theta_quadratic", 1.0),
    ("theta_progress_linear", 1.0),
    ("support_columns", 1.0),
]
for column, sign in signal_specs:
    values = sign * bywell[column].to_numpy(np.float64)
    signal_rows.append(
        {
            "signal": column,
            "spearman_delta_mse": _safe_spearman(values[eligible], outcome[eligible]),
            "auc_action_win": _safe_auc(win[eligible], values[eligible]),
        }
    )
signal_report = pd.DataFrame(signal_rows)

scores = pd.read_csv(SCORES_PATH)
score_lookup = scores.set_index("variant")["rmse"].to_dict()
frozen_rmse = float(np.sqrt(np.mean((frozen_prediction - truth) ** 2)))
adapted_rmse = float(np.sqrt(np.mean((adapted_prediction - truth) ** 2)))
shuffled_rmse = float(score_lookup.get("shuffled_adapter", np.nan))

correlation_pass = bool(
    primary_spearman >= SPEARMAN_GATE or primary_auc >= AUC_GATE
)
bootstrap_pass = bool(
    primary_bootstrap["spearman_p05"] > 0.0
    or primary_bootstrap["auc_p05"] > 0.50
)
wrong_control_pass = bool(
    abs(wrong_spearman) <= WRONG_SIGNAL_ABS_SPEARMAN_MAX
    and wrong_bootstrap["spearman_p05"] <= 0.0 <= wrong_bootstrap["spearman_p95"]
)
coverage_pass = coverage >= COVERAGE_GATE
strict_pass = bool(
    correlation_pass and bootstrap_pass and wrong_control_pass and coverage_pass
)

bywell.to_csv(OUT_DIR / "g3a_bywell.csv", index=False)
signal_report.to_csv(OUT_DIR / "signal_scan_diagnostic.csv", index=False)

verdict = {
    "experiment": EXPERIMENT,
    "source": str(META_DIR),
    "valid_wells": int(len(bywell)),
    "eligible_wells": int(eligible.sum()),
    "coverage": coverage,
    "primary_signal": "inner_loss_improvement",
    "primary_spearman": primary_spearman,
    "primary_auc": primary_auc,
    "primary_bootstrap": primary_bootstrap,
    "wrong_well_spearman": wrong_spearman,
    "wrong_well_auc": wrong_auc,
    "wrong_well_bootstrap": wrong_bootstrap,
    "frozen_rmse": frozen_rmse,
    "optimistic_adapted_rmse": adapted_rmse,
    "optimistic_gain": frozen_rmse - adapted_rmse,
    "shuffled_adapter_rmse": shuffled_rmse,
    "gates": {
        "spearman_ge": SPEARMAN_GATE,
        "or_auc_ge": AUC_GATE,
        "coverage_ge": COVERAGE_GATE,
        "wrong_abs_spearman_le": WRONG_SIGNAL_ABS_SPEARMAN_MAX,
        "bootstrap_direction_stable": True,
    },
    "correlation_pass": correlation_pass,
    "bootstrap_pass": bootstrap_pass,
    "wrong_control_pass": wrong_control_pass,
    "coverage_pass": coverage_pass,
    "strict_pass": strict_pass,
    "interpretation": (
        "Optimistic same-fold screen only; a pass authorises frozen T1, not promotion."
    ),
    "next": (
        "PASS G3a -> implement one frozen low-capacity T1 real/wrong-prefix experiment"
        if strict_pass
        else "FAIL G3a -> close online/T1 branch and enter PDF fallback"
    ),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("\nPRIMARY G3a")
print(json.dumps(verdict, indent=2, ensure_ascii=False))
print("\nSIGNAL DIAGNOSTICS")
print(signal_report.to_string(index=False))
print("saved:", OUT_DIR)

