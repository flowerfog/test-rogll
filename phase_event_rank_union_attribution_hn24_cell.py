"""HN24: read-only equal-budget rank-union attribution for HN23.

HN23 showed that phase/event scoring is weaker than amplitude scoring alone.
This audit asks the only remaining useful question: does the frozen phase arm
rescue amplitude misses?  No GR scoring, training, or configuration selection
is repeated here.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/oof_exports")
HN23_DIR = ROOT / "multiscale_phase_event_observability_hn23"
RANK_PATH = HN23_DIR / "sampled_candidate_ranks.parquet"
VERDICT_PATH = HN23_DIR / "verdict.json"
OUT_DIR = ROOT / "phase_event_rank_union_attribution_hn24"
OUT_DIR.mkdir(parents=True, exist_ok=True)
started = time.time()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


for required in (RANK_PATH, VERDICT_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)

hn23 = json.loads(VERDICT_PATH.read_text(encoding="utf-8"))
selected_config = str(hn23["selected"]["config"])
selected_window = int(hn23["selected"]["half_window"])
if selected_config == "amplitude":
    raise RuntimeError("HN23 did not freeze a distinct phase arm")

records = pd.read_parquet(
    RANK_PATH,
    columns=["control", "well", "fold", "row_idx", "half_window", "config", "rank"],
)
records = records.loc[
    records["half_window"].eq(selected_window)
    & records["config"].isin(["amplitude", selected_config])
].copy()
if records.empty:
    raise RuntimeError("No selected-window HN23 ranks")


def _paired(control):
    local = records.loc[records["control"].eq(control)].copy()
    duplicate = local.duplicated(["well", "row_idx", "config"])
    if duplicate.any():
        raise RuntimeError(f"Duplicate rank keys for {control}")
    wide = local.pivot(
        index=["well", "row_idx"], columns="config", values="rank"
    ).dropna(subset=["amplitude", selected_config])
    wide = wide.rename(columns={"amplitude": "amplitude_rank", selected_config: "phase_rank"})
    if len(wide) < 8000:
        raise RuntimeError(f"Only {len(wide)} paired rows for {control}")
    return wide.reset_index()


def _metrics(control):
    frame = _paired(control)
    amplitude_rank = frame["amplitude_rank"].to_numpy(np.int64)
    phase_rank = frame["phase_rank"].to_numpy(np.int64)
    result = {
        "control": control,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "selected_phase": selected_config,
    }
    for budget in (8, 16, 32):
        half = budget // 2
        amplitude = amplitude_rank < budget
        phase = phase_rank < budget
        equal_union = (amplitude_rank < half) | (phase_rank < half)
        result[f"amplitude_top{budget}"] = float(np.mean(amplitude))
        result[f"phase_top{budget}"] = float(np.mean(phase))
        result[f"union_top{half}_each_budget{budget}"] = float(np.mean(equal_union))
        result[f"equal_budget_gain_{budget}"] = float(np.mean(equal_union) - np.mean(amplitude))
        result[f"phase_rescue_top{half}"] = float(
            np.mean((amplitude_rank >= half) & (phase_rank < half))
        )
        result[f"top{half}_overlap"] = float(
            np.mean((amplitude_rank < half) & (phase_rank < half))
        )
    result["rank_spearman"] = float(
        pd.Series(amplitude_rank).corr(pd.Series(phase_rank), method="spearman")
    )
    return result


aligned = _metrics("aligned")
shuffled = _metrics("typewell_shuffle")

aligned_gain = float(aligned["equal_budget_gain_32"])
shuffle_gain = float(shuffled["equal_budget_gain_32"])
aligned_rescue = float(aligned["phase_rescue_top16"])
shuffle_rescue = float(shuffled["phase_rescue_top16"])
checks = {
    "aligned_rows_ge_9000": int(aligned["rows"]) >= 9000,
    "equal_budget32_gain_ge_0p02": aligned_gain >= 0.02,
    "aligned_gain_exceeds_shuffle_by_0p015": aligned_gain - shuffle_gain >= 0.015,
    "aligned_rescue_exceeds_shuffle_by_0p02": aligned_rescue - shuffle_rescue >= 0.02,
    "rank_spearman_le_0p85": float(aligned["rank_spearman"]) <= 0.85,
}
strict_pass = bool(all(checks.values()))

summary = pd.DataFrame([aligned, shuffled])
summary.to_csv(OUT_DIR / "summary.csv", index=False)
verdict = {
    "experiment": "phase_event_rank_union_attribution_hn24",
    "stage": "frozen_hn23_equal_budget_rank_union_attribution",
    "selection_or_tuning_count": 0,
    "source": str(RANK_PATH),
    "selected": {"config": selected_config, "half_window": selected_window},
    "comparison": "amplitude top32 versus amplitude top16 union phase top16",
    "aligned": aligned,
    "typewell_shuffle": shuffled,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> retain phase only as a candidate-menu expansion signal"
        if strict_pass
        else "FAIL -> phase does not add equal-budget candidate recall; close HN23/HN24"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8"
)
print("HN24 SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nHN24 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
