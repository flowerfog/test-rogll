"""M342-X1: multiscale raw-GR rank-consistency observability gateway.

For each station, candidate offsets are scored with raw robust NCC at 64, 128
and 256-row half windows. Aggregate scores use only candidates valid at every
scale. Fold 3 selects one fixed rank/z-score aggregation; Fold 2 confirms it
against a frozen best-single-scale baseline and a matched-range typewell
shuffle. No model or target-derived feature is trained.
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
HN23_SOURCE = ROOT / "multiscale_phase_event_observability_hn23_cell.py"
OUT = ROOT / "oof_exports" / "main342_multiscale_rank_consistency_x1"
OUT.mkdir(parents=True, exist_ok=True)

DISCOVERY_FOLD = 3
DISCOVERY_WELLS = 24
CONFIRMATION_FOLD = 2
UNTOUCHED_FOLD = 1
ROWS_PER_WELL = 40
HALF_WINDOWS = (64, 128, 256)
MIN_WINDOW_COVERAGE = 0.35


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


def _load_hn23():
    source = HN23_SOURCE.read_text(encoding="utf-8")
    marker = "v66 = pd.read_parquet(V66_PATH)"
    if marker not in source:
        raise RuntimeError("HN23 execution marker not found")
    name = "m342_x1_hn23_definitions"
    module = types.ModuleType(name)
    module.__file__ = str(HN23_SOURCE)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(HN23_SOURCE), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


H = _load_hn23()
DATA_ROOT = H["DATA_ROOT"]
OFFSETS = H["OFFSETS"].astype(np.float64)
v66 = pd.read_parquet(H["V66_PATH"])
prediction_column = H["_prediction_column"](v66)
if "well" not in v66:
    v66["well"] = v66["id"].astype(str).str.rsplit("_", n=1).str[0]
if "row_idx" not in v66:
    v66["row_idx"] = v66["id"].astype(str).str.rsplit("_", n=1).str[1].astype(int)
v66 = v66[["well", "row_idx", "fold", "tvt_true", prediction_column]].rename(
    columns={prediction_column: "base"}
)
v66["well"] = v66["well"].astype(str).str.lower()
fold_by_well = v66.groupby("well")["fold"].first().astype(int).to_dict()


def _robust_z(values, valid):
    output = np.full(len(values), -np.inf, np.float64)
    local = np.asarray(values, np.float64)[valid]
    centre = float(np.median(local))
    scale = float(np.median(np.abs(local - centre)) * 1.4826)
    if scale < 1e-6:
        scale = max(float(np.std(local)), 1e-6)
    output[valid] = (local - centre) / scale
    return output


def _rank_score(values, valid):
    output = np.full(len(values), -np.inf, np.float64)
    indices = np.flatnonzero(valid)
    order = np.argsort(-np.asarray(values)[indices], kind="stable")
    rank = np.empty(len(indices), np.float64)
    rank[order] = np.arange(len(indices), dtype=np.float64)
    output[indices] = -rank / max(len(indices) - 1, 1)
    return output


def _aggregate(scale_scores, common_valid):
    z = {scale: _robust_z(score, common_valid) for scale, score in scale_scores.items()}
    rank = {scale: _rank_score(score, common_valid) for scale, score in scale_scores.items()}
    output = {f"single_{scale}": np.asarray(scale_scores[scale], np.float64) for scale in HALF_WINDOWS}
    output.update({
        "zmean_all": np.mean(np.vstack([z[scale] for scale in HALF_WINDOWS]), axis=0),
        "zmedian_all": np.median(np.vstack([z[scale] for scale in HALF_WINDOWS]), axis=0),
        "zmean_long": 0.35 * z[128] + 0.65 * z[256],
        "rankmean_all": np.mean(np.vstack([rank[scale] for scale in HALF_WINDOWS]), axis=0),
        "rankmedian_all": np.median(np.vstack([rank[scale] for scale in HALF_WINDOWS]), axis=0),
        "rankmean_long": 0.35 * rank[128] + 0.65 * rank[256],
        "rank_worstcase": np.min(np.vstack([rank[scale] for scale in HALF_WINDOWS]), axis=0),
    })
    for score in output.values():
        score[~common_valid] = -np.inf
    return output


def _evaluate(wells, control, shuffle_map=None):
    records, failures = [], []
    for number, well in enumerate(wells, 1):
        local = v66.loc[v66["well"].eq(well)].sort_values("row_idx")
        before = len(records)
        try:
            override = shuffle_map.get(well) if shuffle_map else None
            base, hgr, truth, tw_tvt, tw_gr = H["_load_well"](well, local, override)
            positions = H["_sample_positions"](local["row_idx"].to_numpy(np.int64), ROWS_PER_WELL)
            for centre in positions:
                if not np.isfinite(truth[centre]):
                    continue
                truth_offset = float(truth[centre] - base[centre])
                if truth_offset < OFFSETS[0] or truth_offset > OFFSETS[-1]:
                    continue
                scale_scores, scale_valid = {}, {}
                accepted = True
                for half_window in HALF_WINDOWS:
                    left = max(0, int(centre) - half_window)
                    right = min(len(base), int(centre) + half_window + 1)
                    horizontal = hgr[left:right]
                    if len(horizontal) < half_window or np.isfinite(horizontal).mean() < MIN_WINDOW_COVERAGE:
                        accepted = False
                        break
                    horizontal = H["_fill"](horizontal)
                    if horizontal is None:
                        accepted = False
                        break
                    candidate_tvt = base[left:right][None, :] + OFFSETS[:, None]
                    candidate_gr = np.interp(
                        candidate_tvt.ravel(), tw_tvt, tw_gr, left=np.nan, right=np.nan
                    ).reshape(len(OFFSETS), -1)
                    valid = np.isfinite(candidate_gr).mean(axis=1) >= 0.95
                    if valid.sum() < 64:
                        accepted = False
                        break
                    candidate_gr = np.where(
                        np.isfinite(candidate_gr), candidate_gr,
                        np.nanmedian(candidate_gr, axis=1, keepdims=True),
                    )
                    component = H["_components"](horizontal, candidate_gr)["amp"]
                    scale_scores[half_window] = np.asarray(component, np.float64)
                    scale_valid[half_window] = valid
                if not accepted:
                    continue
                common_valid = np.logical_and.reduce([scale_valid[scale] for scale in HALF_WINDOWS])
                if common_valid.sum() < 64:
                    continue
                for config, score in _aggregate(scale_scores, common_valid).items():
                    rank, winner, _ = H["_rank"](score, truth_offset)
                    records.append({
                        "control": control, "well": well,
                        "fold": int(local["fold"].iloc[0]), "row_idx": int(centre),
                        "config": config, "truth_offset": truth_offset,
                        "winner_offset": float(OFFSETS[winner]), "rank": rank,
                    })
        except Exception as exc:
            failures.append({"well": well, "control": control, "reason": repr(exc)})
        if len(records) == before and not any(item["well"] == well for item in failures):
            failures.append({"well": well, "control": control, "reason": "no_common_scale_rows"})
        if number % 25 == 0 or number == len(wells):
            print(f"M342-X1 {control} {number}/{len(wells)} rows={len(records)} failures={len(failures)}", flush=True)
    return pd.DataFrame(records), failures


def _summary(records):
    rows = []
    for keys, local in records.groupby(["control", "config"], sort=False):
        error = local["winner_offset"].to_numpy(np.float64) - local["truth_offset"].to_numpy(np.float64)
        rows.append({
            "control": keys[0], "config": keys[1], "rows": len(local),
            "wells": local["well"].nunique(), "median_rank": float(local["rank"].median()),
            "top8_recall": float(local["rank"].lt(8).mean()),
            "top32_recall": float(local["rank"].lt(32).mean()),
            "argmax_rmse": float(np.sqrt(np.mean(error * error))),
        })
    return pd.DataFrame(rows)


fold3_wells = sorted(well for well, fold in fold_by_well.items() if fold == DISCOVERY_FOLD)
discovery_wells = fold3_wells[:DISCOVERY_WELLS]
confirmation_wells = sorted(well for well, fold in fold_by_well.items() if fold == CONFIRMATION_FOLD)
discovery, failures_a = _evaluate(discovery_wells, "aligned")
discovery_summary = _summary(discovery)
single_discovery = discovery_summary[discovery_summary["config"].str.startswith("single_")].sort_values(
    ["top32_recall", "top8_recall", "argmax_rmse"], ascending=[False, False, True]
)
aggregate_discovery = discovery_summary[~discovery_summary["config"].str.startswith("single_")].sort_values(
    ["top32_recall", "top8_recall", "argmax_rmse"], ascending=[False, False, True]
)
if single_discovery.empty or aggregate_discovery.empty:
    raise RuntimeError("Incomplete X1 discovery")
baseline_config = str(single_discovery.iloc[0]["config"])
selected_config = str(aggregate_discovery.iloc[0]["config"])

aligned_records, failures_b = _evaluate(confirmation_wells, "aligned")
centres = {}
for well in confirmation_wells:
    table = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv", usecols=["TVT"])
    centres[well] = float(pd.to_numeric(table["TVT"], errors="coerce").median())
ordered = sorted(confirmation_wells, key=lambda well: (centres[well], well))
shuffle_map = dict(zip(ordered, ordered[1:] + ordered[:1]))
shuffled_records, failures_c = _evaluate(confirmation_wells, "typewell_shuffle", shuffle_map)
confirmation_summary = _summary(pd.concat([aligned_records, shuffled_records], ignore_index=True))


def _row(control, config):
    local = confirmation_summary[
        confirmation_summary["control"].eq(control) & confirmation_summary["config"].eq(config)
    ]
    if len(local) != 1:
        raise RuntimeError(f"Missing X1 row: {control}/{config}")
    return local.iloc[0]


baseline = _row("aligned", baseline_config)
selected = _row("aligned", selected_config)
shuffle_baseline = _row("typewell_shuffle", baseline_config)
shuffle_selected = _row("typewell_shuffle", selected_config)
gain32 = float(selected["top32_recall"] - baseline["top32_recall"])
gain8 = float(selected["top8_recall"] - baseline["top8_recall"])
shuffle_gain32 = float(shuffle_selected["top32_recall"] - shuffle_baseline["top32_recall"])
argmax_gain = float(baseline["argmax_rmse"] - selected["argmax_rmse"])
checks = {
    "confirmation_wells_ge_120": int(selected["wells"]) >= 120,
    "top32_gain_ge_0p03": gain32 >= 0.03,
    "top8_gain_ge_0p015": gain8 >= 0.015,
    "aligned_increment_exceeds_shuffle_by_0p015": gain32 - shuffle_gain32 >= 0.015,
    "argmax_rmse_improves_ge_2ft": argmax_gain >= 2.0,
}
strict_pass = bool(all(checks.values()))
verdict = {
    "experiment": "main342_multiscale_rank_consistency_x1",
    "stage": "fold3_multiscale_rank_selection_fold2_observability_confirmation",
    "model_training": False,
    "protocol": {
        "discovery_fold": DISCOVERY_FOLD, "discovery_wells": len(discovery_wells),
        "confirmation_fold": CONFIRMATION_FOLD, "untouched_fold": UNTOUCHED_FOLD,
        "half_windows": HALF_WINDOWS,
        "common_candidate_support_required": True,
    },
    "selected": {"baseline": baseline_config, "aggregate": selected_config},
    "confirmation": {
        "baseline": baseline.to_dict(), "aggregate": selected.to_dict(),
        "shuffle_baseline": shuffle_baseline.to_dict(),
        "shuffle_aggregate": shuffle_selected.to_dict(),
        "top32_gain": gain32, "top8_gain": gain8,
        "shuffle_top32_gain": shuffle_gain32, "argmax_rmse_gain": argmax_gain,
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "failures": {"discovery": len(failures_a), "aligned": len(failures_b), "shuffle": len(failures_c)},
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> add frozen multiscale rank consistency to candidate path generation"
        if strict_pass else
        "FAIL -> multiscale consistency does not resolve aliases; close X1"
    ),
}

discovery_summary.to_csv(OUT / "discovery_summary.csv", index=False)
confirmation_summary.to_csv(OUT / "confirmation_summary.csv", index=False)
(OUT / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-X1 DISCOVERY")
print(discovery_summary.to_string(index=False), flush=True)
print("\nM342-X1 CONFIRMATION")
print(confirmation_summary.to_string(index=False), flush=True)
print("\nM342-X1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT, flush=True)
