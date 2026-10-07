"""Audit the fixed absolute-S row axis required by the RankSDF-CP design.

Run this as one AutoDL notebook cell before rebuilding any RankSDF panels.  The
v1 implementation used a column-dependent PF centre, so a horizontal image row
did not represent one geological S value.  This audit measures whether a fixed

    S_j = S_at_prediction_start + offset_j

axis is practical for the existing official and pseudo prediction starts.  It
does not build image tensors and should therefore use very little disk space.
"""

from pathlib import Path
import json
import shutil
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
BANK_DIR = OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1"
EXTENSION_DIR = OOF_ROOT / "multicut_fleongg_matched_k5_extension_v1"
OUT_DIR = OOF_ROOT / "ranksdf_cp_v2_absolute_s_axis_audit"

TAGS = ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4")
HALF_WIDTHS_FT = (96.0, 120.0, 160.0, 240.0, 320.0)
STATE_STEP_FT = 1.0
TARGET_MD_STEP_FT = 16.0
MAX_COLUMNS = 768
N_CHANNELS = 24

# A fixed axis must retain essentially all row labels.  State-level p10 catches
# a small group of wells that a pooled statistic can otherwise hide.
PASS_POOLED_EVAL_COVERAGE = 0.995
PASS_STATE_P10_EVAL_COVERAGE = 0.95
PASS_POOLED_LABEL_TYPEWELL_VALID = 0.995


def _weighted_mean(frame, value, weight):
    values = frame[value].to_numpy(float)
    weights = frame[weight].to_numpy(float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _axis_typewell_valid_fraction(anchor_s, z, tw_min, tw_max, half_width):
    """Mean valid state fraction without materialising an H x W array."""
    offsets = np.arange(-half_width, half_width + 0.5 * STATE_STEP_FT, STATE_STEP_FT)
    lower_offset = tw_min + np.asarray(z, float) - anchor_s
    upper_offset = tw_max + np.asarray(z, float) - anchor_s
    left = np.searchsorted(offsets, lower_offset, side="left")
    right = np.searchsorted(offsets, upper_offset, side="right")
    return float(np.mean(np.clip(right - left, 0, len(offsets)) / len(offsets)))


for path in (DATA_ROOT / "train", BANK_DIR / "cut_manifest.csv", BANK_DIR / "well_metadata.csv"):
    if not path.exists():
        raise FileNotFoundError(path)
OUT_DIR.mkdir(parents=True, exist_ok=True)

manifests = [pd.read_csv(BANK_DIR / "cut_manifest.csv", dtype={"well": str})]
extension_manifest = EXTENSION_DIR / "new_cut_manifest.csv"
if extension_manifest.exists():
    manifests.append(pd.read_csv(extension_manifest, dtype={"well": str}))
manifest = (
    pd.concat(manifests, ignore_index=True)
    .drop_duplicates(["well", "tag"], keep="last")
)
manifest = manifest[manifest["tag"].isin(TAGS)].copy()
meta = pd.read_csv(BANK_DIR / "well_metadata.csv", dtype={"well": str})
fold_map = meta.set_index("well")["fold"].astype(int).to_dict()
manifest["fold"] = manifest["well"].map(fold_map)
if "actual_frac" not in manifest:
    manifest["actual_frac"] = manifest["cut_position"] / manifest["n_rows"]

started = time.time()
state_rows = []
coverage_rows = []
failures = []

for well, states in tqdm(manifest.groupby("well", sort=False), desc="absolute-S axis audit"):
    try:
        horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
        horizontal = pd.read_csv(horizontal_path, usecols=["MD", "Z", "TVT"])
        typewell = pd.read_csv(typewell_path, usecols=["TVT"])

        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
        z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
        tvt = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(float)
        tw_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(float)
        if not (np.isfinite(md).all() and np.isfinite(z).all() and np.isfinite(tvt).all()):
            raise ValueError("non-finite horizontal coordinates or truth")
        tw_tvt = tw_tvt[np.isfinite(tw_tvt)]
        if len(tw_tvt) < 2:
            raise ValueError("insufficient finite typewell TVT")
        tw_min, tw_max = float(tw_tvt.min()), float(tw_tvt.max())
        true_s = tvt + z
        n_columns = int(np.ceil((md[-1] - md[0]) / TARGET_MD_STEP_FT)) + 1

        for state in states.itertuples(index=False):
            cut = int(np.clip(int(state.cut_position), 32, len(horizontal) - 32))
            anchor = cut - 1
            anchor_s = float(true_s[anchor])
            known_delta = np.abs(true_s[:cut] - anchor_s)
            eval_delta = np.abs(true_s[cut:] - anchor_s)
            state_rows.append(
                {
                    "well": str(well),
                    "tag": str(state.tag),
                    "fold": int(state.fold),
                    "actual_frac": float(state.actual_frac),
                    "rows": len(horizontal),
                    "known_rows": cut,
                    "eval_rows": len(horizontal) - cut,
                    "anchor_s": anchor_s,
                    "known_abs_s_p95": float(np.quantile(known_delta, 0.95)),
                    "known_abs_s_max": float(known_delta.max()),
                    "eval_abs_s_p50": float(np.quantile(eval_delta, 0.50)),
                    "eval_abs_s_p95": float(np.quantile(eval_delta, 0.95)),
                    "eval_abs_s_max": float(eval_delta.max()),
                    "n_columns": n_columns,
                }
            )

            for half_width in HALF_WIDTHS_FT:
                known_coverage = float(np.mean(known_delta <= half_width))
                eval_coverage = float(np.mean(eval_delta <= half_width))
                typewell_all = _axis_typewell_valid_fraction(
                    anchor_s, z, tw_min, tw_max, half_width
                )
                typewell_eval = _axis_typewell_valid_fraction(
                    anchor_s, z[cut:], tw_min, tw_max, half_width
                )
                states_count = int(round(2.0 * half_width / STATE_STEP_FT)) + 1
                raw_panel_gb = N_CHANNELS * states_count * MAX_COLUMNS * 2 / 2**30
                coverage_rows.append(
                    {
                        "well": str(well),
                        "tag": str(state.tag),
                        "fold": int(state.fold),
                        "half_width_ft": float(half_width),
                        "states": states_count,
                        "known_rows": cut,
                        "eval_rows": len(horizontal) - cut,
                        "known_coverage": known_coverage,
                        "eval_coverage": eval_coverage,
                        "typewell_valid_all": typewell_all,
                        "typewell_valid_eval": typewell_eval,
                        "label_typewell_valid_eval": float(
                            np.mean((tvt[cut:] >= tw_min) & (tvt[cut:] <= tw_max))
                        ),
                        "raw_panel_gb": raw_panel_gb,
                    }
                )
    except Exception as exc:
        failures.append({"well": str(well), "reason": repr(exc)})

state_report = pd.DataFrame(state_rows)
coverage = pd.DataFrame(coverage_rows)
failure_report = pd.DataFrame(failures)
state_report.to_csv(OUT_DIR / "state_drift_report.csv", index=False)
coverage.to_csv(OUT_DIR / "axis_coverage_by_state.csv", index=False)
failure_report.to_csv(OUT_DIR / "failures.csv", index=False)
if failures:
    raise RuntimeError(f"failed wells: {len(failures)}; see {OUT_DIR / 'failures.csv'}")
if len(state_report) != len(manifest):
    raise RuntimeError(f"state mismatch: built {len(state_report)} / expected {len(manifest)}")

summary_rows = []
subsets = {
    "all": np.ones(len(coverage), dtype=bool),
    "official": coverage["tag"].eq("official").to_numpy(),
    "pseudo": ~coverage["tag"].eq("official").to_numpy(),
}
for subset, subset_mask in subsets.items():
    for half_width in HALF_WIDTHS_FT:
        frame = coverage[subset_mask & coverage["half_width_ft"].eq(half_width)].copy()
        summary_rows.append(
            {
                "subset": subset,
                "half_width_ft": half_width,
                "states": int(frame["states"].iloc[0]),
                "examples": len(frame),
                "wells": frame["well"].nunique(),
                "pooled_known_coverage": _weighted_mean(frame, "known_coverage", "known_rows"),
                "pooled_eval_coverage": _weighted_mean(frame, "eval_coverage", "eval_rows"),
                "state_p10_eval_coverage": float(frame["eval_coverage"].quantile(0.10)),
                "state_min_eval_coverage": float(frame["eval_coverage"].min()),
                "pooled_typewell_valid_all": _weighted_mean(frame, "typewell_valid_all", "known_rows"),
                "pooled_typewell_valid_eval": _weighted_mean(frame, "typewell_valid_eval", "eval_rows"),
                "pooled_label_typewell_valid_eval": _weighted_mean(
                    frame, "label_typewell_valid_eval", "eval_rows"
                ),
                "raw_cache_gb": float(frame["raw_panel_gb"].sum()),
            }
        )
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "axis_coverage_summary.csv", index=False)

all_summary = summary[summary["subset"].eq("all")].copy()
all_summary["coverage_gate"] = (
    (all_summary["pooled_eval_coverage"] >= PASS_POOLED_EVAL_COVERAGE)
    & (all_summary["state_p10_eval_coverage"] >= PASS_STATE_P10_EVAL_COVERAGE)
    & (
        all_summary["pooled_label_typewell_valid_eval"]
        >= PASS_POOLED_LABEL_TYPEWELL_VALID
    )
)
passing = all_summary[all_summary["coverage_gate"]]
recommended = None if passing.empty else float(passing.iloc[0]["half_width_ft"])

drift_summary = (
    state_report.groupby("tag", as_index=False)
    .agg(
        states=("well", "size"),
        wells=("well", "nunique"),
        eval_abs_s_p50=("eval_abs_s_p50", "median"),
        eval_abs_s_p95_median=("eval_abs_s_p95", "median"),
        eval_abs_s_p95_p90=("eval_abs_s_p95", lambda x: float(np.quantile(x, 0.90))),
        eval_abs_s_max_p95=("eval_abs_s_max", lambda x: float(np.quantile(x, 0.95))),
        max_columns=("n_columns", "max"),
    )
)
drift_summary.to_csv(OUT_DIR / "drift_summary_by_tag.csv", index=False)

verdict = {
    "experiment": "ranksdf_cp_v2_absolute_s_axis_audit",
    "axis_definition": "S_j = (TVT + Z) at the last known row + fixed offset_j",
    "column_axis_invariance_ft": 0.0,
    "states": len(state_report),
    "wells": int(state_report["well"].nunique()),
    "half_widths_ft": list(HALF_WIDTHS_FT),
    "recommended_half_width_ft": recommended,
    "max_columns": int(state_report["n_columns"].max()),
    "free_disk_gb": shutil.disk_usage(OOF_ROOT).free / 2**30,
    "gates": {
        "pooled_eval_coverage_min": PASS_POOLED_EVAL_COVERAGE,
        "state_p10_eval_coverage_min": PASS_STATE_P10_EVAL_COVERAGE,
        "pooled_label_typewell_valid_eval_min": PASS_POOLED_LABEL_TYPEWELL_VALID,
        "max_columns": MAX_COLUMNS,
    },
    "strict_pass": bool(recommended is not None and state_report["n_columns"].max() <= MAX_COLUMNS),
    "next": (
        f"PASS -> build fold-1 one-seed fixed absolute-S panels at +/-{recommended:.0f} ft"
        if recommended is not None and state_report["n_columns"].max() <= MAX_COLUMNS
        else "FAIL -> fixed anchor-S panel is not viable at the audited widths"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("\nDRIFT BY TAG")
print(drift_summary.to_string(index=False))
print("\nAXIS COVERAGE")
print(summary.to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
