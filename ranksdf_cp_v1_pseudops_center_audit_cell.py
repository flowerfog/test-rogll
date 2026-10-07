"""Audit target-free pseudo-PS centres before RankSDF real-well training.

The matched K5 fleongg cache regenerated PF features independently at each
pseudo prediction start. This cell verifies that pf_ancc can centre the fixed
RankSDF +/-96 ft candidate grid without using the official-mask strong curve.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
BANK_DIR = OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1"
EXTENSION_DIR = OOF_ROOT / "multicut_fleongg_matched_k5_extension_v1"
FEATURE_DIR = BANK_DIR / "features"
OUT_DIR = OOF_ROOT / "ranksdf_cp_v1_pseudops_center_audit"

TAGS = ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4")
PRIMARY_CENTER = "pf_ancc"
STATE_HALF_WIDTH_FT = 96.0
TRAIN_KNOWN_FRAC_MIN = 0.10
TRAIN_KNOWN_FRAC_MAX = 0.40

PASS_OVERALL_COVERAGE = 0.995
PASS_MIN_TAG_COVERAGE = 0.990
PASS_COMPLETE_WELLS = 700

ABSOLUTE_CANDIDATES = (
    "pf_ancc",
    "pf_z",
    "likpf_scale_3",
    "likpf_scale_5",
    "likpf_scale_8",
    "likpf_scale_12",
    "likpf_mean",
)


def _sample_path(well, tag):
    return FEATURE_DIR / f"{well}__{tag}__sample.pkl"


started = time.time()
required = [
    BANK_DIR / "well_metadata.csv",
    BANK_DIR / "cut_manifest.csv",
    FEATURE_DIR,
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Matched-K5 cache is unavailable. Restore/rebuild it first: " + repr(missing)
    )

meta = pd.read_csv(BANK_DIR / "well_metadata.csv", dtype={"well": str})
fold_map = meta.set_index("well")["fold"].astype(int).to_dict()
manifests = [pd.read_csv(BANK_DIR / "cut_manifest.csv", dtype={"well": str})]
extension_manifest = EXTENSION_DIR / "new_cut_manifest.csv"
if extension_manifest.exists():
    manifests.append(pd.read_csv(extension_manifest, dtype={"well": str}))
manifest = pd.concat(manifests, ignore_index=True)
manifest = manifest.drop_duplicates(["well", "tag"], keep="last")
manifest = manifest[manifest["tag"].isin(TAGS)].copy()
if "actual_frac" not in manifest:
    manifest["actual_frac"] = manifest["cut_position"] / manifest["n_rows"]

complete = []
for well in meta["well"].astype(str):
    if all(_sample_path(well, tag).exists() for tag in TAGS):
        complete.append(well)
if len(complete) < PASS_COMPLETE_WELLS:
    raise RuntimeError(f"Only {len(complete)} wells have all K5 centre files")
manifest = manifest[manifest["well"].isin(complete)].copy()

print(
    {
        "experiment": "ranksdf_cp_v1_pseudops_center_audit",
        "bank": str(BANK_DIR),
        "complete_wells": len(complete),
        "states": len(manifest),
        "primary_center": PRIMARY_CENTER,
        "candidate_half_width_ft": STATE_HALF_WIDTH_FT,
    }
)
print("CUT DISTRIBUTION")
print(
    manifest.groupby("tag")["actual_frac"].describe()[
        ["count", "mean", "min", "25%", "50%", "75%", "max"]
    ].to_string()
)

rows = []
for record in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="pseudo-PS centre coverage"):
    path = _sample_path(str(record.well), str(record.tag))
    frame = pd.read_pickle(path)
    required_columns = {"target", "last_known_tvt", PRIMARY_CENTER}
    if not required_columns.issubset(frame.columns):
        raise RuntimeError(f"{path} missing {sorted(required_columns - set(frame.columns))}")
    truth = frame["target"].to_numpy(float) + frame["last_known_tvt"].to_numpy(float)
    row = {
        "well": str(record.well),
        "tag": str(record.tag),
        "fold": int(fold_map[str(record.well)]),
        "actual_frac": float(record.actual_frac),
        "rows": len(frame),
        "in_pdf_train_range": bool(
            TRAIN_KNOWN_FRAC_MIN <= float(record.actual_frac) <= TRAIN_KNOWN_FRAC_MAX
        ),
    }
    for candidate in ABSOLUTE_CANDIDATES:
        if candidate not in frame:
            continue
        prediction = frame[candidate].to_numpy(float)
        finite = np.isfinite(truth) & np.isfinite(prediction)
        error = prediction[finite] - truth[finite]
        row[f"{candidate}_rows"] = int(finite.sum())
        row[f"{candidate}_sse"] = float(np.sum(error ** 2))
        row[f"{candidate}_covered"] = int(np.sum(np.abs(error) <= STATE_HALF_WIDTH_FT))
    rows.append(row)

detail = pd.DataFrame(rows)
candidate_rows = []
for candidate in ABSOLUTE_CANDIDATES:
    row_column = f"{candidate}_rows"
    if row_column not in detail:
        continue
    n = int(detail[row_column].sum())
    tag_coverages = []
    for _, group in detail.groupby("tag", sort=False):
        tag_coverages.append(
            group[f"{candidate}_covered"].sum()
            / max(group[row_column].sum(), 1)
        )
    candidate_rows.append(
        {
            "candidate": candidate,
            "rows": n,
            "rmse": float(np.sqrt(detail[f"{candidate}_sse"].sum() / max(n, 1))),
            "coverage": float(detail[f"{candidate}_covered"].sum() / max(n, 1)),
            "min_tag_coverage": float(np.min(tag_coverages)),
            "pdf_range_coverage": float(
                detail.loc[detail["in_pdf_train_range"], f"{candidate}_covered"].sum()
                / max(detail.loc[detail["in_pdf_train_range"], row_column].sum(), 1)
            ),
        }
    )
candidate_report = pd.DataFrame(candidate_rows).sort_values(
    ["coverage", "rmse"], ascending=[False, True]
)

tag_rows = []
for tag, group in detail.groupby("tag", sort=False):
    n = int(group[f"{PRIMARY_CENTER}_rows"].sum())
    tag_rows.append(
        {
            "tag": tag,
            "states": len(group),
            "rows": n,
            "known_frac_mean": float(group["actual_frac"].mean()),
            "known_frac_min": float(group["actual_frac"].min()),
            "known_frac_max": float(group["actual_frac"].max()),
            "rmse": float(np.sqrt(group[f"{PRIMARY_CENTER}_sse"].sum() / max(n, 1))),
            "coverage": float(group[f"{PRIMARY_CENTER}_covered"].sum() / max(n, 1)),
        }
    )
tag_report = pd.DataFrame(tag_rows)
primary = candidate_report[candidate_report["candidate"] == PRIMARY_CENTER].iloc[0]
strict_pass = bool(
    len(complete) >= PASS_COMPLETE_WELLS
    and float(primary["coverage"]) >= PASS_OVERALL_COVERAGE
    and float(primary["min_tag_coverage"]) >= PASS_MIN_TAG_COVERAGE
)
verdict = {
    "experiment": "ranksdf_cp_v1_pseudops_center_audit",
    "bank": str(BANK_DIR),
    "feature_dir": str(FEATURE_DIR),
    "complete_wells": len(complete),
    "complete_states": int(len(detail)),
    "states_in_pdf_known_frac_range": int(detail["in_pdf_train_range"].sum()),
    "primary_center": PRIMARY_CENTER,
    "primary_rmse": float(primary["rmse"]),
    "primary_coverage": float(primary["coverage"]),
    "primary_min_tag_coverage": float(primary["min_tag_coverage"]),
    "primary_pdf_range_coverage": float(primary["pdf_range_coverage"]),
    "gates": {
        "complete_wells_min": PASS_COMPLETE_WELLS,
        "overall_coverage_min": PASS_OVERALL_COVERAGE,
        "min_tag_coverage_min": PASS_MIN_TAG_COVERAGE,
    },
    "strict_pass": strict_pass,
    "next": (
        "PASS -> build three-seed one-fold RankSDF training from K5 centres"
        if strict_pass
        else "FAIL -> select/rebuild a fold-safe pseudo-PS centre before training"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
detail.to_csv(OUT_DIR / "state_coverage.csv", index=False)
candidate_report.to_csv(OUT_DIR / "candidate_centres.csv", index=False)
tag_report.to_csv(OUT_DIR / "primary_by_tag.csv", index=False)
with (OUT_DIR / "verdict.json").open("w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2, sort_keys=True)

print("\nCANDIDATE CENTRES")
print(candidate_report.to_string(index=False))
print("\nPRIMARY BY TAG")
print(tag_report.to_string(index=False))
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
