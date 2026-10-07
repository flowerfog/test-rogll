"""Apply a validated offset-krige lookup to the current Kaggle submission.

Upload exactly these two files from the promoted AutoDL output as one dataset:
    offset_krige_lookup.csv
    offset_krige_config.json

Run this cell immediately after producing the lineage named by base_lineage in
the config.  Applying a CP12-trained lookup after Gold is a new, unvalidated
lineage; for that ordering, build the lookup from honest final-chain OOF first.
The cell preserves a before-copy and writes both a named candidate and
/kaggle/working/submission.csv.
"""

from pathlib import Path
import glob
import json
import math

import numpy as np
import pandas as pd


WORK = Path("/kaggle/working")
SUBMISSION_PATH = WORK / "submission.csv"
LOOKUP_OVERRIDE = None
CONFIG_OVERRIDE = None


def _find_unique(pattern, override=None):
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    matches = [Path(path) for path in glob.glob(pattern, recursive=True)]
    if not matches:
        raise FileNotFoundError(pattern)
    # Prefer the shallowest path to avoid accidentally taking notebook outputs.
    return sorted(matches, key=lambda path: (len(path.parts), str(path)))[0]


def _transform(x, y, geometry):
    theta = float(geometry["theta"])
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.asarray(
        [
            x * cosine + y * sine,
            (-x * sine + y * cosine) / float(geometry["anisotropy"]),
        ],
        dtype=np.float64,
    )


if not SUBMISSION_PATH.exists():
    raise FileNotFoundError(
        f"Run the intended base/CP/Gold chain first; missing {SUBMISSION_PATH}"
    )

lookup_path = _find_unique(
    "/kaggle/input/**/offset_krige_lookup.csv", LOOKUP_OVERRIDE
)
config_path = _find_unique(
    "/kaggle/input/**/offset_krige_config.json", CONFIG_OVERRIDE
)
with config_path.open(encoding="utf-8") as handle:
    config = json.load(handle)
lookup = pd.read_csv(lookup_path)
required = {"well", "cx", "cy", "b_w"}
if not required.issubset(lookup.columns):
    raise RuntimeError(f"lookup missing {sorted(required - set(lookup.columns))}")
lookup = lookup[["well", "cx", "cy", "b_w"]].copy()
lookup["well"] = lookup["well"].astype(str)
for column in ("cx", "cy", "b_w"):
    lookup[column] = pd.to_numeric(lookup[column], errors="raise")
if lookup["well"].duplicated().any() or not np.isfinite(
    lookup[["cx", "cy", "b_w"]].to_numpy(float)
).all():
    raise RuntimeError("invalid lookup rows")

geometry = config["geometry"]
application = config["application"]
guard = config.get("coverage_guard", {})
print(
    json.dumps(
        {
            "lookup": str(lookup_path),
            "config": str(config_path),
            "base_lineage_required": config.get("base_lineage"),
            "geometry": geometry,
            "application": application,
            "coverage_guard": guard,
        },
        indent=2,
    )
)

submission = pd.read_csv(SUBMISSION_PATH)
if list(submission.columns) != ["id", "tvt"]:
    raise RuntimeError(f"unexpected submission columns: {list(submission.columns)}")
submission["id"] = submission["id"].astype(str)
submission["well"] = submission["id"].str.rsplit("_", n=1).str[0]
submission["row_idx"] = submission["id"].str.rsplit("_", n=1).str[-1].astype(int)
submission[["id", "tvt"]].to_csv(
    WORK / "submission_before_offset_krige.csv", index=False
)

horizontal_files = {}
horizontal_priority = {}
for raw_path in glob.glob(
    "/kaggle/input/**/*__horizontal_well.csv", recursive=True
):
    path = Path(raw_path)
    well = path.name.replace("__horizontal_well.csv", "")
    # Test-schema files must win over same-id train/demo copies.
    parts_lower = [part.lower() for part in path.parts]
    priority = (
        0 if "test" in parts_lower else 1,
        0 if "competitions" in parts_lower else 1,
        len(path.parts),
        str(path),
    )
    if well not in horizontal_priority or priority < horizontal_priority[well]:
        horizontal_priority[well] = priority
        horizontal_files[well] = path

theta = float(geometry["theta"])
cosine, sine = np.cos(theta), np.sin(theta)
train_x = lookup["cx"].to_numpy(float)
train_y = lookup["cy"].to_numpy(float)
train_xy = np.column_stack(
    [
        train_x * cosine + train_y * sine,
        (-train_x * sine + train_y * cosine) / float(geometry["anisotropy"]),
    ]
)
train_label = lookup["b_w"].to_numpy(float)
train_well = lookup["well"].to_numpy(str)

output = submission["tvt"].to_numpy(float).copy()
audit_rows = []
for well, group in submission.groupby("well", sort=False):
    path = horizontal_files.get(str(well))
    if path is None:
        audit_rows.append({"well": well, "applied": False, "reason": "missing horizontal"})
        continue
    horizontal = pd.read_csv(path)
    row_index = group["row_idx"].to_numpy(int)
    if row_index.max() >= len(horizontal):
        raise RuntimeError(f"row index outside horizontal file for {well}")
    x = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(float)[row_index]
    y = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(float)[row_index]
    qx, qy = float(np.nanmean(x)), float(np.nanmean(y))
    if not np.isfinite(qx + qy):
        audit_rows.append({"well": well, "applied": False, "reason": "invalid coordinates"})
        continue
    query = _transform(qx, qy, geometry)
    distance2 = np.sum((train_xy - query[None, :]) ** 2, axis=1)
    keep = train_well != str(well)
    distance2 = distance2[keep]
    labels = train_label[keep]
    k = min(int(geometry["k"]), len(labels))
    order = np.argsort(distance2)[:k]
    selected_distance2 = distance2[order]
    weights = np.exp(
        -selected_distance2 / (2.0 * float(geometry["range_ft"]) ** 2)
    )
    weight_sum = float(np.sum(weights))
    nearest = float(np.sqrt(np.min(selected_distance2)))
    ess = float(weight_sum**2 / max(np.sum(weights**2), 1e-12))
    if weight_sum <= 1e-12:
        audit_rows.append({"well": well, "applied": False, "reason": "zero kernel weight"})
        continue
    if nearest > float(guard.get("nearest_distance_max", np.inf)):
        audit_rows.append(
            {"well": well, "applied": False, "reason": "outside distance guard", "nearest": nearest, "ess": ess}
        )
        continue
    if ess < float(guard.get("effective_neighbors_min", 0.0)):
        audit_rows.append(
            {"well": well, "applied": False, "reason": "below ESS guard", "nearest": nearest, "ess": ess}
        )
        continue

    offset = float(np.sum(weights * labels[order]) / weight_sum)
    scalar_shift = float(application["weight"]) * offset
    cap = application.get("applied_cap_ft")
    if cap is not None:
        scalar_shift = float(np.clip(scalar_shift, -float(cap), float(cap)))

    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(tvt_input))
    if not len(known):
        raise RuntimeError(f"no known anchor for {well}")
    md_since = np.maximum(md[row_index] - md[int(known[-1])], 0.0)
    ramp_name = str(application["ramp"])
    if ramp_name == "constant":
        ramp = np.ones(len(group), np.float64)
    elif ramp_name.startswith("tau"):
        tau = float(ramp_name.replace("tau", ""))
        ramp = 1.0 - np.exp(-md_since / tau)
    else:
        raise RuntimeError(f"unknown ramp {ramp_name}")
    move = scalar_shift * ramp
    output[group.index.to_numpy(int)] += move
    audit_rows.append(
        {
            "well": well,
            "applied": True,
            "reason": "ok",
            "cx": qx,
            "cy": qy,
            "nearest": nearest,
            "ess": ess,
            "predicted_b_w": offset,
            "scalar_shift": scalar_shift,
            "mean_abs_move": float(np.mean(np.abs(move))),
            "max_abs_move": float(np.max(np.abs(move))),
        }
    )

candidate = submission[["id"]].copy()
candidate["tvt"] = output
if not np.isfinite(candidate["tvt"].to_numpy(float)).all():
    raise RuntimeError("non-finite final predictions")
candidate.to_csv(WORK / "submission_offset_krige.csv", index=False)
candidate.to_csv(SUBMISSION_PATH, index=False)
audit = pd.DataFrame(audit_rows)
audit.to_csv(WORK / "offset_krige_hidden_audit.csv", index=False)

move = output - submission["tvt"].to_numpy(float)
summary = {
    "rows": len(candidate),
    "wells": int(submission["well"].nunique()),
    "applied_wells": int(audit.get("applied", pd.Series(dtype=bool)).fillna(False).sum()),
    "base_lineage_required": config.get("base_lineage"),
    "mean_abs_move": float(np.mean(np.abs(move))),
    "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
    "max_abs_move": float(np.max(np.abs(move))),
    "output": str(SUBMISSION_PATH),
}
with (WORK / "offset_krige_hidden_audit.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
print("OFFSET-KRIGE COMPLETE")
print(json.dumps(summary, indent=2))
print(audit.to_string(index=False))
