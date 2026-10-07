"""Audit fixed TVT-drift and structure-S state-axis coverage on all train wells."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT_OVERRIDE = None
DATA_ROOT_CANDIDATES = (
    Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
    Path("/root/autodl-tmp/wellbore-geology-prediction-artifacts"),
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    Path(r"E:\rogii-wellbore-geology-prediction"),
)
WIDTHS_FT = (32, 48, 64, 80, 96, 128, 160, 240, 320)


def _horizontal_files(root):
    patterns = (
        "train/*__horizontal_well.csv",
        "*__horizontal_well.csv",
        "**/train/*__horizontal_well.csv",
    )
    for pattern in patterns:
        files = sorted(root.glob(pattern))
        if files:
            return files
    return []


candidate_roots = []
if DATA_ROOT_OVERRIDE:
    candidate_roots.append(Path(DATA_ROOT_OVERRIDE))
candidate_roots.extend(DATA_ROOT_CANDIDATES)

data_root = None
horizontal_files = []
for candidate in candidate_roots:
    if not candidate.exists():
        continue
    files = _horizontal_files(candidate)
    if files:
        data_root = candidate
        horizontal_files = files
        break

if data_root is None:
    searched = "\n".join(f"  - {path}" for path in candidate_roots)
    raise FileNotFoundError(
        "Could not find train horizontal-well CSV files. Searched:\n"
        f"{searched}\nSet DATA_ROOT_OVERRIDE to the dataset root."
    )

print(
    "axis audit input:",
    {"data_root": str(data_root), "horizontal_files": len(horizontal_files)},
    flush=True,
)

tvt_rows = []
s_rows = []
tvt_state_coverage = {width: [] for width in WIDTHS_FT}
s_state_coverage = {width: [] for width in WIDTHS_FT}
well_rows = []
failures = []

for number, path in enumerate(horizontal_files, 1):
    try:
        frame = pd.read_csv(path, usecols=["Z", "TVT", "TVT_input"])
        tvt = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(float)
        z = pd.to_numeric(frame["Z"], errors="coerce").to_numpy(float)
        tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        evaluation = np.flatnonzero(~np.isfinite(tvt_input))
        if not len(known) or not len(evaluation):
            raise ValueError("missing known prefix or evaluation suffix")
        anchor = int(known[-1])
        if int(evaluation[0]) != anchor + 1:
            raise ValueError("TVT_input mask is not a contiguous prefix")

        tvt_drift = tvt[evaluation] - tvt[anchor]
        structure = tvt + z
        s_drift = structure[evaluation] - structure[anchor]
        tvt_abs = np.abs(tvt_drift)
        s_abs = np.abs(s_drift)
        finite = np.isfinite(tvt_abs) & np.isfinite(s_abs)
        tvt_abs = tvt_abs[finite]
        s_abs = s_abs[finite]
        if not len(tvt_abs):
            raise ValueError("no finite evaluation labels")

        tvt_rows.append(tvt_abs)
        s_rows.append(s_abs)
        row = {
            "well": path.name.split("__", 1)[0],
            "eval_rows": len(tvt_abs),
            "tvt_abs_p50": float(np.quantile(tvt_abs, 0.50)),
            "tvt_abs_p95": float(np.quantile(tvt_abs, 0.95)),
            "tvt_abs_max": float(np.max(tvt_abs)),
            "s_abs_p50": float(np.quantile(s_abs, 0.50)),
            "s_abs_p95": float(np.quantile(s_abs, 0.95)),
            "s_abs_max": float(np.max(s_abs)),
        }
        for width in WIDTHS_FT:
            tvt_coverage = float(np.mean(tvt_abs <= width))
            s_coverage = float(np.mean(s_abs <= width))
            tvt_state_coverage[width].append(tvt_coverage)
            s_state_coverage[width].append(s_coverage)
            row[f"tvt_coverage_{width}"] = tvt_coverage
            row[f"s_coverage_{width}"] = s_coverage
        well_rows.append(row)
    except Exception as exc:
        failures.append({"path": str(path), "reason": repr(exc)})

    if number % 100 == 0 or number == len(horizontal_files):
        print(
            f"axis audit {number}/{len(horizontal_files)} "
            f"ok={len(well_rows)} failures={len(failures)}",
            flush=True,
        )

if not tvt_rows:
    preview = failures[:10]
    raise RuntimeError(
        f"Found {len(horizontal_files)} horizontal files but built no valid wells. "
        f"First failures: {preview}"
    )

tvt_rows = np.concatenate(tvt_rows)
s_rows = np.concatenate(s_rows)
well_report = pd.DataFrame(well_rows)
quantiles = (0.50, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0)

coverage = pd.DataFrame(
    {
        "half_width_ft": WIDTHS_FT,
        "pooled_tvt_coverage": [np.mean(tvt_rows <= width) for width in WIDTHS_FT],
        "state_p10_tvt_coverage": [
            np.quantile(tvt_state_coverage[width], 0.10) for width in WIDTHS_FT
        ],
        "pooled_s_coverage": [np.mean(s_rows <= width) for width in WIDTHS_FT],
        "state_p10_s_coverage": [
            np.quantile(s_state_coverage[width], 0.10) for width in WIDTHS_FT
        ],
        "columns_at_1ft": [2 * width + 1 for width in WIDTHS_FT],
    }
)

print("\nSUMMARY")
print(
    {
        "wells": len(well_report),
        "failures": len(failures),
        "rows": len(tvt_rows),
        "carry_last_tvt_rmse": float(np.sqrt(np.mean(tvt_rows * tvt_rows))),
    }
)
print("\nPOOLED ABSOLUTE DRIFT QUANTILES")
print(
    pd.DataFrame(
        {
            "quantile": quantiles,
            "tvt_abs_ft": np.quantile(tvt_rows, quantiles),
            "structure_s_abs_ft": np.quantile(s_rows, quantiles),
        }
    ).to_string(index=False)
)
print("\nAXIS COVERAGE")
print(coverage.to_string(index=False))
print("\nPER-WELL P95")
print(
    well_report[["tvt_abs_p95", "s_abs_p95"]]
    .describe(percentiles=[0.10, 0.50, 0.90, 0.95, 0.99])
    .to_string()
)

recommended = coverage[
    (coverage["pooled_tvt_coverage"] >= 0.9995)
    & (coverage["state_p10_tvt_coverage"] >= 0.99)
]
recommended_width = (
    int(recommended.iloc[0]["half_width_ft"]) if len(recommended) else None
)
print(
    "\nVERDICT",
    {
        "recommended_half_width_ft": recommended_width,
        "strict_pass": recommended_width is not None,
        "next": (
            "build fixed-TVT-drift zero-learning DP"
            if recommended_width is not None
            else "fixed TVT-drift axis is not compact enough"
        ),
    },
)
