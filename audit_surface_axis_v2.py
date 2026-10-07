from pathlib import Path

import numpy as np
import pandas as pd


root = Path(__file__).resolve().parent / "train"
rows = []
prior_rows = {window: [] for window in (64, 128, 256, 512, 1024, 2048)}
for path in sorted(root.glob("*__horizontal_well.csv")):
    frame = pd.read_csv(path, usecols=["MD", "Z", "TVT", "TVT_input"])
    values = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        for column in frame.columns
    }
    known = np.flatnonzero(np.isfinite(values["TVT_input"]))
    hidden = np.flatnonzero(~np.isfinite(values["TVT_input"]) & np.isfinite(values["TVT"]))
    if not len(known) or not len(hidden):
        continue
    anchor = int(known[-1])
    tvt_offset = values["TVT"][hidden] - values["TVT_input"][anchor]
    z_relative = values["Z"][hidden] - values["Z"][anchor]
    surface_offset = tvt_offset + z_relative
    distance = values["MD"][hidden] - values["MD"][anchor]
    reconstructed_z = np.clip(z_relative / 512.0, -4.0, 4.0) * 512.0
    rows.append(
        pd.DataFrame(
            {
                "well": path.name[:8],
                "distance": distance,
                "tvt_offset": tvt_offset,
                "z_relative": z_relative,
                "surface_offset": surface_offset,
                "z_reconstruction_error": reconstructed_z - z_relative,
            }
        )
    )
    known_surface = values["TVT_input"][known] + values["Z"][known]
    anchor_surface = known_surface[-1]
    for window_ft in (64, 128, 256, 512, 1024, 2048):
        selected = known[values["MD"][known] >= values["MD"][anchor] - window_ft]
        if len(selected) < 8:
            slope = 0.0
        else:
            x_fit = values["MD"][selected] - values["MD"][anchor]
            y_fit = values["TVT_input"][selected] + values["Z"][selected] - anchor_surface
            x_centre = x_fit - np.mean(x_fit)
            slope = float(
                np.dot(x_centre, y_fit - np.mean(y_fit))
                / max(np.dot(x_centre, x_centre), 1e-12)
            )
        prior = slope * distance
        residual = surface_offset - prior
        prior_rows[window_ft].append(
            (residual.astype(np.float32), distance.astype(np.float32))
        )

audit = pd.concat(rows, ignore_index=True)
print("rows/wells", len(audit), audit["well"].nunique())
for name in ("tvt_offset", "z_relative", "surface_offset", "z_reconstruction_error"):
    values = audit[name].to_numpy(np.float64)
    print(
        name,
        {
            "min": float(np.min(values)),
            "p01": float(np.quantile(values, 0.01)),
            "p50": float(np.median(values)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
            "abs_p95": float(np.quantile(np.abs(values), 0.95)),
            "abs_p99": float(np.quantile(np.abs(values), 0.99)),
        },
    )
for bound in (64, 128, 192, 256, 384, 512):
    print(
        "surface coverage",
        bound,
        float(np.mean(np.abs(audit["surface_offset"].to_numpy()) <= bound)),
    )
for threshold in (500, 1000, 2000):
    selected = audit.loc[audit["distance"] > threshold, "surface_offset"].to_numpy()
    print(
        "distance",
        threshold,
        "rows",
        len(selected),
        "surface_abs_p95",
        float(np.quantile(np.abs(selected), 0.95)),
        "coverage128",
        float(np.mean(np.abs(selected) <= 128)),
    )

print("\nPREFIX SURFACE TREND PRIORS")
for window_ft, payloads in prior_rows.items():
    residual = np.concatenate([payload[0] for payload in payloads]).astype(np.float64)
    distance = np.concatenate([payload[1] for payload in payloads]).astype(np.float64)
    far = distance > 2000
    print(
        int(window_ft),
        {
            "prior_rmse": float(np.sqrt(np.mean(residual**2))),
            "residual_abs_p95": float(np.quantile(np.abs(residual), 0.95)),
            "coverage128": float(np.mean(np.abs(residual) <= 128)),
            "far_rmse": float(np.sqrt(np.mean(residual[far] ** 2))),
            "far_coverage128": float(np.mean(np.abs(residual[far]) <= 128)),
        },
    )
