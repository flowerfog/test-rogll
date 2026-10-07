"""Measure the unattainable CV produced by train-only formation surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FORMATIONS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")


def _metrics(frame: pd.DataFrame, prediction: str) -> dict[str, float | int | str]:
    valid = frame[prediction].notna() & frame["truth"].notna()
    covered = frame.loc[valid]
    error = covered[prediction].to_numpy(np.float64) - covered["truth"].to_numpy(np.float64)
    well_sse = covered.assign(squared_error=error**2).groupby("well")["squared_error"].mean()
    return {
        "endpoint": prediction,
        "rows": len(covered),
        "wells": covered["well"].nunique(),
        "row_coverage": float(len(covered) / len(frame)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.sqrt(well_sse).median()),
        "p95_well_rmse": float(np.sqrt(well_sse).quantile(0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=None)
    args, _ = parser.parse_known_args()

    root = args.root.resolve()
    out_dir = (args.out or root / "output" / "train_only_formation_leakage_audit").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []
    well_reports: list[dict[str, object]] = []
    paths = sorted((root / "train").glob("*__horizontal_well.csv"))
    for number, path in enumerate(paths, 1):
        well = path.name.removesuffix("__horizontal_well.csv")
        columns = ["Z", "TVT", "TVT_input", *FORMATIONS]
        source = pd.read_csv(path, usecols=columns)
        known = source["TVT_input"].notna().to_numpy()
        hidden = ~known & source["TVT"].notna().to_numpy()
        if known.sum() < 2 or hidden.sum() == 0:
            continue

        local = pd.DataFrame({"well": well, "truth": source.loc[hidden, "TVT"].to_numpy()})
        prefix_scores: dict[str, float] = {}
        for formation in FORMATIONS:
            surface = source[formation].to_numpy(np.float64)
            datum_values = (
                source.loc[known, "TVT_input"].to_numpy(np.float64)
                + source.loc[known, "Z"].to_numpy(np.float64)
                - surface[known]
            )
            datum = float(np.nanmedian(datum_values)) if np.isfinite(datum_values).any() else np.nan
            local[formation] = surface[hidden] - source.loc[hidden, "Z"].to_numpy(np.float64) + datum
            prefix_error = (
                surface[known]
                - source.loc[known, "Z"].to_numpy(np.float64)
                + datum
                - source.loc[known, "TVT_input"].to_numpy(np.float64)
            )
            prefix_scores[formation] = (
                float(np.sqrt(np.nanmean(prefix_error**2)))
                if np.isfinite(prefix_error).any()
                else np.inf
            )

        selected = min(prefix_scores, key=prefix_scores.get)
        if not np.isfinite(prefix_scores[selected]):
            continue
        local["prefix_selected"] = local[selected]
        prediction_matrix = local[list(FORMATIONS)].to_numpy(np.float64)
        truth = local["truth"].to_numpy(np.float64)
        candidate_mse = np.nanmean((prediction_matrix - truth[:, None]) ** 2, axis=0)
        oracle_index = int(np.nanargmin(candidate_mse))
        local["perwell_oracle"] = prediction_matrix[:, oracle_index]
        parts.append(local)
        well_reports.append(
            {
                "well": well,
                "prefix_selected": selected,
                "prefix_selected_rmse": prefix_scores[selected],
                "oracle_formation": FORMATIONS[int(oracle_index)],
            }
        )
        if number % 100 == 0:
            print(f"formation leakage audit {number}/{len(paths)}", flush=True)

    frame = pd.concat(parts, ignore_index=True)
    summary = pd.DataFrame(
        [_metrics(frame, endpoint) for endpoint in (*FORMATIONS, "prefix_selected", "perwell_oracle")]
    ).sort_values("rmse")
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(well_reports).to_csv(out_dir / "by_well.csv", index=False)

    verdict = {
        "experiment": "train_only_formation_leakage_audit",
        "deployable": False,
        "truth_usage": "train-only formation surfaces and oracle diagnostic",
        "summary": summary.to_dict(orient="records"),
        "best_nonoracle_rmse": float(
            summary.loc[summary["endpoint"] != "perwell_oracle", "rmse"].min()
        ),
        "interpretation": (
            "A low score here is a leakage ceiling, not a model result. "
            "These formation columns do not exist for competition test wells."
        ),
    }
    (out_dir / "audit.json").write_text(
        json.dumps(verdict, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(verdict, indent=2, allow_nan=False))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
