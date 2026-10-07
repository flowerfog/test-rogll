"""Finalize the completed gateway when optional parquet support is absent."""

from pathlib import Path
import json

import pandas as pd


root = Path(__file__).resolve().parent
out = root / "output" / "event_aware_neighbor_transfer_gateway_v0"
summary = pd.read_csv(out / "candidate_summary.csv")
selection = pd.read_csv(out / "nested_selection.csv")
event = summary[summary["candidate"].str.startswith("event_")].sort_values("rmse").iloc[0].to_dict()
fullcurve = summary[summary["candidate"].eq("fullcurve_delta")].iloc[0].to_dict()
name = str(event["candidate"])
checks = {
    "gain_ge_0p10": bool(event["gain"] >= 0.10),
    "far_gain_ge_0p20": bool(event["far_gain"] >= 0.20),
    "positive_folds_ge_4": bool(event["positive_folds"] >= 4),
    "nonzero_folds_ge_4": bool(event["nonzero_folds"] >= 4),
    "shuffle_loss_ge_0p05": bool(event["shuffle_loss"] >= 0.05),
}
verdict = {
    "experiment": "event_aware_neighbor_transfer_gateway_v0",
    "stage": "fivefold_sparse_structural_event_transfer_gateway",
    "sample_unit": "complete target well; historical reference truth excludes target fold",
    "base": str(root / "full_oof_cp_crossfit_viewer.csv"),
    "wells": 773,
    "rows": 3783989,
    "base_rmse": float(event["base_rmse"]),
    "best_sparse_event": event,
    "fullcurve_benchmark": fullcurve,
    "fold_doses": selection[selection["candidate"].eq(name)].to_dict(orient="records"),
    "checks": checks,
    "strict_pass": bool(all(checks.values())),
    "next": "FAIL -> sparse event isolation removes the transferable signal; retain complete-curve neighbor transfer",
    "note": "Computation completed; this finalizer bypasses only the optional parquet export.",
}
(out / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print(json.dumps(verdict, indent=2))
