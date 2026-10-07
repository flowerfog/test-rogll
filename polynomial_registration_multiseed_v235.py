"""V235: deterministic multi-seed variance audit for the frozen V231 recipe."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF = ROOT / "oof_exports"
SOURCE = ROOT / "canonical_polynomial_registration_full773_v231.py"
REFERENCE = OOF / "canonical_polynomial_registration_full773_v231" / "fullrow_oof.parquet"
OUT = OOF / "polynomial_registration_multiseed_v235"
SEEDS = (231, 1231, 2231, 3231, 4231)
CAP = 12.0
DOSE = 0.25


def rmse(truth, prediction):
    return float(np.sqrt(np.mean(np.square(np.asarray(truth) - np.asarray(prediction)))))


def shaped(move):
    return CAP * np.tanh(np.asarray(move, np.float64) / CAP)


def main():
    started = time.time()
    if not SOURCE.is_file() or not REFERENCE.is_file():
        raise FileNotFoundError({"source": str(SOURCE), "reference": str(REFERENCE)})
    OUT.mkdir(parents=True, exist_ok=True)

    reference = pd.read_parquet(REFERENCE)[["id", "well", "fold", "truth", "v168", "polynomial_move"]]
    if len(reference) != 3_783_989 or reference.well.nunique() != 773:
        raise RuntimeError("Incomplete V231 reference")
    moves = [reference["polynomial_move"].to_numpy(np.float32)]
    run_reports = [{"seed": 231, "source": str(REFERENCE), "reused": True}]

    temp_root = ROOT / "tmp_v235_multiseed"
    resolved_temp = temp_root.resolve()
    if resolved_temp != ROOT.resolve() / "tmp_v235_multiseed":
        raise RuntimeError(f"Unsafe temporary path: {resolved_temp}")
    temp_root.mkdir(parents=True, exist_ok=True)

    completed = False
    try:
        for seed in SEEDS[1:]:
            run_dir = temp_root / f"seed_{seed}"
            output = run_dir / "fullrow_oof.parquet"
            if not output.is_file():
                environment = os.environ.copy()
                environment["V231_SEED"] = str(seed)
                environment["V231_OUT"] = str(run_dir)
                print(f"V235 running seed {seed}", flush=True)
                subprocess.run([sys.executable, str(SOURCE)], env=environment, check=True)
            current = pd.read_parquet(output)[["id", "truth", "v168", "polynomial_move"]]
            if not current["id"].astype(str).equals(reference["id"].astype(str)):
                raise RuntimeError(f"Seed {seed}: ID order mismatch")
            if np.max(np.abs(current["truth"] - reference["truth"])) > 1e-6:
                raise RuntimeError(f"Seed {seed}: truth mismatch")
            if np.max(np.abs(current["v168"] - reference["v168"])) > 1e-6:
                raise RuntimeError(f"Seed {seed}: parent mismatch")
            moves.append(current["polynomial_move"].to_numpy(np.float32))
            run_reports.append({"seed": seed, "source": str(output), "reused": False})
            del current
        completed = True
    finally:
        if completed and temp_root.is_dir() and temp_root.resolve().parent == ROOT.resolve():
            shutil.rmtree(temp_root)

    move_matrix = np.stack(moves).astype(np.float64)
    truth = reference["truth"].to_numpy(np.float64)
    parent = reference["v168"].to_numpy(np.float64)
    folds = reference["fold"].to_numpy(np.int8)
    recipes = {f"seed_{seed}": move_matrix[index] for index, seed in enumerate(SEEDS)}
    recipes["mean5"] = move_matrix.mean(axis=0)
    recipes["median5"] = np.median(move_matrix, axis=0)
    recipes["trimmed_mean5"] = np.sort(move_matrix, axis=0)[1:-1].mean(axis=0)

    predictions = {name: parent + DOSE * shaped(move) for name, move in recipes.items()}
    rows = []
    reference_prediction = predictions["seed_231"]
    for name, prediction in predictions.items():
        row = {
            "recipe": name,
            "rmse": rmse(truth, prediction),
            "gain_vs_seed231": rmse(truth, reference_prediction) - rmse(truth, prediction),
        }
        for fold in range(1, 6):
            mask = folds == fold
            row[f"fold{fold}_gain"] = rmse(truth[mask], reference_prediction[mask]) - rmse(truth[mask], prediction[mask])
        row["positive_folds"] = sum(row[f"fold{fold}_gain"] > 0 for fold in range(1, 6))
        row["minimum_fold_gain"] = min(row[f"fold{fold}_gain"] for fold in range(1, 6))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)

    aggregate_names = ("mean5", "median5", "trimmed_mean5")
    nested = np.empty(len(reference), np.float64)
    fold_rows = []
    for held_fold in range(1, 6):
        donor = folds != held_fold
        selected = min(aggregate_names, key=lambda name: rmse(truth[donor], predictions[name][donor]))
        held = folds == held_fold
        nested[held] = predictions[selected][held]
        row = {
            "fold": held_fold,
            "selected_aggregate": selected,
            "seed231_rmse": rmse(truth[held], reference_prediction[held]),
            "candidate_rmse": rmse(truth[held], nested[held]),
        }
        row["gain"] = row["seed231_rmse"] - row["candidate_rmse"]
        fold_rows.append(row)

    nested_gain = rmse(truth, reference_prediction) - rmse(truth, nested)
    verdict = {
        "experiment": "polynomial_registration_multiseed_v235",
        "stage": "five_seed_variance_reduction_of_frozen_V231b_recipe",
        "changed_variable": "candidate-bank random seed aggregation only",
        "seeds": list(SEEDS),
        "recipe": {"cap": CAP, "dose": DOSE},
        "runs": run_reports,
        "move_pairwise_correlation_min": float(np.min(np.corrcoef(move_matrix)[np.triu_indices(len(SEEDS), 1)])),
        "summary": summary.to_dict("records"),
        "nested_rmse": rmse(truth, nested),
        "nested_gain_vs_seed231": nested_gain,
        "folds": fold_rows,
        "checks": {
            "nested_gain_ge_0p02": nested_gain >= 0.02,
            "positive_folds_ge_4": sum(row["gain"] > 0 for row in fold_rows) >= 4,
            "minimum_fold_gain_ge_minus0p03": min(row["gain"] for row in fold_rows) >= -0.03,
        },
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    verdict["strict_pass"] = bool(all(verdict["checks"].values()))
    verdict["next"] = (
        "PASS -> deploy the selected five-seed aggregate in the V231 Kaggle cell"
        if verdict["strict_pass"]
        else "FAIL -> retain the single deterministic seed-231 deployment"
    )

    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "folds.csv", index=False)
    pd.DataFrame({
        "id": reference["id"], "well": reference["well"], "fold": folds,
        "truth": truth, "v168": parent, "seed231": reference_prediction, "nested": nested,
    }).to_parquet(OUT / "oof.parquet", index=False)
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("\nV235 SUMMARY")
    print(summary.to_string(index=False))
    print("\nV235 FOLDS")
    print(pd.DataFrame(fold_rows).to_string(index=False))
    print("\nV235 FINAL VERDICT")
    print(json.dumps(verdict, indent=2))
    print("saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
