"""HN18: row-level Fold2/Fold3 confirmation of the frozen HN16 residual.

Reconstruct HN17's selected paths, align them to the complete V66 incumbent,
then apply HN14's already-fitted hold2/hold3 residual models.  No model fitting,
dose scan, cap scan, or endpoint selection occurs in this script.
"""

from pathlib import Path
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
V4_SOURCE = ROOT / "rawdata_lattice_coarse_ranker_v4_cell.py"
if not V4_SOURCE.is_file():
    V4_SOURCE = Path("rawdata_lattice_coarse_ranker_v4_cell.py")
HN17_DIR = OOF_ROOT / "realistic_pairwise_deployment_crossfold_hn17"
HN14_DIR = OOF_ROOT / "realistic_pairwise_residual_feature_hn14"
V66_PATH = (
    OOF_ROOT / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
OUT_DIR = OOF_ROOT / "realistic_pairwise_deployment_residual_hn18"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ROW_PATH = OUT_DIR / "hn17_deployment_rows_fold23.parquet"
EVALUATION_FOLDS = (2, 3)
CAP_FT = 8.0
SEED = 20260730


def _clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _metrics(frame, prediction):
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = np.asarray(prediction, np.float64) - truth
    distance = frame["md_since"].to_numpy(np.float64)
    far = distance > 2000.0
    well_rmse = (
        pd.DataFrame({"well": frame["well"].astype(str), "error2": error * error})
        .groupby("well", sort=False)["error2"].mean().pow(0.5)
    )
    return {
        "rows": int(len(frame)),
        "wells": int(well_rmse.size),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p75_well_rmse": float(well_rmse.quantile(0.75)),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))),
    }


def _features(frame):
    out = pd.DataFrame(index=frame.index)
    group = frame.groupby("well", sort=False)
    distance = frame["md_since"].to_numpy(np.float64)
    base = frame["base"].to_numpy(np.float64)
    hn = frame["hn"].to_numpy(np.float64)
    delta = hn - base
    max_distance = group["md_since"].transform("max").to_numpy(np.float64)
    progress = distance / np.maximum(max_distance, 1.0)
    first_base = group["base"].transform("first").to_numpy(np.float64)
    first_hn = group["hn"].transform("first").to_numpy(np.float64)
    out["distance_kft"] = distance / 1000.0
    out["progress"] = progress
    out["progress2"] = progress * progress
    out["log1p_distance"] = np.log1p(np.maximum(distance, 0.0))
    out["base_relative"] = base - first_base
    out["base_relative_per_kft"] = (base - first_base) / np.maximum(distance / 1000.0, 0.25)
    for lag in (32, 128, 512):
        shifted_base = group["base"].shift(lag).to_numpy(np.float64)
        shifted_hn = group["hn"].shift(lag).to_numpy(np.float64)
        out[f"base_delta_lag{lag}"] = np.nan_to_num(base - shifted_base)
        out[f"hn_delta_lag{lag}"] = np.nan_to_num(hn - shifted_hn)
        out[f"disagreement_delta_lag{lag}"] = np.nan_to_num(
            delta - (shifted_hn - shifted_base)
        )
    out["hn_relative"] = hn - first_hn
    out["disagreement"] = delta
    out["abs_disagreement"] = np.abs(delta)
    out["disagreement_per_kft"] = delta / np.maximum(distance / 1000.0, 0.25)
    series = pd.Series(delta, index=frame.index)
    out["well_delta_mean"] = series.groupby(frame["well"], sort=False).transform("mean")
    out["well_delta_std"] = series.groupby(frame["well"], sort=False).transform("std").fillna(0.0)
    out["well_abs_delta_mean"] = (
        pd.Series(np.abs(delta), index=frame.index)
        .groupby(frame["well"], sort=False).transform("mean")
    )
    out["delta_minus_well_mean"] = delta - out["well_delta_mean"].to_numpy(np.float64)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    base_names = [
        "distance_kft", "progress", "progress2", "log1p_distance",
        "base_relative", "base_relative_per_kft",
        "base_delta_lag32", "base_delta_lag128", "base_delta_lag512",
    ]
    return out, base_names, list(out.columns)


def _bootstrap_v4():
    text = V4_SOURCE.read_text(encoding="utf-8")
    marker = "# --- V4 EXECUTION ---"
    if marker not in text:
        raise RuntimeError(f"V4 marker missing from {V4_SOURCE}")
    namespace = {"__name__": "hn18_v4_definitions", "__file__": str(V4_SOURCE)}
    exec(compile(text.split(marker, 1)[0], str(V4_SOURCE), "exec"), namespace)
    return namespace


def _reconstruct_rows():
    if ROW_PATH.is_file():
        frame = pd.read_parquet(ROW_PATH)
        print("HN18 ROW CACHE", {"rows": len(frame), "wells": frame["well"].nunique()}, flush=True)
        return frame
    choices = pd.read_parquet(HN17_DIR / "deployment_choices_fold23.parquet")
    choices = choices.loc[choices["control"].eq("aligned")].copy()
    if choices["well"].duplicated().any():
        raise RuntimeError("HN17 choices contain duplicate wells")
    G = _bootstrap_v4()
    np.random.seed(G["RANDOM_SEED"])
    all_wells = [
        path.name.split("__", 1)[0]
        for path in sorted((G["DATA_ROOT"] / "train").glob("*__horizontal_well.csv"))
    ]
    stable = G["V"]["_stable_folds"](all_wells)
    stable_fold = dict(zip(stable["well"].astype(str), stable["fold"].astype(int)))
    prior_wells = [well for well in all_wells if stable_fold[well] in (4, 5)]
    needed = sorted(set(prior_wells).union(choices["well"].astype(str)))
    records = {}
    failures = []
    for number, well in enumerate(needed, 1):
        try:
            records[well] = G["V"]["_load_record"](well)
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 50 == 0 or number == len(needed):
            print(f"HN18 records {number}/{len(needed)} failures={len(failures)}", flush=True)
    prior_records = [records[well] for well in prior_wells if well in records]
    if len(prior_records) < 307:
        raise RuntimeError(f"Only {len(prior_records)} prior records")
    prior = G["V"]["_fit_prior"](prior_records, G["SEGMENT_LENGTH_FT"])
    blocks = []
    reconstruction_failures = []
    for number, choice in enumerate(choices.itertuples(index=False), 1):
        well = str(choice.well)
        try:
            record = records[well]
            paths, _, _, _, segments, _ = G["_candidate_union"](record, prior)
            panel = G["_load_panel"](well)
            selected = int(choice.deployment_index)
            if len(paths) != len(panel["mse"]):
                raise RuntimeError(f"candidate count {len(paths)} != panel {len(panel['mse'])}")
            eval_rows = np.asarray(record["eval_rows"], np.int64)
            truth = np.asarray(record["truth"], np.float64)[eval_rows]
            prediction = G["B"]["_path_prediction"](
                record, prior, paths[selected], segments
            )[eval_rows]
            actual_sse = float(np.sum((prediction - truth) ** 2))
            expected_sse = float(panel["sse"][selected])
            tolerance = max(0.05, 2e-6 * max(expected_sse, 1.0))
            if abs(actual_sse - expected_sse) > tolerance:
                raise RuntimeError(f"path SSE mismatch {actual_sse}/{expected_sse}")
            md = np.asarray(record["md"], np.float64)[eval_rows]
            blocks.append(pd.DataFrame({
                "id": [f"{well}_{int(row)}" for row in eval_rows],
                "well": well,
                "row_idx": eval_rows,
                "fold": int(choice.fold),
                "tvt_true": truth.astype(np.float32),
                "md_since": (md - float(record["anchor_md"])).astype(np.float32),
                "hn": np.asarray(prediction, np.float32),
            }))
        except Exception as exc:
            reconstruction_failures.append({"well": well, "reason": repr(exc)})
        if number % 20 == 0 or number == len(choices):
            print(
                f"HN18 reconstruct {number}/{len(choices)} success={len(blocks)} "
                f"failures={len(reconstruction_failures)}", flush=True
            )
    pd.DataFrame(failures + reconstruction_failures).to_csv(
        OUT_DIR / "reconstruction_failures.csv", index=False
    )
    if len(blocks) < 290:
        raise RuntimeError(f"Only {len(blocks)} deployment wells reconstructed")
    frame = pd.concat(blocks, ignore_index=True)
    frame.to_parquet(ROW_PATH, index=False, compression="zstd")
    return frame


started = time.time()
for required in (V4_SOURCE, HN17_DIR / "deployment_choices_fold23.parquet", V66_PATH):
    if not required.is_file():
        raise FileNotFoundError(required)
for fold in EVALUATION_FOLDS:
    for arm in ("base_only", "hn_features"):
        path = HN14_DIR / "models" / f"{arm}_hold{fold}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)

hn = _reconstruct_rows()
v66 = pd.read_parquet(
    V66_PATH,
    columns=["id", "well", "row_idx", "tvt_true", "md_since", "incumbent_6p377_replay"],
).rename(columns={"incumbent_6p377_replay": "base", "tvt_true": "v66_truth"})
frame = hn.merge(v66, on=["id", "well", "row_idx"], how="inner", validate="one_to_one")
truth_gap = np.abs(frame["tvt_true"].to_numpy(np.float64) - frame["v66_truth"].to_numpy(np.float64))
if len(frame) < 1_400_000 or frame["well"].nunique() < 290 or truth_gap.max() > 0.001:
    raise RuntimeError(
        f"Alignment failed rows/wells/truth={len(frame)}/{frame['well'].nunique()}/{truth_gap.max()}"
    )
frame = frame.drop(columns=["v66_truth", "md_since_y"]).rename(columns={"md_since_x": "md_since"})
frame = frame.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
features, base_names, hn_names = _features(frame)
hn_only = [name for name in hn_names if name not in set(base_names)]

predictions = {name: np.full(len(frame), np.nan) for name in ("base_only", "hn_residual", "shuffle")}
fold_rows = []
for fold in EVALUATION_FOLDS:
    mask = frame["fold"].to_numpy() == fold
    positions = np.flatnonzero(mask)
    base_model = lgb.Booster(model_file=str(HN14_DIR / "models" / f"base_only_hold{fold}.txt"))
    hn_model = lgb.Booster(model_file=str(HN14_DIR / "models" / f"hn_features_hold{fold}.txt"))
    if base_model.num_feature() != len(base_names) or hn_model.num_feature() != len(hn_names):
        raise RuntimeError(f"Fold {fold} residual schema mismatch")
    predictions["base_only"][positions] = np.clip(
        base_model.predict(features.iloc[positions][base_names]), -CAP_FT, CAP_FT
    )
    predictions["hn_residual"][positions] = np.clip(
        hn_model.predict(features.iloc[positions][hn_names]), -CAP_FT, CAP_FT
    )
    shuffled = features.iloc[positions][hn_names].copy()
    well_values = frame.iloc[positions]["well"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED + fold)
    for well in np.unique(well_values):
        local = np.flatnonzero(well_values == well)
        values = shuffled.iloc[local][hn_only].to_numpy(copy=True)
        shuffled.iloc[local, shuffled.columns.get_indexer(hn_only)] = values[rng.permutation(len(local))]
    predictions["shuffle"][positions] = np.clip(
        hn_model.predict(shuffled), -CAP_FT, CAP_FT
    )

    local_frame = frame.iloc[positions]
    base = local_frame["base"].to_numpy(np.float64)
    base_metric = _metrics(local_frame, base)
    base_only_metric = _metrics(local_frame, base + predictions["base_only"][positions])
    hn_metric = _metrics(local_frame, base + predictions["hn_residual"][positions])
    shuffle_metric = _metrics(local_frame, base + predictions["shuffle"][positions])
    fold_rows.append({
        "fold": fold,
        "rows": len(local_frame),
        "wells": int(local_frame["well"].nunique()),
        "base_rmse": base_metric["rmse"],
        "base_only_rmse": base_only_metric["rmse"],
        "rmse": hn_metric["rmse"],
        "gain": base_metric["rmse"] - hn_metric["rmse"],
        "increment_vs_base_only": base_only_metric["rmse"] - hn_metric["rmse"],
        "far_gain": base_metric["far_gt2000_rmse"] - hn_metric["far_gt2000_rmse"],
        "shuffle_loss": shuffle_metric["rmse"] - hn_metric["rmse"],
        "p95_change": hn_metric["p95_well_rmse"] - base_metric["p95_well_rmse"],
    })

base = frame["base"].to_numpy(np.float64)
final = base + predictions["hn_residual"]
base_only = base + predictions["base_only"]
shuffle = base + predictions["shuffle"]
base_metrics = _metrics(frame, base)
metrics = _metrics(frame, final)
base_only_metrics = _metrics(frame, base_only)
shuffle_metrics = _metrics(frame, shuffle)
fold_report = pd.DataFrame(fold_rows)
checks = {
    "aligned_wells_ge_290": int(frame["well"].nunique()) >= 290,
    "overall_gain_ge_0p10": base_metrics["rmse"] - metrics["rmse"] >= 0.10,
    "increment_vs_base_only_ge_0p05": base_only_metrics["rmse"] - metrics["rmse"] >= 0.05,
    "both_folds_positive": bool((fold_report["gain"] > 0).all()),
    "both_fold_increments_positive": bool((fold_report["increment_vs_base_only"] > 0).all()),
    "both_shuffle_losses_ge_0p05": float(fold_report["shuffle_loss"].min()) >= 0.05,
    "maximum_fold_p95_change_le_1p0": float(fold_report["p95_change"].max()) <= 1.0,
}
strict_pass = all(checks.values())
output = frame[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
output["base"] = base.astype(np.float32)
output["hn_deployment"] = frame["hn"].to_numpy(np.float32)
output["hn_residual"] = final.astype(np.float32)
output.to_parquet(OUT_DIR / "hn18_oof.parquet", index=False, compression="zstd")
fold_report.to_csv(OUT_DIR / "folds.csv", index=False)
verdict = {
    "experiment": "realistic_pairwise_deployment_residual_hn18",
    "stage": "frozen_hn14_residual_on_clean_hn17_fold2_fold3_deployment_endpoints",
    "selection_or_tuning_count": 0,
    "protocol": {
        "evaluation_folds": list(EVALUATION_FOLDS),
        "selector": "HN17 four-model candidate-wise score median",
        "residual_models": "unchanged HN14 hold2/hold3 models",
        "correction_cap_ft": CAP_FT,
        "dose": 1.0,
    },
    "coverage": {"rows": len(frame), "wells": int(frame["well"].nunique())},
    "base": base_metrics,
    "matched_base_only": base_only_metrics,
    "hn_residual": metrics,
    "alignment_shuffle": shuffle_metrics,
    "gain": base_metrics["rmse"] - metrics["rmse"],
    "increment_vs_base_only": base_only_metrics["rmse"] - metrics["rmse"],
    "shuffle_loss": shuffle_metrics["rmse"] - metrics["rmse"],
    "folds": fold_rows,
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> HN16 has clean deployment-equivalent support on Fold1/2/3; await LB"
        if strict_pass
        else "FAIL -> HN16 residual does not transfer across deployment-equivalent folds"
    ),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nHN18 FOLDS")
print(fold_report.to_string(index=False), flush=True)
print("\nHN18 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
