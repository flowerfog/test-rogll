"""Fold-clean local reconstruction of the learned endpoint in the 6.47 notebook.

Run this as one AutoDL cell after uploading the exact 6.47 notebook.  The cell
executes the notebook's feature/PF definitions directly, then creates honest
whole-well OOF predictions for:

    GS x 1.3 lik-PF -> FLE LightGBM mean -> 0.60 learned + 0.40 likPF-scale-5

Spatial references are rebuilt from each outer fold's training wells only.
Tree counts are selected on an inner whole-well fold and then frozen before the
outer validation fold is predicted.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

EXPERIMENT = os.environ.get(
    "ROGII_FLE_EXPERIMENT", "mainline_6p47_foldclean_fle_gs13_v0"
)
EXPECTED_NOTEBOOK_SHA256 = (
    "249019dfc160599ad7639c66173024dba73a1c8cc6c8ce86fb73b0e65f502b71"
)
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260723
N_FOLDS = 5
TRAIN_ROWS_PER_WELL = int(os.environ.get("ROGII_FLE_TRAIN_ROWS", "2048"))
FEATURE_JOBS = int(os.environ.get("ROGII_FLE_FEATURE_JOBS", "2"))
MODEL_CONFIGS = int(os.environ.get("ROGII_FLE_MODEL_CONFIGS", "3"))
REQUESTED_FOLDS = tuple(
    int(value)
    for value in os.environ.get("ROGII_FLE_FOLDS", "1,2,3,4,5").split(",")
    if value.strip()
)
MIN_FREE_GB = float(os.environ.get("ROGII_FLE_MIN_FREE_GB", "1.5"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_data_root() -> Path:
    candidates = [
        Path(os.environ.get("ROGII_DATA", "")),
        ROOT / "rogii-wellbore-geology-prediction",
        ROOT / "dataset",
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    ]
    for candidate in candidates:
        if candidate and (candidate / "train").is_dir():
            return candidate.resolve()
    for path in ROOT.glob("**/sample_submission.csv"):
        if (path.parent / "train").is_dir():
            return path.parent.resolve()
    raise FileNotFoundError("Cannot locate the competition data root under /root/autodl-tmp")


def _find_source_notebook() -> Path:
    explicit = os.environ.get("ROGII_6P47_NOTEBOOK")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            ROOT / "rogii-luck-is-all-you-need.ipynb",
            Path.cwd() / "rogii-luck-is-all-you-need.ipynb",
        ]
    )
    candidates.extend(ROOT.glob("*.ipynb"))
    seen = set()
    mismatches = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        digest = _sha256(candidate)
        if digest == EXPECTED_NOTEBOOK_SHA256:
            return candidate
        mismatches.append((str(candidate), digest))
    raise FileNotFoundError(
        "Upload the exact 6.47 notebook. Expected SHA256 "
        f"{EXPECTED_NOTEBOOK_SHA256}; inspected={mismatches[:10]}"
    )


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


DATA_ROOT = _find_data_root()
SOURCE_NOTEBOOK = _find_source_notebook()
os.environ["ROGII_DATA"] = str(DATA_ROOT)
os.environ["SHOW_FIGS"] = "0"
os.environ["FAST"] = "0"

if _free_gb(ROOT) < MIN_FREE_GB:
    raise RuntimeError(
        f"Only {_free_gb(ROOT):.2f} GB free; require at least {MIN_FREE_GB:.2f} GB"
    )


def _load_source_definitions(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index in (33, 36, 37, 39, 40, 41, 42):
        source = "".join(notebook["cells"][index].get("source", []))
        # Dynamically executed notebook cells have a synthetic ``filename::cell``
        # origin. Numba cannot derive a persistent cache locator from that name.
        # Disabling only the disk cache leaves JIT compilation and numerics intact.
        source = source.replace("cache=True", "cache=False")
        exec(compile(source, f"{path}::cell_{index}", "exec"), globals())


_load_source_definitions(SOURCE_NOTEBOOK)
CFG.DATA = DATA_ROOT
CFG.OUT = OUT_DIR
CFG.n_jobs = FEATURE_JOBS
CFG.SHOW_FIGS = False

source_cell_37 = "".join(
    json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))["cells"][37]["source"]
)
if "* 1.3" not in source_cell_37 and "*1.3" not in source_cell_37:
    raise RuntimeError("The source notebook does not contain the required GS x 1.3 lik-PF")


FOLD_PATH_CANDIDATES = [
    Path(os.environ.get("ROGII_FOLD_METADATA", "")),
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    ROOT / "well_metadata.csv",
]
FOLD_PATH = next(
    (
        path.resolve()
        for path in FOLD_PATH_CANDIDATES
        if str(path) and path.is_file()
    ),
    None,
)
if FOLD_PATH is None:
    raise FileNotFoundError(
        "Missing frozen well_metadata.csv. Upload the foldclean_full_fleongg_lgb_v0 "
        "metadata or set ROGII_FOLD_METADATA. New folds are deliberately not invented."
    )

folds = pd.read_csv(FOLD_PATH)
if not {"well", "fold"}.issubset(folds.columns):
    raise RuntimeError(f"Fold file lacks well/fold columns: {list(folds.columns)}")
folds = folds[["well", "fold"]].copy()
folds["well"] = folds["well"].astype(str)
folds["fold"] = folds["fold"].astype(int)
if folds["well"].duplicated().any() or set(folds["fold"]) != set(range(1, 6)):
    raise RuntimeError("Frozen fold metadata is malformed")

available_wells = {
    path.stem.replace("__horizontal_well", "")
    for path in (DATA_ROOT / "train").glob("*__horizontal_well.csv")
}
missing_files = sorted(set(folds["well"]) - available_wells)
if missing_files:
    raise RuntimeError(f"Fold metadata references {len(missing_files)} absent wells: {missing_files[:10]}")

fold_by_well = dict(zip(folds["well"], folds["fold"]))
all_wells = folds["well"].tolist()
RUN_SIGNATURE_PAYLOAD = {
    "source_sha256": _sha256(SOURCE_NOTEBOOK),
    "fold_sha256": _sha256(FOLD_PATH),
    "train_rows_per_well": TRAIN_ROWS_PER_WELL,
    "model_configs": MODEL_CONFIGS,
    "seed": SEED,
}
RUN_SIGNATURE = hashlib.sha256(
    json.dumps(RUN_SIGNATURE_PAYLOAD, sort_keys=True).encode("utf-8")
).hexdigest()

print(
    "BASELINE CONFIG",
    {
        "experiment": EXPERIMENT,
        "data_root": str(DATA_ROOT),
        "source_notebook": str(SOURCE_NOTEBOOK),
        "source_sha256": _sha256(SOURCE_NOTEBOOK),
        "fold_source": str(FOLD_PATH),
        "fold_counts": folds.groupby("fold").size().to_dict(),
        "train_rows_per_well": TRAIN_ROWS_PER_WELL,
        "model_configs": MODEL_CONFIGS,
        "requested_folds": REQUESTED_FOLDS,
        "free_gb": _free_gb(ROOT),
        "run_signature": RUN_SIGNATURE,
    },
    flush=True,
)


LIKPF_PATH = OUT_DIR / "likpf_gs13.pkl"


def _build_likpf_cache() -> pd.DataFrame:
    if LIKPF_PATH.exists():
        frame = pd.read_pickle(LIKPF_PATH)
        print(f"loaded lik-PF cache rows={len(frame):,}", flush=True)
        return frame
    started = time.time()
    parts = []
    failures = []
    for number, well in enumerate(all_wells, 1):
        try:
            part = _likpf_rows(well, "train")
            if part is None or part.empty:
                raise RuntimeError("empty lik-PF frame")
            parts.append(part)
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 25 == 0 or number == len(all_wells):
            print(
                f"lik-PF GSx1.3 {number}/{len(all_wells)} failures={len(failures)} "
                f"elapsed_min={(time.time()-started)/60:.1f}",
                flush=True,
            )
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR / "likpf_failures.csv", index=False)
        raise RuntimeError(f"lik-PF failed for {len(failures)} wells: {failures[:5]}")
    frame = pd.concat(parts, ignore_index=True)
    if frame["id"].duplicated().any():
        raise RuntimeError("Duplicate ids in lik-PF cache")
    frame.to_pickle(LIKPF_PATH)
    return frame


likpf = _build_likpf_cache()
likpf_by_well = {
    well: group.drop(columns="well", errors="ignore").copy()
    for well, group in likpf.assign(well=likpf["id"].str.rsplit("_", n=1).str[0]).groupby(
        "well", sort=False
    )
}


def _even_positions(length: int, limit: int) -> np.ndarray:
    if limit <= 0 or length <= limit:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, limit).round().astype(np.int64))


def _one_feature_frame(well: str, sample: bool) -> pd.DataFrame:
    horizontal = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    typewell = DATA_ROOT / "train" / f"{well}__typewell.csv"
    frame = build_well(str(horizontal), str(typewell), is_train=True)
    if frame is None or frame.empty:
        raise RuntimeError("empty build_well result")
    frame = add_likpf_features(frame, likpf_by_well[well])
    frame["source_fold"] = np.int8(fold_by_well[well])
    frame["row_idx"] = frame["id"].str.rsplit("_", n=1).str[1].astype(np.int32)
    full_rows = len(frame)
    if sample:
        frame = frame.iloc[_even_positions(full_rows, TRAIN_ROWS_PER_WELL)].copy()
        frame["sample_weight"] = np.float32(full_rows / len(frame))
    else:
        frame["sample_weight"] = np.float32(1.0)
    return frame


def _build_outer_frames(outer_fold: int):
    global _FI, _DI
    train_wells = folds.loc[folds["fold"] != outer_fold, "well"].tolist()
    valid_wells = folds.loc[folds["fold"] == outer_fold, "well"].tolist()

    # This bank is the central leakage guard: neither train nor validation
    # features may consult a well from the held-out outer fold.
    init_imputers(train_wells)
    started = time.time()
    train_parts = []
    valid_parts = []
    failures = []
    sequence = [(well, True) for well in train_wells] + [
        (well, False) for well in valid_wells
    ]
    def build_item(item):
        well, sample = item
        try:
            return well, sample, _one_feature_frame(well, sample=sample), None
        except Exception as exc:
            return well, sample, None, repr(exc)

    batch_size = max(4, FEATURE_JOBS * 4)
    for start in range(0, len(sequence), batch_size):
        batch = sequence[start : start + batch_size]
        results = Parallel(n_jobs=FEATURE_JOBS, prefer="threads")(
            delayed(build_item)(item) for item in batch
        )
        for well, sample, part, reason in results:
            if reason is None:
                (train_parts if sample else valid_parts).append(part)
            else:
                failures.append(
                    {
                        "outer_fold": outer_fold,
                        "well": well,
                        "role": "train" if sample else "valid",
                        "reason": reason,
                    }
                )
        number = min(start + len(batch), len(sequence))
        if number % 25 < batch_size or number == len(sequence):
            print(
                f"fold {outer_fold} features {number}/{len(sequence)} "
                f"failures={len(failures)} elapsed_min={(time.time()-started)/60:.1f}",
                flush=True,
            )
    if failures:
        failure_path = OUT_DIR / f"fold{outer_fold}_feature_failures.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        valid_failures = [row for row in failures if row["role"] == "valid"]
        if valid_failures:
            raise RuntimeError(f"Outer validation feature failures: {valid_failures[:5]}")
        print(f"warning: excluded {len(failures)} failed outer-train wells", flush=True)
    return pd.concat(train_parts, ignore_index=True), pd.concat(valid_parts, ignore_index=True)


EXCLUDED_FEATURES = {
    "well",
    "id",
    "target",
    "source_fold",
    "row_idx",
    "sample_weight",
}


def _feature_columns(train: pd.DataFrame, valid: pd.DataFrame) -> list[str]:
    return [
        column
        for column in train.columns
        if column not in EXCLUDED_FEATURES
        and not (column.startswith("likpf_scale_") or column == "likpf_mean")
        and column in valid.columns
        and pd.api.types.is_numeric_dtype(train[column])
    ]


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        frame.reindex(columns=columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _fit_lgb_config(train, valid, columns, params, outer_fold, config_index):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    inner_fold = outer_fold % N_FOLDS + 1
    inner_train = train[train["source_fold"] != inner_fold]
    inner_valid = train[train["source_fold"] == inner_fold]
    if inner_train.empty or inner_valid.empty:
        raise RuntimeError(f"Empty nested split outer={outer_fold} inner={inner_fold}")

    tune_params = dict(params)
    tune_fit = dict(
        X=_matrix(inner_train, columns),
        y=inner_train["target"].to_numpy(np.float32),
        sample_weight=inner_train["sample_weight"].to_numpy(np.float32),
        eval_set=[
            (
                _matrix(inner_valid, columns),
                inner_valid["target"].to_numpy(np.float32),
            )
        ],
        eval_metric="rmse",
        callbacks=[early_stopping(250, verbose=False), log_evaluation(0)],
    )
    try:
        tune_model = LGBMRegressor(**tune_params)
        tune_model.fit(**tune_fit)
    except Exception as exc:
        if "device_type" not in tune_params:
            raise
        print(f"LightGBM GPU fallback for config {config_index}: {exc!r}", flush=True)
        tune_params.pop("device_type", None)
        tune_params.pop("gpu_use_dp", None)
        tune_model = LGBMRegressor(**tune_params)
        tune_model.fit(**tune_fit)
    best_iteration = int(tune_model.best_iteration_ or tune_params["n_estimators"])
    del tune_model, inner_train, inner_valid
    gc.collect()

    final_params = dict(tune_params)
    final_params["n_estimators"] = best_iteration
    final_model = LGBMRegressor(**final_params)
    final_model.fit(
        _matrix(train, columns),
        train["target"].to_numpy(np.float32),
        sample_weight=train["sample_weight"].to_numpy(np.float32),
    )
    model_path = MODEL_DIR / f"lgb{config_index}_fold{outer_fold}.joblib"
    joblib.dump(final_model, model_path)
    prediction = final_model.predict(_matrix(valid, columns)).astype(np.float64)
    del final_model
    gc.collect()
    return prediction, best_iteration, inner_fold


def _metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = prediction - truth
    result = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }
    per_well = (
        pd.DataFrame({"well": frame["well"], "error2": error**2})
        .groupby("well")["error2"]
        .mean()
        .pow(0.5)
    )
    result.update(
        {
            "median_well_rmse": float(per_well.median()),
            "p75_well_rmse": float(per_well.quantile(0.75)),
            "p95_well_rmse": float(per_well.quantile(0.95)),
            "max_well_rmse": float(per_well.max()),
        }
    )
    md = frame["md_since"].to_numpy(np.float64)
    strata = {
        "near_0_500_rmse": (md <= 500),
        "mid_500_1000_rmse": (md > 500) & (md <= 1000),
        "mid_1000_2000_rmse": (md > 1000) & (md <= 2000),
        "far_gt2000_rmse": (md > 2000),
    }
    for name, mask in strata.items():
        result[name] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None
    return result


started = time.time()
fold_outputs = []
fold_reports = []
for outer_fold in REQUESTED_FOLDS:
    output_path = OUT_DIR / f"fold{outer_fold}_oof.pkl"
    report_path = OUT_DIR / f"fold{outer_fold}_report.json"
    if output_path.exists() and report_path.exists():
        saved_report = json.loads(report_path.read_text(encoding="utf-8"))
        if saved_report.get("run_signature") != RUN_SIGNATURE:
            raise RuntimeError(
                f"fold {outer_fold} artifacts have a different configuration. "
                "Move/remove that fold's OOF, report, and models before rerunning."
            )
        print(f"fold {outer_fold}: resume from completed artifacts", flush=True)
        fold_outputs.append(pd.read_pickle(output_path))
        fold_reports.append(saved_report)
        continue

    train_frame, valid_frame = _build_outer_frames(outer_fold)
    columns = _feature_columns(train_frame, valid_frame)
    if len(columns) < 100:
        raise RuntimeError(f"Only {len(columns)} usable features in fold {outer_fold}")
    (OUT_DIR / f"fold{outer_fold}_features.json").write_text(
        json.dumps(columns, indent=2), encoding="utf-8"
    )

    lgb_device, lgb_device_name = _device()
    print(f"fold {outer_fold}: LightGBM device={lgb_device_name}", flush=True)
    configs = lgb_configs(lgb_device)[:MODEL_CONFIGS]
    model_predictions = []
    model_reports = []
    for config_index, params in enumerate(configs):
        prediction, best_iteration, inner_fold = _fit_lgb_config(
            train_frame,
            valid_frame,
            columns,
            params,
            outer_fold,
            config_index,
        )
        model_predictions.append(prediction)
        model_reports.append(
            {
                "config_index": config_index,
                "inner_validation_fold": inner_fold,
                "best_iteration": best_iteration,
            }
        )
        print(
            f"fold {outer_fold} model {config_index}: inner={inner_fold} "
            f"best_iteration={best_iteration}",
            flush=True,
        )

    learned_delta = np.mean(model_predictions, axis=0)
    prediction = make_prediction(valid_frame, learned_delta, None)
    truth = (
        valid_frame["last_known_tvt"].to_numpy(np.float64)
        + valid_frame["target"].to_numpy(np.float64)
    )
    output = pd.DataFrame(
        {
            "id": valid_frame["id"].astype(str).to_numpy(),
            "well": valid_frame["well"].astype(str).to_numpy(),
            "row_idx": valid_frame["row_idx"].to_numpy(np.int32),
            "fold": np.int8(outer_fold),
            "md_since": valid_frame["md_since"].to_numpy(np.float32),
            "tvt_true": truth,
            "fle_gs13": prediction,
            "learned_delta": learned_delta,
            "likpf_scale_5": valid_frame["likpf_scale_5"].to_numpy(np.float32),
        }
    )
    report = {
        "run_signature": RUN_SIGNATURE,
        "outer_fold": outer_fold,
        "train_wells": int(train_frame["well"].nunique()),
        "valid_wells": int(valid_frame["well"].nunique()),
        "train_rows": int(len(train_frame)),
        "valid_rows": int(len(valid_frame)),
        "features": len(columns),
        "models": model_reports,
        "metrics": _metrics(output, output["fle_gs13"].to_numpy()),
    }
    output.to_pickle(output_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FOLD RESULT", json.dumps(report, indent=2), flush=True)
    fold_outputs.append(output)
    fold_reports.append(report)
    del train_frame, valid_frame, output, learned_delta, model_predictions
    gc.collect()


oof = pd.concat(fold_outputs, ignore_index=True)
if oof["id"].duplicated().any():
    raise RuntimeError("Duplicate OOF ids")

per_well = []
for well, group in oof.groupby("well", sort=False):
    error = group["fle_gs13"].to_numpy() - group["tvt_true"].to_numpy()
    far = group["md_since"].to_numpy() > 2000
    per_well.append(
        {
            "well": well,
            "fold": int(group["fold"].iloc[0]),
            "rows": len(group),
            "exact_sse": float(np.dot(error, error)),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "far_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else np.nan,
        }
    )
per_well = pd.DataFrame(per_well).sort_values("rmse", ascending=False)
worst_count = max(1, int(np.ceil(0.20 * len(per_well))))
worst_wells = set(per_well.head(worst_count)["well"])
total_sse = float(per_well["exact_sse"].sum())
worst20_share = float(
    per_well.loc[per_well["well"].isin(worst_wells), "exact_sse"].sum() / total_sse
)

expected_wells = set(
    folds.loc[folds["fold"].isin(REQUESTED_FOLDS), "well"].astype(str)
)
covered_wells = set(oof["well"].astype(str))
coverage = len(covered_wells) / len(expected_wells)
expected_rows = 0
for well in expected_wells:
    horizontal = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
        usecols=["TVT", "TVT_input"],
    )
    expected_rows += int((horizontal["TVT_input"].isna() & horizontal["TVT"].notna()).sum())
row_coverage = len(oof) / expected_rows
all_folds_complete = set(REQUESTED_FOLDS) == set(range(1, 6))
baseline_complete = bool(
    all_folds_complete
    and coverage >= 0.995
    and row_coverage >= 0.995
    and MODEL_CONFIGS == 3
    and _sha256(SOURCE_NOTEBOOK) == EXPECTED_NOTEBOOK_SHA256
)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "strict_outer_foldclean_6p47_learned_endpoint_baseline",
    "source_notebook": str(SOURCE_NOTEBOOK),
    "source_sha256": _sha256(SOURCE_NOTEBOOK),
    "run_signature": RUN_SIGNATURE,
    "run_signature_payload": RUN_SIGNATURE_PAYLOAD,
    "fold_source": str(FOLD_PATH),
    "sample_unit": "complete physical well",
    "official_prefix_hidden_mask": True,
    "same_id_lookup": False,
    "uses_neighbor_cp12_gold": False,
    "spatial_reference_scope": "outer_train_only_for_both_training_and_validation_features",
    "tree_count_selection": "one_inner_whole_well_fold_then_outer_train_refit",
    "train_rows_per_well": TRAIN_ROWS_PER_WELL,
    "model_configs": MODEL_CONFIGS,
    "folds": fold_reports,
    "coverage": {
        "expected_wells": len(expected_wells),
        "covered_wells": len(covered_wells),
        "fraction": coverage,
        "missing_wells": sorted(expected_wells - covered_wells),
        "expected_rows": expected_rows,
        "covered_rows": len(oof),
        "row_fraction": row_coverage,
    },
    "oof": _metrics(oof, oof["fle_gs13"].to_numpy()),
    "worst20_percent_wells": {
        "count": worst_count,
        "sse_share": worst20_share,
        "wells": per_well.head(worst_count)["well"].tolist(),
    },
    "baseline_complete": baseline_complete,
    "training_fidelity": (
        "all_outer_train_rows"
        if TRAIN_ROWS_PER_WELL <= 0
        else f"evenly_sampled_{TRAIN_ROWS_PER_WELL}_rows_per_outer_train_well"
    ),
    "performance_gate": "none; this experiment establishes the local zero point",
    "elapsed_hours": (time.time() - started) / 3600,
    "free_gb": _free_gb(ROOT),
    "next": (
        "Use this exact OOF as the immutable local 6.47 baseline"
        if baseline_complete
        else "Complete all five folds with all three model configs before testing gains"
    ),
}

oof.to_pickle(OUT_DIR / "mainline_6p47_fle_gs13_oof.pkl")
oof.to_csv(OUT_DIR / "mainline_6p47_fle_gs13_oof.csv", index=False)
per_well.to_csv(OUT_DIR / "oof_per_well.csv", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("FINAL VERDICT", json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
