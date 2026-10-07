"""Reconstruct the 6.39 notebook's model core with honest whole-well OOF.

Run as one AutoDL cell after uploading the exact 6.39 notebook and keeping the
original SP45/ridge and FLE model artifacts below /root/autodl-tmp.

The reconstructed endpoint is:

    SP45 = project(0.30 * ridge_pp + 0.70 * selector_128seed)
    FLE  = 0.60 * warmup(fold_model_delta) + 0.40 * likPF_scale_5
    core = 0.60 * SP45 + 0.40 * FLE

Saved Trainer.oof_preds recover the ridge branch without retraining. Completed
FLE fold OOF files and models are reused; when they do not exist, only the FLE
branch is trained with held-out whole-well folds. The full-training deployment
models are inventory/provenance inputs, never OOF substitutes. The unsaved
five-fold ridge meta stacker is deterministically
reconstructed with five tiny closed-form Ridge fits; this is provenance replay,
not a new model.
Hidden-id patches and test-only overlap overrides are deliberately excluded.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


STARTED = time.time()
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
MODEL_PACKAGE_ROOT = Path(
    os.environ.get("ROGII_CLAUDE_MODEL_ROOT", ROOT / "rogii-claude-models-pub")
).resolve()
EXPERIMENT = "frontier_6p39_existing_models_oof_v0"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_NOTEBOOK_SHA256 = (
    "da460516b1aeda7404b1b1f2abb5702c418c31f64f4088ecaf81ca08fc795438"
)
MIN_COVERAGE = float(os.environ.get("ROGII_6P39_MIN_COVERAGE", "0.995"))
SP45_WEIGHT = 0.60
FLE_WEIGHT = 0.40
FLE_TRAIN_ROWS_PER_WELL = int(os.environ.get("ROGII_FLE_TRAIN_ROWS", "2048"))
FLE_MODEL_CONFIGS = int(os.environ.get("ROGII_FLE_MODEL_CONFIGS", "3"))
FLE_FEATURE_JOBS = int(os.environ.get("ROGII_FLE_FEATURE_JOBS", "2"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _first(candidates, label: str) -> Path:
    for path in candidates:
        if path is None or not str(path).strip():
            continue
        path = Path(path)
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Cannot find {label}: {[str(x) for x in candidates]}")


def _find_data_root() -> Path:
    candidates = [
        os.environ.get("ROGII_DATA"),
        ROOT / "rogii-wellbore-geology-prediction",
        ROOT / "dataset",
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    ]
    for path in candidates:
        if path is None or not str(path).strip():
            continue
        path = Path(path)
        if (path / "train").is_dir():
            return path.resolve()
    for path in ROOT.glob("**/sample_submission.csv"):
        if (path.parent / "train").is_dir():
            return path.parent.resolve()
    raise FileNotFoundError("Cannot locate raw competition data")


def _find_notebook() -> Path:
    explicit = os.environ.get("ROGII_6P39_NOTEBOOK")
    candidates = [Path(explicit)] if explicit else []
    candidates += [ROOT / "rogii-public-score-frontier-lab-visuals.ipynb"]
    candidates += list(ROOT.glob("*.ipynb"))
    inspected = []
    for path in candidates:
        if not path.is_file():
            continue
        digest = _sha256(path)
        inspected.append((str(path), digest))
        if digest.lower() == EXPECTED_NOTEBOOK_SHA256:
            return path.resolve()
    raise FileNotFoundError(
        f"Exact 6.39 notebook not found; expected {EXPECTED_NOTEBOOK_SHA256}; "
        f"inspected={inspected[:12]}"
    )


DATA_ROOT = _find_data_root()
NOTEBOOK_PATH = _find_notebook()
if not MODEL_PACKAGE_ROOT.is_dir():
    raise FileNotFoundError(
        "Missing original 6.39 model package: "
        f"{MODEL_PACKAGE_ROOT}. Set ROGII_CLAUDE_MODEL_ROOT only if it was moved."
    )

FLE_DEPLOYMENT_FEATURES = MODEL_PACKAGE_ROOT / "features.json"
FLE_DEPLOYMENT_MODELS = sorted(MODEL_PACKAGE_ROOT.glob("lgb*.pkl"))
if not FLE_DEPLOYMENT_FEATURES.is_file() or not FLE_DEPLOYMENT_MODELS:
    raise FileNotFoundError(
        "The original model package does not contain the FLE deployment artifacts "
        f"features.json + lgb*.pkl: {MODEL_PACKAGE_ROOT}"
    )
FOLD_PATH = _first(
    [
        os.environ.get("ROGII_FOLD_METADATA"),
        OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
        OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "folds.csv",
    ],
    "frozen fold metadata",
)
SP45_EXPORT = _first(
    [
        os.environ.get("ROGII_SP45_OOF"),
        OOF_ROOT / "sp45_foldclean_feature_export_v1" / "sp45_feature_oof.parquet",
    ],
    "128-seed SP45 replay OOF",
)


def _find_fle_root() -> Path:
    candidates = []
    explicit = os.environ.get("ROGII_FLE_FOLD_ROOT")
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(OUT_DIR / "fle_foldclean")
    candidates.append(OOF_ROOT / "mainline_6p47_foldclean_fle_gs13_v0")
    candidates.extend(path.parent.parent for path in ROOT.glob("**/models/lgb0_fold1.joblib"))
    candidates.extend(path.parent for path in ROOT.glob("**/fold1_oof.pkl"))
    reports = []
    for path in dict.fromkeys(candidates):
        if not path.is_dir():
            continue
        model_paths = list((path / "models").glob("lgb*_fold*.joblib"))
        model_count = len(model_paths)
        oof_count = len(list(path.glob("fold*_oof.pkl")))
        if model_count or oof_count:
            resolvable = 0
            for outer_fold in range(1, 6):
                has_oof = (path / f"fold{outer_fold}_oof.pkl").is_file()
                has_schema = (path / f"fold{outer_fold}_features.json").is_file()
                has_models = any(
                    model_path.name.endswith(f"_fold{outer_fold}.joblib")
                    for model_path in model_paths
                )
                resolvable += int(has_oof or (has_schema and has_models))
            reports.append(
                (resolvable * 1000 + oof_count * 100 + model_count, path.resolve())
            )
    if not reports:
        generated = OUT_DIR / "fle_foldclean"
        (generated / "models").mkdir(parents=True, exist_ok=True)
        print(
            "No fold-clean FLE artifacts found; missing folds will be trained under "
            f"{generated}. Deployment models in {MODEL_PACKAGE_ROOT} remain inference-only.",
            flush=True,
        )
        return generated.resolve()
    reports.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return reports[0][1]


FLE_ROOT = _find_fle_root()


def _normalise_folds(frame: pd.DataFrame) -> pd.DataFrame:
    well_col = next((c for c in ("well", "well_id", "Well") if c in frame), None)
    fold_col = next((c for c in ("fold", "outer_fold", "Fold") if c in frame), None)
    if well_col is None or fold_col is None:
        raise RuntimeError(f"Cannot identify fold columns: {frame.columns.tolist()}")
    out = frame[[well_col, fold_col]].copy()
    out.columns = ["well", "fold"]
    out["well"] = out["well"].astype(str)
    out["fold"] = pd.to_numeric(out["fold"], errors="raise").astype(int)
    out = out.drop_duplicates("well")
    if set(out["fold"]) != {1, 2, 3, 4, 5}:
        raise RuntimeError("Fold metadata must contain folds 1..5")
    return out


folds = _normalise_folds(pd.read_csv(FOLD_PATH))


def _ridge_artifact_candidates() -> list[dict]:
    reports = []
    train_paths = list(ROOT.glob("**/data/train.csv"))
    package_train = MODEL_PACKAGE_ROOT / "data" / "train.csv"
    if package_train.is_file():
        train_paths.insert(0, package_train)
    for train_path in dict.fromkeys(train_paths):
        artifact_root = train_path.parent.parent
        lgb = [
            sorted(directory.glob("*.pkl"))[0]
            for directory in sorted(artifact_root.glob("models/lightgbm-*"))
            if sorted(directory.glob("*.pkl"))
        ]
        cb = [
            sorted(directory.glob("*.pkl"))[0]
            for directory in sorted(artifact_root.glob("models/catboost-*"))
            if sorted(directory.glob("*.pkl"))
        ]
        if len(lgb) >= 3:
            reports.append(
                {
                    "root": artifact_root,
                    "train": train_path,
                    "lgb": lgb,
                    "cb": cb,
                    "score": (
                        len(lgb) * 10
                        + len(cb)
                        + (1000 if artifact_root.resolve() == MODEL_PACKAGE_ROOT else 0)
                        + (100 if "wellbore-geology-prediction-artifacts" in str(artifact_root) else 0)
                    ),
                }
            )
    return sorted(reports, key=lambda x: (x["score"], str(x["root"])), reverse=True)


ridge_candidates = _ridge_artifact_candidates()
if not ridge_candidates:
    raise FileNotFoundError("No ridge artifact root with data/train.csv and saved Trainers")


def _choose(candidates, env_name: str, label: str):
    explicit = os.environ.get(env_name)
    if explicit:
        explicit_path = Path(explicit).resolve()
        matched = [item for item in candidates if item["root"].resolve() == explicit_path]
        if not matched:
            raise FileNotFoundError(f"{label} root from {env_name} is invalid: {explicit_path}")
        return matched[0]
    best_score = candidates[0]["score"]
    tied = [item for item in candidates if item["score"] == best_score]
    if len(tied) > 1:
        raise RuntimeError(
            f"Ambiguous {label} roots. Set {env_name}. Candidates: "
            + repr([str(item["root"]) for item in tied])
        )
    return candidates[0]


RIDGE_ARTIFACT = _choose(ridge_candidates, "ROGII_RIDGE_ARTIFACT_ROOT", "ridge")

inventory = {
    "notebook": str(NOTEBOOK_PATH),
    "notebook_sha256": _sha256(NOTEBOOK_PATH),
    "data_root": str(DATA_ROOT),
    "fold_source": str(FOLD_PATH),
    "sp45_export": str(SP45_EXPORT),
    "original_model_package": str(MODEL_PACKAGE_ROOT),
    "fle_deployment_features": str(FLE_DEPLOYMENT_FEATURES),
    "fle_deployment_models": [str(path) for path in FLE_DEPLOYMENT_MODELS],
    "ridge_root": str(RIDGE_ARTIFACT["root"]),
    "ridge_lgb_trainers": [str(p) for p in RIDGE_ARTIFACT["lgb"]],
    "ridge_cb_trainers": [str(p) for p in RIDGE_ARTIFACT["cb"]],
    "fle_fold_root": str(FLE_ROOT),
    "base_endpoint_training_calls": 0,
    "fle_missing_fold_policy": "train held-out whole-well fold and checkpoint",
}
print("6P39 ARTIFACT INVENTORY", json.dumps(inventory, indent=2), flush=True)
(OUT_DIR / "artifact_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")

# Register the source notebook's koolbox compatibility class before unpickling
# saved Trainer objects. This cell performs no training by itself.
_bootstrap_notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
_bootstrap_source = "".join(_bootstrap_notebook["cells"][2].get("source", []))
exec(compile(_bootstrap_source, f"{NOTEBOOK_PATH}::cell_2", "exec"), globals())


def _trainer_oof(path: Path, expected_rows: int) -> np.ndarray:
    trainer = joblib.load(path)
    prediction = np.asarray(getattr(trainer, "oof_preds", None), dtype=np.float64)
    if prediction.ndim != 1 or len(prediction) != expected_rows:
        raise RuntimeError(f"Invalid saved OOF in {path}: {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Non-finite saved OOF in {path}")
    return prediction


# -----------------------------------------------------------------------------
# SP45 ridge branch: recover base-model OOF from the saved Trainer objects.
# -----------------------------------------------------------------------------
ridge_frame = pd.read_csv(RIDGE_ARTIFACT["train"], low_memory=False)
required = {"id", "well", "target", "last_known_tvt", "pf_ancc"}
if not required.issubset(ridge_frame.columns):
    raise RuntimeError(f"Ridge train artifact lacks {sorted(required - set(ridge_frame))}")
ridge_frame["id"] = ridge_frame["id"].astype(str)
ridge_frame["well"] = ridge_frame["well"].astype(str)
if ridge_frame["id"].duplicated().any():
    raise RuntimeError("Ridge train artifact contains duplicate ids")

base_oof = {}
for index, path in enumerate(RIDGE_ARTIFACT["lgb"], 1):
    base_oof[f"lightgbm-{index}"] = _trainer_oof(path, len(ridge_frame))
for index, path in enumerate(RIDGE_ARTIFACT["cb"], 1):
    base_oof[f"catboost-{index}"] = _trainer_oof(path, len(ridge_frame))
if len(base_oof) < 3:
    raise RuntimeError("Too few saved ridge base OOF columns")
base_oof = pd.DataFrame(base_oof)

ridge_params = {
    "random_state": 42,
    "alpha": 1.6602834637650032,
    "tol": 0.0005030247295617308,
    "positive": True,
    "fit_intercept": True,
}
groups = ridge_frame["well"].to_numpy()
target_delta = pd.to_numeric(ridge_frame["target"], errors="raise").to_numpy(np.float64)
meta_oof = np.full(len(ridge_frame), np.nan, dtype=np.float64)

ridge_trainer_paths = sorted(RIDGE_ARTIFACT["root"].glob("models/ridge*/*.pkl"))
ridge_trainer_paths += sorted(RIDGE_ARTIFACT["root"].glob("models/*ridge*trainer*.pkl"))
if ridge_trainer_paths:
    meta_oof = _trainer_oof(ridge_trainer_paths[0], len(ridge_frame))
else:
    # The original notebook never saved this cheap meta learner. Reconstructing
    # it is required for exact OOF and does not train a new predictive endpoint.
    # Record the five deterministic closed-form fits explicitly in provenance.
    meta_oof[:] = np.nan
    for fold_index, (train_index, valid_index) in enumerate(
        GroupKFold(5).split(base_oof, target_delta, groups), 1
    ):
        meta = Ridge(**ridge_params)
        meta.fit(base_oof.iloc[train_index], target_delta[train_index])
        meta_oof[valid_index] = meta.predict(base_oof.iloc[valid_index])
    inventory["meta_ridge_reconstructed"] = True
    inventory["meta_ridge_fits"] = 5

last = pd.to_numeric(ridge_frame["last_known_tvt"], errors="raise").to_numpy(np.float64)
pf_delta = pd.to_numeric(ridge_frame["pf_ancc"], errors="raise").to_numpy(np.float64) - last
if "md_since" in ridge_frame:
    md_since = pd.to_numeric(ridge_frame["md_since"], errors="coerce").fillna(0.0).to_numpy(np.float64)
else:
    md_since = np.zeros(len(ridge_frame), dtype=np.float64)
ridge_delta = (0.91 * meta_oof + 0.09 * pf_delta)
ridge_delta *= 1.0 - np.exp(-np.maximum(md_since, 0.0) / 85.0)
ridge_pp = last + ridge_delta

ridge_stage = ridge_frame[["id", "well"]].copy()
ridge_stage["truth"] = last + target_delta
ridge_stage["ridge_pp"] = ridge_pp


# -----------------------------------------------------------------------------
# SP45 selector replay and projection of the 0.3/0.7 combined endpoint.
# -----------------------------------------------------------------------------
sp45 = pd.read_parquet(SP45_EXPORT)
sp45["id"] = sp45["id"].astype(str)
sp45["well"] = sp45["well"].astype(str)
selector_column = "sp45_selector_raw"
if selector_column not in sp45:
    raise RuntimeError(f"SP45 replay lacks {selector_column}")
sp45_for_merge = sp45[["id", "row_idx", "truth", selector_column]].rename(
    columns={"truth": "sp45_truth"}
)
stage = ridge_stage.merge(sp45_for_merge, on="id", how="inner", validate="one_to_one")
alignment = len(stage) / max(len(ridge_stage), 1)
if alignment < MIN_COVERAGE:
    missing = ridge_stage.loc[~ridge_stage["id"].isin(stage["id"]), "well"].value_counts().head(10)
    raise RuntimeError(f"SP45/ridge alignment {alignment:.6%} < {MIN_COVERAGE:.3%}; missing={missing.to_dict()}")
truth_gap = np.abs(
    stage["truth"].to_numpy(np.float64) - stage["sp45_truth"].to_numpy(np.float64)
)
if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.01:
    raise RuntimeError(f"SP45/ridge truth mismatch: max={float(np.nanmax(truth_gap))}")
stage["sp45_preprojection"] = (
    0.30 * stage["ridge_pp"].to_numpy(np.float64)
    + 0.70 * stage[selector_column].to_numpy(np.float64)
)


def _robfit(s: np.ndarray, y: np.ndarray, degree: int = 3) -> np.ndarray:
    if len(s) < degree + 2:
        return y.copy()
    coefficients = np.polyfit(s, y, degree)
    for _ in range(4):
        residual = y - np.polyval(coefficients, s)
        scale = np.median(np.abs(residual)) * 1.4826 + 1e-6
        coefficients = np.polyfit(
            s, y, degree, w=1.0 / (1.0 + (residual / (2.0 * scale)) ** 2)
        )
    return np.polyval(coefficients, s)


def _project_sp45(frame: pd.DataFrame) -> np.ndarray:
    output = frame["sp45_preprojection"].to_numpy(np.float64).copy()
    for well, index in frame.groupby("well", sort=False).groups.items():
        positions = frame.index.get_indexer(index)
        group = frame.iloc[positions].sort_values("row_idx")
        ordered_positions = positions[np.argsort(frame.iloc[positions]["row_idx"].to_numpy())]
        hw_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
        hw = pd.read_csv(hw_path)
        known = hw[hw["TVT_input"].notna()]
        if len(known) < 5:
            continue
        anchor_row = known.iloc[-1]
        anchor = float(anchor_row["TVT_input"]) + float(anchor_row["Z"])
        start_md = float(anchor_row["MD"])
        end_md = float(hw["MD"].iloc[-1])
        rows = group["row_idx"].to_numpy(int)
        md = hw["MD"].to_numpy(np.float64)[rows]
        z = hw["Z"].to_numpy(np.float64)[rows]
        s = (md - start_md) / max(end_md - start_md, 1e-6)
        raw = group["sp45_preprojection"].to_numpy(np.float64)
        surface_fit = _robfit(s, raw + z - anchor, degree=3)
        projected = anchor + surface_fit - z
        output[ordered_positions] = 0.25 * raw + 0.75 * projected
    return output


stage = stage.reset_index(drop=True)
stage["sp45"] = _project_sp45(stage)


# -----------------------------------------------------------------------------
# FLE branch: resume completed folds, then infer any missing fold from existing
# held-out-fold models. Full-training deployment models are deliberately ignored.
# -----------------------------------------------------------------------------
def _read_fle_frame(path: Path, expected_fold: int) -> pd.DataFrame:
    frame = pd.read_pickle(path) if path.suffix.lower() in (".pkl", ".pickle") else pd.read_csv(path)
    if "id" not in frame:
        raise RuntimeError(f"FLE artifact lacks id: {path}")
    column = next(
        (name for name in ("fle_gs13", "fle", "prediction", "tvt") if name in frame),
        None,
    )
    if column is None:
        raise RuntimeError(f"Cannot identify FLE prediction in {path}: {frame.columns.tolist()}")
    output = frame[["id", column]].copy().rename(columns={column: "fle"})
    output["id"] = output["id"].astype(str)
    output["well"] = output["id"].str.rsplit("_", n=1).str[0]
    output["fle"] = pd.to_numeric(output["fle"], errors="raise").astype(np.float64)
    output["fle_model_fold"] = np.int8(expected_fold)
    expected = output["well"].map(dict(zip(folds["well"], folds["fold"])))
    if expected.isna().any() or not (expected.astype(int) == expected_fold).all():
        raise RuntimeError(f"{path} is not a held-out Fold-{expected_fold} prediction")
    if output["id"].duplicated().any() or not np.isfinite(output["fle"]).all():
        raise RuntimeError(f"Invalid FLE predictions in {path}")
    return output.drop(columns="well")


def _load_6p39_fle_definitions() -> None:
    for index in (27, 30, 31, 33, 34, 35, 36):
        source = "".join(_bootstrap_notebook["cells"][index].get("source", []))
        source = source.replace("cache=True", "cache=False")
        exec(compile(source, f"{NOTEBOOK_PATH}::cell_{index}", "exec"), globals())
    CFG.DATA = DATA_ROOT
    CFG.OUT = OUT_DIR
    CFG.n_jobs = int(os.environ.get("ROGII_FLE_FEATURE_JOBS", "2"))
    CFG.SHOW_FIGS = False


def _even_positions(length: int, limit: int) -> np.ndarray:
    if limit <= 0 or length <= limit:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, limit).round().astype(np.int64))


def _fle_feature_columns(train: pd.DataFrame, valid: pd.DataFrame) -> list[str]:
    excluded = {"well", "id", "target", "source_fold", "row_idx", "sample_weight"}
    return [
        column
        for column in train.columns
        if column not in excluded
        and not (column.startswith("likpf_scale_") or column == "likpf_mean")
        and column in valid.columns
        and pd.api.types.is_numeric_dtype(train[column])
    ]


def _fle_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        frame.reindex(columns=columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )


def _fle_one_frame(
    well: str,
    likpf_by_well: dict[str, pd.DataFrame],
    fold_by_well: dict[str, int],
    sample: bool,
) -> pd.DataFrame:
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
        frame = frame.iloc[_even_positions(full_rows, FLE_TRAIN_ROWS_PER_WELL)].copy()
        frame["sample_weight"] = np.float32(full_rows / len(frame))
    else:
        frame["sample_weight"] = np.float32(1.0)
    return frame


def _build_fle_outer_frames(
    outer_fold: int,
    likpf_by_well: dict[str, pd.DataFrame],
    fold_by_well: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_wells = folds.loc[folds["fold"] != outer_fold, "well"].tolist()
    valid_wells = folds.loc[folds["fold"] == outer_fold, "well"].tolist()
    init_imputers(train_wells)
    sequence = [(well, True) for well in train_wells] + [(well, False) for well in valid_wells]
    train_parts, valid_parts, failures = [], [], []

    def build_item(item):
        well, sample = item
        try:
            return well, sample, _fle_one_frame(well, likpf_by_well, fold_by_well, sample), None
        except Exception as exc:
            return well, sample, None, repr(exc)

    started = time.time()
    batch_size = max(4, FLE_FEATURE_JOBS * 4)
    for start in range(0, len(sequence), batch_size):
        batch = sequence[start : start + batch_size]
        results = Parallel(n_jobs=FLE_FEATURE_JOBS, prefer="threads")(
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
        if number % 50 < batch_size or number == len(sequence):
            print(
                f"FLE Fold-{outer_fold} features {number}/{len(sequence)} "
                f"failures={len(failures)} elapsed_min={(time.time()-started)/60:.1f}",
                flush=True,
            )
    if failures:
        pd.DataFrame(failures).to_csv(
            FLE_ROOT / f"fold{outer_fold}_feature_failures.csv", index=False
        )
        valid_failures = [row for row in failures if row["role"] == "valid"]
        if valid_failures:
            raise RuntimeError(f"Fold-{outer_fold} validation feature failures: {valid_failures[:5]}")
        print(f"warning: excluded {len(failures)} failed Fold-{outer_fold} train wells", flush=True)
    return pd.concat(train_parts, ignore_index=True), pd.concat(valid_parts, ignore_index=True)


def _fit_fle_fold(
    outer_fold: int,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, list[dict]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    device, device_name = _device()
    configs = lgb_configs(device)[:FLE_MODEL_CONFIGS]
    model_dir = FLE_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions, reports = [], []
    inner_fold = outer_fold % 5 + 1
    inner_train = train[train["source_fold"] != inner_fold]
    inner_valid = train[train["source_fold"] == inner_fold]
    if inner_train.empty or inner_valid.empty:
        raise RuntimeError(f"Empty nested split outer={outer_fold} inner={inner_fold}")
    print(f"FLE Fold-{outer_fold}: LightGBM device={device_name}", flush=True)

    for config_index, original_params in enumerate(configs):
        params = dict(original_params)
        fit_args = dict(
            X=_fle_matrix(inner_train, columns),
            y=inner_train["target"].to_numpy(np.float32),
            sample_weight=inner_train["sample_weight"].to_numpy(np.float32),
            eval_set=[(
                _fle_matrix(inner_valid, columns),
                inner_valid["target"].to_numpy(np.float32),
            )],
            eval_metric="rmse",
            callbacks=[early_stopping(250, verbose=False), log_evaluation(0)],
        )
        try:
            tune_model = LGBMRegressor(**params)
            tune_model.fit(**fit_args)
        except Exception as exc:
            if "device_type" not in params:
                raise
            print(f"FLE config {config_index} GPU fallback: {exc!r}", flush=True)
            params.pop("device_type", None)
            params.pop("gpu_use_dp", None)
            tune_model = LGBMRegressor(**params)
            tune_model.fit(**fit_args)
        best_iteration = int(tune_model.best_iteration_ or params["n_estimators"])
        del tune_model
        gc.collect()

        final_params = dict(params)
        final_params["n_estimators"] = best_iteration
        model = LGBMRegressor(**final_params)
        model.fit(
            _fle_matrix(train, columns),
            train["target"].to_numpy(np.float32),
            sample_weight=train["sample_weight"].to_numpy(np.float32),
        )
        model_path = model_dir / f"lgb{config_index}_fold{outer_fold}.joblib"
        joblib.dump(model, model_path)
        predictions.append(model.predict(_fle_matrix(valid, columns)).astype(np.float64))
        reports.append(
            {
                "config_index": config_index,
                "inner_validation_fold": inner_fold,
                "best_iteration": best_iteration,
                "model": str(model_path),
            }
        )
        print(
            f"FLE Fold-{outer_fold} model {config_index}: best_iteration={best_iteration}",
            flush=True,
        )
        del model
        gc.collect()
    return np.mean(predictions, axis=0), reports


FLE_FOLDS_TRAINED: list[int] = []


def _infer_missing_fle_folds(missing_folds: list[int]) -> list[pd.DataFrame]:
    _load_6p39_fle_definitions()
    model_root = FLE_ROOT / "models"
    training_required = any(
        not (FLE_ROOT / f"fold{outer_fold}_features.json").is_file()
        or not list(model_root.glob(f"lgb*_fold{outer_fold}.joblib"))
        for outer_fold in missing_folds
    )
    needed_wells = (
        folds["well"].tolist()
        if training_required
        else folds.loc[folds["fold"].isin(missing_folds), "well"].tolist()
    )
    cache_candidates = [
        FLE_ROOT / "likpf_gs13.pkl",
        OUT_DIR / "likpf_gs13_replay.pkl",
    ]
    cache_path = next((path for path in cache_candidates if path.is_file()), None)
    likpf = pd.read_pickle(cache_path) if cache_path else pd.DataFrame()
    if not likpf.empty:
        likpf["id"] = likpf["id"].astype(str)
    cached_wells = set(
        likpf["id"].str.rsplit("_", n=1).str[0].unique()
    ) if not likpf.empty else set()
    missing_cache_wells = [well for well in needed_wells if well not in cached_wells]
    if missing_cache_wells:
        print(
            f"FLE replay: building missing lik-PF cache for {len(missing_cache_wells)} wells",
            flush=True,
        )
        parts = [likpf] if not likpf.empty else []
        failures = []
        for number, well in enumerate(missing_cache_wells, 1):
            try:
                part = _likpf_rows(well, "train")
                if part is None or part.empty:
                    raise RuntimeError("empty lik-PF frame")
                parts.append(part)
            except Exception as exc:
                failures.append({"well": well, "reason": repr(exc)})
            if number % 25 == 0 or number == len(missing_cache_wells):
                print(f"FLE replay lik-PF {number}/{len(missing_cache_wells)} failures={len(failures)}", flush=True)
        if failures:
            pd.DataFrame(failures).to_csv(OUT_DIR / "fle_replay_likpf_failures.csv", index=False)
            raise RuntimeError(f"FLE replay lik-PF failures: {failures[:5]}")
        likpf = pd.concat(parts, ignore_index=True)
        likpf = likpf.drop_duplicates("id", keep="last")
        likpf.to_pickle(OUT_DIR / "likpf_gs13_replay.pkl")
    likpf_by_well = {
        well: group.drop(columns="well", errors="ignore").copy()
        for well, group in likpf.assign(
            well=likpf["id"].str.rsplit("_", n=1).str[0]
        ).groupby("well", sort=False)
    }

    outputs = []
    fold_by_well = dict(zip(folds["well"], folds["fold"]))
    for outer_fold in missing_folds:
        schema_path = FLE_ROOT / f"fold{outer_fold}_features.json"
        model_paths = sorted(model_root.glob(f"lgb*_fold{outer_fold}.joblib"))
        if not schema_path.is_file() or not model_paths:
            print(
                f"FLE Fold-{outer_fold}: no complete held-out artifact; training this fold",
                flush=True,
            )
            train, valid = _build_fle_outer_frames(outer_fold, likpf_by_well, fold_by_well)
            columns = _fle_feature_columns(train, valid)
            if len(columns) < 100:
                raise RuntimeError(f"Only {len(columns)} usable FLE features in Fold-{outer_fold}")
            schema_path.write_text(json.dumps(columns, indent=2), encoding="utf-8")
            learned_delta, model_reports = _fit_fle_fold(
                outer_fold, train, valid, columns
            )
            FLE_FOLDS_TRAINED.append(int(outer_fold))
            prediction = make_prediction(valid, learned_delta, None)
            output = pd.DataFrame(
                {
                    "id": valid["id"].astype(str).to_numpy(),
                    "well": valid["well"].astype(str).to_numpy(),
                    "row_idx": valid["row_idx"].to_numpy(np.int32),
                    "fold": np.int8(outer_fold),
                    "tvt_true": (
                        valid["last_known_tvt"].to_numpy(np.float64)
                        + valid["target"].to_numpy(np.float64)
                    ),
                    "fle_gs13": prediction,
                    "learned_delta": learned_delta,
                    "likpf_scale_5": valid["likpf_scale_5"].to_numpy(np.float32),
                }
            )
            output_path = FLE_ROOT / f"fold{outer_fold}_oof.pkl"
            output.to_pickle(output_path)
            (FLE_ROOT / f"fold{outer_fold}_report.json").write_text(
                json.dumps(
                    {
                        "outer_fold": outer_fold,
                        "train_wells": int(train["well"].nunique()),
                        "valid_wells": int(valid["well"].nunique()),
                        "train_rows": int(len(train)),
                        "valid_rows": int(len(valid)),
                        "features": len(columns),
                        "models": model_reports,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            outputs.append(_read_fle_frame(output_path, outer_fold))
            del train, valid, learned_delta, prediction, output
            gc.collect()
            continue
        columns = json.loads(schema_path.read_text(encoding="utf-8"))
        train_wells = folds.loc[folds["fold"] != outer_fold, "well"].tolist()
        valid_wells = folds.loc[folds["fold"] == outer_fold, "well"].tolist()
        init_imputers(train_wells)
        parts = []
        failures = []
        for number, well in enumerate(valid_wells, 1):
            try:
                horizontal = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
                typewell = DATA_ROOT / "train" / f"{well}__typewell.csv"
                frame = build_well(str(horizontal), str(typewell), is_train=True)
                if frame is None or frame.empty:
                    raise RuntimeError("empty build_well result")
                frame = add_likpf_features(frame, likpf_by_well[well])
                frame["source_fold"] = np.int8(fold_by_well[well])
                frame["row_idx"] = frame["id"].str.rsplit("_", n=1).str[1].astype(np.int32)
                parts.append(frame)
            except Exception as exc:
                failures.append({"fold": outer_fold, "well": well, "reason": repr(exc)})
            if number % 25 == 0 or number == len(valid_wells):
                print(f"FLE replay Fold-{outer_fold} features {number}/{len(valid_wells)} failures={len(failures)}", flush=True)
        if failures:
            pd.DataFrame(failures).to_csv(
                OUT_DIR / f"fle_replay_fold{outer_fold}_failures.csv", index=False
            )
        if not parts:
            raise RuntimeError(f"No valid FLE features for Fold-{outer_fold}")
        valid = pd.concat(parts, ignore_index=True)
        matrix = (
            valid.reindex(columns=columns)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(np.float32)
        )
        model_predictions = []
        for path in model_paths:
            model = joblib.load(path)
            expected_features = getattr(model, "n_features_in_", matrix.shape[1])
            if int(expected_features) != matrix.shape[1]:
                raise RuntimeError(
                    f"Feature mismatch for {path}: model={expected_features}, schema={matrix.shape[1]}"
                )
            model_predictions.append(np.asarray(model.predict(matrix), dtype=np.float64))
        learned_delta = np.mean(model_predictions, axis=0)
        prediction = make_prediction(valid, learned_delta, None)
        output = pd.DataFrame(
            {
                "id": valid["id"].astype(str).to_numpy(),
                "fle_gs13": prediction,
                "fold": np.int8(outer_fold),
            }
        )
        replay_path = OUT_DIR / f"fle_fold{outer_fold}_replayed_oof.pkl"
        output.to_pickle(replay_path)
        outputs.append(_read_fle_frame(replay_path, outer_fold))
        print(
            f"FLE replay Fold-{outer_fold}: rows={len(output):,} wells={valid['well'].nunique()} "
            f"models={len(model_paths)}",
            flush=True,
        )
        del valid, matrix, model_predictions, learned_delta, output
        gc.collect()
    return outputs


fle_parts = []
missing_folds = []
for outer_fold in range(1, 6):
    candidates = [
        FLE_ROOT / f"fold{outer_fold}_oof.pkl",
        OUT_DIR / f"fle_fold{outer_fold}_replayed_oof.pkl",
    ]
    completed = next((path for path in candidates if path.is_file()), None)
    if completed:
        print(f"FLE Fold-{outer_fold}: reuse {completed}", flush=True)
        fle_parts.append(_read_fle_frame(completed, outer_fold))
    else:
        missing_folds.append(outer_fold)
if missing_folds:
    print(f"FLE folds requiring replay or fold training: {missing_folds}", flush=True)
    fle_parts.extend(_infer_missing_fle_folds(missing_folds))

fle_stage = pd.concat(fle_parts, ignore_index=True)
if fle_stage["id"].duplicated().any():
    raise RuntimeError("Combined FLE OOF contains duplicate ids")
stage = stage.merge(fle_stage, on="id", how="inner", validate="one_to_one")
if len(stage) / len(ridge_stage) < MIN_COVERAGE:
    raise RuntimeError("FLE alignment fell below the required coverage")

stage["frontier_6p39_core"] = SP45_WEIGHT * stage["sp45"] + FLE_WEIGHT * stage["fle"]
stage["fold"] = stage["well"].map(dict(zip(folds["well"], folds["fold"]))).astype("Int8")


def _metrics(column: str) -> dict:
    error = stage[column].to_numpy(np.float64) - stage["truth"].to_numpy(np.float64)
    bywell = pd.DataFrame({"well": stage["well"], "error2": error * error}).groupby("well")["error2"].mean()
    return {
        "endpoint": column,
        "rows": int(len(error)),
        "wells": int(stage["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mean_well_rmse": float(np.mean(np.sqrt(bywell))),
        "median_well_rmse": float(np.median(np.sqrt(bywell))),
        "p95_well_rmse": float(np.quantile(np.sqrt(bywell), 0.95)),
    }


endpoint_columns = [
    "ridge_pp",
    selector_column,
    "sp45_preprojection",
    "sp45",
    "fle",
    "frontier_6p39_core",
]
summary = pd.DataFrame([_metrics(column) for column in endpoint_columns])
summary.to_csv(OUT_DIR / "endpoint_summary.csv", index=False)

output_columns = [
    "id", "well", "row_idx", "fold", "truth", "ridge_pp", selector_column,
    "sp45_preprojection", "sp45", "fle", "fle_model_fold", "frontier_6p39_core",
]
stage[output_columns].to_parquet(
    OUT_DIR / "frontier_6p39_core_oof.parquet", index=False, compression="zstd"
)

verdict = {
    "experiment": EXPERIMENT,
    "stage": "existing_model_6p39_core_oof_reconstruction",
    "notebook_sha256": _sha256(NOTEBOOK_PATH),
    "formula": "0.60*project(0.30*ridge_pp+0.70*selector)+0.40*FLE",
    "coverage": {
        "rows": int(len(stage)),
        "wells": int(stage["well"].nunique()),
        "alignment_vs_ridge": float(len(stage) / len(ridge_stage)),
    },
    "training": {
        "base_endpoint_fit_calls": 0,
        "fle_folds_trained": FLE_FOLDS_TRAINED,
        "fle_fit_calls": int(len(FLE_FOLDS_TRAINED) * FLE_MODEL_CONFIGS),
        "fle_train_rows_per_well": FLE_TRAIN_ROWS_PER_WELL,
        "fle_model_configs": FLE_MODEL_CONFIGS,
        "meta_ridge_closed_form_replays": int(inventory.get("meta_ridge_fits", 0)),
    },
    "endpoints": summary.to_dict(orient="records"),
    "excluded_deployment_layers": [
        "Q0522 hidden-id correction",
        "guarded same-well hidden override",
        "test-only model-package gate",
    ],
    "warning": (
        "This is the reproducible model core of the 6.39 notebook, not a claim "
        "that its hidden-id deployment overlays have an OOF analogue."
    ),
    "output": str(OUT_DIR / "frontier_6p39_core_oof.parquet"),
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "free_gb": shutil.disk_usage(ROOT).free / (1024**3),
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
(OUT_DIR / "artifact_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")

print("\n6P39 CORE OOF SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR, flush=True)
