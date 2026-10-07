"""V97: paired upstream PF repeatability gate for the 6.342 lineage.

Run this cell immediately after sub, cv_final = main() in the current
Kaggle/AutoDL mainline, or let its notebook bootstrap load the definitions.
It measures only the stochastic ANCC/Z PF contribution before CP8, Neighbor,
V69 coarse, and V76 adaptive dose. It does not claim to score the final
6.342 endpoint.
"""

from pathlib import Path
import json
import os
import time
import zlib

import joblib
import numpy as np
import pandas as pd
from numba import njit


EXPERIMENT_P5 = "mainline_pf_repeatability_v97"
OUT_DIR_P5 = Path("/root/autodl-tmp/oof_exports") / EXPERIMENT_P5
OUT_DIR_P5.mkdir(parents=True, exist_ok=True)

N_WELLS_P5 = 24
N_SEEDS_P5 = 6
SEEDS_P5 = tuple(range(N_SEEDS_P5))
OOF_CANDIDATES_P5 = (
    Path("/root/autodl-tmp/oof_exports/frontier_6p377_full_fivefold_lineage_v66/incumbent_6p377_full_fivefold_oof.parquet"),
    Path("/root/autodl-tmp/oof_exports/frontier_6p39_full_oof_cp_increment_v18/full_oof_cp_endpoints.parquet"),
    Path("/root/autodl-tmp/full_oof_cp_endpoints.parquet"),
)


def _find_raw_data_p5():
    explicit = os.environ.get("ROGII_DATA", "").strip()
    if explicit and (Path(explicit) / "train").exists():
        return Path(explicit)
    roots = (Path("/root/autodl-tmp"), Path("/kaggle/input"), Path.cwd())
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for sample in root.rglob("sample_submission.csv"):
            if (sample.parent / "train").exists():
                count = sum(1 for _ in (sample.parent / "train").glob("*__horizontal_well.csv"))
                candidates.append((count, sample.parent))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _find_mainline_notebook_p5():
    explicit = os.environ.get("ROGII_MAINLINE_NOTEBOOK", "").strip()
    candidates = [
        Path(explicit) if explicit else None,
        Path("/root/autodl-tmp/rogii-cp8-neighbor.ipynb"),
        Path.cwd() / "rogii-cp8-neighbor.ipynb",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    for root in (Path("/root/autodl-tmp"), Path.cwd()):
        if root.exists():
            matches = list(root.rglob("rogii-cp8-neighbor.ipynb"))
            if matches:
                return matches[0]
    return None


def _discover_model_roots_p5():
    roots = []
    for root in (Path("/root/autodl-tmp"), Path("/kaggle/input"), Path.cwd()):
        if not root.exists():
            continue
        for feature_file in root.rglob("features.json"):
            directory = feature_file.parent
            if list(directory.glob("lgb*.pkl")):
                try:
                    names = json.loads(feature_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                signature = sum(
                    any(str(name).startswith(prefix) for name in names)
                    for prefix in ("pf_ancc", "likpf_", "beam_", "sc8_", "tvtF_")
                )
                roots.append((signature, len(names), directory))
    roots.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return list(dict.fromkeys(item[2] for item in roots))


def _bootstrap_mainline_p5():
    needed = {"CFG", "build_well", "_likpf_rows", "add_likpf_features", "make_prediction", "_find_models"}
    if not needed.difference(globals()):
        return None
    notebook_path = _find_mainline_notebook_p5()
    if notebook_path is None:
        return None
    data_root = _find_raw_data_p5()
    if data_root is not None:
        os.environ["ROGII_DATA"] = str(data_root)
    model_roots = _discover_model_roots_p5()
    if model_roots:
        globals()["LEARNED_MODEL_ROOTS"] = tuple(str(path) for path in model_roots)

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell_index in (34, 37, 38, 40, 41, 42, 43):
        source = "".join(notebook["cells"][cell_index].get("source", []))
        # Dynamically compiled notebook cells have no physical .py locator, so
        # Numba's disk cache cannot be enabled for their decorated functions.
        source = source.replace("cache=True", "cache=False")
        exec(compile(source, f"{notebook_path}:cell_{cell_index}", "exec"), globals())
    cell_44 = "".join(notebook["cells"][44].get("source", []))
    marker = "def main():"
    if marker not in cell_44:
        raise RuntimeError("Cannot locate main() marker in mainline cell 44")
    exec(compile(cell_44.split(marker, 1)[0], f"{notebook_path}:cell_44_prefix", "exec"), globals())
    return str(notebook_path)


BOOTSTRAP_NOTEBOOK_P5 = _bootstrap_mainline_p5()

required_p5 = {
    "CFG", "build_well", "_likpf_rows", "add_likpf_features",
    "make_prediction", "_find_models", "_FI", "_DI", "init_imputers",
}
missing_p5 = sorted(required_p5.difference(globals()))
if missing_p5:
    raise RuntimeError(
        "Missing mainline definitions: " + repr(missing_p5) + ". Upload "
        "rogii-cp8-neighbor.ipynb to /root/autodl-tmp, or set "
        "ROGII_MAINLINE_NOTEBOOK to its path."
    )

if _FI is None or _DI is None:
    bootstrap_train_wells_p5 = sorted(
        path.stem.replace("__horizontal_well", "")
        for path in (Path(CFG.DATA) / "train").glob("*__horizontal_well.csv")
    )
    if len(bootstrap_train_wells_p5) < 700:
        raise RuntimeError(f"Raw data discovery found only {len(bootstrap_train_wells_p5)} train wells at {CFG.DATA}")
    print(f"P5 initializing fold-blind inference imputers from {len(bootstrap_train_wells_p5)} train wells", flush=True)
    init_imputers(bootstrap_train_wells_p5)


@njit(cache=True)
def _seed_numba_p5(seed):
    np.random.seed(seed)


def _stable_seed_p5(well, seed):
    token = zlib.crc32(str(well).encode("ascii")) & 0x7FFFFFFF
    return int((token + 1000003 * int(seed)) % 2147483647)


def _rmse_p5(prediction, truth):
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _metrics_p5(frame, prediction, label):
    error = prediction - frame["truth"].to_numpy(np.float64)
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    by_well = pd.DataFrame({"well": frame["well"], "e2": error ** 2}).groupby("well")["e2"].mean()
    return {
        "variant": label,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(np.sqrt(by_well.to_numpy()))),
        "p95_well_rmse": float(np.quantile(np.sqrt(by_well.to_numpy()), 0.95)),
        "far_gt2000_rmse": float(np.sqrt(np.mean(error[far] ** 2))) if far.any() else float("nan"),
    }


def _select_wells_p5():
    oof_path = next((path for path in OOF_CANDIDATES_P5 if path.exists()), None)
    train_wells = sorted(
        path.stem.replace("__horizontal_well", "")
        for path in (Path(CFG.DATA) / "train").glob("*__horizontal_well.csv")
    )
    if oof_path is None:
        rng = np.random.default_rng(20260728)
        chosen = sorted(rng.choice(train_wells, min(N_WELLS_P5, len(train_wells)), replace=False).tolist())
        return chosen, None

    columns = pd.read_parquet(oof_path).columns.tolist()
    truth_column = next((c for c in ("tvt_true", "truth", "TVT") if c in columns), None)
    pred_column = next(
        (c for c in (
            "incumbent_6p377_replay",
            "full_parent_plus_raw_cp_crossfit", "full_cp8_ordered",
            "full_parent_matched_vp_replay", "core",
        ) if c in columns),
        None,
    )
    if truth_column is None or pred_column is None or "well" not in columns:
        raise RuntimeError(f"Cannot rank wells from {oof_path}: {columns}")
    oof = pd.read_parquet(oof_path, columns=["well", truth_column, pred_column])
    oof["e2"] = (oof[pred_column].astype(float) - oof[truth_column].astype(float)) ** 2
    difficulty = oof.groupby("well")["e2"].mean().pow(0.5).sort_values()
    difficulty = difficulty[difficulty.index.astype(str).isin(train_wells)]
    groups = np.array_split(difficulty.index.to_numpy(), 3)
    per_group = N_WELLS_P5 // 3
    chosen = []
    for group in groups:
        positions = np.linspace(0, len(group) - 1, per_group).round().astype(int)
        chosen.extend(str(group[position]) for position in positions)
    return chosen, str(oof_path)


def _load_models_p5():
    models_dir = _find_models()
    if models_dir is None:
        raise RuntimeError("No pretrained mainline model directory found")
    models_dir = Path(models_dir)
    feature_names = json.loads((models_dir / "features.json").read_text(encoding="utf-8"))
    models = [joblib.load(path) for path in sorted(models_dir.glob("lgb*.pkl"))]
    if not models:
        raise RuntimeError(f"No lgb*.pkl files in {models_dir}")
    return models_dir, feature_names, models


def _one_prediction_p5(well, seed, feature_names, models, likpf_cache):
    train_dir = Path(CFG.DATA) / "train"
    horizontal = train_dir / f"{well}__horizontal_well.csv"
    typewell = train_dir / f"{well}__typewell.csv"
    _seed_numba_p5(_stable_seed_p5(well, seed))
    features = build_well(str(horizontal), str(typewell), True)
    if features is None or features.empty:
        raise RuntimeError(f"{well}: build_well returned no rows")
    if well not in likpf_cache:
        likpf_cache[well] = _likpf_rows(well, "train")
    features = add_likpf_features(features, likpf_cache[well]).reset_index(drop=True)
    for column in feature_names:
        if column not in features:
            features[column] = 0.0
    matrix = features[feature_names].to_numpy(np.float32)
    model_delta = np.mean([model.predict(matrix) for model in models], axis=0)
    prediction = make_prediction(features, model_delta, None).astype(np.float64)
    truth = (
        features["last_known_tvt"].to_numpy(np.float64)
        + features["target"].to_numpy(np.float64)
    )
    output = features[["well", "id", "md_since", "pf_ancc", "pf_z"]].copy()
    output["truth"] = truth
    output["prediction"] = prediction
    output["seed"] = int(seed)
    return output


started_p5 = time.time()
selected_wells_p5, ranking_source_p5 = _select_wells_p5()
models_dir_p5, feature_names_p5, models_p5 = _load_models_p5()
likpf_cache_p5 = {}

# First prove that explicit Numba seeding actually controls the stochastic kernels.
control_a_p5 = _one_prediction_p5(selected_wells_p5[0], 0, feature_names_p5, models_p5, likpf_cache_p5)
control_b_p5 = _one_prediction_p5(selected_wells_p5[0], 0, feature_names_p5, models_p5, likpf_cache_p5)
replay_max_gap_p5 = float(np.max(np.abs(
    control_a_p5["prediction"].to_numpy() - control_b_p5["prediction"].to_numpy()
)))
if replay_max_gap_p5 > 1e-7:
    raise RuntimeError(f"Seed control is not deterministic: max gap={replay_max_gap_p5}")

parts_p5 = []
failures_p5 = []
for number_p5, well_p5 in enumerate(selected_wells_p5, 1):
    for seed_p5 in SEEDS_P5:
        try:
            parts_p5.append(
                _one_prediction_p5(well_p5, seed_p5, feature_names_p5, models_p5, likpf_cache_p5)
            )
        except Exception as error_p5:
            failures_p5.append({"well": well_p5, "seed": seed_p5, "reason": repr(error_p5)})
    print(f"P5 repeatability {number_p5}/{len(selected_wells_p5)} failures={len(failures_p5)}", flush=True)

if failures_p5:
    pd.DataFrame(failures_p5).to_csv(OUT_DIR_P5 / "failures.csv", index=False)
if not parts_p5:
    raise RuntimeError(f"No successful P5 predictions: {failures_p5[:5]}")

long_p5 = pd.concat(parts_p5, ignore_index=True)
coverage_p5 = long_p5.groupby("well")["seed"].nunique()
complete_wells_p5 = coverage_p5[coverage_p5 == N_SEEDS_P5].index
long_p5 = long_p5[long_p5["well"].isin(complete_wells_p5)].copy()
if len(complete_wells_p5) < 20:
    raise RuntimeError(f"Only {len(complete_wells_p5)} complete wells")

keys_p5 = ["well", "id", "md_since", "truth"]
wide_prediction_p5 = long_p5.pivot(index=keys_p5, columns="seed", values="prediction").sort_index()
wide_pf_p5 = long_p5.pivot(index=keys_p5, columns="seed", values="pf_ancc").sort_index()
base_frame_p5 = wide_prediction_p5.reset_index()[keys_p5]

summary_rows_p5 = []
for seed_p5 in SEEDS_P5:
    summary_rows_p5.append(
        _metrics_p5(base_frame_p5, wide_prediction_p5[seed_p5].to_numpy(), f"seed_{seed_p5}")
    )
for count_p5 in (2, 3, 4, N_SEEDS_P5):
    prediction_p5 = wide_prediction_p5[list(SEEDS_P5[:count_p5])].mean(axis=1).to_numpy()
    summary_rows_p5.append(_metrics_p5(base_frame_p5, prediction_p5, f"mean_{count_p5}_seeds"))
summary_p5 = pd.DataFrame(summary_rows_p5)

seed_rmse_p5 = summary_p5[summary_p5["variant"].str.startswith("seed_")]["rmse"]
ensemble_rmse_p5 = float(summary_p5.loc[summary_p5["variant"] == f"mean_{N_SEEDS_P5}_seeds", "rmse"].iloc[0])
median_seed_rmse_p5 = float(seed_rmse_p5.median())
best_seed_rmse_p5 = float(seed_rmse_p5.min())

prediction_std_p5 = wide_prediction_p5.std(axis=1, ddof=0).to_numpy()
pf_std_p5 = wide_pf_p5.std(axis=1, ddof=0).to_numpy()
far_p5 = base_frame_p5["md_since"].to_numpy(np.float64) > 2000.0

pair_rms_p5 = []
error_corr_p5 = []
truth_p5 = base_frame_p5["truth"].to_numpy(np.float64)
for left_p5 in SEEDS_P5:
    for right_p5 in SEEDS_P5:
        if right_p5 <= left_p5:
            continue
        left_values_p5 = wide_prediction_p5[left_p5].to_numpy()
        right_values_p5 = wide_prediction_p5[right_p5].to_numpy()
        pair_rms_p5.append(_rmse_p5(left_values_p5, right_values_p5))
        error_corr_p5.append(float(np.corrcoef(left_values_p5 - truth_p5, right_values_p5 - truth_p5)[0, 1]))

verdict_p5 = {
    "experiment": EXPERIMENT_P5,
    "stage": "paired_seed_repeatability_before_all_downstream_layers",
    "current_deployed_reference": "V69 All773 coarse + V76 adaptive dose, LB 6.342",
    "scope": (
        "learned trajectory core only; not CP8, Neighbor, V69, V76, "
        "or the final 6.342 endpoint"
    ),
    "ranking_source": ranking_source_p5,
    "models_dir": str(models_dir_p5),
    "wells_requested": int(len(selected_wells_p5)),
    "wells_complete": int(len(complete_wells_p5)),
    "seeds": list(SEEDS_P5),
    "deterministic_replay_max_gap": replay_max_gap_p5,
    "median_single_seed_rmse": median_seed_rmse_p5,
    "best_single_seed_rmse": best_seed_rmse_p5,
    "mean_seed_rmse": ensemble_rmse_p5,
    "mean_gain_vs_median_seed": median_seed_rmse_p5 - ensemble_rmse_p5,
    "mean_gain_vs_best_seed": best_seed_rmse_p5 - ensemble_rmse_p5,
    "prediction_stochastic_std_mean": float(np.mean(prediction_std_p5)),
    "prediction_stochastic_std_p95": float(np.quantile(prediction_std_p5, 0.95)),
    "prediction_stochastic_std_far_mean": float(np.mean(prediction_std_p5[far_p5])) if far_p5.any() else float("nan"),
    "pf_ancc_stochastic_std_mean": float(np.mean(pf_std_p5)),
    "pairwise_prediction_rms_mean": float(np.mean(pair_rms_p5)),
    "pairwise_error_correlation_mean": float(np.mean(error_corr_p5)),
    "checks": {
        "explicit_seed_is_deterministic": replay_max_gap_p5 <= 1e-7,
        "prediction_stochastic_std_ge_0p25": float(np.mean(prediction_std_p5)) >= 0.25,
        "mean_gain_vs_median_seed_ge_0p05": median_seed_rmse_p5 - ensemble_rmse_p5 >= 0.05,
        "mean_not_worse_than_best_by_gt_0p03": ensemble_rmse_p5 - best_seed_rmse_p5 <= 0.03,
    },
    "elapsed_minutes": (time.time() - started_p5) / 60.0,
}
verdict_p5["strict_pass"] = bool(all(verdict_p5["checks"].values()))
verdict_p5["next"] = (
    "PASS -> run a matched downstream-survival audit before any full OOF rebuild"
    if verdict_p5["strict_pass"]
    else "FAIL -> keep one stable per-well seed; close PF averaging"
)

summary_p5.to_csv(OUT_DIR_P5 / "seed_ensemble_summary.csv", index=False)
long_p5[["well", "id", "md_since", "truth", "seed", "prediction", "pf_ancc", "pf_z"]].to_parquet(
    OUT_DIR_P5 / "paired_seed_predictions.parquet", index=False
)
(OUT_DIR_P5 / "result.json").write_text(json.dumps(verdict_p5, indent=2), encoding="utf-8")

print("\nP5 SEED ENSEMBLE SUMMARY")
print(summary_p5.to_string(index=False))
print("\nP5 FINAL VERDICT")
print(json.dumps(verdict_p5, indent=2), flush=True)
print("saved:", OUT_DIR_P5)
