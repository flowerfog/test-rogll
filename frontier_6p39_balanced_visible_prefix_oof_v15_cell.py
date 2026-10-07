"""Matched replay of the 6.39 notebook's balanced visible-prefix overlay.

Default execution is a fixed 24-well Fold-3 discovery gate. Set
``ROGII_VP_STAGE=full`` to process all five folds after the discovery result is
reviewed. Candidate generation and dosing are extracted from the exact public
notebook; no parameters are tuned in this cell.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


EXPERIMENT = "frontier_6p39_balanced_visible_prefix_oof_v15"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT
CACHE_DIR = OUT_DIR / "well_cache"
PARENT_PATH = (
    OOF_ROOT
    / "frontier_6p39_cp8_modelpkg_replay_v14"
    / "modelpkg_replay_oof.parquet"
)
FOLD_PATH = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv"
)
EXPECTED_NOTEBOOK_SHA256 = (
    "da460516b1aeda7404b1b1f2abb5702c418c31f64f4088ecaf81ca08fc795438"
)

STAGE = os.environ.get("ROGII_VP_STAGE", "discovery").strip().lower()
if STAGE not in {"discovery", "full"}:
    raise ValueError("ROGII_VP_STAGE must be discovery or full")
DISCOVERY_FOLD = 3
DISCOVERY_WELLS = 24
PROFILE = "balanced"
CUT_FRACS = (0.50, 0.65, 0.75)
CAL_SEEDS = 24
FINAL_SEEDS = 48
PARTICLES = 350
REPEAT_GUARD = os.environ.get("ROGII_VP_REPEAT_GUARD", "1") == "1"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_notebook() -> Path:
    configured = os.environ.get("ROGII_FRONTIER_NOTEBOOK")
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [
            ROOT / "rogii-public-score-frontier-lab-visuals.ipynb",
            ROOT / "working_note_kaggle.ipynb",
        ]
    )
    candidates.extend(ROOT.glob("**/rogii-public-score-frontier-lab-visuals.ipynb"))
    for path in candidates:
        if path.exists() and _sha256(path) == EXPECTED_NOTEBOOK_SHA256:
            return path.resolve()
    found = [(str(path), _sha256(path)) for path in candidates if path.exists()]
    raise FileNotFoundError(
        "Exact 6.39 notebook not found. Upload "
        "rogii-public-score-frontier-lab-visuals.ipynb to /root/autodl-tmp. "
        f"Observed: {found[:5]}"
    )


def _find_data_root() -> Path:
    configured = os.environ.get("ROGII_DATA_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [
            ROOT / "rogii-wellbore-geology-prediction",
            ROOT / "data",
            Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        ]
    )
    for pattern in ("**/train/*__horizontal_well.csv", "**/train/*__typewell.csv"):
        for hit in ROOT.glob(pattern):
            candidates.append(hit.parent.parent)
    seen = set()
    for path in candidates:
        try:
            path = path.resolve()
        except Exception:
            continue
        if path in seen:
            continue
        seen.add(path)
        train = path / "train"
        if train.exists() and next(train.glob("*__horizontal_well.csv"), None):
            return path
    raise FileNotFoundError(
        "Cannot locate raw competition data. Set ROGII_DATA_ROOT to the directory "
        "containing train/*.csv."
    )


def _fold_map(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path)
    well_col = next((name for name in ("well", "well_id", "Well") if name in frame), None)
    fold_col = next((name for name in ("fold", "Fold", "outer_fold") if name in frame), None)
    if well_col is None or fold_col is None:
        raise RuntimeError(f"Cannot identify well/fold columns in {path}: {list(frame)}")
    result = dict(zip(frame[well_col].astype(str), frame[fold_col].astype(int)))
    if len(result) != 773 or set(result.values()) != {1, 2, 3, 4, 5}:
        raise RuntimeError(f"Unexpected fold metadata: wells={len(result)} folds={set(result.values())}")
    return result


def _even_sample(values: list[str], count: int) -> list[str]:
    values = sorted(values)
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[index] for index in indices]


def _load_exact_notebook_definitions(notebook_path: Path, data_root: Path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    source5 = "".join(cells[5].get("source", []))
    source8 = "".join(cells[8].get("source", []))
    source44 = "".join(cells[44].get("source", []))
    if not all(token in source44 for token in (
        "def _gold_calibrate_well", "def _gold_profile_output", "'balanced'"
    )):
        raise RuntimeError("Notebook cell 44 is not the expected visible-prefix implementation")

    class CFG:
        dataset_path = data_root

    namespace = {
        "np": np,
        "pd": pd,
        "Path": Path,
        "cKDTree": cKDTree,
        "joblib": joblib,
        "CFG": CFG,
    }
    exec(compile(source5, str(notebook_path) + "::cell5", "exec"), namespace)

    # The surface candidates need only these exact cell-8 definitions. Avoid
    # executing the later dataset/model construction in that cell.
    wanted = {
        "robust_slope",
        "seg_b_well",
        "FormationPlaneKNN",
        "DenseANCCImputer",
    }
    tree = ast.parse(source8, filename=str(notebook_path) + "::cell8")
    selected_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]
    found = {node.name for node in selected_nodes}
    if found != wanted:
        raise RuntimeError(f"Missing cell-8 definitions: {sorted(wanted - found)}")
    namespace.update(
        {
            "FORMATIONS": ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"],
            "PLANE_K": 10,
            "DENSE_SPW": 60,
            "DENSE_K": 20,
        }
    )
    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(notebook_path) + "::cell8_selected", "exec"), namespace)

    # Execute imports, constants and function definitions, but never the cell's
    # automatic Kaggle submission loop.
    prefix = source44.split("if not _GOLD_ENABLE:", 1)[0]
    old_env = {
        key: os.environ.get(key)
        for key in (
            "ROGII_GOLD_PROFILE",
            "ROGII_GOLD_INCLUDE_PF",
            "ROGII_GOLD_CAL_SEEDS",
            "ROGII_GOLD_FINAL_SEEDS",
            "ROGII_GOLD_PARTICLES",
            "ROGII_GOLD_CUT_FRACS",
            "ROGII_GOLD_SKIP_BIMODAL",
            "ROGII_GOLD_CONTACT_OVERRIDE",
            "ROGII_GOLD_FINAL_SELECTION",
        )
    }
    os.environ.update(
        {
            "ROGII_GOLD_PROFILE": PROFILE,
            "ROGII_GOLD_INCLUDE_PF": "1",
            "ROGII_GOLD_CAL_SEEDS": str(CAL_SEEDS),
            "ROGII_GOLD_FINAL_SEEDS": str(FINAL_SEEDS),
            "ROGII_GOLD_PARTICLES": str(PARTICLES),
            "ROGII_GOLD_CUT_FRACS": ",".join(map(str, CUT_FRACS)),
            "ROGII_GOLD_SKIP_BIMODAL": "0",
            "ROGII_GOLD_CONTACT_OVERRIDE": "0",
            "ROGII_GOLD_FINAL_SELECTION": "profile",
        }
    )
    try:
        exec(compile(prefix, str(notebook_path) + "::cell44_defs", "exec"), namespace)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # In Kaggle this branch exists only for the few test IDs also present in
    # train. A train-well replay would otherwise expose every validation well's
    # complete TVT through its own train file, which is not representative of
    # hidden test wells. Guarded same-well overlap is audited separately.
    namespace["_gold_contact_candidate"] = lambda wid, hw, root: {}
    return namespace


def _metric(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    prediction = np.asarray(prediction, np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    error = prediction - truth
    if len(error) != len(frame) or not np.isfinite(error).all():
        raise RuntimeError(f"Invalid prediction for {endpoint}")
    work = pd.DataFrame({"well": frame["well"].to_numpy(), "se": error * error})
    well_rmse = np.sqrt(work.groupby("well", sort=False)["se"].mean())
    row = {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p75_well_rmse": float(well_rmse.quantile(0.75)),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "max_well_rmse": float(well_rmse.max()),
    }
    distance = frame["md_since"].to_numpy(np.float64)
    for name, low, high in (
        ("near_0_500", 0.0, 500.0),
        ("mid_500_1000", 500.0, 1000.0),
        ("mid_1000_2000", 1000.0, 2000.0),
        ("far_gt2000", 2000.0, np.inf),
    ):
        mask = (distance > low) & (distance <= high)
        row[f"{name}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else np.nan
    return row


def _worst_twenty(frame: pd.DataFrame, prediction: np.ndarray, endpoint: str) -> dict:
    error = np.asarray(prediction, np.float64) - frame["tvt_true"].to_numpy(np.float64)
    work = pd.DataFrame({"well": frame["well"].to_numpy(), "se": error * error})
    per_well = work.groupby("well", sort=False)["se"].mean().sort_values(ascending=False)
    count = max(1, int(np.ceil(0.20 * len(per_well))))
    wells = set(per_well.head(count).index.astype(str))
    mask = frame["well"].astype(str).isin(wells).to_numpy()
    return {
        "endpoint": endpoint,
        "rows": int(mask.sum()),
        "wells": int(count),
        "exact_sse": float(np.dot(error[mask], error[mask])),
        "rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
    }


started = time.time()
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
for path in (PARENT_PATH, FOLD_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

notebook_path = _find_notebook()
data_root = _find_data_root()
fold_by_well = _fold_map(FOLD_PATH)
parent = pd.read_parquet(PARENT_PATH)
required = {
    "id", "well", "row_idx", "fold", "tvt_true", "md_since",
    "cp8_midpoint_modelpkg",
}
missing = sorted(required.difference(parent.columns))
if missing:
    raise RuntimeError(f"Parent OOF missing columns: {missing}")
if parent["id"].duplicated().any():
    raise RuntimeError("Duplicate parent ids")
parent["well"] = parent["well"].astype(str)

if STAGE == "discovery":
    selected_wells = _even_sample(
        [well for well, fold in fold_by_well.items() if fold == DISCOVERY_FOLD],
        DISCOVERY_WELLS,
    )
else:
    selected_wells = sorted(fold_by_well)
parent = parent[parent["well"].isin(selected_wells)].copy().reset_index(drop=True)
if parent["well"].nunique() != len(selected_wells):
    raise RuntimeError("Parent OOF does not cover all selected wells")

namespace = _load_exact_notebook_definitions(notebook_path, data_root)
variants = namespace["_gold_variant_grid"]()
reports = []
candidate_by_id: dict[str, float] = {}
cut_rows = []
repeat_guard = None


def _run_one(well: str, repeat: bool = False):
    fold = fold_by_well[well]
    cache_path = CACHE_DIR / f"{well}.joblib"
    if cache_path.exists() and not repeat:
        payload = joblib.load(cache_path)
        if payload.get("recipe") == {
            "cache_version": 2,
            "notebook_sha256": EXPECTED_NOTEBOOK_SHA256,
            "profile": PROFILE,
            "cuts": CUT_FRACS,
            "cal_seeds": CAL_SEEDS,
            "final_seeds": FINAL_SEEDS,
            "particles": PARTICLES,
            "fold": fold,
        }:
            return payload

    hw_path = data_root / "train" / f"{well}__horizontal_well.csv"
    tw_path = data_root / "train" / f"{well}__typewell.csv"
    if not hw_path.exists() or not tw_path.exists():
        raise FileNotFoundError(f"Missing raw files for {well}")
    hw = pd.read_csv(hw_path)
    tw = pd.read_csv(tw_path)
    if not {"TVT", "TVT_input", "MD", "Z", "GR"}.issubset(hw.columns):
        raise RuntimeError(f"{well}: raw horizontal schema incomplete")

    well_parent = parent[parent["well"].eq(well)].sort_values("row_idx")
    rows = well_parent["row_idx"].to_numpy(np.int64)
    hidden = hw["TVT_input"].isna().to_numpy()
    if not np.all((rows >= 0) & (rows < len(hw))) or not hidden[rows].all():
        raise RuntimeError(f"{well}: parent rows do not match the official hidden suffix")
    truth_gap = np.abs(
        hw["TVT"].to_numpy(np.float64)[rows]
        - well_parent["tvt_true"].to_numpy(np.float64)
    )
    if not np.isfinite(truth_gap).all() or float(truth_gap.max()) > 0.01:
        raise RuntimeError(f"{well}: truth alignment gap {truth_gap.max()}")

    report = namespace["_gold_calibrate_well"](well, hw, tw, data_root, variants)
    if report is None:
        report = {"well": well, "status": "skip_none"}
    report = dict(report)
    local_cuts = report.pop("cut_rows", [])
    candidate = np.full(len(rows), np.nan, dtype=np.float64)
    if report.get("status") == "ok":
        best_name = str(report["best_name"])
        need_pf = best_name.startswith("pf|")
        pool = namespace["_gold_candidate_pool"](
            well,
            hw,
            tw,
            data_root,
            variants,
            include_pf=need_pf,
            n_seeds=FINAL_SEEDS,
            n_particles=PARTICLES,
        )
        if best_name not in pool and need_pf:
            pool = namespace["_gold_candidate_pool"](
                well, hw, tw, data_root, variants,
                include_pf=False, n_seeds=0, n_particles=PARTICLES,
            )
        if best_name in pool:
            candidate = np.asarray(pool[best_name], np.float64)[rows]
            report["final_candidate_available"] = bool(np.isfinite(candidate).all())
            if not np.isfinite(candidate).all():
                report["status"] = "skip_nonfinite_final_candidate"
        else:
            report["final_candidate_available"] = False
            report["status"] = "skip_no_final_candidate"

    payload = {
        "recipe": {
            "cache_version": 2,
            "notebook_sha256": EXPECTED_NOTEBOOK_SHA256,
            "profile": PROFILE,
            "cuts": CUT_FRACS,
            "cal_seeds": CAL_SEEDS,
            "final_seeds": FINAL_SEEDS,
            "particles": PARTICLES,
            "fold": fold,
        },
        "well": well,
        "fold": fold,
        "ids": well_parent["id"].astype(str).tolist(),
        "candidate": candidate.astype(np.float32),
        "report": report,
        "cut_rows": local_cuts,
    }
    if not repeat:
        joblib.dump(payload, cache_path, compress=3)
    return payload


processed = 0
for fold in sorted({fold_by_well[well] for well in selected_wells}):
    fold_wells = [well for well in selected_wells if fold_by_well[well] == fold]
    reference_wells = sorted(well for well, value in fold_by_well.items() if value != fold)
    print(
        "VP REFERENCE",
        {"heldout_fold": fold, "outer_train_wells": len(reference_wells), "eval_wells": len(fold_wells)},
        flush=True,
    )
    namespace["FI"] = namespace["FormationPlaneKNN"](reference_wells, data_root / "train")
    namespace["DI"] = namespace["DenseANCCImputer"](reference_wells, data_root / "train")
    namespace["_FI"] = namespace["FI"]
    namespace["_DI"] = namespace["DI"]

    for well in fold_wells:
        payload = _run_one(well)
        reports.append(payload["report"])
        cut_rows.extend(payload["cut_rows"])
        for row_id, value in zip(payload["ids"], payload["candidate"]):
            if np.isfinite(value):
                candidate_by_id[row_id] = float(value)
        processed += 1
        report = payload["report"]
        print(
            f"VP {processed}/{len(selected_wells)}",
            {
                "well": well,
                "status": report.get("status"),
                "best": report.get("best_name"),
                "gain": report.get("gain"),
                "consistency": report.get("consistency"),
            },
            flush=True,
        )

    if REPEAT_GUARD and repeat_guard is None and fold_wells:
        first = fold_wells[0]
        cached = joblib.load(CACHE_DIR / f"{first}.joblib")
        repeated = _run_one(first, repeat=True)
        same_candidate = np.array_equal(
            np.asarray(cached["candidate"]), np.asarray(repeated["candidate"]), equal_nan=True
        )
        report_keys = ("status", "best_name", "best_score", "default_score", "gain", "consistency")
        same_report = all(cached["report"].get(key) == repeated["report"].get(key) for key in report_keys)
        repeat_guard = {
            "well": first,
            "candidate_bitwise_equal": bool(same_candidate),
            "report_equal": bool(same_report),
            "pass": bool(same_candidate and same_report),
        }
        if not repeat_guard["pass"]:
            raise RuntimeError(f"Repeated-forward guard failed: {repeat_guard}")

base_sub = parent[["id", "well"]].copy()
base_sub["tvt"] = parent["cp8_midpoint_modelpkg"].to_numpy(np.float64)
parent_sub = parent[["id", "well"]].copy()
parent_sub["tvt"] = parent["parent_midpoint_modelpkg"].to_numpy(np.float64)
reports_by_well = {str(report.get("well")): report for report in reports if report.get("well")}
profile_sub, move_rows = namespace["_gold_profile_output"](
    base_sub, candidate_by_id, reports_by_well, PROFILE
)
parent_profile_sub, parent_move_rows = namespace["_gold_profile_output"](
    parent_sub, candidate_by_id, reports_by_well, PROFILE
)
profile_prediction = profile_sub["tvt"].to_numpy(np.float64)
base_prediction = base_sub["tvt"].to_numpy(np.float64)
move = profile_prediction - base_prediction
parent_base_prediction = parent_sub["tvt"].to_numpy(np.float64)
parent_profile_prediction = parent_profile_sub["tvt"].to_numpy(np.float64)
parent_move = parent_profile_prediction - parent_base_prediction

# Matched controls preserve the exact selected dose and move distribution.
reverse_prediction = base_prediction - move
rolled_move = np.zeros_like(move)
for _, indices in parent.groupby("well", sort=False).groups.items():
    idx = np.asarray(list(indices), dtype=np.int64)
    rolled_move[idx] = np.roll(move[idx], max(1, len(idx) // 2))
rolled_prediction = base_prediction + rolled_move

predictions = {
    "parent_midpoint_modelpkg": parent_base_prediction,
    "parent_balanced_visible_prefix": parent_profile_prediction,
    "cp8_midpoint_modelpkg": base_prediction,
    "balanced_visible_prefix": profile_prediction,
    "visible_prefix_reverse": reverse_prediction,
    "visible_prefix_rolled_control": rolled_prediction,
}
summary_rows = []
for endpoint, prediction in predictions.items():
    summary_rows.append({"fold": 0, **_metric(parent, prediction, endpoint)})
for fold in sorted(parent["fold"].astype(int).unique()):
    mask = parent["fold"].astype(int).eq(fold).to_numpy()
    fold_frame = parent.loc[mask].reset_index(drop=True)
    for endpoint, prediction in predictions.items():
        summary_rows.append({"fold": int(fold), **_metric(fold_frame, prediction[mask], endpoint)})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / f"{STAGE}_summary.csv", index=False)
pd.DataFrame(reports).to_csv(OUT_DIR / f"{STAGE}_well_reports.csv", index=False)
pd.DataFrame(cut_rows).to_csv(OUT_DIR / f"{STAGE}_cut_reports.csv", index=False)
move_frame = pd.DataFrame(move_rows)
move_frame.to_csv(OUT_DIR / f"{STAGE}_move_reports.csv", index=False)
pd.DataFrame(parent_move_rows).to_csv(
    OUT_DIR / f"{STAGE}_parent_move_reports.csv", index=False
)

output = parent[["id", "well", "row_idx", "fold", "tvt_true", "md_since"]].copy()
for endpoint, prediction in predictions.items():
    output[endpoint] = prediction.astype(np.float32)
output.to_parquet(OUT_DIR / f"{STAGE}_visible_prefix_oof.parquet", index=False, compression="zstd")

overall = summary[summary["fold"].eq(0)].set_index("endpoint")
base_rmse = float(overall.loc["cp8_midpoint_modelpkg", "rmse"])
candidate_rmse = float(overall.loc["balanced_visible_prefix", "rmse"])
gain = base_rmse - candidate_rmse
active_wells = int((move_frame.get("apply_status", pd.Series(dtype=str)) == "applied").sum())
ok_reports = [report for report in reports if report.get("status") == "ok"]
available_reports = [
    report for report in ok_reports if bool(report.get("final_candidate_available"))
]
final_candidate_rate = float(len(available_reports) / max(1, len(ok_reports)))
worst = [_worst_twenty(parent, prediction, endpoint) for endpoint, prediction in predictions.items()]
gates = {
    "all_selected_wells_processed": bool(len(reports) == len(selected_wells)),
    "no_runtime_error_reports": bool(not any(report.get("status") == "error" for report in reports)),
    "final_candidate_available_for_ok_reports": bool(final_candidate_rate >= 0.99),
    "repeat_forward_bitwise_equal": bool(repeat_guard is None or repeat_guard["pass"]),
    "correct_direction_beats_reverse": bool(
        candidate_rmse < float(overall.loc["visible_prefix_reverse", "rmse"])
    ),
    "correct_direction_beats_rolled_control": bool(
        candidate_rmse < float(overall.loc["visible_prefix_rolled_control", "rmse"])
    ),
    "gain_vs_parent_positive": bool(gain > 0.0),
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": f"{STAGE}_exact_balanced_visible_prefix_replay",
    "notebook": str(notebook_path),
    "notebook_sha256": _sha256(notebook_path),
    "data_root": str(data_root),
    "protocol": {
        "profile": PROFILE,
        "cut_fracs": CUT_FRACS,
        "calibration_seeds": CAL_SEEDS,
        "final_seeds": FINAL_SEEDS,
        "particles": PARTICLES,
        "candidate_variants": len(variants),
        "surface_reference": "outer-fold train wells only",
        "same_id_train_contact_candidate": "disabled_for_hidden_well_matched_oof",
        "hidden_truth_used_for_candidate_generation_or_dose": False,
        "test_time_visible_prefix_candidate_search": True,
        "parameter_optimizer_count": 0,
    },
    "coverage": {
        "rows": int(len(parent)),
        "wells": int(parent["well"].nunique()),
        "candidate_rows": int(len(candidate_by_id)),
        "candidate_row_coverage": float(len(candidate_by_id) / len(parent)),
        "ok_reports": int(len(ok_reports)),
        "final_candidate_rate_among_ok_reports": final_candidate_rate,
        "active_wells": active_wells,
    },
    "repeat_forward_guard": repeat_guard,
    "metrics": overall.reset_index().to_dict(orient="records"),
    "gain_vs_parent": float(gain),
    "matched_parent_profile_gain": float(
        overall.loc["parent_midpoint_modelpkg", "rmse"]
        - overall.loc["parent_balanced_visible_prefix", "rmse"]
    ),
    "cp8_gain_after_matched_visible_prefix": float(
        overall.loc["parent_balanced_visible_prefix", "rmse"]
        - overall.loc["balanced_visible_prefix", "rmse"]
    ),
    "mean_abs_move": float(np.mean(np.abs(move))),
    "parent_mean_abs_move": float(np.mean(np.abs(parent_move))),
    "p95_abs_move": float(np.quantile(np.abs(move), 0.95)),
    "max_abs_move": float(np.max(np.abs(move))),
    "worst_twenty_percent_wells": worst,
    "condition_number": "not_applicable_to_candidate_replay",
    "solve_failures": int(sum(report.get("status") == "error" for report in reports)),
    "gates": gates,
    "strict_pass": bool(all(gates.values())),
    "elapsed_minutes": float((time.time() - started) / 60.0),
    "next": (
        "DISCOVERY PASS -> rerun with ROGII_VP_STAGE=full"
        if STAGE == "discovery" and all(gates.values())
        else "FULL PASS -> export the exact Kaggle CP8+tail chain"
        if STAGE == "full" and all(gates.values())
        else "FAIL -> visible-prefix overlay does not survive matched replay"
    ),
}
(OUT_DIR / f"{STAGE}_verdict.json").write_text(
    json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8"
)

print("BALANCED VISIBLE-PREFIX SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
