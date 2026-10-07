"""Phase 0 gateway for Chronolog GR self-supervised pretraining.

This cell is deliberately target-free. It compares Chronolog vertical GR
windows with two competition domains on the same geological-depth axis:

* typewell GR(TVT)
* horizontal-well known-prefix GR(TVT)

It does not mix Chronolog rows into supervised training. The output decides
whether a frozen GR descriptor pretraining experiment is defensible.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
DATA_ROOTS = (
    ROOT / "rogii-wellbore-geology-prediction",
    ROOT / "replay_data",
    Path.cwd(),
)
DATA_ROOT = next((path for path in DATA_ROOTS if (path / "train").is_dir()), DATA_ROOTS[0])
OUT_DIR = ROOT / "oof_exports" / "chronolog_gr_domain_gateway_v81"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALES_FT = (64.0, 128.0, 256.0, 512.0)
POINTS = 128
WINDOWS_PER_WELL_SCALE = 4
MAX_COMPETITION_WELLS = 773
RANDOM_STATE = 20260730
CHUNK_ROWS = 500_000


def find_chronolog_root():
    explicit = os.environ.get("CHRONOLOG_ROOT")
    roots = [Path(explicit)] if explicit else []
    roots.extend(
        [
            ROOT / "rogii-chronolog-gr-ccby4-v1",
            ROOT / "rogii-chronolog-gr-ccby4-v1" / "rogii-chronolog-gr-ccby4-v1",
            Path("/kaggle/input/rogii-chronolog-gr-ccby4-v1"),
            Path.cwd() / "rogii-chronolog-gr-ccby4-v1",
        ]
    )
    for root in roots:
        if root and (root / "chronolog_logs.csv.gz.bin").is_file():
            return root
    search_roots = [ROOT, Path("/kaggle/input")]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        matches = list(search_root.glob("**/chronolog_logs.csv.gz.bin"))
        if matches:
            return matches[0].parent
    raise FileNotFoundError(
        "chronolog_logs.csv.gz.bin was not found. Upload the dataset directory "
        "under /root/autodl-tmp or set CHRONOLOG_ROOT."
    )


CHRONOLOG_ROOT = find_chronolog_root()
CHRONOLOG_PATH = CHRONOLOG_ROOT / "chronolog_logs.csv.gz.bin"


def json_if_exists(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compression_for(path):
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic[:2] == b"\x1f\x8b":
        return "gzip"
    if magic == b"PAR1":
        return "parquet"
    return None


def numeric_frame(frame, columns):
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce")


def robust_scale(values):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 8:
        return 0.0, 1.0
    centre = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values), 1.0))
    return centre, scale


def deduplicate_axis(axis, values):
    frame = pd.DataFrame({"axis": axis, "value": values}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 16:
        return np.empty(0), np.empty(0)
    grouped = frame.groupby("axis", sort=True, as_index=False)["value"].mean()
    return grouped["axis"].to_numpy(np.float64), grouped["value"].to_numpy(np.float64)


def extract_windows(well, source, axis, values):
    axis, values = deduplicate_axis(axis, values)
    if len(axis) < 16:
        return []
    centre, scale = robust_scale(values)
    output = []
    available = float(axis[-1] - axis[0])
    for scale_ft in SCALES_FT:
        if available < scale_ft * 1.05:
            continue
        starts = np.linspace(axis[0], axis[-1] - scale_ft, WINDOWS_PER_WELL_SCALE + 2)[1:-1]
        for slot, start in enumerate(starts):
            query = np.linspace(float(start), float(start + scale_ft), POINTS)
            raw = np.interp(query, axis, values)
            normalised = np.clip((raw - centre) / scale, -8.0, 8.0)
            output.append(
                {
                    "well": str(well),
                    "group": f"{source}:{well}",
                    "source": source,
                    "scale_ft": float(scale_ft),
                    "slot": int(slot),
                    "axis_step_ft": float(np.median(np.diff(axis))),
                    "raw": raw.astype(np.float32),
                    "normalised": normalised.astype(np.float32),
                }
            )
    return output


def load_chronolog():
    compression = compression_for(CHRONOLOG_PATH)
    if compression == "parquet":
        iterator = [pd.read_parquet(CHRONOLOG_PATH)]
    else:
        iterator = pd.read_csv(
            CHRONOLOG_PATH,
            compression=compression,
            chunksize=CHUNK_ROWS,
            low_memory=False,
        )
    curves = defaultdict(lambda: [[], []])
    rows = 0
    detected = None
    for chunk_number, chunk in enumerate(iterator, 1):
        lower = {str(column).strip().lower(): column for column in chunk.columns}
        well_column = next((lower[name] for name in ("well_id", "well", "id") if name in lower), None)
        depth_column = next((lower[name] for name in ("depth_m", "depth", "md_m") if name in lower), None)
        gr_column = next((lower[name] for name in ("gamma_ray", "gamma ray", "gr") if name in lower), None)
        if None in (well_column, depth_column, gr_column):
            raise RuntimeError(f"Cannot identify Chronolog columns: {list(chunk.columns)}")
        detected = {"well": str(well_column), "depth": str(depth_column), "gr": str(gr_column)}
        local = chunk[[well_column, depth_column, gr_column]].copy()
        local.columns = ["well", "depth_m", "gr"]
        local["well"] = local["well"].astype(str)
        local[["depth_m", "gr"]] = numeric_frame(local, ("depth_m", "gr"))
        local = local.replace([np.inf, -np.inf], np.nan).dropna(subset=["well", "depth_m", "gr"])
        rows += len(local)
        for well, group in local.groupby("well", sort=False):
            curves[str(well)][0].append(group["depth_m"].to_numpy(np.float64) * 3.280839895013123)
            curves[str(well)][1].append(group["gr"].to_numpy(np.float64))
        if chunk_number % 4 == 0:
            print(f"Chronolog chunks {chunk_number}: rows={rows:,} wells={len(curves)}", flush=True)
    windows, failures = [], []
    for well, (depth_parts, gr_parts) in curves.items():
        try:
            windows.extend(extract_windows(well, "chronolog", np.concatenate(depth_parts), np.concatenate(gr_parts)))
        except Exception as error:
            failures.append({"well": well, "reason": repr(error)})
    return windows, {"rows": rows, "wells": len(curves), "columns": detected, "failures": failures}


def load_competition():
    typewell_windows, prefix_windows, failures = [], [], []
    horizontal_paths = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))[:MAX_COMPETITION_WELLS]
    for number, horizontal_path in enumerate(horizontal_paths, 1):
        well = horizontal_path.name.split("__", 1)[0]
        typewell_path = DATA_ROOT / "train" / f"{well}__typewell.csv"
        try:
            typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
            typewell = numeric_frame(typewell, ("TVT", "GR"))
            typewell_windows.extend(
                extract_windows(well, "competition_typewell", typewell["TVT"], typewell["GR"])
            )
            horizontal = pd.read_csv(horizontal_path, usecols=["GR", "TVT_input"])
            horizontal = numeric_frame(horizontal, ("GR", "TVT_input"))
            known = horizontal["TVT_input"].notna() & horizontal["GR"].notna()
            prefix_windows.extend(
                extract_windows(
                    well,
                    "competition_prefix",
                    horizontal.loc[known, "TVT_input"],
                    horizontal.loc[known, "GR"],
                )
            )
        except Exception as error:
            failures.append({"well": well, "reason": repr(error)})
        if number % 100 == 0 or number == len(horizontal_paths):
            print(
                f"Competition {number}/{len(horizontal_paths)}: "
                f"type={len(typewell_windows)} prefix={len(prefix_windows)} failures={len(failures)}",
                flush=True,
            )
    return typewell_windows + prefix_windows, {
        "wells_requested": len(horizontal_paths),
        "typewell_wells": len({row["well"] for row in typewell_windows}),
        "prefix_wells": len({row["well"] for row in prefix_windows}),
        "failures": failures,
    }


def correlation(values, lag):
    values = np.asarray(values, np.float64)
    if lag >= len(values):
        return 0.0
    left, right = values[:-lag], values[lag:]
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 1e-9 else 0.0


def features(window, shape_only):
    raw = np.asarray(window["raw"], np.float64)
    values = np.asarray(window["normalised"], np.float64)
    delta = np.diff(values)
    second = np.diff(values, 2)
    power = np.abs(np.fft.rfft(values - values.mean())) ** 2
    total = float(power[1:].sum() + 1e-9)
    n = len(power)
    vector = [
        np.mean(values), np.std(values), np.quantile(values, 0.10), np.quantile(values, 0.25),
        np.quantile(values, 0.50), np.quantile(values, 0.75), np.quantile(values, 0.90),
        np.mean(np.abs(delta)), np.std(delta), np.quantile(np.abs(delta), 0.90),
        np.mean(np.abs(second)), np.std(second), correlation(values, 1), correlation(values, 4),
        correlation(values, 8), correlation(values, 16),
        power[1 : max(2, n // 8)].sum() / total,
        power[max(2, n // 8) : max(3, n // 3)].sum() / total,
        power[max(3, n // 3) :].sum() / total,
        np.mean(np.sign(delta[:-1]) != np.sign(delta[1:])),
    ]
    if not shape_only:
        centre, scale = robust_scale(raw)
        vector.extend(
            [centre, scale, np.mean(raw), np.std(raw), np.quantile(raw, 0.05), np.quantile(raw, 0.95)]
        )
    return np.asarray(vector, np.float64)


def domain_auc(windows, left_source, right_source, scale_ft, shape_only):
    selected = [
        row for row in windows
        if row["scale_ft"] == scale_ft and row["source"] in {left_source, right_source}
    ]
    x = np.stack([features(row, shape_only) for row in selected])
    y = np.asarray([row["source"] == right_source for row in selected], np.int8)
    groups = np.asarray([row["group"] for row in selected])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    prediction = np.full(len(selected), np.nan)
    for train, valid in splitter.split(x, y, groups):
        model = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x[train], y[train])
        prediction[valid] = model.predict_proba(x[valid])[:, 1]
    auc = float(roc_auc_score(y, prediction))
    return max(auc, 1.0 - auc), x, y


def cross_domain_nn(x, y):
    scaler = StandardScaler().fit(np.vstack([x, y]))
    x = scaler.transform(x)
    y = scaler.transform(y)
    x_to_y = NearestNeighbors(n_neighbors=1).fit(y).kneighbors(x, return_distance=True)[0][:, 0]
    y_to_x = NearestNeighbors(n_neighbors=1).fit(x).kneighbors(y, return_distance=True)[0][:, 0]
    return {
        "chronolog_to_competition_median": float(np.median(x_to_y)),
        "competition_to_chronolog_median": float(np.median(y_to_x)),
        "chronolog_to_competition_p90": float(np.quantile(x_to_y, 0.90)),
        "competition_to_chronolog_p90": float(np.quantile(y_to_x, 0.90)),
    }


chronolog_windows, chronolog_info = load_chronolog()
competition_windows, competition_info = load_competition()
windows = chronolog_windows + competition_windows
if len({row["well"] for row in chronolog_windows}) < 500:
    raise RuntimeError(f"Too few Chronolog wells: {chronolog_info}")
if competition_info["typewell_wells"] < 700 or competition_info["prefix_wells"] < 700:
    raise RuntimeError(f"Too few competition wells: {competition_info}")

records = []
for scale_ft in SCALES_FT:
    for right_source in ("competition_typewell", "competition_prefix"):
        shape_auc, x, y = domain_auc(windows, "chronolog", right_source, scale_ft, True)
        raw_auc, _, _ = domain_auc(windows, "chronolog", right_source, scale_ft, False)
        nn = cross_domain_nn(x[y == 0], x[y == 1])
        left_steps = [row["axis_step_ft"] for row in windows if row["scale_ft"] == scale_ft and row["source"] == "chronolog"]
        right_steps = [row["axis_step_ft"] for row in windows if row["scale_ft"] == scale_ft and row["source"] == right_source]
        records.append(
            {
                "scale_ft": scale_ft,
                "target_domain": right_source,
                "shape_domain_auc": shape_auc,
                "raw_domain_auc": raw_auc,
                "chronolog_axis_step_median_ft": float(np.median(left_steps)),
                "competition_axis_step_median_ft": float(np.median(right_steps)),
                **nn,
            }
        )

summary = pd.DataFrame(records)
type_rows = summary[summary["target_domain"] == "competition_typewell"]
prefix_rows = summary[summary["target_domain"] == "competition_prefix"]
checks = {
    "chronolog_wells_ge_500": bool(chronolog_info["wells"] >= 500),
    "competition_typewell_wells_ge_700": bool(competition_info["typewell_wells"] >= 700),
    "competition_prefix_wells_ge_700": bool(competition_info["prefix_wells"] >= 700),
    "typewell_shape_auc_mean_le_0p85": bool(type_rows["shape_domain_auc"].mean() <= 0.85),
    "typewell_shape_auc_max_le_0p95": bool(type_rows["shape_domain_auc"].max() <= 0.95),
    "prefix_shape_auc_mean_le_0p90": bool(prefix_rows["shape_domain_auc"].mean() <= 0.90),
}
strict_pass = bool(all(checks.values()))
ready = json_if_exists(CHRONOLOG_ROOT / "READY.json")
verdict = {
    "experiment": "chronolog_gr_domain_gateway_v81",
    "stage": "target_free_real_gr_domain_overlap_gateway",
    "chronolog_root": str(CHRONOLOG_ROOT),
    "chronolog_file": str(CHRONOLOG_PATH),
    "compression": compression_for(CHRONOLOG_PATH),
    "ready_manifest_present": ready is not None,
    "chronolog": chronolog_info,
    "competition": competition_info,
    "windows": {
        source: int(sum(row["source"] == source for row in windows))
        for source in ("chronolog", "competition_typewell", "competition_prefix")
    },
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
    "next": (
        "PASS -> pretrain a small frozen multiscale contrastive GR encoder"
        if strict_pass
        else "FAIL -> do not pretrain; use Chronolog only to calibrate augmentation statistics"
    ),
}

summary.to_csv(OUT_DIR / "domain_summary.csv", index=False)
pd.DataFrame(chronolog_info["failures"]).to_csv(OUT_DIR / "chronolog_failures.csv", index=False)
pd.DataFrame(competition_info["failures"]).to_csv(OUT_DIR / "competition_failures.csv", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nV81 DOMAIN SUMMARY")
print(summary.to_string(index=False))
print("\nV81 FINAL VERDICT")
print(json.dumps(verdict, indent=2), flush=True)
print("saved:", OUT_DIR)
