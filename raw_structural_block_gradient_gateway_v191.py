"""V191: leakage-safe structural block-gradient gateway.

This is a new standalone engine. It consumes no historical predictions, OOF
endpoints, candidate banks, PF paths, spatial neighbours, or validation labels
at inference. The model predicts blockwise derivatives of S = TVT + Z from
deployment-legal raw geometry, GR, the visible TVT prefix, and the assigned
typewell. The hidden path is obtained by integrating those derivatives from
the last visible S anchor.

Protocol: folds 4/5 train, fold 2 selects block size and shrinkage, fold 3 is
touched once, and fold 1 is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = "raw_structural_block_gradient_gateway_v191"
SEED = 20260803
TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 2
GATEWAY_FOLD = 3
BLOCK_SIZES = (64, 128, 256)
ALPHAS = (0.25, 0.50, 0.75, 1.00)
PREFIX_WINDOWS = (64, 128, 256, 512)
TYPEWELL_POINTS = 24


@dataclass
class WellRecord:
    well: str
    fold: int
    md: np.ndarray
    z: np.ndarray
    truth: np.ndarray
    hidden: np.ndarray
    anchor_index: int
    prefix_rate: float
    prefix_plane_rate_x: float
    prefix_plane_rate_y: float
    common: dict[str, float]
    tw_features: dict[str, float]
    raw: pd.DataFrame


def _find_data_root(root: Path) -> Path:
    candidates = (
        root / "rogii-wellbore-geology-prediction",
        root,
        Path.cwd(),
    )
    for candidate in candidates:
        if (candidate / "train").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the competition train directory")


def _find_metadata(root: Path) -> Path:
    candidates = (
        root / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
        root / "oof_exports" / "multicut_k5_full_spatial_lgb_bundle_v1" / "well_metadata.csv",
        Path.cwd() / "oof_exports" / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Frozen well_metadata.csv was not found")


def _robust_rate(md: np.ndarray, surface: np.ndarray) -> float:
    if len(md) < 3:
        return 0.0
    delta_md = np.diff(md)
    valid = np.isfinite(delta_md) & (delta_md > 1e-6)
    rates = np.diff(surface)[valid] / delta_md[valid]
    rates = rates[np.isfinite(rates)]
    if not len(rates):
        return 0.0
    lo, hi = np.quantile(rates, (0.05, 0.95)) if len(rates) >= 20 else (-np.inf, np.inf)
    trimmed = rates[(rates >= lo) & (rates <= hi)]
    return float(np.median(trimmed if len(trimmed) else rates))


def _prefix_plane(x: np.ndarray, y: np.ndarray, surface: np.ndarray) -> tuple[float, float]:
    if len(surface) < 8:
        return 0.0, 0.0
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    design = np.column_stack((x0, y0))
    lhs = design.T @ design + np.eye(2) * 100.0
    rhs = design.T @ (surface - np.mean(surface))
    coefficient = np.linalg.solve(lhs, rhs)
    return float(coefficient[0]), float(coefficient[1])


def _safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        finite = np.array([0.0])
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_q10": float(np.quantile(finite, 0.10)),
        f"{prefix}_q50": float(np.quantile(finite, 0.50)),
        f"{prefix}_q90": float(np.quantile(finite, 0.90)),
        f"{prefix}_range": float(np.ptp(finite)),
    }


def _typewell_features(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path, usecols=lambda column: column in {"TVT", "GR"}).sort_values("TVT")
    tvt = frame["TVT"].to_numpy(np.float64)
    gr = frame["GR"].interpolate(limit_direction="both").to_numpy(np.float64)
    finite = np.isfinite(tvt) & np.isfinite(gr)
    tvt, gr = tvt[finite], gr[finite]
    if len(gr) < 2:
        tvt, gr = np.array([0.0, 1.0]), np.array([0.0, 0.0])
    axis = np.linspace(0.0, 1.0, len(gr))
    sampled = np.interp(np.linspace(0.0, 1.0, TYPEWELL_POINTS), axis, gr)
    median = float(np.median(sampled))
    scale = float(np.median(np.abs(sampled - median))) * 1.4826
    if scale < 1e-6:
        scale = float(np.std(sampled)) + 1e-6
    sampled = (sampled - median) / scale
    result = {
        "tw_tvt_span": float(np.ptp(tvt)),
        "tw_gr_mean": float(np.mean(gr)),
        "tw_gr_std": float(np.std(gr)),
    }
    result.update({f"tw_shape_{index:02d}": float(value) for index, value in enumerate(sampled)})
    return result


def _load_record(data_root: Path, well: str, fold: int) -> WellRecord:
    horizontal_path = data_root / "train" / f"{well}__horizontal_well.csv"
    typewell_path = data_root / "train" / f"{well}__typewell.csv"
    columns = ["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"]
    frame = pd.read_csv(horizontal_path, usecols=columns)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["GR"] = frame["GR"].interpolate(limit_direction="both")
    known = frame["TVT_input"].notna().to_numpy()
    hidden = ~known & frame["TVT"].notna().to_numpy()
    hidden_indices = np.flatnonzero(hidden)
    if not len(hidden_indices):
        raise RuntimeError(f"{well}: no hidden evaluation rows")
    anchor = int(hidden_indices[0] - 1)
    if anchor < 2 or not known[: anchor + 1].all() or known[anchor + 1 :].any():
        raise RuntimeError(f"{well}: TVT_input is not one contiguous visible prefix")

    md = frame["MD"].to_numpy(np.float64)
    z = frame["Z"].to_numpy(np.float64)
    truth = frame["TVT"].to_numpy(np.float64)
    prefix_surface = frame.loc[:anchor, "TVT_input"].to_numpy(np.float64) + z[: anchor + 1]
    prefix_rate = _robust_rate(md[max(0, anchor - 255) : anchor + 1], prefix_surface[max(0, anchor - 255) :])
    gx, gy = _prefix_plane(
        frame.loc[:anchor, "X"].to_numpy(np.float64)[-512:],
        frame.loc[:anchor, "Y"].to_numpy(np.float64)[-512:],
        prefix_surface[-512:],
    )
    common: dict[str, float] = {
        "anchor_x_kft": float(frame.loc[anchor, "X"] / 1000.0),
        "anchor_y_kft": float(frame.loc[anchor, "Y"] / 1000.0),
        "anchor_z_kft": float(frame.loc[anchor, "Z"] / 1000.0),
        "anchor_md_kft": float(frame.loc[anchor, "MD"] / 1000.0),
        "known_rows": float(anchor + 1),
        "hidden_rows": float(hidden.sum()),
        "prefix_rate": prefix_rate,
        "prefix_plane_gx": gx,
        "prefix_plane_gy": gy,
    }
    for window in PREFIX_WINDOWS:
        start = max(0, anchor + 1 - window)
        common[f"prefix_rate_{window}"] = _robust_rate(md[start : anchor + 1], prefix_surface[start:])
    common.update(_safe_stats(frame.loc[:anchor, "GR"].to_numpy(np.float64)[-512:], "prefix_gr"))
    return WellRecord(
        well=well,
        fold=fold,
        md=md,
        z=z,
        truth=truth,
        hidden=hidden,
        anchor_index=anchor,
        prefix_rate=prefix_rate,
        prefix_plane_rate_x=gx,
        prefix_plane_rate_y=gy,
        common=common,
        tw_features=_typewell_features(typewell_path),
        raw=frame,
    )


def _block_features(record: WellRecord, indices: np.ndarray, previous: int, block_number: int) -> dict[str, float]:
    frame = record.raw
    current = frame.iloc[indices]
    md = current["MD"].to_numpy(np.float64)
    x = current["X"].to_numpy(np.float64)
    y = current["Y"].to_numpy(np.float64)
    z = current["Z"].to_numpy(np.float64)
    gr = current["GR"].to_numpy(np.float64)
    dm = max(float(md[-1] - record.md[previous]), 1e-6)
    dx = float(x[-1] - frame.iloc[previous]["X"])
    dy = float(y[-1] - frame.iloc[previous]["Y"])
    dz = float(z[-1] - frame.iloc[previous]["Z"])
    step_md = np.diff(np.r_[record.md[previous], md])
    step_md = np.where(step_md > 1e-6, step_md, 1.0)
    tx = np.diff(np.r_[frame.iloc[previous]["X"], x]) / step_md
    ty = np.diff(np.r_[frame.iloc[previous]["Y"], y]) / step_md
    tz = np.diff(np.r_[frame.iloc[previous]["Z"], z]) / step_md
    plane_rate = record.prefix_plane_rate_x * dx / dm + record.prefix_plane_rate_y * dy / dm
    values: dict[str, float] = {
        **record.common,
        **record.tw_features,
        "block_number": float(block_number),
        "block_rows": float(len(indices)),
        "block_md_span": dm,
        "md_since_anchor_kft": float((md[-1] - record.md[record.anchor_index]) / 1000.0),
        "mid_x_kft": float(np.mean(x) / 1000.0),
        "mid_y_kft": float(np.mean(y) / 1000.0),
        "mid_z_kft": float(np.mean(z) / 1000.0),
        "dx_per_md": dx / dm,
        "dy_per_md": dy / dm,
        "dz_per_md": dz / dm,
        "plane_rate": plane_rate,
        "horizontal_span": float(np.hypot(dx, dy)),
        "tangent_x_std": float(np.std(tx)),
        "tangent_y_std": float(np.std(ty)),
        "tangent_z_std": float(np.std(tz)),
        "turn_rms": float(np.sqrt(np.mean(np.diff(np.arctan2(ty, tx)) ** 2))) if len(tx) > 1 else 0.0,
        "gr_start": float(gr[0]),
        "gr_end": float(gr[-1]),
        "gr_delta": float(gr[-1] - gr[0]),
    }
    values.update(_safe_stats(gr, "block_gr"))
    values.update(_safe_stats(np.diff(gr), "block_dgr"))
    return values


def _build_blocks(records: list[WellRecord], block_size: int) -> tuple[pd.DataFrame, dict[str, list[tuple[np.ndarray, int]]]]:
    rows: list[dict[str, object]] = []
    slices: dict[str, list[tuple[np.ndarray, int]]] = {}
    for number, record in enumerate(records, 1):
        hidden = np.flatnonzero(record.hidden)
        previous = record.anchor_index
        blocks: list[tuple[np.ndarray, int]] = []
        for block_number, start in enumerate(range(0, len(hidden), block_size)):
            indices = hidden[start : start + block_size]
            features = _block_features(record, indices, previous, block_number)
            surface = record.truth + record.z
            md_span = max(float(record.md[indices[-1]] - record.md[previous]), 1e-6)
            features.update(
                {
                    "well": record.well,
                    "fold": record.fold,
                    "target_rate": float((surface[indices[-1]] - surface[previous]) / md_span),
                    "row_weight": float(len(indices)),
                }
            )
            rows.append(features)
            blocks.append((indices, previous))
            previous = int(indices[-1])
        slices[record.well] = blocks
        if number % 100 == 0:
            print(f"V191 blocks b{block_size} {number}/{len(records)}", flush=True)
    return pd.DataFrame(rows), slices


def _predict_rows(
    records: list[WellRecord],
    panel: pd.DataFrame,
    slices: dict[str, list[tuple[np.ndarray, int]]],
    rates: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    panel = panel.copy()
    panel["model_rate"] = rates
    lookup = {well: local for well, local in panel.groupby("well", sort=False)}
    parts: list[pd.DataFrame] = []
    for record in records:
        local = lookup[record.well].sort_values("block_number")
        block_rates = local["model_rate"].to_numpy(np.float64)
        current_surface = float(record.raw.loc[record.anchor_index, "TVT_input"] + record.z[record.anchor_index])
        prediction = np.full(len(record.md), np.nan, np.float64)
        for rate, (indices, previous) in zip(block_rates, slices[record.well], strict=True):
            applied_rate = (1.0 - alpha) * record.prefix_rate + alpha * float(rate)
            prediction[indices] = current_surface + applied_rate * (record.md[indices] - record.md[previous])
            current_surface = float(prediction[indices[-1]])
        hidden = np.flatnonzero(record.hidden)
        parts.append(
            pd.DataFrame(
                {
                    "well": record.well,
                    "fold": record.fold,
                    "row_idx": hidden,
                    "truth": record.truth[hidden],
                    "prediction": prediction[hidden] - record.z[hidden],
                    "distance": record.md[hidden] - record.md[record.anchor_index],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _metrics(frame: pd.DataFrame, endpoint: str) -> dict[str, float | int | str]:
    error = frame["prediction"].to_numpy(np.float64) - frame["truth"].to_numpy(np.float64)
    squared = error**2
    by_well = frame.assign(squared_error=squared).groupby("well")["squared_error"].mean().pow(0.5)
    result: dict[str, float | int | str] = {
        "endpoint": endpoint,
        "rows": len(frame),
        "wells": frame["well"].nunique(),
        "exact_sse": float(np.sum(squared)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(by_well.median()),
        "p95_well_rmse": float(by_well.quantile(0.95)),
    }
    for name, mask in (
        ("near_0_500_rmse", frame["distance"] <= 500.0),
        ("mid_500_1000_rmse", (frame["distance"] > 500.0) & (frame["distance"] <= 1000.0)),
        ("mid_1000_2000_rmse", (frame["distance"] > 1000.0) & (frame["distance"] <= 2000.0)),
        ("far_gt2000_rmse", frame["distance"] > 2000.0),
    ):
        local = squared[mask.to_numpy()]
        result[name] = float(np.sqrt(np.mean(local))) if len(local) else np.nan
    return result


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def main() -> None:
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("V191 requires LightGBM in the AutoDL environment") from error

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--threads", type=int, default=12)
    args, _ = parser.parse_known_args()
    started = time.time()
    root = args.root.resolve()
    data_root = _find_data_root(root)
    metadata_path = _find_metadata(root)
    out_dir = root / "oof_exports" / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    if not {"well", "fold"}.issubset(metadata.columns):
        raise RuntimeError(f"Unexpected metadata columns: {metadata.columns.tolist()}")
    metadata = metadata[["well", "fold"]].drop_duplicates("well")
    metadata["well"] = metadata["well"].astype(str).str.strip().str.lower()
    metadata["fold"] = metadata["fold"].astype(int)
    selected_folds = set(TRAIN_FOLDS) | {SELECTION_FOLD, GATEWAY_FOLD}
    metadata = metadata[metadata["fold"].isin(selected_folds)].sort_values(["fold", "well"])

    records: list[WellRecord] = []
    failures: list[dict[str, str | int]] = []
    for number, row in enumerate(metadata.itertuples(index=False), 1):
        try:
            records.append(_load_record(data_root, row.well, int(row.fold)))
        except Exception as error:
            failures.append({"well": row.well, "fold": int(row.fold), "reason": repr(error)})
        if number % 100 == 0:
            print(f"V191 records {number}/{len(metadata)} failures={len(failures)}", flush=True)
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)
        raise RuntimeError(f"V191 record failures: {failures[:5]}")

    coverage = pd.Series([record.fold for record in records]).value_counts().sort_index().to_dict()
    expected = {2: 155, 3: 154, 4: 155, 5: 154}
    if coverage != expected:
        raise RuntimeError(f"Unexpected fold coverage: {coverage}, expected {expected}")
    print("V191 CONFIG", json.dumps({"coverage": coverage, "data_root": str(data_root)}, indent=2), flush=True)

    selection_scan: list[dict[str, float | int]] = []
    artifacts: dict[int, tuple[object, list[str], pd.DataFrame, dict, np.ndarray]] = {}
    for block_size in BLOCK_SIZES:
        panel, slices = _build_blocks(records, block_size)
        feature_names = [
            column
            for column in panel.columns
            if column not in {"well", "fold", "target_rate", "row_weight"}
        ]
        train = panel["fold"].isin(TRAIN_FOLDS)
        selection = panel["fold"].eq(SELECTION_FOLD)
        model = lgb.LGBMRegressor(
            objective="huber",
            learning_rate=0.025,
            n_estimators=1800,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.80,
            reg_alpha=0.2,
            reg_lambda=2.0,
            random_state=SEED + block_size,
            n_jobs=args.threads,
            verbosity=-1,
            force_col_wise=True,
        )
        model.fit(
            panel.loc[train, feature_names],
            panel.loc[train, "target_rate"],
            sample_weight=panel.loc[train, "row_weight"],
            eval_set=[(panel.loc[selection, feature_names], panel.loc[selection, "target_rate"])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        train_rates = panel.loc[train, "target_rate"].to_numpy(np.float64)
        rate_limits = np.quantile(train_rates, (0.002, 0.998))
        selection_rates = np.clip(
            model.predict(panel.loc[selection, feature_names], num_iteration=model.best_iteration_),
            rate_limits[0],
            rate_limits[1],
        )
        selection_records = [record for record in records if record.fold == SELECTION_FOLD]
        selection_panel = panel.loc[selection].copy()
        selection_slices = {record.well: slices[record.well] for record in selection_records}
        for alpha in ALPHAS:
            prediction = _predict_rows(
                selection_records, selection_panel, selection_slices, selection_rates, alpha
            )
            metric = _metrics(prediction, f"b{block_size}_a{alpha:.2f}")
            selection_scan.append(
                {
                    "block_size": block_size,
                    "alpha": alpha,
                    "best_iteration": int(model.best_iteration_),
                    **{key: value for key, value in metric.items() if key != "endpoint"},
                }
            )
        artifacts[block_size] = (model, feature_names, panel, slices, rate_limits)
        print(
            "V191 trained",
            {"block_size": block_size, "best_iteration": model.best_iteration_},
            flush=True,
        )

    scan = pd.DataFrame(selection_scan).sort_values(["rmse", "block_size", "alpha"])
    scan.to_csv(out_dir / "fold2_selection_scan.csv", index=False)
    selected = scan.iloc[0]
    block_size = int(selected["block_size"])
    alpha = float(selected["alpha"])
    model, feature_names, panel, slices, rate_limits = artifacts[block_size]

    gateway = panel["fold"].eq(GATEWAY_FOLD)
    gateway_rates = np.clip(
        model.predict(panel.loc[gateway, feature_names], num_iteration=model.best_iteration_),
        rate_limits[0],
        rate_limits[1],
    )
    gateway_records = [record for record in records if record.fold == GATEWAY_FOLD]
    gateway_panel = panel.loc[gateway].copy()
    gateway_slices = {record.well: slices[record.well] for record in gateway_records}
    candidate = _predict_rows(gateway_records, gateway_panel, gateway_slices, gateway_rates, alpha)
    baseline = _predict_rows(
        gateway_records,
        gateway_panel,
        gateway_slices,
        np.zeros(len(gateway_panel), np.float64),
        0.0,
    )
    candidate_metric = _metrics(candidate, "structural_block_gradient")
    baseline_metric = _metrics(baseline, "prefix_rate_baseline")
    summary = pd.DataFrame([baseline_metric, candidate_metric])
    summary.to_csv(out_dir / "matched_summary.csv", index=False)
    candidate.to_parquet(out_dir / "fold3_fullrow_oof.parquet", index=False)
    by_well = candidate.assign(
        squared_error=np.square(candidate["prediction"] - candidate["truth"])
    ).groupby("well", as_index=False).agg(rows=("row_idx", "size"), rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))))
    by_well.to_csv(out_dir / "fold3_by_well.csv", index=False)

    gain = float(baseline_metric["rmse"] - candidate_metric["rmse"])
    far_gain = float(baseline_metric["far_gt2000_rmse"] - candidate_metric["far_gt2000_rmse"])
    checks = {
        "fold3_rmse_le_7": candidate_metric["rmse"] <= 7.0,
        "fold3_rmse_le_6": candidate_metric["rmse"] <= 6.0,
        "gain_vs_prefix_ge_0p25": gain >= 0.25,
        "far_gain_positive": far_gain > 0.0,
        "fold1_not_loaded": True,
    }
    strict_pass = bool(checks["fold3_rmse_le_7"] and checks["gain_vs_prefix_ge_0p25"] and checks["far_gain_positive"])
    prediction_hash = hashlib.sha256(candidate["prediction"].to_numpy(np.float32).tobytes()).hexdigest()
    verdict = {
        "experiment": EXPERIMENT,
        "stage": "Fold45_train_Fold2_selection_Fold3_single_gateway",
        "historical_prediction_inputs": [],
        "legal_inputs": ["MD", "X", "Y", "Z", "GR", "TVT_input", "typewell_TVT_GR"],
        "target_representation": "blockwise derivative of S=TVT+Z",
        "protocol": {
            "train_folds": list(TRAIN_FOLDS),
            "selection_fold": SELECTION_FOLD,
            "gateway_fold": GATEWAY_FOLD,
            "fold1_loaded": False,
        },
        "selected_recipe": {
            "block_size": block_size,
            "alpha": alpha,
            "best_iteration": int(model.best_iteration_),
            "rate_clip": [float(rate_limits[0]), float(rate_limits[1])],
        },
        "fold2_selected_rmse": float(selected["rmse"]),
        "baseline": baseline_metric,
        "candidate": candidate_metric,
        "gain": gain,
        "far_gain": far_gain,
        "checks": checks,
        "strict_pass": strict_pass,
        "prediction_sha256": prediction_hash,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "next": (
            "PASS -> repeat frozen recipe on untouched Fold1, then build full fivefold OOF"
            if strict_pass
            else "FAIL -> permanently close structural block-gradient engine; do not blend or repair"
        ),
    }
    verdict = _clean(verdict)
    (out_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("\nV191 FOLD2 TOP 20")
    print(scan.head(20).to_string(index=False))
    print("\nV191 MATCHED SUMMARY")
    print(summary.to_string(index=False))
    print("\nV191 FINAL VERDICT")
    print(json.dumps(verdict, indent=2, allow_nan=False))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
