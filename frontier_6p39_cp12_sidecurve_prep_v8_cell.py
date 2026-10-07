"""Build the historical CP12 side curve without replacing the 6.39 parent.

Run after cell 39 of rogii-public-score-frontier-lab-visuals.ipynb. The cell
reconstructs the historical 0.55 SP45/FLE projected parent, applies the exact
CP12 model trained on strong_base_projected_oof.csv, saves CP12 as an auxiliary
curve, and restores the immutable 6.39 core to submission.csv.
"""

from pathlib import Path
import glob
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


warnings.filterwarnings("ignore")
WORK = Path("/kaggle/working")
DATA = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
if "CFG" in globals():
    DATA = Path(getattr(CFG, "DATA", getattr(CFG, "dataset_path", DATA)))

CP_CTRL_N = 12
CP_KNOTS = np.linspace(0.0, 1.0, CP_CTRL_N)
CP_ALPHA = 0.7
CP_CAP = 3.0


def _read_submission(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label}: {path}")
    frame = pd.read_csv(path)
    if not {"id", "tvt"}.issubset(frame.columns):
        raise RuntimeError(f"{label}: malformed columns {list(frame.columns)}")
    frame = frame[["id", "tvt"]].copy()
    frame["id"] = frame["id"].astype(str)
    frame["tvt"] = pd.to_numeric(frame["tvt"], errors="coerce")
    if frame["id"].duplicated().any() or not np.isfinite(frame["tvt"]).all():
        raise RuntimeError(f"{label}: duplicate ids or non-finite predictions")
    return frame


def _save_global_submission(global_name, filename):
    frame = globals().get(global_name)
    if isinstance(frame, pd.DataFrame) and {"id", "tvt"}.issubset(frame.columns):
        _read_submission_frame = frame[["id", "tvt"]].copy()
        _read_submission_frame.to_csv(WORK / filename, index=False)


def _robust_projection(frame, raw_weight=0.25, fit_weight=0.75, degree=4):
    base = frame.copy()
    base["well"] = base["id"].str.rsplit("_", n=1).str[0]
    base["row_idx"] = base["id"].str.rsplit("_", n=1).str[1].astype(int)
    output = dict(zip(base["id"], base["tvt"].astype(float)))
    applied = 0

    def robfit(x, y):
        if len(x) < degree + 2:
            return y.copy()
        coef = np.polyfit(x, y, degree)
        for _ in range(4):
            residual = y - np.polyval(coef, x)
            scale = np.median(np.abs(residual)) * 1.4826 + 1e-6
            weight = 1.0 / (1.0 + (residual / (2.0 * scale)) ** 2)
            coef = np.polyfit(x, y, degree, w=weight)
        return np.polyval(coef, x)

    for well, group in base.groupby("well", sort=False):
        hw_path = DATA / "test" / f"{well}__horizontal_well.csv"
        hw = pd.read_csv(hw_path)
        known = hw[hw["TVT_input"].notna()]
        if len(known) < 5:
            continue
        last = known.iloc[-1]
        anchor = float(last["TVT_input"]) + float(last["Z"])
        anchor_md = float(last["MD"])
        end_md = float(hw["MD"].iloc[-1])
        ordered = group.sort_values("row_idx")
        ridx = ordered["row_idx"].to_numpy(int)
        md = hw["MD"].to_numpy(float)[ridx]
        z = hw["Z"].to_numpy(float)[ridx]
        progress = (md - anchor_md) / max(end_md - anchor_md, 1e-6)
        tvt = ordered["tvt"].to_numpy(float)
        fitted_surface = anchor + robfit(progress, (tvt + z) - anchor)
        projected = raw_weight * tvt + fit_weight * (fitted_surface - z)
        if not np.isfinite(projected).all():
            raise RuntimeError(f"{well}: non-finite historical projection")
        output.update(dict(zip(ordered["id"], projected)))
        applied += 1

    result = frame[["id"]].copy()
    result["tvt"] = result["id"].map(output).astype(float)
    print("historical projection wells:", applied, flush=True)
    return result


def _slope01(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return 0.0
    return float(np.polyfit(np.linspace(0.0, 1.0, len(values)), values, 1)[0])


def _quantile(values, q):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else np.nan


def _build_cp_features(well, hw, ids, row_idx, tvt_base, training=False, truth=None):
    idx = np.asarray(row_idx, int)
    if len(idx) < 20 or idx.min() < 0 or idx.max() >= len(hw):
        return None, None
    evaluated = hw.iloc[idx].copy()
    md = evaluated["MD"].to_numpy(float)
    z = evaluated["Z"].to_numpy(float)
    gr = evaluated["GR"].to_numpy(float) if "GR" in evaluated else np.full(len(evaluated), np.nan)
    order = np.argsort(md)
    md, z, gr = md[order], z[order], gr[order]
    ids = np.asarray(ids)[order]
    tvt_base = np.asarray(tvt_base, float)[order]
    progress = (md - md[0]) / max(float(md[-1] - md[0]), 1e-6)
    surface = tvt_base + z
    control_surface = np.interp(CP_KNOTS, progress, surface)
    row = {
        "well": well,
        "n_eval": len(idx),
        "md_start": float(md[0]), "md_end": float(md[-1]),
        "md_span": float(md[-1] - md[0]),
        "z_start": float(z[0]), "z_end": float(z[-1]),
        "z_span": float(np.nanmax(z) - np.nanmin(z)), "z_slope": _slope01(z),
        "gr_mean": float(np.nanmean(gr)), "gr_std": float(np.nanstd(gr)),
        "gr_span": float(np.nanmax(gr) - np.nanmin(gr)), "gr_slope": _slope01(gr),
        "base_tvt_mean": float(np.mean(tvt_base)), "base_tvt_std": float(np.std(tvt_base)),
        "base_tvt_span": float(np.max(tvt_base) - np.min(tvt_base)),
        "base_tvt_slope": _slope01(tvt_base),
        "base_s_mean": float(np.mean(surface)), "base_s_std": float(np.std(surface)),
        "base_s_span": float(np.max(surface) - np.min(surface)),
        "base_s_slope": _slope01(surface),
        "base_s_diff_std": float(np.std(np.diff(surface))) if len(surface) > 1 else 0.0,
        "base_s_diff_p95_abs": _quantile(np.abs(np.diff(surface)), 0.95) if len(surface) > 1 else 0.0,
    }
    for number, value in enumerate(control_surface):
        row[f"s_base_ctrl_{number:02d}"] = float(value)
    for number, value in enumerate(np.diff(control_surface)):
        row[f"s_base_ctrl_diff_{number:02d}"] = float(value)

    if {"X", "Y", "Z"}.issubset(evaluated.columns):
        x = evaluated["X"].to_numpy(float)[order]
        y = evaluated["Y"].to_numpy(float)[order]
        dx, dy, dz, dmd = np.diff(x), np.diff(y), np.diff(z), np.diff(md)
        horizontal = np.sqrt(dx * dx + dy * dy)
        build = dz / (dmd + 1e-9)
        heading = np.unwrap(np.arctan2(dy, dx)) if len(dx) else np.array([])
        row.update({
            "x_mean": float(np.nanmean(x)), "y_mean": float(np.nanmean(y)),
            "x_span": float(np.nanmax(x) - np.nanmin(x)),
            "y_span": float(np.nanmax(y) - np.nanmin(y)),
            "path_horiz": float(np.nansum(horizontal)),
            "build_mean": float(np.nanmean(build)) if len(build) else 0.0,
            "build_std": float(np.nanstd(build)) if len(build) else 0.0,
            "build_p95_abs": _quantile(np.abs(build), 0.95) if len(build) else 0.0,
            "heading_turn_std": float(np.nanstd(np.diff(heading))) if len(heading) > 2 else 0.0,
            "heading_turn_p95_abs": _quantile(np.abs(np.diff(heading)), 0.95) if len(heading) > 2 else 0.0,
        })

    known = hw[hw["TVT_input"].notna()] if "TVT_input" in hw else pd.DataFrame()
    row["prefix_n"] = len(known)
    if len(known):
        last = known.iloc[-1]
        for column in ("MD", "X", "Y", "Z", "GR", "TVT_input"):
            if column in known:
                row[f"last_{column}"] = float(last[column])
        tail = known.tail(min(200, len(known)))
        if len(tail) > 5:
            for column in ("Z", "GR", "TVT_input"):
                if column in tail:
                    values = tail[column].to_numpy(float)
                    row[f"prefix_tail_{column}_slope"] = _slope01(values)
                    row[f"prefix_tail_{column}_std"] = float(np.nanstd(values))

    cache = {"id": ids, "progress": progress, "tvt_base": tvt_base}
    if training:
        truth = np.asarray(truth, float)[order]
        target = np.interp(CP_KNOTS, progress, truth + z) - control_surface
        for number, value in enumerate(target):
            row[f"target_{number:02d}"] = float(value)
    return row, cache


# Freeze the 6.39 core and all directly available endpoint files.
selected_weight = float(globals().get("SP45_BLEND_WEIGHT", 0.60))
named_core = WORK / f"submission_sp45_learned_w{selected_weight:.2f}.csv"
if not named_core.exists():
    named_core = WORK / "submission.csv"
core = _read_submission(named_core, "6.39 core")
core.to_csv(WORK / "frontier_6p39_core_before_cp.csv", index=False)

sp45 = _read_submission(WORK / "sp45_projection_submission.csv", "SP45")
learned_path = WORK / "learned_trajectory_submission.csv"
if not learned_path.exists():
    raise FileNotFoundError("Cell 39 must first save learned_trajectory_submission.csv")
learned = _read_submission(learned_path, "learned/FLE")
learned.to_csv(WORK / "fleongg_pretrained_submission.csv", index=False)
_save_global_submission("sub_1", "ridge_pp_submission.csv")
_save_global_submission("sub_2", "selector_submission.csv")

# Reconstruct the historical CP12 parent exactly: 0.55 SP45 + 0.45 FLE,
# followed by the old degree-4, 75%-dose robust projection.
historical = sp45.rename(columns={"tvt": "sp45"}).merge(
    learned.rename(columns={"tvt": "fle"}), on="id", validate="one_to_one"
)
old_parent = historical[["id"]].copy()
old_parent["tvt"] = 0.55 * historical["sp45"] + 0.45 * historical["fle"]
old_parent = _robust_projection(old_parent)
old_parent.to_csv(WORK / "historical_strong_before_cp12.csv", index=False)

# Locate the historical fold-clean OOF used by the original CP12 cell.
oof_candidates = [
    WORK / "strong_base_projected_oof.csv",
    *[Path(path) for path in glob.glob("/kaggle/input/**/strong_base_projected_oof.csv", recursive=True)],
]
oof_path = next((path for path in oof_candidates if path.exists()), None)
if oof_path is None:
    raise FileNotFoundError(
        "Upload the dataset containing strong_base_projected_oof.csv; "
        "this is the original CP12 training bloodline."
    )
oof = pd.read_csv(oof_path)
if "well" not in oof:
    oof["well"] = oof["id"].astype(str).str.rsplit("_", n=1).str[0]
pred_column = next((c for c in ("tvt_base", "pred", "tvt_pred", "prediction") if c in oof), None)
if pred_column is None or "tvt_true" not in oof:
    raise RuntimeError(f"Malformed CP12 OOF columns: {list(oof.columns)}")
oof = oof[["id", "well", "tvt_true", pred_column]].rename(columns={pred_column: "tvt_base"})
oof["id"] = oof["id"].astype(str)
oof["well"] = oof["well"].astype(str)
oof["row_idx"] = oof["id"].str.rsplit("_", n=1).str[1].astype(int)

train_rows = []
for number, (well, group) in enumerate(oof.groupby("well", sort=False), 1):
    hw = pd.read_csv(DATA / "train" / f"{well}__horizontal_well.csv")
    group = group.sort_values("row_idx")
    row, _ = _build_cp_features(
        well, hw, group["id"], group["row_idx"], group["tvt_base"],
        training=True, truth=group["tvt_true"],
    )
    if row is not None:
        train_rows.append(row)
    if number % 100 == 0:
        print(f"CP12 train features {number}/{oof['well'].nunique()}", flush=True)

train = pd.DataFrame(train_rows)
target_columns = [f"target_{number:02d}" for number in range(CP_CTRL_N)]
feature_columns = [
    column for column in train.columns
    if column != "well" and column not in target_columns
    and pd.api.types.is_numeric_dtype(train[column])
]
x_train = train[feature_columns]
y_train = train[target_columns].to_numpy(float)
models = []
for target_number in range(CP_CTRL_N):
    model = make_pipeline(
        SimpleImputer(strategy="median"), RobustScaler(),
        HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=1000),
    )
    model.fit(x_train, y_train[:, target_number])
    models.append(model)
print("CP12 trained:", {"wells": len(train), "features": len(feature_columns)}, flush=True)

test = old_parent.copy()
test["well"] = test["id"].str.rsplit("_", n=1).str[0]
test["row_idx"] = test["id"].str.rsplit("_", n=1).str[1].astype(int)
test_rows, caches = [], {}
for well, group in test.groupby("well", sort=False):
    hw = pd.read_csv(DATA / "test" / f"{well}__horizontal_well.csv")
    group = group.sort_values("row_idx")
    row, cache = _build_cp_features(
        well, hw, group["id"], group["row_idx"], group["tvt"], training=False
    )
    if row is not None:
        test_rows.append(row)
        caches[well] = cache

test_features = pd.DataFrame(test_rows)
for column in feature_columns:
    if column not in test_features:
        test_features[column] = np.nan
control_prediction = np.column_stack([
    model.predict(test_features[feature_columns]) for model in models
])
control_by_well = dict(zip(test_features["well"].astype(str), control_prediction))

moved_parts = []
all_moves = []
for well, cache in caches.items():
    raw_move = np.interp(cache["progress"], CP_KNOTS, control_by_well[well])
    move = np.clip(CP_ALPHA * raw_move, -CP_CAP, CP_CAP)
    moved_parts.append(pd.DataFrame({"id": cache["id"], "tvt": cache["tvt_base"] + move}))
    all_moves.append(move)
moved = pd.concat(moved_parts, ignore_index=True)
cp12 = old_parent[["id", "tvt"]].merge(
    moved.rename(columns={"tvt": "moved"}), on="id", how="left", validate="one_to_one"
)
cp12["tvt"] = cp12["moved"].fillna(cp12["tvt"]).astype(float)
cp12 = cp12[["id", "tvt"]]
cp12.to_csv(WORK / "submission_controlpoint_before_gold.csv", index=False)

# CP12 is a side curve. The current parent remains the 6.39 core.
core.to_csv(WORK / "submission.csv", index=False)
moves = np.concatenate(all_moves) if all_moves else np.array([0.0])
print("CP12 side curve ready:", {
    "oof": str(oof_path), "rows": len(cp12), "wells": test["well"].nunique(),
    "mean_abs_move": float(np.mean(np.abs(moves))),
    "p95_abs_move": float(np.quantile(np.abs(moves), 0.95)),
    "restored_submission": "frontier_6p39_core_before_cp.csv",
}, flush=True)
