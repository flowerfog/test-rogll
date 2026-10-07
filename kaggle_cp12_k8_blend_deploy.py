from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


warnings.filterwarnings("ignore")

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
DATA = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
if "CFG" in globals():
    DATA = Path(getattr(CFG, "DATA", getattr(CFG, "dataset_path", DATA)))


def find_bundle():
    direct = [
        WORK / "strong_exact_cpfam_k8_bundle.joblib",
        Path("strong_exact_cpfam_k8_bundle.joblib"),
    ]
    for path in direct:
        if path.exists():
            return path
    found = list(Path("/kaggle/input").rglob("strong_exact_cpfam_k8_bundle.joblib"))
    if found:
        return found[0]
    raise RuntimeError("strong_exact_cpfam_k8_bundle.joblib not found")


def read_submission(path, name, sample):
    path = Path(path)
    if not path.exists():
        return None
    d = pd.read_csv(path)
    if not {"id", "tvt"}.issubset(d.columns):
        raise RuntimeError(f"bad submission columns: {path} {list(d.columns)}")
    d = d[["id", "tvt"]].rename(columns={"tvt": name})
    d["id"] = d["id"].astype(str)
    out = sample[["id"]].merge(d, on="id", how="left")
    if out[name].isna().any() or not np.isfinite(out[name].to_numpy(float)).all():
        raise RuntimeError(f"missing/nonfinite {name}: {path}")
    return out


def submission_from_global(obj, name, sample):
    if obj is None or not isinstance(obj, pd.DataFrame):
        return None
    if not {"id", "tvt"}.issubset(obj.columns):
        return None
    d = obj[["id", "tvt"]].rename(columns={"tvt": name}).copy()
    d["id"] = d["id"].astype(str)
    out = sample[["id"]].merge(d, on="id", how="left")
    if out[name].isna().any() or not np.isfinite(out[name].to_numpy(float)).all():
        return None
    return out


def first_submission(name, sample, global_names, paths):
    for global_name in global_names:
        d = submission_from_global(globals().get(global_name), name, sample)
        if d is not None:
            print(f"loaded {name} from global {global_name}")
            return d
    for path in paths:
        d = read_submission(path, name, sample)
        if d is not None:
            print(f"loaded {name} from {path}")
            return d
    raise RuntimeError(f"cannot locate exact test candidate: {name}")


def safe_stats(v, prefix):
    v = np.asarray(v, float)
    x = v[np.isfinite(v)]
    if len(x) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_span": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_p10": float(np.quantile(x, 0.10)),
        f"{prefix}_p50": float(np.quantile(x, 0.50)),
        f"{prefix}_p90": float(np.quantile(x, 0.90)),
        f"{prefix}_span": float(np.max(x) - np.min(x)),
    }


bundle_path = find_bundle()
bundle = joblib.load(bundle_path)
model = bundle["model"]
feature_cols = list(bundle["feature_cols"])
aux_cols = list(bundle["aux_cols"])
N_CP = int(bundle["n_cp"])
WIN_ROWS = int(bundle["win_rows"])
CP_ALPHA = float(bundle["cp_alpha"])
CP_CAP = float(bundle["cp_cap"])
BLEND_CP12 = float(bundle["blend_cp12"])
BLEND_K8 = float(bundle["blend_k8"])

print("DATA:", DATA)
print("WORK:", WORK)
print("bundle:", bundle_path)
print({k: v for k, v in bundle.items() if k != "model"})

sample = pd.read_csv(DATA / "sample_submission.csv")
sample["id"] = sample["id"].astype(str)
sample["well"] = sample["id"].str.rsplit("_", n=1).str[0]
sample["row_idx"] = sample["id"].str.rsplit("_", n=1).str[1].astype(np.int32)

# Strong is the exact pre-CP base. CP12 must already have been run and saved.
candidate_sources = {
    "strong": first_submission(
        "strong", sample, [], [WORK / "submission_before_gold.csv"]
    ),
    "cp12": first_submission(
        "cp12", sample, [], [
            WORK / "submission_controlpoint_before_gold.csv",
            WORK / "submission_cp12_before_gold.csv",
        ]
    ),
    "ridge": first_submission(
        "ridge", sample, ["sub_1"], [WORK / "ridge_pp_submission.csv"]
    ),
    "selector": first_submission(
        "selector", sample, ["sub_2"], [WORK / "selector_submission.csv"]
    ),
    "fleongg": first_submission(
        "fleongg", sample, [], [WORK / "fleongg_pretrained_submission.csv"]
    ),
    "sp45": first_submission(
        "sp45", sample, [], [WORK / "sp45_projection_submission.csv"]
    ),
}

test = sample[["id", "well", "row_idx"]].copy()
for name, d in candidate_sources.items():
    test = test.merge(d[["id", name]], on="id", how="left")

required_curves = ["strong", *aux_cols]
missing_curves = [c for c in required_curves if c not in test.columns]
if missing_curves:
    raise RuntimeError(f"missing candidate curves: {missing_curves}")

for c in required_curves:
    if test[c].isna().any() or not np.isfinite(test[c].to_numpy(float)).all():
        raise RuntimeError(f"nonfinite candidate: {c}")

test = test.sort_values(["well", "row_idx"]).reset_index(drop=True)
base_vec = test["strong"].to_numpy(np.float32)
candidate_arrays = {c: test[c].to_numpy(np.float32) for c in aux_cols}
row_idx_vec = test["row_idx"].to_numpy(np.int32)


def load_hw_features(wid):
    path = DATA / "test" / f"{wid}__horizontal_well.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    hw = pd.read_csv(path)
    for c in ["MD", "X", "Y", "Z", "GR", "TVT_input"]:
        if c not in hw.columns:
            hw[c] = np.nan
    md = hw["MD"].to_numpy(float)
    x = hw["X"].to_numpy(float)
    y = hw["Y"].to_numpy(float)
    z = hw["Z"].to_numpy(float)
    gr = hw["GR"].to_numpy(float)
    tvt_input = hw["TVT_input"].to_numpy(float)
    dmd = np.gradient(md)
    dmd = np.where(np.abs(dmd) < 1e-6, np.nan, dmd)
    build = np.gradient(z) / dmd
    dx = np.gradient(x) / dmd
    dy = np.gradient(y) / dmd
    dogleg = np.sqrt(
        np.nan_to_num(np.gradient(dx), nan=0.0) ** 2
        + np.nan_to_num(np.gradient(dy), nan=0.0) ** 2
        + np.nan_to_num(np.gradient(build), nan=0.0) ** 2
    )
    gr_fill = pd.Series(gr).interpolate(limit_direction="both").fillna(0).to_numpy(float)
    gr_d1 = np.gradient(gr_fill)
    gr_d2 = np.gradient(gr_d1)
    known = np.flatnonzero(np.isfinite(tvt_input))
    last_known_idx = int(known[-1]) if len(known) else 0
    n = len(hw)
    return {
        "n_total": n,
        "MD": md, "X": x, "Y": y, "Z": z, "GR": gr,
        "build": build, "dogleg": dogleg,
        "gr_d1": gr_d1, "gr_d2": gr_d2,
        "pos": np.arange(n, dtype=float) / max(n - 1, 1),
        "since_anchor_norm": (
            np.arange(n, dtype=float) - last_known_idx
        ) / max(n - last_known_idx - 1, 1),
        "prefix_n": int(len(known)),
    }


records = []
caches = {}
t0 = time.time()
group_items = list(test.groupby("well", sort=False).indices.items())

for wi, (wid, idx) in enumerate(group_items, 1):
    idx = np.asarray(idx, dtype=np.int64)
    n = len(idx)
    if n < 32:
        continue
    hwf = load_hw_features(wid)
    ridx = row_idx_vec[idx]
    if ridx.min() < 0 or ridx.max() >= hwf["n_total"]:
        continue
    centers = np.unique(np.round(np.linspace(0, n - 1, N_CP)).astype(int))
    row_norm = np.arange(n, dtype=float) / max(n - 1, 1)
    caches[wid] = {"idx": idx, "n": n}

    for order, center_local in enumerate(centers):
        lo = max(0, center_local - WIN_ROWS)
        hi = min(n, center_local + WIN_ROWS + 1)
        local_positions = np.arange(lo, hi)
        loc = idx[local_positions]
        ridx_win = ridx[local_positions]
        center_global = int(idx[center_local])
        center_ridx = int(ridx[center_local])
        base_window = base_vec[loc]
        feat = {
            "well": wid,
            "center_global": center_global,
            "center_local": int(center_local),
            "n_well": int(n),
            "n_window": int(len(loc)),
            "cp_order": int(order),
            "cp_order_norm": float(order / max(len(centers) - 1, 1)),
            "row_norm": float(row_norm[center_local]),
            "prefix_n": hwf["prefix_n"],
            "center_GR": float(np.nan_to_num(hwf["GR"][center_ridx], nan=0.0)),
            "center_Z": float(np.nan_to_num(hwf["Z"][center_ridx], nan=0.0)),
            "center_build": float(np.nan_to_num(hwf["build"][center_ridx], nan=0.0)),
            "center_since_anchor_norm": float(
                np.nan_to_num(hwf["since_anchor_norm"][center_ridx], nan=0.0)
            ),
            "base_center": float(base_vec[center_global]),
            "base_slope_win": float(
                (base_window[-1] - base_window[0]) / max(len(base_window) - 1, 1)
            ),
        }
        for col in [
            "MD", "X", "Y", "Z", "GR", "build", "dogleg",
            "gr_d1", "gr_d2", "pos", "since_anchor_norm",
        ]:
            feat.update(safe_stats(hwf[col][ridx_win], col))
        feat.update(safe_stats(base_window, "base"))
        pred_stack = np.column_stack([candidate_arrays[c][loc] for c in aux_cols])
        feat["pred_std_mean"] = float(np.mean(np.std(pred_stack, axis=1)))
        feat["pred_range_mean"] = float(
            np.mean(np.max(pred_stack, axis=1) - np.min(pred_stack, axis=1))
        )
        for c in aux_cols:
            diff = candidate_arrays[c][loc] - base_window
            feat.update(safe_stats(diff, f"{c}_minus_base"))
            feat[f"{c}_minus_base_center"] = float(
                candidate_arrays[c][center_global] - base_vec[center_global]
            )
            feat[f"{c}_minus_base_slope"] = float(
                (diff[-1] - diff[0]) / max(len(diff) - 1, 1)
            )
        records.append(feat)

    if wi % 25 == 0:
        print(f"K8 inference features {wi}/{len(group_items)} elapsed={time.time()-t0:.1f}s", flush=True)

cp_test = pd.DataFrame(records)
for c in feature_cols:
    if c not in cp_test.columns:
        cp_test[c] = 0.0

X_test = (
    cp_test[feature_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .astype(np.float32)
)
cp_test["pred_resid"] = model.predict(X_test).astype(np.float32)

raw_move = np.zeros(len(test), dtype=np.float32)
for wid, cache in caches.items():
    idx = cache["idx"]
    n = cache["n"]
    sub = cp_test[cp_test["well"].eq(wid)].sort_values("center_local")
    xp = sub["center_local"].to_numpy(float)
    fp = sub["pred_resid"].to_numpy(float)
    keep = np.r_[True, np.diff(xp) > 0]
    xp, fp = xp[keep], fp[keep]
    move = np.full(n, fp[0], dtype=float) if len(xp) == 1 else np.interp(
        np.arange(n, dtype=float), xp, fp
    )
    sigma = max(4.0, n / N_CP / 8.0)
    raw_move[idx] = gaussian_filter1d(move, sigma=sigma, mode="nearest").astype(np.float32)

k8_move = np.clip(CP_ALPHA * raw_move, -CP_CAP, CP_CAP)
k8_pred = base_vec + k8_move
cp12_pred = test["cp12"].to_numpy(np.float32)
blend_pred = BLEND_CP12 * cp12_pred + BLEND_K8 * k8_pred

k8_out = test[["id"]].copy()
k8_out["tvt"] = k8_pred.astype(float)
k8_out.to_csv(WORK / "submission_strong_exact_k8_before_gold.csv", index=False)

blend_out = test[["id"]].copy()
blend_out["tvt"] = blend_pred.astype(float)
blend_out.to_csv(WORK / "submission_cp12_k8_equal_blend_before_gold.csv", index=False)
blend_out.to_csv(WORK / "submission.csv", index=False)

diag = test[["id", "well", "row_idx"]].copy()
diag["strong"] = base_vec
diag["cp12"] = cp12_pred
diag["k8"] = k8_pred
diag["blend"] = blend_pred
diag["k8_move"] = k8_move
diag["blend_move_vs_cp12"] = blend_pred - cp12_pred
diag.to_csv(WORK / "cp12_k8_equal_blend_diag.csv", index=False)

summary = {
    "rows": int(len(blend_out)),
    "wells": int(test["well"].nunique()),
    "mean_abs_k8_move_vs_strong": float(np.mean(np.abs(k8_move))),
    "p95_abs_k8_move_vs_strong": float(np.quantile(np.abs(k8_move), 0.95)),
    "mean_abs_blend_move_vs_cp12": float(np.mean(np.abs(blend_pred - cp12_pred))),
    "p95_abs_blend_move_vs_cp12": float(np.quantile(np.abs(blend_pred - cp12_pred), 0.95)),
    "output": str(WORK / "submission_cp12_k8_equal_blend_before_gold.csv"),
}
(WORK / "cp12_k8_equal_blend_audit.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print("CP12/K8 equal blend ready BEFORE Gold")
print(summary)
print("Next: run the original Gold conservative cell. Do not run CP12 again.")
