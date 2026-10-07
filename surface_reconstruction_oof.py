from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parent
SURFACES = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
SAMPLE_EVERY = 40
DEGREES = [1, 2, 3]


def load_training_points(files):
    parts = []
    meta = []
    for i, path in enumerate(files):
        wid = path.name.split("__")[0]
        use = ["X", "Y", *SURFACES]
        d = pd.read_csv(path, usecols=use).iloc[::SAMPLE_EVERY].copy()
        d = d.dropna(subset=["X", "Y", *SURFACES])
        if len(d) == 0:
            continue
        d["well"] = wid
        parts.append(d)
        meta.append({"well": wid, "x": d["X"].mean(), "y": d["Y"].mean()})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(meta)


def make_model(degree):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1.0),
    )


def score_fold(model, valid_files, degree, fold_name):
    rows = []
    total_sse = {"mean": 0.0, "median": 0.0}
    total_n = 0
    for path in valid_files:
        wid = path.name.split("__")[0]
        d = pd.read_csv(path, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
        known = np.flatnonzero(d["TVT_input"].notna().to_numpy())
        hidden = np.flatnonzero(d["TVT_input"].isna().to_numpy())
        if len(known) < 2 or len(hidden) == 0:
            continue
        anchor = int(known[-1])
        surface_hat = model.predict(d[["X", "Y"]].to_numpy(float))
        xhat = surface_hat - d["Z"].to_numpy(float)[:, None]
        paths = (
            float(d["TVT_input"].iloc[anchor])
            + xhat[hidden]
            - xhat[anchor][None, :]
        )
        candidates = {
            "mean": paths.mean(axis=1),
            "median": np.median(paths, axis=1),
        }
        y = d["TVT"].to_numpy(float)[hidden]
        one = {"well": wid, "degree": degree, "fold": fold_name, "n": len(y)}
        for name, pred in candidates.items():
            sse = float(np.sum((pred - y) ** 2))
            total_sse[name] += sse
            one[f"sse_{name}"] = sse
            one[f"rmse_{name}"] = float(np.sqrt(sse / len(y)))
        total_n += len(y)
        rows.append(one)
    return rows, total_sse, total_n


files = sorted((ROOT / "train").glob("*__horizontal_well.csv"))
points, meta = load_training_points(files)
well_to_file = {p.name.split("__")[0]: p for p in files}
print("points:", points.shape, "wells:", meta.shape)

X = points[["X", "Y"]].to_numpy(float)
Y = points[SURFACES].to_numpy(float)
groups = points["well"].to_numpy()

# Random held-out-well CV approximates an interspersed hidden test set.
gkf = GroupKFold(n_splits=5)
random_splits = list(gkf.split(X, Y, groups))

# Spatial-block CV is intentionally harsher and checks geographic extrapolation.
km = KMeans(n_clusters=5, random_state=42, n_init=20)
meta["spatial_fold"] = km.fit_predict(meta[["x", "y"]])
spatial_map = dict(zip(meta["well"], meta["spatial_fold"]))
point_spatial = np.array([spatial_map[w] for w in groups])
spatial_splits = [
    (np.flatnonzero(point_spatial != f), np.flatnonzero(point_spatial == f))
    for f in range(5)
]

all_rows = []
summary_rows = []
t0 = time.time()

for split_name, splits in [("group", random_splits), ("spatial", spatial_splits)]:
    for degree in DEGREES:
        degree_sse = {"mean": 0.0, "median": 0.0}
        degree_n = 0
        fold_gains = []
        for fold, (tr, va) in enumerate(splits, 1):
            model = make_model(degree)
            model.fit(X[tr], Y[tr])
            valid_wells = sorted(set(groups[va]))
            valid_files = [well_to_file[w] for w in valid_wells]
            rows, sse, n = score_fold(model, valid_files, degree, f"{split_name}_{fold}")
            all_rows.extend(rows)
            degree_n += n
            for name in degree_sse:
                degree_sse[name] += sse[name]
            print(split_name, "degree", degree, "fold", fold,
                  "mean_rmse", np.sqrt(sse["mean"] / n),
                  "median_rmse", np.sqrt(sse["median"] / n), flush=True)
        for name, sse in degree_sse.items():
            summary_rows.append({
                "split": split_name,
                "degree": degree,
                "aggregate": name,
                "rmse": float(np.sqrt(sse / degree_n)),
                "rows": degree_n,
            })

summary = pd.DataFrame(summary_rows).sort_values(["split", "rmse"])
print(summary.to_string(index=False))
pd.DataFrame(all_rows).to_csv(ROOT / "surface_reconstruction_oof_bywell.csv", index=False)
summary.to_csv(ROOT / "surface_reconstruction_oof_summary.csv", index=False)
print("elapsed", time.time() - t0)
