from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor


ROOT = Path(__file__).resolve().parent
SAMPLE_EVERY = 80
NEIGHBORS = [16, 64]


files = sorted((ROOT / "train").glob("*__horizontal_well.csv"))
parts = []
well_files = {}
for path in files:
    wid = path.name.split("__")[0]
    well_files[wid] = path
    d = pd.read_csv(path, usecols=["X", "Y", "Z", "TVT"]).iloc[::SAMPLE_EVERY].copy()
    d["datum"] = d["TVT"] + d["Z"]
    d["well"] = wid
    parts.append(d[["X", "Y", "datum", "well"]])

points = pd.concat(parts, ignore_index=True).dropna()
X = points[["X", "Y"]].to_numpy(float)
y = points["datum"].to_numpy(float)
groups = points["well"].to_numpy()
print("points", points.shape, "wells", len(well_files))

gkf = GroupKFold(n_splits=5)
totals = {k: [0.0, 0] for k in NEIGHBORS}
fold_rows = []
t0 = time.time()

for fold, (tr, va) in enumerate(gkf.split(X, y, groups), 1):
    valid_wells = sorted(set(groups[va]))
    models = {}
    for k in NEIGHBORS:
        model = KNeighborsRegressor(
            n_neighbors=k,
            weights="distance",
            algorithm="kd_tree",
            leaf_size=60,
            n_jobs=-1,
        )
        model.fit(X[tr], y[tr])
        models[k] = model

    fold_sse = {k: 0.0 for k in NEIGHBORS}
    fold_n = 0
    for wid in valid_wells:
        d = pd.read_csv(well_files[wid], usecols=["X", "Y", "Z", "TVT", "TVT_input"])
        known = np.flatnonzero(d["TVT_input"].notna().to_numpy())
        hidden = np.flatnonzero(d["TVT_input"].isna().to_numpy())
        if len(known) == 0 or len(hidden) == 0:
            continue
        anchor = int(known[-1])
        query_idx = np.concatenate(([anchor], hidden))
        q = d[["X", "Y"]].to_numpy(float)[query_idx]
        z = d["Z"].to_numpy(float)
        truth = d["TVT"].to_numpy(float)[hidden]
        anchor_datum = float(d["TVT_input"].iloc[anchor] + z[anchor])
        for k, model in models.items():
            dhat = model.predict(q)
            pred_datum = anchor_datum + dhat[1:] - dhat[0]
            pred_tvt = pred_datum - z[hidden]
            sse = float(np.sum((pred_tvt - truth) ** 2))
            fold_sse[k] += sse
            totals[k][0] += sse
            totals[k][1] += len(hidden)
        fold_n += len(hidden)

    for k in NEIGHBORS:
        fold_rows.append({
            "fold": fold,
            "k": k,
            "rmse": float(np.sqrt(fold_sse[k] / fold_n)),
            "rows": fold_n,
        })
    print("fold", fold, {k: np.sqrt(fold_sse[k] / fold_n) for k in NEIGHBORS}, flush=True)

summary = pd.DataFrame([
    {"k": k, "rmse": float(np.sqrt(sse / n)), "rows": n}
    for k, (sse, n) in totals.items()
]).sort_values("rmse")
print(pd.DataFrame(fold_rows).to_string(index=False))
print(summary.to_string(index=False))
print("elapsed", time.time() - t0)
