from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
PREFIX_EVERY = 20
EVAL_EVERY = 25
QUERY_NEIGHBORS = 192
IDW_K = (8, 32, 64)


def rmse_from_sse(sse, n):
    return float(np.sqrt(sse / max(n, 1)))


files = sorted((ROOT / "train").glob("*__horizontal_well.csv"))
well_to_code = {p.name.split("__")[0]: i for i, p in enumerate(files)}

source_xy = []
source_datum = []
source_center_mean = []
source_center_last = []
source_well = []
well_frames = {}

for i, path in enumerate(files, 1):
    wid = path.name.split("__")[0]
    d = pd.read_csv(path, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
    well_frames[wid] = d
    known = np.flatnonzero(d["TVT_input"].notna().to_numpy())
    if len(known) == 0:
        continue
    take = known[::PREFIX_EVERY]
    if take[-1] != known[-1]:
        take = np.append(take, known[-1])
    xy = d[["X", "Y"]].to_numpy(float)[take]
    datum = (
        d["TVT_input"].to_numpy(float)[take]
        + d["Z"].to_numpy(float)[take]
    )
    ok = np.isfinite(xy).all(axis=1) & np.isfinite(datum)
    source_xy.append(xy[ok])
    source_datum.append(datum[ok])
    source_center_mean.append(datum[ok] - np.mean(datum[ok]))
    source_center_last.append(datum[ok] - datum[ok][-1])
    source_well.append(np.full(ok.sum(), well_to_code[wid], dtype=np.int32))
    if i % 100 == 0:
        print(f"load {i}/{len(files)}", flush=True)

source_xy = np.concatenate(source_xy)
source_datum = np.concatenate(source_datum)
source_center_mean = np.concatenate(source_center_mean)
source_center_last = np.concatenate(source_center_last)
source_well = np.concatenate(source_well)
tree = cKDTree(source_xy, leafsize=64)
print("source prefix points:", len(source_xy), "wells:", len(files))


def other_well_neighbors(query_xy, target_code, need_k):
    query_k = min(max(QUERY_NEIGHBORS, need_k + 8), len(source_xy))
    dist, idx = tree.query(query_xy, k=query_k, workers=-1)
    if query_k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    keep = source_well[idx] != target_code
    out_dist = np.full((len(query_xy), need_k), np.nan, dtype=float)
    out_idx = np.full((len(query_xy), need_k), -1, dtype=np.int64)
    for row in range(len(query_xy)):
        valid = np.flatnonzero(keep[row])[:need_k]
        if len(valid):
            out_dist[row, : len(valid)] = dist[row, valid]
            out_idx[row, : len(valid)] = idx[row, valid]
    return out_dist, out_idx


totals = {"constant_anchor": [0.0, 0]}
for k in IDW_K:
    totals[f"prefix_idw_k{k}"] = [0.0, 0]
    totals[f"prefix_center_mean_k{k}"] = [0.0, 0]
    totals[f"prefix_center_last_k{k}"] = [0.0, 0]

distance_rows = []
well_rows = []
t0 = time.time()

for i, (wid, d) in enumerate(well_frames.items(), 1):
    known = np.flatnonzero(d["TVT_input"].notna().to_numpy())
    hidden = np.flatnonzero(d["TVT_input"].isna().to_numpy())
    if len(known) == 0 or len(hidden) == 0:
        continue
    hidden = hidden[::EVAL_EVERY]
    anchor = int(known[-1])
    query_idx = np.concatenate(([anchor], hidden))
    query_xy = d[["X", "Y"]].to_numpy(float)[query_idx]
    z = d["Z"].to_numpy(float)
    truth = d["TVT"].to_numpy(float)[hidden]
    anchor_datum = float(d["TVT_input"].iloc[anchor] + z[anchor])

    dist, idx = other_well_neighbors(query_xy, well_to_code[wid], max(IDW_K))
    nearest = dist[1:, 0]
    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) == 0:
        continue
    distance_rows.extend(
        {"well": wid, "distance": float(value)} for value in nearest
    )

    constant_pred = anchor_datum - z[hidden]
    constant_sse = float(np.sum((constant_pred - truth) ** 2))
    totals["constant_anchor"][0] += constant_sse
    totals["constant_anchor"][1] += len(hidden)
    wr = {
        "well": wid,
        "n": len(hidden),
        "nearest_median": float(np.median(nearest)),
        "nearest_p90": float(np.quantile(nearest, 0.9)),
        "rmse_constant_anchor": rmse_from_sse(constant_sse, len(hidden)),
    }

    for k in IDW_K:
        dk = dist[:, :k]
        ik = idx[:, :k]
        valid = np.isfinite(dk) & (ik >= 0)
        weights = np.where(valid, 1.0 / np.maximum(dk, 10.0) ** 2, 0.0)
        denom = weights.sum(axis=1)
        field = np.full(len(query_idx), np.nan, dtype=float)
        good = denom > 0
        safe_idx = np.maximum(ik, 0)
        field[good] = (
            weights[good] * source_datum[safe_idx[good]]
        ).sum(axis=1) / denom[good]
        pred_datum = anchor_datum + field[1:] - field[0]
        pred_tvt = pred_datum - z[hidden]
        ok = np.isfinite(pred_tvt) & np.isfinite(truth)
        sse = float(np.sum((pred_tvt[ok] - truth[ok]) ** 2))
        totals[f"prefix_idw_k{k}"][0] += sse
        totals[f"prefix_idw_k{k}"][1] += int(ok.sum())
        wr[f"rmse_prefix_idw_k{k}"] = rmse_from_sse(sse, int(ok.sum()))

        for center_name, center_values in (
            ("center_mean", source_center_mean),
            ("center_last", source_center_last),
        ):
            centered_field = np.full(len(query_idx), np.nan, dtype=float)
            centered_field[good] = (
                weights[good] * center_values[safe_idx[good]]
            ).sum(axis=1) / denom[good]
            centered_pred_datum = (
                anchor_datum + centered_field[1:] - centered_field[0]
            )
            centered_pred_tvt = centered_pred_datum - z[hidden]
            centered_ok = np.isfinite(centered_pred_tvt) & np.isfinite(truth)
            centered_sse = float(
                np.sum((centered_pred_tvt[centered_ok] - truth[centered_ok]) ** 2)
            )
            key = f"prefix_{center_name}_k{k}"
            totals[key][0] += centered_sse
            totals[key][1] += int(centered_ok.sum())
            wr[f"rmse_{key}"] = rmse_from_sse(centered_sse, int(centered_ok.sum()))
    well_rows.append(wr)

    if i % 100 == 0:
        print(f"audit {i}/{len(well_frames)} elapsed={time.time() - t0:.1f}s", flush=True)

distances = pd.DataFrame(distance_rows)
by_well = pd.DataFrame(well_rows)
summary = pd.DataFrame(
    [
        {"candidate": name, "rmse": rmse_from_sse(sse, n), "rows": n}
        for name, (sse, n) in totals.items()
    ]
).sort_values("rmse")

print("\nOTHER-WELL PREFIX DISTANCE QUANTILES")
print(distances["distance"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_string())
print("\nNAIVE PREFIX-SPATIAL FIELD SANITY")
print(summary.to_string(index=False))

distances.to_csv(ROOT / "spatial_prefix_distance_rows.csv", index=False)
by_well.to_csv(ROOT / "spatial_prefix_transduction_bywell.csv", index=False)
summary.to_csv(ROOT / "spatial_prefix_transduction_summary.csv", index=False)
print("elapsed", time.time() - t0)
