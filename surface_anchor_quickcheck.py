from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SURFACES = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def rmse_from_sse(sse, n):
    return float(np.sqrt(sse / max(n, 1)))


rows = []
totals = {}
files = sorted((ROOT / "train").glob("*__horizontal_well.csv"))

for i, path in enumerate(files, 1):
    wid = path.name.split("__")[0]
    hw = pd.read_csv(path, usecols=["Z", "TVT", "TVT_input", *SURFACES])
    known = np.flatnonzero(hw["TVT_input"].notna().to_numpy())
    hidden = np.flatnonzero(hw["TVT_input"].isna().to_numpy())
    if len(known) < 100 or len(hidden) == 0:
        continue

    anchor = int(known[-1])
    y = hw["TVT"].to_numpy(float)[hidden]
    tvt_input = hw["TVT_input"].to_numpy(float)
    z = hw["Z"].to_numpy(float)
    last_tvt = float(tvt_input[anchor])

    pred_by_surface = {}
    replay_rmse = {}

    # Score each structural surface on the final visible-prefix block using
    # an earlier anchor. Deployment itself is re-anchored at the true last
    # visible TVT, so prefix bias is never transferred across the boundary.
    score_n = min(500, len(known) - 20)
    score_idx = known[-score_n:]
    replay_anchor_pos = max(0, len(known) - score_n - 1)
    replay_anchor = int(known[replay_anchor_pos])

    for surface in SURFACES:
        sv = hw[surface].to_numpy(float)
        x = sv - z
        if not np.isfinite(x[[anchor, replay_anchor]]).all():
            continue
        if np.isfinite(x[hidden]).mean() < 0.99:
            continue

        pred_by_surface[surface] = last_tvt + x[hidden] - x[anchor]
        replay = tvt_input[replay_anchor] + x[score_idx] - x[replay_anchor]
        valid = np.isfinite(replay) & np.isfinite(tvt_input[score_idx])
        if valid.sum() >= 50:
            replay_rmse[surface] = float(
                np.sqrt(np.mean((replay[valid] - tvt_input[score_idx][valid]) ** 2))
            )

    if not pred_by_surface:
        continue

    names = list(pred_by_surface)
    matrix = np.column_stack([pred_by_surface[name] for name in names])
    candidates = {
        "surface_mean": np.mean(matrix, axis=1),
        "surface_median": np.median(matrix, axis=1),
    }

    if replay_rmse:
        best = min(replay_rmse, key=replay_rmse.get)
        candidates["surface_prefix_best"] = pred_by_surface[best]

        common = [name for name in names if name in replay_rmse]
        loss = np.array([replay_rmse[name] for name in common], float)
        pred_common = np.column_stack([pred_by_surface[name] for name in common])
        for tau in (0.25, 0.5, 1.0, 2.0):
            w = np.exp(-(loss - loss.min()) / tau)
            w /= w.sum()
            candidates[f"surface_soft_tau{tau:g}"] = pred_common @ w

    for surface, pred in pred_by_surface.items():
        candidates[f"surface_{surface}"] = pred

    well_row = {"well": wid, "n": len(hidden)}
    for name, pred in candidates.items():
        valid = np.isfinite(pred) & np.isfinite(y)
        if not valid.any():
            continue
        sse = float(np.sum((pred[valid] - y[valid]) ** 2))
        n = int(valid.sum())
        totals.setdefault(name, [0.0, 0])
        totals[name][0] += sse
        totals[name][1] += n
        well_row[f"rmse_{name}"] = rmse_from_sse(sse, n)
    rows.append(well_row)

    if i % 100 == 0:
        print(f"{i}/{len(files)}")

summary = pd.DataFrame(
    [
        {"candidate": name, "rmse": rmse_from_sse(sse, n), "rows": n}
        for name, (sse, n) in totals.items()
    ]
).sort_values("rmse")

print(summary.to_string(index=False))
pd.DataFrame(rows).to_csv(ROOT / "surface_anchor_quickcheck_bywell.csv", index=False)
summary.to_csv(ROOT / "surface_anchor_quickcheck_summary.csv", index=False)
