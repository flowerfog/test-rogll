"""V153: one-run PF seed posterior reweighting by visible-prefix continuity."""
from __future__ import annotations

import json
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

STARTED = time.time()
ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
OUT_DIR = ROOT / "oof_exports" / "pf_seed_prefix_reweight_v153"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / "mainline_pf_repeatability_v97_cell.py"
if not SOURCE.is_file():
    SOURCE = Path.cwd() / "mainline_pf_repeatability_v97_cell.py"
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

# Bootstrap only definitions and mainline discovery; do not execute V97's
# repeatability experiment.
source = SOURCE.read_text(encoding="utf-8")
marker = "started_p5 = time.time()"
if marker not in source:
    raise RuntimeError("V97 bootstrap marker missing")
module = types.ModuleType("v153_mainline_defs")
module.__file__ = str(SOURCE)
exec(compile(source.split(marker, 1)[0] + "\n", str(SOURCE), "exec"), module.__dict__)

CFG = module.CFG
DATA = Path(CFG.DATA)
if not callable(module.load_well) or not callable(module._pf_lik_allseeds):
    raise RuntimeError("Mainline PF seed interface is unavailable")

OOF_CANDIDATES = (
    ROOT / "oof_exports" / "frontier_6p377_full_fivefold_lineage_v66" / "incumbent_6p377_full_fivefold_oof.parquet",
    ROOT / "oof_exports" / "frontier_6p39_full_oof_cp_increment_v18" / "full_oof_cp_endpoints.parquet",
)
oof_path = next((path for path in OOF_CANDIDATES if path.is_file()), None)
if oof_path is None:
    raise FileNotFoundError(OOF_CANDIDATES)
oof = pd.read_parquet(oof_path, columns=["well", "fold"])
oof["well"] = oof["well"].astype(str)
fold_map = oof.groupby("well", as_index=False)["fold"].first()
fold3 = sorted(fold_map.loc[fold_map["fold"].eq(3), "well"].tolist())
discovery_wells = fold3[:24]
confirmation_wells = fold3[24:]
if len(confirmation_wells) < 120:
    raise RuntimeError(f"Fold3 coverage too small: {len(discovery_wells)}/{len(confirmation_wells)}")

N_SEEDS = min(64, int(getattr(CFG, "PF_SEEDS", 64)))
N_PARTICLES = int(getattr(CFG, "PF_PARTICLES", 350))
SCALE = 5.0
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)
HORIZONS = (128.0, 256.0)


def _one(well):
    hw, tw = module.load_well(well)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(kn) < 30 or len(ev) < 128:
        raise RuntimeError(f"{well}: insufficient prefix/evaluation rows")
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    last = kn.iloc[-1]
    ls = float(last["TVT_input"]) + float(last["Z"])
    tw_at_known = np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).to_numpy(float) - tw_at_known), 10.0, 60.0))
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(float))
    dz = np.diff(tail["Z"].to_numpy(float))
    dm = np.diff(tail["MD"].to_numpy(float))
    valid = dm > 0
    ir = float(np.median((dt[valid] + dz[valid]) / dm[valid])) if valid.sum() >= 3 else 0.0
    gg, gmin, gst = module._grid(tw_tvt, tw_gr)
    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy(float)
    gr_eval = gr_all[ev.index.to_numpy()]
    seed_base = module._stable_seed_p5(well, 153)
    preds, liks = module._pf_lik_allseeds(
        ev["MD"].to_numpy(float), ev["Z"].to_numpy(float), gr_eval,
        gg, gmin, gst, gs, ls, ir, N_PARTICLES, N_SEEDS, seed_base,
        0.998, 0.002, 0.005, 0.1, 0.001, 0.5, 4.5,
    )
    distance = ev["MD"].to_numpy(float) - float(last["MD"])
    truth = ev["TVT"].to_numpy(float)
    scores = []
    for horizon in HORIZONS:
        keep = distance <= horizon
        def fit_slope(x, y):
            if len(x) < 8 or np.ptp(x) < 16.0:
                return 0.0
            return float(np.polyfit(x - x[-1], y, 1)[0])
        slopes = np.asarray([fit_slope(distance[keep], row[keep]) for row in preds]) if keep.sum() >= 8 else np.zeros(N_SEEDS)
        scale = max(float(np.nanstd(slopes)), 0.002)
        scores.append(np.abs(slopes - ir) / scale)
    continuity = np.mean(np.stack(scores), axis=0)
    return {"well": well, "distance": distance, "truth": truth, "preds": preds, "liks": liks, "continuity": continuity}


def _weighted(preds, log_likelihood, continuity, beta):
    logw = (log_likelihood - np.max(log_likelihood)) / SCALE - beta * continuity
    logw -= np.max(logw)
    weights = np.exp(logw)
    weights /= np.sum(weights)
    return np.sum(weights[:, None] * preds, axis=0)


def _rmse(values, truth):
    return float(np.sqrt(np.mean((np.asarray(values) - np.asarray(truth)) ** 2)))


def _evaluate(wells):
    rows = []
    for number, well in enumerate(wells, 1):
        record = _one(well)
        base = _weighted(record["preds"], record["liks"], record["continuity"], 0.0)
        for beta in BETAS:
            pred = _weighted(record["preds"], record["liks"], record["continuity"], beta)
            rows.append({"well": well, "beta": beta, "rmse": _rmse(pred, record["truth"]), "far_rmse": _rmse(pred[record["distance"] > 2000], record["truth"][record["distance"] > 2000])})
        oracle = np.asarray([_rmse(row, record["truth"]) for row in record["preds"]])
        shuffled = _weighted(record["preds"], record["liks"], record["continuity"][np.random.default_rng(153).permutation(len(record["continuity"]))], 1.0)
        rows.append({"well": well, "beta": "baseline", "rmse": _rmse(base, record["truth"]), "far_rmse": _rmse(base[record["distance"] > 2000], record["truth"][record["distance"] > 2000])})
        rows.append({"well": well, "beta": "oracle", "rmse": float(np.min(oracle)), "far_rmse": float(np.min([_rmse(row[record["distance"] > 2000], record["truth"][record["distance"] > 2000]) for row in record["preds"]]))})
        rows.append({"well": well, "beta": "shuffled", "rmse": _rmse(shuffled, record["truth"]), "far_rmse": _rmse(shuffled[record["distance"] > 2000], record["truth"][record["distance"] > 2000])})
        if number % 10 == 0 or number == len(wells):
            print(f"V153 PF seed audit {number}/{len(wells)}", flush=True)
    return pd.DataFrame(rows)


discovery = _evaluate(discovery_wells)
confirmation = _evaluate(confirmation_wells)
discovery.to_csv(OUT_DIR / "discovery_bywell.csv", index=False)
confirmation.to_csv(OUT_DIR / "confirmation_bywell.csv", index=False)

def _summary(part, split):
    out = []
    for beta, local in part.groupby("beta", sort=False):
        out.append({"split": split, "beta": beta, "wells": int(local["well"].nunique()), "rmse": float(np.sqrt(np.mean(local["rmse"] ** 2))), "mean_well_rmse": float(local["rmse"].mean()), "far_rmse": float(np.sqrt(np.mean(local["far_rmse"] ** 2)))})
    return out

summary = pd.DataFrame(_summary(discovery, "discovery") + _summary(confirmation, "confirmation"))
summary.to_csv(OUT_DIR / "summary.csv", index=False)
base = confirmation[confirmation["beta"].eq("baseline")]["rmse"].to_numpy(float)
best_beta = discovery[discovery["beta"].isin(BETAS)].groupby("beta")["rmse"].mean().idxmin()
selected = confirmation[confirmation["beta"].eq(best_beta)]["rmse"].to_numpy(float)
verdict = {
    "experiment": "pf_seed_prefix_reweight_v153",
    "protocol": {"discovery_wells": len(discovery_wells), "confirmation_wells": len(confirmation_wells), "seeds": N_SEEDS, "particles": N_PARTICLES},
    "selected_beta": float(best_beta),
    "confirmation_gain": float(np.mean(base) - np.mean(selected)),
    "summary": summary.to_dict(orient="records"),
    "next": "PASS -> integrate fixed seed continuity weighting" if np.mean(base) - np.mean(selected) >= 0.05 else "FAIL -> prefix continuity cannot reweight PF seeds; close PF posterior branch",
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print(summary.to_string(index=False))
print(json.dumps(verdict, indent=2))
