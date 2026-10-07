"""Full-well, leakage-safe A/B test for the Gold candidate judge.

Run in the original 6.991 notebook after the cells defining the Gold helpers,
including ``_gold_poly_candidates``, ``_gold_pf_candidates`` and
``_gold_variant_grid``. This cell does not change or submit predictions.

The first two non-overlapping visible-prefix windows select a candidate. The
third window is never used during selection and is reserved for evaluation.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Frozen experiment configuration. Do not tune these values on this run.
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
OOF_DIR = Path("/root/autodl-tmp/oof_exports")
CACHE_VERSION = str(
    globals().get(
        "_GOLD_BLOCK_CACHE_VERSION_OVERRIDE",
        "v2_foldclean_s24_p350_b128",
    )
)
CACHE_DIR = OOF_DIR / f"block_paired_gold_forward_cache_{CACHE_VERSION}"

CUT_FRACS = (0.55, 0.70, 0.84)
CAL_SEEDS = 24
N_PARTICLES = 350
BLOCK_ROWS = 128
MIN_BLOCK_ROWS = 32
MIN_FINITE_FRAC = 0.95

# Same conservative quality floor as the confirmed Gold profile.
MIN_RMSE_GAIN = 1.00
MAX_CAND_RMSE = 9.00
MIN_CONSISTENCY = 0.67
MIN_POSITIVE_BLOCK_FRAC = 0.70

# None means all eligible train wells. Set to 5 only for a syntax/runtime smoke
# test, then restore None and clear no cache: completed wells are reusable.
MAX_WELLS = None

required_globals = [
    "_gold_poly_candidates",
    "_gold_pf_candidates",
    "_gold_variant_grid",
    "_gold_rmse",
    "FormationPlaneKNN",
    "DenseANCCImputer",
]
missing = [name for name in required_globals if name not in globals()]
if missing:
    raise RuntimeError(
        "Run the original Gold function-definition cell first; missing: "
        + repr(missing)
    )
if not (DATA_ROOT / "train").exists():
    raise RuntimeError(f"Missing train directory: {DATA_ROOT / 'train'}")

OOF_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# _gold_pf_candidates checks this global flag dynamically.
_GOLD_INCLUDE_PF = True
variants = _gold_variant_grid()

print(
    {
        "experiment": "block_paired_gold_forward_chaining",
        "cuts": CUT_FRACS,
        "cal_seeds": CAL_SEEDS,
        "particles": N_PARTICLES,
        "block_rows": BLOCK_ROWS,
        "variants": len(variants),
        "cache": str(CACHE_DIR),
        "max_wells": MAX_WELLS,
    }
)


def rmse_from_sse(sse, n):
    return float(np.sqrt(float(sse) / max(1, int(n))))


def _safe_surface_candidates(hw_masked, wid, fi, di):
    """Original Gold surface candidates with target-well exclusion.

    The train-only contact lookup is deliberately omitted because it reads the
    target well's TVT and would leak in local replay.
    """
    out = {}
    tvt = hw_masked["TVT_input"].to_numpy(dtype=float)
    z = hw_masked["Z"].to_numpy(dtype=float)
    xy = hw_masked[["X", "Y"]].to_numpy(dtype=float)
    known = np.isfinite(tvt) & np.isfinite(z) & np.isfinite(xy).all(axis=1)
    if int(known.sum()) < 30:
        return out

    formations = list(
        globals().get(
            "FORMATIONS", ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
        )
    )
    surface_names = []

    if fi is not None:
        try:
            form_all, _ = fi.impute(xy, self_wid=wid)
            form_all = np.asarray(form_all, dtype=float)
            for i, formation in enumerate(formations[: form_all.shape[1]]):
                surface = form_all[:, i]
                good = known & np.isfinite(surface)
                if int(good.sum()) < 30:
                    continue
                b_median = float(np.nanmedian(tvt[good] + z[good] - surface[good]))
                name = f"surface_{formation}_median"
                out[name] = (-z + surface + b_median).astype(float)
                surface_names.append(name)

                if callable(globals().get("seg_b_well")):
                    try:
                        b_full, _, _, b_late, b_wls = seg_b_well(
                            tvt[good].astype(np.float32),
                            z[good].astype(np.float32),
                            surface[good].astype(np.float32),
                        )
                        extra = {
                            f"surface_{formation}_full": b_full,
                            f"surface_{formation}_late": b_late,
                            f"surface_{formation}_wls": b_wls,
                        }
                        for extra_name, offset in extra.items():
                            out[extra_name] = (-z + surface + float(offset)).astype(float)
                            surface_names.append(extra_name)
                    except Exception:
                        pass
        except Exception as exc:
            print("safe FI skipped", wid, repr(exc))

    if di is not None:
        try:
            dense, _, _ = di.impute(xy, self_wid=wid)
            dense = np.asarray(dense, dtype=float)
            good = known & np.isfinite(dense)
            if int(good.sum()) >= 30:
                b_median = float(np.nanmedian(tvt[good] + z[good] - dense[good]))
                out["dense_ancc_median"] = (-z + dense + b_median).astype(float)
                surface_names.append("dense_ancc_median")

                if callable(globals().get("seg_b_well")):
                    try:
                        b_full, _, _, b_late, b_wls = seg_b_well(
                            tvt[good].astype(np.float32),
                            z[good].astype(np.float32),
                            dense[good].astype(np.float32),
                        )
                        extra = {
                            "dense_ancc_full": b_full,
                            "dense_ancc_late": b_late,
                            "dense_ancc_wls": b_wls,
                        }
                        for extra_name, offset in extra.items():
                            out[extra_name] = (-z + dense + float(offset)).astype(float)
                            surface_names.append(extra_name)
                    except Exception:
                        pass
        except Exception as exc:
            print("safe DI skipped", wid, repr(exc))

    names = [name for name in surface_names if name in out]
    if len(names) >= 2:
        errors = np.asarray(
            [_gold_rmse(out[name][known], tvt[known]) for name in names], dtype=float
        )
        finite = np.isfinite(errors)
        if int(finite.sum()) >= 2:
            names = [name for name, ok in zip(names, finite) if ok]
            errors = errors[finite]
            weights = 1.0 / np.maximum(errors, 0.25) ** 2
            weights /= weights.sum()
            matrix = np.vstack([out[name] for name in names])
            out["surface_weighted_prefix"] = (
                weights[:, None] * matrix
            ).sum(axis=0).astype(float)

    return out


def _safe_candidate_pool(wid, hw_masked, tw, fi, di):
    pool = {}
    pool.update(_gold_poly_candidates(hw_masked))
    pool.update(_safe_surface_candidates(hw_masked, wid, fi, di))
    pool.update(
        _gold_pf_candidates(
            hw_masked,
            tw,
            variants,
            n_seeds=CAL_SEEDS,
            n_particles=N_PARTICLES,
        )
    )
    extra_factory_with_pool = globals().get(
        "_GOLD_EXTRA_CANDIDATE_FACTORY_WITH_POOL"
    )
    extra_factory = globals().get("_GOLD_EXTRA_CANDIDATE_FACTORY")
    if callable(extra_factory_with_pool):
        extra = extra_factory_with_pool(wid, hw_masked, pool)
        if isinstance(extra, dict):
            pool.update(extra)
    elif callable(extra_factory):
        extra = extra_factory(wid, hw_masked)
        if isinstance(extra, dict):
            pool.update(extra)
    clean = {}
    for name, prediction in pool.items():
        values = np.asarray(prediction, dtype=float)
        if (
            len(values) == len(hw_masked)
            and np.isfinite(values).sum() >= max(20, len(hw_masked) // 20)
        ):
            clean[name] = values
    return clean


def _default_pf_name(hw):
    try:
        return "pf|" + selector_well_code(hw)[1]
    except Exception:
        try:
            return "pf|" + SELECTOR_GLOBAL_VARIANT
        except Exception:
            return None


def _block_effects(truth, default, candidate):
    truth = np.asarray(truth, dtype=float)
    default = np.asarray(default, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    finite = np.isfinite(truth) & np.isfinite(default) & np.isfinite(candidate)
    if int(finite.sum()) < max(30, int(MIN_FINITE_FRAC * len(truth))):
        return None

    # Keep chronological order and retain partial final blocks when informative.
    index = np.flatnonzero(finite)
    effects = []
    for start in range(0, len(index), BLOCK_ROWS):
        block_index = index[start : start + BLOCK_ROWS]
        if len(block_index) < MIN_BLOCK_ROWS:
            continue
        base_loss = (truth[block_index] - default[block_index]) ** 2
        candidate_loss = (truth[block_index] - candidate[block_index]) ** 2
        effects.append(float(np.mean(base_loss - candidate_loss)))
    if not effects:
        return None

    base_error = truth[finite] - default[finite]
    candidate_error = truth[finite] - candidate[finite]
    return {
        "n": int(finite.sum()),
        "default_sse": float(np.sum(base_error**2)),
        "candidate_sse": float(np.sum(candidate_error**2)),
        "default_rmse": float(np.sqrt(np.mean(base_error**2))),
        "candidate_rmse": float(np.sqrt(np.mean(candidate_error**2))),
        "effects": effects,
        "effect_mean": float(np.mean(effects)),
        "effect_median": float(np.median(effects)),
        "positive_block_frac": float(np.mean(np.asarray(effects) > 0)),
    }


def _cut_positions(known_prefix):
    cut_positions = []
    for fraction in CUT_FRACS:
        position = int(round(len(known_prefix) * fraction))
        position = max(50, min(position, len(known_prefix) - 35))
        cut_positions.append(position)
    if not (cut_positions[0] < cut_positions[1] < cut_positions[2]):
        return None
    return cut_positions


def _build_well_metrics(wid, fold, fi, di):
    cache_path = CACHE_DIR / f"{wid}.pkl"
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if cached.get("cache_version") == CACHE_VERSION:
            return cached

    started = time.time()
    try:
        hw_path = DATA_ROOT / "train" / f"{wid}__horizontal_well.csv"
        tw_path = DATA_ROOT / "train" / f"{wid}__typewell.csv"
        hw_source = pd.read_csv(hw_path)
        tw_source = pd.read_csv(tw_path)

        # Enforce the hidden-test schema. Reference-well FI/DI objects may retain
        # train surfaces, but the target well passed into candidate code cannot.
        hw_columns = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
        tw_columns = ["TVT", "GR"]
        missing_hw = [c for c in hw_columns if c not in hw_source.columns]
        missing_tw = [c for c in tw_columns if c not in tw_source.columns]
        if missing_hw or missing_tw:
            raise RuntimeError(
                f"schema mismatch: missing_hw={missing_hw} missing_tw={missing_tw}"
            )
        hw = hw_source[hw_columns].copy()
        tw = tw_source[tw_columns].copy()
        visible_tvt = hw["TVT_input"].to_numpy(dtype=float)
        is_known = np.isfinite(visible_tvt)
        is_hidden = ~is_known
        if not bool(is_hidden.any()):
            raise RuntimeError("no hidden suffix")
        first_hidden = int(np.flatnonzero(is_hidden)[0])
        known_prefix = np.flatnonzero(is_known & (np.arange(len(hw)) < first_hidden))
        if len(known_prefix) < 240:
            raise RuntimeError(f"short visible prefix: {len(known_prefix)}")

        positions = _cut_positions(known_prefix)
        if positions is None:
            raise RuntimeError("invalid forward windows")
        boundaries = positions + [len(known_prefix)]
        default_name = _default_pf_name(hw)
        if default_name is None:
            raise RuntimeError("could not determine default PF name")

        window_metrics = []
        candidate_sets = []
        for window_index, cut_position in enumerate(positions):
            next_position = boundaries[window_index + 1]
            cutoff_idx = int(known_prefix[cut_position - 1])
            score_idx = known_prefix[cut_position:next_position]
            if len(score_idx) < 30:
                raise RuntimeError(
                    f"window {window_index + 1} has only {len(score_idx)} rows"
                )

            hw_masked = hw.copy(deep=True)
            hw_masked.loc[hw_masked.index > cutoff_idx, "TVT_input"] = np.nan
            pool = _safe_candidate_pool(wid, hw_masked, tw, fi, di)
            if default_name not in pool:
                raise RuntimeError(
                    f"default candidate missing in window {window_index + 1}: {default_name}"
                )

            truth = visible_tvt[score_idx]
            default_prediction = pool[default_name][score_idx]
            metrics = {}
            for candidate_name, prediction in pool.items():
                result = _block_effects(
                    truth,
                    default_prediction,
                    np.asarray(prediction, dtype=float)[score_idx],
                )
                if result is not None:
                    metrics[candidate_name] = result
            if default_name not in metrics:
                raise RuntimeError(
                    f"default metrics missing in window {window_index + 1}"
                )
            window_metrics.append(metrics)
            candidate_sets.append(set(metrics))

        common_candidates = sorted(set.intersection(*candidate_sets))
        if default_name not in common_candidates:
            raise RuntimeError("default candidate not common across windows")
        if len(common_candidates) < 2:
            raise RuntimeError("fewer than two common candidates")

        compact = {}
        for candidate_name in common_candidates:
            compact[candidate_name] = [
                window_metrics[i][candidate_name] for i in range(3)
            ]

        record = {
            "cache_version": CACHE_VERSION,
            "ok": True,
            "well": wid,
            "fold": int(fold),
            "known_prefix": int(len(known_prefix)),
            "default_name": default_name,
            "candidate_count": int(len(common_candidates)),
            "metrics": compact,
            "elapsed_sec": float(time.time() - started),
        }
    except Exception as exc:
        record = {
            "cache_version": CACHE_VERSION,
            "ok": False,
            "well": wid,
            "reason": repr(exc),
            "elapsed_sec": float(time.time() - started),
        }

    pd.to_pickle(record, cache_path)
    return record


def _aggregate_rmse(metrics, windows=(0, 1)):
    values = np.asarray([metrics[i]["candidate_rmse"] for i in windows], dtype=float)
    return float(np.median(values) + 0.10 * np.std(values))


def _choose_original(record):
    metrics = record["metrics"]
    default_name = record["default_name"]
    aggregate = {
        name: _aggregate_rmse(candidate_metrics)
        for name, candidate_metrics in metrics.items()
    }
    ordered = sorted((score, name) for name, score in aggregate.items())
    best_score, best_name = ordered[0]
    default_score = aggregate[default_name]

    # Reproduce the original Gold consistency definition on W1/W2.
    consistent = 0
    for window_index in (0, 1):
        local_best = min(
            candidate_metrics[window_index]["candidate_rmse"]
            for candidate_metrics in metrics.values()
        )
        default_rmse = metrics[default_name][window_index]["candidate_rmse"]
        consistent += int(local_best <= default_rmse - 0.25)
    consistency = consistent / 2.0
    gain = default_score - best_score
    selected = bool(
        best_name != default_name
        and best_score <= MAX_CAND_RMSE
        and gain >= MIN_RMSE_GAIN
        and consistency >= MIN_CONSISTENCY
    )
    return {
        "candidate": best_name if selected else default_name,
        "selected": selected,
        "train_score": best_score,
        "train_gain": gain,
        "consistency": consistency,
    }


def _choose_block_paired(record):
    metrics = record["metrics"]
    default_name = record["default_name"]
    default_score = _aggregate_rmse(metrics[default_name])
    eligible = []

    for name, candidate_metrics in metrics.items():
        if name == default_name:
            continue
        candidate_score = _aggregate_rmse(candidate_metrics)
        rmse_gain = default_score - candidate_score
        window_effects = [candidate_metrics[i]["effect_mean"] for i in (0, 1)]
        blocks = np.asarray(
            candidate_metrics[0]["effects"] + candidate_metrics[1]["effects"],
            dtype=float,
        )
        if len(blocks) == 0:
            continue
        positive_block_frac = float(np.mean(blocks > 0))
        robust_effect = float(np.median(blocks))
        if (
            candidate_score <= MAX_CAND_RMSE
            and rmse_gain >= MIN_RMSE_GAIN
            and min(window_effects) > 0
            and positive_block_frac >= MIN_POSITIVE_BLOCK_FRAC
            and robust_effect > 0
        ):
            eligible.append(
                (
                    robust_effect,
                    rmse_gain,
                    name,
                    candidate_score,
                    positive_block_frac,
                )
            )

    if not eligible:
        return {
            "candidate": default_name,
            "selected": False,
            "train_score": default_score,
            "train_gain": 0.0,
            "robust_effect": 0.0,
            "positive_block_frac": 0.0,
        }

    robust_effect, rmse_gain, name, score, positive_block_frac = max(eligible)
    return {
        "candidate": name,
        "selected": True,
        "train_score": score,
        "train_gain": rmse_gain,
        "robust_effect": robust_effect,
        "positive_block_frac": positive_block_frac,
    }


def _evaluate_choice(record, choice):
    default_name = record["default_name"]
    metrics = record["metrics"]
    default_w3 = metrics[default_name][2]
    chosen_w3 = metrics[choice["candidate"]][2]
    default_rmse = default_w3["candidate_rmse"]
    chosen_rmse = chosen_w3["candidate_rmse"]
    return {
        "base_sse": float(default_w3["candidate_sse"]),
        "pred_sse": float(chosen_w3["candidate_sse"]),
        "rows": int(default_w3["n"]),
        "base_rmse_well": default_rmse,
        "pred_rmse_well": chosen_rmse,
        "gain_well": default_rmse - chosen_rmse,
    }


# ---------------------------------------------------------------------------
# Build target-free folds before feature construction. Each validation well's
# FI/DI reference database excludes its entire fold, not only that one well.
# ---------------------------------------------------------------------------
well_paths = sorted((DATA_ROOT / "train").glob("*__horizontal_well.csv"))
meta_rows = []
for path in tqdm(well_paths, desc="scan Gold metadata"):
    wid = path.name.replace("__horizontal_well.csv", "")
    visible = pd.read_csv(path, usecols=["TVT_input"])["TVT_input"].to_numpy(float)
    finite = np.isfinite(visible)
    hidden = ~finite
    if not hidden.any():
        continue
    first_hidden = int(np.flatnonzero(hidden)[0])
    known_prefix = int(np.sum(finite & (np.arange(len(visible)) < first_hidden)))
    if known_prefix >= 240:
        meta_rows.append({"well": wid, "known_prefix": known_prefix})

meta = pd.DataFrame(meta_rows).sort_values("well").reset_index(drop=True)
meta["stratum"] = pd.qcut(
    meta["known_prefix"].rank(method="first"),
    q=min(10, len(meta)),
    labels=False,
).astype(int)
meta["fold"] = -1
splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260711)
for fold, (_, valid_index) in enumerate(splitter.split(meta, meta["stratum"]), 1):
    meta.loc[valid_index, "fold"] = fold

all_reference_wells = meta["well"].astype(str).tolist()
fold_imputers = {}
for fold in range(1, 6):
    reference_wells = meta.loc[meta["fold"] != fold, "well"].astype(str).tolist()
    print(
        f"building fold-clean FI/DI fold={fold} references={len(reference_wells)}",
        flush=True,
    )
    fold_imputers[fold] = (
        FormationPlaneKNN(reference_wells, DATA_ROOT / "train"),
        DenseANCCImputer(reference_wells, DATA_ROOT / "train"),
    )

tasks = list(meta[["well", "fold"]].itertuples(index=False, name=None))
if MAX_WELLS is not None:
    tasks = tasks[: int(MAX_WELLS)]

results = []
started = time.time()
for index, (wid, fold) in enumerate(tqdm(tasks, desc="Gold forward wells"), 1):
    fi, di = fold_imputers[int(fold)]
    result = _build_well_metrics(wid, int(fold), fi, di)
    results.append(result)
    if index % 25 == 0:
        ok_count = sum(bool(item.get("ok")) for item in results)
        print(
            f"{index}/{len(tasks)} ok={ok_count} "
            f"elapsed={(time.time() - started) / 60:.1f} min",
            flush=True,
        )

successes = [result for result in results if result.get("ok")]
failures = pd.DataFrame(
    [
        {
            "well": result.get("well"),
            "reason": result.get("reason"),
            "elapsed_sec": result.get("elapsed_sec"),
        }
        for result in results
        if not result.get("ok")
    ]
)
if len(successes) < 0.85 * len(results):
    display(failures.head(50))
    raise RuntimeError(f"Too many failures: {len(failures)} / {len(results)}")


# ---------------------------------------------------------------------------
# Keep only successful wells while preserving the precomputed folds.
# ---------------------------------------------------------------------------
success_meta = pd.DataFrame(
    [
        {
            "well": result["well"],
            "fold": int(result["fold"]),
            "known_prefix": result["known_prefix"],
            "candidate_count": result["candidate_count"],
            "elapsed_sec": result["elapsed_sec"],
        }
        for result in successes
    ]
).sort_values("well").reset_index(drop=True)
fold_map = dict(zip(success_meta["well"], success_meta["fold"]))
result_map = {result["well"]: result for result in successes}


# ---------------------------------------------------------------------------
# Select on W1/W2 and evaluate once on untouched W3.
# ---------------------------------------------------------------------------
detail_records = []
for wid in success_meta["well"]:
    record = result_map[wid]
    original = _choose_original(record)
    paired = _choose_block_paired(record)
    original_eval = _evaluate_choice(record, original)
    paired_eval = _evaluate_choice(record, paired)
    detail_records.append(
        {
            "well": wid,
            "fold": int(fold_map[wid]),
            "known_prefix": record["known_prefix"],
            "candidate_count": record["candidate_count"],
            "default_name": record["default_name"],
            "original_candidate": original["candidate"],
            "original_selected": original["selected"],
            "original_train_gain": original["train_gain"],
            "original_consistency": original["consistency"],
            "paired_candidate": paired["candidate"],
            "paired_selected": paired["selected"],
            "paired_train_gain": paired["train_gain"],
            "paired_robust_effect": paired["robust_effect"],
            "paired_positive_block_frac": paired["positive_block_frac"],
            "base_sse": original_eval["base_sse"],
            "rows": original_eval["rows"],
            "base_rmse_well": original_eval["base_rmse_well"],
            "original_sse": original_eval["pred_sse"],
            "original_rmse_well": original_eval["pred_rmse_well"],
            "original_gain_well": original_eval["gain_well"],
            "paired_sse": paired_eval["pred_sse"],
            "paired_rmse_well": paired_eval["pred_rmse_well"],
            "paired_gain_well": paired_eval["gain_well"],
        }
    )

detail = pd.DataFrame(detail_records)


def _method_summary(method):
    base_sse = float(detail["base_sse"].sum())
    pred_sse = float(detail[f"{method}_sse"].sum())
    n = int(detail["rows"].sum())
    selected = detail[f"{method}_selected"].to_numpy(bool)
    base_score = rmse_from_sse(base_sse, n)
    score = rmse_from_sse(pred_sse, n)
    selected_positive = (
        float((detail.loc[selected, f"{method}_gain_well"] > 0).mean())
        if selected.any()
        else 0.0
    )
    return {
        "method": method,
        "rows": n,
        "base_rmse": base_score,
        "rmse": score,
        "gain": base_score - score,
        "selected_wells": int(selected.sum()),
        "selected_frac": float(selected.mean()),
        "positive_selected_frac": selected_positive,
        "mean_selected_well_gain": (
            float(detail.loc[selected, f"{method}_gain_well"].mean())
            if selected.any()
            else 0.0
        ),
    }


summary = pd.DataFrame(
    [_method_summary("original"), _method_summary("paired")]
)
fold_records = []
for fold, group in detail.groupby("fold", sort=True):
    n = int(group["rows"].sum())
    base_score = rmse_from_sse(group["base_sse"].sum(), n)
    original_score = rmse_from_sse(group["original_sse"].sum(), n)
    paired_score = rmse_from_sse(group["paired_sse"].sum(), n)
    fold_records.append(
        {
            "fold": int(fold),
            "rows": n,
            "base_rmse": base_score,
            "original_rmse": original_score,
            "original_gain": base_score - original_score,
            "paired_rmse": paired_score,
            "paired_gain": base_score - paired_score,
            "paired_gain_vs_original": original_score - paired_score,
            "original_selected_wells": int(group["original_selected"].sum()),
            "paired_selected_wells": int(group["paired_selected"].sum()),
        }
    )
folds = pd.DataFrame(fold_records)

paired_row = summary.loc[summary["method"] == "paired"].iloc[0]
original_row = summary.loc[summary["method"] == "original"].iloc[0]
strict_gate = bool(
    paired_row["gain"] > 0
    and (folds["paired_gain"] > 0).all()
    and folds["paired_gain"].min() >= 0
    and paired_row["rmse"] < original_row["rmse"]
)

print("\nFORWARD W3 SUMMARY")
display(summary)
print("\nFORWARD W3 FOLDS")
display(folds)
print("\nSELECTION OVERLAP")
display(
    pd.crosstab(
        detail["original_selected"],
        detail["paired_selected"],
        rownames=["original_selected"],
        colnames=["paired_selected"],
    )
)
print("\nTOP PAIRED CANDIDATES")
display(
    detail.loc[detail["paired_selected"], "paired_candidate"]
    .value_counts()
    .head(30)
    .rename_axis("candidate")
    .reset_index(name="wells")
)
print("STRICT GATE PASS:", strict_gate)

prefix = OOF_DIR / "block_paired_gold_forward_chaining"
detail.to_csv(str(prefix) + "_bywell.csv", index=False)
summary.to_csv(str(prefix) + "_summary.csv", index=False)
folds.to_csv(str(prefix) + "_folds.csv", index=False)
success_meta.to_csv(str(prefix) + "_meta.csv", index=False)
failures.to_csv(str(prefix) + "_failures.csv", index=False)
print("saved prefix:", prefix)
