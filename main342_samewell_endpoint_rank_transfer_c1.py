"""M342-C1: same-well pseudo-cut endpoint rank-transfer gateway.

Earlier pseudo cuts are scored only on rows before the official prediction
start. Those rows are visible for the real target well, so selecting an
existing PF/beam endpoint from them is legal at inference. Fold 4/5 truth is
used only to establish a population fixed-candidate control. Fold 3 selects
top-k and a conservative blend dose; Fold 2 is opened only after the Fold 3
mechanism gate passes. Fold 1 is never loaded.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
CACHE_DIR = OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1"
FEATURE_DIR = CACHE_DIR / "features"
DONOR_FEATURE_DIR = (
    OOF_ROOT / "multicut_fleongg_k5_full_rebuild_v2_ps35_50_65_75" / "features"
)
V168_PATH = OOF_ROOT / "historical_analog_metadata_fullrow_v168" / "fullrow_oof.parquet"
OUT_DIR = OOF_ROOT / "main342_samewell_endpoint_rank_transfer_c1"

TRAIN_FOLDS = (4, 5)
SELECTION_FOLD = 3
CONFIRMATION_FOLD = 2
PSEUDO_TAGS = ("pseudo1", "pseudo2")
MIN_REPLAY_ROWS = 16
DOSES = np.round(np.arange(0.0, 0.2001, 0.025), 3)

ABSOLUTE_CANDIDATES = (
    "pf_ancc", "pf_z", "likpf_scale_3", "likpf_scale_5",
    "likpf_scale_8", "likpf_scale_12", "likpf_mean",
)
DELTA_CANDIDATES = (
    "beam_cons_d", "beam_loose_d", "beam_vcons_d", "beam_sm5_d",
    "beam_vloose_d", "beam_mid_d", "beam_stiff_d", "beam_mean_d",
    "beam_med_d", "sc8_d", "sc15_d", "sc25_d", "sc_cons_d",
    "sc_ens_d", "hyb_d", "slp_b_d_all", "slp_b_d_50",
)


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _paths(well, tag, full=False):
    if full:
        path = FEATURE_DIR / f"{well}__official_full.pkl"
        donor = DONOR_FEATURE_DIR / f"{well}__official_full.pkl"
    else:
        path = FEATURE_DIR / f"{well}__{tag}__sample.pkl"
        donor = DONOR_FEATURE_DIR / f"{well}__official__sample.pkl" if tag == "official" else None
    if path.is_file():
        return path
    if donor is not None and donor.is_file():
        return donor
    return path


def _finite(values, fallback):
    output = np.asarray(values, np.float64).copy()
    fallback = np.asarray(fallback, np.float64)
    output[~np.isfinite(output)] = fallback[~np.isfinite(output)]
    return output


def _curves(frame, menu):
    last = frame["last_known_tvt"].to_numpy(np.float64)
    last = np.where(np.isfinite(last), last, np.nanmedian(last))
    curves = {}
    for name in menu:
        if name in ABSOLUTE_CANDIDATES:
            curves[name] = _finite(frame[name].to_numpy(np.float64), last)
        else:
            delta = _finite(frame[name].to_numpy(np.float64), np.zeros(len(frame)))
            curves[name] = last + delta
    return curves


def _truth(frame):
    return (
        frame["last_known_tvt"].to_numpy(np.float64)
        + frame["target"].to_numpy(np.float64)
    )


def _rmse(truth, prediction):
    return float(np.sqrt(np.mean(np.square(np.asarray(prediction) - np.asarray(truth)))))


def _metrics(frame, prediction, endpoint):
    truth = frame["tvt_true"].to_numpy(np.float64)
    prediction = np.asarray(prediction, np.float64)
    error = prediction - truth
    far = frame["md_since"].to_numpy(np.float64) > 2000.0
    local = pd.DataFrame({"well": frame["well"].astype(str), "e2": error * error})
    well_rmse = np.sqrt(local.groupby("well", sort=False)["e2"].mean())
    return {
        "endpoint": endpoint,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "exact_sse": float(np.dot(error, error)),
        "rmse": _rmse(truth, prediction),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(well_rmse.median()),
        "p95_well_rmse": float(well_rmse.quantile(0.95)),
        "far_gt2000_rmse": _rmse(truth[far], prediction[far]) if far.any() else np.nan,
    }


def _discover_menu(probe_wells):
    available = set(ABSOLUTE_CANDIDATES + DELTA_CANDIDATES)
    for well in probe_wells:
        for tag in ("official",) + PSEUDO_TAGS:
            path = _paths(well, tag)
            if not path.is_file():
                continue
            columns = set(pd.read_pickle(path).columns)
            available &= columns
    menu = [name for name in ABSOLUTE_CANDIDATES + DELTA_CANDIDATES if name in available]
    if len(menu) < 8 or "pf_ancc" not in menu or "likpf_scale_5" not in menu:
        raise RuntimeError(f"Candidate menu is too small: {menu}")
    return menu


def _score_frame(frame, menu, mask=None):
    if mask is None:
        mask = np.ones(len(frame), dtype=bool)
    mask = np.asarray(mask, bool)
    if mask.sum() < MIN_REPLAY_ROWS:
        return None
    truth = _truth(frame)[mask]
    curves = _curves(frame, menu)
    return {
        name: {
            "sse": float(np.sum(np.square(curve[mask] - truth))),
            "rows": int(mask.sum()),
        }
        for name, curve in curves.items()
    }


def _collect_rank_transfer(wells, fold_by_well, official_cut, menu):
    details = []
    failures = []
    for number, well in enumerate(wells, 1):
        accumulator = {name: [0.0, 0] for name in menu}
        usable_tags = 0
        try:
            for tag in PSEUDO_TAGS:
                path = _paths(well, tag)
                if not path.is_file():
                    continue
                frame = pd.read_pickle(path).reset_index(drop=True)
                row_column = "_row_idx" if "_row_idx" in frame else None
                if row_column is None:
                    continue
                visible_replay = frame[row_column].to_numpy(np.int64) < int(official_cut[well])
                scores = _score_frame(frame, menu, visible_replay)
                if scores is None:
                    continue
                usable_tags += 1
                for name in menu:
                    accumulator[name][0] += scores[name]["sse"]
                    accumulator[name][1] += scores[name]["rows"]

            official_path = _paths(well, "official")
            if not official_path.is_file():
                raise FileNotFoundError(official_path)
            official_sample = pd.read_pickle(official_path).reset_index(drop=True)
            official_scores = _score_frame(official_sample, menu)
            if official_scores is None:
                raise RuntimeError("official sample has too few rows")

            pseudo_rmse = {
                name: (
                    np.sqrt(values[0] / values[1]) if values[1] >= MIN_REPLAY_ROWS else np.nan
                )
                for name, values in accumulator.items()
            }
            ranked = [
                name for name in sorted(menu, key=lambda candidate: (pseudo_rmse[candidate], candidate))
                if np.isfinite(pseudo_rmse[name])
            ]
            official_rmse = {
                name: np.sqrt(official_scores[name]["sse"] / official_scores[name]["rows"])
                for name in menu
            }
            details.append({
                "well": well,
                "fold": int(fold_by_well[well]),
                "usable_tags": usable_tags,
                "replay_rows": int(max(values[1] for values in accumulator.values())),
                "pseudo_top1": ranked[0] if ranked else None,
                "pseudo_top2": "|".join(ranked[:2]) if len(ranked) >= 2 else None,
                "official_oracle": min(menu, key=lambda name: official_rmse[name]),
                **{f"pseudo_rmse__{name}": pseudo_rmse[name] for name in menu},
                **{f"official_rmse__{name}": official_rmse[name] for name in menu},
                **{f"official_sse__{name}": official_scores[name]["sse"] for name in menu},
                **{f"official_rows__{name}": official_scores[name]["rows"] for name in menu},
            })
        except Exception as exc:
            failures.append({"well": well, "reason": repr(exc)})
        if number % 100 == 0 or number == len(wells):
            print(f"M342-C1 rank transfer {number}/{len(wells)} failures={len(failures)}", flush=True)
    return pd.DataFrame(details), failures


def _fixed_candidate(detail, menu):
    train = detail[detail["fold"].isin(TRAIN_FOLDS)]
    scores = []
    for name in menu:
        sse = train[f"official_sse__{name}"].sum()
        rows = train[f"official_rows__{name}"].sum()
        scores.append({"candidate": name, "rmse": np.sqrt(sse / rows)})
    return pd.DataFrame(scores).sort_values(["rmse", "candidate"]).reset_index(drop=True)


def _build_fold(fold, detail, base, menu, fixed_name, shuffled=False):
    local_detail = detail[detail["fold"].eq(fold)].copy().sort_values("well")
    active = local_detail["pseudo_top1"].notna() & local_detail["pseudo_top2"].notna()
    top1 = local_detail["pseudo_top1"].copy()
    top2 = local_detail["pseudo_top2"].copy()
    if shuffled and active.sum() > 1:
        active_index = local_detail.index[active]
        top1.loc[active_index] = np.roll(top1.loc[active_index].to_numpy(copy=True), 1)
        top2.loc[active_index] = np.roll(top2.loc[active_index].to_numpy(copy=True), 1)
    selections = {
        str(row.well): {
            "top1": top1.loc[index] if pd.notna(top1.loc[index]) else fixed_name,
            "top2": top2.loc[index].split("|") if pd.notna(top2.loc[index]) else [fixed_name],
        }
        for index, row in local_detail.iterrows()
    }
    rows = []
    for number, well in enumerate(sorted(selections), 1):
        path = _paths(well, "official", full=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_pickle(path).reset_index(drop=True)
        frame["id"] = frame["id"].astype(str)
        aligned = frame.merge(
            base.loc[base["well"].eq(well), ["id", "well", "fold", "tvt_true", "md_since", "base_v168"]],
            on="id", how="inner", validate="one_to_one", suffixes=("_feature", ""), sort=False,
        )
        curves = _curves(aligned, menu)
        truth_gap = np.max(np.abs(_truth(aligned) - aligned["tvt_true"].to_numpy(np.float64)))
        if not np.isfinite(truth_gap) or truth_gap > 0.01:
            raise RuntimeError(f"{well}: truth gap {truth_gap}")
        selected = selections[well]
        names2 = [name for name in selected["top2"] if name in curves]
        rows.append(pd.DataFrame({
            "id": aligned["id"].astype(str),
            "well": well,
            "fold": fold,
            "tvt_true": aligned["tvt_true"].to_numpy(np.float64),
            "md_since": aligned["md_since"].to_numpy(np.float64),
            "base_v168": aligned["base_v168"].to_numpy(np.float64),
            "fixed": curves[fixed_name],
            "pseudo_top1": curves[selected["top1"]],
            "pseudo_top2": np.mean(np.column_stack([curves[name] for name in names2]), axis=1),
            "oracle": curves[min(menu, key=lambda name: _rmse(aligned["tvt_true"], curves[name]))],
        }))
        if number % 50 == 0 or number == len(selections):
            print(f"M342-C1 build fold{fold} {number}/{len(selections)} shuffled={shuffled}", flush=True)
    return pd.concat(rows, ignore_index=True)


def _scan(frame, endpoint_names):
    base = frame["base_v168"].to_numpy(np.float64)
    base_rmse = _rmse(frame["tvt_true"], base)
    rows = []
    for endpoint in endpoint_names:
        values = frame[endpoint].to_numpy(np.float64)
        for dose in DOSES:
            prediction = base + float(dose) * (values - base)
            metric = _metrics(frame, prediction, endpoint)
            rows.append({
                "endpoint": endpoint,
                "dose": float(dose),
                "rmse": metric["rmse"],
                "far_gt2000_rmse": metric["far_gt2000_rmse"],
                "gain": base_rmse - metric["rmse"],
            })
    return pd.DataFrame(rows)


def _best(scan, endpoint):
    return scan[scan["endpoint"].eq(endpoint)].sort_values(["rmse", "dose"]).iloc[0]


started = time.time()
for required in (CACHE_DIR / "cut_manifest.csv", FEATURE_DIR, V168_PATH):
    if not required.exists():
        raise FileNotFoundError(required)
OUT_DIR.mkdir(parents=True, exist_ok=True)

base = pd.read_parquet(V168_PATH)
base = base[["id", "well", "fold", "tvt_true", "md_since", "metadata_analog_w0p05"]].copy()
base = base.rename(columns={"metadata_analog_w0p05": "base_v168"})
base["id"] = base["id"].astype(str)
base["well"] = base["well"].astype(str)
if base["well"].nunique() != 773 or base["id"].duplicated().any():
    raise RuntimeError("V168 is not the complete unique 773-well OOF")
fold_by_well = base.groupby("well")["fold"].first().astype(int).to_dict()

manifest = pd.read_csv(CACHE_DIR / "cut_manifest.csv")
manifest["well"] = manifest["well"].astype(str)
manifest["tag"] = manifest["tag"].astype(str)
official_manifest = manifest[manifest["tag"].eq("official")]
official_cut = dict(zip(official_manifest["well"], official_manifest["cut_position"].astype(int)))
complete_wells = sorted(set(base["well"]) & set(official_cut))
complete_wells = [
    well for well in complete_wells
    if all(_paths(well, tag).is_file() for tag in ("official",) + PSEUDO_TAGS)
    and _paths(well, "official", full=True).is_file()
]
coverage = pd.Series({well: fold_by_well[well] for well in complete_wells}).value_counts().sort_index().to_dict()
if sum(coverage.get(fold, 0) for fold in TRAIN_FOLDS) < 295 or coverage.get(3, 0) < 145 or coverage.get(2, 0) < 145:
    raise RuntimeError(f"Insufficient C1 coverage: {coverage}")

discovery_wells = [
    well for well in complete_wells
    if fold_by_well[well] in TRAIN_FOLDS + (SELECTION_FOLD,)
]
menu = _discover_menu(discovery_wells[:20])
print("M342-C1 CONFIG", {"coverage": coverage, "menu": menu, "fold1_touched": False}, flush=True)
detail, failures = _collect_rank_transfer(discovery_wells, fold_by_well, official_cut, menu)
pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)
fixed_scan = _fixed_candidate(detail, menu)
fixed_name = str(fixed_scan.iloc[0]["candidate"])
active_detail = detail[detail["pseudo_top1"].notna()].copy()
top1_hit_rate = float(
    np.mean(active_detail["pseudo_top1"].eq(active_detail["official_oracle"]))
) if len(active_detail) else np.nan
rank_correlations = []
for _, row in active_detail.iterrows():
    pseudo_values = np.asarray(
        [row[f"pseudo_rmse__{name}"] for name in menu], np.float64
    )
    official_values = np.asarray(
        [row[f"official_rmse__{name}"] for name in menu], np.float64
    )
    finite = np.isfinite(pseudo_values) & np.isfinite(official_values)
    if finite.sum() >= 5:
        pseudo_rank = pd.Series(pseudo_values[finite]).rank().to_numpy(np.float64)
        official_rank = pd.Series(official_values[finite]).rank().to_numpy(np.float64)
        rank_correlations.append(float(np.corrcoef(pseudo_rank, official_rank)[0, 1]))
median_rank_correlation = float(np.nanmedian(rank_correlations)) if rank_correlations else np.nan
detail.to_parquet(OUT_DIR / "rank_transfer_detail.parquet", index=False, compression="zstd")
fixed_scan.to_csv(OUT_DIR / "fixed_candidate_scan.csv", index=False)

selection = _build_fold(SELECTION_FOLD, detail, base, menu, fixed_name, shuffled=False)
selection_shuffle = _build_fold(SELECTION_FOLD, detail, base, menu, fixed_name, shuffled=True)
selection_scan = _scan(selection, ("fixed", "pseudo_top1", "pseudo_top2"))
best1 = _best(selection_scan, "pseudo_top1")
best2 = _best(selection_scan, "pseudo_top2")
selected_rule = "pseudo_top1" if best1.rmse <= best2.rmse else "pseudo_top2"
selected = best1 if selected_rule == "pseudo_top1" else best2
fixed_best = _best(selection_scan, "fixed")
selected_dose = float(selected.dose)
base_values = selection["base_v168"].to_numpy(np.float64)
selected_prediction = base_values + selected_dose * (selection[selected_rule].to_numpy(np.float64) - base_values)
shuffle_prediction = base_values + selected_dose * (selection_shuffle[selected_rule].to_numpy(np.float64) - base_values)
selection_metrics = {
    "base": _metrics(selection, base_values, "base"),
    "fixed_endpoint": _metrics(selection, selection["fixed"], "fixed_endpoint"),
    "selected_endpoint": _metrics(selection, selection[selected_rule], selected_rule + "_endpoint"),
    "menu_oracle": _metrics(selection, selection["oracle"], "menu_oracle"),
    "selected": _metrics(selection, selected_prediction, selected_rule),
    "shuffle": _metrics(selection, shuffle_prediction, "samewell_selection_shuffle"),
}
selection_gain = selection_metrics["base"]["rmse"] - selection_metrics["selected"]["rmse"]
selection_increment = float(fixed_best.rmse - selected.rmse)
selection_shuffle_loss = selection_metrics["shuffle"]["rmse"] - selection_metrics["selected"]["rmse"]
active_fraction = float(detail.loc[detail["fold"].eq(3), "pseudo_top1"].notna().mean())
mechanism_pass = bool(
    selected_dose > 0
    and selection_gain >= 0.03
    and selection_increment >= 0.015
    and selection_shuffle_loss >= 0.03
    and active_fraction >= 0.60
)

print("\nM342-C1 FIXED CANDIDATES")
print(fixed_scan.to_string(index=False))
print("\nM342-C1 SELECTION SCAN")
print(selection_scan.to_string(index=False))
print("M342-C1 SELECTION GATE", {
    "rule": selected_rule, "dose": selected_dose, "gain": selection_gain,
    "increment_vs_fixed": selection_increment, "shuffle_loss": selection_shuffle_loss,
    "active_fraction": active_fraction, "pass": mechanism_pass,
}, flush=True)

confirmation_metrics = None
confirmation_checks = None
if mechanism_pass:
    confirmation_wells = [
        well for well in complete_wells if fold_by_well[well] == CONFIRMATION_FOLD
    ]
    confirmation_detail, confirmation_failures = _collect_rank_transfer(
        confirmation_wells, fold_by_well, official_cut, menu
    )
    detail = pd.concat([detail, confirmation_detail], ignore_index=True)
    failures.extend(confirmation_failures)
    detail.to_parquet(
        OUT_DIR / "rank_transfer_detail.parquet", index=False, compression="zstd"
    )
    pd.DataFrame(failures).to_csv(OUT_DIR / "failures.csv", index=False)
    confirmation = _build_fold(CONFIRMATION_FOLD, detail, base, menu, fixed_name, shuffled=False)
    confirmation_shuffle = _build_fold(CONFIRMATION_FOLD, detail, base, menu, fixed_name, shuffled=True)
    con_base = confirmation["base_v168"].to_numpy(np.float64)
    con_prediction = con_base + selected_dose * (confirmation[selected_rule].to_numpy(np.float64) - con_base)
    con_shuffle = con_base + selected_dose * (confirmation_shuffle[selected_rule].to_numpy(np.float64) - con_base)
    confirmation_metrics = {
        "base": _metrics(confirmation, con_base, "base"),
        "selected": _metrics(confirmation, con_prediction, selected_rule),
        "shuffle": _metrics(confirmation, con_shuffle, "samewell_selection_shuffle"),
    }
    con_gain = confirmation_metrics["base"]["rmse"] - confirmation_metrics["selected"]["rmse"]
    con_far_gain = confirmation_metrics["base"]["far_gt2000_rmse"] - confirmation_metrics["selected"]["far_gt2000_rmse"]
    con_shuffle_loss = confirmation_metrics["shuffle"]["rmse"] - confirmation_metrics["selected"]["rmse"]
    confirmation_checks = {
        "gain_ge_0p03": con_gain >= 0.03,
        "far_gain_ge_0": con_far_gain >= 0.0,
        "shuffle_loses_ge_0p03": con_shuffle_loss >= 0.03,
    }

strict_pass = bool(mechanism_pass and confirmation_checks and all(confirmation_checks.values()))
verdict = {
    "experiment": "main342_samewell_endpoint_rank_transfer_c1",
    "stage": "samewell_visible_replay_endpoint_rank_transfer",
    "legal_boundary": "pseudo-cut candidates scored only before official PS",
    "population_control_folds": list(TRAIN_FOLDS),
    "selection_fold": SELECTION_FOLD,
    "confirmation_fold": CONFIRMATION_FOLD if mechanism_pass else None,
    "fold1_touched": False,
    "coverage": coverage,
    "failures": len(failures),
    "candidate_menu": menu,
    "rank_transfer": {
        "active_wells": int(len(active_detail)),
        "pseudo_top1_official_oracle_hit_rate": top1_hit_rate,
        "median_within_well_rank_spearman": median_rank_correlation,
    },
    "population_fixed_candidate": fixed_name,
    "selected_rule": selected_rule,
    "selected_dose": selected_dose,
    "selection": selection_metrics,
    "selection_gate": {
        "gain": selection_gain,
        "increment_vs_fixed": selection_increment,
        "shuffle_loss": selection_shuffle_loss,
        "active_fraction": active_fraction,
        "mechanism_pass": mechanism_pass,
    },
    "confirmation": confirmation_metrics,
    "confirmation_checks": confirmation_checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> replay fixed endpoint menu on each Kaggle test well and apply frozen rule/dose"
        if strict_pass else
        "FAIL -> pseudo-cut endpoint winners do not transfer to the official suffix; close C1"
    ),
}
selection_scan.to_csv(OUT_DIR / "selection_scan.csv", index=False)
(OUT_DIR / "verdict.json").write_text(json.dumps(_clean(verdict), indent=2), encoding="utf-8")
print("\nM342-C1 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2), flush=True)
print("saved:", OUT_DIR)
