"""V149: Fold5 candidate-source and consumption-formula oracle attribution.

Read-only diagnostic built on the V148 engine.  Truth is used only to choose
the best candidate for each oracle metric.  It does not train a selector and
does not produce a deployable endpoint.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = ROOT / "full5_blendaware_candidate_support_oracle_v145_cell.py"
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

source = SOURCE.read_text(encoding="utf-8")
source = source.replace(
    'EXPERIMENT = "full5_blendaware_candidate_support_oracle_v145"',
    'EXPERIMENT = "fold5_candidate_source_formula_oracle_v149"',
    1,
)
source = source.replace("FOLDS = (1, 2, 3, 4, 5)", "FOLDS = (5,)", 1)
source = source.replace(
    "TOPK_VALUES = (1, 4, 8, 16, 32, 64)",
    "TOPK_VALUES = (512, 1024, 1536, 2048)",
    1,
)
source = source.replace("V145 MATCHED SUMMARY", "V149 MATCHED SUMMARY")
source = source.replace("V145 FINAL VERDICT", "V149 FINAL VERDICT")
source = source.replace("V145 candidate oracle", "V149 candidate oracle")
source = source.replace(
    'frame["well"] = frame["well"].astype(str)\nrequired_columns = {',
    'frame["well"] = frame["well"].astype(str)\n'
    'frame = frame[frame["fold"].isin(FOLDS)].copy()\n'
    'required_columns = {',
    1,
)
source = source.replace(
    'if fold_counts != {1: 155, 2: 155, 3: 154, 4: 155, 5: 154}:',
    'if fold_counts != {5: 154}:',
    1,
)
source = source.replace(
    'if len(frame) != 3_783_989 or frame["well"].nunique() != 773 or frame["id"].duplicated().any():',
    'if len(frame) != 756_245 or frame["well"].nunique() != 154 or frame["id"].duplicated().any():',
    1,
)

new_loop = '''FORMULAS = ("current", "dose025", "dose050", "dose100", "full_move", "anchor_exact", "raw_path")
SOURCE_FORMULAS = ("current", "full_move", "anchor_exact", "raw_path")
ARM_IDS = [str(arm["id"]) for arm in getattr(Q4, "SELECTED_ARMS", [Q4.ARM_BY_ID[arm_id] for arm_id in Q4.SELECTED_ARM_IDS])]
formula_parts = {name: [] for name in FORMULAS}
source_formula_parts = {(arm, name): [] for arm in ARM_IDS for name in SOURCE_FORMULAS}
top_parts = {topk: [] for topk in TOPK_VALUES}
base_parts = []
v144_parts = []
matched_parts = []
report_rows = []
dose_parities = []

for number, (well, local) in enumerate(frame.groupby("well", sort=True), 1):
    local = local.sort_values("row_idx", kind="stable").copy()
    fold = int(local["fold"].iloc[0])
    truth = local[TRUTH].to_numpy(np.float64)
    exact = local["exact_v76"].to_numpy(np.float64)
    parent = local[PARENT].to_numpy(np.float64)
    coarse = local[COARSE].to_numpy(np.float64)
    distance = local["md_since"].to_numpy(np.float64)
    base = local[BASE].to_numpy(np.float64)
    dose, dose_parity = _recover_dose(local)
    dose_parities.append(dose_parity)
    top_predictions = {topk: base.copy() for topk in TOPK_VALUES}
    formula_predictions = {name: base.copy() for name in FORMULAS}
    source_predictions = {(arm, name): base.copy() for arm in ARM_IDS for name in SOURCE_FORMULAS}
    report = {"well": str(well), "fold": fold, "status": "fallback", "reason": "", "dose": dose}
    try:
        record = V45["_load_record"](str(well))
        eval_rows = record["eval_rows"]
        paths, source_mask, arm_rank, arm_score_z, segments, _ = Q4._candidate_union(record, prior123)
        panel_path = V96_PANEL_DIR / f"{well}.npz"
        if not panel_path.is_file():
            raise FileNotFoundError(panel_path)
        with np.load(panel_path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32, copy=False)
        if len(features) != len(paths):
            raise RuntimeError(f"candidate count mismatch {len(features)}/{len(paths)}")
        scores = models45[fold].predict(features)
        order = _deduplicated_order(paths, scores)
        available_max = min(MAX_TOPK, len(order))
        if available_max < 1:
            raise RuntimeError("no unique candidates")
        ordered_sources = source_mask[order[:available_max]]
        taper = _smoothstep((distance - FREEZE_FT) / RAMP_FT)
        sse_by_formula = {name: np.empty(available_max, np.float64) for name in FORMULAS}
        predictions_by_formula = {name: [] for name in FORMULAS}
        for rank, index in enumerate(order[:available_max]):
            trajectory = np.asarray(B45["_path_prediction"](record, prior123, paths[index], segments)[eval_rows], np.float64)
            if len(trajectory) != len(local) or not np.isfinite(trajectory).all():
                raise RuntimeError(f"invalid candidate rank {rank + 1}")
            formulas = {
                "current": exact + taper * np.clip(dose * (trajectory - coarse), -CAP_FT, CAP_FT),
                "dose025": exact + taper * np.clip(0.25 * (trajectory - coarse), -CAP_FT, CAP_FT),
                "dose050": exact + taper * np.clip(0.50 * (trajectory - coarse), -CAP_FT, CAP_FT),
                "dose100": exact + taper * np.clip(1.00 * (trajectory - coarse), -CAP_FT, CAP_FT),
                "full_move": exact + taper * np.clip(trajectory - coarse, -CAP_FT, CAP_FT),
                "anchor_exact": exact + taper * np.clip(trajectory - exact, -CAP_FT, CAP_FT),
                "raw_path": trajectory,
            }
            for name, prediction in formulas.items():
                prediction = np.asarray(prediction, np.float64)
                predictions_by_formula[name].append(prediction)
                sse_by_formula[name][rank] = float(np.sum((prediction - truth) ** 2))
        predictions_by_formula = {name: np.asarray(values, np.float64) for name, values in predictions_by_formula.items()}
        for name in FORMULAS:
            best_index = int(np.argmin(sse_by_formula[name]))
            formula_predictions[name] = predictions_by_formula[name][best_index]
        for topk in TOPK_VALUES:
            effective_topk = min(topk, available_max)
            best_index = int(np.argmin(sse_by_formula["current"][:effective_topk]))
            top_predictions[topk] = predictions_by_formula["current"][best_index]
        for arm_index, arm_id in enumerate(ARM_IDS):
            eligible = ((ordered_sources.astype(np.uint64) >> arm_index) & 1) == 1
            for name in SOURCE_FORMULAS:
                if not eligible.any():
                    continue
                eligible_indices = np.flatnonzero(eligible)
                best_local = int(eligible_indices[np.argmin(sse_by_formula[name][eligible_indices])])
                source_predictions[(arm_id, name)] = predictions_by_formula[name][best_local]
        report.update({
            "status": "reused",
            "candidate_count": int(len(paths)),
            "unique_candidates": int(len(order)),
            "available_candidates_evaluated": int(available_max),
            "best_rank_current": int(np.argmin(sse_by_formula["current"]) + 1),
        })
    except Exception as exc:
        report["reason"] = repr(exc)
    matched_parts.append(local)
    base_parts.append(base)
    v144_parts.append(local[V144_ENDPOINT].to_numpy(np.float64))
    for name in FORMULAS:
        formula_parts[name].append(formula_predictions[name])
    for key in source_formula_parts:
        source_formula_parts[key].append(source_predictions[key])
    for topk in TOPK_VALUES:
        top_parts[topk].append(top_predictions[topk])
    report_rows.append(report)
    if number % 20 == 0 or number == frame["well"].nunique():
        fallbacks = sum(row["status"] == "fallback" for row in report_rows)
        print(f"V149 candidate audit {number}/{frame['well'].nunique()} fallbacks={fallbacks} elapsed_min={(time.time() - STARTED) / 60.0:.1f}", flush=True)
'''
loop_start = source.find('prediction_parts = {topk: [] for topk in TOPK_VALUES}')
tail_start = source.find('matched = pd.concat(matched_parts, ignore_index=True)', loop_start)
if loop_start < 0 or tail_start < 0:
    raise RuntimeError("V149 loop anchors not found")
source = source[:loop_start] + new_loop + source[tail_start:]

tail_marker = 'matched = pd.concat(matched_parts, ignore_index=True)\n'
tail_start = source.index(tail_marker)
source = source[:tail_start] + '''matched = pd.concat(matched_parts, ignore_index=True)
base_prediction = np.concatenate(base_parts)
v144_prediction = np.concatenate(v144_parts)
formula_predictions = {name: np.concatenate(values) for name, values in formula_parts.items()}
source_predictions = {key: np.concatenate(values) for key, values in source_formula_parts.items()}
top_predictions = {topk: np.concatenate(values) for topk, values in top_parts.items()}
report = pd.DataFrame(report_rows)
OUT_DIR.mkdir(parents=True, exist_ok=True)
report.to_csv(OUT_DIR / "bywell_source_formula_oracle.csv", index=False)
fallbacks = int(report["status"].eq("fallback").sum())
if fallbacks > 5:
    failed = report.loc[report["status"].eq("fallback"), ["well", "fold", "reason"]]
    raise RuntimeError(f"Too many V149 fallbacks ({fallbacks}): {failed.head().to_dict('records')}")

summary_rows = []
for fold in (0, 5):
    mask = np.ones(len(matched), bool) if fold == 0 else matched["fold"].to_numpy() == fold
    local = matched.loc[mask]
    summary_rows.append({"fold": fold, **_metric(local, base_prediction[mask], BASE)})
    summary_rows.append({"fold": fold, **_metric(local, v144_prediction[mask], V144_ENDPOINT)})
    for name in FORMULAS:
        summary_rows.append({"fold": fold, **_metric(local, formula_predictions[name][mask], f"formula_oracle_{name}")})
    for topk in TOPK_VALUES:
        summary_rows.append({"fold": fold, **_metric(local, top_predictions[topk][mask], f"current_top{topk}")})
    for arm_id in ARM_IDS:
        for name in SOURCE_FORMULAS:
            summary_rows.append({"fold": fold, **_metric(local, source_predictions[(arm_id, name)][mask], f"source_{arm_id}_{name}")})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "matched_summary.csv", index=False)
pooled = summary[summary["fold"].eq(0)].set_index("endpoint")
formula_values = {name: float(pooled.loc[f"formula_oracle_{name}", "rmse"]) for name in FORMULAS}
source_values = {
    arm_id: {name: float(pooled.loc[f"source_{arm_id}_{name}", "rmse"]) for name in SOURCE_FORMULAS}
    for arm_id in ARM_IDS
}
verdict = {
    "experiment": EXPERIMENT,
    "stage": "Fold5_full_candidate_source_and_formula_oracle",
    "deployable": False,
    "truth_usage": "oracle candidate selection and metrics only",
    "coverage": {"rows": len(matched), "wells": int(matched["well"].nunique()), "fallbacks": fallbacks},
    "candidate_depth": {"requested": list(TOPK_VALUES), "effective": int(report["available_candidates_evaluated"].max())},
    "formula_oracle_rmse": formula_values,
    "source_oracle_rmse": source_values,
    "best_formula": min(formula_values, key=formula_values.get),
    "best_source_formula": min(
        ((arm_id, name) for arm_id in ARM_IDS for name in SOURCE_FORMULAS),
        key=lambda item: source_values[item[0]][item[1]],
    ),
    "checks": {
        "coverage_exact": len(matched) == 756245 and matched["well"].nunique() == 154,
        "fallbacks_le_5": fallbacks <= 5,
        "formula_oracle_lt_6p5": min(formula_values.values()) < 6.5,
        "source_formula_oracle_lt_6p5": min(value for values in source_values.values() for value in values.values()) < 6.5,
    },
    "next": "Use the winning source/formula family for a legal selector" if min(formula_values.values()) < 6.5 else "Replace candidate generation or transition prior; current Fold5 bank remains above 6.5 even under oracle",
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
_write_json(OUT_DIR / "verdict.json", verdict)
print("\\nV149 MATCHED SUMMARY")
print(summary.to_string(index=False), flush=True)
print("\\nV149 FINAL VERDICT")
print(json.dumps(_clean(verdict), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_DIR, flush=True)
'''

code = compile(source, str(SOURCE), "exec")
exec(code, {"__name__": "__main__", "__file__": str(SOURCE)})
