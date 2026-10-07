"""Pseudo-rerun and impostor controls for samewell_crossid_retrieval_audit_v0.

This script first runs the read-only v0 audit, then calibrates cross-ID recall
and false activation without touching submission.csv.
"""

from __future__ import annotations

import json
import runpy
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "samewell_pseudorerun_controls_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONTROL_WELLS = int(globals().get("SAMEWELL_CONTROL_WELLS", 24))
SEED = 20260720


namespace = runpy.run_path(
    str(ROOT / "samewell_crossid_retrieval_audit_v0_cell.py"),
    init_globals={
        "SAMEWELL_DATA_ROOT": ROOT,
        "SAMEWELL_OUTPUT_DIR": OUT_DIR / "bootstrap_audit",
    },
)

Fingerprint = namespace["Fingerprint"]
_numeric = namespace["_numeric"]
_resample = namespace["_resample"]
_standardise = namespace["_standardise"]
_unit = namespace["_unit"]
_relative_geometry = namespace["_relative_geometry"]
_canonical_typewell_hash = namespace["_canonical_typewell_hash"]
_preliminary_scores = namespace["_preliminary_scores"]
_search_alignment = namespace["_search_alignment"]
_fine_score = namespace["_fine_score"]
_suggest_action = namespace["_suggest_action"]
_jsonable = namespace["_jsonable"]
train_bank = namespace["train_bank"]
bank_lookup = namespace["bank_lookup"]
GR_POINTS = namespace["GR_POINTS"]
TYPEWELL_POINTS = namespace["TYPEWELL_POINTS"]


def _pseudo_fingerprint(alias, horizontal, source):
    md = _numeric(horizontal, "MD")
    gr_raw = _numeric(horizontal, "GR")
    gr = _unit(_standardise(_resample(md, gr_raw, GR_POINTS)))
    gr_reverse = _unit(
        _standardise(_resample(-md[::-1], gr_raw[::-1], GR_POINTS))
    )
    geometry = _relative_geometry(
        md,
        _numeric(horizontal, "X"),
        _numeric(horizontal, "Y"),
        _numeric(horizontal, "Z"),
        reverse=False,
    )
    geometry_reverse = _relative_geometry(
        md,
        _numeric(horizontal, "X"),
        _numeric(horizontal, "Y"),
        _numeric(horizontal, "Z"),
        reverse=True,
    )
    typewell_vector = np.zeros(TYPEWELL_POINTS, np.float32)
    typewell_hash = ""
    if source.typewell_path is not None:
        typewell = pd.read_csv(source.typewell_path)
        tvt = _numeric(typewell, "TVT")
        tw_gr = _numeric(typewell, "GR")
        typewell_vector = _unit(
            _standardise(_resample(tvt, tw_gr, TYPEWELL_POINTS))
        )
        typewell_hash = _canonical_typewell_hash(tvt, tw_gr)
    finite_md = md[np.isfinite(md)]
    return Fingerprint(
        well=alias,
        horizontal_path=Path(f"pseudo://{alias}"),
        typewell_path=source.typewell_path,
        rows=len(horizontal),
        md_min=float(finite_md.min()),
        md_max=float(finite_md.max()),
        gr=gr,
        gr_reverse=gr_reverse,
        geometry=geometry,
        geometry_reverse=geometry_reverse,
        typewell=typewell_vector,
        typewell_hash=typewell_hash,
    )


def _pseudo_query(source_frame, variant, rng):
    frame = source_frame.copy().reset_index(drop=True)
    if variant == "renamed_full":
        return frame
    if variant == "shifted_noisy":
        frame["MD"] = pd.to_numeric(frame["MD"], errors="coerce") + rng.uniform(80, 240)
        frame["X"] = pd.to_numeric(frame["X"], errors="coerce") + rng.uniform(-5000, 5000)
        frame["Y"] = pd.to_numeric(frame["Y"], errors="coerce") + rng.uniform(-5000, 5000)
        gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(float)
        scale = max(float(np.nanstd(gr)), 1.0)
        frame["GR"] = gr + rng.normal(0.0, 0.015 * scale, len(frame))
        return frame
    if variant == "partial_shifted":
        lower = int(round(0.03 * len(frame)))
        upper = int(round(0.90 * len(frame)))
        frame = frame.iloc[lower:max(upper, lower + 100)].copy().reset_index(drop=True)
        frame["MD"] = pd.to_numeric(frame["MD"], errors="coerce") + rng.uniform(60, 180)
        frame["X"] = pd.to_numeric(frame["X"], errors="coerce") + rng.uniform(-3000, 3000)
        frame["Y"] = pd.to_numeric(frame["Y"], errors="coerce") + rng.uniform(-3000, 3000)
        return frame
    raise KeyError(variant)


def _aligned_row(query_fp, query_frame, candidate_row):
    train_well = str(candidate_row["train_well"])
    train_frame = pd.read_csv(bank_lookup[train_well].horizontal_path)
    aligned = _search_alignment(
        query_frame,
        train_frame,
        int(candidate_row["fingerprint_orientation"]),
    )
    row = candidate_row.to_dict()
    row.update(aligned)
    row["query_well"] = query_fp.well
    row["fine_score"] = _fine_score(row)
    return row


rng = np.random.default_rng(SEED)
sample_indices = np.sort(
    rng.choice(len(train_bank), size=min(CONTROL_WELLS, len(train_bank)), replace=False)
)
positive_rows = []
negative_rows = []
started = time.time()

for sample_number, bank_index in enumerate(sample_indices, 1):
    source = train_bank[int(bank_index)]
    source_frame = pd.read_csv(source.horizontal_path)
    for variant in ("renamed_full", "shifted_noisy", "partial_shifted"):
        query_frame = _pseudo_query(source_frame, variant, rng)
        alias = f"alias_{variant}_{source.well}"
        query_fp = _pseudo_fingerprint(alias, query_frame, source)
        preliminary = _preliminary_scores(query_fp, train_bank)
        top = preliminary.iloc[0]
        preliminary_retrieved = str(top["train_well"]) == source.well
        selected = (
            pd.concat(
                [
                    preliminary.head(3),
                    preliminary[preliminary["typewell_hash_match"]],
                ],
                ignore_index=True,
            )
            .drop_duplicates("train_well")
            .sort_values("preliminary_score", ascending=False)
        )
        fine_candidates = []
        for candidate in selected.itertuples(index=False):
            try:
                fine_candidates.append(
                    _aligned_row(
                        query_fp,
                        query_frame,
                        pd.Series(candidate._asdict()),
                    )
                )
            except Exception:
                continue
        try:
            if not fine_candidates:
                raise RuntimeError("all positive candidate alignments failed")
            fine_candidates.sort(
                key=lambda item: item["fine_score"], reverse=True
            )
            row = fine_candidates[0]
            second_score = (
                fine_candidates[1]["fine_score"]
                if len(fine_candidates) > 1
                else -np.inf
            )
            fine_margin = (
                float(row["fine_score"] - second_score)
                if np.isfinite(second_score)
                else 1.0
            )
            final_retrieved = str(row["train_well"]) == source.well
            action = _suggest_action(pd.Series(row), fine_margin)
            row.update(
                {
                    "source_well": source.well,
                    "variant": variant,
                    "preliminary_top1": str(top["train_well"]),
                    "preliminary_retrieved_source": preliminary_retrieved,
                    "final_retrieved_source": final_retrieved,
                    "fine_margin": fine_margin,
                    "suggested_action": action,
                    "status": "ok",
                }
            )
        except Exception as error:
            row = {
                "source_well": source.well,
                "variant": variant,
                "preliminary_top1": str(top["train_well"]),
                "preliminary_retrieved_source": preliminary_retrieved,
                "final_retrieved_source": False,
                "suggested_action": "failed",
                "status": f"{type(error).__name__}:{error}",
            }
        positive_rows.append(row)

    alias = f"impostor_{source.well}"
    query_fp = _pseudo_fingerprint(alias, source_frame, source)
    preliminary = _preliminary_scores(query_fp, train_bank)
    impostors = preliminary[~preliminary["train_well"].eq(source.well)].head(2)
    fine_impostors = []
    for candidate in impostors.itertuples(index=False):
        try:
            fine_impostors.append(
                _aligned_row(query_fp, source_frame, pd.Series(candidate._asdict()))
            )
        except Exception:
            continue
    if fine_impostors:
        fine_impostors.sort(key=lambda row: row["fine_score"], reverse=True)
        best = fine_impostors[0]
        second = fine_impostors[1]["fine_score"] if len(fine_impostors) > 1 else -np.inf
        margin = float(best["fine_score"] - second) if np.isfinite(second) else 0.0
        action = _suggest_action(pd.Series(best), margin)
        best.update(
            {
                "source_well": source.well,
                "excluded_true_well": True,
                "suggested_action": action,
                "fine_margin": margin,
                "status": "ok",
            }
        )
        negative_rows.append(best)
    else:
        negative_rows.append(
            {
                "source_well": source.well,
                "excluded_true_well": True,
                "suggested_action": "failed",
                "status": "all impostor alignments failed",
            }
        )
    print(f"control wells: {sample_number}/{len(sample_indices)}", flush=True)


positive = pd.DataFrame(positive_rows)
negative = pd.DataFrame(negative_rows)
positive.to_csv(OUT_DIR / "positive_pseudoreruns.csv", index=False)
negative.to_csv(OUT_DIR / "negative_impostors.csv", index=False)

positive_summary = (
    positive.groupby("variant", dropna=False)
    .agg(
        samples=("source_well", "size"),
        preliminary_recall=("preliminary_retrieved_source", "mean"),
        final_recall=("final_retrieved_source", "mean"),
        full_or_graded_rate=(
            "suggested_action",
            lambda values: float(pd.Series(values).isin(["full", "graded"]).mean()),
        ),
        prefix_bias_rmse_p50=("prefix_bias_rmse", "median"),
        prefix_bias_rmse_p95=("prefix_bias_rmse", lambda values: float(np.nanquantile(values, 0.95))),
        gr_ncc_p05=("gr_smooth21_ncc", lambda values: float(np.nanquantile(values, 0.05))),
        eval_coverage_p05=("eval_overlap_coverage", lambda values: float(np.nanquantile(values, 0.05))),
    )
    .reset_index()
)
false_activation = float(
    negative["suggested_action"].isin(["full", "graded"]).mean()
)
verdict = {
    "experiment": "samewell_pseudorerun_controls_v1",
    "control_wells": int(len(sample_indices)),
    "positive_variants": positive_summary.to_dict(orient="records"),
    "impostor_false_activation_rate": false_activation,
    "impostor_full_count": int(negative["suggested_action"].eq("full").sum()),
    "impostor_graded_count": int(negative["suggested_action"].eq("graded").sum()),
    "gates": {
        "each_positive_final_recall_ge": 0.95,
        "each_positive_activation_ge": 0.90,
        "impostor_false_activation_le": 0.01,
    },
    "checks": {
        "positive_recall": bool((positive_summary["final_recall"] >= 0.95).all()),
        "positive_activation": bool((positive_summary["full_or_graded_rate"] >= 0.90).all()),
        "impostor": bool(false_activation <= 0.01),
    },
    "strict_pass": False,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "outputs": {
        "positive": str(OUT_DIR / "positive_pseudoreruns.csv"),
        "negative": str(OUT_DIR / "negative_impostors.csv"),
    },
}
verdict["strict_pass"] = bool(all(verdict["checks"].values()))
verdict["next"] = (
    "PASS -> run read-only hidden audit and inspect cross-ID candidates"
    if verdict["strict_pass"]
    else "FAIL -> recalibrate retrieval/alignment guards before any override"
)
verdict = _jsonable(verdict)
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

print("\nPOSITIVE CONTROL SUMMARY")
print(positive_summary.to_string(index=False))
print("\nIMPOSTOR ACTIONS")
print(negative["suggested_action"].value_counts(dropna=False).to_string())
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
