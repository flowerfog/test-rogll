"""Mode-level posterior audit for the 3216-path full-well candidate bank.

The previous full-bank unary retriever established three facts on Fold 2:

* the label-free 3216-path bank has a strong whole-well oracle;
* the learned energy retains useful candidates in its top-192 set;
* hard candidate argmax is catastrophically vulnerable to repeated aliases.

This is a zero-training, Fold-2-only audit.  It clusters label-free candidate
paths into geological modes before looking at truth, measures the oracle cost
of replacing every mode by a medoid path, then evaluates multiplicity-corrected
mode scores from the frozen unary energy model.  It deliberately does not tune
any setting on Fold 2 and never reads Fold 1.

Prerequisites
-------------
Upload these three files to /root/autodl-tmp (or run them from the repository):

* candidate_conditioned_fullwell_gateway_v0_cell.py
* candidate_conditioned_fullbank_retriever_v1_cell.py
* this file

The V1 Fold-2 checkpoint must already exist.  No V0/V1 execution tail is
rerun; definitions are loaded by parsing their source files.
"""

from __future__ import annotations

from pathlib import Path
import gc
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------------------
# Restore the candidate generator and the frozen V1 energy architecture only.
# ---------------------------------------------------------------------------


SOURCE_CANDIDATES_MODE = (
    Path("/root/autodl-tmp/candidate_conditioned_fullbank_retriever_v1_cell.py"),
    Path.cwd() / "candidate_conditioned_fullbank_retriever_v1_cell.py",
)
SOURCE_PATH_MODE = next(
    (path for path in SOURCE_CANDIDATES_MODE if path.exists()), None
)
if SOURCE_PATH_MODE is None:
    raise FileNotFoundError(
        "Upload candidate_conditioned_fullbank_retriever_v1_cell.py and its V0 "
        "source to /root/autodl-tmp before running this audit."
    )
source_mode = SOURCE_PATH_MODE.read_text(encoding="utf-8")
marker_mode = "# Execute\n"
if marker_mode not in source_mode:
    raise RuntimeError(f"Cannot locate V1 execution marker in {SOURCE_PATH_MODE}")
exec(
    compile(source_mode.split(marker_mode, 1)[0], str(SOURCE_PATH_MODE), "exec"),
    globals(),
)

REQUIRED_MODE = {
    "OOF_ROOT",
    "OUT_DIR_V1",
    "MODEL_DIR_V1",
    "VALID_FOLD",
    "_load_folds",
    "_render_full_bank_v1",
    "_score_arrays_v1",
    "_row_for_index_v1",
    "_summarize_rows_v1",
    "FullBankUnaryEnergyV1",
    "_seed_everything",
    "_write_json",
}
missing_mode = sorted(REQUIRED_MODE.difference(globals()))
if missing_mode:
    raise RuntimeError("V1 definition bootstrap missing: " + repr(missing_mode))
print("MODE audit bootstrap:", SOURCE_PATH_MODE, flush=True)


# ---------------------------------------------------------------------------
# Frozen audit configuration.  These are not selected on Fold 2.
# ---------------------------------------------------------------------------


EXPERIMENT_MODE = "candidate_mode_posterior_audit_v0"
OUT_DIR_MODE = Path(OOF_ROOT) / EXPERIMENT_MODE
CACHE_DIR_MODE = OUT_DIR_MODE / "mode_memberships"
for directory_mode in (OUT_DIR_MODE, CACHE_DIR_MODE):
    directory_mode.mkdir(parents=True, exist_ok=True)

SEED_MODE = 20260730
MODE_COUNTS_MODE = (64, 96, 128)
PATH_FEATURE_POINTS_MODE = 65
KMEANS_BATCH_MODE = 512
KMEANS_MAX_ITER_MODE = 160
KMEANS_N_INIT_MODE = 3
TOP_K_MODES_MODE = (1, 2, 4, 8, 16, 32)
POSTERIOR_TEMPERATURE_MODE = 1.0

# The audit authorises a set-aware scorer only when mode compression does not
# destroy the candidate ceiling and the frozen unary energy retains the modes.
MODE_MEDOID_ORACLE_GATE_MODE = 4.50
TOP8_MEMBER_ORACLE_GATE_MODE = 5.20
TOP8_MEDOID_ORACLE_GATE_MODE = 6.20
MODE_HARD_RMSE_GATE_MODE = 8.00


def _json_default_mode(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_mode_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default_mode), encoding="utf-8"
    )


def _path_features_mode(path_delta):
    """Low-frequency shape and local slope, independent of any truth labels."""
    count = int(path_delta.shape[0])
    positions = np.linspace(0, path_delta.shape[1] - 1, PATH_FEATURE_POINTS_MODE)
    positions = np.rint(positions).astype(np.int32)
    low = path_delta[:, positions].astype(np.float32)
    anchor = low[:, :1]
    low = low - anchor
    slope = np.diff(low, axis=1)
    # Translation, low-frequency form and local changes all identify a mode.
    endpoint = path_delta[:, -1:] - path_delta[:, :1]
    feature = np.concatenate([low / 48.0, slope / 12.0, endpoint / 48.0], axis=1)
    if feature.shape[0] != count or not np.isfinite(feature).all():
        raise RuntimeError("Invalid candidate shape features")
    return feature


def _cluster_modes_mode(path_delta, mode_count, well):
    """Return deterministic labels and one candidate medoid per mode."""
    features = _path_features_mode(path_delta)
    candidates = len(features)
    clusters = min(int(mode_count), candidates)
    model = MiniBatchKMeans(
        n_clusters=clusters,
        batch_size=min(KMEANS_BATCH_MODE, candidates),
        max_iter=KMEANS_MAX_ITER_MODE,
        n_init=KMEANS_N_INIT_MODE,
        random_state=SEED_MODE + int(str(well), 16) % 1000003 + clusters,
        reassignment_ratio=0.0,
    )
    labels = model.fit_predict(features).astype(np.int16)

    # MiniBatchKMeans can leave an unused centre for a highly duplicated
    # candidate bank.  Split the most diffuse occupied mode deterministically
    # instead of treating this numerical artefact as a data failure.
    counts = np.bincount(labels, minlength=clusters).astype(np.int32)
    for empty in np.flatnonzero(counts == 0):
        donors = np.flatnonzero(counts > 1)
        if not len(donors):
            raise RuntimeError(f"cannot repair empty mode {empty} for {well}")
        donor = int(donors[np.argmax(counts[donors])])
        donor_indices = np.flatnonzero(labels == donor)
        donor_centre = features[donor_indices].mean(axis=0)
        donor_distance = np.sum(
            (features[donor_indices] - donor_centre) ** 2, axis=1
        )
        moved = int(donor_indices[np.argmax(donor_distance)])
        labels[moved] = empty
        counts[donor] -= 1
        counts[empty] += 1
    members = []
    medoids = []
    for cluster in range(clusters):
        indices = np.flatnonzero(labels == cluster)
        if not len(indices):
            raise RuntimeError(f"empty mode {cluster} for {well}")
        centre = features[indices].mean(axis=0)
        distance = np.sum((features[indices] - centre) ** 2, axis=1)
        members.append(indices.astype(np.int32))
        medoids.append(int(indices[np.argmin(distance)]))
    medoids = np.asarray(medoids, np.int32)
    sizes = np.asarray([len(values) for values in members], np.int32)
    if int(sizes.sum()) != candidates or len(np.unique(medoids)) != clusters:
        raise RuntimeError("Invalid mode partition")
    return labels, members, medoids, sizes


def _logmeanexp_mode(values):
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).mean()))


def _mode_scores_mode(scores, members):
    """All aggregations are label-free and correct for candidate multiplicity."""
    result = {"max": [], "mean": [], "logmeanexp": [], "top3mean": []}
    for indices in members:
        values = scores[indices]
        ordered = np.sort(values)[::-1]
        result["max"].append(float(ordered[0]))
        result["mean"].append(float(values.mean()))
        result["logmeanexp"].append(_logmeanexp_mode(values))
        result["top3mean"].append(float(ordered[: min(3, len(ordered))].mean()))
    return {key: np.asarray(value, np.float32) for key, value in result.items()}


def _risk_medoid_mode(path_delta, medoids, mode_scores):
    """Choose an existing mode medoid with minimal posterior shape risk."""
    posterior = np.exp(mode_scores - float(np.max(mode_scores)))
    posterior /= posterior.sum()
    paths = path_delta[medoids].astype(np.float64)
    paths = paths[:, ::8]
    distance = ((paths[:, None, :] - paths[None, :, :]) ** 2).mean(axis=2)
    risk = distance @ posterior
    return int(np.argmin(risk))


def _append_row_mode(rows, label, well, bank, candidate_index, **extra):
    row = _row_for_index_v1(well, bank, int(candidate_index), label)
    row.update(extra)
    rows.append(row)


def _summarize_frame_mode(frame):
    return _summarize_rows_v1(frame)


def _mode_cache_path(well):
    return CACHE_DIR_MODE / f"{well}.npz"


def _save_membership_mode(well, labels_by_count, medoids_by_count, sizes_by_count):
    values = {"well": np.asarray(str(well))}
    for count, labels in labels_by_count.items():
        values[f"labels_{count}"] = labels.astype(np.int16)
        values[f"medoids_{count}"] = medoids_by_count[count].astype(np.int16)
        values[f"sizes_{count}"] = sizes_by_count[count].astype(np.int16)
    np.savez_compressed(_mode_cache_path(well), **values)


# ---------------------------------------------------------------------------
# Frozen Fold-2 application.
# ---------------------------------------------------------------------------


_seed_everything(SEED_MODE)
folds_mode = _load_folds()
valid_wells_mode = sorted(
    folds_mode.loc[folds_mode["fold"].eq(VALID_FOLD), "well"].astype(str).tolist()
)
if len(valid_wells_mode) < 150:
    raise RuntimeError(f"Unexpected Fold-{VALID_FOLD} size: {len(valid_wells_mode)}")

checkpoint_path_mode = Path(MODEL_DIR_V1) / "best_fold2_unary_retriever.pt"
if not checkpoint_path_mode.exists():
    raise FileNotFoundError(
        checkpoint_path_mode,
        "Run candidate_conditioned_fullbank_retriever_v1 through its Fold-2 "
        "evaluation first.",
    )
device_mode = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_mode = torch.load(
    checkpoint_path_mode, map_location=device_mode, weights_only=False
)
model_mode = FullBankUnaryEnergyV1().to(device_mode)
model_mode.load_state_dict(checkpoint_mode["model"])
model_mode.eval()

rows_mode = {}
for count in MODE_COUNTS_MODE:
    for score_name in ("max", "mean", "logmeanexp", "top3mean"):
        rows_mode[f"b{count}_{score_name}_medoid"] = []
        rows_mode[f"b{count}_{score_name}_risk_medoid"] = []
        for top_k in TOP_K_MODES_MODE:
            rows_mode[f"b{count}_{score_name}_top{top_k}_medoid_oracle"] = []
            rows_mode[f"b{count}_{score_name}_top{top_k}_member_oracle"] = []
    rows_mode[f"b{count}_medoid_oracle"] = []

diagnostic_rows_mode = []
started_mode = time.time()
for number_mode, well_mode in enumerate(valid_wells_mode, start=1):
    bank_mode = _render_full_bank_v1(well_mode)
    scores_mode = _score_arrays_v1(
        model_mode,
        bank_mode["query"],
        bank_mode["reference"],
        bank_mode["valid"],
        bank_mode["path_delta"],
        bank_mode["heuristic"],
        device_mode,
        control="full",
    )
    labels_by_count_mode = {}
    medoids_by_count_mode = {}
    sizes_by_count_mode = {}

    for count_mode in MODE_COUNTS_MODE:
        deployable_indices = np.flatnonzero(
            np.isfinite(bank_mode["candidate_mse"])
        )
        if len(deployable_indices) < int(count_mode):
            raise RuntimeError(
                f"{well_mode} has only {len(deployable_indices)} deployable paths"
            )
        local_labels, local_members, local_medoids, sizes = _cluster_modes_mode(
            bank_mode["path_delta"][deployable_indices], count_mode, well_mode
        )
        labels = np.full(len(bank_mode["candidate_mse"]), -1, np.int16)
        labels[deployable_indices] = local_labels
        members = [deployable_indices[indices] for indices in local_members]
        medoids = deployable_indices[local_medoids]
        labels_by_count_mode[count_mode] = labels
        medoids_by_count_mode[count_mode] = medoids
        sizes_by_count_mode[count_mode] = sizes
        prefix = f"b{count_mode}"

        medoid_oracle = int(medoids[np.argmin(bank_mode["candidate_mse"][medoids])])
        _append_row_mode(
            rows_mode[f"{prefix}_medoid_oracle"],
            "medoid_oracle",
            well_mode,
            bank_mode,
            medoid_oracle,
            mode_count=int(count_mode),
        )

        score_sets = _mode_scores_mode(scores_mode, members)
        for score_name, mode_scores in score_sets.items():
            selected_mode = int(np.argmax(mode_scores))
            selected_medoid = int(medoids[selected_mode])
            _append_row_mode(
                rows_mode[f"{prefix}_{score_name}_medoid"],
                f"{score_name}_medoid",
                well_mode,
                bank_mode,
                selected_medoid,
                selected_mode=selected_mode,
                selected_mode_size=int(sizes[selected_mode]),
            )
            risk_mode = _risk_medoid_mode(
                bank_mode["path_delta"], medoids, mode_scores / POSTERIOR_TEMPERATURE_MODE
            )
            _append_row_mode(
                rows_mode[f"{prefix}_{score_name}_risk_medoid"],
                f"{score_name}_risk_medoid",
                well_mode,
                bank_mode,
                int(medoids[risk_mode]),
                selected_mode=risk_mode,
                selected_mode_size=int(sizes[risk_mode]),
            )
            order = np.argsort(-mode_scores, kind="mergesort")
            for top_k in TOP_K_MODES_MODE:
                active_modes = order[: min(int(top_k), len(order))]
                active_medoids = medoids[active_modes]
                medoid_index = int(
                    active_medoids[
                        np.argmin(bank_mode["candidate_mse"][active_medoids])
                    ]
                )
                active_members = np.concatenate([members[index] for index in active_modes])
                member_index = int(
                    active_members[
                        np.argmin(bank_mode["candidate_mse"][active_members])
                    ]
                )
                _append_row_mode(
                    rows_mode[
                        f"{prefix}_{score_name}_top{top_k}_medoid_oracle"
                    ],
                    f"{score_name}_top{top_k}_medoid_oracle",
                    well_mode,
                    bank_mode,
                    medoid_index,
                    top_k=int(top_k),
                )
                _append_row_mode(
                    rows_mode[
                        f"{prefix}_{score_name}_top{top_k}_member_oracle"
                    ],
                    f"{score_name}_top{top_k}_member_oracle",
                    well_mode,
                    bank_mode,
                    member_index,
                    top_k=int(top_k),
                )

        diagnostic_rows_mode.append(
            {
                "well": well_mode,
                "mode_count": int(count_mode),
                "candidate_count": int(len(bank_mode["candidate_mse"])),
                "deployable_candidates": int(np.isfinite(bank_mode["candidate_mse"]).sum()),
                "mode_size_min": int(sizes.min()),
                "mode_size_median": float(np.median(sizes)),
                "mode_size_max": int(sizes.max()),
                "medoid_oracle_rmse": float(
                    np.sqrt(bank_mode["candidate_mse"][medoids].min())
                ),
            }
        )
    _save_membership_mode(
        well_mode, labels_by_count_mode, medoids_by_count_mode, sizes_by_count_mode
    )
    del bank_mode, scores_mode, labels_by_count_mode, medoids_by_count_mode
    gc.collect()
    if number_mode % 5 == 0 or number_mode == len(valid_wells_mode):
        print(
            f"MODE audit fold{VALID_FOLD} {number_mode}/{len(valid_wells_mode)} "
            f"elapsed={(time.time() - started_mode) / 60.0:.1f}m",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Reports, explicit gates and a stop/go decision for a set-aware reranker.
# ---------------------------------------------------------------------------


reports_mode = {}
for label_mode, values_mode in rows_mode.items():
    frame_mode = pd.DataFrame(values_mode)
    frame_mode.to_csv(OUT_DIR_MODE / f"fold{VALID_FOLD}_{label_mode}_bywell.csv", index=False)
    report_mode = _summarize_frame_mode(frame_mode)
    report_mode["variant"] = label_mode
    reports_mode[label_mode] = report_mode

diagnostics_mode = pd.DataFrame(diagnostic_rows_mode)
diagnostics_mode.to_csv(OUT_DIR_MODE / f"fold{VALID_FOLD}_mode_diagnostics.csv", index=False)
summary_mode = pd.DataFrame(reports_mode.values()).sort_values("rmse")
summary_mode.to_csv(OUT_DIR_MODE / f"fold{VALID_FOLD}_mode_summary.csv", index=False)
print("MODE SUMMARY TOP 30")
print(summary_mode.head(30).to_string(index=False), flush=True)

recommended_count_mode = 96
recommended_score_mode = "logmeanexp"
medoid_oracle_mode = reports_mode[f"b{recommended_count_mode}_medoid_oracle"]
hard_mode = reports_mode[
    f"b{recommended_count_mode}_{recommended_score_mode}_medoid"
]
top8_member_mode = reports_mode[
    f"b{recommended_count_mode}_{recommended_score_mode}_top8_member_oracle"
]
top8_medoid_mode = reports_mode[
    f"b{recommended_count_mode}_{recommended_score_mode}_top8_medoid_oracle"
]

gates_mode = {
    "mode_medoid_oracle_rmse_le": MODE_MEDOID_ORACLE_GATE_MODE,
    "top8_member_oracle_rmse_le": TOP8_MEMBER_ORACLE_GATE_MODE,
    "top8_medoid_oracle_rmse_le": TOP8_MEDOID_ORACLE_GATE_MODE,
    "hard_mode_rmse_le": MODE_HARD_RMSE_GATE_MODE,
}
checks_mode = {
    "mode_medoid_oracle": bool(medoid_oracle_mode["rmse"] <= MODE_MEDOID_ORACLE_GATE_MODE),
    "top8_member_oracle": bool(top8_member_mode["rmse"] <= TOP8_MEMBER_ORACLE_GATE_MODE),
    "top8_medoid_oracle": bool(top8_medoid_mode["rmse"] <= TOP8_MEDOID_ORACLE_GATE_MODE),
    "hard_mode": bool(hard_mode["rmse"] <= MODE_HARD_RMSE_GATE_MODE),
}
strict_pass_mode = bool(
    checks_mode["mode_medoid_oracle"]
    and checks_mode["top8_member_oracle"]
    and checks_mode["top8_medoid_oracle"]
)
verdict_mode = {
    "experiment": EXPERIMENT_MODE,
    "stage": "fold2_zero_training_mode_multiplicity_audit",
    "sample_unit": "one complete physical well",
    "candidate_generation": "same label-free 24-base x 134-template bank as V1",
    "truth_usage": "Fold-2 truth only measures frozen mode/score decisions",
    "fold1_touched": False,
    "checkpoint": str(checkpoint_path_mode),
    "checkpoint_epoch": int(checkpoint_mode["epoch"]),
    "mode_counts": list(MODE_COUNTS_MODE),
    "recommended_frozen_recipe": {
        "mode_count": recommended_count_mode,
        "score": recommended_score_mode,
        "decoder": "mode_medoid",
        "posterior_temperature": POSTERIOR_TEMPERATURE_MODE,
    },
    "recommended_metrics": {
        "medoid_oracle": medoid_oracle_mode,
        "hard_mode": hard_mode,
        "top8_member_oracle": top8_member_mode,
        "top8_medoid_oracle": top8_medoid_mode,
    },
    "mode_diagnostics": {
        "median_mode_size": float(diagnostics_mode["mode_size_median"].median()),
        "p95_mode_size_max": float(diagnostics_mode["mode_size_max"].quantile(0.95)),
        "median_medoid_oracle_rmse": float(diagnostics_mode["medoid_oracle_rmse"].median()),
    },
    "gates": gates_mode,
    "checks": checks_mode,
    "strict_pass": strict_pass_mode,
    "elapsed_minutes": (time.time() - started_mode) / 60.0,
}
if strict_pass_mode:
    verdict_mode["next"] = (
        "PASS -> train a cross-fitted set-context mode scorer on the exact "
        "label-free clustered bank; predict mode posterior, not candidate argmax."
    )
elif checks_mode["top8_member_oracle"]:
    verdict_mode["next"] = (
        "PARTIAL -> mode coverage survives but medoid compression loses too much; "
        "train an in-mode local decoder only after exact inner-OOF mode retrieval."
    )
else:
    verdict_mode["next"] = (
        "FAIL -> the frozen unary energy does not retrieve enough coherent modes; "
        "do not train a mode scorer from this checkpoint."
    )
_write_mode_json(OUT_DIR_MODE / "verdict.json", verdict_mode)
print("MODE AUDIT FINAL VERDICT")
print(json.dumps(verdict_mode, indent=2, default=_json_default_mode), flush=True)
print("saved:", OUT_DIR_MODE, flush=True)
