"""M342-W2: whitened in-domain GR latent-distance observability gateway.

W1 showed that PCA denoising suppresses candidate recall. W2 tests the opposite
hypothesis: discriminative events live in lower-variance components. It reuses
W1's Fold4/5 PCA fit and HN23 candidate protocol, replacing reconstruction
cosine with variance-whitened latent Mahalanobis similarity.
"""

from pathlib import Path
import json
import sys
import time
import types

import numpy as np
import pandas as pd


STARTED_W2 = time.time()
ROOT_W2 = Path("/root/autodl-tmp")
W1_SOURCE_W2 = ROOT_W2 / "main342_indomain_gr_denoising_observability_w1.py"
OUT_W2 = ROOT_W2 / "oof_exports" / "main342_indomain_gr_whitened_observability_w2"
OUT_W2.mkdir(parents=True, exist_ok=True)

COMPONENTS_W2 = (8, 16, 24, 32)
GAMMAS_W2 = (0.0, 0.25, 0.50, 1.0)
MIN_CONFIRMATION_WELLS_W2 = 120


def _clean_w2(value):
    if isinstance(value, dict):
        return {str(key): _clean_w2(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_w2(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _load_w1_prefix_w2():
    source = W1_SOURCE_W2.read_text(encoding="utf-8")
    marker = "fold3_wells = sorted("
    if marker not in source:
        raise RuntimeError("W1 execution marker not found")
    name = "m342_w2_w1_runtime"
    module = types.ModuleType(name)
    module.__file__ = str(W1_SOURCE_W2)
    sys.modules[name] = module
    try:
        exec(compile(source.split(marker, 1)[0], str(W1_SOURCE_W2), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module.__dict__


W = _load_w1_prefix_w2()
pca_w2 = W["pca"]
H_w2 = W["H"]


def _whitened_scores_w2(horizontal, candidates):
    horizontal = W["_standardise_rows"](np.asarray(horizontal, np.float64)[None, :])[0]
    candidates = W["_standardise_rows"](candidates)
    output = {"raw_amplitude": H_w2["_cosine_rows"](horizontal, candidates)}
    h_score = pca_w2.transform(horizontal[None, :])[0]
    q_score = pca_w2.transform(candidates)
    variance = np.maximum(pca_w2.explained_variance_, 1e-8)
    for components in COMPONENTS_W2:
        difference = q_score[:, :components] - h_score[None, :components]
        for gamma in GAMMAS_W2:
            weight = np.power(variance[:components], -float(gamma))
            distance = np.sum(np.square(difference) * weight[None, :], axis=1)
            distance /= max(float(np.sum(weight)), 1e-12)
            gamma_label = str(gamma).replace(".", "p")
            output[f"white_k{components}_g{gamma_label}"] = -distance
    return output


# _evaluate resolves _scores through its original module global dictionary.
W["_scores"] = _whitened_scores_w2
fold_by_well_w2 = W["fold_by_well"]
fold3_wells_w2 = sorted(
    well for well, fold in fold_by_well_w2.items() if fold == W["DISCOVERY_FOLD"]
)
discovery_wells_w2 = fold3_wells_w2[: W["DISCOVERY_WELLS"]]
confirmation_wells_w2 = sorted(
    well for well, fold in fold_by_well_w2.items() if fold == W["CONFIRMATION_FOLD"]
)

discovery_w2, failures_a_w2 = W["_evaluate"](discovery_wells_w2, "aligned")
discovery_summary_w2 = W["_summary"](discovery_w2)
latent_discovery_w2 = discovery_summary_w2[
    discovery_summary_w2["config"].str.startswith("white_")
].sort_values(
    ["top32_recall", "top8_recall", "argmax_rmse"],
    ascending=[False, False, True],
)
if latent_discovery_w2.empty:
    raise RuntimeError("No W2 discovery rows")
selected_w2 = latent_discovery_w2.iloc[0].to_dict()
selected_config_w2 = str(selected_w2["config"])
selected_window_w2 = int(selected_w2["half_window"])

aligned_records_w2, failures_b_w2 = W["_evaluate"](
    confirmation_wells_w2, "aligned", half_windows=(selected_window_w2,)
)
centres_w2 = {}
for well in confirmation_wells_w2:
    table = pd.read_csv(W["DATA_ROOT"] / "train" / f"{well}__typewell.csv", usecols=["TVT"])
    centres_w2[well] = float(pd.to_numeric(table["TVT"], errors="coerce").median())
ordered_w2 = sorted(confirmation_wells_w2, key=lambda well: (centres_w2[well], well))
shuffle_map_w2 = dict(zip(ordered_w2, ordered_w2[1:] + ordered_w2[:1]))
shuffled_records_w2, failures_c_w2 = W["_evaluate"](
    confirmation_wells_w2,
    "typewell_shuffle",
    shuffle_map_w2,
    half_windows=(selected_window_w2,),
)
confirmation_summary_w2 = W["_summary"](
    pd.concat([aligned_records_w2, shuffled_records_w2], ignore_index=True)
)


def _row_w2(control, config):
    local = confirmation_summary_w2[
        confirmation_summary_w2["control"].eq(control)
        & confirmation_summary_w2["config"].eq(config)
    ]
    if len(local) != 1:
        raise RuntimeError(f"Missing W2 summary row: {control}/{config}")
    return local.iloc[0]


aligned_w2 = _row_w2("aligned", selected_config_w2)
raw_w2 = _row_w2("aligned", "raw_amplitude")
shuffle_w2 = _row_w2("typewell_shuffle", selected_config_w2)
shuffle_raw_w2 = _row_w2("typewell_shuffle", "raw_amplitude")
gain32_w2 = float(aligned_w2["top32_recall"] - raw_w2["top32_recall"])
gain8_w2 = float(aligned_w2["top8_recall"] - raw_w2["top8_recall"])
shuffle_gain32_w2 = float(shuffle_w2["top32_recall"] - shuffle_raw_w2["top32_recall"])
argmax_gain_w2 = float(raw_w2["argmax_rmse"] - aligned_w2["argmax_rmse"])

checks_w2 = {
    "confirmation_wells_ge_120": int(aligned_w2["wells"]) >= MIN_CONFIRMATION_WELLS_W2,
    "top32_gain_ge_0p03": gain32_w2 >= 0.03,
    "top8_gain_ge_0p015": gain8_w2 >= 0.015,
    "aligned_increment_exceeds_shuffle_by_0p015": gain32_w2 - shuffle_gain32_w2 >= 0.015,
    "argmax_rmse_improves_ge_2ft": argmax_gain_w2 >= 2.0,
}
strict_pass_w2 = bool(all(checks_w2.values()))
verdict_w2 = {
    "experiment": "main342_indomain_gr_whitened_observability_w2",
    "stage": "fold45_selfsupervised_whitening_fold3_selection_fold2_observability",
    "target_usage_in_representation": False,
    "source": W1_SOURCE_W2,
    "protocol": {
        "representation_folds": W["REPRESENTATION_FOLDS"],
        "discovery_fold": W["DISCOVERY_FOLD"],
        "discovery_wells": len(discovery_wells_w2),
        "confirmation_fold": W["CONFIRMATION_FOLD"],
        "untouched_fold": W["UNTOUCHED_FOLD"],
    },
    "representation": {
        "patches": len(W["patch_matrix"]),
        "components": W["PCA_COMPONENTS"],
        "gammas": GAMMAS_W2,
    },
    "selected": {"config": selected_config_w2, "half_window": selected_window_w2},
    "confirmation": {
        "raw": raw_w2.to_dict(), "whitened": aligned_w2.to_dict(),
        "shuffle_raw": shuffle_raw_w2.to_dict(), "shuffle_whitened": shuffle_w2.to_dict(),
        "top32_gain": gain32_w2, "top8_gain": gain8_w2,
        "shuffle_top32_gain": shuffle_gain32_w2, "argmax_rmse_gain": argmax_gain_w2,
    },
    "checks": checks_w2,
    "strict_pass": strict_pass_w2,
    "failures": {
        "discovery": len(failures_a_w2), "aligned": len(failures_b_w2),
        "shuffle": len(failures_c_w2),
    },
    "elapsed_minutes": (time.time() - STARTED_W2) / 60.0,
    "next": (
        "PASS -> add frozen whitened distance to a candidate-bank ranker"
        if strict_pass_w2 else
        "FAIL -> close self-supervised GR representation; low-variance events do not resolve aliases"
    ),
}

discovery_summary_w2.to_csv(OUT_W2 / "discovery_summary.csv", index=False)
confirmation_summary_w2.to_csv(OUT_W2 / "confirmation_summary.csv", index=False)
(OUT_W2 / "verdict.json").write_text(
    json.dumps(_clean_w2(verdict_w2), indent=2), encoding="utf-8"
)
print("\nM342-W2 DISCOVERY")
print(discovery_summary_w2.to_string(index=False), flush=True)
print("\nM342-W2 CONFIRMATION")
print(confirmation_summary_w2.to_string(index=False), flush=True)
print("\nM342-W2 FINAL VERDICT")
print(json.dumps(_clean_w2(verdict_w2), indent=2), flush=True)
print("saved:", OUT_W2, flush=True)
