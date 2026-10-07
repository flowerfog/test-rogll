"""HN23: multiscale GR phase/event observability gate.

This is a new emission family.  A candidate is a constant TVT shift of the
frozen V66 local path, so the candidate retains the mainline's local shape.
Horizontal and typewell GR are compared through multiscale derivatives and
curvature rather than absolute amplitude.  Truth is used only for ranking and
oracle audit, never in a candidate score.
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


warnings.filterwarnings("ignore")

ROOT = Path("/root/autodl-tmp/oof_exports")
DATA_ROOT = Path("/root/autodl-tmp/rogii-wellbore-geology-prediction")
V66_PATH = (
    ROOT / "frontier_6p377_full_fivefold_lineage_v66"
    / "incumbent_6p377_full_fivefold_oof.parquet"
)
OUT_DIR = ROOT / "multiscale_phase_event_observability_hn23"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260731
DISCOVERY_FOLD = 3
DISCOVERY_WELLS = 24
CONFIRMATION_FOLD = 2
ROWS_PER_WELL = 64
OFFSETS = np.arange(-128.0, 128.0 + 1e-6, 2.0, dtype=np.float64)
HALF_WINDOWS = (64, 128, 256)
MIN_WINDOW_COVERAGE = 0.35

CONFIGS = {
    "amplitude": {"amp": 1.0, "grad4": 0.0, "grad16": 0.0, "curve16": 0.0},
    "phase_short": {"amp": 0.10, "grad4": 0.55, "grad16": 0.25, "curve16": 0.10},
    "phase_balanced": {"amp": 0.15, "grad4": 0.30, "grad16": 0.35, "curve16": 0.20},
    "phase_long": {"amp": 0.05, "grad4": 0.15, "grad16": 0.50, "curve16": 0.30},
}
started = time.time()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _prediction_column(frame):
    for column in (
        "incumbent_6p377_replay", "prediction", "tvt_pred", "pred"
    ):
        if column in frame:
            return column
    raise RuntimeError(f"Cannot identify V66 prediction: {frame.columns.tolist()}")


def _fill(values):
    values = np.asarray(values, np.float64)
    good = np.isfinite(values)
    if good.sum() < 2:
        return None
    return np.interp(np.arange(len(values)), np.flatnonzero(good), values[good])


def _standardise(values, axis=-1):
    values = np.asarray(values, np.float64)
    centre = np.median(values, axis=axis, keepdims=True)
    scale = np.median(np.abs(values - centre), axis=axis, keepdims=True) * 1.4826
    scale = np.where(scale > 1e-5, scale, np.std(values, axis=axis, keepdims=True))
    return (values - centre) / np.maximum(scale, 1e-5)


def _cosine_rows(reference, candidates):
    reference = np.asarray(reference, np.float64)
    candidates = np.asarray(candidates, np.float64)
    numerator = candidates @ reference
    denominator = np.linalg.norm(candidates, axis=1) * np.linalg.norm(reference)
    return numerator / np.maximum(denominator, 1e-8)


def _components(horizontal_gr, candidate_gr):
    h = _standardise(horizontal_gr)
    q = _standardise(candidate_gr, axis=1)
    components = {"amp": _cosine_rows(h, q)}
    for sigma, name in ((4.0, "grad4"), (16.0, "grad16")):
        hs = gaussian_filter1d(h, sigma=sigma, mode="nearest")
        qs = gaussian_filter1d(q, sigma=sigma, axis=1, mode="nearest")
        hg = _standardise(np.gradient(hs))
        qg = _standardise(np.gradient(qs, axis=1), axis=1)
        correlation = _cosine_rows(hg, qg)
        # Event polarity is deliberately amplitude-free.
        polarity = np.mean(np.sign(qg) == np.sign(hg)[None, :], axis=1) * 2.0 - 1.0
        components[name] = 0.75 * correlation + 0.25 * polarity
    hs = gaussian_filter1d(h, sigma=16.0, mode="nearest")
    qs = gaussian_filter1d(q, sigma=16.0, axis=1, mode="nearest")
    hc = _standardise(np.gradient(np.gradient(hs)))
    qc = _standardise(np.gradient(np.gradient(qs, axis=1), axis=1), axis=1)
    components["curve16"] = _cosine_rows(hc, qc)
    return components


def _scores(components):
    return {
        name: sum(float(weights[key]) * components[key] for key in weights)
        for name, weights in CONFIGS.items()
    }


def _rank(scores, truth_offset):
    truth_state = int(np.argmin(np.abs(OFFSETS - truth_offset)))
    order = np.argsort(-np.asarray(scores, np.float64), kind="stable")
    rank = int(np.flatnonzero(order == truth_state)[0])
    return rank, int(order[0]), order


def _sample_positions(rows, n_take):
    rows = np.asarray(rows, np.int64)
    if len(rows) <= n_take:
        return rows
    return rows[np.unique(np.round(np.linspace(0, len(rows) - 1, n_take)).astype(int))]


def _load_well(well, local, typewell_override=None):
    horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
    typewell_name = typewell_override or well
    typewell = pd.read_csv(DATA_ROOT / "train" / f"{typewell_name}__typewell.csv")
    required_h = {"GR", "TVT", "TVT_input"}
    if not required_h.issubset(horizontal):
        raise RuntimeError(f"{well}: missing horizontal columns")
    if not {"TVT", "GR"}.issubset(typewell):
        raise RuntimeError(f"{typewell_name}: missing typewell columns")

    n_rows = len(horizontal)
    base = (
        pd.to_numeric(horizontal["TVT_input"], errors="coerce")
        .to_numpy(np.float64)
        .copy()
    )
    row_idx = local["row_idx"].to_numpy(np.int64)
    if row_idx.min() < 0 or row_idx.max() >= n_rows:
        raise RuntimeError(f"{well}: row_idx outside horizontal file")
    base[row_idx] = local["base"].to_numpy(np.float64)
    base = _fill(base)
    if base is None:
        raise RuntimeError(f"{well}: cannot build complete base path")

    horizontal_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(np.float64)
    tw_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(np.float64)
    tw_gr = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(np.float64)
    good = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    if good.sum() < 64:
        raise RuntimeError(f"{typewell_name}: too few typewell rows")
    order = np.argsort(tw_tvt[good])
    return base, horizontal_gr, truth, tw_tvt[good][order], tw_gr[good][order]


def _evaluate_wells(frame, wells, control, shuffle_map=None, half_windows=None):
    half_windows = tuple(HALF_WINDOWS if half_windows is None else half_windows)
    records = []
    failures = []
    for number, well in enumerate(wells, 1):
        local = frame.loc[frame["well"].eq(well)].sort_values("row_idx")
        well_start = len(records)
        skipped = {
            "truth_nonfinite": 0,
            "offset_outside": 0,
            "short_window": 0,
            "low_gr_coverage": 0,
            "unfillable_gr": 0,
            "candidate_support": 0,
        }
        try:
            override = shuffle_map.get(well) if shuffle_map else None
            base, hgr, truth, tw_tvt, tw_gr = _load_well(well, local, override)
            positions = _sample_positions(local["row_idx"].to_numpy(np.int64), ROWS_PER_WELL)
            for centre in positions:
                if not np.isfinite(truth[centre]):
                    skipped["truth_nonfinite"] += 1
                    continue
                truth_offset = float(truth[centre] - base[centre])
                if truth_offset < OFFSETS[0] or truth_offset > OFFSETS[-1]:
                    skipped["offset_outside"] += 1
                    continue
                for half_window in half_windows:
                    left = max(0, int(centre) - half_window)
                    right = min(len(base), int(centre) + half_window + 1)
                    if right - left < half_window:
                        skipped["short_window"] += 1
                        continue
                    horizontal_window = hgr[left:right]
                    coverage = float(np.isfinite(horizontal_window).mean())
                    if coverage < MIN_WINDOW_COVERAGE:
                        skipped["low_gr_coverage"] += 1
                        continue
                    filled_horizontal = _fill(horizontal_window)
                    if filled_horizontal is None:
                        skipped["unfillable_gr"] += 1
                        continue
                    candidate_tvt = base[left:right][None, :] + OFFSETS[:, None]
                    candidate_gr = np.interp(
                        candidate_tvt.ravel(), tw_tvt, tw_gr, left=np.nan, right=np.nan
                    ).reshape(len(OFFSETS), -1)
                    valid_candidates = np.isfinite(candidate_gr).mean(axis=1) >= 0.95
                    if valid_candidates.sum() < 64:
                        skipped["candidate_support"] += 1
                        continue
                    # Edge extrapolation is forbidden; invalid states rank last.
                    candidate_gr = np.where(
                        np.isfinite(candidate_gr), candidate_gr,
                        np.nanmedian(candidate_gr, axis=1, keepdims=True),
                    )
                    components = _components(filled_horizontal, candidate_gr)
                    score_map = _scores(components)
                    for config, score in score_map.items():
                        score = np.asarray(score, np.float64).copy()
                        score[~valid_candidates] = -np.inf
                        rank, winner, order_states = _rank(score, truth_offset)
                        records.append(
                            {
                                "control": control,
                                "well": well,
                                "fold": int(local["fold"].iloc[0]),
                                "row_idx": int(centre),
                                "half_window": int(half_window),
                                "config": config,
                                "coverage": coverage,
                                "truth_offset": truth_offset,
                                "winner_offset": float(OFFSETS[winner]),
                                "rank": rank,
                                "top8_oracle_offset": float(
                                    OFFSETS[order_states[:8]][
                                        np.argmin(np.abs(OFFSETS[order_states[:8]] - truth_offset))
                                    ]
                                ),
                                "top32_oracle_offset": float(
                                    OFFSETS[order_states[:32]][
                                        np.argmin(np.abs(OFFSETS[order_states[:32]] - truth_offset))
                                    ]
                                ),
                            }
                        )
            if len(records) == well_start:
                failures.append(
                    {
                        "well": well,
                        "control": control,
                        "reason": "no_accepted_windows",
                        **skipped,
                    }
                )
        except Exception as exc:
            failures.append({"well": well, "control": control, "reason": repr(exc)})
        if number % 25 == 0 or number == len(wells):
            print(
                f"HN23 {control} {number}/{len(wells)} records={len(records)} "
                f"failures={len(failures)}",
                flush=True,
            )
    return pd.DataFrame(records), failures


def _summary(records):
    if records.empty:
        raise RuntimeError("No accepted phase/event rows")
    rows = []
    keys = ["control", "half_window", "config"]
    for values, local in records.groupby(keys, sort=False):
        truth = local["truth_offset"].to_numpy(np.float64)
        winner = local["winner_offset"].to_numpy(np.float64)
        top8 = local["top8_oracle_offset"].to_numpy(np.float64)
        top32 = local["top32_oracle_offset"].to_numpy(np.float64)
        rows.append(
            {
                **dict(zip(keys, values)),
                "rows": len(local),
                "wells": local["well"].nunique(),
                "base_rmse": float(np.sqrt(np.mean(truth**2))),
                "argmax_rmse": float(np.sqrt(np.mean((winner - truth) ** 2))),
                "argmax_within5": float(np.mean(np.abs(winner - truth) <= 5.0)),
                "median_rank": float(np.median(local["rank"])),
                "top8_recall": float(np.mean(local["rank"] < 8)),
                "top32_recall": float(np.mean(local["rank"] < 32)),
                "top8_oracle_rmse": float(np.sqrt(np.mean((top8 - truth) ** 2))),
                "top32_oracle_rmse": float(np.sqrt(np.mean((top32 - truth) ** 2))),
            }
        )
    return pd.DataFrame(rows)


for required in (V66_PATH, DATA_ROOT / "train"):
    if not required.exists():
        raise FileNotFoundError(required)

v66 = pd.read_parquet(V66_PATH)
prediction_column = _prediction_column(v66)
if "well" not in v66 and "id" in v66:
    v66["well"] = v66["id"].astype(str).str.rsplit("_", n=1).str[0]
if "row_idx" not in v66 and "id" in v66:
    v66["row_idx"] = v66["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
v66 = v66[["well", "row_idx", "fold", "tvt_true", prediction_column]].rename(
    columns={prediction_column: "base"}
)
v66["well"] = v66["well"].astype(str).str.lower()
if v66.duplicated(["well", "row_idx"]).any():
    raise RuntimeError("Duplicate V66 keys")

fold3_wells = sorted(v66.loc[v66["fold"].eq(DISCOVERY_FOLD), "well"].unique())
rng = np.random.RandomState(SEED)
rng.shuffle(fold3_wells)
discovery_wells = sorted(fold3_wells[:DISCOVERY_WELLS])
confirmation_wells = sorted(v66.loc[v66["fold"].eq(CONFIRMATION_FOLD), "well"].unique())

discovery, failures_a = _evaluate_wells(v66, discovery_wells, "aligned")
if discovery.empty:
    diagnostic = pd.DataFrame(failures_a)
    diagnostic.to_csv(OUT_DIR / "discovery_failures.csv", index=False)
    raise RuntimeError(
        "HN23 discovery produced zero rows. Diagnostics:\n"
        + diagnostic.head(24).to_string(index=False)
    )
discovery_summary = _summary(discovery)
phase_discovery = discovery_summary.loc[
    discovery_summary["config"].ne("amplitude")
].sort_values(
    ["top32_recall", "top8_recall", "argmax_rmse"],
    ascending=[False, False, True],
)
if phase_discovery.empty:
    raise RuntimeError("No phase discovery results")
selected = phase_discovery.iloc[0]
selected_config = str(selected["config"])
selected_window = int(selected["half_window"])
print(
    "HN23 SELECTED",
    {"config": selected_config, "half_window": selected_window},
    flush=True,
)

confirmation, failures_b = _evaluate_wells(
    v66, confirmation_wells, "aligned", half_windows=(selected_window,)
)
# Match the shuffle by absolute typewell TVT range.  A fully random typewell
# often has no overlapping candidate depths and would create a trivial control.
typewell_centres = {}
for well in confirmation_wells:
    table = pd.read_csv(
        DATA_ROOT / "train" / f"{well}__typewell.csv", usecols=["TVT"]
    )
    typewell_centres[well] = float(
        np.nanmedian(pd.to_numeric(table["TVT"], errors="coerce"))
    )
ordered_for_shuffle = sorted(
    confirmation_wells, key=lambda well: (typewell_centres[well], well)
)
shifted_for_shuffle = ordered_for_shuffle[1:] + ordered_for_shuffle[:1]
shuffle_map = dict(zip(ordered_for_shuffle, shifted_for_shuffle))
shuffled, failures_c = _evaluate_wells(
    v66,
    confirmation_wells,
    "typewell_shuffle",
    shuffle_map,
    half_windows=(selected_window,),
)

all_records = pd.concat([discovery, confirmation, shuffled], ignore_index=True)
confirmation_summary = _summary(pd.concat([confirmation, shuffled], ignore_index=True))
chosen = confirmation_summary.loc[
    confirmation_summary["half_window"].eq(selected_window)
    & confirmation_summary["config"].eq(selected_config)
]
aligned = chosen.loc[chosen["control"].eq("aligned")].iloc[0]
shuffle = chosen.loc[chosen["control"].eq("typewell_shuffle")].iloc[0]
amplitude = confirmation_summary.loc[
    confirmation_summary["control"].eq("aligned")
    & confirmation_summary["half_window"].eq(selected_window)
    & confirmation_summary["config"].eq("amplitude")
].iloc[0]

checks = {
    "confirmation_wells_ge_145": int(aligned["wells"]) >= 145,
    "phase_top32_gain_vs_amplitude_ge_0p03": (
        float(aligned["top32_recall"]) - float(amplitude["top32_recall"]) >= 0.03
    ),
    "phase_top8_gain_vs_amplitude_ge_0p02": (
        float(aligned["top8_recall"]) - float(amplitude["top8_recall"]) >= 0.02
    ),
    "typewell_shuffle_top32_loss_ge_0p05": (
        float(aligned["top32_recall"]) - float(shuffle["top32_recall"]) >= 0.05
    ),
    "top32_oracle_rmse_le_6": float(aligned["top32_oracle_rmse"]) <= 6.0,
    "argmax_beats_amplitude_by_2ft": (
        float(amplitude["argmax_rmse"]) - float(aligned["argmax_rmse"]) >= 2.0
    ),
}
strict_pass = bool(all(checks.values()))

all_records.to_parquet(OUT_DIR / "sampled_candidate_ranks.parquet", index=False)
discovery_summary.to_csv(OUT_DIR / "discovery_summary.csv", index=False)
confirmation_summary.to_csv(OUT_DIR / "confirmation_summary.csv", index=False)
pd.DataFrame(failures_a + failures_b + failures_c).to_csv(
    OUT_DIR / "failures.csv", index=False
)
verdict = {
    "experiment": "multiscale_phase_event_observability_hn23",
    "stage": "local_shifted_path_multiscale_phase_event_observability_gate",
    "sample_unit": "fixed hidden-suffix rows nested in complete physical wells",
    "truth_usage": "rank and oracle audit only",
    "candidate_path": "frozen V66 local path plus one constant offset",
    "protocol": {
        "discovery_fold": DISCOVERY_FOLD,
        "discovery_wells": len(discovery_wells),
        "confirmation_fold": CONFIRMATION_FOLD,
        "confirmation_wells": len(confirmation_wells),
        "shuffle": "cyclic nearest-range typewell permutation",
    },
    "selected": {"config": selected_config, "half_window": selected_window},
    "confirmation": aligned.to_dict(),
    "amplitude_control": amplitude.to_dict(),
    "typewell_shuffle": shuffle.to_dict(),
    "checks": checks,
    "strict_pass": strict_pass,
    "elapsed_minutes": (time.time() - started) / 60.0,
    "next": (
        "PASS -> insert phase emission into a frozen candidate-path decoder"
        if strict_pass
        else "FAIL -> phase/events do not resolve the alias; close this emission family"
    ),
}
(OUT_DIR / "verdict.json").write_text(
    json.dumps(verdict, indent=2, default=_json_default), encoding="utf-8"
)
print("\nHN23 DISCOVERY")
print(discovery_summary.to_string(index=False), flush=True)
print("\nHN23 CONFIRMATION")
print(confirmation_summary.to_string(index=False), flush=True)
print("\nHN23 FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=_json_default), flush=True)
print("saved:", OUT_DIR, flush=True)
