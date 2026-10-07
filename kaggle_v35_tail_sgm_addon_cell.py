"""Paste this entire file as the final Kaggle notebook cell.

Prerequisite: the CP8 + Neighbor pipeline has already written
``/kaggle/working/submission.csv``.  This cell applies the frozen V35
tail-only SGM dose and overwrites submission.csv after validation.
"""

from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd


WORK = Path("/kaggle/working")
BASE_PATH = WORK / "submission.csv"
BACKUP_PATH = WORK / "submission_before_v35_tail_sgm.csv"
CANDIDATE_PATH = WORK / "submission_v35_tail_sgm_dose_0p0375.csv"
AUDIT_PATH = WORK / "v35_tail_sgm_submit_audit.json"

DOSE = 0.0375
START_FT = 1000.0
TRANSITION_FT = 250.0
NODE_SPACING_FT = 128.0
WINDOW_HALF_FT = 96.0
MAX_WINDOW_ROWS = 49
MIN_WINDOW_ROWS = 8
OFFSETS_FT = np.arange(-24.0, 24.01, 2.0, dtype=np.float64)
ZERO_STATE = int(np.argmin(np.abs(OFFSETS_FT)))
TEMPERATURE = 2.0
SMOOTH_PENALTY = 1.0
JUMP_PENALTY = 3.0
CENTRE_PENALTY = 0.02
started = time.time()


def find_data_root():
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ]
    candidates.extend(Path("/kaggle/input").glob("**/sample_submission.csv"))
    checked = []
    for candidate in candidates:
        root = candidate.parent if candidate.is_file() else candidate
        if root in checked:
            continue
        checked.append(root)
        test = root / "test"
        if (root / "sample_submission.csv").is_file() and test.is_dir():
            if next(test.glob("*__horizontal_well.csv"), None):
                return root.resolve()
    raise FileNotFoundError("Cannot locate competition test data")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def smoothstep(distance):
    t = np.clip((np.asarray(distance) - START_FT) / TRANSITION_FT, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def robust_scale(values, floor=5.0):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float(floor)
    centre = np.median(values)
    return float(max(1.4826 * np.median(np.abs(values - centre)), floor))


def robust_affine(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, np.float64)[valid]
    y = np.asarray(y, np.float64)[valid]
    if len(x) < 24 or np.std(x) < 1e-6:
        bias = float(np.median(y - x)) if len(x) else 0.0
        return 1.0, bias, robust_scale(y - x - bias)
    design = np.column_stack((x, np.ones(len(x))))
    weight = np.ones(len(x))
    beta = np.array([1.0, np.median(y - x)])
    for _ in range(8):
        lhs = design.T @ (weight[:, None] * design) + np.diag((1e-3, 1e-6))
        rhs = design.T @ (weight * y)
        beta = np.linalg.solve(lhs, rhs)
        beta[0] = np.clip(beta[0], 0.50, 1.50)
        residual = y - design @ beta
        scale = robust_scale(residual, 3.0)
        ratio = np.abs(residual) / max(1.5 * scale, 1e-6)
        weight = np.where(ratio <= 1.0, 1.0, 1.0 / np.maximum(ratio, 1e-6))
    residual = y - design @ beta
    return float(beta[0]), float(beta[1]), robust_scale(residual, 5.0)


def deduplicate(axis, values):
    table = pd.DataFrame({"axis": axis, "value": values}).replace([np.inf, -np.inf], np.nan)
    table = table.dropna().groupby("axis", as_index=False)["value"].median().sort_values("axis")
    if len(table) < 32:
        raise RuntimeError("typewell has too few finite unique rows")
    return table["axis"].to_numpy(np.float64), table["value"].to_numpy(np.float64)


def decode(costs):
    states = len(OFFSETS_FT)
    differences = np.abs(OFFSETS_FT[:, None] - OFFSETS_FT[None, :])
    strain = 4.0
    transition = (
        SMOOTH_PENALTY * (np.minimum(differences, strain) / strain) ** 2
        + JUMP_PENALTY * (differences > strain)
        + 0.02 * differences
    )
    dp = np.full(states, np.inf)
    dp[ZERO_STATE] = 0.0
    back = np.full((len(costs), states), -1, np.int16)
    for node in range(1, len(costs)):
        unary = costs[node] / TEMPERATURE + CENTRE_PENALTY * (OFFSETS_FT / 24.0) ** 2
        candidate = dp[:, None] + transition
        predecessor = np.argmin(candidate, axis=0)
        dp = candidate[predecessor, np.arange(states)] + unary
        back[node] = predecessor.astype(np.int16)
    path = np.empty(len(costs), np.int16)
    path[-1] = int(np.argmin(dp))
    for node in range(len(costs) - 1, 0, -1):
        path[node - 1] = back[node, path[node]]
    path[0] = ZERO_STATE
    return OFFSETS_FT[path]


def build_correction(well, row_idx, baseline, test_dir):
    horizontal = pd.read_csv(
        test_dir / f"{well}__horizontal_well.csv",
        usecols=lambda column: column in {"MD", "GR", "TVT_input"},
    ).apply(pd.to_numeric, errors="coerce")
    typewell = pd.read_csv(
        test_dir / f"{well}__typewell.csv",
        usecols=lambda column: column in {"TVT", "GR"},
    ).apply(pd.to_numeric, errors="coerce")
    if row_idx.min() < 0 or row_idx.max() >= len(horizontal):
        raise RuntimeError("submission row index outside horizontal file")

    md = horizontal["MD"].to_numpy(np.float64)
    gr = horizontal["GR"].to_numpy(np.float64)
    tvt_input = horizontal["TVT_input"].to_numpy(np.float64)
    positive_steps = np.diff(md)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0)]
    first_step = float(np.median(positive_steps)) if len(positive_steps) else 1.0
    distance = md[row_idx] - md[row_idx[0]] + first_step

    tw_tvt, tw_gr_raw = deduplicate(
        typewell["TVT"].to_numpy(np.float64), typewell["GR"].to_numpy(np.float64)
    )
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(gr))
    expected_known = np.interp(tvt_input[known], tw_tvt, tw_gr_raw, left=np.nan, right=np.nan)
    gain, bias, gamma = robust_affine(expected_known, gr[known])
    tw_gr = gain * tw_gr_raw + bias
    eval_gr = gr[row_idx]

    active = np.flatnonzero(distance >= START_FT)
    if not len(active):
        return np.zeros(len(row_idx)), {
            "nodes": 0, "usable_fraction": 0.0, "gain": gain, "bias": bias,
            "gamma": gamma, "max_abs_correction": 0.0,
        }
    targets = np.arange(START_FT, float(distance[-1]) + 0.5, NODE_SPACING_FT)
    nodes = np.searchsorted(distance, targets, side="left")
    nodes = np.clip(nodes, active[0], len(distance) - 1)
    nodes = np.unique(np.concatenate(([active[0]], nodes, [len(distance) - 1]))).astype(np.int64)
    costs = np.zeros((len(nodes), len(OFFSETS_FT)), np.float64)
    usable = np.zeros(len(nodes), bool)
    costs[0] = 1e4
    costs[0, ZERO_STATE] = 0.0
    usable[0] = True

    for node_number, node in enumerate(nodes[1:], 1):
        rows = np.flatnonzero(np.abs(distance - distance[node]) <= WINDOW_HALF_FT)
        rows = rows[np.isfinite(eval_gr[rows])]
        if len(rows) > MAX_WINDOW_ROWS:
            rows = rows[np.linspace(0, len(rows) - 1, MAX_WINDOW_ROWS).round().astype(int)]
        if len(rows) < MIN_WINDOW_ROWS:
            continue
        observed = eval_gr[rows]
        obs_centre = np.median(observed)
        obs_scale = robust_scale(observed - obs_centre, 3.0)
        state_cost = np.full(len(OFFSETS_FT), 20.0)
        for state, offset in enumerate(OFFSETS_FT):
            expected = np.interp(
                baseline[rows] + offset, tw_tvt, tw_gr, left=np.nan, right=np.nan
            )
            valid = np.isfinite(expected)
            if valid.sum() < MIN_WINDOW_ROWS:
                continue
            raw_residual = (observed[valid] - expected[valid]) / max(gamma, 1e-6)
            raw_cost = np.mean(np.log1p(np.minimum(raw_residual**2, 1e4)))
            exp_centre = np.median(expected[valid])
            exp_scale = robust_scale(expected[valid] - exp_centre, 3.0)
            shape_residual = (
                (observed[valid] - obs_centre) / obs_scale
                - (expected[valid] - exp_centre) / exp_scale
            )
            shape_cost = np.mean(np.log1p(np.minimum(shape_residual**2, 1e4)))
            coverage = valid.mean()
            state_cost[state] = (
                0.65 * raw_cost + 0.35 * shape_cost
                + 4.0 * max(0.85 - coverage, 0.0)
            )
        costs[node_number] = state_cost - np.min(state_cost)
        usable[node_number] = True

    node_offsets = decode(costs)
    correction = np.interp(
        distance, distance[nodes], node_offsets,
        left=0.0, right=float(node_offsets[-1]),
    )
    correction *= smoothstep(distance)
    correction[distance <= START_FT] = 0.0
    return correction, {
        "nodes": int(len(nodes)),
        "usable_fraction": float(np.mean(usable)),
        "gain": gain,
        "bias": bias,
        "gamma": gamma,
        "max_abs_correction": float(np.max(np.abs(correction))),
    }


if not BASE_PATH.is_file():
    raise FileNotFoundError("Run the complete CP8 + Neighbor pipeline first")
DATA_ROOT = find_data_root()
TEST_DIR = DATA_ROOT / "test"
SAMPLE_PATH = DATA_ROOT / "sample_submission.csv"

base_original = pd.read_csv(BASE_PATH)
sample = pd.read_csv(SAMPLE_PATH)
if list(base_original.columns) != ["id", "tvt"]:
    raise RuntimeError(f"Unexpected submission columns: {list(base_original.columns)}")
if not base_original["id"].astype(str).equals(sample["id"].astype(str)):
    raise RuntimeError("Current submission IDs/order do not match sample_submission")
if not np.isfinite(base_original["tvt"]).all():
    raise RuntimeError("Current submission contains non-finite TVT")

if CANDIDATE_PATH.is_file():
    previous = pd.read_csv(CANDIDATE_PATH)
    if previous["id"].astype(str).equals(base_original["id"].astype(str)):
        if np.allclose(previous["tvt"], base_original["tvt"], rtol=0.0, atol=1e-8):
            raise RuntimeError("V35 is already present in submission.csv; do not apply it twice")

base_hash = sha256(BASE_PATH)
base_original.to_csv(BACKUP_PATH, index=False)
work = base_original.copy()
work["id"] = work["id"].astype(str)
work["well"] = work["id"].str.rsplit("_", n=1).str[0]
work["row_idx"] = work["id"].str.rsplit("_", n=1).str[-1].astype(np.int64)
work["original_order"] = np.arange(len(work), dtype=np.int64)
work = work.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)

base = work["tvt"].to_numpy(np.float64)
prediction = base.copy()
audit_rows = []
failures = []
groups = work.groupby("well", sort=False).indices
for number, (well, indices) in enumerate(groups.items(), 1):
    indices = np.asarray(indices, np.int64)
    try:
        correction, diagnostics = build_correction(
            str(well),
            work.loc[indices, "row_idx"].to_numpy(np.int64),
            base[indices],
            TEST_DIR,
        )
        candidate = base[indices] + DOSE * correction
        prediction[indices] = candidate
        audit_rows.append({
            "well": str(well), "rows": len(indices), "status": "applied",
            "mean_abs_move": float(np.mean(np.abs(candidate - base[indices]))),
            "max_abs_move": float(np.max(np.abs(candidate - base[indices]))),
            **diagnostics,
        })
    except Exception as exc:
        failures.append({"well": str(well), "reason": repr(exc)})
        audit_rows.append({
            "well": str(well), "rows": len(indices), "status": "fallback_base",
            "mean_abs_move": 0.0, "max_abs_move": 0.0,
        })
    if number % 25 == 0 or number == len(groups):
        print(f"V35 hidden SGM {number}/{len(groups)} failures={len(failures)}", flush=True)

if len(failures) > max(2, int(math.ceil(0.01 * len(groups)))):
    raise RuntimeError(f"Too many V35 hidden failures: {failures[:10]}")
if not np.isfinite(prediction).all():
    raise RuntimeError("V35 produced non-finite predictions")
if float(np.max(np.abs(prediction - base))) > DOSE * 24.0 + 1e-8:
    raise RuntimeError("V35 move exceeded the frozen OOF cap")

work["tvt"] = prediction
output = work.sort_values("original_order", kind="stable")[["id", "tvt"]].reset_index(drop=True)
if not output["id"].astype(str).equals(sample["id"].astype(str)):
    raise RuntimeError("V35 output IDs/order do not match sample_submission")
output.to_csv(CANDIDATE_PATH, index=False)
output.to_csv(BASE_PATH, index=False)

audit = pd.DataFrame(audit_rows)
audit.to_csv(WORK / "v35_tail_sgm_test_audit.csv", index=False)
pd.DataFrame(failures).to_csv(WORK / "v35_tail_sgm_test_failures.csv", index=False)
payload = {
    "experiment": "frontier_6p377_tail_sgm_dose_robustness_v35_kaggle",
    "lineage": "current submission.csv after CP8 + Neighbor",
    "data_root": str(DATA_ROOT),
    "dose": DOSE,
    "rows": int(len(output)),
    "wells": int(len(groups)),
    "successful_wells": int((audit["status"] == "applied").sum()),
    "failed_wells": int(len(failures)),
    "mean_abs_move": float(np.mean(np.abs(prediction - base))),
    "max_abs_move": float(np.max(np.abs(prediction - base))),
    "base_sha256": base_hash,
    "candidate_sha256": sha256(CANDIDATE_PATH),
    "submission_sha256": sha256(BASE_PATH),
    "elapsed_minutes": (time.time() - started) / 60.0,
    "outputs": {
        "backup": str(BACKUP_PATH),
        "candidate": str(CANDIDATE_PATH),
        "final": str(BASE_PATH),
    },
}
AUDIT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("\nV35 TAIL-SGM KAGGLE SUBMISSION COMPLETE")
print(json.dumps(payload, indent=2), flush=True)
print("final Kaggle file:", BASE_PATH, flush=True)
