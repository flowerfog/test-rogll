"""Heel-conditioned, PF-centred standalone CorrNet v4 one-fold gateway.

This is a new-base experiment.  It does not consume strong, CP12, Gold, or
Neighbor predictions.  It fixes the two structural errors in CorrNet v3:

1. the state axis is centred on the deployment-legal likelihood-PF trajectory
   at every column instead of on one constant anchor TVT;
2. the complete visible heel trajectory is retained as a condition, including
   its known TVT path and a horizontal-GR self-reference emission.

Outer-train wells use official plus four fold-safe pseudo prediction starts.
Outer-valid wells use only their untouched official start.  The first run is a
single-seed fold-1 gateway; five-fold expansion is allowed only after it beats
the pre-registered absolute and causal gates near the end of this file.
"""

from pathlib import Path
import gc
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


ROOT = Path("/root/autodl-tmp")
DATA_ROOT_CANDIDATES = (
    ROOT / "rogii-wellbore-geology-prediction",
    Path.cwd(),
)
DATA_ROOT = next(
    (path for path in DATA_ROOT_CANDIDATES if (path / "train").exists()),
    DATA_ROOT_CANDIDATES[0],
)
OOF_ROOT = ROOT / "oof_exports" if (ROOT / "oof_exports").exists() else DATA_ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / "tvt_corrnet_heel_conditioned_onefold_v4"
PANEL_DIR = OUT_DIR / "panels"
MODEL_DIR = OUT_DIR / "models"

FOLD_CANDIDATES = (
    OOF_ROOT / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
    OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1" / "well_metadata.csv",
)
BANK_DIR = OOF_ROOT / "multicut_fleongg_matched_k3_gateway_v1"
EXTENSION_DIR = OOF_ROOT / "multicut_fleongg_matched_k5_extension_v1"
FEATURE_DIR = BANK_DIR / "features"
MANIFEST_CANDIDATES = (
    BANK_DIR / "cut_manifest.csv",
    BANK_DIR / "cut_manifest_v1.csv",
)
EXTRA_MANIFEST_CANDIDATES = (
    EXTENSION_DIR / "new_cut_manifest.csv",
    EXTENSION_DIR / "cut_manifest.csv",
)

VALID_FOLD = 1
TRAIN_TAGS = ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4")
TAG_MASS = {
    "official": 0.40,
    "pseudo1": 0.15,
    "pseudo2": 0.15,
    "pseudo3": 0.15,
    "pseudo4": 0.15,
}
PF_CENTER_COLUMN = "likpf_scale_12"
PANEL_VERSION = 4

PREFIX_COLUMNS = 64
EVAL_COLUMNS = 256
N_COLUMNS = PREFIX_COLUMNS + EVAL_COLUMNS
STATE_OFFSETS = np.arange(-96.0, 96.01, 2.0, dtype=np.float32)
TARGET_SIGMA_FT = 2.0
BASE_CHANNELS = 32
BATCH_SIZE = 2
EPOCHS = 30
WARMUP_EPOCHS = 4
REPORT_EVERY = 1
EARLY_STOPPING_REPORTS = 6
LEARNING_RATE = 4e-4
MIN_LEARNING_RATE = 4e-5
WEIGHT_DECAY = 1e-4
SEED = 20260717

LOSS_CE = 1.0
LOSS_TVT = 0.35
LOSS_SLOPE = 0.04
LOSS_CURVATURE = 0.01
PREFIX_LOSS_WEIGHT = 0.15

# Input channels before the per-column geometry channels are appended.
RAW_CHANNEL_NAMES = (
    "horizontal_gr",
    "horizontal_gr_valid",
    "typewell_gr",
    "typewell_valid",
    "horizontal_x_typewell",
    "negative_abs_typewell_difference",
    "heel_reference_gr",
    "heel_reference_valid",
    "horizontal_x_heel",
    "negative_abs_heel_difference",
    "known_tvt_heatmap",
    "known_prefix_mask",
    "state_offset",
)
TYPEWELL_CHANNELS = (2, 3, 4, 5)
GR_VALUE_CHANNELS = (0, 2, 4, 5, 6, 8, 9)


def _first_existing(paths):
    return next((path for path in paths if path.exists()), None)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rmse(truth, prediction):
    truth = np.asarray(truth, np.float64)
    prediction = np.asarray(prediction, np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _robust_normalise(values):
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    output = np.zeros(values.shape, np.float32)
    if int(valid.sum()) < 4:
        return output
    centre = float(np.median(values[valid]))
    scale = float(1.4826 * np.median(np.abs(values[valid] - centre)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(max(np.std(values[valid]), 1.0))
    output[valid] = np.clip((values[valid] - centre) / scale, -8.0, 8.0)
    return output


def _interp_finite(x, y, query, left=np.nan, right=np.nan):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    query = np.asarray(query, np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return np.full(query.shape, np.nan, np.float64)
    order = np.argsort(x[valid], kind="mergesort")
    xv = x[valid][order]
    yv = y[valid][order]
    unique, inverse = np.unique(xv, return_inverse=True)
    if len(unique) != len(xv):
        sums = np.bincount(inverse, weights=yv)
        counts = np.bincount(inverse)
        yv = sums / np.maximum(counts, 1)
        xv = unique
    return np.interp(query, xv, yv, left=left, right=right)


def _nearest_distance(x, query):
    x = np.sort(np.asarray(x, np.float64)[np.isfinite(x)])
    query = np.asarray(query, np.float64)
    if not len(x):
        return np.full(query.shape, np.inf)
    position = np.searchsorted(x, query)
    left = np.abs(query - x[np.clip(position - 1, 0, len(x) - 1)])
    right = np.abs(query - x[np.clip(position, 0, len(x) - 1)])
    return np.minimum(left, right)


def _frame_row_index(frame):
    if "_row_idx" in frame.columns:
        return pd.to_numeric(frame["_row_idx"], errors="coerce").to_numpy(float)
    if "row_idx" in frame.columns:
        return pd.to_numeric(frame["row_idx"], errors="coerce").to_numpy(float)
    if "id" in frame.columns:
        return pd.to_numeric(
            frame["id"].astype(str).str.rsplit("_", n=1).str[-1],
            errors="coerce",
        ).to_numpy(float)
    raise RuntimeError("PF feature frame has no row identity")


def _feature_path(well, tag):
    if tag == "official":
        full = FEATURE_DIR / f"{well}__official_full.pkl"
        if full.exists():
            return full
    sample = FEATURE_DIR / f"{well}__{tag}__sample.pkl"
    if sample.exists():
        return sample
    raise FileNotFoundError(f"missing PF feature frame for {well}/{tag}")


def _panel_key(well, tag):
    return f"{well}__{tag}"


def _load_manifest(wells):
    primary = _first_existing(MANIFEST_CANDIDATES)
    extra = _first_existing(EXTRA_MANIFEST_CANDIDATES)
    if primary is None or extra is None:
        raise FileNotFoundError(
            "The matched K5 cut bank is required: missing primary or extension manifest"
        )
    frames = [pd.read_csv(primary, dtype={"well": str})]
    if extra != primary:
        frames.append(pd.read_csv(extra, dtype={"well": str}))
    manifest = pd.concat(frames, ignore_index=True)
    if "tag" not in manifest or "cut_position" not in manifest:
        raise RuntimeError("cut manifest misses tag/cut_position")
    manifest["well"] = manifest["well"].astype(str)
    manifest["tag"] = manifest["tag"].astype(str)
    manifest = manifest[manifest["well"].isin(set(wells))]
    manifest = manifest[manifest["tag"].isin(TRAIN_TAGS)]
    manifest = manifest.drop_duplicates(["well", "tag"], keep="last")

    metadata = []
    for well in tqdm(wells, desc="scan v4 cut metadata"):
        frame = pd.read_csv(
            DATA_ROOT / "train" / f"{well}__horizontal_well.csv",
            usecols=["TVT_input", "TVT"],
        )
        tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce").to_numpy(float)
        truth = pd.to_numeric(frame["TVT"], errors="coerce").to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        hidden = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
        if len(known) < 64 or len(hidden) < 128:
            raise RuntimeError(f"invalid official mask for {well}")
        metadata.append(
            {
                "well": well,
                "n_rows": len(frame),
                "official_cut": int(hidden[0]),
                "official_n_eval": len(hidden),
            }
        )
    metadata = pd.DataFrame(metadata)
    # The historical cut manifests may already carry these columns.  They are
    # stale bookkeeping rather than cut definitions; keeping them would make
    # pandas suffix the freshly audited values as _x/_y.
    manifest = manifest.drop(
        columns=[
            column
            for column in ("n_rows", "official_cut", "official_n_eval")
            if column in manifest.columns
        ]
    )
    manifest = manifest.merge(metadata, on="well", how="left", validate="many_to_one")
    manifest["cut_position"] = manifest["cut_position"].astype(int)
    manifest.loc[manifest["tag"] == "official", "cut_position"] = manifest.loc[
        manifest["tag"] == "official", "official_cut"
    ]
    manifest["actual_frac"] = manifest["cut_position"] / manifest["n_rows"]
    manifest["state_mass"] = manifest["tag"].map(TAG_MASS).astype(float)

    counts = manifest.groupby("well")["tag"].nunique()
    incomplete = sorted(set(wells) - set(counts[counts == len(TRAIN_TAGS)].index))
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} wells do not have all K5 states; first 20: {incomplete[:20]}"
        )
    return manifest.sort_values(["well", "tag"]).reset_index(drop=True)


def _build_panel(task):
    well = str(task["well"])
    tag = str(task["tag"])
    cut = int(task["cut_position"])
    state_mass = float(task["state_mass"])
    official_n_eval = int(task["official_n_eval"])
    key = _panel_key(well, tag)
    output = PANEL_DIR / f"{key}.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as cached:
            if (
                "panel_version" in cached.files
                and int(cached["panel_version"][0]) == PANEL_VERSION
                and int(cached["cut_position"][0]) == cut
                and str(cached["pf_center_column"][0]) == PF_CENTER_COLUMN
            ):
                return {"well": well, "tag": tag, "status": "cached", "key": key}
        output.unlink()

    try:
        horizontal = pd.read_csv(DATA_ROOT / "train" / f"{well}__horizontal_well.csv")
        typewell = pd.read_csv(DATA_ROOT / "train" / f"{well}__typewell.csv")
        required = {"MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"}
        if not required.issubset(horizontal.columns):
            raise RuntimeError(f"horizontal misses {sorted(required-set(horizontal.columns))}")
        if not {"TVT", "GR"}.issubset(typewell.columns):
            raise RuntimeError("typewell misses TVT/GR")

        md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(float)
        x = pd.to_numeric(horizontal["X"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(horizontal["Y"], errors="coerce").to_numpy(float)
        z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(float)
        gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(float)
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(float)
        truth = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        official_eval = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(truth))
        if not len(known) or not len(official_eval):
            raise RuntimeError("missing official prefix/eval")
        official_cut = int(official_eval[0])
        if tag == "official":
            if cut != official_cut:
                raise RuntimeError(f"official cut mismatch {cut} != {official_cut}")
            visible_tvt = tvt_input.copy()
            eval_rows = official_eval
        else:
            visible_tvt = np.full(len(truth), np.nan, np.float64)
            visible_tvt[:cut] = truth[:cut]
            eval_rows = np.arange(cut, len(truth), dtype=np.int32)
        anchor = cut - 1
        if anchor < 63 or len(eval_rows) < 128 or not np.isfinite(visible_tvt[anchor]):
            raise RuntimeError("invalid prediction start")
        anchor_tvt = float(visible_tvt[anchor])

        prefix_md = np.linspace(md[0], md[anchor], PREFIX_COLUMNS, endpoint=True)
        eval_md_columns = np.linspace(
            md[anchor], md[eval_rows[-1]], EVAL_COLUMNS + 1, endpoint=True
        )[1:]
        column_md = np.r_[prefix_md, eval_md_columns]
        prefix_mask = np.r_[np.ones(PREFIX_COLUMNS), np.zeros(EVAL_COLUMNS)].astype(np.float32)
        eval_mask = 1.0 - prefix_mask
        column_x = _interp_finite(md, x, column_md)
        column_y = _interp_finite(md, y, column_md)
        column_z = _interp_finite(md, z, column_md)
        truth_tvt = _interp_finite(md, truth, column_md)

        feature_frame = pd.read_pickle(_feature_path(well, tag))
        if PF_CENTER_COLUMN not in feature_frame.columns:
            raise RuntimeError(f"PF frame misses {PF_CENTER_COLUMN}")
        if "last_known_tvt" in feature_frame.columns:
            cached_anchor = pd.to_numeric(
                feature_frame["last_known_tvt"], errors="coerce"
            ).to_numpy(float)
            cached_anchor = cached_anchor[np.isfinite(cached_anchor)]
            if not len(cached_anchor) or abs(float(np.median(cached_anchor)) - anchor_tvt) > 1e-3:
                raise RuntimeError(
                    "PF feature/cut lineage mismatch: "
                    f"cached_anchor={float(np.median(cached_anchor)) if len(cached_anchor) else np.nan}, "
                    f"expected={anchor_tvt}"
                )
        feature_rows = _frame_row_index(feature_frame)
        feature_values = pd.to_numeric(
            feature_frame[PF_CENTER_COLUMN], errors="coerce"
        ).to_numpy(float)
        finite_feature = (
            np.isfinite(feature_rows)
            & np.isfinite(feature_values)
            & (feature_rows >= cut)
            & (feature_rows < len(md))
        )
        if int(finite_feature.sum()) < 32:
            raise RuntimeError("too few finite PF centre rows")
        pf_rows = feature_rows[finite_feature].astype(np.int32)
        pf_md = np.r_[md[anchor], md[pf_rows]]
        pf_values = np.r_[anchor_tvt, feature_values[finite_feature]]
        right_value = float(pf_values[int(np.nanargmax(pf_md))])
        eval_center = _interp_finite(
            pf_md, pf_values, eval_md_columns, left=anchor_tvt, right=right_value
        )
        prefix_center = _interp_finite(
            md[:cut], visible_tvt[:cut], prefix_md, left=visible_tvt[0], right=anchor_tvt
        )
        center_tvt = np.r_[prefix_center, eval_center]
        center_tvt[PREFIX_COLUMNS - 1] = anchor_tvt
        if not np.isfinite(center_tvt).all():
            raise RuntimeError("non-finite dynamic centre")
        truth_residual = truth_tvt - center_tvt

        gr_norm = _robust_normalise(gr)
        finite_gr = np.isfinite(gr)
        horizontal_gr = _interp_finite(
            md[finite_gr], gr_norm[finite_gr], column_md, left=np.nan, right=np.nan
        )
        md_step = float(np.nanmedian(np.diff(md[np.isfinite(md)])))
        horizontal_valid = (
            _nearest_distance(md[finite_gr], column_md) <= max(6.0, 2.5 * md_step)
        ).astype(np.float32)
        horizontal_gr = np.nan_to_num(horizontal_gr, nan=0.0).astype(np.float32)

        tw = typewell[["TVT", "GR"]].apply(pd.to_numeric, errors="coerce").dropna()
        tw = tw.groupby("TVT", as_index=False)["GR"].mean().sort_values("TVT")
        tw_tvt = tw["TVT"].to_numpy(float)
        tw_gr = _robust_normalise(tw["GR"].to_numpy(float))
        if len(tw_tvt) < 32:
            raise RuntimeError("typewell too short")

        candidate_tvt = center_tvt[None, :] + STATE_OFFSETS[:, None]
        typewell_gr = _interp_finite(
            tw_tvt, tw_gr, candidate_tvt.ravel(), left=np.nan, right=np.nan
        ).reshape(len(STATE_OFFSETS), N_COLUMNS)
        typewell_valid = np.isfinite(typewell_gr).astype(np.float32)
        typewell_gr = np.nan_to_num(typewell_gr, nan=0.0).astype(np.float32)

        visible_rows = np.arange(cut, dtype=np.int32)
        self_valid_rows = (
            np.isfinite(visible_tvt[:cut]) & np.isfinite(gr_norm[:cut])
        )
        self_tvt = visible_tvt[:cut][self_valid_rows]
        self_gr = gr_norm[:cut][self_valid_rows]
        if len(self_tvt) >= 16:
            heel_reference_gr = _interp_finite(
                self_tvt,
                self_gr,
                candidate_tvt.ravel(),
                left=np.nan,
                right=np.nan,
            ).reshape(len(STATE_OFFSETS), N_COLUMNS)
        else:
            heel_reference_gr = np.full(candidate_tvt.shape, np.nan, np.float64)
        heel_reference_valid = np.isfinite(heel_reference_gr).astype(np.float32)
        heel_reference_gr = np.nan_to_num(heel_reference_gr, nan=0.0).astype(np.float32)

        state_valid = (typewell_valid > 0.5) | (heel_reference_valid > 0.5)
        if np.any(state_valid.sum(axis=0) < 2):
            raise RuntimeError("some columns have fewer than two valid states")

        full_progress = (column_md - column_md[0]) / max(column_md[-1] - column_md[0], 1.0)
        md_since_anchor = (column_md - md[anchor]) / 1000.0
        eval_progress = np.clip(
            (column_md - md[anchor]) / max(column_md[-1] - md[anchor], 1.0), 0.0, 1.0
        )
        dx = np.gradient(column_x, column_md, edge_order=1)
        dy = np.gradient(column_y, column_md, edge_order=1)
        dz = np.gradient(column_z, column_md, edge_order=1)
        center_dip = np.gradient(center_tvt, column_md, edge_order=1)
        center_curvature = np.gradient(center_dip, column_md, edge_order=1) * 1000.0
        geometry = np.stack(
            [
                full_progress,
                eval_progress,
                prefix_mask,
                eval_mask,
                md_since_anchor,
                (column_x - column_x[PREFIX_COLUMNS - 1]) / 1000.0,
                (column_y - column_y[PREFIX_COLUMNS - 1]) / 1000.0,
                (column_z - column_z[PREFIX_COLUMNS - 1]) / 1000.0,
                dx,
                dy,
                dz,
                (center_tvt - anchor_tvt) / 64.0,
                center_dip,
                center_curvature,
                horizontal_valid,
                np.full(N_COLUMNS, cut / len(md)),
                np.full(N_COLUMNS, np.mean(~np.isfinite(gr))),
                np.mean(typewell_valid, axis=0),
                np.mean(heel_reference_valid, axis=0),
            ],
            axis=0,
        ).astype(np.float32)

        distance = np.abs(STATE_OFFSETS[:, None] - truth_residual[None, :])
        distance[~state_valid] = np.inf
        teacher_index = np.argmin(distance, axis=0)
        teacher_available = np.isfinite(distance[teacher_index, np.arange(N_COLUMNS)])
        teacher_residual = STATE_OFFSETS[teacher_index]
        teacher_rmse = _rmse(
            truth_residual[teacher_available], teacher_residual[teacher_available]
        )
        label_coverage = float(
            np.mean(np.abs(truth_residual[PREFIX_COLUMNS:]) <= np.max(np.abs(STATE_OFFSETS)))
        )
        true_state_valid_fraction = float(np.mean(teacher_available[PREFIX_COLUMNS:]))

        np.savez_compressed(
            output,
            panel_version=np.asarray([PANEL_VERSION], np.int16),
            pf_center_column=np.asarray([PF_CENTER_COLUMN], str),
            well=np.asarray([well], str),
            tag=np.asarray([tag], str),
            cut_position=np.asarray([cut], np.int32),
            state_offsets=STATE_OFFSETS,
            horizontal_gr=horizontal_gr.astype(np.float16),
            horizontal_valid=horizontal_valid.astype(np.float16),
            typewell_gr=typewell_gr.astype(np.float16),
            typewell_valid=typewell_valid.astype(np.float16),
            heel_reference_gr=heel_reference_gr.astype(np.float16),
            heel_reference_valid=heel_reference_valid.astype(np.float16),
            state_valid=state_valid.astype(np.uint8),
            geometry=geometry.astype(np.float16),
            prefix_mask=prefix_mask.astype(np.float16),
            eval_mask=eval_mask.astype(np.float16),
            center_tvt=center_tvt.astype(np.float32),
            truth_residual=truth_residual.astype(np.float32),
            column_md=column_md.astype(np.float32),
            column_z=column_z.astype(np.float32),
            eval_rows=eval_rows.astype(np.int32),
            eval_md=md[eval_rows].astype(np.float32),
            eval_z=z[eval_rows].astype(np.float32),
            eval_truth=truth[eval_rows].astype(np.float32),
            panel_weight=np.asarray([state_mass * max(official_n_eval, 1)], np.float32),
            center_eval_rmse=np.asarray(
                [_rmse(truth_tvt[PREFIX_COLUMNS:], center_tvt[PREFIX_COLUMNS:])], np.float32
            ),
            teacher_rmse=np.asarray([teacher_rmse], np.float32),
            label_coverage=np.asarray([label_coverage], np.float32),
            true_state_valid_fraction=np.asarray([true_state_valid_fraction], np.float32),
        )
        return {
            "well": well,
            "tag": tag,
            "key": key,
            "status": "built",
            "center_eval_rmse": _rmse(
                truth_tvt[PREFIX_COLUMNS:], center_tvt[PREFIX_COLUMNS:]
            ),
            "label_coverage": label_coverage,
            "teacher_rmse": teacher_rmse,
        }
    except Exception as exc:
        return {
            "well": well,
            "tag": tag,
            "key": key,
            "status": "failed",
            "reason": repr(exc),
        }


class PanelStore:
    def __init__(self, keys):
        self.records = []
        for key in tqdm(keys, desc="load v4 panels into RAM"):
            with np.load(PANEL_DIR / f"{key}.npz", allow_pickle=False) as data:
                self.records.append(
                    {
                        "well": str(data["well"][0]),
                        "tag": str(data["tag"][0]),
                        "horizontal_gr": data["horizontal_gr"].astype(np.float32),
                        "horizontal_valid": data["horizontal_valid"].astype(np.float32),
                        "typewell_gr": data["typewell_gr"].astype(np.float32),
                        "typewell_valid": data["typewell_valid"].astype(np.float32),
                        "heel_reference_gr": data["heel_reference_gr"].astype(np.float32),
                        "heel_reference_valid": data["heel_reference_valid"].astype(np.float32),
                        "state_valid": data["state_valid"].astype(bool),
                        "geometry": data["geometry"].astype(np.float32),
                        "prefix_mask": data["prefix_mask"].astype(np.float32),
                        "eval_mask": data["eval_mask"].astype(np.float32),
                        "center_tvt": data["center_tvt"].astype(np.float32),
                        "truth_residual": data["truth_residual"].astype(np.float32),
                        "column_md": data["column_md"].astype(np.float32),
                        "column_z": data["column_z"].astype(np.float32),
                        "eval_rows": data["eval_rows"].astype(np.int32),
                        "eval_md": data["eval_md"].astype(np.float32),
                        "eval_z": data["eval_z"].astype(np.float32),
                        "eval_truth": data["eval_truth"].astype(np.float32),
                        "panel_weight": float(data["panel_weight"][0]),
                    }
                )

    def __len__(self):
        return len(self.records)


def _donor_index(index, length):
    return index if length < 2 else (index + max(1, length // 3)) % length


def _compose_input(store, index, mode):
    record = store.records[index]
    horizontal = record["horizontal_gr"]
    horizontal_valid = record["horizontal_valid"]
    typewell = record["typewell_gr"]
    typewell_valid = record["typewell_valid"]
    heel = record["heel_reference_gr"]
    heel_valid = record["heel_reference_valid"]
    if mode == "shuffled_typewell":
        donor = store.records[_donor_index(index, len(store))]
        typewell = donor["typewell_gr"]
        typewell_valid = donor["typewell_valid"]
    elif mode not in ("full", "no_gr"):
        raise ValueError(mode)

    states = len(STATE_OFFSETS)
    h = np.broadcast_to(horizontal[None, :], (states, N_COLUMNS))
    h_valid = np.broadcast_to(horizontal_valid[None, :], (states, N_COLUMNS))
    known_mask = np.broadcast_to(record["prefix_mask"][None, :], (states, N_COLUMNS))
    known_heat = (
        np.exp(-0.5 * (STATE_OFFSETS / TARGET_SIGMA_FT) ** 2)[:, None] * known_mask
    )
    offset = np.broadcast_to(
        (STATE_OFFSETS / np.max(np.abs(STATE_OFFSETS)))[:, None],
        (states, N_COLUMNS),
    )
    raw = np.stack(
        [
            h,
            h_valid,
            typewell,
            typewell_valid,
            h * typewell,
            -np.abs(h - typewell),
            heel,
            heel_valid,
            h * heel,
            -np.abs(h - heel),
            known_heat,
            known_mask,
            offset,
        ],
        axis=0,
    ).astype(np.float32)
    if mode == "no_gr":
        raw[list(GR_VALUE_CHANNELS)] = 0.0
    geometry = np.broadcast_to(
        record["geometry"][:, None, :],
        (record["geometry"].shape[0], states, N_COLUMNS),
    )
    return np.concatenate([raw, geometry], axis=0).astype(np.float32)


class PanelDataset(Dataset):
    def __init__(self, store, mode="full"):
        self.store = store
        self.mode = mode

    def __len__(self):
        return len(self.store)

    def __getitem__(self, index):
        record = self.store.records[index]
        return {
            "input": torch.from_numpy(_compose_input(self.store, index, self.mode)),
            "state_valid": torch.from_numpy(record["state_valid"]),
            "truth_residual": torch.from_numpy(record["truth_residual"]),
            "center_tvt": torch.from_numpy(record["center_tvt"]),
            "column_md": torch.from_numpy(record["column_md"]),
            "column_z": torch.from_numpy(record["column_z"]),
            "eval_mask": torch.from_numpy(record["eval_mask"]),
            "panel_weight": torch.tensor(record["panel_weight"], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.project = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
        )

    def forward(self, values):
        return F.gelu(self.block(values) + self.project(values))


class HeelConditionedCorrNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        base = BASE_CHANNELS
        self.enc1 = ResidualBlock(input_channels, base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = ResidualBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base * 4, base * 4),
            ResidualBlock(base * 4, base * 4),
        )
        self.dec2 = ResidualBlock(base * 4 + base * 2, base * 2)
        self.dec1 = ResidualBlock(base * 2 + base, base)
        self.head = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.log_prior_sigma = nn.Parameter(torch.tensor(np.log(16.0), dtype=torch.float32))

    def forward(self, values, state_offsets):
        enc1 = self.enc1(values)
        enc2 = self.enc2(self.down1(enc1))
        bottleneck = self.bottleneck(self.down2(enc2))
        up2 = F.interpolate(
            bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False
        )
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        residual_logits = self.head(dec1).squeeze(1)
        sigma = torch.exp(self.log_prior_sigma).clamp(4.0, 64.0)
        prior = -0.5 * (state_offsets[:, None] / sigma) ** 2
        return residual_logits + prior[None]


def _weighted_panel_mean(values, panel_weight):
    panel_weight = panel_weight.float().clamp_min(1e-6)
    return (values * panel_weight).sum() / panel_weight.sum()


def _loss(
    logits,
    state_valid,
    truth_residual,
    center_tvt,
    column_md,
    column_z,
    eval_mask,
    panel_weight,
    state_offsets,
    ce_only,
):
    logits = logits.float().masked_fill(~state_valid[:, :, :], -30.0)
    distance = state_offsets[None, :, None] - truth_residual[:, None, :]
    target = torch.softmax(-0.5 * (distance / TARGET_SIGMA_FT) ** 2, dim=1)
    target = target * state_valid
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    probability = torch.softmax(logits, dim=1)
    prediction = (probability * state_offsets[None, :, None]).sum(dim=1)

    column_weight = PREFIX_LOSS_WEIGHT + (1.0 - PREFIX_LOSS_WEIGHT) * eval_mask
    first_eval = torch.argmax(eval_mask, dim=1)
    column_weight[torch.arange(len(column_weight), device=column_weight.device), first_eval] = 4.0
    denominator = column_weight.sum(dim=1).clamp_min(1e-6)

    ce_columns = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    ce = _weighted_panel_mean((ce_columns * column_weight).sum(dim=1) / denominator, panel_weight)
    tvt_columns = F.smooth_l1_loss(
        prediction, truth_residual, beta=2.0, reduction="none"
    )
    tvt = _weighted_panel_mean((tvt_columns * column_weight).sum(dim=1) / denominator, panel_weight)

    predicted_s = center_tvt + prediction + column_z
    true_s = center_tvt + truth_residual + column_z
    dmd = (column_md[:, 1:] - column_md[:, :-1]).clamp_min(1e-3)
    predicted_dip = (predicted_s[:, 1:] - predicted_s[:, :-1]) / dmd
    true_dip = (true_s[:, 1:] - true_s[:, :-1]) / dmd
    transition_weight = torch.maximum(eval_mask[:, 1:], eval_mask[:, :-1])
    transition_denominator = transition_weight.sum(dim=1).clamp_min(1e-6)
    slope_columns = F.smooth_l1_loss(
        predicted_dip, true_dip, beta=0.01, reduction="none"
    )
    slope = _weighted_panel_mean(
        (slope_columns * transition_weight).sum(dim=1) / transition_denominator,
        panel_weight,
    )
    curvature_columns = F.smooth_l1_loss(
        predicted_dip[:, 1:] - predicted_dip[:, :-1],
        true_dip[:, 1:] - true_dip[:, :-1],
        beta=0.005,
        reduction="none",
    )
    curvature_weight = torch.maximum(transition_weight[:, 1:], transition_weight[:, :-1])
    curvature = _weighted_panel_mean(
        (curvature_columns * curvature_weight).sum(dim=1)
        / curvature_weight.sum(dim=1).clamp_min(1e-6),
        panel_weight,
    )
    total = ce if ce_only else (
        LOSS_CE * ce
        + LOSS_TVT * tvt
        + LOSS_SLOPE * slope
        + LOSS_CURVATURE * curvature
    )
    return total, {
        "ce": ce,
        "tvt": tvt,
        "slope": slope,
        "curvature": curvature,
    }


@torch.no_grad()
def _predict_record(model, store, index, mode, device, state_offsets):
    record = store.records[index]
    values = torch.from_numpy(_compose_input(store, index, mode))[None].to(device)
    state_valid = torch.from_numpy(record["state_valid"])[None].to(device)
    logits = model(values, state_offsets).float().masked_fill(~state_valid, -30.0)
    probability = torch.softmax(logits, dim=1)
    correction = (
        probability * state_offsets[None, :, None]
    ).sum(dim=1)[0].cpu().numpy()
    correction[:PREFIX_COLUMNS] = 0.0
    column_s = record["center_tvt"] + correction + record["column_z"]
    eval_s = np.interp(record["eval_md"], record["column_md"], column_s)
    prediction = eval_s - record["eval_z"]
    center_s = record["center_tvt"] + record["column_z"]
    center_prediction = (
        np.interp(record["eval_md"], record["column_md"], center_s) - record["eval_z"]
    )
    return prediction, center_prediction


@torch.no_grad()
def _evaluate(model, store, mode, device, state_offsets, return_rows=False):
    errors = []
    center_errors = []
    by_well = []
    row_frames = []
    for index, record in enumerate(store.records):
        prediction, center_prediction = _predict_record(
            model, store, index, mode, device, state_offsets
        )
        truth = record["eval_truth"].astype(float)
        error = prediction - truth
        center_error = center_prediction - truth
        errors.append(error)
        center_errors.append(center_error)
        by_well.append(
            {
                "well": record["well"],
                "rows": len(error),
                "center_rmse": float(np.sqrt(np.mean(center_error**2))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
            }
        )
        if return_rows:
            row_frames.append(
                pd.DataFrame(
                    {
                        "well": record["well"],
                        "row_idx": record["eval_rows"],
                        "truth": truth,
                        "center": center_prediction,
                        "prediction": prediction,
                    }
                )
            )
    errors = np.concatenate(errors)
    center_errors = np.concatenate(center_errors)
    by_well = pd.DataFrame(by_well)
    metrics = {
        "center_rmse": float(np.sqrt(np.mean(center_errors**2))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "gain_vs_center": float(
            np.sqrt(np.mean(center_errors**2)) - np.sqrt(np.mean(errors**2))
        ),
        "mae": float(np.mean(np.abs(errors))),
        "median_well_rmse": float(by_well["rmse"].median()),
        "p95_well_rmse": float(by_well["rmse"].quantile(0.95)),
    }
    rows = pd.concat(row_frames, ignore_index=True) if return_rows else None
    return metrics, by_well, rows


def _train(train_store, valid_store, input_channels, device, state_offsets):
    _seed_everything(SEED)
    model = HeelConditionedCorrNet(input_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE
    )
    dataset = PanelDataset(train_store, "full")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    checkpoint_path = MODEL_DIR / f"seed_{SEED}.pt"
    history = []
    best_rmse = np.inf
    stale_reports = 0
    started = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_rows = []
        ce_only = epoch <= WARMUP_EPOCHS
        for batch in loader:
            values = batch["input"].to(device, non_blocking=True)
            state_valid = batch["state_valid"].to(device, non_blocking=True)
            truth_residual = batch["truth_residual"].to(device, non_blocking=True)
            center_tvt = batch["center_tvt"].to(device, non_blocking=True)
            column_md = batch["column_md"].to(device, non_blocking=True)
            column_z = batch["column_z"].to(device, non_blocking=True)
            eval_mask = batch["eval_mask"].to(device, non_blocking=True)
            panel_weight = batch["panel_weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(values, state_offsets)
                loss, components = _loss(
                    logits,
                    state_valid,
                    truth_residual,
                    center_tvt,
                    column_md,
                    column_z,
                    eval_mask,
                    panel_weight,
                    state_offsets,
                    ce_only,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    **{name: float(value.detach().cpu()) for name, value in components.items()},
                }
            )
        scheduler.step()

        if epoch % REPORT_EVERY and epoch != EPOCHS:
            continue
        model.eval()
        metrics, by_well, _ = _evaluate(
            model, valid_store, "full", device, state_offsets
        )
        record = {
            "epoch": epoch,
            "phase": "ce_only" if ce_only else "joint",
            **{
                f"train_{name}": float(np.mean([row[name] for row in loss_rows]))
                for name in loss_rows[0]
            },
            **{f"valid_{name}": value for name, value in metrics.items()},
            "prior_sigma": float(torch.exp(model.log_prior_sigma).detach().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        history.append(record)
        print("epoch", record, flush=True)
        if metrics["rmse"] < best_rmse - 1e-4:
            best_rmse = metrics["rmse"]
            stale_reports = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "input_channels": input_channels,
                    "seed": SEED,
                },
                checkpoint_path,
            )
            by_well.to_csv(OUT_DIR / "best_bywell.csv", index=False)
        elif epoch > WARMUP_EPOCHS:
            stale_reports += 1
        if epoch >= WARMUP_EPOCHS + 5 and stale_reports >= EARLY_STOPPING_REPORTS:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics, by_well, rows = _evaluate(
        model, valid_store, "full", device, state_offsets, return_rows=True
    )
    by_well.to_csv(OUT_DIR / "final_bywell.csv", index=False)
    rows.to_pickle(OUT_DIR / "fold1_predictions.pkl")
    result = {
        "best_epoch": int(checkpoint["epoch"]),
        **metrics,
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    del optimizer, scheduler, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, result


for directory in (OUT_DIR, PANEL_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

fold_path = _first_existing(FOLD_CANDIDATES)
if fold_path is None:
    raise FileNotFoundError("fold metadata not found")
if not FEATURE_DIR.exists():
    raise FileNotFoundError(FEATURE_DIR)
fold = pd.read_csv(fold_path, dtype={"well": str})
fold = fold.groupby("well", as_index=False)["fold"].first()
fold["well"] = fold["well"].astype(str)
fold["fold"] = fold["fold"].astype(int)
wells = fold["well"].tolist()
train_wells = fold.loc[fold["fold"] != VALID_FOLD, "well"].tolist()
valid_wells = fold.loc[fold["fold"] == VALID_FOLD, "well"].tolist()
manifest = _load_manifest(wells)
manifest.to_csv(OUT_DIR / "cut_manifest.csv", index=False)

train_set = set(train_wells)
valid_set = set(valid_wells)
train_tasks = manifest[manifest["well"].isin(train_set)].copy()
valid_tasks = manifest[
    manifest["well"].isin(valid_set) & (manifest["tag"] == "official")
].copy()
build_tasks = pd.concat([train_tasks, valid_tasks], ignore_index=True).to_dict("records")
missing_feature_states = []
for task in build_tasks:
    try:
        _feature_path(str(task["well"]), str(task["tag"]))
    except FileNotFoundError:
        missing_feature_states.append((str(task["well"]), str(task["tag"])))
if missing_feature_states:
    raise FileNotFoundError(
        f"Missing {len(missing_feature_states)} PF feature states; "
        f"first 20: {missing_feature_states[:20]}"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")
print(
    {
        "experiment": "tvt_corrnet_heel_conditioned_onefold_v4",
        "device": str(device),
        "fold": VALID_FOLD,
        "train_wells": len(train_wells),
        "train_panels": len(train_tasks),
        "valid_wells": len(valid_wells),
        "valid_panels": len(valid_tasks),
        "state_range_ft": [float(STATE_OFFSETS[0]), float(STATE_OFFSETS[-1])],
        "states": len(STATE_OFFSETS),
        "columns": N_COLUMNS,
        "pf_center": PF_CENTER_COLUMN,
        "standalone": True,
    },
    flush=True,
)
print("CUT DISTRIBUTION")
print(
    manifest.groupby("tag")["actual_frac"].describe()[
        ["count", "mean", "min", "25%", "50%", "75%", "max"]
    ].to_string()
)

build_started = time.time()
build_report = pd.DataFrame(
    [_build_panel(task) for task in tqdm(build_tasks, desc="build heel-conditioned panels")]
)
build_report.to_csv(OUT_DIR / "panel_build_report.csv", index=False)
failed = build_report[build_report["status"] == "failed"]
if len(failed):
    print(failed.head(30).to_string(index=False))
    raise RuntimeError(f"failed to build {len(failed)} panels")
panel_build_minutes = (time.time() - build_started) / 60.0

guard_rows = []
for task in build_tasks:
    key = _panel_key(task["well"], task["tag"])
    with np.load(PANEL_DIR / f"{key}.npz", allow_pickle=False) as data:
        guard_rows.append(
            {
                "well": str(task["well"]),
                "tag": str(task["tag"]),
                "center_eval_rmse": float(data["center_eval_rmse"][0]),
                "teacher_rmse": float(data["teacher_rmse"][0]),
                "label_coverage": float(data["label_coverage"][0]),
                "true_state_valid_fraction": float(data["true_state_valid_fraction"][0]),
            }
        )
guard_frame = pd.DataFrame(guard_rows)
guard_frame["key"] = guard_frame.apply(
    lambda row: _panel_key(str(row["well"]), str(row["tag"])), axis=1
)
guard_frame["usable"] = (
    (guard_frame["label_coverage"] >= 0.95)
    & (guard_frame["true_state_valid_fraction"] >= 0.99)
    & (guard_frame["teacher_rmse"] <= 2.0)
)
guard_frame.to_csv(OUT_DIR / "panel_guards.csv", index=False)
official_bad = guard_frame[(guard_frame["tag"] == "official") & ~guard_frame["usable"]]
if len(official_bad):
    print("BAD OFFICIAL PANELS")
    print(official_bad.sort_values("label_coverage").head(20).to_string(index=False))
    raise RuntimeError(f"{len(official_bad)} official panels fail the v4 state-axis guard")

train_tasks["key"] = train_tasks.apply(
    lambda row: _panel_key(str(row["well"]), str(row["tag"])), axis=1
)
valid_tasks["key"] = valid_tasks.apply(
    lambda row: _panel_key(str(row["well"]), str(row["tag"])), axis=1
)
usable_keys = set(guard_frame.loc[guard_frame["usable"], "key"])
before = len(train_tasks)
train_tasks = train_tasks[train_tasks["key"].isin(usable_keys)].copy()
dropped = before - len(train_tasks)
if dropped / max(before, 1) > 0.10:
    raise RuntimeError(f"too many pseudo panels rejected: {dropped}/{before}")

guard_summary = guard_frame.groupby("tag", as_index=False).agg(
    panels=("well", "size"),
    usable=("usable", "sum"),
    center_rmse=("center_eval_rmse", "mean"),
    label_coverage=("label_coverage", "mean"),
    min_label_coverage=("label_coverage", "min"),
    teacher_rmse=("teacher_rmse", "mean"),
)
guard_summary.to_csv(OUT_DIR / "panel_guard_by_tag.csv", index=False)
print("\nPANEL GUARDS")
print(guard_summary.to_string(index=False))
print("retained train panels:", {"before": before, "after": len(train_tasks), "dropped": dropped})

train_keys = train_tasks["key"].astype(str).tolist()
valid_keys = valid_tasks["key"].astype(str).tolist()
train_store = PanelStore(train_keys)
valid_store = PanelStore(valid_keys)
retained_mass = train_tasks.groupby("well")["state_mass"].sum().to_dict()
for record in train_store.records:
    record["panel_weight"] /= max(float(retained_mass[record["well"]]), 1e-6)

sample_input = _compose_input(train_store, 0, "full")
input_channels = int(sample_input.shape[0])
state_offsets_tensor = torch.from_numpy(STATE_OFFSETS).to(device)
print(
    "MODEL INPUT",
    {
        "channels": input_channels,
        "raw_channels": len(RAW_CHANNEL_NAMES),
        "geometry_channels": train_store.records[0]["geometry"].shape[0],
        "panel_shape": sample_input.shape,
    },
)

model, result = _train(
    train_store, valid_store, input_channels, device, state_offsets_tensor
)
intervention_rows = []
for mode in ("full", "no_gr", "shuffled_typewell"):
    metrics, _, rows = _evaluate(
        model, valid_store, mode, device, state_offsets_tensor, return_rows=True
    )
    rows.to_pickle(OUT_DIR / f"intervention_{mode}.pkl")
    intervention_rows.append({"mode": mode, **metrics})
interventions = pd.DataFrame(intervention_rows)
full_rmse = float(interventions.loc[interventions["mode"] == "full", "rmse"].iloc[0])
interventions["degradation_vs_full"] = interventions["rmse"] - full_rmse
interventions.to_csv(OUT_DIR / "interventions.csv", index=False)

no_gr_degradation = float(
    interventions.loc[interventions["mode"] == "no_gr", "degradation_vs_full"].iloc[0]
)
shuffled_degradation = float(
    interventions.loc[
        interventions["mode"] == "shuffled_typewell", "degradation_vs_full"
    ].iloc[0]
)
architecture_pass = bool(result["rmse"] < 8.5 and result["gain_vs_center"] >= 2.0)
causal_pass = bool(no_gr_degradation >= 0.30 and shuffled_degradation >= 0.20)
strict_pass = bool(architecture_pass and causal_pass)
verdict = {
    "experiment": "tvt_corrnet_heel_conditioned_onefold_v4",
    "standalone": True,
    "fold": VALID_FOLD,
    "fold_source": str(fold_path),
    "train_wells": len(train_wells),
    "train_panels": len(train_tasks),
    "valid_wells": len(valid_wells),
    "state_half_width_ft": float(np.max(np.abs(STATE_OFFSETS))),
    "states": len(STATE_OFFSETS),
    "columns": N_COLUMNS,
    "input_channels": input_channels,
    "pf_center": PF_CENTER_COLUMN,
    "result": result,
    "no_gr_degradation": no_gr_degradation,
    "shuffled_typewell_degradation": shuffled_degradation,
    "panel_build_minutes": panel_build_minutes,
    "dropped_pseudo_panels": dropped,
    "architecture_pass": architecture_pass,
    "causal_pass": causal_pass,
    "strict_pass": strict_pass,
    "next": (
        "PASS -> add two seeds, then five outer folds; standalone target <=5.8 ft"
        if strict_pass
        else "FAIL -> stop this v4 architecture before multi-seed/five-fold expansion"
    ),
}
with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)

print("\nRESULT")
print(json.dumps(result, indent=2))
print("\nINTERVENTIONS")
print(interventions.to_string(index=False))
print("\nFINAL VERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", OUT_DIR)
