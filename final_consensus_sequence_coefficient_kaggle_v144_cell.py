"""Apply the legal all-773 V144 coefficient model after V116 on Kaggle."""

from pathlib import Path
import hashlib
import json
import math
import os
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn


WORK_V144K = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
INPUT_V144K = Path("/kaggle/input")
PARENT_V144K = WORK_V144K / "submission.csv"
PARENT_COPY_V144K = WORK_V144K / "submission_v144_v116_parent.csv"
OUTPUT_V144K = WORK_V144K / "submission.csv"
NAMED_V144K = WORK_V144K / "submission_v144_sequence_coefficient.csv"

required_live_v144k = (
    "sample", "sample_value_column", "data_root", "well_rows", "distance",
    "parent_values", "v77_coarse", "v77_output_values", "final", "_load_test_record",
)
missing_live_v144k = [name for name in required_live_v144k if name not in globals()]
if missing_live_v144k:
    raise RuntimeError("Run V69, V77 and V116 first. Missing: " + repr(missing_live_v144k))


def _sha256_v144k(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_bundle_v144k():
    configured = os.environ.get("V144_BUNDLE_DIR")
    candidates = [Path(configured)] if configured else []
    if INPUT_V144K.is_dir():
        candidates.extend(path.parent for path in INPUT_V144K.rglob("manifest.json"))
    for directory in candidates:
        path = directory / "manifest.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get(
            "experiment"
        ) == "final_consensus_sequence_coefficient_bundle_v144":
            return directory, payload
    # Kaggle can mount an uploaded archive as a file instead of expanding it.
    # Accept the exact AutoDL export without requiring a separate dataset build step.
    if INPUT_V144K.is_dir():
        for archive in INPUT_V144K.rglob("final_consensus_sequence_coefficient_bundle_v144.zip"):
            extracted = WORK_V144K / "v144_bundle_extracted"
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(extracted)
            for manifest_path in extracted.rglob("manifest.json"):
                try:
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and payload.get(
                    "experiment"
                ) == "final_consensus_sequence_coefficient_bundle_v144":
                    return manifest_path.parent, payload
    raise FileNotFoundError(
        "Cannot locate the V144 deployment bundle. Upload the AutoDL export "
        "final_consensus_sequence_coefficient_bundle_v144.zip as a Kaggle dataset."
    )


def _smoothstep_v144k(value):
    value = np.clip(np.asarray(value, np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _fill_v144k(values, coordinate):
    values = np.asarray(values, np.float64)
    coordinate = np.asarray(coordinate, np.float64)
    finite = np.isfinite(values) & np.isfinite(coordinate)
    if finite.sum() < 2:
        raise RuntimeError("Insufficient finite sequence values")
    output = values.copy()
    output[~finite] = np.interp(coordinate[~finite], coordinate[finite], values[finite])
    return output


def _robust_v144k(values):
    values = np.asarray(values, np.float64)
    centre = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - centre)))
    if not np.isfinite(scale) or scale < 1e-4:
        scale = float(np.std(values))
    return (values - centre) / max(scale, 1e-3)


def _interp_v144k(progress, values, grid):
    return np.interp(grid, progress, np.asarray(values, np.float64))


def _features_v144k(well, record, positions, base, exact, parent, coarse, nodes):
    eval_rows = record["eval_rows"]
    if len(positions) != len(eval_rows):
        raise RuntimeError(f"{well}: row mismatch {len(positions)}/{len(eval_rows)}")
    full_md = record["md"].astype(np.float64)
    md = full_md[eval_rows]
    local_distance = md - float(record["anchor_md"])
    progress = local_distance / max(float(local_distance[-1]), 1.0)
    grid = np.linspace(0.0, 1.0, int(nodes))
    gr = _fill_v144k(record["gr"], full_md)[eval_rows]
    z = _fill_v144k(record["z"], full_md)[eval_rows]
    type_tvt = np.asarray(record["type_tvt"], np.float64)
    type_gr = np.asarray(record["type_gr"], np.float64)
    finite = np.isfinite(type_tvt) & np.isfinite(type_gr)
    type_frame = pd.DataFrame({"tvt": type_tvt[finite], "gr": type_gr[finite]})
    type_frame = type_frame.groupby("tvt", as_index=False, sort=True)["gr"].mean()
    if len(type_frame) < 16:
        raise RuntimeError(f"{well}: insufficient typewell samples")
    type_tvt = type_frame["tvt"].to_numpy(float)
    type_gr = type_frame["gr"].to_numpy(float)
    mismatch = [gr - np.interp(base + offset, type_tvt, type_gr) for offset in (-24, -12, 0, 12, 24)]
    spacing = max(float(np.median(np.diff(md))), 1e-3)
    channels = [
        grid,
        np.log1p(grid * max(float(local_distance[-1]), 1.0)) / math.log1p(max(float(local_distance[-1]), 1.0)),
        _interp_v144k(progress, base - base[0], grid) / 32.0,
        _interp_v144k(progress, exact - base, grid) / 8.0,
        _interp_v144k(progress, parent - base, grid) / 16.0,
        _interp_v144k(progress, coarse - base, grid) / 16.0,
        _interp_v144k(progress, base - exact, grid) / 8.0,
        _interp_v144k(progress, coarse - parent, grid) / 16.0,
        _interp_v144k(progress, _robust_v144k(gr), grid),
        _interp_v144k(progress, np.gradient(_robust_v144k(gr), spacing), grid),
        _interp_v144k(progress, (z - z[0]) / 100.0, grid),
        _interp_v144k(progress, np.gradient(z, md), grid),
        _interp_v144k(progress, np.gradient(base, md), grid),
        _interp_v144k(progress, np.gradient(np.gradient(base, md), md) * 1000.0, grid),
    ]
    channels.extend(
        _interp_v144k(progress, values / max(float(record["noise_scale"]), 5.0), grid)
        for values in mismatch
    )
    known = np.flatnonzero(np.isfinite(record["tvt_input"]) & np.isfinite(record["gr"]))
    prefix = known[record["md"][known] >= float(record["anchor_md"]) - 500.0]
    if len(prefix) < 2:
        raise RuntimeError(f"{well}: visible prefix unavailable")
    prefix_gr = record["gr"][prefix].astype(np.float64)
    metadata = np.asarray([
        float(local_distance[-1]) / 5000.0,
        spacing / 2.0,
        float(record["noise_scale"]) / 20.0,
        float(record["anchor_surface"]) / 15000.0,
        float(record["tvt_input"][record["anchor_index"]]) / 15000.0,
        float(record["z"][record["anchor_index"]]) / 15000.0,
        float(np.nanmean(prefix_gr)) / 100.0,
        float(np.nanstd(prefix_gr)) / 50.0,
        float(base[0]) / 15000.0,
        float(base[-1] - base[0]) / 100.0,
        float(np.std(base - exact)) / 10.0,
        float(np.std(coarse - parent)) / 20.0,
    ], np.float32)
    sequence = np.asarray(channels, np.float32)
    if not np.isfinite(sequence).all() or not np.isfinite(metadata).all():
        raise RuntimeError(f"{well}: non-finite features")
    return sequence, metadata, local_distance


class _ResidualBlockV144K(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(4, width), nn.SiLU(), nn.Dropout(0.10),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, width),
        )
        self.activation = nn.SiLU()

    def forward(self, value):
        return self.activation(value + self.net(value))


class _SequenceCoefficientEncoderV144K(nn.Module):
    def __init__(self, channels, metadata_features, outputs=6, width=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(channels, width, 5, padding=2), nn.SiLU())
        self.blocks = nn.Sequential(*(_ResidualBlockV144K(width, d) for d in (1, 2, 4, 8)))
        self.attention = nn.Conv1d(width, 1, 1)
        self.metadata = nn.Sequential(nn.Linear(metadata_features, 24), nn.LayerNorm(24), nn.SiLU(), nn.Dropout(0.10))
        self.head = nn.Sequential(
            nn.Linear(width * 3 + 24, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(0.15), nn.Linear(96, outputs)
        )

    def forward(self, sequence, metadata):
        encoded = self.blocks(self.stem(sequence))
        attention = torch.softmax(self.attention(encoded), dim=-1)
        return self.head(torch.cat([
            torch.sum(attention * encoded, dim=-1),
            torch.mean(encoded, dim=-1),
            torch.amax(encoded, dim=-1),
            self.metadata(metadata),
        ], dim=1))


bundle_dir_v144k, manifest_v144k = _find_bundle_v144k()
checkpoint_path_v144k = bundle_dir_v144k / manifest_v144k["checkpoint"]
if _sha256_v144k(checkpoint_path_v144k) != manifest_v144k["checkpoint_sha256"]:
    raise RuntimeError("V144 checkpoint hash mismatch")
checkpoint_v144k = torch.load(checkpoint_path_v144k, map_location="cpu", weights_only=False)
model_v144k = _SequenceCoefficientEncoderV144K(
    checkpoint_v144k["sequence_channels"], checkpoint_v144k["metadata_features"]
)
model_v144k.load_state_dict(checkpoint_v144k["state_dict"])
model_v144k.eval()

parent_frame_v144k = pd.read_csv(PARENT_V144K)
summary_path_v144k = WORK_V144K / "v116_consensus_center_summary.json"
if not summary_path_v144k.is_file() or _sha256_v144k(PARENT_V144K) != json.loads(summary_path_v144k.read_text())["sha256"]:
    raise RuntimeError("Current submission.csv is not the audited V116 parent")
parent_frame_v144k.to_csv(PARENT_COPY_V144K, index=False)

sequence_rows_v144k, metadata_rows_v144k, records_v144k = [], [], []
for number_v144k, (well_v144k, positions_v144k) in enumerate(sorted(well_rows.items()), 1):
    positions_v144k = np.asarray(positions_v144k, np.int64)
    record_v144k = _load_test_record(Path(data_root), well_v144k, 1, V["_interp_nan"])
    sequence_v144k, metadata_v144k, local_distance_v144k = _features_v144k(
        well_v144k, record_v144k, positions_v144k,
        np.asarray(final, np.float64)[positions_v144k],
        np.asarray(v77_output_values, np.float64)[positions_v144k],
        np.asarray(parent_values, np.float64)[positions_v144k],
        np.asarray(v77_coarse, np.float64)[positions_v144k],
        checkpoint_v144k["nodes"],
    )
    sequence_rows_v144k.append(sequence_v144k)
    metadata_rows_v144k.append(metadata_v144k)
    records_v144k.append((str(well_v144k), positions_v144k, local_distance_v144k))
    if number_v144k % 25 == 0 or number_v144k == len(well_rows):
        print(f"V144 features {number_v144k}/{len(well_rows)}", flush=True)

sequence_v144k = np.stack(sequence_rows_v144k).astype(np.float32)
metadata_v144k = np.stack(metadata_rows_v144k).astype(np.float32)
sequence_v144k = (sequence_v144k - checkpoint_v144k["sequence_mean"]) / checkpoint_v144k["sequence_std"]
metadata_v144k = (metadata_v144k - checkpoint_v144k["metadata_mean"]) / checkpoint_v144k["metadata_std"]
with torch.no_grad():
    coefficient_v144k = model_v144k(
        torch.from_numpy(sequence_v144k.astype(np.float32)),
        torch.from_numpy(metadata_v144k.astype(np.float32)),
    ).numpy()
coefficient_v144k = coefficient_v144k * checkpoint_v144k["target_std"] + checkpoint_v144k["target_mean"]
coefficient_v144k = np.clip(coefficient_v144k, checkpoint_v144k["target_low"], checkpoint_v144k["target_high"])

base_values_v144k = np.asarray(final, np.float64)
output_values_v144k = base_values_v144k.copy()
reports_v144k = []
for index_v144k, (well_v144k, positions_v144k, local_distance_v144k) in enumerate(records_v144k):
    taper_v144k = _smoothstep_v144k((local_distance_v144k - manifest_v144k["freeze_ft"]) / manifest_v144k["ramp_ft"])
    x_v144k = (local_distance_v144k - np.mean(local_distance_v144k)) / 1000.0
    design_v144k = np.column_stack([taper_v144k, taper_v144k * x_v144k])
    move_v144k = np.clip(design_v144k @ coefficient_v144k[index_v144k, 1:3], -4.0, 4.0)
    output_values_v144k[positions_v144k] += move_v144k
    reports_v144k.append({"well": well_v144k, "mean_abs_move": float(np.mean(np.abs(move_v144k))), "max_abs_move": float(np.max(np.abs(move_v144k)))})

if not np.isfinite(output_values_v144k).all():
    raise RuntimeError("V144 produced non-finite predictions")
output_v144k = sample.copy()
output_v144k[sample_value_column] = output_values_v144k
output_v144k.to_csv(OUTPUT_V144K, index=False)
output_v144k.to_csv(NAMED_V144K, index=False)
pd.DataFrame(reports_v144k).to_csv(WORK_V144K / "v144_inference_report.csv", index=False)
summary_v144k = {
    "experiment": "final_consensus_sequence_coefficient_kaggle_v144",
    "parent": "V116 consensus_center",
    "rows": len(output_v144k),
    "wells": len(records_v144k),
    "parent_sha256": _sha256_v144k(PARENT_COPY_V144K),
    "submission_sha256": _sha256_v144k(NAMED_V144K),
    "mean_abs_move": float(np.mean(np.abs(output_values_v144k - base_values_v144k))),
    "max_abs_move": float(np.max(np.abs(output_values_v144k - base_values_v144k))),
    "submit": str(NAMED_V144K),
}
(WORK_V144K / "v144_kaggle_summary.json").write_text(json.dumps(summary_v144k, indent=2), encoding="utf-8")
print("V144 KAGGLE SUMMARY")
print(json.dumps(summary_v144k, indent=2), flush=True)
print("SUBMIT:", NAMED_V144K, flush=True)
