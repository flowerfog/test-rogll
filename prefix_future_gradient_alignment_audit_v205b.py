"""V205b: truth-only audit of prefix-to-future adaptation gradient alignment.

No parameters are updated. Hidden truth is used only to measure whether the
legal same-well prefix gradient points toward lower future loss. Fold3's fixed
24 discovery wells are compared with rotated donor-prefix gradients.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


STARTED_V205B = time.time()
EXPERIMENT_V205B = "prefix_future_gradient_alignment_audit_v205b"
required_v205b = (
    "model", "select_records", "device", "_block_nodes", "_inputs",
    "STATE_OFFSETS", "OUT",
)
missing_v205b = [name for name in required_v205b if name not in globals()]
if missing_v205b:
    raise RuntimeError("Run V202/V205 in this live kernel first. Missing: " + repr(missing_v205b))

OUT_V205B = Path(OUT).parent / EXPERIMENT_V205B
OUT_V205B.mkdir(parents=True, exist_ok=True)
WELLS_V205B = 24
PREFIX_WINDOW_FT_V205B = 2048.0
PREFIX_STEP_FT_V205B = 24.0


def _clean_v205b(value):
    if isinstance(value, dict):
        return {str(key): _clean_v205b(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_v205b(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _prefix_nodes_v205b(record):
    known = np.flatnonzero(record.condition[0] > 0.5)
    anchor_md = float(record.md[record.anchor_index])
    known = known[record.md[known] >= anchor_md - PREFIX_WINDOW_FT_V205B]
    grid = np.arange(
        max(float(record.md[known[0]]), anchor_md - PREFIX_WINDOW_FT_V205B),
        anchor_md + 0.5 * PREFIX_STEP_FT_V205B,
        PREFIX_STEP_FT_V205B,
    )
    positions = np.clip(np.searchsorted(record.md[known], grid), 0, len(known) - 1)
    rows = np.unique(known[positions]).astype(np.int64)
    label = record.condition[1, rows].astype(np.float32) * 128.0
    valid = np.isfinite(label) & (label >= STATE_OFFSETS[0]) & (label <= STATE_OFFSETS[-1])
    return rows[valid], label[valid]


def _supervised_loss_v205b(record, rows, label):
    logits = model(*_inputs(record, rows, device))[0]
    states = torch.as_tensor(STATE_OFFSETS, device=device)[:, None]
    target = torch.as_tensor(label, device=device)[None]
    distribution = torch.exp(-0.5 * ((states - target) / 2.0) ** 2)
    distribution /= distribution.sum(dim=0, keepdim=True).clamp_min(1e-8)
    logp = F.log_softmax(logits, dim=0)
    ce = -(distribution * logp).sum(dim=0).mean()
    probability = F.softmax(logits, dim=0)
    expected = (probability * states).sum(dim=0)
    path = F.smooth_l1_loss(expected / 8.0, target[0] / 8.0)
    return ce + 0.20 * path


parameters_v205b = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def _gradient_v205b(loss):
    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in parameters_v205b],
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    output = {}
    for (name, parameter), gradient in zip(parameters_v205b, gradients):
        output[name] = (
            torch.zeros_like(parameter).flatten()
            if gradient is None else gradient.detach().flatten()
        )
    return output


def _cosine_v205b(left, right, prefix=None):
    names = [name for name in left if prefix is None or name.startswith(prefix)]
    dot = sum(torch.dot(left[name], right[name]) for name in names)
    left_norm = torch.sqrt(sum(torch.dot(left[name], left[name]) for name in names))
    right_norm = torch.sqrt(sum(torch.dot(right[name], right[name]) for name in names))
    denominator = left_norm * right_norm
    return float((dot / denominator.clamp_min(1e-12)).cpu())


if "base_state_v205" in globals():
    model.load_state_dict(base_state_v205)
elif "base_state_v203" in globals():
    model.load_state_dict(base_state_v203)
model.eval()
records_v205b = sorted(select_records, key=lambda record: record.well)[:WELLS_V205B]
layer_prefixes_v205b = ("encoder", "condition", "input", "context", "output")
rows_v205b = []

for index_v205b, record_v205b in enumerate(records_v205b):
    prefix_rows_v205b, prefix_label_v205b = _prefix_nodes_v205b(record_v205b)
    hidden_rows_v205b, hidden_label_v205b, _, _ = _block_nodes(record_v205b)
    donor_v205b = records_v205b[(index_v205b + max(1, WELLS_V205B // 2)) % WELLS_V205B]
    donor_rows_v205b, donor_label_v205b = _prefix_nodes_v205b(donor_v205b)

    aligned_gradient_v205b = _gradient_v205b(
        _supervised_loss_v205b(record_v205b, prefix_rows_v205b, prefix_label_v205b)
    )
    hidden_gradient_v205b = _gradient_v205b(
        _supervised_loss_v205b(record_v205b, hidden_rows_v205b, hidden_label_v205b)
    )
    donor_gradient_v205b = _gradient_v205b(
        _supervised_loss_v205b(donor_v205b, donor_rows_v205b, donor_label_v205b)
    )
    aligned_cosine_v205b = _cosine_v205b(aligned_gradient_v205b, hidden_gradient_v205b)
    donor_cosine_v205b = _cosine_v205b(donor_gradient_v205b, hidden_gradient_v205b)
    row_v205b = {
        "well": record_v205b.well,
        "prefix_nodes": len(prefix_rows_v205b),
        "hidden_nodes": len(hidden_rows_v205b),
        "aligned_cosine": aligned_cosine_v205b,
        "donor_cosine": donor_cosine_v205b,
        "aligned_increment": aligned_cosine_v205b - donor_cosine_v205b,
    }
    for layer_v205b in layer_prefixes_v205b:
        row_v205b[f"{layer_v205b}_cosine"] = _cosine_v205b(
            aligned_gradient_v205b, hidden_gradient_v205b, prefix=layer_v205b
        )
    rows_v205b.append(row_v205b)
    print(f"V205b gradients {index_v205b + 1}/{WELLS_V205B}", flush=True)

detail_v205b = pd.DataFrame(rows_v205b)
detail_v205b.to_csv(OUT_V205B / "well_gradient_alignment.csv", index=False)
summary_v205b = {
    "wells": len(detail_v205b),
    "mean_aligned_cosine": float(detail_v205b["aligned_cosine"].mean()),
    "median_aligned_cosine": float(detail_v205b["aligned_cosine"].median()),
    "positive_aligned_fraction": float((detail_v205b["aligned_cosine"] > 0).mean()),
    "mean_donor_cosine": float(detail_v205b["donor_cosine"].mean()),
    "mean_aligned_increment": float(detail_v205b["aligned_increment"].mean()),
    "positive_increment_fraction": float((detail_v205b["aligned_increment"] > 0).mean()),
    "layer_mean_cosines": {
        layer: float(detail_v205b[f"{layer}_cosine"].mean())
        for layer in layer_prefixes_v205b
    },
}
checks_v205b = {
    "median_aligned_cosine_ge_0p05": summary_v205b["median_aligned_cosine"] >= 0.05,
    "positive_aligned_fraction_ge_0p65": summary_v205b["positive_aligned_fraction"] >= 0.65,
    "aligned_increment_ge_0p05": summary_v205b["mean_aligned_increment"] >= 0.05,
}
strict_pass_v205b = bool(all(checks_v205b.values()))
verdict_v205b = {
    "experiment": EXPERIMENT_V205B,
    "stage": "read_only_Fold3_prefix_future_gradient_observability_audit",
    "training_or_selection_count": 0,
    "truth_usage": "future gradients and audit metrics only",
    "fold2_touched": 0,
    "fold1_touched": 0,
    "summary": summary_v205b,
    "checks": checks_v205b,
    "strict_pass": strict_pass_v205b,
    "elapsed_minutes": (time.time() - STARTED_V205B) / 60.0,
    "next": (
        "PASS -> train an episodic gradient-aligned correlation model"
        if strict_pass_v205b
        else "FAIL -> prefix gradients do not identify future improvements; close test-time adaptation"
    ),
}
(OUT_V205B / "verdict.json").write_text(
    json.dumps(_clean_v205b(verdict_v205b), indent=2, allow_nan=False), encoding="utf-8"
)
print("\nV205b FINAL VERDICT")
print(json.dumps(_clean_v205b(verdict_v205b), indent=2, allow_nan=False), flush=True)
print("saved:", OUT_V205B, flush=True)
