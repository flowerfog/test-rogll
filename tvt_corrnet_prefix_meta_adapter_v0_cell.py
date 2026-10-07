"""Deployment-legal prefix meta-adapter gateway for heel CorrNet v4.

Run immediately after ``tvt_corrnet_heel_conditioned_onefold_v4_cell.py`` in
the same kernel.  The shared CorrNet checkpoint is frozen.  For each outer
validation well, only pseudo starts earlier than its official prediction start
are used as support.  Adapter labels are further restricted to the interval
between that pseudo start and the official start, which is visible at hidden
deployment.  Official toe truth never enters adaptation.

The adapter has four parameters: logit temperature, linear state bias,
quadratic state bias, and progress-dependent linear state bias.  A shuffled
adapter control tests whether any gain comes from well-specific prefix evidence.
"""

from pathlib import Path
import ast
import gc
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def _bootstrap_v4_namespace():
    candidates = (
        Path("/root/autodl-tmp/tvt_corrnet_heel_conditioned_onefold_v4_cell.py"),
        Path.cwd() / "tvt_corrnet_heel_conditioned_onefold_v4_cell.py",
    )
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        current = globals()
        required = (
            "HeelConditionedCorrNet",
            "PanelStore",
            "_build_panel",
            "_panel_key",
            "_compose_input",
            "_first_existing",
            "_feature_path",
            "DATA_ROOT",
            "OOF_ROOT",
        )
        missing_current = [name for name in required if name not in current]
        if missing_current:
            raise FileNotFoundError(
                "v4 source is absent and notebook definitions are incomplete; "
                "missing: " + repr(missing_current)
            )
        oof_root = Path(current["OOF_ROOT"])
        root = Path("/root/autodl-tmp")
        base_out = oof_root / "tvt_corrnet_heel_conditioned_onefold_v4"
        bank = oof_root / "multicut_fleongg_matched_k3_gateway_v1"
        extension = oof_root / "multicut_fleongg_matched_k5_extension_v1"
        overrides = {
            "ROOT": root,
            "OOF_ROOT": oof_root,
            "OUT_DIR": base_out,
            "PANEL_DIR": base_out / "panels",
            "MODEL_DIR": base_out / "models",
            "BANK_DIR": bank,
            "EXTENSION_DIR": extension,
            "FEATURE_DIR": bank / "features",
            "FOLD_CANDIDATES": (
                oof_root / "foldclean_full_fleongg_lgb_v0" / "well_metadata.csv",
                bank / "well_metadata.csv",
            ),
            "MANIFEST_CANDIDATES": (
                bank / "cut_manifest.csv",
                bank / "cut_manifest_v1.csv",
            ),
            "EXTRA_MANIFEST_CANDIDATES": (
                extension / "new_cut_manifest.csv",
                extension / "cut_manifest.csv",
            ),
            "VALID_FOLD": 1,
            "TRAIN_TAGS": ("official", "pseudo1", "pseudo2", "pseudo3", "pseudo4"),
            "TAG_MASS": {
                "official": 0.40,
                "pseudo1": 0.15,
                "pseudo2": 0.15,
                "pseudo3": 0.15,
                "pseudo4": 0.15,
            },
            "PF_CENTER_COLUMN": "likpf_scale_12",
            "PANEL_VERSION": 4,
            "PREFIX_COLUMNS": 64,
            "EVAL_COLUMNS": 256,
            "N_COLUMNS": 320,
            "STATE_OFFSETS": np.arange(-96.0, 96.01, 2.0, dtype=np.float32),
            "TARGET_SIGMA_FT": 2.0,
            "BASE_CHANNELS": 32,
            "SEED": 20260717,
            "RAW_CHANNEL_NAMES": (
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
            ),
            "TYPEWELL_CHANNELS": (2, 3, 4, 5),
            "GR_VALUE_CHANNELS": (0, 2, 4, 5, 6, 8, 9),
        }
        current.update(overrides)
        # Functions and classes pasted into an earlier notebook cell may retain
        # that cell's globals dictionary.  Restore it explicitly as well.
        objects = [
            current[name]
            for name in (
                "_build_panel",
                "_feature_path",
                "_compose_input",
                "_panel_key",
                "PanelStore",
                "HeelConditionedCorrNet",
            )
        ]
        for obj in objects:
            function_globals = getattr(obj, "__globals__", None)
            if function_globals is not None:
                function_globals.update(overrides)
            if isinstance(obj, type):
                for member in vars(obj).values():
                    member_globals = getattr(member, "__globals__", None)
                    if member_globals is not None:
                        member_globals.update(overrides)
        namespace = dict(current)
        namespace.update(overrides)
        return namespace, "notebook_globals_fallback"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    cutoff = next(
        (
            node.lineno
            for node in tree.body
            if isinstance(node, ast.For) and node.lineno > 900
        ),
        None,
    )
    if cutoff is None:
        raise RuntimeError("cannot locate the v4 execution boundary")
    definition_types = (
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.FunctionDef,
        ast.ClassDef,
    )
    definition_nodes = [
        node
        for node in tree.body
        if node.lineno < cutoff and isinstance(node, definition_types)
    ]
    namespace = {
        "__name__": "corrnet_v4_meta_bootstrap",
        "__file__": str(source),
    }
    exec(
        compile(
            ast.Module(body=definition_nodes, type_ignores=[]),
            filename=str(source),
            mode="exec",
        ),
        namespace,
    )
    return namespace, source


V4, V4_SOURCE = _bootstrap_v4_namespace()
OOF_ROOT = Path(V4["OOF_ROOT"])
BASE_OUT_DIR = Path(V4["OUT_DIR"])
PANEL_DIR = Path(V4["PANEL_DIR"])
MODEL_DIR = Path(V4["MODEL_DIR"])
SEED = int(V4["SEED"])
STATE_OFFSETS = np.asarray(V4["STATE_OFFSETS"], np.float32)
TARGET_SIGMA_FT = float(V4["TARGET_SIGMA_FT"])
PREFIX_COLUMNS = int(V4["PREFIX_COLUMNS"])
HeelConditionedCorrNet = V4["HeelConditionedCorrNet"]
PanelStore = V4["PanelStore"]
_build_panel = V4["_build_panel"]
_panel_key = V4["_panel_key"]
_compose_input = V4["_compose_input"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")

fold_path = V4["_first_existing"](V4["FOLD_CANDIDATES"])
if fold_path is None:
    raise FileNotFoundError("v4 fold metadata not found")
fold = pd.read_csv(fold_path, dtype={"well": str})
fold = fold.groupby("well", as_index=False)["fold"].first()
fold["well"] = fold["well"].astype(str)
fold["fold"] = fold["fold"].astype(int)
valid_wells = fold.loc[fold["fold"] == int(V4["VALID_FOLD"]), "well"].tolist()

manifest_path = BASE_OUT_DIR / "cut_manifest.csv"
if not manifest_path.exists():
    raise FileNotFoundError(manifest_path)
manifest = pd.read_csv(manifest_path, dtype={"well": str, "tag": str})
manifest["well"] = manifest["well"].astype(str)
manifest["tag"] = manifest["tag"].astype(str)
valid_set_for_bootstrap = set(valid_wells)
official_tasks = manifest[
    manifest["well"].isin(valid_set_for_bootstrap)
    & (manifest["tag"] == "official")
].copy()
for task in official_tasks.to_dict("records"):
    panel_path = PANEL_DIR / f"{_panel_key(task['well'], task['tag'])}.npz"
    if not panel_path.exists():
        result = _build_panel(task)
        if result.get("status") == "failed":
            raise RuntimeError(f"failed to restore official panel: {result}")
official_keys = [
    _panel_key(str(row.well), str(row.tag))
    for row in official_tasks.itertuples(index=False)
]
valid_store = PanelStore(official_keys)
input_channels = int(_compose_input(valid_store, 0, "full").shape[0])

META_OUT_DIR = Path(OOF_ROOT) / "tvt_corrnet_prefix_meta_adapter_v0"
META_OUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_STEPS = 120
ADAPTER_LR = 0.05
ADAPTER_L2 = 0.03
MIN_SUPPORT_COLUMNS = 12
DOSES = (0.0, 0.25, 0.50, 0.75, 1.00)


def _load_frozen_model():
    checkpoint_path = Path(MODEL_DIR) / f"seed_{SEED}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict"))
    if state_dict is None:
        raise RuntimeError(
            f"checkpoint has no model state; keys={sorted(checkpoint.keys())}"
        )
    saved_input_channels = checkpoint.get("input_channels")
    if saved_input_channels is None:
        stem_key = next(
            (
                key
                for key in state_dict
                if key.endswith("enc1.block.0.weight")
                and getattr(state_dict[key], "ndim", 0) == 4
            ),
            None,
        )
        if stem_key is None:
            saved_input_channels = int(input_channels)
        else:
            saved_input_channels = int(state_dict[stem_key].shape[1])
    frozen = HeelConditionedCorrNet(int(saved_input_channels)).to(device)
    frozen.load_state_dict(state_dict)
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen, checkpoint_path, checkpoint


@torch.no_grad()
def _panel_logits(model, store, index):
    values = torch.from_numpy(_compose_input(store, index, "full"))[None].to(device)
    valid = torch.from_numpy(store.records[index]["state_valid"])[None].to(device)
    state_offsets = torch.from_numpy(STATE_OFFSETS).to(device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(device.type == "cuda"),
    ):
        logits = model(values, state_offsets)
    return logits[0].float().masked_fill(~valid[0], -30.0)


def _adapt_logits(logits, theta, progress):
    state = torch.as_tensor(
        STATE_OFFSETS / float(np.max(np.abs(STATE_OFFSETS))),
        dtype=logits.dtype,
        device=logits.device,
    )[:, None]
    centred_square = state.square() - state.square().mean()
    log_scale = 0.50 * torch.tanh(theta[0])
    linear = 2.00 * torch.tanh(theta[1])
    quadratic = 1.00 * torch.tanh(theta[2])
    progress_linear = 2.00 * torch.tanh(theta[3])
    scale = torch.exp(log_scale)
    bias = (
        linear * state
        + quadratic * centred_square
        + progress_linear * state * progress[None]
    )
    return scale * logits + bias


def _fit_adapter(support_items):
    theta = torch.zeros(4, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=ADAPTER_LR)
    state_offsets = torch.from_numpy(STATE_OFFSETS).to(device)
    before = None
    final_data = None
    for step in range(ADAPTER_STEPS + 1):
        losses = []
        ce_rows = []
        regression_rows = []
        for item in support_items:
            logits = item["logits"]
            adapted = _adapt_logits(logits, theta, item["progress"])
            distance = state_offsets[:, None] - item["truth_residual"][None]
            target = torch.softmax(-0.5 * (distance / TARGET_SIGMA_FT) ** 2, dim=0)
            target = target * item["state_valid"]
            target = target / target.sum(dim=0, keepdim=True).clamp_min(1e-8)
            log_probability = F.log_softmax(adapted, dim=0)
            probability = torch.softmax(adapted, dim=0)
            correction = (probability * state_offsets[:, None]).sum(dim=0)
            mask = item["support_mask"]
            ce = (-(target * log_probability).sum(dim=0)[mask]).mean()
            regression = F.smooth_l1_loss(
                correction[mask], item["truth_residual"][mask], beta=2.0
            )
            losses.append(ce + 0.10 * regression)
            ce_rows.append(ce)
            regression_rows.append(regression)
        data_loss = torch.stack(losses).mean()
        regularisation = ADAPTER_L2 * theta.square().sum()
        loss = data_loss + regularisation
        if step == 0:
            before = {
                "loss": float(data_loss.detach().cpu()),
                "ce": float(torch.stack(ce_rows).mean().detach().cpu()),
                "regression": float(
                    torch.stack(regression_rows).mean().detach().cpu()
                ),
            }
        if step == ADAPTER_STEPS:
            final_data = {
                "loss": float(data_loss.detach().cpu()),
                "ce": float(torch.stack(ce_rows).mean().detach().cpu()),
                "regression": float(
                    torch.stack(regression_rows).mean().detach().cpu()
                ),
            }
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], 5.0)
        optimizer.step()
    return theta.detach(), before, final_data


@torch.no_grad()
def _official_prediction(model, store, index, theta, dose):
    record = store.records[index]
    logits = _panel_logits(model, store, index)
    progress = torch.linspace(0.0, 1.0, logits.shape[1], device=device)
    adapted = _adapt_logits(logits, theta * float(dose), progress)
    probability = torch.softmax(adapted, dim=0)
    state_offsets = torch.from_numpy(STATE_OFFSETS).to(device)
    correction = (probability * state_offsets[:, None]).sum(dim=0).cpu().numpy()
    correction[:PREFIX_COLUMNS] = 0.0
    column_s = record["center_tvt"] + correction + record["column_z"]
    prediction = (
        np.interp(record["eval_md"], record["column_md"], column_s)
        - record["eval_z"]
    )
    center_s = record["center_tvt"] + record["column_z"]
    center = (
        np.interp(record["eval_md"], record["column_md"], center_s)
        - record["eval_z"]
    )
    return prediction, center


frozen_model, checkpoint_path, checkpoint = _load_frozen_model()
valid_set = set(map(str, valid_wells))
official_rows = (
    manifest[(manifest["well"].isin(valid_set)) & (manifest["tag"] == "official")]
    .set_index("well")
)
support_tasks = manifest[
    manifest["well"].isin(valid_set) & (manifest["tag"] != "official")
].copy()
support_tasks["official_cut"] = support_tasks["well"].map(
    official_rows["cut_position"].to_dict()
)
support_tasks = support_tasks[
    support_tasks["cut_position"] < support_tasks["official_cut"]
].copy()

build_rows = []
for task in tqdm(support_tasks.to_dict("records"), desc="build legal meta-support panels"):
    build_rows.append(_build_panel(task))
build_report = pd.DataFrame(build_rows)
build_report.to_csv(META_OUT_DIR / "support_panel_build.csv", index=False)
failed = build_report[build_report["status"] == "failed"] if len(build_report) else build_report
if len(failed):
    print("SUPPORT BUILD FAILURES")
    print(failed.head(30).to_string(index=False))

support_keys = []
support_metadata = []
for task in support_tasks.itertuples(index=False):
    key = _panel_key(str(task.well), str(task.tag))
    path = Path(PANEL_DIR) / f"{key}.npz"
    if path.exists():
        support_keys.append(key)
        support_metadata.append(
            {
                "well": str(task.well),
                "tag": str(task.tag),
                "official_cut": int(task.official_cut),
            }
        )
support_store = PanelStore(support_keys)
support_by_well = {}
for index, metadata in enumerate(support_metadata):
    support_by_well.setdefault(metadata["well"], []).append((index, metadata))

official_index = {
    record["well"]: index for index, record in enumerate(valid_store.records)
}
adapter_rows = []
theta_by_well = {}
started = time.time()
for well in tqdm(valid_wells, desc="fit prefix meta-adapters"):
    well = str(well)
    official_record = valid_store.records[official_index[well]]
    official_anchor_md = float(official_record["column_md"][PREFIX_COLUMNS - 1])
    support_items = []
    for support_index, metadata in support_by_well.get(well, []):
        record = support_store.records[support_index]
        support_mask_np = (
            (record["eval_mask"] > 0.5)
            & (record["column_md"] <= official_anchor_md + 1e-6)
        )
        if int(support_mask_np.sum()) < MIN_SUPPORT_COLUMNS:
            continue
        logits = _panel_logits(frozen_model, support_store, support_index)
        support_items.append(
            {
                "logits": logits,
                "truth_residual": torch.from_numpy(record["truth_residual"]).to(device),
                "state_valid": torch.from_numpy(record["state_valid"]).to(device),
                "support_mask": torch.from_numpy(support_mask_np).to(device),
                "progress": torch.linspace(0.0, 1.0, logits.shape[1], device=device),
            }
        )
    if support_items:
        theta, before, after = _fit_adapter(support_items)
    else:
        theta = torch.zeros(4, dtype=torch.float32, device=device)
        before = {"loss": np.nan, "ce": np.nan, "regression": np.nan}
        after = before.copy()
    theta_by_well[well] = theta.cpu().numpy()
    adapter_rows.append(
        {
            "well": well,
            "support_panels": len(support_items),
            "support_columns": int(
                sum(int(item["support_mask"].sum().cpu()) for item in support_items)
            ),
            "loss_before": before["loss"],
            "loss_after": after["loss"],
            "theta_log_scale": float(theta[0].cpu()),
            "theta_linear": float(theta[1].cpu()),
            "theta_quadratic": float(theta[2].cpu()),
            "theta_progress_linear": float(theta[3].cpu()),
        }
    )
adapter_frame = pd.DataFrame(adapter_rows)
adapter_frame.to_csv(META_OUT_DIR / "adapter_bywell.csv", index=False)

eligible_wells = adapter_frame.loc[adapter_frame["support_panels"] > 0, "well"].tolist()
shifted_theta = {}
if eligible_wells:
    for position, well in enumerate(eligible_wells):
        shifted_theta[well] = theta_by_well[eligible_wells[(position + 1) % len(eligible_wells)]]

truth_chunks = []
center_chunks = []
prediction_by_variant = {
    f"adapted_dose_{dose:.2f}": [] for dose in DOSES
}
prediction_by_variant["shuffled_adapter"] = []
row_metadata = []
for well in valid_wells:
    well = str(well)
    index = official_index[well]
    record = valid_store.records[index]
    truth = record["eval_truth"].astype(np.float64)
    truth_chunks.append(truth)
    theta = torch.from_numpy(theta_by_well[well]).to(device)
    for dose in DOSES:
        prediction, center = _official_prediction(
            frozen_model, valid_store, index, theta, dose
        )
        prediction_by_variant[f"adapted_dose_{dose:.2f}"].append(prediction)
    center_chunks.append(center)
    donor_theta = torch.from_numpy(
        shifted_theta.get(well, np.zeros(4, np.float32))
    ).to(device)
    shuffled, _ = _official_prediction(
        frozen_model, valid_store, index, donor_theta, 1.0
    )
    prediction_by_variant["shuffled_adapter"].append(shuffled)
    row_metadata.append(
        pd.DataFrame(
            {
                "well": well,
                "row_idx": record["eval_rows"],
                "truth": truth,
            }
        )
    )

truth_all = np.concatenate(truth_chunks)
center_all = np.concatenate(center_chunks)
score_rows = []
best_variant = None
best_prediction = None
best_rmse = np.inf
for variant, chunks in prediction_by_variant.items():
    prediction = np.concatenate(chunks)
    error = prediction - truth_all
    well_rmse = np.asarray(
        [
            np.sqrt(np.mean((candidate - truth) ** 2))
            for candidate, truth in zip(chunks, truth_chunks)
        ]
    )
    row = {
        "variant": variant,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_well_rmse": float(np.median(well_rmse)),
        "p95_well_rmse": float(np.quantile(well_rmse, 0.95)),
    }
    score_rows.append(row)
    if variant.startswith("adapted_dose_") and row["rmse"] < best_rmse:
        best_rmse = row["rmse"]
        best_variant = variant
        best_prediction = prediction.copy()
scores = pd.DataFrame(score_rows).sort_values("rmse")
base_rmse = float(
    scores.loc[scores["variant"] == "adapted_dose_0.00", "rmse"].iloc[0]
)
checkpoint_metrics = checkpoint.get("metrics", {})
checkpoint_rmse = checkpoint_metrics.get("rmse")
if checkpoint_rmse is None and isinstance(globals().get("result"), dict):
    checkpoint_rmse = globals()["result"].get("rmse")
if checkpoint_rmse is not None and abs(base_rmse - float(checkpoint_rmse)) > 0.02:
    raise RuntimeError(
        f"frozen CorrNet lineage mismatch: reconstructed={base_rmse:.6f}, "
        f"checkpoint={float(checkpoint_rmse):.6f}"
    )
full_rmse = float(
    scores.loc[scores["variant"] == "adapted_dose_1.00", "rmse"].iloc[0]
)
shuffled_rmse = float(
    scores.loc[scores["variant"] == "shuffled_adapter", "rmse"].iloc[0]
)
scores["gain_vs_frozen"] = base_rmse - scores["rmse"]
scores.to_csv(META_OUT_DIR / "scores.csv", index=False)

best_rows = pd.concat(row_metadata, ignore_index=True)
best_rows["prediction"] = best_prediction
best_rows.to_pickle(META_OUT_DIR / "best_predictions.pkl")

coverage = float(np.mean(adapter_frame["support_panels"] > 0))
inner_improvement = float(
    np.nanmean(adapter_frame["loss_before"] - adapter_frame["loss_after"])
)
meta_signal_pass = bool(
    full_rmse <= base_rmse - 0.30
    and shuffled_rmse >= full_rmse + 0.10
    and coverage >= 0.55
)
architecture_pass = bool(full_rmse <= 7.0)
target_pass = bool(full_rmse <= 5.5)
verdict = {
    "experiment": "tvt_corrnet_prefix_meta_adapter_v0",
    "base_checkpoint": str(checkpoint_path),
    "base_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
    "valid_wells": len(valid_wells),
    "adapted_wells": int(np.sum(adapter_frame["support_panels"] > 0)),
    "adaptation_coverage": coverage,
    "mean_inner_loss_improvement": inner_improvement,
    "frozen_rmse": base_rmse,
    "full_adapter_rmse": full_rmse,
    "full_adapter_gain": float(base_rmse - full_rmse),
    "shuffled_adapter_rmse": shuffled_rmse,
    "best_same_fold_variant": best_variant,
    "best_same_fold_rmse": float(best_rmse),
    "meta_signal_pass": meta_signal_pass,
    "architecture_pass": architecture_pass,
    "target_5p5_pass": target_pass,
    "strict_pass": bool(meta_signal_pass and architecture_pass),
    "next": (
        "PASS -> implement end-to-end meta-training and a second seed"
        if meta_signal_pass
        else "FAIL -> close low-dimensional prefix adaptation and proceed to S3"
    ),
    "elapsed_minutes": (time.time() - started) / 60.0,
}
with open(META_OUT_DIR / "verdict.json", "w", encoding="utf-8") as handle:
    json.dump(verdict, handle, indent=2)

print("\nMETA-ADAPTER SCORES")
print(scores.to_string(index=False))
print("\nADAPTER SUMMARY")
print(
    adapter_frame[
        [
            "support_panels",
            "support_columns",
            "loss_before",
            "loss_after",
            "theta_log_scale",
            "theta_linear",
            "theta_quadratic",
            "theta_progress_linear",
        ]
    ].describe().to_string()
)
print("\nVERDICT")
print(json.dumps(verdict, indent=2))
print("saved:", META_OUT_DIR)

del frozen_model
gc.collect()
if device.type == "cuda":
    torch.cuda.empty_cache()
