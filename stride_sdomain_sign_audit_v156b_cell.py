"""V156b: cheap S-domain sign/convention audit before another beam run."""
from __future__ import annotations

import ast
import json
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT = "stride_sdomain_sign_audit_v156b"
ROOT = Path("/root/autodl-tmp")
OOF_ROOT = ROOT / "oof_exports"
OUT_DIR = OOF_ROOT / EXPERIMENT
OUT_DIR.mkdir(parents=True, exist_ok=True)
STARTED = time.time()
SOURCE_CANDIDATES = (ROOT / "stride_joint_segment_decoder_gateway_v1_cell.py", Path.cwd() / "stride_joint_segment_decoder_gateway_v1_cell.py")
SOURCE = next((p for p in SOURCE_CANDIDATES if p.exists()), None)
if SOURCE is None:
    raise FileNotFoundError("Upload stride_joint_segment_decoder_gateway_v1_cell.py")


def _bootstrap(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    cutoff = next((i for i, node in enumerate(tree.body) if isinstance(node, ast.If) and "FOLD_PATH is None" in (ast.unparse(node.test) if hasattr(ast, "unparse") else "")), None)
    if cutoff is None:
        raise RuntimeError("Cannot locate STRIDE definition boundary")
    class DisableCache(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "njit":
                    for kw in dec.keywords:
                        if kw.arg == "cache": kw.value = ast.Constant(False)
            return node
    tree = DisableCache().visit(tree)
    ast.fix_missing_locations(tree)
    module = types.ModuleType("stride_v156b_defs")
    module.__file__ = str(path)
    exec(compile(ast.Module(body=tree.body[:cutoff], type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


V = _bootstrap(SOURCE)
fold_path = V.FOLD_PATH
load_record = V._load_record
NODE_STEP = 32.0
folds = pd.read_csv(fold_path, dtype={"well": str})[["well", "fold"]].drop_duplicates("well")
wells = folds.loc[folds["fold"].eq(3), "well"].sort_values().head(24).tolist()
if len(wells) != 24:
    raise RuntimeError(f"Expected 24 Fold3 audit wells, got {len(wells)}")


def _profile(record, sign):
    rows = record["evaluation"]
    index = np.r_[record["anchor"], rows]
    md = record["md"][index]
    tvt = np.r_[record["anchor_tvt"], record["truth"][rows]]
    z = record["z"][index]
    axis = np.arange(md[0], md[-1] + 0.5 * NODE_STEP, NODE_STEP)
    if axis[-1] < md[-1] - 1e-6: axis = np.r_[axis, md[-1]]
    s = np.interp(axis, md, tvt + sign * z)
    rate = np.diff(s) / np.maximum(np.diff(axis), 1e-6)
    change = np.diff(rate)
    return rate, change


rows = []
for number, well in enumerate(wells, 1):
    record = load_record(well)
    for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
        rate, change = _profile(record, sign)
        rows.append({
            "well": well,
            "sign": sign_name,
            "rate_median": float(np.median(rate)),
            "rate_std": float(np.std(rate)),
            "rate_abs_p95": float(np.quantile(np.abs(rate), 0.95)),
            "jump_abs_median": float(np.median(np.abs(change))),
            "jump_abs_p95": float(np.quantile(np.abs(change), 0.95)),
            "jump_total_variation": float(np.sum(np.abs(change))),
            "sign_switch_fraction": float(np.mean(rate[1:] * rate[:-1] < 0.0)) if len(rate) > 1 else 0.0,
        })
    if number % 8 == 0 or number == len(wells):
        print(f"V156b audit {number}/{len(wells)}", flush=True)

frame = pd.DataFrame(rows)
frame.to_csv(OUT_DIR / "bywell.csv", index=False)
summary = frame.groupby("sign", as_index=False).agg({
    "rate_std": "mean",
    "rate_abs_p95": "mean",
    "jump_abs_median": "mean",
    "jump_abs_p95": "mean",
    "jump_total_variation": "mean",
    "sign_switch_fraction": "mean",
})
summary.to_csv(OUT_DIR / "summary.csv", index=False)
plus = summary.loc[summary["sign"].eq("plus")].iloc[0]
minus = summary.loc[summary["sign"].eq("minus")].iloc[0]
verdict = {
    "experiment": EXPERIMENT,
    "stage": "24well_truth_audit_of_S_sign_convention",
    "wells": wells,
    "summary": summary.to_dict(orient="records"),
    "lower_variation_sign": "plus" if plus["jump_total_variation"] <= minus["jump_total_variation"] else "minus",
    "checks": {
        "plus_rate_abs_p95_finite": bool(np.isfinite(plus["rate_abs_p95"])),
        "minus_rate_abs_p95_finite": bool(np.isfinite(minus["rate_abs_p95"])),
        "variation_ratio_ge_1p10": bool(max(plus["jump_total_variation"], minus["jump_total_variation"]) / max(min(plus["jump_total_variation"], minus["jump_total_variation"]), 1e-9) >= 1.10),
    },
    "next": "Run V156 with the lower-variation S sign only" if max(plus["jump_total_variation"], minus["jump_total_variation"]) / max(min(plus["jump_total_variation"], minus["jump_total_variation"]), 1e-9) >= 1.10 else "Sign is not the main issue; close S-domain branch",
    "elapsed_minutes": (time.time() - STARTED) / 60.0,
}
(OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding="utf-8")
print("V156b SUMMARY")
print(summary.to_string(index=False), flush=True)
print("V156b FINAL VERDICT")
print(json.dumps(verdict, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x), flush=True)
