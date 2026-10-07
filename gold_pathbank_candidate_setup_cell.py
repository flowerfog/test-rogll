"""Register a restricted fold-clean path-bank menu in the original Gold judge.

Run this after the original Gold helper-definition cells and immediately before
``block_paired_gold_forward_chaining_cell.py``. It does not alter the Gold
judge, thresholds, shrinkage, or caps. It only contributes 12 anchored residual
shape candidates around Gold's existing default PF path.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


OOF_ROOT = Path("/root/autodl-tmp/oof_exports")
PATHBANK_DIR = OOF_ROOT / "gr_path_bank_oracle_v0"
BANK_DIR = PATHBANK_DIR / "banks"
META_PATH = PATHBANK_DIR / "well_metadata.csv"
OUT_DIR = OOF_ROOT / "gold_pathbank_restricted_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The path-bank oracle used raw corrections as large as 64-192 ft. Gold should
# consume a conservative version, not those oracle-scale moves directly.
TEMPLATE_DOSE = 0.50
CORRECTION_CAP_FT = 24.0
EXPECTED_CANDIDATES = 12

for path in (BANK_DIR, META_PATH):
    if not path.exists():
        raise FileNotFoundError(path)

meta = pd.read_csv(META_PATH, dtype={"well": str})
if not {"well", "fold"}.issubset(meta.columns):
    raise RuntimeError("path-bank metadata must contain well/fold")
_PB_FOLD_MAP = meta.set_index("well")["fold"].astype(int).to_dict()
_PB_BANK_CACHE = {}
_PB_SELECTION_ROWS = []


def _quantile_indices(count, quantiles):
    if count <= 0:
        return []
    return sorted(
        set(
            int(np.clip(round(float(q) * (count - 1)), 0, count - 1))
            for q in quantiles
        )
    )


def _family(name):
    name = str(name)
    if name.startswith("line_"):
        return "line"
    if name.startswith("raw_"):
        return "raw"
    if name.startswith("lenhalf_"):
        return "lenhalf"
    if name.startswith("pca_"):
        return "pca"
    return "other"


def _select_bank_templates(bank, fold):
    templates = np.asarray(bank["templates"], np.float32)
    alphas = np.asarray(bank["length_alphas"], np.float32)
    names = np.asarray(bank["names"], str)
    selected = []

    # Endpoint quantiles provide low/mid/high residual dynamics without using
    # the held-out target well or any candidate performance on it.
    family_spec = {
        "line": (4, (0.15, 0.38, 0.62, 0.85)),
        "raw": (3, (0.20, 0.50, 0.80)),
        "lenhalf": (3, (0.20, 0.50, 0.80)),
    }
    families = np.asarray([_family(name) for name in names], str)
    for family, (wanted, quantiles) in family_spec.items():
        indices = np.flatnonzero(families == family)
        order = indices[np.argsort(templates[indices, -1], kind="mergesort")]
        positions = _quantile_indices(len(order), quantiles)
        for slot, position in enumerate(positions[:wanted]):
            selected.append((f"{family}{slot + 1}", int(order[position])))

    # PCA components are only orientation-defined up to sign. Select the two
    # moderate c00 branches by name when available; otherwise use moderate-RMS
    # PCA paths deterministically.
    pca_preferred = []
    for token in ("pca_c00_l-1", "pca_c00_l+1"):
        matches = np.flatnonzero(np.char.startswith(names.astype(str), token))
        if len(matches):
            pca_preferred.append(int(matches[0]))
    if len(pca_preferred) < 2:
        pca_indices = np.flatnonzero(families == "pca")
        rms = np.sqrt(np.mean(templates[pca_indices] ** 2, axis=1))
        order = pca_indices[np.argsort(rms, kind="mergesort")]
        for position in _quantile_indices(len(order), (0.40, 0.60)):
            candidate = int(order[position])
            if candidate not in pca_preferred:
                pca_preferred.append(candidate)
            if len(pca_preferred) == 2:
                break
    for slot, index in enumerate(pca_preferred[:2]):
        selected.append((f"pca{slot + 1}", index))

    if len(selected) != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"fold {fold}: selected {len(selected)} instead of "
            f"{EXPECTED_CANDIDATES} templates"
        )

    output = []
    for slot, index in selected:
        output.append(
            {
                "slot": slot,
                "index": index,
                "source_name": str(names[index]),
                "family": _family(names[index]),
                "template": templates[index],
                "alpha": float(alphas[index]),
            }
        )
        _PB_SELECTION_ROWS.append(
            {
                "fold": int(fold),
                "slot": slot,
                "index": index,
                "source_name": str(names[index]),
                "family": _family(names[index]),
                "endpoint_ft": float(templates[index, -1]),
                "template_rms_ft": float(
                    np.sqrt(np.mean(templates[index] ** 2))
                ),
                "length_alpha": float(alphas[index]),
            }
        )
    return output


def _load_fold_bank(fold):
    fold = int(fold)
    if fold in _PB_BANK_CACHE:
        return _PB_BANK_CACHE[fold]
    path = BANK_DIR / f"fold{fold}_global.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as source:
        raw = {name: source[name] for name in source.files}
    train_wells = set(np.asarray(raw.get("train_wells", []), str))
    record = {
        "path": str(path),
        "progress_grid": np.asarray(raw["progress_grid"], np.float64),
        "median_length": float(np.ravel(raw["median_length"])[0]),
        "train_wells": train_wells,
        "selected": _select_bank_templates(raw, fold),
    }
    _PB_BANK_CACHE[fold] = record
    return record


def _default_curve(hw_masked, pool):
    default_name = None
    default_factory = globals().get("_default_pf_name")
    if callable(default_factory):
        try:
            default_name = default_factory(hw_masked)
        except Exception:
            default_name = None
    if default_name in pool:
        return str(default_name), np.asarray(pool[default_name], np.float64)

    pf_names = sorted(name for name in pool if str(name).startswith("pf|"))
    if pf_names:
        name = pf_names[0]
        return str(name), np.asarray(pool[name], np.float64)

    fallback_names = [
        name
        for name in ("poly_u_deg1_tail160", "poly_u_deg1_tail320")
        if name in pool
    ]
    if fallback_names:
        name = fallback_names[0]
        return str(name), np.asarray(pool[name], np.float64)
    raise RuntimeError("Gold pool has no usable PF/polynomial base candidate")


def _pathbank_candidate_factory(wid, hw_masked, existing_pool):
    wid = str(wid)
    fold = _PB_FOLD_MAP.get(wid)
    if fold is None:
        # This setup is intentionally OOF-only. A deployment all-well bank is
        # built only after the restricted menu survives the sanity audit.
        return {}
    bank = _load_fold_bank(fold)
    if wid in bank["train_wells"]:
        raise RuntimeError(f"fold contamination: {wid} appears in fold {fold} bank")

    required = {"MD", "TVT_input"}
    if not required.issubset(hw_masked.columns):
        return {}
    md = pd.to_numeric(hw_masked["MD"], errors="coerce").to_numpy(np.float64)
    tvt_input = pd.to_numeric(
        hw_masked["TVT_input"], errors="coerce"
    ).to_numpy(np.float64)
    known = np.flatnonzero(np.isfinite(tvt_input) & np.isfinite(md))
    if len(known) < 30:
        return {}
    anchor = int(known[-1])
    future = np.arange(anchor + 1, len(hw_masked), dtype=np.int32)
    if len(future) < 20 or not np.isfinite(md[future]).all():
        return {}

    eval_span = float(max(md[future[-1]] - md[anchor], 1.0))
    progress = np.clip((md[future] - md[anchor]) / eval_span, 0.0, 1.0)
    length_ratio = max(eval_span / max(bank["median_length"], 1.0), 0.1)
    base_name, base = _default_curve(hw_masked, existing_pool)
    if len(base) != len(hw_masked):
        return {}

    output = {}
    for item in bank["selected"]:
        correction = np.interp(
            progress, bank["progress_grid"], item["template"]
        )
        correction *= length_ratio ** item["alpha"]
        correction = np.clip(
            TEMPLATE_DOSE * correction,
            -CORRECTION_CAP_FT,
            CORRECTION_CAP_FT,
        )
        values = np.asarray(base, np.float64).copy()
        values[future] = values[future] + correction
        # Every correction is exactly zero at the current masked-prefix anchor.
        values[anchor] = base[anchor]
        name = f"pathbank_{item['slot']}"
        output[name] = values

    if len(output) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"{wid}: generated {len(output)} path candidates")
    return output


_GOLD_EXTRA_CANDIDATE_FACTORY_WITH_POOL = _pathbank_candidate_factory
_GOLD_EXTRA_CANDIDATE_PREFIXES = ("pathbank_",)
_GOLD_BLOCK_CACHE_VERSION_OVERRIDE = "v4_pathbank12_d050_cap24_s24_p350_b128"

# Preload and audit all fold selections before the expensive replay starts.
for _fold in sorted(set(_PB_FOLD_MAP.values())):
    _load_fold_bank(_fold)
selection = pd.DataFrame(_PB_SELECTION_ROWS).drop_duplicates(
    ["fold", "slot"]
).sort_values(["fold", "slot"])
selection.to_csv(OUT_DIR / "selected_templates.csv", index=False)
setup = {
    "experiment": "gold_pathbank_restricted_v1",
    "folds": sorted(set(_PB_FOLD_MAP.values())),
    "candidates_per_fold": EXPECTED_CANDIDATES,
    "template_dose": TEMPLATE_DOSE,
    "correction_cap_ft": CORRECTION_CAP_FT,
    "candidate_prefix": "pathbank_",
    "cache_version": _GOLD_BLOCK_CACHE_VERSION_OVERRIDE,
    "deployment_ready": False,
}
(OUT_DIR / "setup.json").write_text(
    json.dumps(setup, indent=2, ensure_ascii=True), encoding="utf-8"
)

print("PATHBANK GOLD SETUP")
print(json.dumps(setup, indent=2))
print(selection.to_string(index=False))
print("Next: run block_paired_gold_forward_chaining_cell.py")
