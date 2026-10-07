"""Export four raw single-well input sections from the competition train set.

Sections:
1. Heel/prefix known TVT_input sequence through the prediction-start anchor.
2. Toe/eval trajectory after the anchor: MD, X, Y, Z only.
3. Full horizontal GR aligned by original row index and MD.
4. One row per well with first-row/wellhead, anchor, and toe-end XY.

No hidden/eval TVT label is exported in the toe trajectory section.
"""

from pathlib import Path
import json
import shutil
import time
import zipfile

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


DATA_ROOT_CANDIDATES = (
    Path("/root/autodl-tmp/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    Path(r"E:\rogii-wellbore-geology-prediction"),
)
CREATE_ZIP = True


def _find_data_root():
    for root in DATA_ROOT_CANDIDATES:
        train = root / "train"
        if train.exists() and next(train.glob("*__horizontal_well.csv"), None):
            return root
    raise FileNotFoundError("cannot locate competition train directory")


def _size(path):
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _finite_xy_row(x, y):
    valid = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if not len(valid):
        raise RuntimeError("well contains no finite XY pair")
    return int(valid[0])


DATA_ROOT = _find_data_root()
TRAIN_DIR = DATA_ROOT / "train"
OOF_ROOT_CANDIDATES = (
    Path("/root/autodl-tmp/oof_exports"),
    DATA_ROOT / "oof_exports",
)
OOF_ROOT = next(
    (path for path in OOF_ROOT_CANDIDATES if path.exists()),
    DATA_ROOT / "oof_exports",
)
OOF_ROOT.mkdir(parents=True, exist_ok=True)
OUT_DIR = OOF_ROOT / "raw_single_well_inputs_v2"
ZIP_PATH = OOF_ROOT / "raw_single_well_inputs_v2.zip"
directories = {
    "heel_prefix_tvt": OUT_DIR / "heel_prefix_tvt",
    "toe_trajectory": OUT_DIR / "toe_trajectory",
    "horizontal_gr": OUT_DIR / "horizontal_gr",
    "metadata": OUT_DIR / "metadata",
}
for directory in directories.values():
    directory.mkdir(parents=True, exist_ok=True)

horizontal_files = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
if len(horizontal_files) < 700:
    raise RuntimeError(f"expected about 773 train wells, found {len(horizontal_files)}")

started = time.time()
well_rows = []
failures = []
required = ["MD", "X", "Y", "Z", "GR", "TVT_input"]

print(
    {
        "experiment": "export_raw_single_well_inputs_v2",
        "data_root": str(DATA_ROOT),
        "wells": len(horizontal_files),
        "output": str(OUT_DIR),
        "free_gb_before": shutil.disk_usage(OOF_ROOT).free / 2**30,
    },
    flush=True,
)

for source in tqdm(horizontal_files, desc="export raw single-well inputs"):
    well = source.name.removesuffix("__horizontal_well.csv")
    try:
        frame = pd.read_csv(source, usecols=required)
        if not set(required).issubset(frame.columns):
            raise RuntimeError(f"missing columns: {sorted(set(required) - set(frame.columns))}")
        if len(frame) < 2:
            raise RuntimeError("well has fewer than two rows")

        md = pd.to_numeric(frame["MD"], errors="coerce").to_numpy(np.float64)
        x = pd.to_numeric(frame["X"], errors="coerce").to_numpy(np.float64)
        y = pd.to_numeric(frame["Y"], errors="coerce").to_numpy(np.float64)
        z = pd.to_numeric(frame["Z"], errors="coerce").to_numpy(np.float64)
        gr = pd.to_numeric(frame["GR"], errors="coerce").to_numpy(np.float64)
        tvt_input = pd.to_numeric(
            frame["TVT_input"], errors="coerce"
        ).to_numpy(np.float64)
        row_idx = np.arange(len(frame), dtype=np.int32)

        known = np.isfinite(tvt_input)
        if not known.any() or known.all():
            raise RuntimeError("TVT_input does not define a prefix/eval split")
        anchor_row = int(np.flatnonzero(known)[-1])
        prefix_rows = np.arange(0, anchor_row + 1, dtype=np.int32)
        toe_rows = np.arange(anchor_row + 1, len(frame), dtype=np.int32)
        if not len(toe_rows):
            raise RuntimeError("empty toe/eval segment")

        # Preserve every prefix row and an explicit validity mask. Normally all
        # prefix rows are known, but the mask makes occasional internal holes
        # unambiguous to downstream code.
        prefix_valid = np.isfinite(tvt_input[prefix_rows])
        np.savez_compressed(
            directories["heel_prefix_tvt"] / f"{well}.npz",
            row_idx=prefix_rows,
            md=md[prefix_rows].astype(np.float32),
            tvt_input=tvt_input[prefix_rows].astype(np.float32),
            tvt_valid=prefix_valid.astype(np.uint8),
            anchor_row=np.asarray([anchor_row], np.int32),
            anchor_md=np.asarray([md[anchor_row]], np.float32),
            anchor_tvt=np.asarray([tvt_input[anchor_row]], np.float32),
        )

        # Deliberately omit train TVT here: this section mirrors legal test-time
        # toe trajectory inputs only.
        np.savez_compressed(
            directories["toe_trajectory"] / f"{well}.npz",
            row_idx=toe_rows,
            md=md[toe_rows].astype(np.float32),
            x=x[toe_rows].astype(np.float32),
            y=y[toe_rows].astype(np.float32),
            z=z[toe_rows].astype(np.float32),
            geometry_valid=(
                np.isfinite(md[toe_rows])
                & np.isfinite(x[toe_rows])
                & np.isfinite(y[toe_rows])
                & np.isfinite(z[toe_rows])
            ).astype(np.uint8),
            anchor_row=np.asarray([anchor_row], np.int32),
        )

        # GR retains NaN values and carries an explicit mask. row_idx and MD
        # align it exactly with both prefix and toe sections.
        np.savez_compressed(
            directories["horizontal_gr"] / f"{well}.npz",
            row_idx=row_idx,
            md=md.astype(np.float32),
            gr=gr.astype(np.float32),
            gr_valid=np.isfinite(gr).astype(np.uint8),
            is_prefix=(row_idx <= anchor_row).astype(np.uint8),
            anchor_row=np.asarray([anchor_row], np.int32),
        )

        wellhead_row = _finite_xy_row(x, y)
        toe_end_row = int(toe_rows[-1])
        well_rows.append(
            {
                "well": well,
                "source": str(source),
                "rows": len(frame),
                "prefix_rows": len(prefix_rows),
                "known_prefix_rows": int(prefix_valid.sum()),
                "prefix_internal_missing_rows": int((~prefix_valid).sum()),
                "toe_rows": len(toe_rows),
                "anchor_row": anchor_row,
                "anchor_md": float(md[anchor_row]),
                "anchor_tvt": float(tvt_input[anchor_row]),
                "wellhead_row": wellhead_row,
                "wellhead_x": float(x[wellhead_row]),
                "wellhead_y": float(y[wellhead_row]),
                "anchor_x": float(x[anchor_row]),
                "anchor_y": float(y[anchor_row]),
                "toe_end_row": toe_end_row,
                "toe_end_x": float(x[toe_end_row]),
                "toe_end_y": float(y[toe_end_row]),
                "gr_valid_rows": int(np.isfinite(gr).sum()),
                "gr_missing_fraction": float(np.mean(~np.isfinite(gr))),
                "md_monotonic": bool(np.all(np.diff(md[np.isfinite(md)]) >= 0)),
            }
        )
    except Exception as exc:
        failures.append({"well": well, "source": str(source), "reason": repr(exc)})

well_manifest = pd.DataFrame(well_rows).sort_values("well").reset_index(drop=True)
failure_report = pd.DataFrame(failures)
well_manifest.to_csv(directories["metadata"] / "well_manifest.csv", index=False)
well_manifest[
    [
        "well",
        "wellhead_row",
        "wellhead_x",
        "wellhead_y",
        "anchor_row",
        "anchor_x",
        "anchor_y",
        "toe_end_row",
        "toe_end_x",
        "toe_end_y",
    ]
].to_csv(OUT_DIR / "wellhead_xy.csv", index=False)
failure_report.to_csv(directories["metadata"] / "failures.csv", index=False)

if failures or len(well_manifest) != len(horizontal_files):
    raise RuntimeError(
        f"exported {len(well_manifest)} / {len(horizontal_files)} wells; "
        f"failures={len(failures)}"
    )

schema = {
    "heel_prefix_tvt/<well>.npz": {
        "row_idx": "int32, original horizontal-well row",
        "md": "float32",
        "tvt_input": "float32, known prefix TVT sequence",
        "tvt_valid": "uint8 mask",
        "anchor_row/anchor_md/anchor_tvt": "one-element arrays",
    },
    "toe_trajectory/<well>.npz": {
        "row_idx": "int32, rows strictly after anchor",
        "md/x/y/z": "float32 legal test-time trajectory inputs",
        "geometry_valid": "uint8 mask",
        "note": "No true TVT label is included.",
    },
    "horizontal_gr/<well>.npz": {
        "row_idx/md/gr": "full prefix+eval sequence",
        "gr_valid": "uint8 finite-GR mask",
        "is_prefix": "uint8; one through anchor row",
    },
    "wellhead_xy.csv": {
        "wellhead_x/y": "first finite XY pair",
        "anchor_x/y": "prediction-start anchor XY",
        "toe_end_x/y": "final trajectory XY",
    },
}
(OUT_DIR / "schema.json").write_text(
    json.dumps(schema, indent=2, ensure_ascii=True), encoding="utf-8"
)

manifest = {
    "bundle": "raw_single_well_inputs_v2",
    "data_root": str(DATA_ROOT),
    "source_split": "train",
    "wells": len(well_manifest),
    "sections": {
        "heel_prefix_tvt": len(list(directories["heel_prefix_tvt"].glob("*.npz"))),
        "toe_trajectory": len(list(directories["toe_trajectory"].glob("*.npz"))),
        "horizontal_gr": len(list(directories["horizontal_gr"].glob("*.npz"))),
        "wellhead_xy": len(well_manifest),
    },
    "rows": {
        "all": int(well_manifest["rows"].sum()),
        "prefix": int(well_manifest["prefix_rows"].sum()),
        "known_prefix": int(well_manifest["known_prefix_rows"].sum()),
        "toe": int(well_manifest["toe_rows"].sum()),
        "gr_valid": int(well_manifest["gr_valid_rows"].sum()),
    },
    "bundle_size_mb": _size(OUT_DIR) / 2**20,
    "elapsed_seconds": time.time() - started,
    "free_gb_after_bundle": shutil.disk_usage(OOF_ROOT).free / 2**30,
}
(OUT_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
)

if CREATE_ZIP:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(Path(OUT_DIR.name) / path.relative_to(OUT_DIR)))
    manifest["zip"] = str(ZIP_PATH)
    manifest["zip_size_mb"] = ZIP_PATH.stat().st_size / 2**20
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

print("\nEXPORT COMPLETE")
print(json.dumps(manifest, indent=2))
print("well manifest:", directories["metadata"] / "well_manifest.csv")
print("bundle:", OUT_DIR)
if CREATE_ZIP:
    print("zip:", ZIP_PATH)
