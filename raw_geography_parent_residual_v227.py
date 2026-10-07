"""V227: add raw absolute geography to the frozen V224 experiment.

The V215 feature cache does not expose raw X/Y/MD.  V227 preserves V224's
folds, target, model and recipe selection, changing only the addition of raw
map position, anchor position and relative trajectory coordinates.
"""

from pathlib import Path


ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()
SOURCE = ROOT / "strong_parent_conditional_residual_v224.py"
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)

source = SOURCE.read_text(encoding="utf-8")
source = source.replace("strong_parent_conditional_residual_v224", "raw_geography_parent_residual_v227")
source = source.replace("V224", "V227")
source = source.replace("SEED = 224", "SEED = 227")
source = source.replace(
    'ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()\n',
    'ROOT = Path("/root/autodl-tmp") if Path("/root/autodl-tmp").is_dir() else Path.cwd()\n'
    'DATA_ROOT = ROOT / "rogii-wellbore-geology-prediction"\n',
    1,
)

old_features = "features = base_features + spatial_features + derived_features\n"
new_features = '''raw_geo_features = [
    "raw_md", "raw_x", "raw_y", "raw_z", "raw_anchor_x", "raw_anchor_y",
    "raw_anchor_z", "raw_rel_x", "raw_rel_y", "raw_rel_z", "raw_xy_radius",
    "raw_rel_xy_radius", "raw_azimuth_sin", "raw_azimuth_cos",
    "v168_surface_z",
]
features = base_features + spatial_features + derived_features + raw_geo_features
'''
if old_features not in source:
    raise RuntimeError("V227 feature marker not found")
source = source.replace(old_features, new_features, 1)

old_load = '''def _load_one(well, full):
    base = pd.read_pickle(_paths(BASE, well, full))
    spatial = pd.read_pickle(_paths(SPATIAL, well, full))
'''
new_load = '''def _load_one(well, full):
    base = pd.read_pickle(_paths(BASE, well, full))
    spatial = pd.read_pickle(_paths(SPATIAL, well, full))
    horizontal_path = DATA_ROOT / "train" / f"{well}__horizontal_well.csv"
    horizontal = pd.read_csv(horizontal_path, usecols=["MD", "X", "Y", "Z", "TVT_input"])
    row_index = base["_row_idx"].to_numpy(np.int64)
    if row_index.min() < 0 or row_index.max() >= len(horizontal):
        raise RuntimeError(f"{well}: raw geography row alignment failed")
    known = np.flatnonzero(horizontal["TVT_input"].notna().to_numpy())
    if len(known) == 0:
        raise RuntimeError(f"{well}: no visible anchor")
    anchor_index = int(known[-1])
    raw_md = horizontal["MD"].to_numpy(np.float32)[row_index]
    raw_x = horizontal["X"].to_numpy(np.float32)[row_index]
    raw_y = horizontal["Y"].to_numpy(np.float32)[row_index]
    raw_z = horizontal["Z"].to_numpy(np.float32)[row_index]
    anchor_x = np.float32(horizontal["X"].iloc[anchor_index])
    anchor_y = np.float32(horizontal["Y"].iloc[anchor_index])
    anchor_z = np.float32(horizontal["Z"].iloc[anchor_index])
    rel_x = raw_x - anchor_x
    rel_y = raw_y - anchor_y
    azimuth = np.arctan2(rel_y, rel_x)
    base["raw_md"] = raw_md
    base["raw_x"] = raw_x
    base["raw_y"] = raw_y
    base["raw_z"] = raw_z
    base["raw_anchor_x"] = anchor_x
    base["raw_anchor_y"] = anchor_y
    base["raw_anchor_z"] = anchor_z
    base["raw_rel_x"] = rel_x
    base["raw_rel_y"] = rel_y
    base["raw_rel_z"] = raw_z - anchor_z
    base["raw_xy_radius"] = np.hypot(raw_x, raw_y).astype(np.float32)
    base["raw_rel_xy_radius"] = np.hypot(rel_x, rel_y).astype(np.float32)
    base["raw_azimuth_sin"] = np.sin(azimuth).astype(np.float32)
    base["raw_azimuth_cos"] = np.cos(azimuth).astype(np.float32)
'''
if old_load not in source:
    raise RuntimeError("V227 load marker not found")
source = source.replace(old_load, new_load, 1)

old_parent = '    base["v168_delta"] = base["v168"].to_numpy(np.float32) - last\n'
new_parent = (
    '    base["v168_delta"] = base["v168"].to_numpy(np.float32) - last\n'
    '    base["v168_surface_z"] = base["v168"].to_numpy(np.float32) + base["raw_z"].to_numpy(np.float32)\n'
)
if old_parent not in source:
    raise RuntimeError("V227 parent marker not found")
source = source.replace(old_parent, new_parent, 1)

source = source.replace(
    '"V168 parent state plus parent-to-trajectory disagreement features"',
    '"V168-conditioned evidence plus raw absolute X/Y/MD and anchor geography"',
)
source = source.replace(
    '"PASS -> build exact V168-conditioned Kaggle inference"',
    '"PASS -> retain raw geography and build exact Kaggle inference"',
)
source = source.replace(
    '"FAIL -> current legal evidence cannot explain strong-parent residual; stop rowwise GBDT"',
    '"FAIL -> raw absolute geography cannot explain V168 residual; close regional rowwise GBDT"',
)

code = compile(source, str(SOURCE), "exec")
exec(code, {"__name__": "__main__", "__file__": str(SOURCE)})
