# First-Place Clean-Room Reproduction Matrix

## Contract

- Reproduction level: **behavioral reproduction**.
- Evidence boundary: first-place writeup, `submit-reproduce.ipynb` logs/configuration, and competition data only.
- No author source, packaged models, private XY fold map, or released implementation is read.
- Publicly disclosed behavior is fixed. Values not disclosed are isolated as `INFERRED` in code.

| Component | Status | Implementation |
|---|---|---|
| 2D alignment, anchor-relative target | source-derived | `data.py` |
| 400 x 345 grid, 0.5 ft state step, 32-row bins | exact | `config.py`, `data.py` |
| Seven version channel tuples | exact from notebook logs | `config.py` |
| Visible-region typewell calibration | inferred blend rule | `data.py` |
| ConvNeXt-small U-Net, BN replacement, pool stem and non-param resize | source-derived architecture; projection/decoder wiring inferred | `model.py` |
| Exponential target + CE/Huber/GR loss | source-derived equations/weights; Huber delta inferred as 1 | `loss.py` |
| BF16, batch 16, AdamW, cosine, EMA .99 | exact | `train.py` |
| Geo-stratified five folds x three seeds | source-derived 20-cluster/seed contract; coordinate map reconstruction inferred | `prepare.py`, `train.py` |
| Z-shift block bootstrap, faults, TVT_bias_v2, GR transform, masks | source-derived parameters; noise plumbing/window conversion inferred; PF/XY maps remapped into each simulated path | `data.py` |
| PF 500 particles x 128 seeds, 64-row update, jumps, FFBSi | source-derived; diverse profile values inferred | `pf.py` |
| XY anisotropic IDW + robust local WLS | source-derived parameters; local-difference and accumulated-TVT channels; singular fallback inferred; outer-fold-clean support | `xy.py`, `prepare.py` |
| Savitzky-Golay decoding | exact windows/blends | `infer.py` |
| Fixed seven-version weights | exact | `config.py`, `infer.py` |
| Four-threshold XY hard gate | exact from inference notebook | `xy.py`, `infer.py` |
| Original `train_png_typewell_map.csv` coordinates | author artifact unavailable | mean well XY reconstruction is implemented and isolated in `prepare.py` |
| Reference OOF 4.627 | reported target, not yet validated | full AutoDL training required |

This directory must not be described as a literal reproduction. A large metric gap triggers a representation audit before tuning.
