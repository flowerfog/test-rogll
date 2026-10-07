from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
AUTHOR_SRC = PACKAGE_ROOT / "author_src"

REFERENCE_FILES = {
    "cfg.pkl": (16473, "ED99CDED75DBD3BEC8AB29D450238A1134EA8C28CD8DC1A0EA34E67B575DA985"),
    "models.pkl": (3220117304, "677475C3D8F1E0C4A18BCA9381469CD466F79CA0A3E0FA8D1686C74932704F45"),
    "oof_df.pqt": (131989244, "3D60B30E4AC4BB98F32BF9BA3C5CC0915334D8A30D9FF4C993AA36BA7D780EF8"),
    "seq_nn.log": (113385, "C18164818FFCD8DEC78136DAA1E2F33940B397BA5BFA8BBBFDDE398F90B7A15E"),
    "submission.csv": (424408, "256A6F07A99D54AF18C8AFA7C0B7E296BEC0380F234482A25A7EF6CBCB7B73BC"),
    "submission_details.pqt": (1030549, "05B47E5E4F43DAD439CB521EF6DCF3EE421A935BC0D2DA8F076DCF2B716F708E"),
}

AUTHOR_SOURCE_HASHES = {
    "seq_NN_cfg.py": "8FAFDE9D2D24D6B6747D9DC6D823614C9BDF14D2B642DD641B4F38883F0663E9",
    "seq_NN_dataset.py": "22CE505940CBEB366D17CFC0F19C33624D98221C56675A7F3E2D93D86BA1D22E",
    "seq_NN_data_prep.py": "CBBBA8E8DBEB5FA097996CAE9F874653BE411179D5AD73571AB9CAC20A86A644",
    "seq_NN_data_prep_pf_opt.py": "93E92EABFE6E47D0EA1F2DD87841952139FD2D0E3FED3577A4B689418150A2CD",
    "seq_NN_geo_prior.py": "B61C6351041D5D2DAE2A7861530F933E3B102E1C6D80D3FE2F4A61FA1F0368EF",
    "seq_NN_main.py": "478B6FF4B4898F20963F2463E3D6A7502033DBBE30B74081CF1523E082B8CC54",
    "seq_NN_models.py": "A8965834E614D0DCBEB5FD98976A79CD1E2C52463A0E9EBC760AB2669F6014BE",
    "seq_NN_pretrained_unet.py": "40F12951E804B103142EBB244172F2A1762D1175A90F0F95E9AD1FB3CFF69F58",
    "seq_NN_train.py": "EBAA2A41B4CB7460EA246A0F7049D7F1F2613BF2AA741BA878FEC260F85D80C9",
}

REFERENCE_OOF_RMSE = 5.0910

VERSIONS = {
    "0719_V1": {"branch": "default", "general": 0.07, "no_xy": 0.40, "cv": 5.09, "status": "literal"},
    "0729_V3": {"branch": "pf", "general": 0.00, "no_xy": 0.20, "cv": 5.53, "status": "retrain"},
    "0801_V1": {"branch": "default", "general": 0.07, "no_xy": 0.40, "cv": 5.16, "status": "retrain"},
    "0724_V1": {"branch": "xy", "general": 0.28, "no_xy": 0.00, "cv": 4.86, "status": "retrain"},
    "0801_V2": {"branch": "xy", "general": 0.28, "no_xy": 0.00, "cv": 4.80, "status": "retrain"},
    "0803_V2": {"branch": "xy_pf", "general": 0.28, "no_xy": 0.00, "cv": 5.00, "status": "retrain"},
}

BASE_CHANNELS = (
    "tw_gr", "gr", "gr_abs_diff", "gr_isnan_rate", "gr_std",
    "gr_first_last_delta", "gr_slope", "gr_quadratic_a",
    "gr_quadratic_b", "gr_quadratic_c", "gr_quadratic_rmse",
    "tw_gr_is_nan", "tw_tvt_rel", "seen_tvt_rel",
    "tw_seen_tvt_abs_diff", "z_diff",
)

# These names map the write-up's disclosed information directly to the
# implementations present in the supplied author source.  The five later
# version-specific cfg pickles were not supplied, so their augmentation deltas
# remain explicitly source-derived rather than literal.
XY_CHANNELS = ("geo_tvt_diff", "geo_tvt_abs_diff", "geo_tvt_rel")
PF_CHANNELS = ("pf_particle_density_prob", "pf_prob_ffbsi")

QUALITY_COLUMNS = (
    "mean_neighbor_distance",
    "p10_neighbor_distance",
    "prefix_visible_neighbor_weight_ratio",
    "query_center_distance_ratio",
)

VERSION_CHANNELS = {
    "0719_V1": BASE_CHANNELS,
    "0729_V3": BASE_CHANNELS + PF_CHANNELS,
    "0801_V1": BASE_CHANNELS,
    "0724_V1": tuple(name for name in BASE_CHANNELS if name != "z_diff") + XY_CHANNELS,
    "0801_V2": tuple(name for name in BASE_CHANNELS if name != "z_diff") + XY_CHANNELS,
    # This ordering is the known 0803_V2 20-channel contract.
    "0803_V2": (
        *tuple(name for name in BASE_CHANNELS if name != "z_diff"),
        "geo_tvt_diff", "pf_particle_density_prob", "pf_prob_ffbsi",
        "geo_tvt_abs_diff", "geo_tvt_rel",
    ),
}
