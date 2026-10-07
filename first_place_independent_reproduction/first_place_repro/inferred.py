"""Only details not numerically disclosed by the writeup live here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InferredPFProfile:
    process_sigma_ft: float
    likelihood_scale_gr: float
    jump_probability: float
    jump_sigma_ft: float
    weight: float


INFERRED_PF_PROFILES = (
    InferredPFProfile(0.35, 5.0, 0.00004, 5.0, 0.20),
    InferredPFProfile(0.55, 6.0, 0.000075, 7.0, 0.35),
    InferredPFProfile(0.80, 8.0, 0.00012, 10.0, 0.30),
    InferredPFProfile(1.10, 11.0, 0.00020, 14.0, 0.15),
)

