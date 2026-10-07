"""Read-only attribution after V40; requires the completed V40 live namespace."""

from pathlib import Path
import json
import time

import pandas as pd


REQUIRED = {
    "V", "records", "packs", "generic_reference", "confirmation_wells",
    "fold2_wells", "PROFILES", "OOF_ROOT",
}
missing = sorted(REQUIRED.difference(globals()))
if missing:
    raise RuntimeError(
        f"Missing live V40 prerequisites: {missing}. Run "
        "unsigned_boundary_spacing_v40_cell.py in this kernel first."
    )

STARTED_V40B = time.time()
OUT_DIR_V40B = Path(OOF_ROOT) / "unsigned_boundary_attribution_v40b"
OUT_DIR_V40B.mkdir(parents=True, exist_ok=True)

profile_by_name = {profile["name"]: profile for profile in PROFILES}
recipes = [
    {
        "profile": "unsigned_loc", "mode": "baseline",
        "exact_dose": 0.0, "generic_dose": 0.0,
    },
    {
        "profile": "unsigned_loc", "mode": "location_joint",
        "exact_dose": 0.1, "generic_dose": 0.1,
    },
    {
        "profile": "unsigned_space20", "mode": "spacing_exact",
        "exact_dose": 0.1, "generic_dose": 0.0,
    },
    {
        "profile": "unsigned_space20", "mode": "spacing_generic",
        "exact_dose": 0.0, "generic_dose": 0.1,
    },
    {
        "profile": "unsigned_space20", "mode": "spacing_joint",
        "exact_dose": 0.1, "generic_dose": 0.1,
    },
]

rows = []
for stage, wells in (
    ("fold3_confirmation", confirmation_wells),
    ("fold2_application", fold2_wells),
):
    for recipe in recipes:
        profile = profile_by_name[recipe["profile"]]
        rows.append(
            {
                "stage": stage,
                **V["_evaluate"](
                    wells, records, packs, generic_reference, profile, recipe
                ),
            }
        )

result = pd.DataFrame(rows)
result.to_csv(OUT_DIR_V40B / "matched_attribution.csv", index=False)


def metric(stage, mode, column):
    return float(
        result.loc[
            result["stage"].eq(stage) & result["mode"].eq(mode), column
        ].iloc[0]
    )


attribution = {}
for stage in ("fold3_confirmation", "fold2_application"):
    base = metric(stage, "baseline", "top32_recall")
    location = metric(stage, "location_joint", "top32_recall")
    exact = metric(stage, "spacing_exact", "top32_recall")
    generic = metric(stage, "spacing_generic", "top32_recall")
    joint = metric(stage, "spacing_joint", "top32_recall")
    location_deep = metric(stage, "location_joint", "deep_top32_recall")
    joint_deep = metric(stage, "spacing_joint", "deep_top32_recall")
    attribution[stage] = {
        "baseline_top32": base,
        "location_joint_gain": location - base,
        "spacing_joint_gain": joint - base,
        "spacing_increment_over_location": joint - location,
        "exact_only_gain": exact - base,
        "generic_only_gain": generic - base,
        "exact_increment_over_generic": joint - generic,
        "spacing_deep_increment_over_location": joint_deep - location_deep,
    }

gates = {
    "spacing_increment_positive_fold3": bool(
        attribution["fold3_confirmation"]["spacing_increment_over_location"] > 0.002
    ),
    "spacing_increment_positive_fold2": bool(
        attribution["fold2_application"]["spacing_increment_over_location"] > 0.002
    ),
    "spacing_deep_increment_positive_fold2": bool(
        attribution["fold2_application"]["spacing_deep_increment_over_location"] > 0.002
    ),
    "exact_identity_increment_fold2_ge_0p002": bool(
        attribution["fold2_application"]["exact_increment_over_generic"] >= 0.002
    ),
}
thickness_pass = bool(
    gates["spacing_increment_positive_fold3"]
    and gates["spacing_increment_positive_fold2"]
    and gates["spacing_deep_increment_positive_fold2"]
)
verdict_v40b = {
    "experiment": "unsigned_boundary_attribution_v40b",
    "stage": "read_only_location_spacing_reference_identity_attribution",
    "selection_or_tuning_count": 0,
    "fold1_touched": False,
    "attribution": attribution,
    "gates": gates,
    "thickness_pass": thickness_pass,
    "elapsed_minutes": (time.time() - STARTED_V40B) / 60.0,
    "next": (
        "PASS -> build explicit ordered thickness transition"
        if thickness_pass
        else "FAIL -> retain boundary location only as a candidate feature; close thickness DP"
    ),
}
(OUT_DIR_V40B / "verdict.json").write_text(
    json.dumps(verdict_v40b, indent=2), encoding="utf-8"
)
print("\nV40B MATCHED ATTRIBUTION")
print(result.to_string(index=False))
print("\nFINAL VERDICT", json.dumps(verdict_v40b, indent=2), flush=True)
print("saved:", OUT_DIR_V40B, flush=True)
