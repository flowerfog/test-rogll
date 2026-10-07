"""Build the self-contained AutoDL cell from the validated native builder."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "native_resolution_dualref_capacity_v0_cell.py"
ADDON = ROOT / "native_fullwell_surface_posterior_fold2_v0_addon.py"
OUTPUT = ROOT / "native_fullwell_surface_posterior_fold2_v0_cell.py"
MARKER = "# ---------------------------------------------------------------------------\n# Execute\n"

source = SOURCE.read_text(encoding="utf-8")
if MARKER not in source:
    raise RuntimeError(f"Cannot find execute boundary in {SOURCE}")
prefix = source.split(MARKER, 1)[0]
addon = ADDON.read_text(encoding="utf-8")
OUTPUT.write_text(prefix.rstrip() + "\n\n\n" + addon, encoding="utf-8")
print(OUTPUT)

