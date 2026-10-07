from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "native_resolution_dualref_capacity_v0_cell.py"
ADDON = ROOT / "native_fullwell_decomposed_episode_fold2_v1_addon.py"
OUTPUT = ROOT / "native_fullwell_decomposed_episode_fold2_v1_cell.py"
MARKER = "# ---------------------------------------------------------------------------\n# Execute\n"

source = BASE.read_text(encoding="utf-8")
if MARKER not in source:
    raise RuntimeError(f"Cannot locate execution marker in {BASE}")
prefix = source.split(MARKER, 1)[0]
addon = ADDON.read_text(encoding="utf-8")
OUTPUT.write_text(prefix.rstrip() + "\n\n" + addon.lstrip(), encoding="utf-8")
print(OUTPUT)
