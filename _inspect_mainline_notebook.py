import json
import re
import sys
from pathlib import Path


path = Path(r"E:\fork-of-fork-of-fork-of-rogii-wellbore-geol-ecf6db (1).ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))
if len(sys.argv) > 1:
    for raw_index in sys.argv[1:]:
        index = int(raw_index)
        print(f"\n===== CELL {index} =====")
        print("".join(notebook["cells"][index].get("source", [])))
    raise SystemExit(0)

print("cells", len(notebook["cells"]))
for index, cell in enumerate(notebook["cells"]):
    source = "".join(cell.get("source", []))
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    definitions = re.findall(r"(?m)^def\s+([A-Za-z0-9_]+)", source)
    first = lines[0][:150] if lines else ""
    print(
        f"[{index}] {cell['cell_type']} chars={len(source)} "
        f"first={first} defs={','.join(definitions)}"
    )
