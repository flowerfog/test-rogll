"""Resume M342-V1 from either a V1 failure or a directly completed V74 cell."""

from pathlib import Path


source_path = Path("/root/autodl-tmp/main342_longcontext_k2_compatibility_v1.py")
source = source_path.read_text(encoding="utf-8")
runtime_marker = "V74 = _execute_v74()"
continuation_marker = 'k2 = P["_endpoint_table"]()[["id", "k2_parent"]]'
if runtime_marker not in source or continuation_marker not in source:
    raise RuntimeError("Corrected V1 resume markers not found")

# Load constants and helper definitions without executing V74 again.
prefix = source.split(runtime_marker, 1)[0]
exec(compile(prefix, str(source_path) + ":resume_prefix", "exec"), globals())

if "V74" not in globals():
    if "frame" not in globals() or "endpoint_predictions" not in globals():
        raise RuntimeError(
            "No completed V74 runtime is present. Run the corrected full V1 script."
        )
    V74 = {"frame": frame, "endpoint_predictions": endpoint_predictions}

P = _load_p1_definitions()
B = P["B"]
base, _, _, coverage = P["O"]["_prepare_inventory"]()

tail = continuation_marker + source.split(continuation_marker, 1)[1]
exec(compile(tail, str(source_path) + ":resume_tail", "exec"), globals())
