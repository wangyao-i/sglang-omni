"""Read-only source inventory for the Qwen3-ASR NPU graph validation task.

Run this with the service Python. It does not import torch or initialize an NPU.
"""

import ast
import hashlib
import importlib.machinery
import importlib.metadata
import json
from pathlib import Path


def main():
    result = {}
    targets = {
        "sglang": [
            "srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py"
        ],
        "sglang_omni": [
            "platforms/device_graph.py",
            "models/qwen3_asr/engine_builder.py",
            "models/qwen3_asr/encoder_service.py",
        ],
        "torch_npu": ["npu/graphs.py"],
    }
    for package, files in targets.items():
        spec = importlib.machinery.PathFinder.find_spec(package)
        entry = {"found": spec is not None, "files": {}}
        result[package] = entry
        try:
            entry["distribution_version"] = importlib.metadata.version(
                package.replace("_", "-")
            )
        except importlib.metadata.PackageNotFoundError:
            entry["distribution_version"] = None
        if spec is None or not spec.submodule_search_locations:
            continue
        root = Path(next(iter(spec.submodule_search_locations)))
        for relative in files:
            path = root / relative
            if not path.is_file():
                entry["files"][relative] = {"exists": False}
                continue
            raw = path.read_bytes()
            source = raw.decode("utf-8-sig")
            tree = ast.parse(source)
            functions = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                    "replay",
                    "replay_with_input_update",
                    "update_capture_record",
                    "__new__",
                }:
                    segment = ast.get_source_segment(source, node)
                    functions[f"{node.name}:{node.lineno}"] = {
                        "sha256": hashlib.sha256(segment.encode()).hexdigest(),
                        "calls": sorted(
                            {
                                ast.unparse(candidate.func)
                                for candidate in ast.walk(node)
                                if isinstance(candidate, ast.Call)
                            }
                        ),
                    }
            entry["files"][relative] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "functions": functions,
                "class_update_stream_assignment": any(
                    isinstance(node, ast.ClassDef)
                    and any(
                        isinstance(statement, ast.Assign)
                        and any(
                            isinstance(target, ast.Name)
                            and target.id == "update_stream"
                            for target in statement.targets
                        )
                        for statement in node.body
                    )
                    for node in ast.walk(tree)
                ),
            }
    result["limitation"] = (
        "Source resolution only; confirm imported paths and effective config in "
        "retained service evidence. No workload run."
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
