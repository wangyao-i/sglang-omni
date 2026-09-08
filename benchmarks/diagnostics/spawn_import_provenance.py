"""Compare Qwen3-ASR runtime imports across a multiprocessing spawn boundary.

This is a no-NPU diagnostic: it constructs neither a model nor a graph.  It
answers whether a fresh ``spawn`` child resolves the same source-defined
encoder runner and model-worker methods as its launcher.
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any


_DEFER_FIELDS = (
    "configured",
    "deferred_count",
    "remaining",
    "run_count",
    "first_capture",
)


def _is_under(path: str | None, root: Path) -> bool:
    if path is None:
        return False
    try:
        Path(path).resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _method_constants(method: Any) -> tuple[Any, ...]:
    return tuple(getattr(getattr(method, "__code__", None), "co_consts", ()))


def collect_provenance(expected_root: Path) -> dict[str, Any]:
    """Return path-free, JSON-safe provenance for the active code objects."""
    encoder_module = importlib.import_module(
        "sglang_omni.models.qwen3_asr.encoder_cuda_graph"
    )
    worker_module = importlib.import_module("sglang_omni.model_runner.model_worker")
    runner_type = encoder_module.Qwen3ASREncoderLayerStackGraphRunner
    runner_constants = _method_constants(runner_type.model_info)
    worker_constants = _method_constants(worker_module.ModelWorker._encoder_cuda_graph_info)
    encoder_path = getattr(encoder_module, "__file__", None)
    encoder_cache = getattr(encoder_module, "__cached__", None)
    worker_path = getattr(worker_module, "__file__", None)
    worker_cache = getattr(worker_module, "__cached__", None)
    pycache_prefix = Path(sys.pycache_prefix).resolve() if sys.pycache_prefix else None

    return {
        "pid": os.getpid(),
        "encoder_module_under_expected_root": _is_under(encoder_path, expected_root),
        "model_worker_module_under_expected_root": _is_under(worker_path, expected_root),
        "encoder_cache_under_pycache_prefix": bool(
            pycache_prefix and _is_under(encoder_cache, pycache_prefix)
        ),
        "model_worker_cache_under_pycache_prefix": bool(
            pycache_prefix and _is_under(worker_cache, pycache_prefix)
        ),
        "runner_model_info_has_defer_provenance": all(
            field in runner_constants for field in _DEFER_FIELDS
        ),
        "model_worker_has_runtime_identity": "runtime_identity" in worker_constants,
        "pycache_prefix_is_set": pycache_prefix is not None,
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
        "defer_env": os.environ.get("SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES"),
    }


def _child_main(queue: multiprocessing.queues.Queue, expected_root: str) -> None:
    queue.put(collect_provenance(Path(expected_root).resolve()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-root",
        required=True,
        help="Checkout root expected to own both imported modules.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_root = Path(args.expected_root).resolve()
    parent = collect_provenance(expected_root)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=_child_main, args=(queue, str(expected_root)))
    child.start()
    child_result = queue.get(timeout=30)
    child.join(timeout=30)
    if child.is_alive():
        child.terminate()
        child.join(timeout=5)
        raise RuntimeError("spawn child did not exit within 30 seconds")
    if child.exitcode != 0:
        raise RuntimeError(f"spawn child exited with code {child.exitcode}")

    result = {"parent": parent, "spawn_child": child_result}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
