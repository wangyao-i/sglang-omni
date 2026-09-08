# SPDX-License-Identifier: Apache-2.0
"""Verify that an Omni process runs an installed, immutable wheel artifact.

This diagnostic intentionally loads no model weights and creates no device
context.  It is designed for isolated-hardware preflight checks where an
editable checkout must never be the serving import source.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.metadata
import json
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


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _record_hash_matches(
    distribution: importlib.metadata.Distribution,
    path: Path,
) -> bool:
    """Validate one installed file against the wheel RECORD digest."""
    root = Path(distribution.locate_file(""))
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return False
    for entry in distribution.files or ():
        if entry.as_posix() != relative:
            continue
        digest = getattr(entry, "hash", None)
        if digest is None or digest.mode != "sha256":
            return False
        actual = base64.urlsafe_b64encode(
            hashlib.sha256(path.read_bytes()).digest()
        ).decode("ascii").rstrip("=")
        return actual == digest.value
    return False


def collect_runtime_artifact_integrity(
    *,
    distribution_name: str = "sglang-omni",
    forbidden_root: Path | None = None,
) -> dict[str, Any]:
    """Return a path-free attestation for the loaded Omni runtime artifact."""
    distribution = importlib.metadata.distribution(distribution_name)
    install_root = Path(distribution.locate_file("")).resolve()
    encoder_module = importlib.import_module(
        "sglang_omni.models.qwen3_asr.encoder_cuda_graph"
    )
    worker_module = importlib.import_module("sglang_omni.model_runner.model_worker")
    encoder_path = Path(encoder_module.__file__).resolve()
    worker_path = Path(worker_module.__file__).resolve()
    runner_constants = getattr(
        encoder_module.Qwen3ASREncoderLayerStackGraphRunner.model_info.__code__,
        "co_consts",
        (),
    )
    worker_constants = getattr(
        worker_module.ModelWorker._encoder_cuda_graph_info.__code__,
        "co_consts",
        (),
    )
    forbidden = forbidden_root.resolve() if forbidden_root is not None else None

    report = {
        "schema_version": 1,
        "distribution": distribution.metadata["Name"],
        "distribution_version": distribution.version,
        "isolated_venv": Path(sys.prefix).resolve() != Path(sys.base_prefix).resolve(),
        "encoder_module_under_install_root": _is_under(encoder_path, install_root),
        "model_worker_module_under_install_root": _is_under(worker_path, install_root),
        "encoder_record_hash_matches": _record_hash_matches(distribution, encoder_path),
        "model_worker_record_hash_matches": _record_hash_matches(
            distribution, worker_path
        ),
        "runner_model_info_has_defer_provenance": all(
            field in runner_constants for field in _DEFER_FIELDS
        ),
        "model_worker_has_runtime_identity": "runtime_identity" in worker_constants,
        "encoder_module_outside_forbidden_root": (
            forbidden is None or not _is_under(encoder_path, forbidden)
        ),
        "model_worker_module_outside_forbidden_root": (
            forbidden is None or not _is_under(worker_path, forbidden)
        ),
    }
    report["valid"] = all(
        value
        for key, value in report.items()
        if key
        not in {
            "schema_version",
            "distribution",
            "distribution_version",
        }
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forbid-root",
        help="Checkout root that must not own loaded runtime modules.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect_runtime_artifact_integrity(
        forbidden_root=Path(args.forbid_root) if args.forbid_root else None
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
