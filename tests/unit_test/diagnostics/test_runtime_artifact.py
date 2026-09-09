# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.diagnostics.runtime_artifact import (
    _constants_contain,
    _is_under,
    _record_hash_matches,
)
from sglang_omni.diagnostics.isolated_launch import (
    collect_isolation_report,
    sanitize_import_path,
)


def test_is_under_rejects_sibling_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    nested = root / "nested" / "module.py"
    nested.parent.mkdir()
    nested.touch()
    sibling = tmp_path / "root-other" / "module.py"
    sibling.parent.mkdir()
    sibling.touch()

    assert _is_under(nested, root)
    assert not _is_under(sibling, root)


def test_constants_contain_handles_compiler_folded_literal_tuple() -> None:
    source = "def report():\n    return {'defer': {'configured': 1, 'run_count': 0}}\n"
    module = compile(source, "report.py", "exec")
    report = next(item for item in module.co_consts if getattr(item, "co_name", None) == "report")

    assert _constants_contain(report.co_consts, "configured")
    assert _constants_contain(report.co_consts, "run_count")
    assert not _constants_contain(report.co_consts, "missing")


def test_record_hash_matches_installed_file(tmp_path: Path) -> None:
    package = tmp_path / "sglang_omni" / "module.py"
    package.parent.mkdir()
    package.write_bytes(b"immutable wheel payload\n")
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(package.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    entry = SimpleNamespace(
        as_posix=lambda: "sglang_omni/module.py",
        hash=SimpleNamespace(mode="sha256", value=digest),
    )
    distribution = SimpleNamespace(
        locate_file=lambda _path: tmp_path,
        files=[entry],
    )

    assert _record_hash_matches(distribution, package)
    package.write_bytes(b"modified payload\n")
    assert not _record_hash_matches(distribution, package)


def test_isolated_launch_removes_checkout_entries(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    outside = tmp_path / "runtime"
    outside.mkdir()
    monkeypatch.chdir(outside)
    monkeypatch.setattr(sys, "path", [str(checkout), str(outside), ""])

    retained = sanitize_import_path(checkout)

    assert retained == [str(outside), ""]
    report = collect_isolation_report(checkout)
    assert report["cwd_outside_forbidden_root"]
    assert report["checkout_absent_from_sys_path"]


def test_isolated_launch_rejects_checkout_working_directory(
    monkeypatch, tmp_path: Path
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)

    with pytest.raises(RuntimeError, match="outside the checkout"):
        sanitize_import_path(checkout)
