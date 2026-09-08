# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace

from sglang_omni.diagnostics.runtime_artifact import _is_under, _record_hash_matches


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
