# SPDX-License-Identifier: Apache-2.0
"""Launch Omni from an installed wheel without a checkout on ``sys.path``.

This is a narrow isolated-hardware bootstrap.  It deliberately runs before
``sglang_omni.cli`` is imported so a task-scoped wheel can coexist with the
server's existing editable development environment.  ``multiprocessing``
spawn children receive the parent's sanitized ``sys.path`` in their
preparation data, so the same invariant applies to model-worker processes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import sysconfig
from pathlib import Path
from typing import Sequence


_SITE_GUARD_MARKER = "# sglang-omni isolated-runtime site guard"
_SITE_GUARD_SOURCE = f'''{_SITE_GUARD_MARKER}
import os
import sys
from pathlib import Path

_root_text = os.environ.get("SGLANG_OMNI_FORBID_IMPORT_ROOT")
if _root_text:
    _root = Path(_root_text).resolve()
    _cwd = Path.cwd().resolve()
    def _under(_path):
        try:
            _path.resolve().relative_to(_root)
        except ValueError:
            return False
        return True
    sys.path[:] = [
        _entry
        for _entry in sys.path
        if not _under(_cwd if not _entry else Path(_entry))
    ]
'''


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def sanitize_import_path(forbidden_root: Path) -> list[str]:
    """Remove entries resolving inside ``forbidden_root`` from ``sys.path``.

    An empty path entry is the current working directory.  Refuse that case
    when the process was started from the checkout rather than silently
    continuing with ambiguous import precedence.
    """
    root = forbidden_root.resolve()
    cwd = Path.cwd().resolve()
    if _is_under(cwd, root):
        raise RuntimeError("isolated launch must start outside the checkout")

    retained: list[str] = []
    for entry in sys.path:
        resolved = cwd if not entry else Path(entry).resolve()
        if not _is_under(resolved, root):
            retained.append(entry)
    sys.path[:] = retained
    os.environ.pop("PYTHONPATH", None)
    return retained


def _site_guard_path(site_packages: Path | None = None) -> Path:
    if site_packages is None:
        site_packages = Path(sysconfig.get_path("purelib"))
    return site_packages / "sitecustomize.py"


def install_site_guard(site_packages: Path | None = None) -> Path:
    """Install a task-venv-only startup guard without replacing user content."""
    destination = _site_guard_path(site_packages)
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if _SITE_GUARD_MARKER not in existing:
            raise RuntimeError(
                "task venv already owns sitecustomize.py; create a fresh task venv"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_SITE_GUARD_SOURCE, encoding="utf-8")
    return destination


def _site_guard_installed() -> bool:
    path = _site_guard_path()
    return path.is_file() and _SITE_GUARD_MARKER in path.read_text(encoding="utf-8")


def collect_isolation_report(forbidden_root: Path) -> dict[str, bool]:
    """Return a path-free gate report after :func:`sanitize_import_path`."""
    root = forbidden_root.resolve()
    cwd = Path.cwd().resolve()
    has_checkout_entry = any(
        _is_under(cwd if not entry else Path(entry).resolve(), root)
        for entry in sys.path
    )
    package_path = Path(__file__).resolve()
    return {
        "isolated_interpreter": bool(sys.flags.isolated),
        "cwd_outside_forbidden_root": not _is_under(cwd, root),
        "checkout_absent_from_sys_path": not has_checkout_entry,
        "launcher_outside_forbidden_root": not _is_under(package_path, root),
        "site_guard_installed": _site_guard_installed(),
        "forbid_root_environment_matches": os.environ.get(
            "SGLANG_OMNI_FORBID_IMPORT_ROOT"
        )
        == str(root),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forbid-root", required=True)
    parser.add_argument(
        "--attest",
        action="store_true",
        help="Print the launch-path gate and exit without importing the CLI.",
    )
    parser.add_argument(
        "--install-site-guard",
        action="store_true",
        help="Install the task-venv site startup guard and exit.",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.install_site_guard:
        destination = install_site_guard()
        print(json.dumps({"site_guard_installed": True, "path_name": destination.name}))
        return 0
    sanitize_import_path(Path(args.forbid_root))
    report = collect_isolation_report(Path(args.forbid_root))
    report["valid"] = all(report.values())
    if args.attest:
        print(json.dumps(report, sort_keys=True))
        return 0 if report["valid"] else 1
    if not report["valid"]:
        raise RuntimeError("isolated launch gate failed")
    if not args.args or args.args[0] != "--":
        raise SystemExit("pass the sgl-omni command after '--'")

    sys.argv = ["sgl-omni", *args.args[1:]]
    from sglang_omni.cli import app

    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
