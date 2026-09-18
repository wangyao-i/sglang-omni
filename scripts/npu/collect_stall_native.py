"""Bounded Linux native snapshot of ONE explicitly selected task-owned PID.

Run before terminating the hung service. GDB temporarily stops the target.
No device reset, service kill, package installation or privilege changes.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def wait_for_blocked_trace(pid, directory, timeout=600):
    """Observe metadata files only; start AFTER service readiness."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            raise RuntimeError("service exited before snapshot trigger")
        for path in directory.glob(f"{pid}-*.jsonl"):
            # Per-thread bounded diagnosis logs are small; tolerate a partial row.
            lines = path.read_bytes().splitlines()
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if row["phase"].endswith(".enter"):
                    age = (time.monotonic_ns() - row["ns"]) / 1e9
                    if age >= 20:
                        return row
                break
        time.sleep(1)
    raise TimeoutError("no blocked instrumented call observed within 600 seconds")


def collect(pid, out):
    proc = Path(f"/proc/{pid}")
    if pid <= 1 or not proc.is_dir():
        raise ValueError("require a live, explicitly selected service PID > 1")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("status", "maps", "stat"):
        (out / name).write_bytes((proc / name).read_bytes())
    rows = []
    for task in sorted((proc / "task").iterdir()):
        row = {"tid": int(task.name)}
        for name in ("comm", "wchan", "syscall"):
            try:
                row[name] = (task / name).read_text()
            except OSError as exc:
                row[name] = type(exc).__name__
        rows.append(row)
    (out / "tasks.json").write_text(json.dumps(rows, indent=2))
    commands = [
        ("python-stacks.txt", ["py-spy", "dump", "--pid", str(pid)]),
        (
            "native-stacks.txt",
            [
                "gdb",
                "-nx",
                "-nh",
                "-batch",
                "-iex",
                "set auto-load python-scripts off",
                "-iex",
                "set auto-load local-gdbinit off",
                "-iex",
                "set auto-load gdb-scripts off",
                "-iex",
                "set debuginfod enabled off",
                "-ex",
                "set pagination off",
                "-ex",
                "set print frame-arguments none",
                "-ex",
                f"attach {pid}",
                "-ex",
                "info threads",
                "-ex",
                "thread apply all bt 40",
                "-ex",
                "info sharedlibrary",
                "-ex",
                "detach",
            ],
        ),
    ]
    results = []
    for filename, command in commands:
        start = time.monotonic()
        with (out / filename).open("wb") as output:
            try:
                result = subprocess.run(
                    command,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=25,
                    check=False,
                    env={**os.environ, "DEBUGINFOD_URLS": ""},
                )
                status = str(result.returncode)
            except (OSError, subprocess.TimeoutExpired) as exc:
                status = type(exc).__name__
        results.append(
            dict(command=command, status=status, seconds=time.monotonic() - start)
        )
    (out / "commands.json").write_text(json.dumps(results, indent=2))
    # Record whether the tracer detached; operator must handle any stopped target.
    (out / "status-after").write_bytes((proc / "status").read_bytes())
    native = (out / "native-stacks.txt").read_text(errors="replace")
    detached = "TracerPid:\t0" in (out / "status-after").read_text()
    return all(row["status"] == "0" for row in results) and "#0" in native and detached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pid", type=int)
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--watch-trace", type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Linux /proc required")
    for tool in ("gdb", "py-spy"):
        if shutil.which(tool) is None:
            parser.error(f"missing {tool}; arrange tools before the hardware run")
    if args.watch_trace and args.preflight:
        parser.error("--watch-trace requires --pid")
    if args.preflight:
        # Disposable CPU-only Python process owned by this script.
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(90)"])
        try:
            time.sleep(0.5)
            ok = collect(child.pid, args.out)
        finally:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
    else:
        trigger = None
        if args.watch_trace:
            trigger = wait_for_blocked_trace(args.pid, args.watch_trace)
        ok = collect(args.pid, args.out)
        if trigger is not None:
            (args.out / "trigger.json").write_text(json.dumps(trigger, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
