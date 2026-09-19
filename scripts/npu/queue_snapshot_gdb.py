"""Source in an already-attached GDB; read-only, targeted-first snapshot.

Set WF_QUEUE_SNAPSHOT to a NEW output filename and WF_QUEUE_TIDS to a comma
separated list of fresh scheduler/copy LWPs. The caller owns timeout and detach.
No inferior function calls, writes, workload launch or device operations.
"""

import hashlib
import json
import os
import time
from pathlib import Path


LIB_SHA = "6f000ee3be32c15d062c9fc7a5c0889e9033e2b1959800608b7a3819249010f6"
WAIT_PCS = {0xCA54B0, 0xCA5524}


def ordered_threads(rows, requested):
    """Names identify candidates, NOT proven consumers; include every match."""
    def rank(row):
        if row["tid"] in requested:
            return 0
        if "acl" in row["name"].lower() or "release" in row["name"].lower():
            return 1
        return 2

    return sorted(rows, key=lambda row: (rank(row), row["tid"])), rank


def parse_tids(value):
    values = {int(item) for item in value.split(",") if item.strip()}
    if not values or min(values) <= 0:
        raise ValueError("supply fresh positive scheduler/copy LWPs")
    return values


def matching_library(maps):
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or int(fields[2], 16) != 0:
            continue
        if Path(fields[5]).name != "libtorch_npu.so":
            continue
        path = Path(fields[5])
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"path": str(path), "sha256": digest.hexdigest(),
                "base": int(fields[0].split("-")[0], 16)}
    return None


def repository_state(frame, inferior, library):
    # Only the two binary-verified AArch64 caller frames permit this layout.
    if not library or library["sha256"] != LIB_SHA:
        return None
    if "aarch64" not in frame.architecture().name().lower():
        return None
    if frame.pc() - library["base"] not in WAIT_PCS:
        return None
    obj = int(frame.read_register("x19"))
    result = {"repository_candidate": hex(obj), "mutex_address": hex(obj + 0x60),
              "provenance": "unwound x19 at binary-matched wait return PC"}
    for name, offset, size in [("read_idx", 0x4C, 4), ("write_idx", 0x54, 4),
                               ("need_empty", 0x5C, 1), ("status", 0x58, 4),
                               ("efd_read", 0x18, 4), ("efd_empty", 0x20, 4)]:
        result[name] = int.from_bytes(bytes(inferior.read_memory(obj + offset, size)), "little")
    result["mutex_owner"] = "unknown: libc mutex layout not assumed"
    return result


def snapshot(gdb):
    requested = parse_tids(os.environ.get("WF_QUEUE_TIDS", ""))
    output = Path(os.environ["WF_QUEUE_SNAPSHOT"])
    budget = float(os.environ.get("WF_QUEUE_SECONDS", "15"))
    if not 0 < budget <= 30:
        raise ValueError("budget must be in (0, 30]")
    inferior = gdb.selected_inferior()
    proc = Path(f"/proc/{inferior.pid}")
    # Exclusive creation prevents destruction of earlier evidence.
    with output.open("x", encoding="utf-8") as destination:
        def emit(row):
            destination.write(json.dumps(row) + "\n")
            destination.flush()

        maps = (proc / "maps").read_text()
        library = matching_library(maps)
        emit({"phase": "identity", "pid": inferior.pid, "library": library})
        env = {}
        allowed = {"TASK_QUEUE_ENABLE", "PER_STREAM_QUEUE", "ASCEND_LAUNCH_BLOCKING"}
        for item in (proc / "environ").read_bytes().split(b"\0"):
            key, _, value = item.partition(b"=")
            if key.decode(errors="replace") in allowed:
                env[key.decode()] = value.decode(errors="replace")
        emit({"phase": "raw_environment", "values": env,
              "effective_values": "unknown; raw unset is not effective zero"})
        threads = {thread.ptid[1]: thread for thread in inferior.threads()}
        rows = []
        for tid, thread in threads.items():
            try:
                name = (proc / "task" / str(tid) / "comm").read_text().strip()
            except OSError:
                name = thread.name or "unknown"
            rows.append({"tid": tid, "name": name})
        rows, rank = ordered_threads(rows, requested)
        emit({"phase": "inventory", "threads": rows,
              "missing_requested": sorted(requested - threads.keys())})
        started = time.monotonic()
        collected = []
        for row in rows:
            if time.monotonic() - started >= budget:
                break
            threads[row["tid"]].switch()
            item = {**row, "phase": "stack", "priority": rank(row), "frames": []}
            try:
                frame = gdb.newest_frame()
                limit = 60 if rank(row) < 2 else 8
                for _ in range(limit):
                    if frame is None:
                        break
                    entry = {"pc": hex(frame.pc()), "name": frame.name()}
                    try:
                        entry["repository"] = repository_state(frame, inferior, library)
                    except Exception as exc:
                        entry["repository_error"] = str(exc)
                    item["frames"].append(entry)
                    frame = frame.older()
                item["stack_truncated"] = frame is not None
            except Exception as exc:
                item["error"] = str(exc)
            emit(item)
            collected.append(row["tid"])
        emit({"phase": "coverage", "collected": collected,
              "missing": sorted(threads.keys() - set(collected)),
              "note": "absence from captured stacks is not proof of absence"})


if __name__ == "__main__":
    import gdb

    try:
        snapshot(gdb)
    finally:
        # Outer runner must still verify TracerPid and stopped state after timeout.
        gdb.execute("detach")
