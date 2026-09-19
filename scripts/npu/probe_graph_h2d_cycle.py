"""Bounded NPU mechanism probe; --help and module import do not import torch."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback


def emit(phase, **fields):
    print(json.dumps(dict(phase=phase, pid=os.getpid(),
                          tid=threading.get_native_id(), ns=time.monotonic_ns(),
                          **fields)), flush=True)


def overlap(copy, update, delay, log=emit):
    """Update never depends on completion of copy; parent owns hard timeout."""
    entered = threading.Event()
    failures = []

    def run_copy():
        try:
            log("copy.enter")
            entered.set()
            copy()
            log("copy.return")
        except BaseException as exc:
            failures.append(exc)
            entered.set()

    worker = threading.Thread(target=run_copy, name="cycle-copy", daemon=True)
    worker.start()
    if not entered.wait(5):
        raise RuntimeError("copy thread did not start")
    # This forces a bounded scheduling window, NOT proof of native enqueue.
    time.sleep(delay)
    update()
    worker.join(5)
    if worker.is_alive():
        raise RuntimeError("copy still blocked after update returned")
    if failures:
        raise failures[0]


def child(arm):
    emit("import.enter", arm=arm)
    import torch
    import torch_npu

    torch.npu.set_device(0)
    torch.manual_seed(42)
    compute = torch.npu.Stream()
    update_stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    event = torch.npu.ExternalEvent()
    query = torch.randn(1, 2, 1, 64, device="npu:0", dtype=torch.float16)
    key = torch.randn(1, 2, 16, 64, device="npu:0", dtype=torch.float16)
    value = torch.randn_like(key)
    kwargs = dict(num_heads=2, input_layout="BNSD", scale=0.125,
                  pre_tokens=65535, next_tokens=65535, softmax_lse_flag=False,
                  actual_seq_lengths=[1], actual_seq_lengths_kv=[16])
    reference = torch_npu.npu_fused_infer_attention_score(query, key, value, **kwargs)
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        query, key, value, **kwargs)
    output = torch.empty_like(reference[0])
    lse = torch.empty_like(reference[1])
    source = torch.full((143360,), 0.25, dtype=torch.bfloat16)
    if arm == "pinned":
        source = source.pin_memory()
    destination = torch.empty_like(source, device="npu:0")
    torch.npu.synchronize()
    emit("identity", torch=torch.__version__, torch_npu=torch_npu.__version__,
         torch_npu_file=torch_npu.__file__, compute=compute.npu_stream,
         update=update_stream.npu_stream, source_pinned=source.is_pinned(),
         bytes=source.numel() * source.element_size(),
         queue_env={k: os.environ.get(k) for k in
                    ("TASK_QUEUE_ENABLE", "PER_STREAM_QUEUE", "ASCEND_LAUNCH_BLOCKING")})

    def operator(length):
        torch_npu.npu_fused_infer_attention_score.out(
            query, key, value, **{**kwargs, "actual_seq_lengths_kv": [length]},
            workspace=workspace, out=[output, lse])

    emit("capture.enter")
    with torch.npu.graph(graph, stream=compute):
        event.wait(compute)
        event.reset(compute)
        torch.npu.graph_task_group_begin(compute)
        operator(8)
        handle = torch.npu.graph_task_group_end(compute)
    emit("capture.return")

    def update():
        with torch.npu.stream(update_stream):
            emit("update.begin.enter")
            torch.npu.graph_task_update_begin(update_stream, handle)
            emit("update.begin.return")
            operator(16)
            emit("update.end.enter")
            torch.npu.graph_task_update_end(update_stream)
            emit("update.end.return")
            event.record(update_stream)
            emit("record.return")

    def replay():
        with torch.npu.stream(compute):
            graph.replay()
        emit("replay.return")

    def check():
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), reference[0].cpu(), rtol=0.01, atol=0.01)

    # All arms have the same legal, upstream-style initialization gate.
    update()
    replay()
    check()
    emit("ordered_gate.pass")
    # Warm the copy path before graph replay is outstanding; retain buffers.
    with torch.npu.stream(compute):
        destination.copy_(source, non_blocking=True)
    torch.npu.synchronize()
    # Preserve the submission protocol used in the failed candidate.
    update_stream.wait_stream(compute)
    replay()
    if arm == "gate":
        update()
    else:
        def copy():
            torch.npu.set_device(0)
            with torch.npu.stream(compute):
                destination.copy_(source, non_blocking=True)
        overlap(copy, update, delay=0.05)
    check()
    torch.testing.assert_close(destination.cpu(), source, rtol=0, atol=0)
    emit("probe.pass", arm=arm)


def stop_owned(proc):
    if proc.poll() is not None:
        return "already_exited"
    proc.terminate()
    try:
        proc.wait(timeout=5)
        return "terminated"
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        return "killed"


def snapshot_command(pid, helper):
    # -ex is a GDB command, not shell syntax. Python parses the path literal;
    # run_name preserves the helper's __main__ entry point and finally-detach.
    load = f"python import runpy; runpy.run_path({str(helper)!r}, run_name='__main__')"
    return ["gdb", "-nx", "-nh", "-batch", "-iex", "set auto-load off",
            "-ex", "set pagination off", "-ex", f"attach {pid}", "-ex", load]


def collect_snapshot(pid, tids, out):
    env = dict(os.environ, WF_QUEUE_SNAPSHOT=str(out / "queue.jsonl"),
               WF_QUEUE_TIDS=",".join(map(str, tids)), WF_QUEUE_SECONDS="15")
    helper = Path(__file__).with_name("queue_snapshot_gdb.py").resolve()
    command = snapshot_command(pid, helper)
    (out / "gdb-command.json").write_text(json.dumps(command), encoding="utf-8")
    rc = None
    error = None
    with (out / "gdb.txt").open("w", encoding="utf-8") as dbg:
        try:
            rc = subprocess.run(command, env=env, stdout=dbg,
                                stderr=subprocess.STDOUT, timeout=25).returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = type(exc).__name__
    rows = []
    try:
        rows = [json.loads(line) for line in (out / "queue.jsonl").read_text().splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        error = f"snapshot: {type(exc).__name__}"
    phases = {row.get("phase") for row in rows}
    status = Path(f"/proc/{pid}/status").read_text()
    (out / "status-after").write_text(status, encoding="utf-8")
    detached = "TracerPid:\t0" in status and not any(
        line.startswith("State:") and line.split()[1] in ("T", "t")
        for line in status.splitlines())
    required = {"maps", "identity", "inventory", "stack", "coverage"}
    captured = {row["tid"] for row in rows if row.get("phase") == "stack"
                and row.get("frames") and not row.get("error")}
    missing_requested = sorted(set(tids) - captured)
    ok = (rc == 0 and error is None and required <= phases
          and detached and not missing_requested)
    result = dict(ok=ok, rc=rc, error=error, phases=sorted(phases), detached=detached,
                  missing_requested=missing_requested)
    (out / "collector.json").write_text(json.dumps(result), encoding="utf-8")
    emit("collector.result", **result)
    return ok


def preflight(out):
    """Run the identical debugger invocation on an owned CPU-only child."""
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(90)"])
    try:
        ok = collect_snapshot(proc.pid, [proc.pid], out)
    finally:
        cleanup = stop_owned(proc)
        (out / "exit.json").write_text(json.dumps(dict(
            rc=proc.returncode, cleanup=cleanup)), encoding="utf-8")
    return 0 if ok else 1


def supervise(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--arm", args.arm]
    (out / "command.json").write_text(json.dumps(command), encoding="utf-8")
    with (out / "events.jsonl").open("w", encoding="utf-8") as log, \
            (out / "stderr.txt").open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(command, stdout=log, stderr=err)
        (out / "pid.txt").write_text(str(proc.pid), encoding="ascii")
        try:
            proc.wait(timeout=120)
            verdict = "completed" if proc.returncode == 0 else "error"
        except subprocess.TimeoutExpired:
            verdict = "timeout_unclassified"
            # Preserve maps even if debugger attachment is unavailable.
            (out / "maps").write_bytes(Path(f"/proc/{proc.pid}/maps").read_bytes())
            rows = []
            for line in (out / "events.jsonl").read_text().splitlines():
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and "tid" in row:
                        rows.append(row)
                except json.JSONDecodeError:
                    pass  # Keep original stdout, including library banners.
            tids = sorted({row["tid"] for row in rows})
            collect_snapshot(proc.pid, tids, out)
        finally:
            cleanup = stop_owned(proc)
            (out / "exit.json").write_text(json.dumps(dict(
                rc=proc.returncode, cleanup=cleanup)), encoding="utf-8")
    (out / "verdict.txt").write_text(verdict, encoding="ascii")
    return 0 if verdict == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("gate", "pinned", "pageable"))
    parser.add_argument("--preflight", action="store_true", help="CPU-only real GDB attach check")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.preflight:
        if args.arm or args.child or args.out is None or sys.platform != "linux":
            parser.error("--preflight requires Linux, --out, and no --arm/--child")
        return preflight(args.out)
    if args.arm is None:
        parser.error("--arm required for a workload")
    if args.child:
        try:
            child(args.arm)
        except BaseException:
            traceback.print_exc()
            return 1
        return 0
    if sys.platform != "linux" or args.out is None:
        parser.error("Linux and a new --out directory required")
    return supervise(args)


if __name__ == "__main__":
    sys.exit(main())
