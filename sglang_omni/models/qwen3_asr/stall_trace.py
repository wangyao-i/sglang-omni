"""Opt-in WF-009 diagnosis only; no tensor values or device synchronization."""

import json
import os
import threading
import time
from contextlib import contextmanager

TRACE_DIR = os.environ.get("SGLANG_ASR_STALL_TRACE_DIR")
_local = threading.local()
_installed = False


def emit(phase, **fields):
    if not TRACE_DIR:
        return
    # Reject unexpected objects instead of invoking repr/default=str in capture.
    if any(type(v) not in (str, int, float, bool, type(None)) for v in fields.values()):
        raise TypeError("stall trace fields must be scalar metadata")
    pid, tid = os.getpid(), threading.get_native_id()
    state = _local.__dict__
    if state.get("owner") != (pid, tid):
        os.makedirs(TRACE_DIR, exist_ok=True)
        state["fd"] = os.open(
            os.path.join(TRACE_DIR, f"{pid}-{tid}.jsonl"),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        state["owner"] = (pid, tid)
        state["seq"] = 0
    state["seq"] += 1
    row = dict(
        ns=time.monotonic_ns(),
        pid=pid,
        tid=tid,
        seq=state["seq"],
        phase=phase,
        thread=threading.current_thread().name,
        **fields,
    )
    data = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    if os.write(state["fd"], data) != len(data):
        raise OSError("short stall trace write")


@contextmanager
def span(phase, **fields):
    emit(phase + ".enter", **fields)
    try:
        yield
    except BaseException as exc:
        emit(phase + ".error", error_type=type(exc).__name__, **fields)
        raise
    else:
        emit(phase + ".exit", **fields)


def install():
    global _installed
    if not TRACE_DIR or _installed:
        return
    from torch_npu.npu import graphs

    _install_hooks(graphs)
    _installed = True
    emit("hooks.installed")


def _install_hooks(graphs):
    # Patch the Python module globals used by installed update_capture_record.
    # Never edit site-packages or inspect operator arguments/tensors.
    begin = graphs.graph_task_update_begin
    end = graphs.graph_task_update_end
    replay = graphs.NPUGraph.replay
    update = graphs.NPUGraph.update

    def traced_begin(stream, handle):
        _local.handle_pyid = id(handle)
        with span("task_begin", stream=int(stream.npu_stream), handle_pyid=id(handle)):
            return begin(stream, handle)

    def traced_end(stream):
        with span(
            "task_end",
            stream=int(stream.npu_stream),
            handle_pyid=_local.__dict__.get("handle_pyid"),
        ):
            return end(stream)

    def traced_replay(self, *args, **kwargs):
        with span("graph_replay", graph_pyid=id(self)):
            return replay(self, *args, **kwargs)

    def traced_update(self, *args, **kwargs):
        with span("graph_update", graph_pyid=id(self)):
            return update(self, *args, **kwargs)

    graphs.graph_task_update_begin = traced_begin
    graphs.graph_task_update_end = traced_end
    graphs.NPUGraph.replay = traced_replay
    graphs.NPUGraph.update = traced_update


@contextmanager
def transfer(embedding, target, device_module):
    if not TRACE_DIR:
        yield
        return
    emit("attach.metadata.enter")
    fields = dict(
        tensor_pyid=id(embedding),
        source_device=str(embedding.device),
        target_device=str(target),
        dtype=str(embedding.dtype),
        rank=embedding.dim(),
        elements=embedding.numel(),
    )
    emit("attach.metadata.exit", **fields)
    # Bracket device metadata queries separately: if one blocks, do not report
    # that as a blocked transfer. No set_device/stream changes are made.
    emit("attach.stream_query.enter")
    fields["current_device"] = int(device_module.current_device())
    fields["current_stream"] = int(device_module.current_stream().npu_stream)
    emit("attach.stream_query.exit", **fields)
    with span("attach.to", **fields):
        yield
