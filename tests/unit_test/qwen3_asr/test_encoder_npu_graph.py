# SPDX-License-Identifier: Apache-2.0
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_asr.encoder_cuda_graph import (
    EncoderGraphUnrecoverableError,
    NpuGraphCaptureAttention,
    NpuGraphCaptureContext,
    NpuGraphUpdateTask,
    Qwen3ASREncoderLayerStackGraphRunner,
)


class _NpuBackend:
    supports_graph_task_update = True

    def replay(self, graph):
        graph.replay()


class _NpuDeviceModule:
    def current_stream(self, device=None):
        return "encoder-compute"

    def stream(self, stream):
        return nullcontext()


def _npu_runner():
    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._max_seqlen = 8
    runner._buckets = (8,)
    runner._failed = set()
    runner._graphs = {}
    runner._capture_failed = False
    runner._graph_submission_unrecoverable = False
    runner._device = SimpleNamespace(type="npu", index=0)
    runner._graph_backend = _NpuBackend()
    runner._device_module = _NpuDeviceModule()
    runner._npu_update_stream = SimpleNamespace(wait_stream=lambda stream: None)
    return runner


def test_npu_capture_waits_for_a_real_window_signature():
    runner = _npu_runner()
    runner._buckets = (128, 256)

    runner.capture_all()

    assert runner._graphs == {}
    assert runner._failed == set()


def test_npu_replay_updates_window_boundaries_for_a_reused_bucket():
    captured = []
    operations = []
    runner = _npu_runner()
    runner.plan = lambda total, windows: (8, [8 - total])
    runner._npu_update_stream.wait_stream = lambda stream: operations.append(
        ("wait", stream)
    )

    def capture(bucket_size, *, window_lens=None):
        captured.append((bucket_size, window_lens))
        task = SimpleNamespace(
            apply=lambda device_module, update_stream, boundaries: operations.append(
                ("update", boundaries)
            )
        )
        return SimpleNamespace(
            hidden_states=torch.zeros(bucket_size, 2),
            graph=SimpleNamespace(
                replay=lambda: operations.append(("replay", bucket_size))
            ),
            output=torch.zeros(bucket_size, 2),
            npu_update_tasks=(task,),
        )

    runner.capture = capture

    assert runner.run(torch.ones(4, 2), [4]) is not None
    assert runner.run(torch.ones(3, 2), [3]) is not None
    assert captured == [(8, (4, 4))]
    assert operations == [
        ("wait", "encoder-compute"),
        ("replay", 8),
        ("update", [4, 8]),
        ("wait", "encoder-compute"),
        ("replay", 8),
        ("update", [3, 8]),
    ]


def test_npu_graph_update_rebinds_fia_lengths_on_the_runner_stream():
    calls = []

    class DeviceModule:
        def graph_task_update_begin(self, stream, handle):
            calls.append(("begin", stream, handle))

        def graph_task_update_end(self, stream):
            calls.append(("end", stream))

    task = NpuGraphUpdateTask(
        operation=lambda **kwargs: calls.append(("operation", kwargs)),
        kwargs={"query": "static-query"},
        handle="fia-handle",
        event=SimpleNamespace(record=lambda stream: calls.append(("event", stream))),
    )

    task.apply(DeviceModule(), "update-stream", [3, 8])

    assert calls == [
        ("begin", "update-stream", "fia-handle"),
        (
            "operation",
            {
                "query": "static-query",
                "actual_seq_lengths": [3, 8],
                "actual_seq_lengths_kv": [3, 8],
            },
        ),
        ("end", "update-stream"),
        ("event", "update-stream"),
    ]


@pytest.mark.parametrize("failure", ["replay", "signal"])
def test_npu_submission_failure_makes_the_graph_unrecoverable(failure):
    calls = []
    runner = _npu_runner()
    runner.plan = lambda total, windows: (8, [8 - total])

    def call(name):
        calls.append(name)
        if name == failure:
            raise torch.OutOfMemoryError("injected submission failure")

    runner._device_module.graph_task_update_begin = lambda *args: call("begin")
    runner._device_module.graph_task_update_end = lambda *args: call("end")
    runner._npu_update_stream.wait_stream = lambda stream: call("wait")
    task = NpuGraphUpdateTask(
        operation=lambda **kwargs: call("operation"),
        kwargs={},
        handle=object(),
        event=SimpleNamespace(record=lambda stream: call("signal")),
    )
    runner._graphs[8] = SimpleNamespace(
        hidden_states=torch.zeros(8, 2),
        graph=SimpleNamespace(replay=lambda: call("replay")),
        output=torch.ones(8, 2),
        npu_update_tasks=(task,),
    )

    with pytest.raises(EncoderGraphUnrecoverableError, match="restart"):
        runner.run(torch.ones(3, 2), [3])
    calls_after_failure = list(calls)
    with pytest.raises(EncoderGraphUnrecoverableError, match="previously failed"):
        runner.run(torch.ones(3, 2), [3])
    assert calls == calls_after_failure


def test_npu_pre_capture_failure_leaves_the_bucket_eager():
    runner = _npu_runner()
    runner.plan = lambda total, windows: (8, [8 - total])
    capture_calls = []

    def capture(bucket_size, *, window_lens=None):
        capture_calls.append((bucket_size, window_lens))
        raise torch.OutOfMemoryError("simulated warmup failure")

    runner.capture = capture

    assert runner.run(torch.ones(4, 2), [4]) is None
    assert runner.run(torch.ones(4, 2), [4]) is None
    assert capture_calls == [(8, (4, 4))]
    assert runner._failed == {8}


def test_npu_attention_capture_restores_partial_setup():
    runner = _npu_runner()
    original = torch.nn.Identity()
    attention = torch.nn.Module()
    attention.qkv_backend_name = "ascend_attn"
    attention.qkv_backend = original
    runner._tower = SimpleNamespace(
        layers=[
            SimpleNamespace(self_attn=attention),
            SimpleNamespace(self_attn=SimpleNamespace(qkv_backend_name="unsupported")),
        ]
    )

    with pytest.raises(RuntimeError, match="ascend_attn"):
        with runner.capture_npu_attention_tasks(NpuGraphCaptureContext(tasks=[])):
            pytest.fail("capture body must not run after partial setup fails")
    assert attention.qkv_backend is original


def test_npu_attention_capture_registers_an_explicit_fia_task(monkeypatch):
    calls = []

    def operation(**kwargs):
        calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            _npu_fused_infer_attention_score_get_max_workspace=lambda **kwargs: torch.empty(
                16
            ),
            npu_fused_infer_attention_score=SimpleNamespace(out=operation),
        ),
    )

    class Event:
        def wait(self, stream):
            calls.append(("wait", stream))

        def reset(self, stream):
            calls.append(("reset", stream))

        def record(self, stream):
            calls.append(("record", stream))

    class DeviceModule:
        def current_stream(self):
            return "capture-stream"

        def ExternalEvent(self):
            return Event()

        def graph_task_group_begin(self, stream):
            calls.append(("begin", stream))

        def graph_task_group_end(self, stream):
            calls.append(("end", stream))
            return "fia-handle"

    context = NpuGraphCaptureContext(tasks=[])
    attention = NpuGraphCaptureAttention(context, DeviceModule())
    metadata = SimpleNamespace(cu_seqlens=torch.tensor([0, 3, 8], dtype=torch.int32))

    output = attention(
        torch.zeros(8, 2, 4),
        torch.zeros(8, 2, 4),
        torch.zeros(8, 2, 4),
        forward_metadata=metadata,
    )

    assert output.shape == (8, 2, 4)
    assert len(context.tasks) == 1
    assert context.tasks[0].handle == "fia-handle"
    assert calls[-2]["actual_seq_lengths"] == [3, 8]
    assert calls[-2]["actual_seq_lengths_kv"] == [3, 8]


def _capture_runner(backend):
    backend.supports_graph_task_update = True

    class DeviceModule:
        pool = object()
        graph_pool_handle_calls = 0

        def graph_pool_handle(self):
            self.graph_pool_handle_calls += 1
            return self.pool

        def Stream(self, device):
            return SimpleNamespace(wait_stream=lambda other: None)

        def current_stream(self, device):
            return SimpleNamespace(wait_stream=lambda other: None)

        def stream(self, side):
            return nullcontext()

        def synchronize(self, device):
            pass

    def identity(hidden_states):
        return hidden_states, None

    class IdentityNorm:
        normalized_shape = (2,)

        def __call__(self, hidden_states):
            return hidden_states

    runner = object.__new__(Qwen3ASREncoderLayerStackGraphRunner)
    runner._tower = SimpleNamespace(
        layers=[],
        ln_post=IdentityNorm(),
        proj1=identity,
        act=lambda hidden_states: hidden_states,
        proj2=identity,
    )
    runner._device = torch.device("cpu")
    runner._dtype = torch.float32
    runner._device_module = DeviceModule()
    runner._max_seqlen = 8
    runner._buckets = (8,)
    runner._graph_backend = backend
    runner._graph_pool = None
    runner._graphs = {}
    runner._failed = set()
    runner._capture_failed = False
    runner._graph_submission_unrecoverable = False
    runner._npu_update_stream = object()
    runner.plan = lambda total, windows: (8, [8 - total])
    return runner


def test_npu_capture_context_failure_is_terminal():
    class CaptureContext:
        def __enter__(self):
            raise RuntimeError("simulated capture context failure")

        def __exit__(self, *args):
            return False

    runner = _capture_runner(SimpleNamespace(capture=lambda **kwargs: CaptureContext()))

    with pytest.raises(EncoderGraphUnrecoverableError, match="entering or executing"):
        runner.run(torch.ones(4, 2), [4])
    with pytest.raises(EncoderGraphUnrecoverableError, match="previously failed"):
        runner.run(torch.ones(4, 2), [4])


def test_npu_captures_share_one_graph_pool():
    pools = []

    class Backend:
        def capture(self, *, pool=None, stream=None, thread_local_errors=False):
            pools.append(pool)
            return nullcontext(SimpleNamespace())

    runner = _capture_runner(Backend())

    runner.capture(8, window_lens=(4, 4))
    runner.capture(8, window_lens=(2, 2, 4))

    assert runner._device_module.graph_pool_handle_calls == 1
    assert pools == [runner._device_module.pool, runner._device_module.pool]
