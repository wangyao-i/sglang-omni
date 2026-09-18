"""CPU source-isolated worker tests; not device or full-stack qualification.

Execute the actual service, queue worker and cache with only SGLang import
boundaries substituted. No torch_npu import or accelerator execution.
"""

from __future__ import annotations

import ast
import contextlib
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]


def load(relative, injected=None):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith(("sglang.", "sglang_omni."))
        )
    ]
    module = types.ModuleType("_transfer_test_" + path.stem)
    sys.modules[module.__name__] = module
    module.__dict__.update(injected or {})
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


base = load("sglang_omni/scheduling/pre_lm_encoder.py")
cache = load("sglang_omni/scheduling/stage_cache.py")
service_module = load(
    "sglang_omni/models/qwen3_asr/encoder_service.py",
    {
        "PreLMEncoderService": base.PreLMEncoderService,
        "QueueEntry": base.QueueEntry,
        "StageOutputCache": cache.StageOutputCache,
        "MultimodalInputFormat": types.SimpleNamespace(PRECOMPUTED_EMBEDDING="ready"),
        "create_device_stream": lambda device: None,
        "device_stream_context": lambda stream: contextlib.nullcontext(),
    },
)


def item(key="audio"):
    return types.SimpleNamespace(
        audio_fingerprint=key,
        num_audio_tokens=2,
        feature=object(),
        precomputed_embeddings=None,
    )


class Model:
    def __init__(self):
        self.audio_tower = torch.nn.Linear(1, 1)
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(hidden_size=4)
        )
        self.calls = 0

    def get_audio_feature(self, items):
        self.calls += 1
        return torch.ones((len(items) * 2, 4))


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.service = service_module.Qwen3ASRPreLMEncoderService(
            self.model, cache_namespace="test", max_batch_wait_ms=20
        )
        self.addCleanup(self.service.close)

    def test_transfer_owned_by_worker_and_ready_after_sync(self):
        source = torch.ones((2, 4))
        target = item()
        reached, release = threading.Event(), threading.Event()
        calls = []
        original = torch.Tensor.to

        def transfer(tensor, *args, **kwargs):
            if tensor is source:
                calls.append(threading.current_thread().name)
            return original(tensor, *args, **kwargs)

        def sync():
            reached.set()
            if not release.wait(3):
                raise RuntimeError("test synchronization timeout")

        self.service.synchronize_batch = sync
        with patch.object(torch.Tensor, "to", transfer):
            future = self.service.submit_cached_embedding(target, source)
            try:
                self.assertTrue(reached.wait(2))
                self.assertFalse(future.done())
                self.assertEqual(calls, ["qwen3-asr-audio-encode"])
            finally:
                release.set()
            self.assertIs(future.result(2), target.precomputed_embeddings)
        self.assertEqual(self.model.calls, 0)

    def test_submit_item_cache_hit_uses_worker(self):
        self.service._cache.put("test:audio", torch.ones((2, 4)))
        calls = []
        original = self.service.attach_embedding

        def attach(target, embedding):
            calls.append(threading.current_thread().name)
            original(target, embedding)

        self.service.attach_embedding = attach
        self.service.submit_item(item()).result(2)
        self.assertEqual(calls, ["qwen3-asr-audio-encode"])
        self.assertEqual(self.model.calls, 0)

    def test_mixed_batch_preserves_result_order_without_encoding_hits(self):
        cached, fresh = item(), item("fresh")
        source = torch.full((2, 4), 9.0)
        outputs = self.service._execute_batch(
            [service_module._CachedEmbeddingTransfer(cached, source), fresh]
        )
        self.assertIs(outputs[0], cached.precomputed_embeddings)
        self.assertTrue(torch.equal(outputs[0], source))
        self.assertIs(outputs[1], fresh.precomputed_embeddings)
        self.assertEqual(self.model.calls, 1)

    def test_sync_error_fails_future_without_encode(self):
        def fail():
            raise RuntimeError("transfer sync failed")

        self.service.synchronize_batch = fail
        with self.assertRaisesRegex(RuntimeError, "transfer sync failed"):
            self.service.submit_cached_embedding(item(), torch.ones((2, 4))).result(2)
        self.assertEqual(self.model.calls, 0)

    def test_transfer_failure_does_not_kill_worker(self):
        original = self.service.attach_embedding

        def fail(target, embedding):
            raise RuntimeError("copy failed")

        self.service.attach_embedding = fail
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            self.service.submit_cached_embedding(item(), torch.ones((2, 4))).result(2)
        self.service.attach_embedding = original
        self.service.submit_cached_embedding(item(), torch.ones((2, 4))).result(2)

    def test_invalid_and_closed_submissions(self):
        with self.assertRaises(ValueError):
            self.service.submit_cached_embedding(item(), torch.ones((3, 4)))
        self.service.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.service.submit_cached_embedding(item(), torch.ones((2, 4)))

    def test_queued_source_survives_cache_eviction(self):
        reached, release = threading.Event(), threading.Event()

        def sync():
            reached.set()
            if not release.wait(3):
                raise RuntimeError("test synchronization timeout")

        self.service.synchronize_batch = sync
        self.service._cache.put("test:audio", torch.ones((2, 4)))
        source = self.service.lookup_cached_embedding("audio", 2)
        future = self.service.submit_cached_embedding(item(), source)
        try:
            self.assertTrue(reached.wait(2))
            self.service._cache.remove_if_same("test:audio", source)
            del source
            self.assertFalse(future.done())
        finally:
            release.set()
        self.assertTrue(torch.equal(future.result(2), torch.ones((2, 4))))

    def test_late_cache_hit_recheck_does_not_encode(self):
        self.service._cache.put("test:audio", torch.ones((2, 4)))
        with patch.object(self.service, "lookup_cached_embedding", return_value=None):
            self.service.submit_item(item()).result(2)
        self.assertEqual(self.model.calls, 0)

    def test_builder_cache_branch_defers_until_transfer_future(self):
        import concurrent.futures

        path = ROOT / "sglang_omni/models/qwen3_asr/request_builders.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        branch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "cached_embedding is not None"
            and any(isinstance(child, ast.Return) for child in node.body)
        )
        fn = ast.parse("def branch():\n    pass\n").body[0]
        fn.body = [branch]
        ready = concurrent.futures.Future()
        target, cached = item(), torch.ones((2, 4))

        def submit(actual_item, actual_cached):
            self.assertIs(actual_item, target)
            self.assertIs(actual_cached, cached)
            return ready

        namespace = dict(
            cached_embedding=cached,
            audio_item=target,
            req_data=object(),
            audio_encoder_service=types.SimpleNamespace(submit_cached_embedding=submit),
            DeferredAdmission=types.SimpleNamespace,
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                str(path),
                "exec",
            ),
            namespace,
        )
        result = namespace["branch"]()
        self.assertIs(result.ready, ready)
        self.assertIs(result.value, namespace["req_data"])
        self.assertFalse(result.ready.done())

    def test_follower_publishes_without_device_work(self):
        import concurrent.futures

        leader = concurrent.futures.Future()
        self.service._inflight["test:audio"] = leader
        follower = item()
        ready = self.service.submit_item(follower)
        with patch.object(
            torch.Tensor, "to", side_effect=AssertionError("caller device work")
        ):
            leader.set_result(torch.ones((2, 4)))
            self.assertIs(ready.result(2), follower.precomputed_embeddings)

    def test_transfer_enters_private_stream_and_registers_consumer(self):
        events = []
        stream = types.SimpleNamespace(synchronize=lambda: events.append("sync"))
        consumer = object()
        self.service._stream = stream

        @contextlib.contextmanager
        def context(actual):
            self.assertIs(actual, stream)
            events.append("enter")
            yield
            events.append("exit")

        def record(tensor, actual):
            self.assertIs(actual, consumer)
            self.assertEqual(threading.current_thread().name, "qwen3-asr-audio-encode")
            events.append("record")

        with (
            patch.object(service_module, "device_stream_context", context),
            patch.object(
                torch,
                "get_device_module",
                return_value=types.SimpleNamespace(
                    default_stream=lambda device: consumer
                ),
            ),
            patch.object(torch.Tensor, "record_stream", record),
        ):
            self.service.submit_cached_embedding(item(), torch.ones((2, 4))).result(2)
        self.assertEqual(events, ["enter", "record", "exit", "sync"])

    def test_cancelled_transfer_does_not_kill_worker(self):
        reached, release = threading.Event(), threading.Event()

        def sync():
            reached.set()
            if not release.wait(3):
                raise RuntimeError("test synchronization timeout")

        self.service.synchronize_batch = sync
        future = self.service.submit_cached_embedding(item(), torch.ones((2, 4)))
        try:
            self.assertTrue(reached.wait(2))
            self.assertTrue(future.cancel())
        finally:
            release.set()
        next_future = self.service.submit_cached_embedding(item(), torch.ones((2, 4)))
        next_future.result(2)
        self.assertTrue(future.cancelled())


if __name__ == "__main__":
    unittest.main()
