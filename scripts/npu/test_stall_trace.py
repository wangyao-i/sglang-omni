"""CPU checks of diagnostic wrappers; does not exercise CANN or ptrace."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PATH = (
    Path(__file__).resolve().parents[2] / "sglang_omni/models/qwen3_asr/stall_trace.py"
)


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        spec = importlib.util.spec_from_file_location("stall_trace_test", PATH)
        self.trace = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"SGLANG_ASR_STALL_TRACE_DIR": self.temp.name}):
            spec.loader.exec_module(self.trace)

    def tearDown(self):
        state = self.trace._local.__dict__
        if "fd" in state:
            os.close(state["fd"])
        self.temp.cleanup()

    def rows(self):
        return [
            json.loads(line)
            for path in Path(self.temp.name).glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]

    def test_rejects_objects_without_repr(self):
        class Poison:
            def __repr__(self):
                raise AssertionError("repr must never run")

        with self.assertRaises(TypeError):
            self.trace.emit("bad", tensor=Poison())
        self.assertEqual(self.rows(), [])

    def test_disabled_does_not_query_device(self):
        self.trace.TRACE_DIR = None
        with self.trace.transfer(object(), object(), object()):
            pass
        self.assertEqual(self.rows(), [])

    def test_transfer_metadata_and_exception_propagation(self):
        tensor = SimpleNamespace(
            device="cpu", dtype="float16", dim=lambda: 2, numel=lambda: 64
        )
        device = SimpleNamespace(
            current_device=lambda: 1,
            current_stream=lambda: SimpleNamespace(npu_stream=123),
        )
        error = RuntimeError("original")
        with self.assertRaises(RuntimeError) as caught:
            with self.trace.transfer(tensor, "npu:1", device):
                raise error
        self.assertIs(caught.exception, error)
        rows = self.rows()
        self.assertEqual(rows[-1]["phase"], "attach.to.error")
        self.assertEqual(rows[-1]["source_device"], "cpu")
        self.assertEqual(rows[-1]["current_stream"], 123)

    def test_hooks_keep_call_order_arguments_and_results(self):
        calls = []
        stream = SimpleNamespace(npu_stream=456)
        handle = object()

        class Graph:
            def replay(self):
                calls.append("replay")
                return 7

            def update(self, *, cpu_update_input):
                calls.append(cpu_update_input)
                graphs.graph_task_update_begin(stream, handle)
                graphs.graph_task_update_end(stream)
                return 8

        graphs = SimpleNamespace(
            NPUGraph=Graph,
            graph_task_update_begin=lambda s, h: calls.append((s, h)),
            graph_task_update_end=lambda s: calls.append(s),
        )
        self.trace._install_hooks(graphs)
        graph = Graph()
        payload = object()
        self.assertEqual(graph.replay(), 7)
        self.assertEqual(graph.update(cpu_update_input=payload), 8)
        self.assertEqual(calls, ["replay", payload, (stream, handle), stream])
        self.assertEqual(
            [row["phase"] for row in self.rows()],
            [
                "graph_replay.enter",
                "graph_replay.exit",
                "graph_update.enter",
                "task_begin.enter",
                "task_begin.exit",
                "task_end.enter",
                "task_end.exit",
                "graph_update.exit",
            ],
        )

    def test_stream_query_failure_not_mislabeled_as_transfer(self):
        def fail():
            raise RuntimeError("query")

        tensor = SimpleNamespace(
            device="cpu", dtype="float16", dim=lambda: 2, numel=lambda: 64
        )
        with self.assertRaises(RuntimeError):
            with self.trace.transfer(
                tensor, "npu:0", SimpleNamespace(current_device=fail)
            ):
                self.fail("transfer must not execute")
        self.assertEqual(self.rows()[-1]["phase"], "attach.stream_query.enter")


if __name__ == "__main__":
    unittest.main()
