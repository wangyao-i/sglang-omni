"""CPU source-isolated constructor checks, not NPU qualification.

Run alongside SGLang's test_async_submission.py to cover the unchanged replay
order. This executes the actual constructor with device doubles because the
local development environment cannot import the full SGLang model stack.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "sglang_omni/models/qwen3_asr/encoder_cuda_graph.py"
)


def load_constructor(namespace):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Qwen3ASREncoderLayerStackGraphRunner"
    )
    constructor = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    exec(
        compile(ast.Module(body=[constructor], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["__init__"]


class PrivateStreamTests(unittest.TestCase):
    def check_ownership(self, is_npu, model_devices):
        created = []
        resolved = []

        def create_stream(*, device):
            stream = object()
            created.append((device, stream))
            return stream

        def get_device_module(device):
            resolved.append(device)
            return SimpleNamespace(Stream=create_stream)

        def forbidden_shared_stream():
            self.fail("encoder must not request decoder's shared update stream")

        constructor = load_constructor(
            {
                "torch": SimpleNamespace(get_device_module=get_device_module),
                "current_platform": SimpleNamespace(is_npu=lambda: is_npu),
                "_get_feat_extract_output_lengths_int": lambda frames: 13,
                "get_npu_graph_update_stream": forbidden_shared_stream,
            }
        )
        runners = []
        for device in model_devices:
            tower = SimpleNamespace(
                parameters=lambda device=device: iter(
                    [SimpleNamespace(device=device, dtype="float16")]
                ),
                config=SimpleNamespace(n_window=50, n_window_infer=100),
            )
            runner = SimpleNamespace()
            constructor(
                runner, tower, buckets=(128,), max_batch_size=8, graph_backend=object()
            )
            runners.append(runner)
        self.assertEqual(resolved, model_devices)
        if is_npu:
            self.assertEqual([device for device, _ in created], model_devices)
            for runner, (_, stream) in zip(runners, created):
                self.assertIs(runner._npu_update_stream, stream)
            self.assertIsNot(
                runners[0]._npu_update_stream, runners[1]._npu_update_stream
            )
        else:
            self.assertEqual(created, [])
            self.assertTrue(all(r._npu_update_stream is None for r in runners))

    def test_same_device_runners_do_not_share(self):
        self.check_ownership(True, ["npu:0", "npu:0"])

    def test_explicit_nondefault_model_device(self):
        self.check_ownership(True, ["npu:1", "npu:2"])

    def test_non_npu_does_not_allocate_update_stream(self):
        self.check_ownership(False, ["cuda:0", "cuda:1"])


if __name__ == "__main__":
    unittest.main()
