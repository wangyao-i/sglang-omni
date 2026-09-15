# Qwen3-ASR NPU bucket-key validation task

This is the only authorized server task for the #2160 bucket-key delivery
candidate. Run gates in order and stop at the first failure.

## Exact inputs

Set these values to the server-local checkout and model locations. Do not
return paths, host details, raw logs, transcripts, or audio.

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export EXPECTED_SGLANG_HEAD=0bcd822377da7b5718e674eaf9c870d349424dd1
export EXPECTED_OMNI_BASE_HEAD=1638c5dddb012686210f85ed3ee050fed1ac4597
export EXPECTED_OMNI_CODE_HEAD=302cf932fcf17ce2f1e836b44a06a6a8d9979451
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export ASCEND_RT_VISIBLE_DEVICES=14
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-pr2160-bucket-key
mkdir -p "${EVIDENCE}"
```

Use a fresh process for Gate 0 and one isolated 910C for Gates 1 and 2. Do not
run a second service or workload on the selected physical device.

## Gate 0: identity and tests

```bash
cd "${SGLANG_REPO}"
test "$(git rev-parse HEAD)" = "${EXPECTED_SGLANG_HEAD}"
test -z "$(git status --porcelain)"
test ! -e python/sglang/srt/hardware_backend/npu/fused_ops.py

cd "${OMNI_REPO}"
test -z "$(git status --porcelain)"
git merge-base --is-ancestor "${EXPECTED_OMNI_BASE_HEAD}" HEAD
git merge-base --is-ancestor "${EXPECTED_OMNI_CODE_HEAD}" HEAD
test -z "$(git diff --name-only "${EXPECTED_OMNI_CODE_HEAD}" HEAD | \
  grep -v '^docs/')"

python - <<'PY'
import os
from pathlib import Path

import sglang
import sglang_omni
import torch
import torch_npu

assert Path(sglang.__file__).resolve().is_relative_to(
    Path(os.environ["SGLANG_REPO"]).resolve()
)
assert Path(sglang_omni.__file__).resolve().is_relative_to(
    Path(os.environ["OMNI_REPO"]).resolve()
)
print("sglang:", sglang.__version__, sglang.__file__)
print("sglang_omni:", sglang_omni.__file__)
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
PY
```

Record the observed Omni head and the CANN, torch, torch_npu, triton-ascend,
and `sgl_kernel_npu` versions. An Omni docs-only descendant of the expected
code head is valid. Any other identity, import, diff, or clean-worktree mismatch
is `blocked`.

Run:

```bash
cd "${OMNI_REPO}"
python -m pytest -q tests/unit_test/platforms/test_device_graph.py \
  >"${EVIDENCE}/test-device-graph.log" 2>&1
python -m pytest -q tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py \
  >"${EVIDENCE}/test-encoder-graph.log" 2>&1
python -m pytest -q tests/unit_test/qwen3_asr \
  >"${EVIDENCE}/test-qwen3-asr.log" 2>&1
git diff --check
```

Require all collected tests to pass. The XPU signature-introspection test may
skip only when this torch build does not provide `torch.xpu.graph`. Stop before
hardware on the first collection or test failure.

## Gate 1: NPU mechanism

Run only the accelerator-backed eager-versus-graph test that exercises one
captured encoder tower with layouts `[500]` and `[450]`:

```bash
cd "${OMNI_REPO}"
python -m pytest -q \
  tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py \
  -k test_graph_matches_eager_tower \
  >"${EVIDENCE}/test-npu-mechanism.log" 2>&1
```

Require:

- `[500]` captures successfully and has max absolute difference below `3e-2`;
- `[450]` reuses the same bucket graph and has max absolute difference below
  `3e-2`;
- the graph registry contains one entry after both cases;
- update and replay complete without hang or exception;
- no eager fallback, capture failure, ACL, allocator, stream, device, ATB, or
  attention error appears.

## Gate 2: two-request all-graph smoke

Resolve and inspect this exact profile:

```bash
cd "${OMNI_REPO}"
sgl-omni config resolve \
  --model-path "${MODEL_PATH}" \
  --asr.gpu 0 \
  --asr.engine.enable_torch_compile false \
  --asr.engine.cuda_graph_backend_prefill breakable \
  --asr.engine.cuda_graph_backend_decode full \
  --asr.engine.max_running_requests 64 \
  --asr.engine.cuda_graph_max_bs 64 \
  --asr.factory.enable_encoder_cuda_graph true \
  >"${EVIDENCE}/resolved-config.yaml"
```

Require compile disabled, prefill `breakable`, decode `full`, CUDA graph not
disabled, and encoder graph enabled. Then start one fresh service:

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
  --asr.gpu 0 \
  --asr.engine.enable_torch_compile false \
  --asr.engine.cuda_graph_backend_prefill breakable \
  --asr.engine.cuda_graph_backend_decode full \
  --asr.engine.max_running_requests 64 \
  --asr.engine.cuda_graph_max_bs 64 \
  --asr.engine.decode_log_interval 1 \
  --asr.factory.enable_encoder_cuda_graph true \
  >"${EVIDENCE}/server.log" 2>&1 &
SERVER_PID=$!
```

After readiness, run two fixed English SeedTTS samples sequentially:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 --port "${PORT}" \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --lang en --max-samples 2 \
  --concurrencies 1 --repeats 1 \
  --disable-resource-monitor \
  --output "${EVIDENCE}/all-graph-smoke.json" \
  --save-raw-dir "${EVIDENCE}/raw-smoke"
```

Require `2/2` completed with zero failures, timeouts, or empty transcripts;
positive encoder, prefill, and decode graph markers; zero encoder fallback
markers; no Torch Compile marker; and no ACL, allocator, stream, device, ATB,
`PagedAttentionOperation`, or other attention error. This smoke has no WER,
latency, or throughput pass threshold.

## Gate 3: shutdown and cleanup

Stop the service normally. Require the port to be free, no service worker,
benchmark client, multiprocessing child, or NPU context holder to remain, and
HBM to return to its recorded pre-run idle baseline.

Do not edit source, tests, dependencies, packages, or site-packages. Do not add
warm-up requests, workload arms, repeats, soak runs, realtime, or performance
measurements.

## Return contract

```text
Task status: passed / failed / blocked
SGLang branch / observed HEAD / clean:
SGLang-Omni branch / observed HEAD / clean:
Diff from expected Omni code head: docs-only / mismatch
Imported module identity: matched / mismatch
Runtime and CANN/torch_npu/triton-ascend/sgl_kernel_npu versions:
NPU model / device count / selected device:
Platform graph tests: passed / failed / skipped
Encoder graph tests: passed / failed / skipped
Full Qwen3-ASR tests: passed / failed / skipped
Mechanism [500] max absolute difference / graph count:
Mechanism [450] max absolute difference / graph count:
All-graph smoke evaluated / total / failures / timeouts / empty:
Encoder capture / replay / fallback markers:
Prefill graph markers:
Decode graph markers:
Shutdown / process cleanup / HBM state:
First complete failure and owning repository:
Server-local artifacts retained:
Suggested next bounded task:
```
