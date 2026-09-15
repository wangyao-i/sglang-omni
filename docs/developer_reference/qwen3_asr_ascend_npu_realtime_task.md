# Qwen3-ASR Ascend NPU realtime validation task

This is a functional qualification task. It is not a performance or soak task.

## Exact Inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export EXPECTED_SGLANG_HEAD=0bcd822377da7b5718e674eaf9c870d349424dd1
export EXPECTED_OMNI_BASE_HEAD=18c8cfd2eeeb495569426875a2e2bf4114133caf
export EXPECTED_OMNI_CODE_HEAD=3b1fee2269cdcee9a3a957d5a456e2477b39f05e
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export ASCEND_RT_VISIBLE_DEVICES=14
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-npu-realtime
mkdir -p "${EVIDENCE}"
```

Use the checkout roots imported by Python. Do not return these values or any
private paths, host details, raw logs, audio, or transcripts.

The selected physical card must be the only card visible to this lane. The
service maps it to logical `npu:0`; never run another lane on the same card.

## Gate 0: Identity And Focused Tests

```bash
cd "${SGLANG_REPO}"
test "$(git rev-parse HEAD)" = "${EXPECTED_SGLANG_HEAD}"
test -z "$(git status --porcelain)"
test ! -e python/sglang/srt/hardware_backend/npu/fused_ops.py

cd "${OMNI_REPO}"
test -z "$(git status --porcelain)"
git merge-base --is-ancestor "${EXPECTED_OMNI_BASE_HEAD}" HEAD
git merge-base --is-ancestor 1638c5dddb012686210f85ed3ee050fed1ac4597 HEAD
git diff --exit-code "${EXPECTED_OMNI_CODE_HEAD}" HEAD -- \
  sglang_omni tests
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

cd "${OMNI_REPO}"
python -m pytest -q \
  tests/unit_test/qwen3_asr/test_streaming.py \
  tests/unit_test/serve/test_realtime_transcription.py \
  tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py \
  >"${EVIDENCE}/test-focused.log" 2>&1

python -m pytest -q tests/unit_test/qwen3_asr \
  >"${EVIDENCE}/test-qwen3-asr.log" 2>&1
```

Stop before hardware on any identity, import, collection, or test failure.

Resolve the runtime profile before starting a service:

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

Require `enable_torch_compile: false`, `disable_cuda_graph: false`, prefill
backend `breakable`, decode backend `full`, and encoder graph enabled.

## Gate 1: Bounded Client Workload

Start one fresh service:

```bash
cd "${OMNI_REPO}"
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --enable-realtime \
  --port "${PORT}" \
  --asr.gpu 0 \
  --asr.engine.enable_torch_compile false \
  --asr.engine.cuda_graph_backend_prefill breakable \
  --asr.engine.cuda_graph_backend_decode full \
  --asr.engine.max_running_requests 64 \
  --asr.engine.cuda_graph_max_bs 64 \
  --asr.factory.enable_encoder_cuda_graph true \
  >"${EVIDENCE}/server.log" 2>&1 &
SERVER_PID=$!
```

Wait for the HTTP health endpoint. Require startup and graph capture to finish
without ACL, ATB, allocator, stream, device, OOM, eager-compile, or
`PagedAttentionOperation` errors.

Run one paced VAD workload and one paced manual workload against the same
server and graph registry:

```bash
cd "${OMNI_REPO}"
python -m benchmarks.eval.benchmark_asr_realtime \
  --host 127.0.0.1 --port "${PORT}" \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --lang en --max-samples 12 \
  --concurrencies 1 --mode vad \
  --output "${EVIDENCE}/realtime-vad.json"

python -m benchmarks.eval.benchmark_asr_realtime \
  --host 127.0.0.1 --port "${PORT}" \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --lang en --max-samples 12 \
  --concurrencies 1 --mode manual \
  --output "${EVIDENCE}/realtime-manual.json"
```

The second command must use the same model identifier as the server. The
benchmark opens and closes one WebSocket session per sample, so the 12-sample
sequence also exercises disconnect followed by reconnect. This task records
correctness and protocol behavior only; latency fields are not a performance
gate.

Require for both runs:

- `12/12` sessions evaluated with zero client errors;
- zero protocol violations;
- zero empty completed transcripts;
- zero samples with per-sample WER above `0.5`;
- at least one non-final partial in each mode;
- a `transcription.completed` event for every session;
- no duplicate or missing final segment;
- no uncontrolled long-lived request.

Record corpus WER but do not use it as a performance threshold.

## Gate 2: Graph Evidence And Cleanup

Scan the server log and require:

- the prerequisite all-graph gate has proved encoder bucket replay; this run
  has at least one encoder capture and zero fallback or update failure;
- positive prefill capture and replay evidence;
- positive decode replay evidence;
- zero encoder fallback or host-input update failure for this bounded workload;
- zero Torch Compile markers;
- zero ACL, ATB, allocator, stream, device, graph-capture, or
  `PagedAttentionOperation` errors.

Stop the service normally. Require:

- the port is free;
- no server, benchmark, multiprocessing child, or NPU context holder remains;
- HBM returns to the pre-run idle baseline;
- no orphaned graph handle or device stream remains.

## Stop Conditions

Stop at the first:

- identity, import, or focused-test failure;
- service start or graph-capture failure;
- protocol-violation event or missing terminal event;
- empty or garbled transcript;
- encoder graph capacity fallback;
- forbidden compile, ACL, ATB, allocator, stream, device, OOM, or
  `PagedAttentionOperation` marker;
- timeout, hang, or cleanup failure.

Do not add another model, card, concurrency, soak repeat, or benchmark
parameter after a failure. Classify the first failure and return it.

## Non-Goals

- No performance, latency, RTF, throughput, or memory-trend claim.
- No word timestamps or forced alignment.
- No encoder graph key redesign.
- No source, test, dependency, site-package, or model edit on the server.

## Return Contract

```text
Task status: passed / failed / blocked
SGLang branch / observed HEAD / clean:
SGLang-Omni branch / observed HEAD / clean:
Imported module identity:
Runtime and CANN/torch_npu/sgl_kernel_npu versions:
NPU model / device count / selected device:
Resolved compile and graph settings:
Focused test result:
Qwen3-ASR unit suite result:
Manual mode: completed / failed / violations / empty / max per-sample WER:
Server VAD mode: completed / failed / violations / empty / max per-sample WER:
Encoder graph capture / replay / fallback:
Prefill graph capture / replay / eager:
Decode graph replay / eager:
Shutdown and cleanup state:
First complete failure and owning repository:
Server-local artifacts retained:
Suggested next bounded task:
```
