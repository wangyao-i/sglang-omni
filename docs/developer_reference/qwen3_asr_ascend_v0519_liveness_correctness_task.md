# Qwen3-ASR Ascend pure v0.5.19 liveness and correctness task

This task qualifies the current pure-base graph-only candidate after the
startup and smoke gates passed. It is intentionally limited to liveness and
correctness. Do not run performance, Torch Compile, realtime, timestamp, or
forced-alignment work in this task.

## Exact Identity

| Repository | Exact runtime head | Required state |
|---|---|---|
| SGLang | `0bcd822377da7b5718e674eaf9c870d349424dd1` (`v0.5.19`) | Clean; no fused-op patch |
| SGLang-Omni | `5190678c463f6c2b01e4ee0007cf788c3fdc2287` | Clean encoder private-stream code |

The SGLang checkout must not contain
`python/sglang/srt/hardware_backend/npu/fused_ops.py`, and
`python/sglang/srt/models/qwen3.py` must have no diff against the tag commit.
Documentation commits on the Omni branch do not replace the runtime head above.

## Inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export EXPECTED_SGLANG_HEAD=0bcd822377da7b5718e674eaf9c870d349424dd1
export EXPECTED_OMNI_HEAD=5190678c463f6c2b01e4ee0007cf788c3fdc2287
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-v0519-liveness-correctness
mkdir -p "${EVIDENCE}/raw-liveness" "${EVIDENCE}/raw-correctness"
```

Use the actual checkout roots imported by Python. Do not return these variable
values, model paths, host details, raw logs, transcripts, or audio.

## Gate 0: Identity And Environment

Run one fresh process:

```bash
cd "${SGLANG_REPO}"
test "$(git rev-parse HEAD)" = "${EXPECTED_SGLANG_HEAD}"
test -z "$(git status --porcelain)"
test ! -e python/sglang/srt/hardware_backend/npu/fused_ops.py
git diff --exit-code "${EXPECTED_SGLANG_HEAD}" -- \
  python/sglang/srt/models/qwen3.py

cd "${OMNI_REPO}"
test "$(git rev-parse HEAD)" = "${EXPECTED_OMNI_HEAD}"
test -z "$(git status --porcelain)"

python - <<'PY'
from pathlib import Path

import sglang
import sglang_omni
import torch
import torch_npu
import triton

print("sglang:", sglang.__version__, sglang.__file__)
print("sglang_omni:", sglang_omni.__file__)
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("triton:", triton.__version__)
PY
```

Record CANN, torch, torch_npu, triton-ascend, and `sgl_kernel_npu` versions.
Record the resolved graph/compile settings and the selected NPU identity.
Stop and return `blocked` on any identity mismatch, unexpected source diff,
unclean worktree, import mismatch, or unavailable device.

Run the focused Omni test:

```bash
cd "${OMNI_REPO}"
python -m pytest -q tests/unit_test/qwen3_asr/test_encoder_service.py \
  >"${EVIDENCE}/test-encoder-service.log" 2>&1
```

There is no SGLang `test_fused_ops.py` gate on the pure base because that file
and patch are absent.

## Gate 1: Cold Concurrency-8 Liveness

Start one fresh server process. Keep the pure-base graph-only profile:

```bash
cd "${OMNI_REPO}"
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
  --asr.engine.enable_torch_compile false \
  >"${EVIDENCE}/server.log" 2>&1 &
SERVER_PID=$!
```

Wait for readiness. Do not add a warm-up request before the first liveness
measurement. Run 70 deterministic SeedTTS English samples at concurrency 8,
one repeat, with no client warm-up:

```bash
cd "${OMNI_REPO}"
python -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 --port "${PORT}" \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --lang en --max-samples 70 \
  --concurrencies 8 --repeats 1 \
  --disable-resource-monitor \
  --output "${EVIDENCE}/liveness-conc8.json" \
  --save-raw-dir "${EVIDENCE}/raw-liveness"
```

Require:

- `evaluated=70`, `total=70`, and `skipped=0`;
- zero transport/HTTP failures, timeouts, empty transcripts, or missing
  results;
- no `PagedAttentionOperation`, ATB setup, ACL, allocator, stream, or device
  error;
- no unexpected eager/compile fallback;
- the server remains responsive through the full 70-request pass.

Stop at the first failure. Do not continue to Gate 2 from a failed or polluted
process.

## Gate 2: Reuse The Qualified 140-Request Correctness Protocol

The `0.078` versus `0.02` discrepancy is already diagnosed. It is an evaluation
protocol difference, not a model or SGLang regression:

- the non-equivalent conc1 path produced WER about `0.0778`;
- the packaged exact10 harness produced `0.0176`-`0.0179`;
- the M1c reference produced `0.0161`.

Do not invent a replacement correctness protocol. The current SeedTTS
benchmark command is not a substitute for the previously qualified exact10
workload, and concurrency 1 is not an acceptance protocol.

Before running any correctness request, perform a read-only recovery of the
exact qualified invocation from the retained M1c, `910C-062`, and `910C-071`
artifacts. Record:

- the exact client and script path;
- the pinned manifest SHA-256 and sample count;
- the exact concurrency, repeat count, warm-up partition, and request timeout;
- whether the cold concurrency-8 gate preceded the correctness phase;
- the raw per-sample JSONL path and its schema;
- the resolved server configuration used by that qualified run.

If that invocation is not recoverable, stop and return `blocked`. Do not run a
new correctness protocol, do not lower the WER threshold, and do not substitute
the `benchmark_asr_seedtts` conc8 command.

Only after the qualified invocation is recovered, stop the Gate 1 process,
confirm the port and device are idle, and launch a fresh pure-base service with
the same graph-only settings. Run that exact 140-request correctness workload.

Require:

- all 140 requests completed with no timeout or transport failure;
- corpus WER is in the qualified `0.016`-`0.019` band;
- zero garbled outputs, where a garbled output is any successful sample with
  WER greater than `0.5` or an empty hypothesis;
- zero unexpected eager or compile fallback.

## Run History

### 2026-09-12: known protocol mismatch, not accepted

The server passed Gate 0 and the 70-request cold concurrency-8 liveness gate.
Its correctness attempt reported 140/140 completion, zero garbled outputs, and
WER `0.0784`, but it used the known non-equivalent exact10 concurrency-1 path.
This result is consistent with the historical `0.0778` protocol and does not
close the correctness gate. The fix is to recover and reuse the qualified
invocation, not to rerun the same path under a new threshold. The shutdown
check also returned without confirming HBM or port state, so cleanup must be
verified before the next run.

## Gate 3: Shutdown And Cleanup

Stop the correctness service normally, then record:

- service exit status and whether the port is free;
- NPU memory/health return to the pre-run idle baseline;
- no retained process, worker, or device holder;
- no monotonic memory growth across Gate 1 and Gate 2.

If cleanup is not clean, return `failed` or `blocked` with the exact first
cleanup failure. Do not delete the evidence before it is classified.

## Stop Conditions

Stop immediately on:

- an identity or focused-test failure;
- startup, capture, or replay failure;
- timeout, missing/duplicate/unexpected result, or empty transcript;
- WER or garbled-count failure;
- OOM, ACL, allocator, stream, device, or `PagedAttentionOperation` error;
- unexpected eager/compile fallback; or
- cleanup or device-health failure.

Do not add a second arm, a warm-up pool, or an alternative server profile after
a failure.

## Non-Goals

- No Torch Compile.
- No C70 or performance target.
- No encoder graph or prefill graph enablement.
- No realtime, timestamps, or forced alignment.
- No source, configuration, dependency, or `site-packages` edits.

## Return Contract

Return only sanitized fields:

```text
Task status: passed / failed / blocked
SGLang branch / observed HEAD / clean:
SGLang-Omni branch / observed HEAD / clean:
Imported module identity:
Runtime and CANN/torch_npu/sgl_kernel_npu versions:
NPU model / device count / selected device:
Resolved compile and graph settings:
Focused encoder-service test:
Gate 1 evaluated / total / skipped / failures / timeouts:
Gate 1 wall clock and completion range:
Gate 1 target-path and fallback markers:
Gate 2 evaluated / total / skipped:
Gate 2 corpus WER:
Gate 2 empty hypotheses / per-sample WER > 0.5:
Gate 2 wall clock:
Gate 2 target-path and fallback markers:
Shutdown and cleanup state:
First complete failure and owning repository:
Server-local artifacts retained:
Suggested next bounded task:
```

Do not return model paths, hostnames, addresses, credentials, raw audio,
transcripts, full logs, or proprietary profiler output.
