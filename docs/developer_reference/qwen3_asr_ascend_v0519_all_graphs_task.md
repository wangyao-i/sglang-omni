# Qwen3-ASR Ascend v0.5.19 all-graph validation task

This is the only authorized server task for the encoder + prefill + decode
graph profile. It is a functional qualification, not a performance task.

## Exact Inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export EXPECTED_SGLANG_HEAD=0bcd822377da7b5718e674eaf9c870d349424dd1
export EXPECTED_OMNI_BASE_HEAD=886ced95b9c0b76429798bb60dbd34d3f71dad95
export EXPECTED_OMNI_CODE_HEAD=8ad2a5a9bcfc6c43ca1c623a933cb6e8479a8b75
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-v0519-all-graphs
mkdir -p "${EVIDENCE}"
```

Use the actual checkout roots imported by Python. Do not return these variable
values, paths, host details, raw logs, transcripts, or audio.

## Gate 0: Identity And Focused Tests

Run one fresh process:

```bash
cd "${SGLANG_REPO}"
test "$(git rev-parse HEAD)" = "${EXPECTED_SGLANG_HEAD}"
test -z "$(git status --porcelain)"
test ! -e python/sglang/srt/hardware_backend/npu/fused_ops.py
git diff --exit-code "${EXPECTED_SGLANG_HEAD}" -- \
  python/sglang/srt/models/qwen3.py

cd "${OMNI_REPO}"
test -z "$(git status --porcelain)"
git merge-base --is-ancestor "${EXPECTED_OMNI_BASE_HEAD}" HEAD
git diff --exit-code "${EXPECTED_OMNI_CODE_HEAD}" HEAD -- \
  sglang_omni tests
test -z "$(git diff --name-only "${EXPECTED_OMNI_CODE_HEAD}" HEAD | \
  grep -v '^docs/')"

python - <<'PY'
import sglang
import sglang_omni
import torch
import torch_npu
import os
from pathlib import Path

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

Record CANN, torch, torch_npu, triton-ascend, and `sgl_kernel_npu` versions.
Stop and return `blocked` on any identity, import, diff, or clean-worktree
mismatch.

Run the focused local regression and then the complete Qwen3-ASR suite:

```bash
cd "${OMNI_REPO}"
python -m pytest -q \
  tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py \
  tests/unit_test/qwen3_asr/test_encoder_service.py \
  >"${EVIDENCE}/test-focused.log" 2>&1
python -m pytest -q tests/unit_test/qwen3_asr \
  >"${EVIDENCE}/test-qwen3-asr.log" 2>&1
```

Stop before hardware on the first collection or test failure.

Resolve the exact profile:

```bash
cd "${OMNI_REPO}"
sgl-omni config resolve \
  --model-path "${MODEL_PATH}" \
  --asr.engine.enable_torch_compile false \
  --asr.engine.cuda_graph_backend_prefill breakable \
  --asr.engine.cuda_graph_backend_decode full \
  --asr.engine.max_running_requests 64 \
  --asr.engine.cuda_graph_max_bs 64 \
  --asr.factory.enable_encoder_cuda_graph true \
  >"${EVIDENCE}/resolved-config.yaml"
```

Require the resolved configuration to show:

- `enable_torch_compile: false`;
- prefill backend `breakable`;
- decode backend `full`;
- `disable_cuda_graph: false`;
- encoder graph enabled.

## Gate 1: Cold Concurrency-8 Liveness Repeats

Run three independent repetitions. Each repetition must use a fresh service
process and a separate log/result pair. Do not reuse a process after a failure.

For repetition 1, start:

```bash
cd "${OMNI_REPO}"
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
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

Wait for readiness, then run the retained 70-sample English SeedTTS workload at
concurrency 8 without a warm-up request:

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

- `70/70` completed, zero failures, zero timeouts, zero empty transcripts;
- no hang or no-progress interval;
- no ACL, allocator, stream, device, ATB, or `PagedAttentionOperation` error;
- no Torch Compile marker;
- all three target graph paths positive;
- zero encoder graph fallback markers.

Repeat the same fresh-process sequence for repetitions 2 and 3, writing
`server-2.log`, `liveness-conc8-2.json`, `server-3.log`, and
`liveness-conc8-3.json`. All three repetitions must satisfy the same criteria.

Stop at the first failure. Do not start Gate 2 from a polluted process.

## Gate 2: 140-Request Functional Correctness

Stop the Gate 1 process and verify the port and device are idle. Start a fresh
service with the exact same command, then run the retained 140-sample
functional-correctness workload at concurrency 8:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 --port "${PORT}" \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --lang en --max-samples 140 \
  --concurrencies 8 --repeats 1 \
  --disable-resource-monitor \
  --output "${EVIDENCE}/correctness-140.json" \
  --save-raw-dir "${EVIDENCE}/raw-correctness"
```

Require:

- `140/140` completed with zero transport or HTTP failure;
- zero empty hypotheses;
- zero successful samples with WER above `0.5`;
- zero encoder graph fallback log lines;
- positive encoder, prefill, and decode replay evidence;
- no compile, graph-capture, graph-replay, device, or stream error.

The corpus WER is recorded for investigation but is not the pass threshold for
this functional gate.

## Gate 3: Steady-State Graph Registry Soak

This gate verifies that the NPU graph registry converges for the retained
workload instead of continuously capturing new layouts. It is a functional and
memory-stability gate, not a performance measurement.

Stop the Gate 2 service and start one fresh service with the same exact command.
First run one 140-request warm-up:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 --port "${PORT}" \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --lang en --max-samples 140 \
  --concurrencies 8 --repeats 1 \
  --disable-resource-monitor \
  --output "${EVIDENCE}/soak-warmup.json" \
  --save-raw-dir "${EVIDENCE}/raw-soak-warmup"
```

After the warm-up, record the current count of encoder capture markers. Then
run three consecutive 140-request soak passes:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 --port "${PORT}" \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --lang en --max-samples 140 \
  --concurrencies 8 --repeats 1 \
  --disable-resource-monitor \
  --output "${EVIDENCE}/soak-1.json" \
  --save-raw-dir "${EVIDENCE}/raw-soak-1"
```

Repeat for `soak-2.json` and `soak-3.json`.

Require after each soak pass:

- `140/140` completed, zero failures, zero timeouts, zero empty hypotheses;
- no new `[qwen3-asr] captured encoder layer-stack graph` line after the
  warm-up;
- zero `encoder graph eager fallback reason=` lines;
- positive encoder, prefill, and decode replay evidence;
- no ACL, ATB, allocator, stream, device, capture, or replay error;
- service remains responsive between passes.

HBM must not show monotonic growth across the three passes. Temporary
capture-time growth is allowed, but the final pass must return to a stable band
consistent with the post-warm-up baseline, within the platform's normal
measurement noise.

## Gate 4: Shutdown And Cleanup

Stop the service normally and verify:

- the port is free;
- no worker, benchmark client, or NPU context holder remains;
- HBM returns to the pre-run idle baseline;
- no monotonic memory growth or retained graph handle remains.

## Stop Conditions

Stop at the first occurrence of:

- identity, import, or focused-test failure;
- startup or graph-capture failure;
- `PagedAttentionOperation`, ATB, ACL, allocator, stream, device, or OOM error;
- timeout, hang, missing or duplicate request result, or empty transcript;
- any `encoder graph eager fallback reason=` line;
- a new encoder capture after the Gate 3 warm-up;
- garbled-output or request-accounting failure;
- a Torch Compile marker;
- cleanup failure.

Do not add another arm, warm-up pool, environment override, or performance
measurement.

## Non-Goals

- No Torch Compile.
- No performance or C70 measurement.
- No realtime, timestamps, or forced alignment.
- No source, test, dependency, package, or site-package edit on the server.

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
Full Qwen3-ASR test result:
Gate 1 evaluated / total / skipped / failures / timeouts:
Gate 1 wall clock:
Gate 1 encoder capture/replay/fallback markers:
Gate 1 prefill capture/replay/eager markers:
Gate 1 decode capture/replay/eager markers:
Gate 1 repetitions 1 / 2 / 3:
Gate 2 evaluated / total / skipped:
Gate 2 empty hypotheses / samples with WER > 0.5:
Gate 2 corpus WER:
Gate 2 graph markers and forbidden errors:
Gate 3 warm-up and soak evaluated / total / skipped:
Gate 3 new encoder captures after warm-up per pass:
Gate 3 graph replay/fallback markers per pass:
Gate 3 HBM baseline / pass 1 / pass 2 / pass 3:
Shutdown and cleanup state:
First complete failure and owning repository:
Server-local artifacts retained:
Suggested next bounded task:
```
