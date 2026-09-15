# Qwen3-ASR Ascend v0.5.19 all-graph validation task

This is the only authorized server task for the encoder + prefill + decode
graph profile. The current phase is functional qualification only. Stability
repetitions, graph-registry soak, HBM trend analysis, and performance
measurement belong to a separate deferred task.

## Exact Inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export EXPECTED_SGLANG_HEAD=0bcd822377da7b5718e674eaf9c870d349424dd1
export EXPECTED_OMNI_BASE_HEAD=18c8cfd2eeeb495569426875a2e2bf4114133caf
export EXPECTED_OMNI_CODE_HEAD=b8a37792829ef402edf7b5c83136ab5c804cf5af
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export ASCEND_RT_VISIBLE_DEVICES=14
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-v0519-all-graphs
mkdir -p "${EVIDENCE}"
```

Use the actual checkout roots imported by Python. Do not return these variable
values, paths, host details, raw logs, transcripts, or audio.

## Execution Lanes

Gate 0 is CPU-only and runs once before any service starts.

Gate 1 has one mandatory NPU lane. Do not shard one 140-request benchmark run
across multiple cards; the service state, scheduler, and graph registry are
part of the unit under test.

If multiple cards and enough host CPU are available:

- run at most one optional mirror lane with the identical Gate 1 profile on a
  different physical card; the primary lane remains the required result;
- after a primary-lane failure, use separate diagnostic lanes for encoder-only,
  prefill-only, and decode-only attribution;
- give every lane a distinct `ASCEND_RT_VISIBLE_DEVICES`, port, evidence
  directory, server PID/PGID, and log.

Each lane maps its selected physical card to logical `npu:0` and uses
`--asr.gpu 0`. Never run two lanes on the same physical card. Parallel lanes
must not share a port or evidence directory. If host CPU contention is present,
stop the extra lanes and treat timeout as environment-blocked, not as a
functional software failure.

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
git merge-base --is-ancestor 1638c5dddb012686210f85ed3ee050fed1ac4597 HEAD
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
  --asr.gpu 0 \
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

## Gate 1: Functional Cold Concurrency-8 Run

Start one fresh service:

```bash
cd "${OMNI_REPO}"
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

Wait for readiness, then run the retained 140-sample English SeedTTS workload
at concurrency 8 without a warm-up request:

```bash
cd "${OMNI_REPO}"
python -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 --port "${PORT}" \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --lang en --max-samples 140 \
  --concurrencies 8 --repeats 1 \
  --disable-resource-monitor \
  --output "${EVIDENCE}/functional-conc8-140.json" \
  --save-raw-dir "${EVIDENCE}/raw-functional"
```

Require:

- `140/140` completed, zero failures, zero timeouts, zero empty transcripts;
- zero successful samples with WER above `0.5`;
- corpus WER is recorded for investigation but is not the pass threshold;
- no hang or no-progress interval;
- no ACL, allocator, stream, device, ATB, or `PagedAttentionOperation` error;
- no Torch Compile marker;
- all three target graph paths positive;
- zero encoder graph fallback markers.

Stop at the first failure.

## Gate 2: Shutdown And Cleanup

Stop the service normally and verify:

- the port is free;
- no worker, benchmark client, or NPU context holder remains;
- HBM returns to the pre-run idle baseline;
- no retained graph handle or orphaned multiprocessing child remains.

## Stop Conditions

Stop at the first occurrence of:

- identity, import, or focused-test failure;
- startup or graph-capture failure;
- `PagedAttentionOperation`, ATB, ACL, allocator, stream, device, or OOM error;
- timeout, hang, missing or duplicate request result, or empty transcript;
- any `encoder graph eager fallback reason=` line;
- garbled-output or request-accounting failure;
- a Torch Compile marker;
- cleanup failure.

Do not add another arm, warm-up pool, environment override, repeat count, soak
pass, or performance measurement.

## Non-Goals

- No Torch Compile.
- No C70, throughput, p95, latency, or other performance measurement.
- No multi-run stability soak or HBM trend analysis.
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
Shutdown and cleanup state:
First complete failure and owning repository:
Server-local artifacts retained:
Suggested next bounded task:
```
