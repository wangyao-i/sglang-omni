# Qwen3-ASR Ascend ordered-update liveness probe

This diagnostic answers one question: with encoder, prefill, and decode graphs
unchanged, does replacing only the SGLang decoder graph background update
helper with ordered same-thread update/replay restore cold concurrency-8
progress?

A pass does not qualify the 140-request all-graph workload. A failure rejects
this mechanism. Stop at the first failure.

## Exact inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export EXPECTED_SGLANG_HEAD=4d819e5aa265e5548a63bdd9bb6ec35b10396950
export EXPECTED_SGLANG_BASE=0bcd822377da7b5718e674eaf9c870d349424dd1
export EXPECTED_OMNI_CODE_HEAD=f55c3b094419b4b8c2aba84d83c1c55c0ebaa1de
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export ASCEND_RT_VISIBLE_DEVICES=14
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-ordered-update-probe
mkdir -p "${EVIDENCE}"
```

Do not return paths, host details, raw logs, transcripts, or audio.

## Gate 0: clean recovery and identity

The previous hung process left 86% HBM. Before checkout or execution:

1. confirm the selected physical card has no unrelated process or context
   holder;
2. stop only processes owned by the previous failed task;
3. reset the selected card only if it is isolated and no unrelated holder is
   present;
4. require HBM to return to the recorded idle baseline before continuing.

If ownership is ambiguous or HBM does not recover, return `blocked`. Preserve
the previous dirty Omni checkout and its two changed paths; do not discard,
stash, or overwrite unknown changes. Use a fresh clean worktree for this task.

```bash
cd "${SGLANG_REPO}"
test "$(git rev-parse HEAD)" = "${EXPECTED_SGLANG_HEAD}"
test -z "$(git status --porcelain)"
git merge-base --is-ancestor "${EXPECTED_SGLANG_BASE}" HEAD
test "$(git diff --name-only "${EXPECTED_SGLANG_BASE}" HEAD | wc -l)" = 2
test -z "$(git diff --name-only "${EXPECTED_SGLANG_BASE}" HEAD | grep -v -E \
  '^(python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py|test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py)$')"

cd "${OMNI_REPO}"
test -z "$(git status --porcelain)"
git merge-base --is-ancestor "${EXPECTED_OMNI_CODE_HEAD}" HEAD
test -z "$(git diff --name-only "${EXPECTED_OMNI_CODE_HEAD}" HEAD | \
  grep -v '^docs/')"
```

Confirm imported `sglang` and `sglang_omni` resolve inside these exact clean
worktrees. Record torch, torch_npu, CANN, triton-ascend, and sgl-kernel-npu
versions.

Run the changed SGLang test and the Omni focused gate:

```bash
cd "${SGLANG_REPO}"
python -m pytest -q \
  test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py \
  >"${EVIDENCE}/test-sglang-graph-runner.log" 2>&1

cd "${OMNI_REPO}"
python -m pytest -q \
  tests/unit_test/platforms/test_device_graph.py \
  tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py \
  tests/unit_test/qwen3_asr/test_encoder_service.py \
  >"${EVIDENCE}/test-omni-focused.log" 2>&1
```

## Gate 1: single-variable 32-request probe

Resolve and require compile disabled, prefill `breakable`, decode `full`, CUDA
graph not disabled, and encoder graph enabled. Do not set
`SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE`; this diagnostic branch changes ordering
directly and the pinned base does not define that environment variable.

Start one fresh service with the same profile as the failed all-graph gate:

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

After readiness, run the first 32 fixed English SeedTTS samples at concurrency
8 without a warm-up request:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --host 127.0.0.1 --port "${PORT}" \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --lang en --max-samples 32 \
  --concurrencies 8 --repeats 1 \
  --disable-resource-monitor \
  --output "${EVIDENCE}/ordered-conc8-32.json" \
  --save-raw-dir "${EVIDENCE}/raw-ordered-32"
```

Pass requires 32/32 completed with zero failure, timeout, empty transcript,
no-progress interval, or encoder fallback/update failure; positive encoder,
prefill, and decode graph evidence; and no ACL, allocator, stream, device, ATB,
PagedAttention, or OOM error. There is no WER or performance threshold.

Stop the service normally. Require a free port, no residual process or NPU
context holder, and HBM at the idle baseline. Do not continue into the
140-request qualification in this process.

## Return contract

```text
Task status: passed / failed / blocked
SGLang branch / observed HEAD / clean / diff paths from base:
Omni branch / observed HEAD / clean:
Old dirty Omni worktree paths retained untouched:
Imported module identity:
Runtime and NPU versions / selected device:
Pre-run process holders / reset action / idle HBM:
SGLang focused test result:
Omni focused test result:
Resolved graph and compile settings:
Probe completed / total / failures / timeouts / empty:
Last progress time and no-progress interval:
Encoder capture / fallback / update-error evidence:
Prefill graph evidence:
Decode graph evidence:
Forbidden error counts:
Shutdown / process cleanup / final HBM:
First complete failure and owning repository:
Server-local artifacts retained:
Suggested next bounded task:
```
