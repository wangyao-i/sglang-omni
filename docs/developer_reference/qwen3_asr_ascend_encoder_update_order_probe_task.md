# Qwen3-ASR Ascend encoder-update ordering probe

This diagnostic answers one question: with pristine SGLang `v0.5.19` and all
three graph paths unchanged, does removing only the Omni encoder graph's
background `NPUGraph.update()` helper restore cold concurrency-8 progress?

The diagnostic executes encoder host-input update and replay in order on the
encoder worker, which already owns the private NPU stream context. A pass does
not prove that the helper used the default stream and does not qualify the
140-request all-graph workload. A failure rejects this mechanism. Stop at the
first failure.

## Exact inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export EXPECTED_SGLANG_HEAD=0bcd822377da7b5718e674eaf9c870d349424dd1
export EXPECTED_OMNI_CODE_HEAD=b00a8b8b884981fd42d0326073291339ab5c8821
export EXPECTED_OMNI_BASE=f55c3b094419b4b8c2aba84d83c1c55c0ebaa1de
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export ASCEND_RT_VISIBLE_DEVICES=14
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-encoder-update-order-probe
mkdir -p "${EVIDENCE}"
```

Use the user-confirmed Omni node. Adapt only the server-local checkout, model,
evidence, visible-device, and port paths. Do not return paths, host details, raw
logs, transcripts, or audio.

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
stash, or overwrite unknown changes. Use fresh clean worktrees for this task.

```bash
cd "${SGLANG_REPO}"
test "$(git rev-parse HEAD)" = "${EXPECTED_SGLANG_HEAD}"
test -z "$(git status --porcelain)"

cd "${OMNI_REPO}"
test "$(git rev-parse HEAD)" = "${EXPECTED_OMNI_CODE_HEAD}"
test -z "$(git status --porcelain)"
git merge-base --is-ancestor "${EXPECTED_OMNI_BASE}" HEAD
test -z "$(git diff --name-only "${EXPECTED_OMNI_BASE}" HEAD | grep -v -E \
  '^(sglang_omni/platforms/device_graph.py|tests/unit_test/platforms/test_device_graph.py)$')"
```

Confirm imported `sglang` and `sglang_omni` resolve inside these exact clean
worktrees. Record torch, torch_npu, CANN, triton-ascend, and sgl-kernel-npu
versions. Do not apply the SGLang ordered-update diagnostic or set an update
mode environment variable.

Run the Omni focused gate:

```bash
cd "${OMNI_REPO}"
python -m pytest -q \
  tests/unit_test/platforms/test_device_graph.py \
  tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py \
  tests/unit_test/qwen3_asr/test_encoder_service.py \
  >"${EVIDENCE}/test-omni-focused.log" 2>&1
```

## Gate 1: single-variable 32-request probe

Resolve and require compile disabled, prefill `breakable`, decode `full`, CUDA
graph not disabled, and encoder graph enabled. Start one fresh service with the
same profile as the failed all-graph gate:

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
  --output "${EVIDENCE}/encoder-ordered-conc8-32.json" \
  --save-raw-dir "${EVIDENCE}/raw-encoder-ordered-32"
```

Pass requires 32/32 completed with zero failure, timeout, empty transcript,
no-progress interval, or encoder fallback/update failure; positive encoder,
prefill, and decode graph evidence; and no ACL, allocator, stream, device, ATB,
PagedAttention, or OOM error. There is no WER or performance threshold.

Stop immediately if ordered encoder `graph.update()` does not return, if
decoder `graph.update()` repeats the previous stall, or if 90 seconds pass with
no request completion. Before cleanup, retain one sanitized Python stack that
identifies the blocked update/replay path. Do not try another ordering or edit
either repository on the server.

Stop the service normally. Require a free port, no residual process or NPU
context holder, and HBM at the idle baseline. Do not continue into the
140-request qualification in this process.

## Result interpretation

- Pass: the new Omni encoder update helper is a necessary participant in the
  observed all-graph liveness failure. This does not prove that it used the
  default stream.
- Encoder update blocks before replay: same-thread ordering is not supported by
  this encoder graph/runtime path; reject the diagnostic and next test explicit
  private-stream propagation or update-transaction serialization in Omni.
- Decoder update still blocks: removing the encoder helper is insufficient;
  investigate overlapping encoder graph work and the CANN graph-task queue
  without changing SGLang.

## Result

Failed. The service completed 24 of 32 requests; eight were missing/timeouts.
Encoder, prefill, and decode graph paths were all observed, but the encoder
batch path raised `NPU graph host input update failed`. Its chained cause was
`graph_task_update_begin` in `NPUGraph.cpp:65`:
`AclmdlRICaptureTaskUpdateBegin(stream, handle.task_group)` returned error
`107033`. This is the task's declared "encoder update blocks or fails before
replay" outcome, not a partial pass. The result rejects same-thread ordered
encoder update for this runtime path; it does not prove that the original
helper inherited the correct private stream, nor that concurrent graph
submitters are unrelated. No symbolic interpretation is assigned to `107033`
without vendor evidence.

## Return contract

```text
Task status: passed / failed / blocked
SGLang branch / observed HEAD / clean:
Omni branch / observed HEAD / clean / diff paths from base:
Old dirty Omni worktree paths retained untouched:
Imported module identity:
Runtime and NPU versions / selected device:
Pre-run process holders / reset action / idle HBM:
Omni focused test result:
Resolved graph and compile settings:
Probe completed / total / failures / timeouts / empty:
Last progress time and no-progress interval:
Encoder capture / replay / fallback / update-error evidence:
Prefill graph evidence:
Decode graph evidence:
Blocked stack, if any:
Forbidden error counts:
Shutdown / process cleanup / final HBM:
First complete failure and owning repository:
Server-local artifacts retained:
Suggested next bounded task:
```
