# Qwen3-ASR Ascend encoder update-stream probe

This diagnostic answers one question: with pristine SGLang `v0.5.19` and the
required concurrent encoder graph update/replay sequence restored, does making
the Omni update helper explicitly enter the caller's private encoder stream
restore cold concurrency-8 progress?

A pass supports a stream-context-loss mechanism but does not by itself qualify
the 140-request all-graph workload. A failure rejects explicit update-stream
propagation as a sufficient repair. Stop at the first failure.

## Exact inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export EXPECTED_SGLANG_HEAD=0bcd822377da7b5718e674eaf9c870d349424dd1
export EXPECTED_OMNI_CODE_HEAD=d7906ce77eacf20a53d9614b7509d0ad7188e55e
export EXPECTED_OMNI_BASE=f55c3b094419b4b8c2aba84d83c1c55c0ebaa1de
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export ASCEND_RT_VISIBLE_DEVICES=14
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-encoder-update-stream-probe
mkdir -p "${EVIDENCE}"
```

Use the user-confirmed Omni node. Adapt only the server-local checkout, model,
evidence, visible-device, and port paths. Do not return paths, host details, raw
logs, transcripts, or audio.

## Gate 0: clean recovery and identity

Use a new process and require the selected card to be at its recorded idle HBM
baseline with no unrelated process or context holder. Stop only processes owned
by the preceding task. If ownership is ambiguous or HBM does not recover,
return `blocked`.

Preserve all previous worktrees. Use fresh clean worktrees for this task:

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
versions. Do not apply either ordered-update diagnostic and do not set an update
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
  --output "${EVIDENCE}/stream-propagated-conc8-32.json" \
  --save-raw-dir "${EVIDENCE}/raw-stream-propagated-32"
```

Pass requires 32/32 completed with zero failure, timeout, empty transcript,
no-progress interval, or encoder fallback/update failure; positive encoder,
prefill, and decode graph evidence; and no ACL, allocator, stream, device, ATB,
PagedAttention, or OOM error. There is no WER or performance threshold.

Stop if any graph update/replay raises, if 90 seconds pass without a request
completion, or if the previous decoder update stall recurs. Before cleanup,
retain one sanitized Python stack identifying the blocked path. Do not edit
either repository or try another variant on the server.

Stop the service normally. Require a free port, no residual process or NPU
context holder, and HBM at the idle baseline. Do not continue into the
140-request qualification in this process.

## Result interpretation

- Pass: losing the encoder worker's private stream in the update helper is a
  necessary participant in the observed failure. Re-run the 140-request gate
  on a separate fresh process before considering a shipping fix.
- Encoder update still fails or hangs: explicit stream propagation is
  insufficient; retain the first error and stop this diagnostic family before
  adding another update-order variant.
- Decoder update still hangs: moving the encoder helper to the private stream
  is insufficient; investigate the CANN graph-task transaction boundary while
  keeping SGLang unchanged.

## Return contract

```text
Task status: passed / failed / blocked
SGLang branch / observed HEAD / clean:
Omni branch / observed HEAD / clean / diff paths from base:
Imported module identity:
Runtime and NPU versions / selected device:
Pre-run process holders / recovery action / idle HBM:
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
