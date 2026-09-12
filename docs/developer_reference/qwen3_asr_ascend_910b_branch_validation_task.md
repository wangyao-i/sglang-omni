# Qwen3-ASR Ascend branch-validation task

> Historical: this task pinned the SGLang main candidate. Do not run it.
> Use [`qwen3_asr_ascend_v0519_validation_task.md`](qwen3_asr_ascend_v0519_validation_task.md).

## Result

Completed as `failed` on 2026-09-12 at the first model-smoke startup. Keep this
task as the first-run record. The next action is the bounded startup-failure
classification in
[`qwen3_asr_ascend_910b_failure_classification_task.md`](qwen3_asr_ascend_910b_failure_classification_task.md);
do not rerun this task unchanged.

Run this task only after reading
[`qwen3_asr_ascend_910b_handoff.md`](qwen3_asr_ascend_910b_handoff.md) and
confirming the exact branch heads.

## Inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export SMOKE_WAV=/server/local/approved-smoke.wav
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-branch-validation
mkdir -p "${EVIDENCE}"
```

Do not return the values of these variables.

## Gate 0: Sign Off The Baseline

In each repository, run:

```bash
git status --short --branch
git fetch origin
git rev-parse HEAD
git log -1 --format='%H %s'
```

Check out detached exact heads:

```bash
cd "${SGLANG_REPO}"
git switch --detach 85e8933dc3ae4ecd44a1e0ebf4595f0fd9011595

cd "${OMNI_REPO}"
git switch --detach 5190678c463f6c2b01e4ee0007cf788c3fdc2287
```

Record:

```bash
python -c "import sys, torch, sglang, torch_npu, triton, sgl_kernel_npu; print(sys.version); print(torch.__version__); print(sglang.__version__, sglang.__file__); print(torch_npu.__version__); print(triton.__version__); print(sgl_kernel_npu.__version__)"
```

Stop and return `blocked` if:

- the checkout or installed package does not resolve to the requested code;
- `sglang.__version__` is not the declared `0.5.19` line;
- torch and torch_npu major/minor versions differ;
- the NPU is unavailable; or
- the worktree contains unexplained changes.

## Gate 1: Focused SGLang Test

Run the test on the NPU host:

```bash
cd "${SGLANG_REPO}"
python -m pytest -q \
  test/registered/unit/hardware_backend/npu/test_fused_ops.py \
  >"${EVIDENCE}/test-sglang-fused-ops.log" 2>&1
```

This test must prove all four properties:

1. the external kernel is registered as
   `torch.ops.sglang.npu_split_qkv_rmsnorm_rope`;
2. fake-tensor output metadata is correct;
3. symbolic tracing keeps the external kernel behind the custom-op node; and
4. eager and TorchAir-compiled values match the external kernel for batches
   1 and 2.

## Gate 2: Focused Omni Tests

Run with the SGLang branch installed or first on `PYTHONPATH`:

```bash
cd "${OMNI_REPO}"
python -m pytest -q tests/unit_test/qwen3_asr/test_encoder_service.py \
  >"${EVIDENCE}/test-omni-encoder-service.log" 2>&1
```

At minimum, record:

```bash
python -m pytest -q \
  tests/unit_test/qwen3_asr/test_encoder_service.py::test_service_creates_device_stream \
  tests/unit_test/qwen3_asr/test_encoder_service.py::test_attach_embedding_records_default_stream \
  >"${EVIDENCE}/test-omni-stream-focus.log" 2>&1
```

## Gate 3: Minimal Model Smoke

Start a fresh process with the Qwen3-ASR default profile. Do not disable
torch.compile and do not add a fallback switch.

```bash
cd "${OMNI_REPO}"
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
  >"${EVIDENCE}/server.log" 2>&1
```

Wait for readiness, then send one transcription request:

```bash
curl --fail --silent --show-error \
  -X POST "http://127.0.0.1:${PORT}/v1/audio/transcriptions" \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F language=en \
  -F response_format=json \
  -F "file=@${SMOKE_WAV}" \
  >"${EVIDENCE}/smoke-response.json"
```

Then send a second sequential request and stop the server normally.

## Stop Conditions

Stop immediately on the first:

- test failure or collection error;
- startup failure;
- request failure or empty response;
- device error, OOM, hang, or timeout;
- unexpected eager fallback;
- stream-context or allocator error; or
- non-clean shutdown.

Keep the first complete traceback and the first repository frame that owns it.

## Non-Goals

- No performance measurement.
- No concurrency 70 or three-repeat campaign.
- No realtime WebSocket work.
- No timestamp or forced-alignment work.
- No installer, feature-matrix, or unrelated cleanup changes in this task.

## Return

Use the return contract in the handoff. Keep full logs and response bodies
server-local.
