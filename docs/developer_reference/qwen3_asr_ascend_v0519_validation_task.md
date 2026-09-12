# Qwen3-ASR Ascend v0.5.19 validation task

This is the active validation task after moving the runtime baseline from
SGLang main to the `v0.5.19` release line. It replaces the historical
main-based classification and retry tasks.

## Inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export SMOKE_WAV=/server/local/approved-smoke.wav
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-v0519-validation
mkdir -p "${EVIDENCE}"
```

Do not return the values of these variables.

## Exact Identity

| Repository | Branch | Exact runtime head | Base |
|---|---|---|---|
| SGLang | `codex/qwen3-asr-v0519-fused-op` | `e0011e30fbdb9690f01fa2083d452c93b37bb214` | `v0.5.19` commit `0bcd822377da7b5718e674eaf9c870d349424dd1` |
| SGLang-Omni | `codex/qwen3-asr-npu-encoder-stream-v0519` | `5190678c463f6c2b01e4ee0007cf788c3fdc2287` | `6ff46426469a1af2746cef71a9fdcfc09613966d` |

The SGLang candidate is the same three-file fused-op boundary cherry-picked
onto the release-line commit. It is not based on SGLang main.

## Gate 0

- Fetch both branches and check out the detached runtime heads above.
- Require clean tracked worktrees. Preserve any unrelated untracked artifact
  outside the checkout; do not use a dirty tree for this gate.
- Verify that the imported SGLang package resolves to the release-line
  checkout:

  ```bash
  python -c "import sglang; print(sglang.__version__, sglang.__file__)"
  ```

- The imported path must resolve to the checked-out release-line repository.
  If it resolves to a main checkout or another site-packages copy, stop and
  return `blocked`; do not edit `site-packages` or mix two SGLang installs.
- Record Python, CANN, torch, torch_npu, triton-ascend, and `sgl_kernel_npu`
  versions.
- Record `LD_PRELOAD`, `LD_LIBRARY_PATH`, `PYTORCH_NPU_ALLOC_CONF`,
  `STREAMS_PER_DEVICE`, `HCCL_BUFFSIZE`, and `ASCEND_LAUNCH_BLOCKING` without
  changing them.
- Confirm no residual process, port `8000` free, and the selected NPU healthy.

Stop and return `blocked` if the code does not resolve to the requested
release-line checkout, `sglang.__version__` does not belong to the `0.5.19`
release line, or the environment is not identifiable.

## Gate 1: Focused SGLang Test

```bash
cd "${SGLANG_REPO}"
python -m pytest -q \
  test/registered/unit/hardware_backend/npu/test_fused_ops.py \
  >"${EVIDENCE}/test-sglang-fused-ops.log" 2>&1
```

## Gate 2: Focused Omni Test

```bash
cd "${OMNI_REPO}"
python -m pytest -q \
  tests/unit_test/qwen3_asr/test_encoder_service.py \
  >"${EVIDENCE}/test-omni-encoder-service.log" 2>&1
```

## Gate 3: Default Startup And Smoke

Use the default Qwen3-ASR profile. Keep `torch.compile` enabled and keep the
decode graph enabled; do not add a fallback switch.

```bash
cd "${OMNI_REPO}"
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
  >"${EVIDENCE}/server.log" 2>&1
```

After readiness, send one transcription request:

```bash
curl --fail --silent --show-error \
  -X POST "http://127.0.0.1:${PORT}/v1/audio/transcriptions" \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F language=en \
  -F response_format=json \
  -F "file=@${SMOKE_WAV}" \
  >"${EVIDENCE}/smoke-response.json"
```

Require a non-empty transcript and clean normal shutdown. Record the actual
compile/graph mode evidence from the startup log or `/server_info`.

## Stop Conditions

Stop at the first complete failure:

- focused test failure or collection error;
- startup failure, graph capture failure, or Scheduler `AttributeError`;
- request failure, empty transcript, hang, or timeout;
- device, ACL, OOM, allocator, or stream error;
- unexpected eager fallback; or
- non-clean shutdown.

Keep the first complete traceback and the first repository frame that owns it.
Do not edit code or environment variables on the server.

## Non-Goals

- No SGLang main validation.
- No scheduler main-compatibility branch.
- No performance, concurrency-70, realtime, timestamp, or forced-alignment
  work.
- No source, package, or configuration edits on the server.

## Return

Return the exact repository heads and worktree state, runtime versions,
environment values, focused test counts, default startup result, smoke result,
target-path evidence, fallback/error counts, device health, cleanup state, and
the first complete failure if any. Keep full logs and response bodies
server-local.
