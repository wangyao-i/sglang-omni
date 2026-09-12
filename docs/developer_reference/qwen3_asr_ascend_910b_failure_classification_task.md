# Qwen3-ASR Ascend startup-failure classification task

> Historical: this task targeted the SGLang main baseline. That baseline was
> abandoned after the main-only Scheduler interface drift. Do not run it.
> Use [`qwen3_asr_ascend_v0519_validation_task.md`](qwen3_asr_ascend_v0519_validation_task.md).

Run this task only after the baseline in
[`qwen3_asr_ascend_910b_handoff.md`](qwen3_asr_ascend_910b_handoff.md) has
been acknowledged and the exact heads have been checked out.

## Inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export SMOKE_WAV=/server/local/approved-smoke.wav
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-failure-classification
mkdir -p "${EVIDENCE}"
```

Do not return the values of these variables.

## Decision To Make

The first bounded branch validation failed during startup with
`PagedAttentionOperation`, `Capture cuda graph failed`, and
`corrupted size vs. prev_size`.

The falsifiable question is:

> Does the startup failure still occur when `torch.compile` is disabled while
> the decode graph remains enabled?

Only run the second arm if the first arm passes.

The `corrupted size vs. prev_size` message is a native heap-allocator corruption
signal. It is not by itself evidence that the custom fused-op boundary changed
the PagedAttention input layout. Do not edit `fused_ops.py` from this report.

## Exact Identity

| Repository | Exact runtime head |
|---|---|
| SGLang | `85e8933dc3ae4ecd44a1e0ebf4595f0fd9011595` |
| SGLang-Omni | `5190678c463f6c2b01e4ee0007cf788c3fdc2287` |

The earlier server report labeled the SGLang branch
`codex/910c-072-chain23-off` while reporting the SGLang hash above. Treat the
branch label as an identity discrepancy and check out and run the detached
hash. Do not substitute a branch tip.

## Preconditions

Use a fresh process for each arm. Before the first arm:

- record `git status --short --branch`, `git rev-parse HEAD`, and
  `git log -1 --format='%H %s'` for both repositories;
- record the exact Python, CANN, torch, torch_npu, triton-ascend, and
  `sgl_kernel_npu` versions;
- record the launch environment without changing it:

  ```bash
  for name in LD_PRELOAD LD_LIBRARY_PATH PYTORCH_NPU_ALLOC_CONF \
    STREAMS_PER_DEVICE HCCL_BUFFSIZE ASCEND_LAUNCH_BLOCKING; do
    printf '%s=%s\n' "${name}" "${!name-}"
  done
  ```

- when the server is containerized or launched non-interactively, compare
  that output with the same variables in an interactive login shell; treat any
  difference as classification evidence, not as permission to change the A/B
  environment;
- confirm that no failed server process remains and that the selected NPU is
  healthy and sufficiently idle;
- preserve the first complete startup traceback and the first repository frame
  from the failed run;
- do not edit either checkout, `site-packages`, model files, or graph policy.

Use the same model, smoke WAV, device, port policy, dtype, memory settings, and
other startup arguments as the failed run except for the one variable named in
each arm.

## Arm A: Disable torch.compile, Keep Decode Graph

Prove that the override reaches the resolved Omni configuration:

```bash
cd "${OMNI_REPO}"
sgl-omni config resolve \
  --model-path "${MODEL_PATH}" \
  --asr.engine.enable_torch_compile false \
  >"${EVIDENCE}/arm-a-config.yaml"

grep -n 'enable_torch_compile: false' "${EVIDENCE}/arm-a-config.yaml"
```

Start a fresh server with decode graph enabled by default:

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
  --asr.engine.enable_torch_compile false \
  >"${EVIDENCE}/arm-a-server.log" 2>&1
```

If the server becomes ready, send one transcription request, stop the server
normally, and retain the response metadata. Do not run Arm B after any failure.

## Arm B: Enable torch.compile, Disable Decode Graph

Run this arm only if Arm A passes.

Prove the overrides before starting:

```bash
cd "${OMNI_REPO}"
sgl-omni config resolve \
  --model-path "${MODEL_PATH}" \
  --asr.engine.enable_torch_compile true \
  --asr.engine.cuda_graph_backend_decode disabled \
  >"${EVIDENCE}/arm-b-config.yaml"

grep -nE 'enable_torch_compile: true|cuda_graph_backend_decode: disabled' \
  "${EVIDENCE}/arm-b-config.yaml"
```

Start a fresh server:

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
  --asr.engine.enable_torch_compile true \
  --asr.engine.cuda_graph_backend_decode disabled \
  >"${EVIDENCE}/arm-b-server.log" 2>&1
```

If the server becomes ready, send one transcription request and stop it
normally. Do not add further combinations in this task.

## Smoke Request

After an arm reaches readiness, set `ARM_EVIDENCE` to that arm's log directory
and send:

```bash
curl --fail --silent --show-error \
  -X POST "http://127.0.0.1:${PORT}/v1/audio/transcriptions" \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F language=en \
  -F response_format=json \
  -F "file=@${SMOKE_WAV}" \
  >"${ARM_EVIDENCE}/smoke-response.json"
```

Require a non-empty transcript, a zero exit status, and a clean normal
shutdown. Retain only response metadata when returning evidence.

## Stop Conditions

Stop at the first complete failure in each arm and do not continue collecting
data from the failed process. Return immediately if any of the following
occurs:

- startup exits before readiness;
- `PagedAttentionOperation`, `Capture cuda graph failed`, or
  `corrupted size vs. prev_size` appears;
- one request fails, returns an empty transcript, or hangs;
- a device, ACL, OOM, allocator, or stream error appears;
- shutdown is non-clean; or
- the resolved overrides do not show the values required by the arm.

## Interpretation

| Observation | Classification for the next bounded task |
|---|---|
| Arm A fails the same way | `torch.compile` and the custom fused-op boundary are cleared; next hold Arm A fixed and run one diagnostic with `ASCEND_LAUNCH_BLOCKING=1` to get a synchronous first failure, then discriminate the graph/runtime, lower stack, and launch environment |
| Arm A passes; Arm B fails | The failure requires compile-enabled model execution but occurs even with decode graph disabled; classify it as a compile-path failure |
| Both arms pass | The failure requires the combination of compile and decode graph capture; next isolate when the decode graph is captured, not the fused-op tensor layout |

In all cases, record the first complete failure and the first repository frame
before assigning an owner. If Arm A still reports heap corruption, include the
interactive/non-interactive environment comparison and ask before changing any
runtime variable. `PagedAttentionOperation` is an asynchronous operator report
in common Ascend failures, so do not treat its name as the exact producer until
a blocking diagnostic confirms it.

## Non-Goals

- No source or test edits.
- No changes to the fused-op boundary.
- No performance, concurrency-70, realtime, timestamp, or forced-alignment
  work.
- No attempt to silently fall back to eager mode as a success.

## Return

Return the exact runtime matrix, one line per arm, the resolved override
evidence, the first complete failure and its first repository frame, the
classification, device health, and the server-local artifact names. Keep full
logs, paths, hostnames, audio, and response bodies on the isolated server.
