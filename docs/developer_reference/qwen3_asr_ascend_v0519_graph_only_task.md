# Qwen3-ASR Ascend v0.5.19 graph-only task

This task tests the smallest configuration change supported by the current
code: disable `torch.compile` while keeping the NPU decode graph enabled.

## Why

- `torch.compile` was added to the Qwen3-ASR pipeline as a CUDA low/mid
  concurrency optimization, not as a model-correctness requirement.
- `create_sglang_qwen3_asr_executor` already defaults
  `enable_torch_compile=False`; the high-level pipeline configuration overrides
  it to `True`.
- The NPU attention backend uses `forward_decode_graph` when graph mode is
  enabled and `enable_torch_compile` is false. With compile enabled, graph
  capture uses a different decode implementation.
- The exact-head run with compile and graph enabled stopped at ATB
  `PagedAttentionOperation` setup. This task determines whether the graph-only
  path avoids that failure.

`torch.compile` remains deferred for NPU. Do not pursue compile+graph again
until a performance task requires it.

## Exact Identity

| Repository | Exact runtime head |
|---|---|
| SGLang | `0bcd822377da7b5718e674eaf9c870d349424dd1` (pure `v0.5.19`) |
| SGLang-Omni | `5190678c463f6c2b01e4ee0007cf788c3fdc2287` |

## Inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export EXPECTED_SGLANG_HEAD=0bcd822377da7b5718e674eaf9c870d349424dd1
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export SMOKE_WAV=/server/local/approved-smoke.wav
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-v0519-graph-only
mkdir -p "${EVIDENCE}"
```

Use the actual checkout roots imported by Python. Do not return the values of
these variables.

## Gate 0

Run the identity checks in
[`qwen3_asr_ascend_v0519_validation_task.md`](qwen3_asr_ascend_v0519_validation_task.md)
before starting the server.

Require:

- exact SGLang head `0bcd82237`;
- exact Omni head `5190678c`;
- clean worktrees;
- imported modules under the corresponding checkout roots;
- no new source, environment, or package changes.

## Resolved Configuration

Resolve the configuration before startup:

```bash
cd "${OMNI_REPO}"
sgl-omni config resolve \
  --model-path "${MODEL_PATH}" \
  --asr.engine.enable_torch_compile false \
  >"${EVIDENCE}/resolved-config.yaml"
```

Require the resolved configuration to show:

- `enable_torch_compile: false`;
- decode graph not disabled;
- no alternative attention backend or fallback selected for this test.

Stop and return `blocked` if the override does not reach the resolved
configuration.

## Startup And Smoke

Start one fresh process:

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
  --asr.engine.enable_torch_compile false \
  >"${EVIDENCE}/server.log" 2>&1 &
SERVER_PID=$!
```

Do not change the model, device, attention backend, decode graph setting,
environment, or other runtime arguments.

Wait for readiness. If startup succeeds, send exactly one request:

```bash
curl --fail --silent --show-error \
  -X POST "http://127.0.0.1:${PORT}/v1/audio/transcriptions" \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F language=en \
  -F response_format=json \
  -F "file=@${SMOKE_WAV}" \
  >"${EVIDENCE}/smoke-response.json"
```

Require a non-empty transcript and a clean normal shutdown.

## Stop Conditions

Stop at the first complete failure:

- resolved configuration is wrong;
- startup or graph capture fails;
- `PagedAttentionOperation`, ATB, device, ACL, allocator, or stream error;
- readiness timeout;
- request failure, empty transcript, or hang; or
- non-clean shutdown.

Do not add another arm or change another variable in this task.

## Classification

- Graph-only passes: this becomes the functional NPU baseline. Reassess the
  SGLang fused-op patch because the verifier no longer needs the
  compile-enabled path for the functional gate.
- Graph-only fails at the same `PagedAttentionOperation` setup: `torch.compile`
  is exonerated for this failure. Continue with log-first analysis of the NPU
  decode graph attention path.
- Graph-only fails somewhere else: return the first complete failure and its
  first owning repository frame. Do not change code in this task.

## Results

### Historical patch-bearing graph-only run

Functional gate: passed. Cleanup gate: passed.

- SGLang `e0011e30` and Omni `5190678c` were confirmed.
- Resolved configuration confirmed `enable_torch_compile: false`.
- The service reached readiness and decode graph capture completed without
  `PagedAttentionOperation`.
- One smoke request returned HTTP 200 with a non-empty transcript and
  `latency=0.283s`.
- Normal shutdown completed.
- Initial HBM remained at 86% with a holder. Follow-up identified the holder
  as a residual process; the server operator cleaned it. It is not attributed
  to the graph-only run.

Functional conclusion:

- The compile-enabled capture path is the source of the
  `PagedAttentionOperation` setup failure.
- Decode graph itself is functional with `torch.compile` disabled.
- The SGLang fused-op patch is not needed for this NPU functional path and
  should be removed or deferred unless a later performance task proves that
  compile is required.

### Pure v0.5.19 follow-up

Functional gate: passed.

- SGLang `0bcd8223` and Omni `5190678c` were confirmed.
- `fused_ops.py` and the Qwen3 diff against the `v0.5.19` tag were both zero.
- Server readiness was reached and decode graph capture completed without
  `PagedAttentionOperation`.
- One smoke request returned HTTP 200 with a non-empty transcript and
  `latency=12.834s`. The cold first-request latency is recorded as functional
  evidence only and is not a performance claim.
- Normal shutdown was observed.

The follow-up proves that the SGLang fused-op patch is not required for the
current functional path. The pure base `0bcd82237` is the active candidate.
The next bounded change is cold concurrency-8 liveness, followed by the agreed
correctness workload.

## Return

```text
Task status: passed
SGLang HEAD / clean / imported module:
Omni HEAD / clean / imported module:
Resolved compile and graph settings:
Server reached readiness:
Smoke transcript non-empty:
Target path evidence:
Fallback/error markers:
First complete failure and owning repository:
Device health and cleanup:
Server-local artifacts retained:
```

Do not return full logs, model paths, transcripts, host details, or
credentials.
