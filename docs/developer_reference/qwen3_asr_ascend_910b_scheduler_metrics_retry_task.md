# Qwen3-ASR Ascend scheduler-metrics retry task

Run this task after reading
[`qwen3_asr_ascend_910b_handoff.md`](qwen3_asr_ascend_910b_handoff.md) and
checking out the exact heads below.

## Inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export SMOKE_WAV=/server/local/approved-smoke.wav
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-scheduler-metrics-retry
mkdir -p "${EVIDENCE}"
```

Do not return the values of these variables.

## Exact Identity

| Repository | Exact runtime head |
|---|---|
| SGLang | `85e8933dc3ae4ecd44a1e0ebf4595f0fd9011595` |
| SGLang-Omni | `24552a65` |

The Omni fix is built on the previously validated code commit:

```text
5190678c fix(qwen3-asr): run the NPU encoder on a private device stream
24552a65 fix(scheduler): mirror upstream stage metrics state
```

When SGLang main provides the stage-metrics module, the fix mirrors SGLang's
`self.scheduler_stage_metrics = self.metrics_reporter.scheduler_stage_metrics`
after metrics-reporter initialization. When running the declared
`sglang==0.5.19` line, where that main-only instance contract does not exist,
it initializes the field to `None` instead. It does not change SGLang, graph
capture, attention, or the fused-op boundary.

## Gate 0

- Confirm both repositories are detached at the exact heads.
- Require a clean tracked worktree. If `fusion_result.json` is still
  untracked, preserve it outside the checkout or use a fresh clean checkout;
  it is not part of this candidate.
- Record Python, CANN, torch, torch_npu, triton-ascend, and `sgl_kernel_npu`
  versions. `triton` and CANN must no longer be reported as uncollected.
- Record `LD_PRELOAD`, `LD_LIBRARY_PATH`, `PYTORCH_NPU_ALLOC_CONF`,
  `STREAMS_PER_DEVICE`, `HCCL_BUFFSIZE`, and `ASCEND_LAUNCH_BLOCKING` without
  changing them.
- Confirm no residual process, port `8000` free, and the selected NPU healthy.

Run the focused scheduler tests first:

```bash
cd "${OMNI_REPO}"
python -m pytest -q \
  tests/unit_test/pipeline/test_scheduler.py::test_omni_scheduler_mirrors_upstream_scheduler_stage_metrics \
  tests/unit_test/pipeline/test_scheduler.py::test_omni_scheduler_supports_v0519_without_scheduler_stage_metrics \
  tests/unit_test/pipeline/test_scheduler.py::test_omni_scheduler_initializes_upstream_queue_limit \
  >"${EVIDENCE}/test-scheduler-focus.log" 2>&1
```

Stop and return `blocked` if either test fails or the environment cannot be
identified.

## Gate R1: Compile Off, Decode Graph On

This is the same startup mode that previously captured both graphs and then
failed in `OmniScheduler`:

```bash
sgl-omni config resolve \
  --model-path "${MODEL_PATH}" \
  --asr.engine.enable_torch_compile false \
  >"${EVIDENCE}/r1-config.yaml"

grep -n 'enable_torch_compile: false' "${EVIDENCE}/r1-config.yaml"

sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
  --asr.engine.enable_torch_compile false \
  >"${EVIDENCE}/r1-server.log" 2>&1
```

If ready, send one smoke request using the command from
[`qwen3_asr_ascend_910b_failure_classification_task.md`](qwen3_asr_ascend_910b_failure_classification_task.md),
then stop the server normally.

Pass requires readiness, one non-empty transcript, no scheduler
`AttributeError`, no graph capture error, and clean shutdown.

Stop immediately on the first failure. Do not run R2 or R3.

## Gate R2: Compile On, Decode Graph Off

Run only if R1 passes.

```bash
sgl-omni config resolve \
  --model-path "${MODEL_PATH}" \
  --asr.engine.enable_torch_compile true \
  --asr.engine.cuda_graph_backend_decode disabled \
  >"${EVIDENCE}/r2-config.yaml"

grep -nE 'enable_torch_compile: true|cuda_graph_backend_decode: disabled' \
  "${EVIDENCE}/r2-config.yaml"

sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
  --asr.engine.enable_torch_compile true \
  --asr.engine.cuda_graph_backend_decode disabled \
  >"${EVIDENCE}/r2-server.log" 2>&1
```

Use a fresh process, one smoke request, and normal shutdown. Stop immediately
on the first failure.

## Gate R3: Default Compile And Decode Graph

Run only if R2 passes. This is the original startup combination:

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
  >"${EVIDENCE}/r3-server.log" 2>&1
```

Use a fresh process. Confirm through the startup log and `/server_info` that
`torch.compile` and decode graph are enabled rather than silently skipped.
Send one smoke request and stop normally.

## Stop Conditions

Stop at the first complete failure and do not run later gates:

- focused scheduler test failure;
- startup failure, scheduler `AttributeError`, or graph capture failure;
- request failure, empty transcript, hang, or timeout;
- device, ACL, OOM, allocator, or stream error;
- unexpected eager fallback; or
- non-clean shutdown.

## Interpretation

| First failing gate | Classification |
|---|---|
| R1 | The scheduler-metrics fix is incomplete or another integration contract changed |
| R2 | The failure depends on compile-enabled execution and is independent of decode graph capture |
| R3 | The failure requires the compile plus decode-graph combination |
| None | The current candidate passes the startup/smoke classification; continue with the separate liveness and correctness gates |

## Non-Goals

- No source edits on the server.
- No changes to the SGLang fused-op boundary.
- No performance, concurrency-70, realtime, timestamp, or forced-alignment
  work.

## Return

Return the exact repository heads and worktree state, runtime versions,
environment values, focused test counts, and one line for each executed gate:
ready/failed, smoke result, forbidden signatures, and cleanup state. Include
the first complete failure and first repository frame if any gate fails. Keep
full logs, paths, audio, and response bodies server-local.
