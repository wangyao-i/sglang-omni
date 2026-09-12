# Qwen3-ASR Ascend v0.5.19 capture classification task

This is the next bounded diagnostic after the exact-head candidate run passed
the focused tests and then stopped at decode graph capture. It is log-first:
analyze the existing first failure before running another arm.

The task has two stages:

- Stage 1 is mandatory: recover the existing failure evidence and identify the
  first failing layer.
- Stage 2 is conditional: run a base-versus-candidate comparison only when the
  recovered logs cannot distinguish two specific competing hypotheses and the
  local owner explicitly requests the experiment.

## Causal Question

What is the first failing layer in the existing exact-head startup failure, and
what is the smallest missing piece of evidence needed to identify its owner?

Do not start an experiment to answer this question. Analyze the failure that
already exists first. Only if that analysis cannot distinguish two named
mechanisms may Stage 2 be approved.

## Inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export EXPECTED_SGLANG_HEAD=0bcd822377da7b5718e674eaf9c870d349424dd1
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export SMOKE_WAV=/server/local/approved-smoke.wav
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-v0519-capture-classification
mkdir -p "${EVIDENCE}"
```

Use the actual checkout roots imported by Python. A legacy directory name is
not a code mismatch. Do not return the values of these variables.

## Fixed Identity

| Repository | Arm | Exact runtime head |
|---|---|---|
| SGLang | Base `v0.5.19` | `0bcd822377da7b5718e674eaf9c870d349424dd1` |
| SGLang | Candidate | `e0011e30fbdb9690f01fa2083d452c93b37bb214` |
| SGLang-Omni | Both arms | `5190678c463f6c2b01e4ee0007cf788c3fdc2287` |

## Known Candidate Result

The candidate arm already produced:

- SGLang `test_fused_ops.py`: `5 passed`.
- Omni `test_encoder_service.py`: `30 passed, 1 skipped`.
- Default startup reached decode graph capture and then stopped with
  `PagedAttentionOperation setup failed` and `Capture cuda graph failed`.

Treat that failure as valid exact-head evidence. Do not modify `fused_ops.py`,
PagedAttention, graph policy, source, or environment during this task.

## Stage 1: Existing-Failure Analysis

Do not start or rerun the model server in this stage.

Recover the original candidate startup failure from the exact run:

- SGLang `e0011e30fbdb9690f01fa2083d452c93b37bb214`;
- Omni `5190678c463f6c2b01e4ee0007cf788c3fdc2287`;
- default `torch.compile` and decode graph settings.

Return redacted evidence for:

1. The first complete ordered traceback, not only the outer
   `RuntimeError`. Include all Python frames and the first frame owned by
   SGLang, SGLang-Omni, torch_npu, torchair, or CANN.
2. The lines immediately before `The Inner error is reported as above`, where
   the actual lower-layer error may appear.
3. The last compile, graph-capture, and operator-setup markers before the
   failure, with enough surrounding lines to identify the capture phase.
4. Which graph capture failed: prefill, decode, or another call site. Include
   the capture call stack and batch size if logged.
5. The first failing operator's available metadata: dtype, shape, stride,
   layout, graph-key or bucket information, and stream/device identifiers.
   Do not return tensor contents.
6. Resolved server arguments and environment values already present in the run
   metadata.
7. Any earlier warning, allocator message, device error, fallback marker, or
   shutdown error that precedes the final `PagedAttentionOperation` report.

Classify the evidence into one provisional category:

- compile or Dynamo tracing;
- graph capture argument construction;
- PagedAttention input or graph-key incompatibility;
- CANN, torchair, or torch_npu failure;
- allocator, heap, stream, or device failure;
- application orchestration;
- insufficient evidence.

For the chosen category, state the strongest supporting lines, the strongest
contrary evidence, and exactly which alternative mechanisms remain
indistinguishable. Do not propose or implement a code fix in Stage 1.

## Stage 2 Gate 0

Do not execute Stage 2 until the local owner approves it in a new task update
or a new handoff commit.

Before each server start:

- Stop the previous process and wait for the port to become free.
- Check out the exact arm head in the actual `${SGLANG_REPO}`.
- Keep `${OMNI_REPO}` at `5190678c463f6c2b01e4ee0007cf788c3fdc2287`.
- Require both worktrees to be clean.
- Verify that the imported `sglang` and `sglang_omni` modules resolve to their
  respective checkout roots.
- Record runtime package versions and environment values once, then keep them
  unchanged for both arms.

Use the Gate 0 identity checks in
[`qwen3_asr_ascend_v0519_validation_task.md`](qwen3_asr_ascend_v0519_validation_task.md).
Set `EXPECTED_SGLANG_HEAD` to the exact head of the arm under test. The
candidate hash in that task is only the default value for the candidate arm.
Stop and return `blocked` if identity cannot be established.

## Stage 2, Arm B: Base v0.5.19

Check out the base commit and start a fresh default process:

```bash
cd "${SGLANG_REPO}"
git checkout --detach 0bcd822377da7b5718e674eaf9c870d349424dd1
test -z "$(git status --porcelain)"

cd "${OMNI_REPO}"
test "$(git rev-parse HEAD)" = \
  "5190678c463f6c2b01e4ee0007cf788c3fdc2287"
test -z "$(git status --porcelain)"

sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
  >"${EVIDENCE}/arm-base-server.log" 2>&1 &
SERVER_PID=$!
```

Keep `torch.compile` enabled and keep the decode graph enabled. Do not add a
fallback, graph-disable flag, alternate model profile, or environment change.

Wait for readiness with the server's standard health check. If startup reaches
readiness, send exactly one transcription request using the same command and
WAV as the validation task. Preserve `SERVER_PID` so the process can be stopped
cleanly after the first failure or completed smoke.

Stop at the first complete failure. Preserve the first traceback, repository
frame, operator name, capture-stage boundary, and process exit state.

## Stage 2 Decision Rule

### Base fails at the same capture point

Stop. Do not run the candidate again.

Conclusion: the candidate fused-op patch is not necessary for the reported
`PagedAttentionOperation` capture failure. Move the investigation to the
`v0.5.19` base or lower runtime layer.

### Base fails earlier or at a different stage

Stop. Do not run the candidate again.

Record the first failing stage, such as Dynamo tracing, model loading, graph
capture, or device setup. The candidate changed the failure boundary, but this
does not prove that the candidate caused the original capture failure.

### Base reaches readiness and completes the smoke request

Stop the base process cleanly and wait for the port to become free. Run the
candidate as a positive control in a fresh process with the same environment,
Omni head, model, and start arguments:

```bash
cd "${SGLANG_REPO}"
git checkout --detach e0011e30fbdb9690f01fa2083d452c93b37bb214
export EXPECTED_SGLANG_HEAD=e0011e30fbdb9690f01fa2083d452c93b37bb214
test -z "$(git status --porcelain)"

sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
  >"${EVIDENCE}/arm-candidate-server.log" 2>&1 &
SERVER_PID=$!
```

Require the candidate capture failure to reproduce.

- Candidate reproduces: the SGLang patch is a valid causal suspect and the next
  task must narrow the specific fused-op interaction.
- Candidate does not reproduce: stop and return both results. The earlier
  failure is not stable enough to support a root-cause change.

## Stop Conditions

In Stage 1, stop after returning the existing-failure analysis. Do not run a
server process.

In an approved Stage 2, stop immediately on:

- Gate 0 identity failure;
- startup failure at any stage;
- `PagedAttentionOperation`, capture, device, ACL, allocator, or stream error;
- readiness timeout;
- request failure, empty transcript, or hang; or
- unexpected eager fallback.

Do not continue from a failed or polluted process.

## Non-Goals

- No server rerun during Stage 1.
- No source or environment edits.
- No `site-packages` edits.
- No disabling `torch.compile` or decode graph.
- No GraphKey, guard, selector, or scheduler work.
- No performance, realtime, timestamp, forced-alignment, or C70 work.
- No bundled test matrix beyond the existing focused-test evidence.

## Cleanup

If Stage 2 was approved and executed, restore `${SGLANG_REPO}` to the active
candidate after preserving logs and the first failure:

```bash
git -C "${SGLANG_REPO}" checkout --detach \
  e0011e30fbdb9690f01fa2083d452c93b37bb214
test -z "$(git -C "${SGLANG_REPO}" status --porcelain)"
```

## Return

Stage 1 return:

```text
Task status: evidence_returned / blocked
Existing candidate run located:
SGLang and Omni identities:
First complete traceback frames:
Lower-layer or inner error:
First repository frame and owning repository:
Failure stage and capture call site:
Operator metadata available:
Resolved arguments and environment:
Provisional category:
Supporting evidence:
Contrary evidence:
Remaining indistinguishable mechanisms:
Missing evidence:
Requested next action:
Server-local artifacts retained:
```

Stage 2 return, only after explicit approval:

```text
Task status: completed / blocked
SGLang arm:
SGLang observed HEAD / clean / imported module:
Omni observed HEAD / clean / imported module:
Runtime and environment unchanged:
Startup reached readiness:
Smoke result:
First failing stage:
First complete failure signature:
First repository frame:
Same as prior candidate failure:
Candidate positive control run:
Device health and cleanup:
Server-local artifacts retained:
```

Do not return full logs, model paths, audio, transcripts, host details, or
credentials.
