# Qwen3-ASR on Ascend roadmap

This page is the current decision and status surface for the Qwen3-ASR Ascend
effort. Detailed hardware evidence belongs in the hardware handoff or task
pages; this roadmap records what is complete, what is still pending, and which
work is deliberately deferred.

Last updated: 2026-09-12.

## Baseline

| Item | Current decision |
|---|---|
| SGLang-Omni | `upstream/main`, currently `6ff46426`; rebase the Omni branch before the final hardware run |
| SGLang | Pure `v0.5.19` tag commit `0bcd82237` is the active runtime baseline; the fused-op patch `e0011e30` is deferred |
| SGLang dependency | `sglang==0.5.19` |
| NPU runtime | CANN, PyTorch, torch_npu, triton-ascend, and sgl-kernel-npu must be selected from their compatibility matrices |
| Performance | Deferred until the functional and correctness path is complete |

The NPU installer and its documentation have been aligned to the SGLang
`0.5.19` release line. The installer test matrix now accepts `0.5.19`
development, pre-release, final, post-release, and local-build spellings and
rejects earlier and later release lines.

SGLang main is no longer an active baseline. The main-only
`scheduler_stage_metrics` interface drift and the main-baseline capture results
are retained as historical evidence only.

## Status

Legend: **Done** means implemented and verified at the stated scope;
**Implemented** means code exists but current-head qualification is pending;
**Planned** means agreed work that has not started; **Historical** means useful
evidence that does not apply to the current heads; **Failed** means an
exact-head run stopped at a first complete failure; **Partial** means the
functional gate passed but a cleanup or qualification gate remains open;
**Deferred** means explicitly outside the current phase.

| Workstream | Status | Current state | Exit condition |
|---|---|---|---|
| Omni NPU encoder private stream | Done | `codex/qwen3-asr-npu-encoder-stream-v0519` uses code commit `5190678c`; focused stream tests pass and a request completed on the pure `v0.5.19` runtime | Keep the implementation under final PR-head validation |
| External fused-kernel compile boundary | Deferred | Pure `v0.5.19` passes graph-only, so `e0011e30` is not required by the active functional path | Keep [#38843](https://github.com/sgl-project/sglang/pull/38843) deferred unless a later compile performance task proves the boundary necessary |
| SGLang v0.5.19 validation | Done | Pure tag commit `0bcd82237`, with no fused_ops or Qwen3 diff against the tag, reached readiness, captured the decode graph, returned a non-empty smoke transcript, and completed normal shutdown | Preserve the pure base as the active runtime baseline |
| NPU installer `0.5.19` alignment | Done | `install_npu.sh`, installation docs, `pyproject_npu.toml`, and installer tests now target `0.5.19` | Run the installer suite in a Linux CI environment |
| Qwen3-ASR feature matrix | Done | [`qwen3_asr_feature_matrix.md`](qwen3_asr_feature_matrix.md) records model, Omni/CUDA, NPU implementation, and NPU qualification separately | Refresh rows when exact-head server evidence arrives |
| Combined NPU liveness and correctness | Partial | Pure `v0.5.19` + Omni `5190678c` passed graph-only startup, smoke, and the 70-request cold concurrency-8 gate; a first correctness attempt used the known non-equivalent exact10 concurrency-1 protocol and WER `0.0784`, so it is not accepted | Confirm cleanup, recover the already-qualified 140-request invocation, then run that exact protocol on the pure base as specified in [`qwen3_asr_ascend_v0519_liveness_correctness_task.md`](qwen3_asr_ascend_v0519_liveness_correctness_task.md) |
| SGLang main interface experiment | Historical | Main exposed `scheduler_stage_metrics` drift and older capture failures; the baseline is abandoned | Do not use for current acceptance |
| Historical `910C-071` result | Historical | It qualified the old combined candidate, but later scope removal changed the candidate and the result is not current-head evidence | Replace it with a new exact-head result or leave it clearly historical |
| Performance target | Deferred | No performance work is part of the current acceptance path | Reopen only after functionality and correctness close |
| Realtime ASR | Deferred | It is not required for the current offline serving fix | Define a separate protocol and acceptance gate before implementation |

## Problem Statement

The retained functional problem is now narrower than the original
compile-supported stack:

1. Pure SGLang `v0.5.19` must run Qwen3-ASR with the NPU decode graph enabled
   and `enable_torch_compile=false`.
2. The Omni encoder must submit audio work without conflicting with generation
   submission; the private device stream is the retained mechanism.
3. The pure-base candidate must pass cold concurrency-8 liveness and the agreed
   correctness workload.

The compile-safe fused-op boundary, GraphKey, execution guards, ordered graph
input update, compile-bucket selectors, TopK/RMSNorm/Silu dispatch overrides,
and global `prepare_model_for_torch_compile` changes are not part of the active
functional problem. They remain deferred or rejected unless a future
performance task supplies independent evidence.

## Execution Plan

### Phase 0: Baseline alignment

- Completed: the NPU installer, installation documentation, and installer
  tests now use `0.5.19`.
- Completed: the active SGLang candidate is based on tag commit `0bcd82237`,
  not main.
- Pin the exact Omni and SGLang heads in the next hardware task before running
  any server command.

### Phase 1: Feature matrix

- Completed: one row per externally meaningful feature is recorded in
  [`qwen3_asr_feature_matrix.md`](qwen3_asr_feature_matrix.md).
- Model capability, stack implementation, and hardware qualification are
  separate columns.

### Phase 2: Pure v0.5.19 minimal baseline

- Completed: pure SGLang `v0.5.19` commit `0bcd82237` is the active candidate.
- Completed: the SGLang checkout has no `fused_ops.py` or Qwen3 diff against
  the tag.
- Completed: readiness, decode graph capture, one non-empty smoke request, and
  normal shutdown pass with `enable_torch_compile=false`.
- Archived: the three-file fused-op boundary at `e0011e30` is deferred and is
  no longer a prerequisite for the functional path.
- Do not restore graph, guard, selector, dispatch, or compile-boundary changes
  without a new isolating experiment.

### Phase 3: Omni branch cleanup

- Rebase the private-stream change onto current Omni main.
- Keep the change platform-neutral and preserve CUDA behavior.
- Keep the focused encoder-service tests with the implementation.

### Phase 4: Exact-head validation

- Run local static checks and focused tests first.
- On the isolated NPU server, pass the hardened Gate 0 repository, editable
  distribution, and imported-module identity checks before any pytest or
  server command.
- Identify a checkout by exact HEAD, clean worktree, and imported module path,
  not by directory name. A legacy worktree name is acceptable when all three
  checks pass.
- On the isolated NPU server, first run default startup and one smoke request.
- Stop at the first failure and classify before adding any more variants.
- The pure-base graph-only smoke has passed. Run cold concurrency-8 liveness
  and the agreed correctness workload next.
- Record exact repository heads, dependency versions, and sanitized results.
- Stop at the first hang, accuracy failure, device error, or unexpected eager
  fallback.

### Phase 5: Upstream preparation

- Rebase and split the SGLang and Omni changes into reviewable units.
- Make the PR narrative state only what the current evidence proves.
- Remove development diagnostics, stale performance claims, and superseded
  candidate identities from the final description.

## Validation Gates

The current phase is complete only when:

- the NPU installer accepts the declared `0.5.19` dependency line;
- pure SGLang `v0.5.19` passes graph-only startup and decode capture without
  the fused-op patch;
- the Omni encoder uses its private device stream without the removed guard;
- cold concurrency-8 liveness passes on the exact pure-base heads; and
- the correctness workload passes with no garbled output and no unexpected
  fallback.

The pure-base startup and smoke gate is complete. Performance measurements,
C70 throughput targets, realtime ASR, timestamps, and forced alignment remain
outside this gate.

## Current Root-Cause Board

| Claim | Status | Evidence |
|---|---|---|
| SGLang main interface drift | Historical, resolved by baseline change | Main added `scheduler_stage_metrics`; the Omni composition layer lacked it. This cannot occur on `v0.5.19` |
| Main-baseline PagedAttention and heap-corruption failure | Historical, needs release-line recheck | It was observed on main and is not current evidence for the `v0.5.19` candidate |
| Fused-op patch is required for graph-only | Rejected | Pure `v0.5.19` commit `0bcd82237` has no fused-op or Qwen3 diff and passes readiness, decode capture, and smoke |
| Compile-enabled capture failure at `PagedAttentionOperation` | Scope resolved | The failure reproduces only with `torch.compile` enabled; graph-only with compile disabled passes exact-head startup and smoke |
| HBM residual after graph-only shutdown | Resolved as external interference | The holder was identified as a residual process and cleaned; it is not attributed to the graph-only run |
| Generic Ascend Qwen3 evidence covers this run | Scope difference, not root cause | Ascend Qwen3 e2e uses eager decode without torch.compile; Qwen3-ASR enables both compile and decode graph, and the NPU attention implementation selects a different branch when compile is enabled |
| `torch.compile` is required for Qwen3-ASR | Rejected as a functional requirement | It was introduced as a CUDA low/mid-concurrency performance optimization; the stage default is `enable_torch_compile=False` |

The pure `v0.5.19` graph-only startup and smoke gates are closed. The next work
is cold concurrency-8 liveness and correctness on `0bcd82237` + Omni
`5190678c`. No compile-supported NPU claim and no fused-op requirement are
active.

## Roadmap Maintenance

- Update this page when a workstream changes state, not after every diagnostic
  command.
- Every status change must name the exact commit, evidence scope, and whether
  the result applies to the current heads.
- Historical results stay historical until rerun on the pinned candidate.
- If a new change is not needed for the narrow problem statement above, add it
  to the deferred section instead of expanding the active patch.
