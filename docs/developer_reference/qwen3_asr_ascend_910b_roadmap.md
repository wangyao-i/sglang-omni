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
| SGLang | `main` is accepted as the development baseline; use the exact tested head in every result |
| SGLang dependency | `sglang==0.5.19` |
| NPU runtime | CANN, PyTorch, torch_npu, triton-ascend, and sgl-kernel-npu must be selected from their compatibility matrices |
| Performance | Deferred until the functional and correctness path is complete |

The NPU installer and its documentation have been aligned to the SGLang
`0.5.19` release line. The installer test matrix now accepts `0.5.19`
development, pre-release, final, post-release, and local-build spellings and
rejects earlier and later release lines.

## Status

Legend: **Done** means implemented and verified at the stated scope;
**Implemented** means code exists but current-head qualification is pending;
**Planned** means agreed work that has not started; **Historical** means useful
evidence that does not apply to the current heads; **Deferred** means explicitly
outside the current phase.

| Workstream | Status | Current state | Exit condition |
|---|---|---|---|
| Omni NPU encoder private stream | Implemented | `codex/qwen3-asr-npu-encoder-stream` at `5190678c` passed `30 passed, 1 skipped` focused tests on the server; end-to-end execution is blocked by startup classification | Complete a request on the classified runtime |
| External fused-kernel compile boundary | Implemented | `codex/qwen3-asr-compile-safe-fused-op` at `85e8933d` passed `5 passed` focused server tests; startup failure attribution is unresolved | Pass exact-head startup and classify whether the failure depends on compile |
| Startup-failure classification | Partial | Compile-off/decode-on completed both graph captures; the first failure was then `OmniScheduler` missing upstream `scheduler_stage_metrics`, not the original attention/heap-corruption signature | Retry exact candidate `24552a65` through compile/graph gates |
| Omni scheduler stage-metrics compatibility | Implemented | `24552a65` mirrors the main-only reporter field when available and uses `None` on the declared `sglang==0.5.19` line; focused tests cover both layouts | Pass the exact-head scheduler tests and R1 startup |
| NPU installer `0.5.19` alignment | Done | `install_npu.sh`, installation docs, `pyproject_npu.toml`, and installer tests now target `0.5.19` | Run the installer suite in a Linux CI environment |
| Qwen3-ASR feature matrix | Done | [`qwen3_asr_feature_matrix.md`](qwen3_asr_feature_matrix.md) records model, Omni/CUDA, NPU implementation, and NPU qualification separately | Refresh rows when exact-head server evidence arrives |
| Combined NPU liveness and correctness | Blocked | Startup classification reached the Scheduler API compatibility fix; liveness and correctness are not yet rerun | Pass the retry gates, then cold concurrency-8 liveness and correctness on exact pinned heads |
| Historical `910C-071` result | Historical | It qualified the old combined candidate, but later scope removal changed the candidate and the result is not current-head evidence | Replace it with a new exact-head result or leave it clearly historical |
| Performance target | Deferred | No performance work is part of the current acceptance path | Reopen only after functionality and correctness close |
| Realtime ASR | Deferred | It is not required for the current offline serving fix | Define a separate protocol and acceptance gate before implementation |

## Problem Statement

The retained problem is narrow:

1. On Ascend, Qwen3's NPU path calls `split_qkv_rmsnorm_rope` from
   `sgl_kernel_npu` while SGLang compiles the model with `fullgraph=True`.
2. Dynamo can trace into the external Triton launcher and reject host-side
   launch logic such as `get_device_properties()`.
3. The boundary must make the external kernel opaque to Dynamo while preserving
   the real kernel's arguments and outputs at runtime.
4. The encoder must not submit audio work on the generation submission lane;
   the private device stream addresses that independent NPU liveness issue.

The following are not part of this problem: GraphKey, execution guards,
ordered graph input update, compile-bucket selectors, TopK/RMSNorm/Silu
dispatch overrides, and global `prepare_model_for_torch_compile` changes.

## Execution Plan

### Phase 0: Baseline alignment

- Completed: the NPU installer, installation documentation, and installer
  tests now use `0.5.19`.
- Pin the exact Omni and SGLang heads in the next hardware task before running
  any server command.

### Phase 1: Feature matrix

- Completed: one row per externally meaningful feature is recorded in
  [`qwen3_asr_feature_matrix.md`](qwen3_asr_feature_matrix.md).
- Model capability, stack implementation, and hardware qualification are
  separate columns.

### Phase 2: Minimal SGLang repair

- Rebase onto current SGLang `main`.
- Keep only the external fused-kernel compile boundary.
- Prefer the existing `register_custom_op_from_extern()` helper; evaluate
  registering the op in `sgl_kernel_npu` as the longer-term owner.
- Keep a focused fake-tensor/opacity test and a value-parity test for the
  compiled path.
- Do not restore removed graph, guard, selector, or dispatch changes without
  new isolating evidence.

### Phase 3: Omni branch cleanup

- Rebase the private-stream change onto current Omni main.
- Keep the change platform-neutral and preserve CUDA behavior.
- Keep the focused encoder-service tests with the implementation.

### Phase 4: Exact-head validation

- Run local static checks and focused tests first.
- Treat the first startup failure as a classification input, not as evidence
  for an implementation repair.
- Run the single-variable failure classification before changing production
  code.
- On the isolated NPU server, run cold concurrency-8 liveness and the agreed
  correctness workload.
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
- the external fused kernel is opaque in the compiled graph and preserves
  values;
- the Omni encoder uses its private device stream without the removed guard;
- cold concurrency-8 liveness passes on the exact pinned heads; and
- the correctness workload passes with no garbled output and no unexpected
  fallback.

Performance measurements, C70 throughput targets, realtime ASR, timestamps,
and forced alignment remain outside this gate.

## Current Root-Cause Board

| Claim | Status | Evidence |
|---|---|---|
| Custom fused-op boundary changed PagedAttention tensor layout | Rejected for the current observed first failure | Compile-disabled execution passed prefill and decode capture; the failure moved to the Omni scheduler before smoke |
| `OmniScheduler` missed an upstream instance contract | Confirmed | The decorated upstream scheduler method reads `self.scheduler_stage_metrics`; `24552a65` mirrors it on main, preserves v0.5.19 compatibility, and adds regression tests |
| Compile-enabled graph path has an independent defect | Pending | R2 and R3 retry the compile separation after the confirmed Scheduler fix |
| Non-interactive launch dropped an allocator or library setting | Alternative | Public Ascend reports show the same heap-corruption signal when `LD_PRELOAD` or related runtime settings apply only to an interactive shell |
| Asynchronous operator attribution is misleading | Diagnostic caveat | Ascend can report the previous `PagedAttentionOperation` when the real fault is asynchronous; use `ASCEND_LAUNCH_BLOCKING=1` only in the follow-up diagnostic |
| `torch.compile` is required for the original capture failure | Pending | R2 disables the decode graph while compile stays enabled |
| Decode graph is required for the original capture failure | Pending | R3 returns to the original compile-plus-decode-graph combination |

The owner is not assigned until the bounded classification returns a first
complete failure and the first repository frame that owns it.

## Roadmap Maintenance

- Update this page when a workstream changes state, not after every diagnostic
  command.
- Every status change must name the exact commit, evidence scope, and whether
  the result applies to the current heads.
- Historical results stay historical until rerun on the pinned candidate.
- If a new change is not needed for the narrow problem statement above, add it
  to the deferred section instead of expanding the active patch.
