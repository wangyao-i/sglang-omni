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
| SGLang | `v0.5.19` release line is the only active runtime baseline; the candidate patch is `e0011e30` on tag commit `0bcd82237` |
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
exact-head run stopped at a first complete failure; **Deferred** means
explicitly outside the current phase.

| Workstream | Status | Current state | Exit condition |
|---|---|---|---|
| Omni NPU encoder private stream | Implemented | `codex/qwen3-asr-npu-encoder-stream-v0519` uses code commit `5190678c`; focused stream tests passed previously and remain applicable | Complete a request on the release-line runtime |
| External fused-kernel compile boundary | Implemented | `codex/qwen3-asr-v0519-fused-op` at `e0011e30` ports the same three-file boundary onto `v0.5.19` | Pass exact-head focused tests and default startup |
| SGLang v0.5.19 validation | Failed | Identity is confirmed: the legacy-named worktree is clean at `e0011e30` and Python imports from that checkout. Focused tests passed; default startup then stopped at `PagedAttentionOperation setup failed` during decode graph capture | Complete the log-first analysis in [`qwen3_asr_ascend_v0519_capture_classification_task.md`](qwen3_asr_ascend_v0519_capture_classification_task.md); run the conditional A/B only after local approval |
| NPU installer `0.5.19` alignment | Done | `install_npu.sh`, installation docs, `pyproject_npu.toml`, and installer tests now target `0.5.19` | Run the installer suite in a Linux CI environment |
| Qwen3-ASR feature matrix | Done | [`qwen3_asr_feature_matrix.md`](qwen3_asr_feature_matrix.md) records model, Omni/CUDA, NPU implementation, and NPU qualification separately | Refresh rows when exact-head server evidence arrives |
| Combined NPU liveness and correctness | Planned | No current v0.5.19 result exists | Pass default smoke first, then cold concurrency-8 liveness and correctness on exact pinned heads |
| SGLang main interface experiment | Historical | Main exposed `scheduler_stage_metrics` drift and older capture failures; the baseline is abandoned | Do not use for current acceptance |
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
- Completed: the active SGLang candidate is based on tag commit `0bcd82237`,
  not main.
- Pin the exact Omni and SGLang heads in the next hardware task before running
  any server command.

### Phase 1: Feature matrix

- Completed: one row per externally meaningful feature is recorded in
  [`qwen3_asr_feature_matrix.md`](qwen3_asr_feature_matrix.md).
- Model capability, stack implementation, and hardware qualification are
  separate columns.

### Phase 2: Minimal SGLang repair

- Apply the current three-file fused-op boundary to the `v0.5.19` release
  line.
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
- On the isolated NPU server, pass the hardened Gate 0 repository, editable
  distribution, and imported-module identity checks before any pytest or
  server command.
- Identify a checkout by exact HEAD, clean worktree, and imported module path,
  not by directory name. A legacy worktree name is acceptable when all three
  checks pass.
- On the isolated NPU server, first run default startup and one smoke request.
- Stop at the first failure and classify before adding any more variants.
- After the default smoke passes, run cold concurrency-8 liveness and the
  agreed
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
| SGLang main interface drift | Historical, resolved by baseline change | Main added `scheduler_stage_metrics`; the Omni composition layer lacked it. This cannot occur on `v0.5.19` |
| Main-baseline PagedAttention and heap-corruption failure | Historical, needs release-line recheck | It was observed on main and is not current evidence for the `v0.5.19` candidate |
| Fused-op tensor-layout mismatch | Unproven | The exact-head startup failure is valid, but no single-variable comparison connects it to `fused_ops.py`; `PagedAttentionOperation` is an asynchronous report |
| Release-line capture failure at `PagedAttentionOperation` | Active first failure, owner unknown | Exact candidate `e0011e30` reached decode graph capture, then reported operator setup failure and `Capture cuda graph failed` |

The next task must preserve this first failure and extract the complete ordered
traceback, lower-layer error, capture call site, and first owning repository
frame. A base-versus-candidate comparison is conditional and requires a new
local decision. No code change is selected yet.

## Roadmap Maintenance

- Update this page when a workstream changes state, not after every diagnostic
  command.
- Every status change must name the exact commit, evidence scope, and whether
  the result applies to the current heads.
- Historical results stay historical until rerun on the pinned candidate.
- If a new change is not needed for the narrow problem statement above, add it
  to the deferred section instead of expanding the active patch.
