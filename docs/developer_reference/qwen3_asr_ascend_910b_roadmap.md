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
exact-head run stopped at a first complete failure; **Partial** means the
functional gate passed but a cleanup or qualification gate remains open;
**Deferred** means explicitly outside the current phase.

| Workstream | Status | Current state | Exit condition |
|---|---|---|---|
| Omni NPU encoder private stream | Implemented | `codex/qwen3-asr-npu-encoder-stream-v0519` uses code commit `5190678c`; focused stream tests passed previously and remain applicable | Complete a request on the release-line runtime |
| External fused-kernel compile boundary | Deferred | `e0011e30` contains the three-file boundary, but the passing NPU graph-only path does not require compile or this patch | Remove or park the patch unless a future compile performance task proves it necessary |
| SGLang v0.5.19 validation | Done | Graph-only passed readiness, decode capture, non-empty smoke, shutdown, and cleanup on `e0011e30` + `5190678c`; the residual HBM holder was an external process that was cleaned | Make graph-only the explicit NPU profile and test the pure `v0.5.19` base after removing the unnecessary SGLang compile patch |
| NPU installer `0.5.19` alignment | Done | `install_npu.sh`, installation docs, `pyproject_npu.toml`, and installer tests now target `0.5.19` | Run the installer suite in a Linux CI environment |
| Qwen3-ASR feature matrix | Done | [`qwen3_asr_feature_matrix.md`](qwen3_asr_feature_matrix.md) records model, Omni/CUDA, NPU implementation, and NPU qualification separately | Refresh rows when exact-head server evidence arrives |
| Combined NPU liveness and correctness | Planned | Graph-only startup, one smoke request, and cleanup pass; the explicit NPU profile and pure-base candidate remain open | Pin the graph-only profile, remove the unused SGLang patch, then run cold concurrency-8 liveness and correctness on exact heads |
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
| Fused-op tensor-layout mismatch | Rejected for the graph-only path | Graph-only passes with the same fused-op code, and the compile-enabled failure does not prove a tensor-layout mismatch |
| Compile-enabled capture failure at `PagedAttentionOperation` | Scope resolved | The failure reproduces only with `torch.compile` enabled; graph-only with compile disabled passes exact-head startup and smoke |
| HBM residual after graph-only shutdown | Resolved as external interference | The holder was identified as a residual process and cleaned; it is not attributed to the graph-only run |
| Generic Ascend Qwen3 evidence covers this run | Scope difference, not root cause | Ascend Qwen3 e2e uses eager decode without torch.compile; Qwen3-ASR enables both compile and decode graph, and the NPU attention implementation selects a different branch when compile is enabled |
| `torch.compile` is required for Qwen3-ASR | Rejected as a functional requirement | It was introduced as a CUDA low/mid-concurrency performance optimization; the stage default is `enable_torch_compile=False` |

The graph-only function and cleanup gates are closed. The next work is an
explicit NPU graph-only profile and a pure `v0.5.19` candidate that removes the
SGLang compile-boundary patch. No compile-supported NPU claim is active.

## Roadmap Maintenance

- Update this page when a workstream changes state, not after every diagnostic
  command.
- Every status change must name the exact commit, evidence scope, and whether
  the result applies to the current heads.
- Historical results stay historical until rerun on the pinned candidate.
- If a new change is not needed for the narrow problem statement above, add it
  to the deferred section instead of expanding the active patch.
