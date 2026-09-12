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
| Omni NPU encoder private stream | Implemented | `codex/qwen3-asr-npu-encoder-stream` at `5190678c` is rebased onto Omni main and includes focused stream/record tests; current-head NPU execution is pending | Pass the focused unit tests on the server |
| External fused-kernel compile boundary | Implemented | `codex/qwen3-asr-compile-safe-fused-op` at `85e8933d` contains the minimal wrapper on current SGLang main and uses `register_custom_op_from_extern` | Pass the NPU trace, registration, and compiled-value parity tests on the server |
| NPU installer `0.5.19` alignment | Done | `install_npu.sh`, installation docs, `pyproject_npu.toml`, and installer tests now target `0.5.19` | Run the installer suite in a Linux CI environment |
| Qwen3-ASR feature matrix | Done | [`qwen3_asr_feature_matrix.md`](qwen3_asr_feature_matrix.md) records model, Omni/CUDA, NPU implementation, and NPU qualification separately | Refresh rows when exact-head server evidence arrives |
| Combined NPU liveness and correctness | Planned | No current-head combined result exists | Cold concurrency-8 liveness and the agreed correctness gate pass on exact pinned heads |
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

## Roadmap Maintenance

- Update this page when a workstream changes state, not after every diagnostic
  command.
- Every status change must name the exact commit, evidence scope, and whether
  the result applies to the current heads.
- Historical results stay historical until rerun on the pinned candidate.
- If a new change is not needed for the narrow problem statement above, add it
  to the deferred section instead of expanding the active patch.
