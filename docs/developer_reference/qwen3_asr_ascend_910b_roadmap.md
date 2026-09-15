# Qwen3-ASR on Ascend roadmap

This page is the current decision and status surface for the Qwen3-ASR Ascend
effort. Detailed hardware evidence belongs in the hardware handoff or task
pages; this roadmap records what is complete, what is still pending, and which
work is deliberately deferred.

Last updated: 2026-09-15.

## Priority Reset

The WER-protocol discrepancy between `0.078` and the CUDA CI threshold was
already classified in `910C-062`. Continuing to refine that protocol before the
minimal NPU deliverable is reviewable has low information value. The priority
order is now:

1. **P0 - retain the merged minimal functional path.** Pure SGLang `v0.5.19`
   plus the merged [#2084](https://github.com/sgl-project/sglang-omni/pull/2084)
   encoder private stream, with decode graph enabled and Torch Compile
   disabled. No fused-op patch.
2. **P1 - qualify the all-graph functional path.** Run encoder, prefill, and
   decode graphs together on the pinned follow-up candidate, still with Torch
   Compile off. The functional gate is one clean cold concurrency-8 pass over
   140 requests, with liveness and correctness assertions, positive graph
   evidence, and clean shutdown.
3. **P2 - keep merged work frozen.** Do not modify, rebase, or rewrite the
   merged #2084 history. New capabilities use follow-up branches and PRs.
4. **P3 - qualify realtime on the graph-only profile.** The realtime session
   and Qwen3-ASR streaming strategy are already device-agnostic. Run a bounded
   protocol and graph-activity gate on the same runtime; do not fork an NPU
   session implementation.
5. **P4 - keep timestamps separate.** Word and character timestamps require
   `Qwen3-ForcedAligner-0.6B`; they are not part of all-graph or realtime
   support.
6. **P5 - defer stability and performance.** Multi-run soak, graph-registry
   convergence, HBM trend analysis, Torch Compile, C70, throughput, p95, and
   latency measurement use dedicated tasks after the functional paths settle.

The exact WER evaluation protocol is a known non-blocking verification issue.
The current functional acceptance uses `140/140` requests, zero empty
hypotheses, zero outputs with WER above `0.5`, and no execution or fallback
errors. Exact corpus/protocol reconciliation is deferred.

## Baseline

| Item | Current decision |
|---|---|
| SGLang-Omni merged base | `upstream/main@886ced95` contains merged #2084 |
| SGLang-Omni server validation base | PR #2016 at `18c8cfd2eeeb495569426875a2e2bf4114133caf` |
| SGLang-Omni PR #2160 candidate | `1638c5dddb012686210f85ed3ee050fed1ac4597`; frozen for maintainer review |
| SGLang-Omni all-graph/realtime integration code | `b8a37792829ef402edf7b5c83136ab5c804cf5af`; combines PR #2016, frozen PR #2160, the NPU graph-only profile, and the bounded Qwen3-ASR final-prefix change |
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
exact-head evidence sequence is incomplete because some gates passed and the
remaining result is pending, invalidated, or blocked; **Deferred** means
explicitly outside the current phase.

| Workstream | Status | Current state | Exit condition |
|---|---|---|---|
| Omni NPU encoder private stream | Done | Merged in [#2084](https://github.com/sgl-project/sglang-omni/pull/2084) through merge commit `886ced95`; the prior branch was reviewed and frozen before merge | Do not rewrite the merged history; use a follow-up PR for new behavior |
| External fused-kernel compile boundary | Deferred | Pure `v0.5.19` passes graph-only, so `e0011e30` is not required by the active functional path | Keep [#38843](https://github.com/sgl-project/sglang/pull/38843) deferred unless a later compile performance task proves the boundary necessary |
| SGLang v0.5.19 validation | Done | Pure tag commit `0bcd82237`, with no fused_ops or Qwen3 diff against the tag, reached readiness, captured the decode graph, returned a non-empty smoke transcript, and completed normal shutdown | Preserve the pure base as the active runtime baseline |
| NPU installer `0.5.19` alignment | Done | `install_npu.sh`, installation docs, `pyproject_npu.toml`, and installer tests now target `0.5.19` | Run the installer suite in a Linux CI environment |
| Qwen3-ASR feature matrix | Done | [`qwen3_asr_feature_matrix.md`](qwen3_asr_feature_matrix.md) records model, Omni/CUDA, NPU implementation, and NPU qualification separately | Refresh rows when exact-head server evidence arrives |
| Combined NPU liveness and correctness | Historical | Pure `v0.5.19` + the pre-merge Omni code `5190678c` passed graph-only startup, smoke, cold concurrency-8 `70/70`, and a 140-request pass with `140/140`, zero empty outputs, and zero garbled outputs | Re-attest the merged-main baseline when the all-graph task runs; the exact WER protocol remains deferred |
| NPU encoder graph | Implemented | PR #2160 at `1638c5dd` contains lazy exact-layout capture, a shared NPU graph pool, a global 32-graph bound, and fail-fast capture errors | Re-run the all-graph task on the PR #2016 integration code head `b8a37792` with zero unexpected fallback markers |
| Encoder graph layout-key consolidation | Deferred | The current qualification keeps the exact `(bucket_size, effective_window_lens)` key. vLLM's encoder graph manager illustrates a budget-key plus replay-buffer contract, but Qwen3-ASR has no reusable audio encoder graph hook; the relevant NPU precedent is vLLM Ascend's FIA graph-task update path | After the exact-layout all-graph gate passes, prove in an isolated POC that one graph per token bucket can replay two window layouts with eager-parity output and no recapture. Proceed only if the measured graph/memory benefit justifies changing the shared SGLang vision-attention path |
| NPU prefill graph | Implemented | The all-graph candidate selects SGLang `breakable` prefill and preserves the NPU graph-only profile even when the typed pipeline default enables Torch Compile | Re-run the all-graph task on `b8a37792` with positive prefill capture and replay |
| All-graph functional qualification | Pending revalidation | Earlier all-graph smoke and 140-request runs used behavior predecessors. The current candidate includes the graph-only config change and must be re-attested | On a clean host, pass one fresh-process cold conc8 run over 140 requests, with zero empty outputs, zero garbled outputs, positive encoder/prefill/decode evidence, and clean shutdown |
| All-graph stability and performance qualification | Deferred | Multi-run liveness, graph-registry soak, HBM trend analysis, C70, throughput, p95, and latency are not part of the functional target | Define a separate exact-head task after the functional gate is reviewable |
| SGLang main interface experiment | Historical | Main exposed `scheduler_stage_metrics` drift and older capture failures; the baseline is abandoned | Do not use for current acceptance |
| Historical `910C-071` result | Historical | It qualified the old combined candidate, but later scope removal changed the candidate and the result is not current-head evidence | Replace it with a new exact-head result or leave it clearly historical |
| Performance target | Deferred | No performance work is part of the current acceptance path | Reopen only after functionality and correctness close |
| Realtime ASR | Development/qualification pending | The integration branch starts from PR #2016 and reuses the existing `/v1/realtime` session and Qwen3-ASR streaming strategy. Add production code only for failures demonstrated on the exact combined stack | After the all-graph gate, pass `qwen3_asr_ascend_npu_realtime_task.md` on a clean card, including manual commit, server VAD, cancellation, disconnect/reconnect, positive graph activity, and no unexpected fallback |
| Timestamps / forced alignment | Planned, separate PR | Qwen3-ASR does not emit word timestamps. `Qwen3-ForcedAligner-0.6B` is a separate NAR checkpoint covering 11 languages and up to 5 minutes | Run the official package probe, define an explicit alignment-stage and response contract, then qualify it independently; see `qwen3_asr_forced_aligner_integration.md` |

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

- Completed: #2084 merged through `886ced95`.
- Do not rewrite or reopen the merged PR history.
- Start new capability work from merged main on a separate follow-up branch.

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
- The all-graph implementation has changed since the earlier smoke and
  140-request runs. Re-run the cold concurrency-8 liveness and agreed
  correctness workload on integration code head `b8a37792`.
- Record exact repository heads, dependency versions, and sanitized results.
- Stop at the first hang, accuracy failure, device error, or unexpected eager
  fallback.

### Phase 5: Upstream preparation

- Keep the merged #2084 encoder-stream unit unchanged.
- Keep PR #2160 frozen while maintainer review is pending.
- Re-run exact-head hardware qualification after PR #2160, then prepare the
  graph-only configuration as a separate follow-up PR.
- Keep SGLang compile-boundary work separate and deferred.
- Keep all-graph and realtime in the shared validation branch, but split their
  eventual delivery PRs; forced alignment remains a separate branch and PR.
- Any post-merge defect gets a new branch and a linked follow-up PR.

## Validation Gates

The current phase is complete only when:

- the NPU installer accepts the declared `0.5.19` dependency line;
- pure SGLang `v0.5.19` passes graph-only startup and decode capture without
  the fused-op patch;
- the Omni encoder uses its private device stream without the removed guard;
- one cold concurrency-8 run over 140 requests succeeds on `b8a37792` with no
  garbled output and no unexpected fallback;
- the realtime task passes manual commit, server VAD, cancellation,
  disconnect/reconnect, and bounded graph activity without unexpected fallback.

The pure-base startup and smoke gate is complete. Multi-run stability soak,
HBM trend analysis, C70 throughput targets, p95/latency measurement, and
forced alignment remain outside this gate.

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

The pure `v0.5.19` graph-only startup and smoke gates are closed, and #2084 is
merged. PR #2160 now carries the bounded NPU encoder graph. The all-graph
candidate adds the graph-only profile needed to prevent typed pipeline defaults
from re-enabling Torch Compile. Earlier all-graph results are useful behavior
evidence, but they are not current-head qualification for `b8a37792`.
Realtime is next as a separate qualification task. Timestamps remain a
separate forced-aligner feature. Performance and stability qualification are
deferred. No compile-supported NPU claim and no fused-op requirement are
active.

## Issue #2076 Update Draft

The repository feature matrix and roadmap now track:

- merged encoder private stream: done;
- NPU encoder graph: implemented under PR #2160, pending current-head
  all-graph revalidation;
- NPU prefill graph: implemented under the all-graph candidate, pending
  current-head qualification;
- decode graph: supported and previously exercised, pending exact-head
  revalidation after the graph-only profile change;
- graph-only profile with Torch Compile forced off: implemented;
- realtime PCM transcription: existing implementation, NPU qualification
  pending;
- timestamps: planned as a separate `Qwen3-ForcedAligner-0.6B` feature;
- performance, soak, and C70: still deferred.

Do not mark encoder, prefill, realtime, or timestamp support complete until
the exact-head gates in this roadmap pass.

## Roadmap Maintenance

- Update this page when a workstream changes state, not after every diagnostic
  command.
- Every status change must name the exact commit, evidence scope, and whether
  the result applies to the current heads.
- Historical results stay historical until rerun on the pinned candidate.
- If a new change is not needed for the narrow problem statement above, add it
  to the deferred section instead of expanding the active patch.
