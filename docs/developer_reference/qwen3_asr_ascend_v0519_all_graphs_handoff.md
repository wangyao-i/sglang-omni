# Qwen3-ASR Ascend v0.5.19 all-graph handoff

Status: the PR #2016-based integration code candidate is `f55c3b09`. Gate 0
passes, but the cold concurrency-8 gate on handoff `773ad0aa` hung after 16 of
140 requests in the SGLang decoder graph update thread. All-graph qualification
is blocked while an Omni-owned encoder-update ordering diagnostic is pending.
Earlier hardware runs remain historical evidence, not current-head
qualification. The first Omni-owned ordered encoder-update probe failed: 24 of
32 requests completed and eight were missing/timeouts after the encoder
`NPUGraph.update()` path raised CANN error `107033` from
`AclmdlRICaptureTaskUpdateBegin`.

## Objective

Validate the smallest no-Torch-Compile profile with encoder, prefill, and decode
graphs enabled together:

- SGLang pure `v0.5.19`;
- Omni PR #2016 at the exact server-validation head, which already contains
  the merged #2084 NPU encoder private stream;
- NPU encoder layer-stack graph;
- SGLang `breakable` prefill graph;
- NPU full decode graph;
- `enable_torch_compile=false`;
- no execution guard, fused-op patch, compile selector, or private encoder graph
  pool.

This integration branch starts from PR #2016, merges the frozen PR #2160 head,
and applies the NPU graph-only profile. It does not modify either community PR.
Realtime development and all-graph qualification share this integration base
so the server does not need to switch branches between the two workstreams.

## Exact Identity

| Repository | Exact runtime head | Required state |
|---|---|---|
| SGLang | `0bcd822377da7b5718e674eaf9c870d349424dd1` (`v0.5.19`) | Clean; no fused-op patch |
| SGLang-Omni PR #2016 base | `18c8cfd2eeeb495569426875a2e2bf4114133caf` | Exact server-validation base; must be an ancestor of the observed HEAD |
| SGLang-Omni PR #2160 candidate | `302cf932fcf17ce2f1e836b44a06a6a8d9979451` | Bucket-key encoder graph code and tests; source-equivalent commits are applied to this integration branch |
| SGLang-Omni all-graph integration base | `cb0ea08c5f852de6e152945a7e71808959f81ee2` | PR #2016 + PR #2160 + NPU graph-only profile |
| SGLang-Omni validation code candidate | `f55c3b094419b4b8c2aba84d83c1c55c0ebaa1de` | Adds the validated NPU bucket-key update path and initializes the test builder device identity required by the full-suite Gate 0; docs-only descendants are allowed |
| SGLang-Omni encoder-update diagnostic | `b00a8b8b884981fd42d0326073291339ab5c8821` | Diagnostic-only descendant: keeps NPU encoder host-input update and replay on the encoder worker instead of creating a second update helper thread |

The Omni checkout must not contain zero-diff assumptions for the SGLang side:
verify the imported SGLang module points at the exact clean `v0.5.19` checkout.

## Local Implementation

`sglang_omni/models/qwen3_asr/encoder_cuda_graph.py` adapts the generic encoder
graph runner for NPU:

- `capture_all()` defers NPU capture because synthetic bucket layouts are not
  valid Ascend attention signatures.
- The first real request captures one graph for its token bucket.
- Cumulative window boundaries remain host-resident on NPU.
- Capture enables torch_npu host-input dispatch. Before replay,
  `actual_seq_lengths` and `actual_seq_lengths_kv` are updated through
  `NPUGraph.update()` using the backend-required concurrent update/replay
  sequence.
- Different real window layouts in one token bucket reuse the same graph. The
  graph registry is bounded by the finite configured token buckets, so the old
  request-layout key, global 32-graph policy, and capacity fallback are removed.
- An NPU capture or host-input update failure remains terminal and visible.

The all-graph candidate also forces `enable_torch_compile=false` on NPU after
typed pipeline defaults are merged. The task pins compile and graph values
explicitly. The NPU test fixture selects `ascend_attn`, matching the production
platform default. `triton_attn` is a CUDA/ROCm test path and expects
device-resident `cu_seqlens`, so it must not be forced on Ascend.

## Prior Art

- `910C-026` and `910C-027` established lazy NPU signatures and bounded
  multi-signature capture.
- The #2160 bucket-key mechanism gate captured layout `[500]`, replayed layout
  `[450]` through the same graph entry, and kept both eager-parity differences
  below `3e-2` on Ascend 910C.
- `910C-032` arm `A0` was correct at WER `0.0183` with encoder, prefill, and
  decode graphs enabled and Torch Compile off.
- The current implementation intentionally excludes the old diagnostic layer,
  process-global graph-pool experiment, execution guard, and compile changes.

## Runtime Contract

Required resolved settings:

- `enable_torch_compile: false`;
- `cuda_graph_backend_prefill: breakable`;
- `cuda_graph_backend_decode: full`;
- encoder graph enabled;
- decode graph not disabled;
- no `npugraph_ex`, inductor, TorchAir compile, or compile-safe fused-op path.

Required positive evidence:

- startup contains the NPU encoder capture deferral marker;
- the focused mechanism test proves two layouts replay one encoder bucket graph;
- the service logs at least one encoder capture and zero encoder fallback or
  host-input update failures;
- prefill startup capture and positive replay count;
- positive decode graph replay;
- zero encoder graph fallback log lines;
- zero forbidden ACL, ATB, PagedAttention, allocator, stream, device, or capture
  failures.

## Open Questions

1. Does current `v0.5.19` still capture the breakable prefill graph on NPU with
   the encoder private stream active?
2. Does the first real NPU encoder signature capture without `aclrtMemcpy`
   `107030` or another captured-stream synchronization failure?
3. Does the 140-request functional-correctness workload keep all three graph
   paths active while request layouts reuse the finite bucket registry without
   host-input update failure or eager fallback?
4. Does graceful shutdown leave no process, port, graph handle, or target-card
   HBM holder behind?

## Latest First Failure

On `773ad0aa`, focused tests passed (`40 passed, 1 skipped`) and the full
Qwen3-ASR suite passed (`618 passed, 4 skipped`). The cold concurrency-8 run
then stopped after 16 of 140 requests. The decoder update helper was blocked in
`graph_task_update_begin`; encoder, prefill, and decode graphs had all captured,
and no ACL, ATB, PagedAttention, or OOM error preceded the stall. Forced cleanup
left the selected card at 86% HBM, so a verified idle reset is required before
another experiment.

This failure resembles historical `910C-065`, but the current stack adds a
captured NPU encoder graph that did not exist in the later private-stream
shipping candidate. Therefore “three submitters” remains a hypothesis for the
current head. The blocked frame is in SGLang, but the new variable and the
second update helper belong to the Omni encoder graph. The next task therefore
keeps pristine SGLang `v0.5.19`, orders only the Omni encoder update/replay, and
uses a bounded 32-request liveness probe. It does not drop a graph path or add a
guard.

The ordered encoder-update probe at `8605c0d5` did not pass its frozen gate.
Although encoder, prefill, and decode graphs were all observed and 24 requests
completed, eight requests did not complete and the encoder batch path raised
`NPU graph host input update failed`. The cause chained to
`graph_task_update_begin` in `NPUGraph.cpp:65`, where
`AclmdlRICaptureTaskUpdateBegin(stream, handle.task_group)` returned `107033`.
This rejects same-thread encoder `update -> replay` for this runtime path. It
does not show that encoder/decoder concurrency is irrelevant, and the numeric
error is not assigned a symbolic meaning without vendor evidence.

## Server Task

Use
[`qwen3_asr_ascend_v0519_all_graphs_task.md`](qwen3_asr_ascend_v0519_all_graphs_task.md).
The current task is a bounded functional sequence: one fresh-process cold
concurrency-8 run over 140 requests, with liveness and correctness assertions,
followed by shutdown cleanup. Do not add repeat counts, a steady-state soak,
HBM trend analysis, or a performance measurement. Those belong to a separate
deferred task after functionality is reviewable.

Before retrying that qualification task, run
[`qwen3_asr_ascend_encoder_update_order_probe_task.md`](qwen3_asr_ascend_encoder_update_order_probe_task.md).

## Required Return

Return only the sanitized fields in the task document. Keep raw logs, model
paths, hostnames, address data, audio, and transcripts in the isolated
environment.
