# Qwen3-ASR Ascend v0.5.19 all-graph handoff

Status: the PR #2016-based integration code candidate is `6a59057e`;
functional isolated-hardware revalidation is pending. Earlier runs remain
historical evidence, not current-head qualification.

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
| SGLang-Omni PR #2160 candidate | `1638c5dddb012686210f85ed3ee050fed1ac4597` | Frozen encoder/prefill graph code and tests |
| SGLang-Omni all-graph integration base | `cb0ea08c5f852de6e152945a7e71808959f81ee2` | PR #2016 + PR #2160 + NPU graph-only profile |
| SGLang-Omni validation code candidate | `6a59057eb744ccb1a03692c369a7d7b288dbe3aa` | Adds bounded Qwen3-ASR final-prefix behavior and aligns PR #2016 final decode with its stability-gate test; docs-only descendants are allowed by Gate 0 |

The Omni checkout must not contain zero-diff assumptions for the SGLang side:
verify the imported SGLang module points at the exact clean `v0.5.19` checkout.

## Local Implementation

`sglang_omni/models/qwen3_asr/encoder_cuda_graph.py` adapts the generic encoder
graph runner for NPU:

- `capture_all()` defers NPU capture because synthetic bucket layouts are not
  valid Ascend attention signatures.
- The first real request captures by exact `(bucket_size, window_lens)`.
- Cumulative window boundaries remain host-resident on NPU.
- Different real window layouts for one token bucket receive separate graphs.
- The NPU graph registry captures lazily by exact
  `(bucket_size, window_lens)` layout and shares one graph pool across captured
  layouts. A runner-wide `max_graphs` limit, defaulting to 32, bounds the total
  number of NPU graphs across all buckets. Once the limit is reached,
  previously captured layouts continue to replay while unseen layouts return
  to the eager path. This capacity fallback is distinct from capture failure:
  an NPU capture failure is terminal, and subsequent requests raise until the
  process is restarted with encoder graphs disabled.
- First replay of each signature is logged for positive target-path evidence.

The all-graph candidate also forces `enable_torch_compile=false` on NPU after
typed pipeline defaults are merged. The task pins compile and graph values
explicitly. The NPU test fixture selects `ascend_attn`, matching the production
platform default. `triton_attn` is a CUDA/ROCm test path and expects
device-resident `cu_seqlens`, so it must not be forced on Ascend.

## Prior Art

- `910C-026` and `910C-027` established lazy NPU signatures and bounded
  multi-signature capture.
- `910C-031` saturated the encoder cache at `8/8` with positive replay and zero
  encoder fallback/capture failure.
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
- at least one encoder capture and replay marker;
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
   paths active without encoder signature-capacity fallback?
4. Does graceful shutdown leave no process, port, graph handle, or target-card
   HBM holder behind?

## Server Task

Use
[`qwen3_asr_ascend_v0519_all_graphs_task.md`](qwen3_asr_ascend_v0519_all_graphs_task.md).
The current task is a bounded functional sequence: one fresh-process cold
concurrency-8 run over 140 requests, with liveness and correctness assertions,
followed by shutdown cleanup. Do not add repeat counts, a steady-state soak,
HBM trend analysis, or a performance measurement. Those belong to a separate
deferred task after functionality is reviewable.

## Required Return

Return only the sanitized fields in the task document. Keep raw logs, model
paths, hostnames, address data, audio, and transcripts in the isolated
environment.
