# Qwen3-ASR Ascend v0.5.19 all-graph handoff

Status: current candidate is `8d2fcaaa`; functional isolated hardware
revalidation is pending after the NPU graph-only profile change. Earlier runs
remain historical evidence, not current-head qualification.

## Objective

Validate the smallest no-Torch-Compile profile with encoder, prefill, and decode
graphs enabled together:

- SGLang pure `v0.5.19`;
- Omni main with the merged #2084 NPU encoder private stream;
- NPU encoder layer-stack graph;
- SGLang `breakable` prefill graph;
- NPU full decode graph;
- `enable_torch_compile=false`;
- no execution guard, fused-op patch, compile selector, or private encoder graph
  pool.

This branch starts from the merge commit for #2084. It does not modify or
reopen that PR; it adds only the NPU encoder layer-stack graph and its
validation task.

## Exact Identity

| Repository | Exact runtime head | Required state |
|---|---|---|
| SGLang | `0bcd822377da7b5718e674eaf9c870d349424dd1` (`v0.5.19`) | Clean; no fused-op patch |
| SGLang-Omni baseline | `886ced95b9c0b76429798bb60dbd34d3f71dad95` | Merge commit for #2084; must be an ancestor of the observed HEAD |
| SGLang-Omni PR #2160 candidate | `1638c5dddb012686210f85ed3ee050fed1ac4597` | Frozen encoder/prefill graph code and tests |
| SGLang-Omni all-graph candidate | `8d2fcaaab24e44a69ba41d06d7ee7467aee0cd11` | Current runtime candidate for encoder, prefill, and decode graph revalidation |

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
- The NPU graph registry mirrors SGLang's `ViTNpuGraphRunner`: lazy capture by
  the exact `(bucket_size, window_lens)` key, with the window boundaries kept
  host-resident. There is no separate capacity knob or capacity-based eager
  fallback; capture failures propagate instead of silently changing paths.
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
