# Qwen3-ASR Ascend v0.5.19 all-graph handoff

Status: local implementation rebased onto merged #2084; isolated hardware
Gate 0 pending.

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
| SGLang-Omni | `codex/qwen3-asr-npu-encoder-prefill-graph-v0519` | Clean observed HEAD must contain this handoff and differ from the runtime-code commit only under `docs/` |
| SGLang-Omni runtime code | `efafac553ccbb644aaed5486cff8e0a0160c3b86` | Exact encoder/prefill/decode graph implementation and tests |

The Omni checkout must not contain zero-diff assumptions for the SGLang side:
verify the imported SGLang module points at the exact clean `v0.5.19` checkout.

## Local Implementation

`sglang_omni/models/qwen3_asr/encoder_cuda_graph.py` now adapts the generic
encoder graph runner for NPU:

- `capture_all()` defers NPU capture because synthetic bucket layouts are not
  valid Ascend attention signatures.
- The first real request captures by exact `(bucket_size, window_lens)`.
- Cumulative window boundaries remain host-resident on NPU.
- Different real window layouts for one token bucket receive separate graphs.
- Admission is fair within one explicit global signature budget: one slot per
  unseen token bucket is reserved before any bucket may take an additional
  signature. The default budget is `max(max_batch_size, bucket_count)` and can
  be raised with `asr.factory.npu_encoder_graph_signature_capacity`; this task
  pins `64`.
  Exhaustion does not evict, and unseen layouts stay eager with explicit
  bounded logging.
- First replay of each signature is logged for positive target-path evidence.

The existing Qwen3-ASR generation defaults already select
`cuda_graph_backend_prefill=breakable`; the task pins that value explicitly.
The NPU test fixture selects `ascend_attn`, matching the production platform
default. `triton_attn` is a CUDA/ROCm test path and expects device-resident
`cu_seqlens`, so it must not be forced on Ascend.

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
- `npu_encoder_graph_signature_capacity: 64`;
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
4. Does graceful shutdown return the NPU and HBM to the pre-run baseline?

## Server Task

Use
[`qwen3_asr_ascend_v0519_all_graphs_task.md`](qwen3_asr_ascend_v0519_all_graphs_task.md).
Do not add another profile or run a performance measurement in this task.

## Required Return

Return only the sanitized fields in the task document. Keep raw logs, model
paths, hostnames, address data, audio, and transcripts in the isolated
environment.
