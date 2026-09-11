# Qwen3-ASR on Ascend roadmap

## Project goals

This project brings Qwen3-ASR inference to Ascend with the complete acceleration
stack enabled together. It has three goals:

1. support offline ASR with Encoder Graph, Prefill Graph, Decode Graph,
   Torch Compile, and the NPU execution guard enabled together;
2. make the implementation reviewable and available through the SGLang and
   SGLang-Omni communities; and
3. challenge an exact-10-second, concurrency-70 performance target of latency
   p95 below 500 ms and throughput above 140 requests/s on one card.

The performance target is an optimization challenge, not a guarantee. Feature
correctness and upstream support do not depend on reaching that target.

## Feature status

The following status describes the development implementation. Community
support remains pending until the corresponding pull requests are merged.

| Feature | Status | Notes |
|---|---|---|
| Offline ASR | Supported | OpenAI-compatible transcription requests are used for qualification. |
| Encoder Graph | Supported | Bounded shape/signature capture and replay are implemented. |
| Prefill Graph | Supported | Can run together with encoder and decode graphs. |
| Decode Graph | Supported | Captured batch sizes through concurrency 70 have been exercised. |
| Torch Compile | Supported | Can run with all graph paths while preserving ASR output quality. |
| NPU execution guard | Supported | Prevents conflicting encoder and generation device work; the narrowest efficient scope is still being optimized. |
| Fully accelerated offline stack | Supported in development | Encoder, prefill, decode graph, compile, and guard have passed combined correctness testing. |
| Realtime/streaming ASR | Not implemented | A realtime protocol, incremental transcript contract, and streaming benchmark are still required. |

## Current fully accelerated reference result

The current reference enables Encoder Graph, Prefill Graph, Decode Graph,
Torch Compile, and the execution guard together.

| Item | Value |
|---|---|
| Hardware target | One Ascend 910B/910C-class card |
| Model | Qwen3-ASR-1.7B, BF16 |
| Workload | 700 distinct exact-10-second samples at concurrency 70 |
| Successful requests | 700/700 |
| Latency p95 | 1.442 s |
| Throughput | 59.88 requests/s |
| RTFx | 598.8 |
| WER | 0.0164 |
| Garbled outputs | 0 |
| Runtime configuration | `max_total_tokens=32768`, `mem_fraction_static=0.80`, `torch_compile_bs=[1,2,70]` |

This is a development reference rather than a portable performance promise.
Results can vary with the model revision, CANN and torch-npu versions, hardware,
request distribution, and server configuration.

## Full-feature validation pull requests

Two integration pull requests will provide the complete code needed to run the
current feature set locally. These links are placeholders and will be updated
after the branches are published.

| Repository | Full-feature validation PR | Scope |
|---|---|---|
| SGLang | **TBD** | NPU graph execution, Torch Compile compatibility, sparse compile batches, and graph input update ordering. |
| SGLang-Omni | **TBD** | Qwen3-ASR Encoder Graph, execution guard integration, configuration, runtime counters, and validation tooling. |

These integration PRs are intended to make end-to-end validation available
early. After the implementation is split into smaller community-ready PRs,
this roadmap will be refreshed with the final dependency and merge order.

## Remaining engineering work

### Runtime and graph integration

- finalize NPU graph input update/replay ownership;
- minimize the execution-guard critical section without reintroducing device
  concurrency hangs; and
- verify the final behavior across supported NPU graph runners.

### Performance optimization

- reduce graph breaks and recompilations around the attention path;
- reduce encoder queueing and long device critical sections;
- improve prefill/decode admission and batch formation; and
- optimize device kernels only after profiling identifies a specific hot
  operator.

The reproducible reference is the guarded `forward` scope measured over three
fresh processes per arm: p95 1.539-1.765 s at 50.66-55.95 requests/s
(`threaded`) and 1.559-1.739 s at 51.34-54.51 requests/s (`ordered`). The single
1.442 s / 59.88 requests/s reading recorded earlier is not reproduced by any of
those six runs and is no longer used as the anchor.

The gap to the challenge target, measured against that reproducible reference,
is approximately 3.3x in p95 latency and 2.6x in throughput.

### Realtime ASR

- define the streaming session and incremental transcript/revision contract;
- implement chunked audio ingestion and partial result emission;
- add session isolation, cancellation, backpressure, and error handling; and
- qualify correctness, partial/final latency, concurrency, and long-running
  stability independently from the offline benchmark.

### Upstream preparation

- remove development-only diagnostics and internal qualification history;
- rebase the retained implementation onto current upstream branches;
- split runtime, benchmark, and documentation changes into reviewable PRs; and
- rerun combined correctness and performance qualification on the final
  upstream candidate commits.
