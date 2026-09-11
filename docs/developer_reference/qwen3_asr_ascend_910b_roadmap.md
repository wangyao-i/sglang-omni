# Qwen3-ASR on Ascend roadmap

## Project goals

This project brings Qwen3-ASR inference to Ascend with the complete acceleration
stack enabled together. It has three goals:

1. support offline ASR with Encoder Graph, Prefill Graph, Decode Graph,
   Torch Compile, and NPU encoder stream isolation enabled together;
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
| NPU encoder stream isolation | Supported | The audio encoder submits on its own device stream, so it never shares the generation submission lane. Replaces the earlier cross-thread execution guard. |
| Fully accelerated offline stack | Supported in development | Encoder, prefill and decode graph, compile, and stream isolation have passed combined correctness testing. |
| Realtime/streaming ASR | Not implemented | A realtime protocol, incremental transcript contract, and streaming benchmark are still required. |

## Current fully accelerated reference result

The current reference enables Encoder Graph, Prefill Graph, Decode Graph,
Torch Compile, and NPU encoder stream isolation together.

| Item | Value |
|---|---|
| Hardware target | One Ascend 910B/910C-class card |
| Model | Qwen3-ASR-1.7B, BF16 |
| Workload | 700 distinct exact-10-second samples at concurrency 70 |
| Successful requests | 700/700 in every repeat |
| Latency p95 | 1.35-1.47 s |
| Throughput | 59.7-62.9 requests/s |
| RTFx | 597-629 |
| WER | 0.0179 at the 140-request correctness gate |
| Garbled outputs | 0 |
| Runtime configuration | `max_total_tokens=32768`, `mem_fraction_static=0.80`, compile coverage selected with `torch_compile_max_bs` |

These values are the `910C-070` arm A measurements, taken over three fresh
processes. Their compile coverage came from a development-only sparse bucket
selector that is not part of the shipped branches; the shipped configuration
selects coverage with `torch_compile_max_bs`. See the reference note under
remaining engineering work for their provenance and for the re-measurement that
follows the guard removal.

This is a development reference rather than a portable performance promise.
Results can vary with the model revision, CANN and torch-npu versions, hardware,
request distribution, and server configuration.

## Full-feature validation pull requests

Two integration pull requests will provide the complete code needed to run the
current feature set locally. These links are placeholders and will be updated
after the branches are published.

| Repository | Full-feature validation PR | Scope |
|---|---|---|
| SGLang | **TBD** | NPU graph execution, Torch Compile compatibility, and sparse compile batches. |
| SGLang-Omni | **TBD** | Qwen3-ASR Encoder Graph, NPU encoder stream isolation, configuration, runtime counters, and validation tooling. |

These integration PRs are intended to make end-to-end validation available
early. After the implementation is split into smaller community-ready PRs,
this roadmap will be refreshed with the final dependency and merge order.

## Remaining engineering work

### Runtime and graph integration

- verify the encoder's private-stream hand-off across supported NPU graph
  runners and on the guard-free commits; and
- keep the encoder and generation submission lanes separate as new device
  producers are added.

### Performance optimization

- choose which decode buckets to compile rather than compiling a prefix of the
  captured set, so a memory-constrained card can cover the hot bucket without
  paying for every smaller one;
- reduce graph breaks and recompilations around the attention path;
- reduce encoder queueing and long device critical sections;
- improve prefill/decode admission and batch formation; and
- optimize device kernels only after profiling identifies a specific hot
  operator.

The reference for the shipping configuration is the `910C-070` arm A band: p95
1.350-1.474 s at 59.65-62.88 requests/s over three fresh processes, measured with
the encoder on its own stream and no guard installed. It was taken on a
development build that expressed both the stream fix and the compile coverage
through switches, so this section will be refreshed by `910C-071`, which
re-measures the shipped heads using the upstream compile-coverage knob. The
earlier 1.442 s / 59.88 requests/s reading, and the guarded band that replaced it
as the anchor, are both superseded.

The gap to the challenge target, measured against the arm A band, is
approximately 2.7-2.9x in p95 latency and 2.2-2.3x in throughput.

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
