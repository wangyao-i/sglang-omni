# Qwen3-ASR Ascend NPU realtime handoff

Status: implementation reuse complete; isolated-hardware qualification pending.

## Objective

Qualify live PCM transcription on the same no-Torch-Compile, all-graph profile
used by the offline Qwen3-ASR path:

- SGLang `v0.5.19`;
- SGLang-Omni `8d2fcaaa`;
- encoder graph, breakable prefill graph, and full decode graph enabled;
- `enable_torch_compile=false`;
- `/v1/realtime?intent=transcription` with manual commit and server VAD;
- cancellation, disconnect, and reconnect behavior.

This workstream does not add a second realtime implementation for NPU.

## Exact Identity

| Repository | Candidate | Required state |
|---|---|---|
| SGLang | `0bcd822377da7b5718e674eaf9c870d349424dd1` (`v0.5.19`) | Clean; no fused-op or Qwen3 model patch |
| SGLang-Omni PR #2160 | `1638c5dddb012686210f85ed3ee050fed1ac4597` | Frozen code candidate |
| SGLang-Omni all-graph/realtime base | `8d2fcaaab24e44a69ba41d06d7ee7467aee0cd11` | Clean; contains the NPU graph-only profile |

The realtime branch is stacked on `8d2fcaaa`; it currently contains no
additional production code.

## Code Finding

The existing realtime path is device-agnostic:

- `RealtimeTranscriptionSession` owns WebSocket state, audio buffering, VAD,
  partial replacement, finalization, cancellation, and reconnect semantics.
- `Qwen3ASRStreamingStrategy` converts each partial or final segment into an
  ordinary `GenerateRequest`.
- The request then follows the same model, encoder, prefill, and decode path as
  an offline upload.

No CUDA-only stream, event, allocator, graph API, or prompt field was found in
the realtime transcription path. Adding an NPU-specific realtime branch would
duplicate code and create a second behavior contract. The correct change is a
hardware qualification task on the existing implementation.

## Prior Art

The current vLLM-Omni tree has a general realtime framework, but no matching
turn-based Qwen3-ASR transcription strategy. Its full-duplex MiniCPM-o
protocol is not a drop-in implementation for this feature. The closest
reusable implementation remains SGLang-Omni's existing transcription session
and Qwen3-ASR streaming strategy.

## Expected Runtime Contract

The resolved server configuration must show:

- `enable_torch_compile: false`;
- `disable_cuda_graph: false`;
- prefill backend `breakable`;
- decode backend `full`;
- encoder graph enabled;
- realtime transcription mounted with `/v1/realtime`.

The realtime workload must show:

- `session.created` and `session.updated`;
- positive partial hypotheses for enough audio;
- one immutable final event per committed segment;
- `transcription.completed`;
- server VAD events in VAD mode and manual commit behavior in manual mode;
- no replacement event after a segment is final;
- no event-index ordering violation;
- clear/cancel leaves the session usable or closes it cleanly;
- disconnect followed by a new session succeeds.

## Current Risk

Realtime partial decodes use increasing audio lengths. On NPU, each new exact
encoder window layout can create a separate encoder graph until the graph
registry reaches its `32`-entry bound. Repeating the same decode interval and
sample durations should reuse layouts, but the first bounded qualification
must prove that the planned workload stays within the bound.

An encoder-capacity fallback is allowed only if the product contract explicitly
accepts eager fallback. It is not accepted in this first all-graph
qualification.

## Non-Goals

- No realtime performance, throughput, RTF, or latency target.
- No C70, p95, or soak run.
- No word timestamps or forced alignment.
- No production code change while the existing device-agnostic path is still
  awaiting exact-head hardware evidence.
- No change to PR #2160.

## Server Task

Use
[`qwen3_asr_ascend_npu_realtime_task.md`](qwen3_asr_ascend_npu_realtime_task.md).

## Required Return

Return only the sanitized fields listed in the task document. Keep raw logs,
audio, transcripts, model paths, hostnames, addresses, and credentials inside
the isolated environment.
