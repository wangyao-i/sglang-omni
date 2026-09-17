# Qwen3-ASR Ascend NPU realtime handoff

Status: PR #2016 integration base prepared; Qwen3-ASR realtime development and
isolated-hardware qualification are pending on the combined validation branch.

## Objective

Qualify live PCM transcription on the same no-Torch-Compile, all-graph profile
used by the offline Qwen3-ASR path:

- SGLang `v0.5.19`;
- SGLang-Omni PR #2016 base `18c8cfd2` plus the integration code candidate
  `63bc33e0`;
- encoder graph, breakable prefill graph, and full decode graph enabled;
- `enable_torch_compile=false`;
- `/v1/realtime?intent=transcription` with manual commit and server VAD;
- cancellation, disconnect, and reconnect behavior.

This workstream does not add a second realtime implementation for NPU.

## Exact Identity

| Repository | Candidate | Required state |
|---|---|---|
| SGLang | `0bcd822377da7b5718e674eaf9c870d349424dd1` (`v0.5.19`) | Clean; no fused-op or Qwen3 model patch |
| SGLang-Omni PR #2016 base | `18c8cfd2eeeb495569426875a2e2bf4114133caf` | Exact branch currently used by the server; contains the realtime work under review and merged #2084 |
| SGLang-Omni PR #2160 | `302cf932fcf17ce2f1e836b44a06a6a8d9979451` | Validated bucket-key code candidate |
| SGLang-Omni all-graph integration base | `cb0ea08c5f852de6e152945a7e71808959f81ee2` | PR #2016 + PR #2160 + NPU graph-only profile |
| SGLang-Omni validation code candidate | `63bc33e05de798d754e9953ebfcee9abd08a11ca` | Serializes NPU encoder graph update+replay on the model execution thread and disables the NPU pre-LM worker; docs-only descendants are allowed |

The validation branch is based on PR #2016 and uses the same integrated code
head as all-graph qualification. The first bounded Qwen3-ASR realtime change
reuses PR #2016's final-decode rule: after the existing two-refresh stability
gate and language check pass, a final decode retains the full stable prefix and
sets token rollback to zero. No NPU-specific realtime branch has been added;
further behavior must be justified by a failing realtime gate on this stack.

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

Realtime partial decodes use increasing audio lengths, so repeated turns can
update the host-side encoder window boundaries many times within one token
bucket. The mechanism gate proves two layouts, but the bounded realtime
qualification must still show that repeated update/replay completes without a
hang, update error, recapture, or eager fallback.

No encoder fallback is accepted in this first all-graph qualification.

## Non-Goals

- No realtime performance, throughput, RTF, or latency target.
- No C70, p95, or soak run.
- No word timestamps or forced alignment.
- No production code change while the existing device-agnostic path is still
  awaiting exact-head hardware evidence.
- No change to PR #2016 or PR #2160.

## Server Task

Use
[`qwen3_asr_ascend_npu_realtime_task.md`](qwen3_asr_ascend_npu_realtime_task.md).

## Required Return

Return only the sanitized fields listed in the task document. Keep raw logs,
audio, transcripts, model paths, hostnames, addresses, and credentials inside
the isolated environment.
