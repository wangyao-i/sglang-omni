# Qwen3-ASR PR #2016 all-graph and realtime validation branch

Status: local integration branch prepared; server Gate 0 acknowledgement and
hardware execution are pending.

## Purpose

Use the exact PR #2016 head currently under server validation as the common
base for two ordered workstreams:

1. qualify Qwen3-ASR encoder, prefill, and decode graphs together with Torch
   Compile disabled;
2. develop and qualify Qwen3-ASR realtime transcription on the same accepted
   graph-only runtime.

This is an integration and qualification branch. It is not a replacement for
PR #2016 or PR #2160, and it must not be submitted upstream as one combined PR.

## Exact Stack

| Layer | Exact head | Role |
|---|---|---|
| PR #2016 base | `18c8cfd2eeeb495569426875a2e2bf4114133caf` | Server's current realtime validation branch |
| PR #2160 | `1638c5dddb012686210f85ed3ee050fed1ac4597` | Frozen NPU encoder graph candidate |
| Integration merge | `4afce5fb` | Records both PR histories without rewriting either one |
| All-graph integration base | `cb0ea08c5f852de6e152945a7e71808959f81ee2` | Adds the NPU graph-only profile |
| Qwen3-ASR realtime code | `b8a37792829ef402edf7b5c83136ab5c804cf5af` | Adds the bounded Qwen3-ASR final-prefix realtime change |
| Validation code | `6a59057eb744ccb1a03692c369a7d7b288dbe3aa` | Preserves PR #2016's stability gate on final decode |
| SGLang runtime | `0bcd822377da7b5718e674eaf9c870d349424dd1` | Clean `v0.5.19`; no fused-op patch |

Local branch:
`codex/pr2016-qwen3-asr-all-graph-realtime-validation`.

The server must report its observed branch, full HEAD, clean-worktree state,
imported module paths, runtime versions, and hardware summary before running a
gate. A material identity mismatch stops the task.

## Ordered Gates

Run the all-graph task first:
[`qwen3_asr_ascend_v0519_all_graphs_task.md`](qwen3_asr_ascend_v0519_all_graphs_task.md).

Only after it passes, continue with the realtime task:
[`qwen3_asr_ascend_npu_realtime_task.md`](qwen3_asr_ascend_npu_realtime_task.md).

The two tasks share a code base but remain separate causal questions and
separate fresh service processes. An all-graph failure blocks realtime work
that depends on the same graph profile. A realtime-only failure may justify a
minimal realtime change on this branch; it does not authorize changing graph
policy or either community PR.

## Development Boundary

- Reuse the existing device-agnostic realtime session and Qwen3-ASR streaming
  strategy.
- Do not add NPU-specific realtime production code unless the exact-stack gate
  demonstrates a platform-specific failure.
- Keep word timestamps and forced alignment out of this branch.
- Keep performance, C70, p95, throughput, and soak work out of these functional
  gates.
- Stop at the first complete failure, preserve the raw evidence on the server,
  return a sanitized classification, and restart from a fresh process after a
  fix.

## Return Contract

For each task return the exact identities, resolved graph/compile settings,
focused-test result, target-path markers, fallback/error counts, request/event
counts, first failure if any, cleanup health, server-local artifact reference,
and any server commit that must be reconstructed locally.
