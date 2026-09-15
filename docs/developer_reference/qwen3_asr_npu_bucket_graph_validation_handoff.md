# Qwen3-ASR NPU bucket-key validation handoff

## Decision

Validate the #2160 follow-up on an isolated Ascend 910C before proposing it
upstream. The change reuses one encoder graph for different effective window
layouts in the same token bucket. It updates the captured Ascend attention
host parameters immediately before replay.

The one causal question is:

> Can layouts `[500]` and `[450]` reuse one encoder token-bucket graph through
> `NPUGraph.update()` while both outputs remain within `3e-2` of eager output?

This task also runs a two-request all-graph smoke test to catch integration
regressions. It does not qualify performance, long-run stability, realtime,
cache eviction, or the complete #2016 integration stack.

## Exact inputs

- SGLang-Omni upstream base: `1638c5dddb012686210f85ed3ee050fed1ac4597`
- SGLang-Omni code/test head: `302cf932fcf17ce2f1e836b44a06a6a8d9979451`
- SGLang head: `0bcd822377da7b5718e674eaf9c870d349424dd1`
- Qualification branch: `codex/pr2160-npu-bucket-key`
- Required hardware: one isolated Ascend 910C

The server operator must use
`qwen3_asr_npu_bucket_graph_validation_task.md` from the qualification branch.
The task file is allowed to be a docs-only descendant of the code/test head.
No source, dependency, package, or site-package changes are authorized on the
server.

## Ownership and evidence

The local owner prepares and reviews the commits. The isolated-hardware owner
runs the task, stops at the first failure, retains raw artifacts on the server,
and returns only the redacted result fields specified by the task.

Acceptance requires all of the following on the recorded exact heads:

1. focused unit tests pass, apart from an explicit unavailable-XPU skip;
2. both NPU mechanism cases match eager within `3e-2` and use one graph entry;
3. the two-request encoder + prefill + decode graph smoke completes with no
   fallback, compile, ACL, allocator, stream, device, ATB, or attention error;
4. shutdown leaves no service process or NPU context holder and HBM returns to
   its pre-run idle baseline.

If the mechanism gate fails, return the first complete failure without running
the service. A passed result authorizes preparation of the #2160 update; it
does not authorize a performance claim or realtime development. The full
140-request concurrency-8 qualification remains a separate gate on the #2016
integration branch before realtime work starts.
