# Qwen3-ASR decoder ordered-update validation

Instruction revision: 1. This supplements
`qwen3_asr_ascend_v0519_all_graphs_task.md`; that task retains ownership of the
service command, input manifest, correctness checks, graph evidence and cleanup.

## Read-only discovery

Before starting a service, use the failed service's Python environment:

```bash
python scripts/npu/inventory_graph_sources.py
```

Return full Omni and SGLang HEADs and relevant dirty state; Python, torch,
torch_npu and CANN versions; the inventory JSON; and the retained effective
encoder, prefill, decode, compile and pre-LM settings. Confirm whether imports
resolve to the intended checkouts without returning private absolute paths.

Clarify whether the reported `encoder_off` condition disabled only the graph,
the pre-LM worker, or all encoder device submissions. For each failed run return
the first `107033` operation, last update/replay boundaries and blocked call
separately. Do not assume all stalls share one stack.

This phase is read-only. Do not install packages, edit site-packages, start a
workload, restart services or reset a device. Stop and return discrepancies if
the observed code, imports or effective configuration cannot be reconciled with
the failed baseline.

## Candidate

The SGLang diagnostic candidate is
`4d819e5aa265e5548a63bdd9bb6ec35b10396950` on branch
`codex/qwen3-asr-all-graph-ordered-update-probe`, based on clean v0.5.19
`0bcd822377da7b5718e674eaf9c870d349424dd1`. It changes decoder graph input
update ordering and its focused test. It is not an unmodified v0.5.19 runtime.

Keep the failed run's exact Omni code, dependencies, dataset and configuration.
The local reference Omni branch is
`codex/qwen3-asr-npu-updatable-encoder-graph` at `9e398a7b`; this is not proof of
the server baseline. Use separate clean worktrees and preserve unrelated state.

Before running a service, execute the focused SGLang test in the real runtime:

```bash
python -m pytest \
  test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py -q
```

Inspect the installed torch_npu update implementation. If update completion
requires replay to be submitted concurrently, stop and return that source
evidence instead of running the ordered candidate.

## Conditional gates

Reuse the owning all-graph task's exact command, manifest, sample ordering and
configuration. This supplement changes only the SGLang decoder update candidate.
Do not change encoder behavior, prefill mode, graph enablement, concurrency,
sample count or dependency versions.

Run in fresh service processes:

1. concurrency 8, 32 samples;
2. concurrency 8, 140 samples;
3. concurrency 8, 140 samples once more in another fresh process.

Require every result exactly once, zero failures, timeouts and empty transcripts,
no successful sample with WER above 0.5, positive encoder/prefill/decode graph
evidence, and zero graph errors or unapproved eager fallback. Preserve any
stricter applicable requirement from the owning task.

Stop at the first graph or CANN error, 60 seconds without a completed request
after readiness, or 10 minutes total per workload. These bounds classify the
gate as failed or bounded; they do not prove a deadlock. Retain the last completed
request, outstanding count, graph counters and available update/replay boundaries.
Do not continue eager after abandoning a blocked update thread.

Use the owning task's shutdown procedure. Restore only task-owned resources and
require its idle baseline before another run. Rollback selects the prior checkout
in a fresh process; never reset a dirty checkout.

Return exact code/build identities, this instruction revision and file SHA256,
effective configuration, manifest SHA256, gates actually run, correctness and
graph counters, first failure, cleanup result and server-local evidence IDs.
A pass qualifies only this exact candidate and does not establish a CANN internal
root cause.
