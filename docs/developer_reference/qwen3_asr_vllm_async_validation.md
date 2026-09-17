# WF-009: asynchronous graph submission versus ordered update

Instruction revision: 1. Operation: qualification and paired comparison.
This is a server-agent packet. No NPU execution has occurred on the local host.

## Purpose and scope

Decide between the established ordered decoder candidate and a vLLM-inspired
asynchronous candidate under the same all-graph Qwen3-ASR workload. This compares
complete implementations, not individual causal mechanisms. A successful run
does not prove the internal cause of CANN 107033.

The asynchronous candidate uses the shared torch_npu update stream, installs
`update_stream.wait_stream(current_compute_stream)` BEFORE submitting replay,
then calls update on the caller thread. Encoder and decoder use one shared Host
RLock spanning wait/replay/update so their submissions cannot interleave. The lock
does not synchronize the device. This is an adaptation for Omni's separate
encoder worker, not a verbatim copy of vLLM.

References: vllm-ascend PRs
[12944](https://github.com/vllm-project/vllm-ascend/pull/12944) and
[13600](https://github.com/vllm-project/vllm-ascend/pull/13600), merge
`ac19e1e647785be51d22a87f336ba03c02357e18`.
The public torch_npu v2.10.0 branch has class-shared update_stream and inline
Begin/op/End/event-record in update_capture_record. The installed post2 build
must be checked independently; a branch name is not its identity.

## Candidates

- Ordered O: SGLang `4d819e5aa265e5548a63bdd9bb6ec35b10396950`,
  Omni `964ddd5f2dc2546f64b1c483edfcc2207edd0297`.
- Async A: SGLang `b950878e03f07c63bdca050e8a854503ebfa7058`, branch
  `codex/npu-vllm-async-probe`; Omni code commit
  `611e08784b714994995feb1c33171192efaca579`, branch
  `codex/qwen3-asr-vllm-async-probe`.
- The branch may include a later documentation-only packet commit. Record both
  code SHA and packet commit/SHA256. Do not substitute WF-008 trace branches or
  the eager chunk-plan workaround. `eager_preamble` stays at the original baseline.
- Keep prior evidence/kernel_meta files. Use separate clean worktrees rather
  than deleting untracked files or resetting a dirty checkout.

## Preflight: stop before load on mismatch

Record actual imported sglang/sglang_omni/torch_npu paths, Git SHA/diff and package
versions: Python 3.11.10, torch 2.10.0, torch_npu 2.10.0.post2, CANN 9.0.1.
Record actual NPU model, driver, device allocation and idle HBM. Only use the
allocated device and task-owned processes; no package changes or device resets.

Save installed source and SHA256 for NPUGraph.update, NPUGraph.replay,
_GraphDispatchMode.update_capture_record, its stream initialization and capture
event handling. Verify:

1. update uses the same `_GraphDispatchMode.update_stream` returned by Omni's
   `get_npu_graph_update_stream` and decoder's accessor;
2. capture has event wait/reset and update has Begin/op/End/event-record;
3. update does not add a wait for the just-submitted replay or a global device
   synchronization. Return unexpected behavior before executing A.

Run for A (paths denote actual isolated worktrees):

```bash
cd "$SGLANG_WORKTREE"
python scripts/npu/test_async_submission.py --omni-root "$OMNI_WORKTREE"
python -m pytest test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py -q
cd "$OMNI_WORKTREE"
python -m pytest tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py -q
```

All must pass, with four standalone tests and no skipped encoder source test.
For O use its existing focused suites. Preserve full collection/results.

## Fixed runtime and data

Use the successful WF-004 launcher and evaluator, record their exact commands
and SHA256, and keep all effective settings identical between arms:

- compile=false; prefill=breakable; decode=full; encoder graph=true;
  disable_cuda_graph=false; max_running_requests=64; graph_max_bs=64;
- pre-LM encoder worker active, with identical worker and stream policy;
- concurrency 8, fixed 140-input manifest SHA256
  `22bb330f7572827fc1e6fbe5b5ec3a261aed9d77651abfbb6c2d704019bf3f1a`;
- identical model, generation settings, seed, timeouts, request order and cache
  policy. Save input/output IDs; do not change data to avoid a failure;
- unset WF-008 trace/serialization settings. Keep equivalent logging levels.

## Bounded execution

1. O health check: fresh process, two known-valid manifest requests. Require
   expected text, no invalid feature lengths/index errors and healthy shutdown.
   If O fails, stop: runtime/input health blocks comparison.
2. A smoke: fresh process, same two requests. Require encoder and decoder
   `NPU async submission completed` markers, compatible shared update-stream
   identity, prefill replay counter increase and correct text. These markers
   attest Host replay/update completion, not device completion by themselves.
   The successful responses supply the corresponding completion evidence.
3. If both pass, run three matched pairs, one service at a time:
   O1 -> A1; A2 -> O2; O3 -> A3. Each run starts a fresh service, uses the same
   32-input warmup, then measures conc8/140. Record warmup separately.
   Use the same warmup inputs/order in all runs. Verify cache hits do not mask
   encoder execution on the measured workload; if measured replay cannot be
   demonstrated, stop comparison and report the cache policy issue.

Capture three graph-family replay evidence and eager-fallback counts, using
available runtime counters/profiler evidence. Capture markers or elapsed time
alone are insufficient. If a counter is unavailable say so; do not report zero
fallback without evidence. Do not introduce runtime monkeypatches or alternate
data/configuration to manufacture attestation.

Stop load expansion at the first exception, 107033, timeout, empty output,
WER > 0.5 output, invalid length, or 60 seconds without completed requests.
Save the first complete traceback and CANN plog context, worker stacks, and
progress. Do not run additional stress rounds after a failure. Shut down only
task-owned processes; report forced termination separately and do not count it
as successful normal shutdown. Preserve logs and recover the allocated idle
resource baseline before another process starts.

## Acceptance and decision rule

Functional gate for A: three measured rounds each 140/140; zero errors/timeouts,
empty/garbled outputs, zero 107033 and unexpected eager fallback; positive
encoder/prefill/decode replay; aggregate WER no worse than paired O; normal
shutdown, free port, no owned process residue and recovered HBM.

Return each raw JSON, wall time, p50/p95/p99, throughput, WER, output differences,
cache/replay/fallback counters and cleanup. Compare measured windows only, not
cold startup against warm execution. Three runs provide a bounded decision aid,
not a statistically established performance claim.

- A functional failure with healthy O: choose O for delivery; retain A evidence.
- O failure or runtime/input/identity failure: comparison invalid; fix the
  identified blocker before deciding. Do not call A a failed hypothesis.
- Both pass: prefer O's simpler integration unless A improves throughput in
  all three pairs, median relative throughput gain >=5%, and no pair has p95
  regression >5%. These are predeclared engineering selection thresholds.
- If A meets that bar: recommend A, subject to review of the two-repository
  integration and runtime-private API dependency. Report tradeoffs explicitly.
- Missing replay, correctness or cleanup evidence: leave final decision pending.

Return: WF-009 revision, status, exact identities/imports, preflight semantics,
effective config, manifest, executed commands, per-run metrics/attestation,
first failure if any, cleanup, evidence directory identifiers, and recommendation.
Keep full private server paths, data and logs on the server.
