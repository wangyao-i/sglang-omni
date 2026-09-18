# WF-009 B: private encoder update stream

Instruction revision: B1. Operation: single-variable diagnosis, then conditional
stability checks. This packet supersedes the old O-versus-A packet for this run.
Not a performance comparison or authorization to merge the candidate.

## Question and fixed identities

Does separating encoder task updates from decoder's shared update stream prevent
the prolonged stall reported for A while retaining A's Host/event ordering?

- SGLang BOTH arms: `b950878e03f07c63bdca050e8a854503ebfa7058`.
- A Omni: `d6aae0541007b7b71004cabaf720db03b10197af`.
- B Omni code: `bd6f335446dea31c7ccd9c6e7baa699bc798e662`.
- B branch: `qwen3-asr/private-encoder-update-stream`.
- B may have this documentation-only commit above its code SHA; record both
  code and packet HEAD plus this file's SHA256. No arbitrary branch-tip updates.

B changes only the encoder runner's update-stream allocation, using its existing
device module and explicit model device. One stream is retained per runner; all
its captured buckets/tasks reuse it. Decoder code is unchanged. Preserve shared
Host RLock, wait-before-replay, replay-before-update, Begin/op/End/event-record,
capture event wait/reset, eager preamble, capture/pool policy and output clones.
Do not remove the unused legacy shared-stream helper in this diagnostic patch;
cleanup is separate from the causal test. No dependency/site-packages changes.

Prior art: vLLM-Ascend 12944/13600 inspired A's ordering, not this isolation
hypothesis. Public Ascend Stream accepts an explicit device; its source is not
proof of the installed build or of private-stream correctness:
https://raw.githubusercontent.com/Ascend/pytorch/v2.10.0/torch_npu/npu/streams.py

## Evidence boundary and readiness

User reports the memory residue is an existing container cleanup issue involving
remaining threads, and owns recovery. Do not diagnose it as a driver defect from
HBM alone. Wait for operator confirmation that cleanup is complete, and record
the allocated device, host/container process ownership and restored idle HBM
before each fresh service. No reset or host-wide kill is authorized by this packet.
If residue remains, stop and return to the operator; no automatic retry.

Historical A stalled with scheduler in graph_task_update_end and encoder in
mask.to(device=device); neither a Device dependency cycle nor its precise owner
was observed. A had unrecoverable launch dirty state outside the verified files.
Use CLEAN tracked worktrees for both new arms; preserve old artifacts elsewhere.
No reset/stash of existing worktrees. Local mocks do not qualify NPU execution.

Record actual imported paths and full hashes of backend, submission module,
encoder graph runner and sglang_model.py, plus Git HEAD/status. Launch from the
selected Omni checkout; an old site-packages Omni must not be imported. Preserve
Python 3.11.10, torch 2.10.0, torch_npu 2.10.0.post2, CANN 9.0.1 and record driver,
model/data identity. If unavailable or different, stop before load; do not install.

## Local/focused preflight

Use actual checkout paths for the following variables. These commands do not
install dependencies. On B:

```bash
cd "$OMNI_WORKTREE"
python scripts/npu/test_private_encoder_update_stream.py
python -m pytest tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py -q
cd "$SGLANG_WORKTREE"
python scripts/npu/test_async_submission.py --omni-root "$OMNI_WORKTREE"
python -m pytest test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py -q
```

Require all tests pass, no skipped encoder path. New standalone constructor tests
are 3; SGLang standalone tests are 4. Native suites must be collected fully; record
actual count rather than forcing an old count. Constructor mocks verify ownership
and non-NPU behavior, not device execution. Stop before workload on test failure.

## Frozen workload

Use the retained WF-009 rev2 SeedTTS EN 140 requests, cold conc8/140, NO warmup.
Recover the FULL evaluation_input_sha256 from that evidence (known prefix
9f631ab7), verify all 140 ordered inputs, and record it BEFORE either arm. Prefix
matching alone is insufficient. Do NOT use exact10s 700 or its 22bb330f manifest.
If the full retained identity cannot be recovered, report blocked, do not invent
a replacement corpus. Save launcher/evaluator scripts, full SHA256 and commands.

Effective profile in BOTH arms:

- enable_torch_compile=false; prefill=breakable; decode=full;
- encoder graph=true; disable_cuda_graph=false;
- max_running_requests=64; cuda_graph_max_bs=64;
- pre-LM encoder worker active; same model, request order, generation parameters,
  cache policy, seed, logging and timeouts as rev2 for both arms.
- Unset WF-008 trace/serialization knobs; no tensor repr diagnostics, monkeypatch
  tracing, extra synchronizations, warmup requests or profiling-only variants.

Initialize the evidence directory variable explicitly, check it is nonempty, and
ensure all client output goes under it (never /measure.log). Manage client/server
as owned processes; retain unbuffered progress and raw completed-response records.

## Sequence, timeouts and cleanup

1. Operator restores baseline. Fresh A, one cold conc8/140 run. Record a pass or
   failure; a failure is a diagnostic control result, not a reason to declare B
   failed. Do not repeat A trying to obtain a preferred result.
2. Shut down A; operator recovers any residue. Only after baseline is restored,
   start fresh B1, identical cold conc8/140.
3. If B1 passes, run B2 and B3, each fresh process and baseline check. Stop B
   expansion at its first failure; no B2/B3 after B1 failure.

Allow up to 15 minutes for service readiness; no requests are sent before ready.
During workload, 60 seconds without a completed response is a failure (including
the first response). Absolute workload limit is 10 minutes. Use an independent
watchdog, not the possibly blocked client's final summary, to enforce limits.
At the first error/timeout/empty output/107033/107027 or no-progress limit, stop
new client load, preserve first failure and partial results, and collect existing
logs plus two bounded simultaneous Python-stack snapshots if the process is still
available. Any snapshot command has a 15-second timeout. Collect already available
runtime logs/native traces; do not change runtime logging or attach GDB ad hoc.
Native data absent is UNKNOWN, not proof that no runtime error occurred.

Then SIGTERM owned service/client processes, wait at most 60 seconds, and request
operator recovery for residual host/container threads. Forced termination must
be explicit in the report, never counted as normal cleanup. No subsequent service
until ownership and idle baseline are confirmed restored. Do not consume another
round solely to recreate lost evidence.

## Attestation and decision

Existing submission markers print ONCE per source. They can attest initial Host
completion and compute/update stream identities, not last progress, per-request
replay counts or device completion. Require BOTH source markers:

- A: encoder update ID equals decoder update ID.
- B: encoder update ID differs from decoder update ID and its compute stream;
  decoder still uses its baseline update path. Compare within each process only.
- A/B: capture logs must show all graph families initialized; report any fallback.

For each completed B round require 140/140, no request errors/timeouts/empty outputs,
no per-sample WER > 0.5, no runtime errors and normal shutdown with recovered
baseline. Report aggregate WER and all raw responses. Partial-run WER is not corpus
WER and cannot qualify correctness. These thresholds are a bounded diagnostic
screen, not a final quality/non-regression claim.

Use existing replay/fallback counters if available. If unavailable, say UNKNOWN;
HTTP completion and capture alone do not attest all-graph replay. Do not add
in-process logging to only one arm to satisfy counters. Missing attestation leaves
final all-graph qualification pending even if the liveness screen passes.

- A fails / B three rounds pass: supports stream-sharing involvement under these
  conditions; does not prove an exclusive CANN root cause. Recommend next focused
  replay/correctness qualification before moving B into PR #2160.
- Both fail: isolation alone insufficient; stop proposing further locks/order
  variants. Review first failure for native/event/capture diagnosis.
- Both pass: historical race not reproduced on restored environment. B passes
  this screen but causal attribution remains unresolved. No performance claim.
- Identity, isolation marker or cleanup mismatch: affected result is invalid or
  blocked, not evidence for/against stream isolation.

Return full identity/config/input digests, instruction revision and packet digest,
tests actually run, per-arm stream IDs, completed/evaluated counts separately,
first failure time and stacks, correctness/replay gaps, cleanup mode/baseline,
and server evidence directory IDs. Keep raw private paths/data/logs on-server.
