# Qwen3-ASR 1.7B on Ascend 910B: hardware handoff

This page is the source of truth for the local-development and isolated-Ascend
qualification effort. It is written for the local SGLang-Omni developer and
the operator of the isolated 910B server. Update this page when a gate changes;
the companion task pages contain executable procedures, not competing status
or support claims.

**Server-task synchronization rule:** every new request for isolated-server
execution must be written into this handoff before the operator runs it. A
server experiment mentioned only in chat is not an executable task. Each
completed run must replace or close the previous task here before another
server-side variable is introduced.

**Server source-code authority rule (hard constraint):** the isolated server
is an execution and evidence environment, not a development authority. Its
operator may check out only the exact repository commits named by the current
handoff, run the authorized commands, retain raw evidence locally, and perform
the declared cleanup. The operator must not edit source, tests, configuration
files, benchmark code, or documentation; create commits; apply unreviewed
patches; or repair a failure in place. If execution indicates that code must
change, stop at the first complete failure and return the affected repository,
file/symbol or operation boundary, proposed change, supporting sanitized
evidence, and required test coverage to the local Codex owner. The local owner
implements, reviews, tests, and commits the change, updates this handoff, and
only then authorizes a fresh-process server verification at that exact commit.
Historical server-side diagnostic commits remain evidence records but are not
precedent for future server edits.

## Scope and status

Target topology: one Ascend 910B, one Qwen3-ASR-1.7B stage, BF16, no model
quantization, and no tensor or data parallelism.

Status at base commit `e7d876b28326c55d777ae62e1c3650b816785d8c`:
**A3 eager functional baseline and generation graph capacities 1 through 70
with torch compile disabled passed capture and single-request replay; the named
encoder-eager candidate passes two concurrent requests after both inputs are
warmed but hangs only when uncached encoder work overlaps enabled generation
graph execution; disabling generation graph removes the hang, while compiled
generation and encoder graph remain unqualified. Synchronous request building
also removes the hang; disabling only prefill graph removes it while decode
graph remains captured, making prefill-eager/decode-graph the bounded-
concurrency candidate. The dataset/client-environment blocker is resolved, but
that candidate hung at the first concurrency-8 cold-input level even though
prefill graph was disabled. Successive diagnostics localized the unmatched
decode forward to the NPU graph input-update lane: `graph.replay()` returned at
the host API boundary, but the update thread never returned from
`graph.update()` and the main thread remained in its join. The local Qwen3-ASR-
and-NPU-specific mutual-exclusion fix completed the cold-input concurrency
ladder at 8, 16, 32, 64, and 70. The clean `910C-022` and `910C-023` runs
recorded balanced guard triplets, zero eager decode fallback, and real replay
at buckets 32, 64, and 70. This disproves an intrinsic high-concurrency FIFO-
guard deadlock; the earlier failed process was environment-contaminated by
residual NPU activity. The two fresh `910C-023` runs each evaluated 70/70,
independently closing the current capacity result despite the historical
`910C-022` 65/70 scoring anomaly. This qualifies the explicit A3 functional
candidate only. No exact-10-second hard-target performance or realtime
qualification has run**.

The first remote run used `Ascend910_9382`, which the current Ascend ecosystem
identifies as A3 hardware. The project owner states that its single-card
compute is equivalent to the target 910B, so this host is accepted as the
single-card performance proxy and compute capacity is not a separate blocker.
It is still recorded under the derived `910C-*` run series: chip identity,
CANN/ATB/driver behavior, and runtime compatibility remain device-class
qualification concerns, so these results do not automatically certify the
same software stack on a physical 910B.

| Area | Repository state | 910B evidence |
|---|---|---|
| Ascend installation | NPU manifest, precheck, and installation guide are implemented | Precheck passed on the compute-equivalent A3 performance proxy; runtime compatibility has not been repeated on a physical 910B |
| Qwen3-ASR model path | Single-stage model, batching, pre-LM encoder, SSE output, and long-audio upload chunking are implemented | A3 eager batch 1, two-concurrent, ten-sequential, health, shutdown, and restart gates passed after repairing the OpenCV environment; not yet started on 910B |
| Generation graph | Enabled by the Qwen3-ASR defaults and delegated to SGLang | In the explicit compile-disabled, prefill-eager profile, A3 decode capacities 1 through 70 captured and the cold-input ladder passed through concurrency 70 with zero eager fallback; bucket 70 replayed 11 times. With compile enabled, batch 1 failed at Dynamo/triton-ascend and batch 64 failed at ATB `PagedAttentionOperation`; the repository-default graph profile remains unqualified |
| Encoder graph | Implemented with `torch.cuda.Stream`, `torch.cuda.CUDAGraph`, and `torch.cuda.graph`; enabled by default | A3 capture failed for every attempted bucket because captured-stream synchronization memcpy is unsupported; each bucket explicitly stayed eager |
| Pre-LM encoder service | NPU tensors use the default device stream; the dedicated stream path is CUDA-only | A3 eager functionality and restart stability passed; local commit `29ca236f` adds a FIFO device-execution guard shared only by the Qwen3-ASR encoder batch and generation forward when NPU generation graph is enabled. Cold-input concurrency 8, 16, 32, 64, and 70 completed on clean processes with balanced guard events and state drain |
| SSE transcription | Emits decoder-token deltas after the complete upload has entered one engine request | Not continuous audio-input realtime |
| Realtime WebSocket | Buffers PCM16 until VAD stops, then runs a response pass followed by a transcription pass | Does not meet the incremental-ASR target by inspection; not yet verified on 910B |

The repository therefore supports installing SGLang-Omni on Ascend, but does
not yet claim that Qwen3-ASR is supported or qualified on 910B. A check mark is
added only after redacted evidence from the isolated server passes the relevant
gate.

## Acceptance contract

The performance values below are targets, not measured results. Correctness and
stability gates are prerequisites; an optimization that changes recognized text
outside the declared accuracy allowance or silently falls back is a failure.

### Offline transcription gate

- Input set: a frozen manifest of 16 kHz, mono, PCM16 speech clips, each
  `10.000 s` with a maximum duration error of one audio sample. The manifest,
  language mix, transcript references, and file hashes stay on the isolated
  server; the returned evidence contains only an aggregate manifest hash and
  aggregate metrics.
- Workload: closed-loop concurrency `70`, one uploaded clip per request, no
  repeated-audio cache hits, explicit language hints matching the manifest,
  and non-streaming JSON responses.
- Warm-up and sample size: discard at least one complete concurrency wave, then
  measure at least `700` requests per repeat and run three repeats in fresh
  server processes.
- Latency interval: client timestamp immediately after the complete multipart
  body is written through receipt of the final response body. Report mean,
  p50, p90, p95, p99, and maximum. The hard target is p95 below `500 ms` in
  every repeat.
- Capacity: all measured requests return success; request shedding, timeouts,
  empty responses, and retries count as failures. At the latency target,
  concurrency 70 implies approximately `140 requests/s` and, for exact 10 s
  clips, `1400 input-audio-seconds/s`; report measured throughput and RTFx
  rather than treating those derived values as separate proof.
- Accuracy: first freeze the eager-NPU baseline on the same manifest. Later
  variants must produce the same normalized transcript for the smoke set and
  must not regress corpus WER/CER from that baseline. Public SeedTTS checks use
  the repository's existing ASR thresholds rather than duplicating them here.

### Realtime transcription gate

“Realtime” means audio is accepted continuously and partial recognition is
produced before end-of-utterance. SSE deltas from a fully uploaded file and a
VAD-triggered whole-utterance request do not satisfy this definition.

- Transport: 70 simultaneous WebSocket sessions; PCM16 mono at 16 kHz; one
  `500 ms` chunk appended every `500 ms` for a 10 s utterance.
- Partial latency: for every eligible chunk after speech start, measure from
  completion of that chunk's append event to the first transcript event that
  incorporates it. The target is p95 below `500 ms` across sessions.
- Final latency: measure from the final audio commit/end-of-speech event to the
  completed transcript event. The target is p95 below `500 ms`.
- Stability: all 70 sessions complete without disconnect, request shedding,
  cross-session transcript leakage, unbounded buffer growth, or device OOM.
- Semantics: partial text may revise only through an explicit protocol event;
  the final normalized text must match the offline result for the same audio.

The first realtime milestone is transcription only. Conversational response,
TTS output, word timestamps, forced alignment, multi-card execution, and model
weight changes are outside this qualification.

## Ownership boundary

- This repository owns Qwen3-ASR configuration, model adaptation, batching,
  audio ingress, benchmark tooling, tests, operator documentation, and any
  platform-neutral graph/stream dispatch added here.
- SGLang owns NPU scheduler, attention backend, generation graph, allocator,
  and device runtime behavior. A defect reproduced below the Omni adapter needs
  a minimal upstream reproducer and an exact SGLang revision.
- `torch_npu`, CANN, `triton-ascend`, and `sgl-kernel-npu` own device operators
  and compiler/runtime behavior. The server operator records exact versions
  chosen from the official compatibility matrices; this repository does not
  invent a second version matrix.

Repository ownership identifies where the local change must land; it does not
authorize the isolated operator to modify that repository. All future code
changes, including diagnostic instrumentation and test-only changes, are made
and committed locally before server execution.

Do not patch `site-packages`, copy private model artifacts into the repository,
or convert a failed graph path into an unreported eager fallback. A fallback is
acceptable only when it is explicit in configuration and independently
qualified.

## Derived A3 qualification record

The isolated operator executed steps 1 through 4 of the first validation task.
The full logs remain in the server-local `qwen3-asr-910c-000` evidence
directory; this section contains only the reviewable, redacted result.

### Frozen environment and completed gates

- SGLang-Omni was detached at clean commit `e7d876b2`; the NPU precheck passed.
- Hardware was `Ascend910_9382`, 16 devices with 64 GiB HBM per device. The
  qualification topology used one device.
- Runtime fingerprint: CANN toolkit 9.0.1, PyTorch 2.10.0+cpu,
  `torch_npu` 2.10.0.post2, SGLang package 0.5.18,
  `triton-ascend` 3.2.1, and SGLang-Omni 0.1.3 at `e7d876b2`. The initial runs
  did not return the exact SGLang Git HEAD. Run `910C-004` later recorded
  `71de97b264b04dcd514cf904003028aefe9775c8`; that commit is required unchanged
  for subsequent comparison but is not retroactive proof of the initial state.
- The NPU installer suite passed 22 tests. The focused Qwen3-ASR suite passed
  588 tests with 3 skipped. An earlier collection failure came from an old
  editable SGLang 0.5.16 fork and disappeared after the operator installed the
  intended 0.5.18 package; it is an environment correction, not model evidence.
- Default startup failed after prefill graph capture succeeded and decode graph
  capture failed while setting up `PagedAttentionOperation`.
- The explicit eager diagnostic profile became ready. Its first smoke request
  returned HTTP 500, so the operator correctly stopped before the two-request,
  ten-sequential, restart, concurrency, and performance gates.
- All service processes were stopped, ports were released, and device memory
  returned to the approximately 3 GiB idle baseline.

### First complete eager failure

With synchronous NPU launch enabled, the first request reached the audio tower
and failed at its first-layer `conv2d`. `torch_npu` reported
`AclSetCompileopt(ACL_PRECISION_MODE)` error 500001. The nested CANN error is
the actionable failure: `GEInitializeV2` could not initialize because
`multiprocessing.Manager` instantiation failed, after which TBE custom-store,
fusion-manager, and ops-manager initialization also failed.

The failure was stable across three requests. Host memory, file descriptors,
shared memory, inodes, and device HBM were all sufficient. Five progressively
closer standalone probes passed, including BF16 `conv2d`, non-JIT compile mode,
a thread-pool worker, and a spawn child process whose worker thread executed
the operator. Those probes rule out a general A3 `conv2d` failure.

### Resolved eager root cause and ownership

The daemon-process diagnosis recorded in commit `fa27495d` is rejected by run
`910C-001`. The operator first confirmed the Python invariant independently:
`multiprocessing.Manager()` raises `AssertionError: daemonic processes are not
allowed to have children` in a daemon spawn child and succeeds in an otherwise
equivalent non-daemon child. The service-side A/B then changed only the ASR
stage process to `daemon=False`; an explicit diagnostic line confirmed the
failing stage PID was non-daemon. Batch 1 nevertheless failed in that same PID
with the unchanged CANN Manager EC0009, `GEInitializeV2`, and error-500001
chain. The checkout was restored to `daemon=True` and a clean worktree after
the diagnostic run.

Run `910C-002` resolved the previously unknown boundary. The CANN wrapper hid
an `EOFError` in the Manager parent. Its `SyncManager` server child exited while
the spawn bootstrap re-imported the `sgl-omni` main module: the import chain
reached `cv2.typing`, then `cv2.mat_wrapper`, which failed because
`libGL.so.1` was absent. The child closed its bootstrap pipe without returning
the Manager address, the parent raised `EOFError`, and CANN converted it to
EC0009 before the outer GE and `AclSetCompileopt` failures.

This was an environment dependency collision, not a CANN, `torch_npu`,
SGLang, SGLang-Omni, daemon-process, or `conv2d` defect. Both OpenCV wheel
variants had been installed into the same `cv2` namespace, and the later
non-headless installation won. The operator removed `opencv-python` and
force-reinstalled `opencv-python-headless` 5.0.0 without dependencies. This is
the exact validated A3 repair, not a repository-wide version pin. Afterward
`cv2` imported successfully and its binary had no unresolved `libGL`
dependency; the CANN Manager/GE failure signature disappeared.

The operator also found and force-stopped a stage process orphaned from the
initial validation for more than 12 hours, with PPID 1 and approximately 55 GiB
of chip-0 HBM. It contaminated earlier retry observations but was not the batch-1
cause: after removal, HBM returned to the approximately 3 GiB idle baseline and
a fresh non-daemon run reproduced the same failure. The orphan is retained as
a separate shutdown/reap defect relevant to the later stability gate.

### A3 eager qualification after environment repair

- The NPU installer suite passed 22 tests and the focused Qwen3-ASR suite passed
  588 tests with 3 skipped after the repair, matching the pre-repair counts.
- Eager batch 1 returned HTTP 200. Its approximately 30.2-second latency
  included first compilation and is not a performance measurement.
- Two concurrent requests both returned HTTP 200 in approximately 0.64 seconds
  wall time, with different output hashes for different inputs and no observed
  cross-request contamination.
- Ten sequential requests all returned HTTP 200 in approximately 0.35--0.40
  seconds each, with one stable output hash. These clips were functional smoke
  inputs, not the frozen exact-10-second performance corpus.
- Peak chip-0 HBM was 55,756 MiB of 65,536 MiB and remained stable. Health,
  graceful shutdown, process cleanup, port release, and a fresh-process restart
  all passed. The restart request reproduced the batch-1 output hash.
- The only scanned `ERROR` was an unrelated optional NIXL import failure; there
  was no traceback, OOM, NaN, device reset, or fallback marker.

The eager result qualifies only this A3 environment and functional workload.
It does not qualify the original 910B target, concurrency 70, exact-10-second
latency, realtime ingress, or generation graph mode.

### Compile-enabled batch-64 ATB failure

Run `910C-003` repeated default startup after the OpenCV repair. Prefill graph
capture succeeded, but decode graph capture failed on its first, largest bucket
at batch size 64 with approximately 9.28 GiB available. The complete log had
zero occurrences of EC0009, Manager instantiation, `GEInitializeV2`,
`EOFError`, `AclSetCompileopt`, error 500001, `libGL`, or `cv2`. The first
failure is instead ATB `PagedAttentionOperation setup failed` from
`OpParamMaker.cpp` and `AtbCommon.cpp` during SGLang's decode NPU graph
capture.

Run `910C-006` later captured the same batch-64 ladder successfully with torch
compile disabled and no ATB signature. The original failure is therefore not a
pure batch-64 shape or HBM-capacity failure. It is conditional on the global
compile-enabled configuration or state established by that path. This does not
prove that bucket 64 itself was compiled: under the reported default
`torch_compile_max_bs=2`, it was not a member of `compile_bs`. The remaining
classification is a compile-state/capture-order interaction at the SGLang NPU
graph-runner and ATB boundary. A minimal reproducer must isolate what persistent
model, backend, workspace, or operation state compile initialization changes
before the largest bucket is captured.

### Capacity-one torch-compile failure

Run `910C-004` followed the stop rule and ended at capacity 1; capacities 16,
32, and 64 were not run. The resolved profile was `cuda_graph=True`, decode
capture buckets `[1]`, `enable_torch_compile=True`,
`max_running_requests=1`, and `mem_fraction_static=0.837`. Prefill capture
succeeded in 7.80 seconds. Decode capture then failed at zero progress in
TorchDynamo before reaching ATB attention setup:

```text
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
triton/backends/ascend/driver.py: NPUUtils.get_device_properties
```

The batch-1 log contains no `PagedAttentionOperation`, `OpParamMaker`, or
`AtbCommon` signature. Conversely, the earlier first failure at batch 64 did
not contain the Dynamo skipped-function signature. These are distinct first
failures, not evidence of a capacity threshold.

At the exact SGLang commit used by the server,
`get_batch_sizes_to_capture()` places only capture buckets less than or equal
to `torch_compile_max_bs` in `compile_bs`. Qwen3-ASR currently defaults that
threshold to 2. Therefore the batch-1 arm is a compiled bucket, while the
first, largest batch-64 bucket from `910C-003` is non-compiled under the
reported unchanged defaults. Run `910C-006` nevertheless shows that disabling
the global compile mode removes both failure signatures. The evidence supports
a compile-configuration dependency, but not the stronger claim that both
failing buckets execute compiled forward code.

The immediate unsupported call is implemented by `triton-ascend`, but final
fix ownership is not established by the stack alone. SGLang owns the compiled
forward boundary and should avoid tracing device discovery if the value can be
resolved and cached before compilation. A minimal reproducer against the exact
SGLang and triton-ascend revisions must determine whether that integration
change is sufficient or whether the driver must make the query safely usable
by compiler consumers. Do not apply Dynamo trace-forcing decorators as a
diagnostic workaround: they can bypass safety checks or introduce graph breaks
without proving capture/replay correctness.

### Capacity-one generation graph pass

Run `910C-005` changed only `enable_torch_compile` from true to false while
keeping capacity 1 and generation graph capture enabled. Its resolved profile
was `cuda_graph=True`, decode buckets `[1]`, `torch_compile=False`,
`max_running_requests=1`, and `mem_fraction_static=0.837`. Prefill capture
succeeded with the breakable backend in 7.61 seconds, and decode capture
succeeded with the full NPU graph backend in 0.88 seconds. The known Dynamo,
ATB, GE/Manager, error-500001, and OpenCV signatures were absent.

Exactly one smoke request returned HTTP 200. Its normalized output hash matched
the frozen eager hash, and the request log explicitly reported
`npu graph: True`; no generation-graph fallback was reported. The 12.87-second
request latency included first-use encoder compilation and is diagnostic only,
not performance evidence. This passes capacity-1 generation graph capture and
replay only for the explicit compile-disabled configuration. It does not
qualify the default compiled path or any larger bucket.

### Compile-disabled generation capacity pass

Run `910C-006` retained the `910C-005` compile-disabled configuration and
changed only paired generation capacity. Fresh-process arms at 16, 32, and 64
all passed prefill and decode capture. Their resolved decode ladders ended at
16, 32, and 64 respectively; decode capture took 1.37, 1.63, and 2.01 seconds,
and left approximately 9.01--9.02 GiB available.

Each arm sent one smoke request. All returned HTTP 200 with the frozen eager
output hash, logged `npu graph: True`, and reported zero generation-graph
fallback. The Dynamo, ATB, GE/Manager, error-500001, and OpenCV signatures were
absent. The six known encoder capture failures retained exactly the previously
classified buckets and error signature. Every process shut down cleanly and
HBM returned to the idle baseline.

Together with `910C-005`, this qualifies generation graph capture and a
single-request replay smoke for configured capacities 1 through 64 with torch
compile explicitly disabled. It does not prove that the maximum bucket replayed
under concurrency, does not cover the target concurrency 70, and does not
qualify the encoder graph or compile-enabled mode.

### Target-capacity generation graph pass

Run `910C-007` changed only the paired generation capacity from 64 to 70 while
retaining the compile-disabled `910C-006` configuration. The resolved decode
list was `[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 70]`; SGLang neither
clamped nor omitted the target bucket. Prefill capture succeeded, and all 13
decode buckets captured in 2.13 seconds using 0.27 GiB, leaving approximately
8.99 GiB available.

The one allowed smoke request returned HTTP 200 with the frozen eager output
hash, explicit `npu graph: True`, zero generation fallback, and none of the
known Dynamo, ATB, GE/Manager, error-500001, or OpenCV signatures. The encoder
capture failure retained its exact six buckets and known error signature. The
service shut down cleanly and HBM returned to the idle baseline.

This passes generation capacity-70 capture and a single-request replay smoke
for the explicit compile-disabled mode. It does not prove that a 70-request
decode batch selected and replayed bucket 70.

### Named-candidate two-request hang

Run `910C-008` changed only `enable_encoder_cuda_graph` from true to false,
making encoder eager execution an explicit configuration instead of a failed
capture fallback. Startup met that contract: generation captured through
bucket 70, while encoder capture attempts, `bucket stays eager`, and encoder
`aclrtMemcpy` 107030 counts were all zero. Both frozen clip hashes remained
unchanged.

The single-request level passed after warm-up in 0.128 seconds with
`npu graph: True`. Its earlier first encoder cache miss took 12.46 seconds. The
two-request wave then submitted the two distinct clips within 2 ms. Both clients
timed out after 120 seconds with no response bytes. The service remained alive
but fixed at two running requests and two pending completions. Only one prefill
was observed and no decode batch followed. HBM remained stable, with no OOM,
traceback, graph fallback, or device error. The operator correctly stopped and
cleaned the process before the sequential and larger-concurrency levels.

The first level warmed only one clip; the failed wave introduced the other clip
as a cold pre-LM encoder cache miss. Therefore the evidence does not yet prove a
general two-request scheduler deadlock or that disabling encoder graph itself
is causal. On NPU, `Qwen3ASRPreLMEncoderService` has no dedicated stream, so
encoder execution in the request-builder path and generation execution share
the default device stream across threads. Separately, the scheduler drains
pending request-build futures in insertion order, so one unfinished build can
hold a later completed build behind it. These are code-level risk boundaries,
not confirmed root causes. The 40 ms prefill-coalescing deadline alone cannot
explain a 120-second stall.

The passing two-request eager result from `910C-003` is not a clean A/B for
this failure: generation graph mode, admission capacity, encoder-graph setting,
and warm-up state all differed. It cannot presently assign causality to
`max_running_requests` or the encoder-graph flag.

### Both-warm two-request pass

Run `910C-009` retained the exact `910C-008` named-candidate configuration and
changed only request warm-up order. In one fresh process, clip A completed in
13.11 seconds with 12.82 seconds of encoder time and the frozen output hash.
Clip B, which had different audio bytes and had not been requested in that
process, then completed sequentially in 0.188 seconds with its own frozen hash.
The latter result shows that the failed overlap is not explained by first
compiler initialization alone.

After both embeddings were warm and all request state had drained, one
synchronized A+B wave completed in 1.87 seconds. Both requests returned HTTP
200 with their respective frozen hashes. The service logged one prefill with
`#new-seq: 2` and `npu graph: True`; graph fallback and forbidden-error counts
were zero, and coordinator state drained after completion.

This rules out a general two-request scheduler deadlock under the tested named
candidate. It establishes that uncached encoder work or its first-use
audio-shape state is a necessary condition for the `910C-008` failure under
the tested ordering. It does not distinguish shared-default-stream interaction
from request-build head-of-line blocking, nor does it prove that encoder cache
miss alone is sufficient. A warm-corpus workaround is not an acceptable
performance qualification because the offline contract requires distinct
audio bytes with no repeated-audio cache hits.

### Generation-graph cold-overlap A/B

Run `910C-010` used two fresh processes with the exact same warm-A/cold-B
request order and changed only generation graph enablement. Arm A retained the
capacity-70 graph profile. After clip A warmed and state drained, the A+B wave
reproduced the failure: both clients timed out after 120 seconds with no bytes,
coordinator state remained at two running requests and two pending completions,
only one `#new-seq: 1` graph prefill appeared, and no decode followed. HBM was
stable and no graph fallback or device error was reported.

Arm B explicitly disabled generation graph. The same warm-A/cold-B wave
completed in 0.44 seconds with two HTTP 200 responses and the two frozen output
hashes. Its two requests coalesced into one `#new-seq: 2` prefill with
`npu graph: False`, and all state drained. Encoder graph remained explicitly
disabled in both arms, so neither arm attempted encoder capture or emitted the
known encoder fallback signature.

This establishes generation graph execution as a necessary condition for the
observed cold-encoder hang under the tested configuration. Together with
`910C-009`, the failure requires both an uncached encoder operation and enabled
generation graph execution; removing either condition avoids it. It does not
yet prove that shared NPU default-stream execution is the complete mechanism.
The asynchronous request-builder lets encoder device work overlap scheduler
generation, and insertion-ordered future draining can amplify a blocked first
build, so one further serialization A/B is required before selecting a source
fix.

### Synchronous request-build pass

Run `910C-011` retained the graph-enabled `910C-010` Arm A profile and changed
only `request_build_max_workers` from 8 to 1. The resolved scheduler reported
one worker and no asynchronous build-pending or backlog growth. Generation
captured through bucket 70, while torch compile and encoder graph remained
explicitly disabled.

After warming only clip A, the warm-A/cold-B wave completed in 1.88 seconds.
Both requests returned HTTP 200 with their respective frozen hashes, generation
logged `npu graph: True`, fallback and forbidden-error counts were zero, and
all state drained. The two requests used serial `#new-seq: 1` prefills, as
expected when request construction and the blocking encode execute on the
scheduler thread.

This establishes asynchronous request-building overlap as another necessary
condition for the observed hang. The tested failure requires enabled generation
graph, asynchronous request building, and an uncached encoder operation at the
same time; removing any one avoids it. Insertion-ordered future draining can
propagate the blocked build to later requests, but is not an independent root
cause. One build worker is a diagnostic result, not a performance candidate:
it removes request-build parallelism and encoder batching that the measured
GPU profile needs at higher concurrency.

### Prefill-eager/decode-graph pass

Run `910C-012` returned to eight asynchronous request-build workers and changed
only prefill graph enablement from the failing `910C-010` Arm A profile.
Prefill graph was explicitly disabled, decode graph still captured every bucket
through 70, and encoder graph remained explicitly disabled. After warming only
clip A, the warm-A/cold-B wave completed in 0.19 seconds with two HTTP 200
responses and both frozen hashes. The two requests coalesced into one
`#new-seq: 2` prefill with `npu graph: False`; fallback and forbidden-error
counts were zero and all state drained.

This isolates prefill graph, rather than decode graph in general, as necessary
for the observed NPU cold-encoder hang. The result supports a zero-source-change
prefill-eager/decode-graph candidate with eight build workers. It does not yet
prove decode replay in that exact run: the short outputs ended before the
default decode logging interval and `GET /model_info` did not expose a decode
replay counter. That missing attestation should be collected during the
bounded-concurrency ladder, where lowering the decode log interval is an
evidence-only setting; another two-request clip experiment would not add a
distinct functional boundary.

### Pinned SeedTTS staging blocker

Run `910C-013` stopped before service startup because the isolated host had no
SeedTTS cache and both the default Hugging Face endpoint and configured mirror
failed through the enterprise TLS proxy. The default endpoint returned a proxy
504 and the mirror connection closed during TLS handling. The operator did not
install a dependency, edit the benchmark, substitute private data, or run the
concurrency ladder. Device, port, process, HBM, and tracked-worktree state
remained clean.

This is an environment prerequisite, not a model or candidate-profile failure.
The approved recovery is an offline transfer of the standard Hugging Face cache
created on a connected environment from repository
`zhaochenyang20/seed-tts-eval-arrow` at exact revision
`27f4c1adee83b5b29b7c4b375f6b976324bda308`. The connected environment must use
the same checked-out benchmark and compatible `datasets` and `huggingface_hub`
versions, set an otherwise empty explicit `HF_HOME`, run the documented
`benchmarks.dataset.prepare` command with the exact revision, then repeat that
command successfully with `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`.

Archive that complete `HF_HOME` while preserving its directory layout and
links, record the archive SHA-256 and package versions, transfer it through the
approved isolated-server channel, and extract it into a new explicit directory.
On the target, verify the archive SHA-256, point `HF_HOME` at the extracted
directory, set both offline variables, and rerun the same prepare command. A
successful offline prepare is the resume signal for `910C-013`; any network
attempt, missing revision, cache rebuild failure, or dataset-schema error keeps
the run blocked. The cache archive, audio, paths, and transcripts remain local;
returned evidence contains only the repo ID, revision, archive hash, package
versions, split/sample counts, and benchmark evaluation-input hash.

For the English-only `910C-013` ladder, a smaller verified local Parquet
snapshot is also approved. Download from that exact revision, not `main`, and
preserve this layout:

```text
seed-tts-eval-arrow-27f4c1ad/
  README.md
  data/
    en-00000-of-00001.parquet
```

The English Parquet must be exactly 247,555,423 bytes with SHA-256
`5849b41b49cae996328c06d2c5791717c3bafc369bddfa1ec4f86761bb8bc0ca`.
Transfer the directory or a hash-recorded archive through the approved channel.
On the server, verify size and SHA-256, then pass the snapshot root to the
benchmark as `--meta <snapshot-root>` with `--lang en`; do not pass a different
dataset revision or rename the Parquet split. Before service startup, call
`load_seedtts_samples(<snapshot-root>, max_samples=70, split="en")` in a
short-lived process and require exactly 70 samples plus 70 readable, distinct
audio inputs. Record the upstream repo/revision and Parquet hash separately,
because local-path benchmark provenance does not infer a Hub revision.

The benchmark also accepts a local `meta.lst`, but that fallback is not approved
for this recovery because the current repository has no pinned export command
that records upstream identity and per-file integrity. The verified Parquet
snapshot above is not that fallback: its exact upstream LFS SHA-256 and split
layout are fixed. Repeated repository test clips cannot replace the dataset:
cache hits and absent corpus references would invalidate cold-input concurrency
and WER evidence.

The verified English Parquet subsequently exposed a second client-environment
blocker before service startup: the existing `pyarrow 25.0.0` reader raised
`ArrowInvalid: Index not in dictionary bounds` while decoding its dictionary-
encoded pages. Do not classify this as a general incompatibility between a file
written by Arrow 24 and a reader at Arrow 25. Apache Arrow issue GH-50503 records
the same deterministic `pyarrow 25.0.0` aarch64 dictionary-decode failure on an
affected SVE CPU path, with 24.0.0 unaffected; the 25.0.1 patch release includes
GH-50503's fix. Record `uname -m`, the relevant `lscpu` model/part fields,
glibc version, wheel filename, and exact traceback category to establish whether
the isolated host matches that failure class.

Do not downgrade or upgrade `pyarrow` in the serving environment. Create a
separate benchmark-client virtual environment with access to the existing eval
dependencies, install only the offline `pyarrow 25.0.1` wheel appropriate for
the host, and run dataset loading plus the HTTP benchmark from that environment;
launch the server with its unchanged interpreter. For the current CPython 3.11
aarch64 stack, the approved wheel is
`pyarrow-25.0.1-cp311-cp311-manylinux_2_28_aarch64.whl`, size 46,834,633 bytes,
SHA-256
`880523be3d29efcf83d3998835d206118ccf35e3871dbd2fb60408cf6b007a80`.
Verify architecture, ABI, glibc compatibility, file size, and hash before an
offline `pip install --no-index --no-deps` into that virtual environment.

The approved loader decision is to use the repository's local-Parquet path,
not a `sitecustomize`/monkeypatch redirect, fabricated Hub API cache, generated
`meta.lst`, or substituted corpus. The repository loader recognizes a snapshot
directory, resolves only `data/<split>-*.parquet`, and invokes the local
`parquet` dataset builder without a Hub repo ID. The benchmark's
`--unique-audio` option hashes staged audio bytes, preserves the first sample
for each content hash, and applies `--max-samples` after deduplication. This is
required for `910C-013`: path or sample-ID uniqueness is insufficient evidence
for 70 encoder-cache misses.

The operator temporarily changed the serving interpreter from pyarrow 25.0.0
to 24.0.0 during diagnosis. Before any model server is started, restore that
interpreter to its exact pre-diagnostic package set (including pyarrow 25.0.0),
run `pip check`, and record the restored freeze hash. Pyarrow 25.0.1 belongs
only in the separate benchmark-client environment; a passing ASR unit-test
collection while the global package set is changed does not qualify the
serving environment.

Before resuming `910C-013`, require all of the following: the client environment
imports `pyarrow==25.0.1`; `pip check` succeeds; a full single-threaded
`ParquetFile.iter_batches` scan reads all 1,088 English rows; the repository
loader invoked through the benchmark with `--unique-audio --max-samples 70`
returns exactly 70 requested samples with readable, distinct audio; and
the original serving interpreter still reports its unchanged package set. If
25.0.1 fails any check, stop and preserve evidence; a pyarrow-24 client-only A/B
requires a new recorded decision, not an in-place serving-environment downgrade.

Set `BENCHMARK_PYTHON` to the benchmark-client virtual environment's Python,
then run this preflight after the full Parquet scan and before starting the
service:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "${BENCHMARK_PYTHON}" - <<'PY'
from pathlib import Path

from benchmarks.dataset.seedtts import (
    load_seedtts_samples,
    select_unique_audio_samples,
)

samples = load_seedtts_samples(
    "/home/w00984239/seed-tts-eval-arrow", split="en"
)
samples = select_unique_audio_samples(samples, 70)
assert len(samples) == 70
assert all(Path(sample.ref_audio).is_file() for sample in samples)
print("SeedTTS local snapshot preflight: 70 distinct audio inputs")
PY
```

### Independent encoder graph failure

The same run produced hardware evidence for the encoder boundary previously
identified by inspection. Encoder graph capture failed for buckets 128, 256,
512, 1024, 2048, and 3159. Each failure reported `aclrtMemcpy` error 107030,
that the current capture mode does not support the operation, and that
synchronizing the captured stream is not allowed. The implementation caught
each failure and logged that the bucket stayed eager; the smoke request
therefore used eager encoder execution while generation decode used NPU graph
replay.

This is an SGLang-Omni encoder-graph integration problem constrained by NPU
captured-stream semantics, independent of the SGLang generation graph. The
fallback is observable and preserved correctness, but it is not acceptable as
an unreported default-graph qualification pass. Explicitly disabling encoder
graph is allowed later as a named baseline mode; retaining that mode for the
performance target requires separate latency, concurrency, and memory evidence.
An NPU-native encoder graph fix must identify the operation that initiates the
synchronizing copy, then hoist it outside capture or replace it with an
NPU-capture-compatible path, and prove replay correctness rather than
suppressing the exception.

### Installation hardening follow-up

The repository should add a non-mutating NPU precheck and operator guidance for
this failure class. The check should import `cv2` in a fresh spawn child, report
the installed `opencv-python` and `opencv-python-headless` distributions, and
reject the ambiguous state where both own the same `cv2` namespace. A missing
`libGL` import should explain the two operator-owned remedies: provide the
system library or use one compatible headless OpenCV distribution. The project
installer must not automatically uninstall, replace, or pin an externally
owned OpenCV stack. This is a local implementation task and is not part of the
next server run.

## Next bounded diagnostic task

Run identifier: `910C-013`. Promote the `910C-012` prefill-eager/decode-graph
profile to a bounded functional concurrency ladder using the existing CUDA
SeedTTS benchmark scenario. This run must attest decode replay and cold-input
stability; its short, non-exact-duration clips and evidence logging mean its
latencies are preliminary and cannot satisfy the hard exact-10-second target.

Before resuming the numbered steps, create the cache in a connected Linux
environment from the same repository revision and compatible evaluation
dependencies:

```bash
export HF_HOME=<new-empty-seedtts-cache-directory>
python -m benchmarks.dataset.prepare --dataset seedtts \
  --revision 27f4c1adee83b5b29b7c4b375f6b976324bda308
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m benchmarks.dataset.prepare --dataset seedtts \
  --revision 27f4c1adee83b5b29b7c4b375f6b976324bda308
```

Archive the complete directory with a link-preserving tool and record its
SHA-256. After approved transfer and extraction, the isolated server must point
`HF_HOME` at that directory and run the second, fully offline command. Do not
resume step 1 until it succeeds and reports the pinned revision from cache.
Alternatively, use the exact English-only Parquet snapshot and preflight
defined in the staging-blocker section; that path does not require an HF cache.
If the server's existing client packages reproduce the documented Arrow 25.0.0
dictionary-decode failure, perform the isolated 25.0.1 benchmark-client virtual-
environment recovery from that section and repeat the full scan and 70-sample
loader preflight. The service must not start until one approved client
environment passes.

1. Keep the repaired A3 stack, torch compile disabled, encoder graph explicitly
   disabled, prefill graph explicitly disabled, decode graph captured through
   bucket 70, `max_running_requests=70`, `request_build_max_workers=8`, and the
   `910C-012` cache, pending-build, coalescing, memory, and version settings.
   Use the pinned SeedTTS EN dataset already supported by the repository. For a
   transferred HF cache, retain `HF_HUB_OFFLINE=1` and
   `HF_DATASETS_OFFLINE=1` for the entire run. For the verified Parquet path,
   pass its snapshot root via `--meta` at every level and retain both offline
   variables to prohibit an accidental network fallback. If neither approved
   source passes its offline preflight, stop rather than enabling network
   access, installing an ad-hoc dependency, or substituting private data.
2. Run concurrency levels 8, 16, 32, 64, and 70 in order. Use a fresh server
   process for every level so a prior level cannot warm measured embeddings.
   Before every startup require a clean worktree, free port, no worker/orphan,
   healthy device, and idle-baseline HBM; verify the exact candidate profile,
   decode buckets ending at 70, and zero prefill/encoder graph capture attempts.
3. In each fresh process, send the existing frozen clip A once to complete
   first-use encoder compilation, then wait for all state to drain. Clip A must
   not belong to the measured SeedTTS subset. Run exactly one repeat over the
   first 70 content-distinct pinned SeedTTS EN samples at that level with
   `benchmark_asr_seedtts --meta /home/w00984239/seed-tts-eval-arrow --lang en
   --unique-audio --max-samples 70 --concurrencies <level> --repeats 1`
   and **without** `--warmup`. Every measured audio byte sequence must appear
   once;
   require the encoder-cache statistics delta to show 70 measured misses and
   zero measured hits or merged same-key requests.
4. Set the upstream decode log interval to 1 for this qualification only, using
   `--asr.engine.decode_log_interval 1`, so short outputs still attest each
   decode batch. Record prefill `npu graph: False`, decode `npu graph: True`,
   maximum observed running decode batch size, fallback counts, request-build
   and encoder statistics, coordinator/scheduler drain, peak/steady HBM, NPU
   utilization, completion count, aggregate WER, and preliminary latency and
   throughput. Do not use these logging-instrumented timings as the hard gate.
5. Stop immediately at the first timeout, HTTP failure, wrong/empty transcript,
   WER regression outside the repository SeedTTS threshold, OOM, device error,
   graph fallback, unexpected cache hit, state leak, or orphan. Preserve the
   first complete sanitized failure and do not run higher levels.
6. A level passes only when all 70 requests complete, all timed inputs are
   cache misses, prefill remains eager, positive decode graph replay is logged,
   no forbidden error/fallback occurs, memory remains bounded, and all state
   drains. At concurrency 70 also record whether an actual running decode batch
   of 70 selected graph bucket 70. If the level passes but scheduling never
   forms batch 70, report the observed maximum; capacity-70 client stability
   passes but maximum-bucket replay remains a separate unproven item.
7. Stop and clean normally after every level. Confirm port/process release,
   healthy device, idle HBM, and no orphan before starting the next process.

Do not enable benchmark warm-up, reuse a service across levels, disable the
encoder cache without a reviewed repository change, alter graph/build/admission
settings, add a longer-output diagnostic clip, or run the exact-10-second hard
gate or realtime in `910C-013`. Raw audio, transcripts, paths, and request logs
remain server-local; return only revisions, aggregate dataset identity,
statistics, hashes, and sanitized failure classes.

### `910C-013` first failure and next diagnostic

The dataset and client-environment preflight passed on server commit
`9bae2619`: the serving interpreter was restored to pyarrow 25.0.0, the isolated
benchmark client used pyarrow 25.0.1, the full English Parquet scan returned
1,088 rows, and content-based selection produced 70 distinct inputs. The first
ladder level then hung with eight requests outstanding for more than 90 seconds.
No higher level ran. Prefill graph remained disabled, decode graph was captured
through bucket 70, and 66 decode log records reported `npu graph: True`; this is
valid positive evidence that decode replay executed, but it does not qualify
the candidate because none of the measured requests completed.

Do not infer from the reported encoder `misses: 1` alone that eight requests
entered the encoder queue and failed to finish. In the checked-in service,
`misses` increments when `submit_item()` establishes a cache-miss leader,
before device encoding begins. The next run must distinguish real-time encoder
stats from a stale or periodic log snapshot and record request-build pending,
admission pending, backlog, encoder misses, queue depth, batches, and items at
the same polling timestamps where available.

Run identifier: `910C-014`. Treat `910C-013` concurrency 8 as Arm A; do not
rerun it. Run only Arm B in a fresh process with the exact same stack, pinned
70-input selection, one clip-A first-use warm-up followed by full drain,
concurrency 8, no benchmark warm-up, eight request-build workers, compile
disabled, encoder graph disabled, prefill graph disabled, memory settings, and
120-second bounded timeout. The sole effective variable is decode graph:
disable generation CUDA/NPU graph completely for Arm B. Confirm at startup that
no prefill or decode graph is captured and require measured prefill/decode logs
to report `npu graph: False`.

Run all 70 distinct requests at concurrency 8 so Arm B has the same closed-loop
workload as Arm A. Stop at the first timeout, HTTP/output/accuracy failure, OOM,
device error, state leak, or orphan. If all 70 finish, decode graph execution is
necessary for the concurrency-8 cold-input hang and the next task may locate
the 2-to-8 threshold or instrument NPU stream ownership. If Arm B also hangs,
decode graph is not necessary and the next task must isolate cold encoder
batching/request-building without graph execution. Do not vary worker count,
encoder batch size, coalescing, admission limits, corpus, or concurrency in
`910C-014`, and do not continue to levels 16/32/64/70.

This cold-input requirement remains part of qualification. “70 distinct” means
70 requests in the measured corpus with at most eight simultaneously in flight
at this level, not 70 simultaneous cold encodes. Production utterances are
normally content-distinct, so prewarming the measured corpus would replace the
target workload with cache-hit performance and cannot close this stability
gate.

### `910C-014` result and client preparation

The graph-disabled Arm B completed all 70 HTTP requests at concurrency 8 and
drained coordinator state. Startup and runtime evidence showed no graph capture
and `npu graph: False`; request-build pending peaked at 1 without admission or
backlog growth, and running batch size peaked at 7. Compared with the otherwise
identical graph-enabled `910C-013` Arm A, this establishes decode graph execution
as a necessary condition for the observed cold-input hang.

Classify this as a passed stability-isolation arm, not a complete benchmark
pass. The client raised after all responses while constructing English WER
because `whisper.normalizers.EnglishTextNormalizer` was unavailable, so it did
not write the result JSON or accuracy/latency aggregates. The latest periodic
encoder statistic reported 68 misses; it demonstrates progress but is not a
final 70-miss attestation. Do not reconstruct missing metrics from access logs
or call the incomplete result a performance measurement.

`openai-whisper==20250625` is an exact dependency declared by this repository,
so installing that exact distribution in the isolated benchmark-client virtual
environment is approved and is not an ad-hoc dependency. Do not install it in
the serving interpreter. Use an offline artifact or approved wheelhouse, retain
pyarrow 25.0.1 in the client environment, run `pip check`, and require this
probe to pass:

```bash
"${BENCHMARK_PYTHON}" -c \
  'from whisper.normalizers import EnglishTextNormalizer; EnglishTextNormalizer()'
```

If the exact package requires a build artifact or a declared transitive
dependency that is absent, stop and stage that repository-declared dependency;
do not enable network access or select a different Whisper version. Installing
the dependency does not authorize another model-server run by itself.

Do not rerun graph-disabled Arm B solely to recover WER or latency: its eager
timings are not the performance candidate, and the isolation conclusion is
already established. Do not scan concurrency 3 through 7 yet; a numerical
threshold would be workload-sensitive and would not identify the blocked device
operation. The next development task is a local, reviewable diagnostic change
that exposes encoder enqueue/batch start/encode return/batch finish and the
corresponding request-build/admission state, with timestamps and bounded logging.
Where decode replay begin/end visibility requires an SGLang change, keep that
patch in the SGLang repository and record its actual HEAD and commit mapping.
Only after the diagnostic change has focused tests and a handoff commit may a
new graph-enabled concurrency-8 cold-input run be issued.

Formal performance prerequisites are not yet met. The prefill-eager/decode-
graph candidate passed one two-request cold-overlap wave but failed the first
bounded concurrency level; maximum-bucket replay also remains unproven. The performance contract
requires distinct timed audio bytes or a disabled repeated-embedding cache, so
prewarming the measured corpus cannot bypass this gate. The repository
performance task also records that the NPU-aware exact-10-second manifest
harness has not yet been implemented and tested. Compile-disabled generation
is a valid explicit candidate mode, but it is not evidence that the repository's
compile-enabled default is supported.

Do not run performance or realtime measurements until the functional gates are
green. A server-side diagnostic edit is not a deliverable fix; any confirmed
change must be rebuilt locally, tested, committed in this repository, and
mapped in the evidence table.

### `910C-015` local diagnostic review and `910C-016` task

The server-only diagnostic commit `544b8cd9` was reviewed as design evidence,
not accepted as proof that the code existed locally. Its local, reviewable
replacement is `b64b16d6`. The local implementation keeps the proposed
environment gate and decode graph counters, with three corrections:

- `encoder_enqueue` is emitted from the common `_submit()` boundary because
  Qwen3-ASR overrides `_enqueue()`; instrumenting only the base `_enqueue()`
  would miss the target path;
- the Qwen3-ASR multimodal item carries its request ID in model-specific data,
  and every encoder batch event is emitted per request with the batch request
  IDs in metadata, so encoder and scheduler events can be joined reliably;
- `encoder_encode_return` is emitted immediately after `encode_batch()`
  returns, before split, clone, host/device copy, synchronization, attachment,
  and cache work. Its absence therefore isolates a block inside the device
  encode call rather than the whole encoder batch lifecycle.

`SGLANG_OMNI_ENCODER_DIAG` is off by default. Setting it to `1`, `true`, `yes`,
or `on` only enables the call sites; the existing request-event recorder must
also be started through `/start_request_profile`, otherwise no JSONL is
written. Diagnostic metadata contains `CLOCK_MONOTONIC`, host boot ID, and
`monotonic_ns`; raw audio, transcripts, tensors, and internal input paths are
not recorded. `model_info.decode_cuda_graph` is always available and reports
the configured backend, runner/backend types, capture buckets, replay count,
standard eager count, and replay buckets without per-request logging.

Local Windows verification passed 10 encoder/diagnostic tests, Python
byte-compilation, focused Ruff fatal/import checks, and `git diff --check`.
The model-worker graph test and the Qwen3-ASR collection could not run locally:
the Windows interpreter lacks the Linux `resource` module and `torchaudio`.
They remain mandatory on the server's exact SGLang `71de97b2` environment.
No SGLang repository change is required by this diagnostic revision.

Run identifier: `910C-016`. Before any hardware action, the server must check
out the handoff commit containing this task, confirm that its sglang-omni code
contains local diagnostic commit `b64b16d6` (or a byte-equivalent transferred
commit), and return the actual sglang-omni and SGLang HEADs plus a clean-worktree
attestation. Do not reuse server-only `544b8cd9` as the implementation under
test. Preserve the repaired OpenCV environment, serving pyarrow 25.0.0,
isolated benchmark-client pyarrow 25.0.1 plus the declared
`openai-whisper==20250625`, and all `910C-013` workload and data-integrity
constraints.

Run the focused tests before service startup:

```bash
python -m pytest -q \
  tests/unit_test/profiler/test_encoder_diag_events.py \
  tests/unit_test/model_runner/test_prefill_cuda_graph_usage.py \
  tests/unit_test/scheduling/test_pre_lm_encoder.py
python -m pytest -q tests/unit_test/qwen3_asr
```

Stop on the first collection or test failure. If they pass, start one fresh
service with the exact graph-enabled `910C-013` concurrency-8 profile: encoder
graph disabled, prefill graph disabled, decode graph captured through bucket
70, torch compile disabled, eight asynchronous request-build workers,
`max_running_requests=70`, and decode log interval 1. The only diagnostic
addition is:

```bash
export SGLANG_OMNI_ENCODER_DIAG=1
```

After startup and before warm-up, start request-event recording without the
Torch profiler. Use a server-local evidence directory and verify the returned
run ID and directory:

```bash
curl -fsS -X POST http://127.0.0.1:8000/start_request_profile \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"910C-016","event_dir":"<server-local-evidence>/events"}'
```

Send frozen clip A once for first-use compilation and wait for all coordinator,
request-build, admission, and scheduler state to drain. Before the measured
wave, require at least one request-correlated sequence containing
`encoder_enqueue`, `encoder_batch_start`, `encoder_encode_return`, and
`encoder_batch_finish`; if it is absent, stop because the recorder contract is
not active. Record the pre-wave `model_info.decode_cuda_graph` snapshot.

Then run exactly the same pinned 70 content-distinct SeedTTS EN requests at
concurrency 8, without benchmark warm-up or retry, as `910C-013`. Poll at
bounded intervals using the same timestamp for coordinator state,
request-build/admission/backlog state, encoder counters, HBM/device health, and
the complete `model_info.decode_cuda_graph` object. Preserve the JSONL files
server-local. A 90-second no-completion interval is the expected diagnostic
stop condition, not permission to wait indefinitely; on timeout, take one
final snapshot, stop request profiling, and perform bounded forced cleanup.
Do not run another arm, scan concurrency, change graph settings, add stream
synchronization, edit runtime packages, or attempt a fix in `910C-016`.

Classify the first missing boundary for each outstanding request:

1. no `encoder_enqueue`: blocked before encoder submission;
2. enqueue without `encoder_batch_start`: encoder queue/worker dispatch;
3. batch start without `encoder_encode_return`: inside `encode_batch()` or its
   device execution;
4. encode return without batch finish: split/copy/synchronize/attach/cache;
5. batch finish without `scheduler_request_admit`: future completion,
   insertion-ordered build drain, or deferred admission;
6. scheduler admit without response: generation scheduling/replay or result
   completion.

For the same monotonic interval, report whether decode replay count advances,
which replay buckets advance, and whether standard eager count changes. Return
only the event-name/request-ID timing matrix, aggregate deltas, first missing
boundary, state counters, sanitized errors, cleanup result, and commit/version
mapping. Do not return raw JSONL, audio, transcript text, dataset paths, host
identity, or proprietary traces. This is a diagnostic run only: even if all 70
requests finish, do not report performance qualification and do not proceed to
realtime or higher concurrency without a new handoff task.

### `910C-016` preflight failure and `910C-017` retry

The first `910C-016` attempt stopped correctly at its focused-test gate before
service startup. On the exact SGLang `71de97b2` environment,
`test_model_worker_reports_actual_decode_graph_replays_by_bucket` observed zero
decode replay and eager counts. The local implementation had treated
`ForwardMode.is_cuda_graph()` as proof that a batch was already represented by
a graph wrapper and returned early. In SGLang, `ForwardMode.DECODE` itself
satisfies both `is_decode()` and `is_cuda_graph()`; the per-forward
`can_run_graph` result distinguishes replay from eager execution. The faulty
guard therefore excluded every decode forward.

Local fix `144316fe` removes the `is_cuda_graph()` exclusion and adds explicit
test assertions for the SGLang mode contract. Ten locally runnable
encoder/diagnostic tests, Python byte-compilation, focused Ruff fatal/import
checks, and `git diff --check` passed. The SGLang-dependent test remains a
mandatory server gate because the local Windows interpreter cannot import the
Linux-only SGLang dependency chain.

Run identifier: `910C-017`. This is a clean retry of the previously authorized
diagnostic task, not a continuation in the failed process. Before executing,
check out the handoff commit containing this section and confirm that
sglang-omni contains `144316fe` on top of `b64b16d6`, while SGLang remains at
`71de97b2`; require a clean worktree and the same serving/client dependency
split. Rerun all focused tests and the complete Qwen3-ASR unit-test directory
from the `910C-016` task. Stop again at the first failure.

Only after every test passes, start a fresh service and execute the unchanged
`910C-016` graph-enabled concurrency-8 cold-input diagnostic procedure under
the new run ID `910C-017`. Keep every profile, corpus, concurrency, timeout,
polling, request-recorder, cleanup, evidence, and prohibited-action requirement
unchanged. In the warm-up precheck, additionally require
`model_info.decode_cuda_graph.replay_count > 0` after observed decode graph
execution; zero or absent counts are a diagnostic-contract failure and must
stop the run before the measured SeedTTS wave.

### `910C-017` result and `910C-018` forward-boundary gate

The isolated-server `910C-017` retry passed all preflight tests and reproduced
the concurrency-8 cold-input hang. At the 90-second stop, eight requests were
outstanding: one had entered `encode_batch()` without an encode return, five
had been admitted without a first prefill start, and two had completed their
first prefill forward without reaching `model_path_end`. The warm-up decode
replay count was 15 and did not increase during the measured interval. Cleanup
returned the server to its clean resource baseline.

Do not interpret those observations as three independent blocking sites. The
five admitted requests may simply be queued behind a scheduler thread blocked
by an earlier model forward. `model_path_end` is a request-terminal event, so
the two post-prefill requests may be anywhere in their decode lifetime. Also,
the decode replay counter is updated only after
`model_runner.forward()` returns. A count fixed at 15 proves that no additional
decode graph forward returned successfully; it does **not** prove that no
measured request entered a replay which then blocked.

Local diagnostic commit `4c25482e` therefore adds one narrower boundary around
the standard `tp_worker.forward_batch_generation()` call. With the existing
`SGLANG_OMNI_ENCODER_DIAG` gate and request recorder active, every request in a
batch receives:

- `generation_forward_start`: call-local `forward_id`, `phase` (`prefill` or
  `decode`), `batch_size`, and the existing monotonic clock metadata;
- `generation_forward_return`: the same phase and batch size plus
  the same `forward_id`, `can_run_graph`, and `error_class`.

Normal serving remains unchanged because the event path is a no-op unless the
diagnostic environment gate and request recorder are both enabled. The new
events do not synchronize a stream, acquire a device lock, invoke graph
eligibility twice, or change scheduling order.

Run identifier: `910C-018`. Check out the handoff commit containing this task
and confirm that sglang-omni contains `4c25482e` on top of `144316fe`, while
SGLang remains exactly `71de97b2`. Require a clean worktree and the established
serving/client dependency split. Before service startup, run:

```bash
python -m pytest -q \
  tests/unit_test/profiler/test_encoder_diag_events.py \
  tests/unit_test/scheduling/test_pre_lm_encoder.py \
  tests/unit_test/model_runner/test_prefill_cuda_graph_usage.py \
  tests/unit_test/model_runner/test_base_hooks.py
python -m pytest -q tests/unit_test/qwen3_asr
```

Stop on the first collection or test failure. In particular, require the new
prefill/decode forward-boundary test to pass; do not patch it on the server.

Only after the tests pass, run one fresh service with the exact `910C-017`
profile, corpus, warm-up, request recorder, concurrency 8, no-retry policy,
polling, 90-second no-completion stop, and bounded cleanup. Keep
`SGLANG_OMNI_ENCODER_DIAG=1` and decode log interval 1. Before the measured
wave, require the warm request to contain paired
`generation_forward_start`/`generation_forward_return` events and a positive
decode replay count; otherwise stop as a diagnostic-contract failure.

For the measured wave, group generation events by `forward_id`, then report:

1. every unmatched `generation_forward_start`, including phase, batch size,
   and the redacted request-ID set;
2. the last successfully paired forward before the hang, including phase and
   returned `can_run_graph`;
3. encoder boundary state for each outstanding request;
4. scheduler admission/prefill/terminal state for each outstanding request;
5. decode replay/eager counter deltas, explicitly described as completed
   forward counts rather than graph-entry counts;
6. coordinator/build/admission/HBM state and cleanup result.

If an unmatched decode start is present, classify the first blocker as inside
the standard decode generation forward; a later task may then add begin/end
visibility at the exact SGLang graph-dispatch call. If an unmatched prefill
start is present, classify it inside the eager prefill generation forward. If
all starts have returns, classify the blocker after the device forward and use
the surrounding events to choose the next boundary. Do not infer a separate
scheduler defect merely from admitted requests that never receive prefill
while another forward is unmatched.

This remains a diagnostic run. Do not change graph settings, scan concurrency,
enable stream diagnostics, add synchronization or mutual exclusion, edit
runtime packages, attempt a fix, report performance qualification, or proceed
to realtime. Return the bounded redacted event matrix and aggregates, not raw
JSONL, audio, transcripts, private paths, host identity, or proprietary logs.

### `910C-018` result and `910C-019` SGLang graph-dispatch gate

The isolated-server `910C-018` run passed 25 focused tests and the complete
Qwen3-ASR unit-test directory (588 passed, 3 skipped), then reproduced the
concurrency-8 cold-input hang. Warm-up produced 16 paired standard generation
forward boundaries and 15 completed decode graph replays. During the measured
wave, one `forward_id` had `generation_forward_start` with `phase=decode` and
no matching return. Its same request had previously completed an eager prefill
forward. The completed decode replay count remained at the warm-up value.

This locates the first observable blocker inside
`tp_worker.forward_batch_generation()` for a standard decode call. It does not
yet establish whether the call blocked while evaluating graph eligibility,
loading static replay buffers, executing the backend graph, publishing a
shared-read fence, shaping graph output, or running the SGLang forward epilogue.
The outer replay counter still cannot distinguish those cases because it is
updated only after the complete call returns.

The failing boundary is owned by the SGLang repository. A separate local
SGLang worktree was created from the exact server dependency commit
`71de97b264b04dcd514cf904003028aefe9775c8` on branch
`codex/qwen3-asr-decode-graph-diag`. Diagnostic commit `f86279db9` extends the
existing `SGLANG_LOG_DECODE_GRAPH_KEY` switch with ordered stage markers:

```text
eligibility_begin
eligibility_return (includes can_run_graph)
execute_begin
runner_enter
replay_session_enter
load_batch_return (includes selected graph key)
backend_replay_begin
backend_replay_return
replay_session_return
execute_return
forward_raw_return
model_forward_return
```

The change only emits sanitized INFO records when the pre-existing switch is
enabled. It does not import sglang-omni, include request content, synchronize a
device stream, alter graph eligibility, or change graph execution order. The
outer sglang-omni `forward_id` remains the request-correlated boundary; on this
single-card, single-target-worker run, ordered SGLang records between its start
and the final snapshot identify the last completed inner stage.

Run identifier: `910C-019`. Check out the sglang-omni handoff commit containing
this task and require a clean worktree. In the SGLang checkout, use exactly
`f86279db9` (parent `71de97b2`) and require a clean worktree; do not reproduce
the patch in site-packages. Preserve the established editable-install mapping,
serving/client dependency split, hardware/runtime stack, and all `910C-018`
profile and corpus settings.

Before startup, run the SGLang-owned focused test followed by the unchanged
sglang-omni gates:

```bash
python -m pytest -q \
  test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py

python -m pytest -q \
  tests/unit_test/profiler/test_encoder_diag_events.py \
  tests/unit_test/scheduling/test_pre_lm_encoder.py \
  tests/unit_test/model_runner/test_prefill_cuda_graph_usage.py \
  tests/unit_test/model_runner/test_base_hooks.py
python -m pytest -q tests/unit_test/qwen3_asr
```

Run each command from its owning repository. Stop on the first collection or
test failure and do not edit either checkout on the server.

Only after all tests pass, start one fresh service with the exact `910C-018`
configuration plus the single diagnostic variable:

```bash
export SGLANG_LOG_DECODE_GRAPH_KEY=1
```

Keep `SGLANG_OMNI_ENCODER_DIAG=1`, request-event recording, decode log interval
1, warm clip, pinned 70-input SeedTTS EN wave at concurrency 8, no retry,
polling, 90-second no-completion stop, final snapshot, and bounded cleanup
unchanged. Before the measured wave, require one warm decode call to show the
complete ordered SGLang stage sequence through `model_forward_return`, a paired
outer generation forward, and a positive completed replay count. Otherwise
stop as a diagnostic-contract failure.

For the first unmatched outer decode `forward_id`, report the last observed
ordered SGLang stage and classify it as follows:

- no `eligibility_return`: graph eligibility;
- eligibility returned `can_run_graph=False`: eager decode path, not replay;
- `execute_begin` without `runner_enter`: graph runner call boundary;
- `runner_enter` without `replay_session_enter`: replay context preparation;
- `replay_session_enter` without `load_batch_return`: static replay input load;
- `load_batch_return` without `backend_replay_begin`: pre-replay shared-read
  publication;
- `backend_replay_begin` without `backend_replay_return`: backend graph replay;
- `backend_replay_return` without `replay_session_return`: shared-read
  publication or replay-session exit;
- `replay_session_return` without `execute_return`: graph output shaping;
- `execute_return` without `forward_raw_return`: `_forward_raw()` return path;
- `forward_raw_return` without `model_forward_return`: SGLang forward epilogue;
- `model_forward_return` without outer `generation_forward_return`: the
  SGLang worker/wrapper path after `ModelRunner.forward()`.

Return the redacted unmatched `forward_id`, phase, batch size, ordered stage
names, relative timestamps, `can_run_graph`, graph key size, completed
replay/eager deltas, coarse encoder/scheduler states, health/HBM, cleanup, and
both repository commit mappings. Do not return raw logs, request content,
audio, transcripts, dataset or host paths, host identity, or proprietary
traces.

This is still diagnostic-only. Do not change graph settings or concurrency,
add stream synchronization or a mutex, enable vendor profilers, modify runtime
packages, attempt a fix, claim performance qualification, or begin realtime.
The next code change must be selected from the first missing inner stage found
by this run.

### `910C-019` result and `910C-020` NPU update/replay gate

The isolated-server `910C-019` run passed the SGLang-owned test (13 passed),
the sglang-omni focused set (25 passed), and the complete Qwen3-ASR unit-test
directory (588 passed, 3 skipped). It reproduced the concurrency-8 cold-input
hang. Of 63 decode dispatches, 62 reached `execute_return`; the final batch-size
2 dispatch emitted `execute_begin` but no return, while the completed replay
counter remained at its warm-up value 15.

The absence of the generic runner markers is now explained by code ownership,
not logging loss: Ascend selects `NPUGraphRunner`, whose NPU-specific
`execute()` overrides `DecodeCudaGraphRunner.execute()`. The generic
`ModelRunner` markers executed, while the overridden implementation bypassed
the instrumented generic body. Therefore `910C-019` proves that the standard
decode call entered the selected NPU graph runner and did not return, but it
does not yet isolate the NPU implementation's device boundary.

Code inspection of the exact SGLang dependency identifies the remaining
ordered operations:

1. load/copy the selected static graph inputs;
2. copy `seq_lens` from NPU to the host;
3. start a background thread which calls `graph.update(...)` after binding the
   NPU device;
4. call `graph.replay()` concurrently on the main thread;
5. join the update thread and return the captured outputs.

The update/replay overlap is a strong mechanism candidate because the hang
also requires cold encoder device work and asynchronous request building in
the established matrix. It is not yet a confirmed defect: torch_npu's graph
dispatch implementation uses a dedicated update stream plus event ordering,
so serializing update and replay without observing both lanes could violate
the intended API contract.

SGLang diagnostic commit `9dbc4f89c`, on top of `f86279db9` and exact parent
`71de97b2`, adds NPU-specific markers under the same existing
`SGLANG_LOG_DECODE_GRAPH_KEY` gate. The markers cover:

```text
npu_execute_begin
load_batch_begin / load_batch_return
input_copy_begin / input_copy_return
seq_lens_host_begin / seq_lens_host_return
input_update_replay_begin / input_update_replay_return
backend_enter / cpu_update_input_ready
update_thread_start_begin / update_thread_start_return
update_thread_enter / update_device_set
graph_update_begin / graph_update_return
graph_replay_begin / graph_replay_return
update_thread_join_begin / update_thread_join_return
```

The diagnostic change emits only stage names and graph/batch sizes. It does not
alter stream selection, thread ordering, graph inputs, synchronization, or
request data.

`910C-019` also exposed a repeatable environment hazard. Installing the new
SGLang editable checkout with dependency resolution reintroduced
`opencv-python` 4.10.0.84, overwrote the headless `cv2` namespace, and restored
the already closed `libGL.so.1` -> Manager EOF -> GE initialization failure.
The operator restored the established single headless OpenCV distribution and
the subsequent warm request passed. Because that repair happened before the
accepted measured run, the graph diagnosis remains usable, but every future
checkout/editable-install transition must verify the OpenCV invariant before
tests or service startup. Do not reinstall the already mapped editable SGLang
checkout with dependency resolution merely to change its Git commit.

Run identifier: `910C-020`. Check out the sglang-omni handoff commit containing
this task and SGLang `9dbc4f89c`; require both worktrees clean. First verify:

- `opencv-python` is not installed;
- exactly one intended `opencv-python-headless` distribution owns `cv2`;
- a fresh process imports `cv2` successfully;
- the loaded `cv2` extension has zero `libGL.so.1` dependencies;
- editable SGLang resolves to the checkout at `9dbc4f89c` without another
  dependency-resolving install.

If any invariant fails, stop and report an environment-preflight failure. Do
not mutate packages and continue under the same run ID. After a separately
authorized repair, restart the task from a fresh process and new run ID.

If the environment passes, rerun the exact SGLang, sglang-omni focused, and
complete Qwen3-ASR test commands from `910C-019`. Stop on the first collection
or test failure. Then start one fresh service and repeat the exact `910C-019`
warm-up and measured concurrency-8 cold-input diagnostic with both
`SGLANG_OMNI_ENCODER_DIAG=1` and `SGLANG_LOG_DECODE_GRAPH_KEY=1`. Preserve all
graph, compile, encoder, request-builder, corpus, timeout, polling, no-retry,
evidence, and cleanup settings. No other variable is permitted.

Before the measured wave, require one warm decode to show the complete NPU
main-thread and update-thread marker sets through
`update_thread_join_return`, followed by the outer paired generation return
and a positive completed replay counter. Missing markers are a diagnostic
contract failure.

At the first 90-second no-completion interval, report the unmatched outer
`forward_id` and the last marker independently for the NPU main and update
lanes. Classify the first missing boundary:

- `load_batch_begin` without return: static graph input load;
- `seq_lens_host_begin` without return: NPU-to-host sequence-length copy or
  synchronization;
- `update_thread_start_begin` without return: Python update-thread startup;
- update thread entered but did not reach `update_device_set`: NPU device bind;
- `graph_update_begin` without return: NPUGraph input update;
- `graph_replay_begin` without return: NPUGraph replay;
- replay returned, then join began without return: update thread still blocked;
- both graph operations returned but `input_update_replay_return` is absent:
  backend post-operation/return path;
- NPU runner returned but the outer SGLang markers did not: retain the
  corresponding `910C-019` outer-stage classification.

If both `graph_update_begin` and `graph_replay_begin` lack returns, report both
lanes as jointly outstanding; do not choose one as causal from log order. If
all 70 requests unexpectedly finish, classify the issue as timing-sensitive
under diagnostic logging and stop; do not claim stability or performance.

Return only commit mappings, test totals, OpenCV invariant results, the bounded
stage/timestamp matrix, batch/key sizes, completed replay/eager deltas,
coarse encoder/scheduler state, health/HBM, and cleanup. Keep raw logs, audio,
transcripts, dataset/host paths, host identity, and proprietary traces on the
server. Do not change graph settings or concurrency, add synchronization or a
mutex, run vendor profilers, modify runtime packages, attempt a fix, claim
performance qualification, or begin realtime.

### `910C-020` result and `910C-021` mutual-exclusion treatment

The isolated-server `910C-020` run passed the SGLang diagnostic tests (13),
the sglang-omni focused tests (25), and the complete Qwen3-ASR unit-test
directory (588 passed, 3 skipped). The warm request completed all NPU graph
markers and incremented the completed decode-replay counter to 15. The same
70-input SeedTTS EN workload at concurrency 8 then reproduced the hang.

The final decode forward selected raw batch size 6 and graph bucket 8. Its
static input load and NPU-to-host `seq_lens` copy returned. The update thread
entered `graph.update()`, while the main thread called `graph.replay()` and the
host replay call returned. The main thread then waited in the update-thread
join for almost five minutes. Marker totals were:

```text
graph_update:            begin 34, return 33
graph_replay:            begin 34, return 34
update_thread_join:      begin 34, return 33
```

This proves the Python update call did not return. It does **not** prove that
all replay device work completed independently: the replay host API may return
after enqueue while device-side event ordering still depends on the update
lane. Therefore the repository must not reorder or serialize torch_npu's
internal update/replay protocol based only on this observation.

Local implementation commit `29ca236f` instead addresses the already proven
cross-thread overlap at the integration boundary. It constructs one shared
FIFO device-execution guard only when all of the following are true:

- the Qwen3-ASR pre-LM encoder is enabled;
- the model device is NPU;
- generation graph execution is enabled.

The encoder holds the guard for its complete batch execution, including the
blocking CPU cache copy that establishes completion of its NPU work. The model
runner holds the same guard around each standard prefill or decode generation
forward, through the graph update-thread join. Request building remains at
eight workers and decode graph remains enabled. CPU, CUDA, graph-disabled, and
pre-LM-encoder-disabled paths receive no guard. FIFO ticket ordering prevents a
tight decode loop from indefinitely overtaking an encoder batch that is
already waiting. Under `SGLANG_OMNI_ENCODER_DIAG=1`, wait, acquired, and
released events identify the owner, FIFO ticket, phase or encoder batch size,
wait time, and hold time.

Local Windows validation ran `git diff --check`, import/order checks, and the
standalone FIFO concurrency test (1 passed). Tests importing SGLang could not
be collected because SGLang is not installed in the local Windows interpreter;
they are mandatory server preconditions below, not claimed local passes.

Run identifier: `910C-021`. This is a single treatment run against the failed
`910C-020` control. Check out the sglang-omni handoff commit containing this
task and verify that it contains implementation parent `29ca236f`. Keep the
exact SGLang diagnostic checkout `9dbc4f89c` and the same editable mapping used
by `910C-020`. Both tracked worktrees must be clean; an install-generated
tracked edit is still a dirty worktree and is not exempted as an environment
side effect.

Before running tests, repeat the `910C-020` environment preflight: no
`opencv-python`, exactly one intended `opencv-python-headless` owner, fresh-
process `cv2` import success, zero `libGL.so.1` dependencies, serving pyarrow
25.0.0, isolated benchmark-client pyarrow 25.0.1 plus the declared
`openai-whisper`, idle device/HBM, no server worker or orphan, and a free port.
Stop on any discrepancy. Do not repair an environment and continue under the
same run ID.

Run these gates from their owning repositories, stopping at the first
collection or test failure:

```bash
# SGLang checkout at 9dbc4f89c
python -m pytest -q \
  test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py

# sglang-omni checkout at the 910C-021 handoff commit
python -m pytest -q \
  tests/unit_test/utils/test_execution_guard.py \
  tests/unit_test/model_runner/test_base_hooks.py \
  tests/unit_test/profiler/test_encoder_diag_events.py \
  tests/unit_test/scheduling/test_pre_lm_encoder.py \
  tests/unit_test/model_runner/test_prefill_cuda_graph_usage.py
python -m pytest -q tests/unit_test/qwen3_asr
```

Only after all tests pass, start a fresh service with the exact accepted
`910C-020` configuration and environment. In particular retain:

```text
enable_encoder_cuda_graph=false
disable_prefill_cuda_graph=true
disable_cuda_graph=false
enable_torch_compile=false
cuda_graph_max_bs=70
max_running_requests=70
request_build_max_workers=8
decode_log_interval=1
SGLANG_OMNI_ENCODER_DIAG=1
SGLANG_LOG_DECODE_GRAPH_KEY=1
```

Do not introduce a guard enable/disable environment flag: presence of local
commit `29ca236f` is the sole treatment variable relative to `910C-020`.
Require decode graph capture buckets to end at 70, zero prefill/encoder graph
capture attempts, and no unexpected eager fallback. Run the same single warm
clip A and wait for full drain. The warm precheck must show a successful
decode replay plus complete generation and guard event pairs.

Then run exactly the same pinned, content-distinct 70-input SeedTTS EN set at
closed-loop concurrency 8, with no benchmark warm-up, retry, input
substitution, graph/configuration change, or service reuse from another run.
Poll the same coordinator, scheduler, encoder-cache, decode-graph, health, HBM,
and diagnostic counters. Preserve raw artifacts server-locally.

The treatment passes only if all of the following hold:

- all 70 measured requests return HTTP 200 with non-empty valid output and the
  repository SeedTTS accuracy gate passes;
- the workload preflight and cache deltas prove the measured inputs remained
  cold and content-distinct; any unexplained hit/merge delta invalidates the
  workload rather than being waived after the run;
- decode graph completed-replay count increases during the measured wave,
  `npu graph: True` is observed, and graph fallback/error counts remain zero;
- every acquired guard interval has one release, encoder and generation
  acquired intervals do not overlap, and at least one encoder acquisition
  after generation contention proves the FIFO path made progress;
- request-build/admission queues and all request states drain, HBM stays
  bounded, the device remains healthy, and shutdown leaves no process, port,
  or device-memory residue.

Stop at the first test failure, startup/capture discrepancy, forbidden error,
wrong or empty result, timeout, 90-second no-completion interval, guard-event
imbalance/overlap, OOM, device reset, fallback, state leak, or orphan. On a
hang, return the last guard owner/ticket ordering and the last NPU graph marker
for the unmatched generation forward; do not add another synchronization,
stream, timeout, worker-count, or graph change on the server.

This is a functional stability qualification of the fix, not a performance or
realtime result. Record the preliminary latency/throughput and guard wait/hold
distributions for regression triage, but do not compare them with the 500 ms
target. If `910C-021` passes, the next handoff will restore the bounded
concurrency ladder under this same guarded graph profile before the exact-10 s
performance task. If it fails, the first unmatched guard or graph boundary
selects the next local change.

Return only the two repository commits, package/test totals, environment
invariants, aggregate dataset identity, request/accuracy/cache results,
completed graph replay and fallback deltas, guard event counts and aggregate
wait/hold statistics, scheduler/health/HBM/cleanup state, and the first
sanitized failure boundary. Do not return raw audio, transcripts, request IDs,
host or dataset paths, host identity, full logs, or proprietary traces.

### `910C-021` result and `910C-022` guard-boundary diagnostic

The `910C-021` concurrency-8 functional workload passed: all 70 cold,
content-distinct requests completed, WER was 0.77%, decode graph replay count
was 211 with zero standard-eager decode, and state drained. This is strong
evidence that local guard implementation `29ca236f` removes the previously
repeatable concurrency-8 failure under the tested profile. The run is not a
complete gate pass because its required guard events were absent. Its
preliminary p95 was 0.61 s, already above the 0.50 s hard target, but this
cold-input, diagnostic-logging run is not the exact-10-second performance gate.

The operator then extended the ladder before a new handoff. That exploration
completed all 70 requests at concurrency 16 (WER 0.77%, decode replay 118,
zero eager fallback), but p95 rose to 3.58 s. A fresh concurrency-32 process
timed out after ten minutes with 64 coordinator-pending operations and only
two measured completions. Preserve these as useful exploratory evidence, not
as completion of a predeclared capacity gate.

Do not yet classify the concurrency-32 failure as a ticket-lock convoy,
reader/writer starvation, or `notify_all` scaling defect. The execution guard
has only two device-submitting host threads in this stage: the single encoder
worker and the generation scheduler. Request concurrency does not create 32
encoder workers or 32 generation schedulers. Moreover, the required guard
events were absent from the returned JSONL evidence. That violated the
`910C-021` diagnostic contract and leaves three materially different boundaries
unseparated:

1. a thread waiting to acquire the FIFO guard;
2. the current holder blocked inside encoder or generation device execution;
3. scheduler/coordinator work outside the guard.

The server also applied an unreported compatibility edit so legacy/mock engine
builders without `_device_execution_guard` would not fail tests. Local commit
`d9df3a74` is the reviewed equivalent: `make_model_runner()` now uses
`getattr(..., None)` and adds a regression test. The next server checkout must
use the handoff commit containing `d9df3a74`; do not carry an uncommitted or
unmapped server edit.

Run identifier: `910C-022`. This is diagnostic-only and runs concurrency 32
once. It does not repeat 8 or 16, change guard policy, or resume the capacity
ladder. Check out the sglang-omni handoff commit containing this task and local
parents `29ca236f` and `d9df3a74`. Keep SGLang at diagnostic commit
`9dbc4f89c`. Require both tracked worktrees clean and repeat the full OpenCV,
pyarrow/client, process/port, device/HBM, model, dataset, and editable-checkout
preflight from `910C-021`.

Run the same SGLang test plus the sglang-omni focused and full Qwen3-ASR tests
listed in `910C-021`. Stop on the first collection or test failure. Then start
one fresh service with the exact accepted guarded profile: encoder graph and
prefill graph disabled, decode graph enabled through bucket 70, torch compile
disabled, eight request-build workers, maximum running requests 70, decode log
interval 1, and both diagnostic environment variables enabled. No runtime,
package, model, cache, batching, admission, graph, worker, timeout, or stream
change is permitted.

Before any measured request, start the request-event recorder through the same
profiling control plane used in `910C-020`. Send warm clip A, drain state, stop
and inspect the file if necessary, and require all of these positive markers:

```text
generation_forward_start / generation_forward_return
npu_execution_guard_wait
npu_execution_guard_acquired (owner, ticket, phase/batch_size, wait_ms)
npu_execution_guard_released (same owner and ticket, held_ms)
graph_update_begin / graph_update_return
graph_replay_begin / graph_replay_return
update_thread_join_begin / update_thread_join_return
```

At least one complete encoder guard interval and one complete generation guard
interval must be present. If the environment variable is set but any guard
event class is absent, stop as a diagnostic-contract failure. Do not proceed
on the strength of graph counters alone.

After the warm precheck passes, run the identical pinned 70-input SeedTTS EN
set once at closed-loop concurrency 32, with no benchmark warm-up and no retry.
Use the same client and cache-integrity checks. Poll as before, but stop after
the first 90-second interval with no measured completion; do not wait ten
minutes. Stop the recorder before bounded cleanup so its line-buffered JSONL
files are closed.

Join events by owner, request/batch, phase, and FIFO ticket. Report exactly one
of these first boundaries:

- an acquired ticket without release: identify whether the holder is encoder,
  prefill, or decode, then report the last inner encoder/generation/NPU marker;
- guard wait events after the last released ticket with no later acquisition:
  acquisition/FIFO progression is blocked; report the last acquired and
  released ticket and owner, without calling it starvation unless ticket order
  proves bypass;
- all acquired tickets are released but requests remain pending: the first
  blocker is outside the guard; use the existing request-build, admission,
  prefill, model-path, and generation-forward boundaries;
- all 70 requests complete: classify the earlier concurrency-32 failure as
  timing-sensitive or invalidated by missing diagnostics, record functional
  results, and stop without running a higher level.

For every acquired ticket, verify that the next acquired ticket is strictly
the next integer and that no acquired intervals overlap. `notify_all` versus
`notify(1)`, a timeout, maximum hold duration, a reader/writer lock, guard
removal, an extra stream, or synchronization is explicitly outside this run.
Those changes require the first complete `910C-022` boundary and a new local
reviewed commit.

Return only repository commits, environment/test totals, aggregate corpus and
accuracy results, completion and timeout counts, guard events grouped by
ticket/owner with relative timestamps and aggregate wait/hold distributions,
the last matched encoder/generation/NPU boundary, graph replay/fallback
deltas, scheduler/health/HBM/cleanup state, and any protocol difference. Keep
raw request IDs, paths, audio, transcripts, logs, and proprietary traces on the
isolated server. This run cannot qualify performance or realtime.

### `910C-022` and `910C-023` guarded capacity result

The isolated-server `910C-022` run used sglang-omni `81177bea`, including
guard `29ca236f` and compatibility fix `d9df3a74`, with SGLang diagnostic
commit `9dbc4f89c`. The required warm marker families were present. The single
concurrency-32 cold-input wave completed without a guard or graph hang:

- the server reported all 70 request completions, normal drain and shutdown;
- decode graph recorded 68 completed replays, zero standard-eager decode, and
  27 bucket-32 replays;
- guard events contained 1,303 waits, 1,303 acquisitions, and 1,303 releases,
  with no reported ticket gap, overlap, or unmatched holder;
- scored WER was 0.77%, wall time 5.64 s, preliminary p95 3.18 s, and RTFx
  57.3.

This clean rerun disproves the hypothesis that `FairDeviceExecutionGuard`
intrinsically deadlocks at concurrency 32. The earlier failed process was
reported to have residual NPU-side activity plus `SetDevice` 507033 and stage
death. Record that older run as environmentally contaminated. Do not generalize
the evidence into a rule that an `hdc` or `tsd` daemon should be killed: future
tasks must stop when unexpected target-device compute PIDs, non-baseline HBM,
or a device-health error is present, and recovery must use the operator's
approved device/runtime procedure under a new run ID.

The isolated-server `910C-023` run then completed fresh concurrency-64 and
concurrency-70 processes against the same guarded candidate. Both reported
70/70 evaluated, WER 0.77%, zero standard-eager decode, balanced guard event
triplets, state drain, and no graph fallback, OOM, or forbidden error:

| Concurrency | Wall | Preliminary p95 | RTFx | Decode replay | Largest observed bucket |
|---:|---:|---:|---:|---:|---:|
| 64 | 5.06 s | 4.94 s | 63.9 | 51 | 64, 12 replays |
| 70 | 4.58 s | 4.49 s | 70.6 | 98 | 70, 11 replays |

The concurrency-70 run observed all 13 configured decode buckets
`[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 70]`; references to 25 buckets
are a reporting typo. Balanced wait/acquire/release events establish forward
progress, FIFO ownership, and the absence of unmatched guard holders in these
runs. They do not establish that lock contention was absent.

The historical `910C-022` `evaluated=65/70` denominator remains an unexplained
artifact of that run. It must not be silently rewritten, but it does not
invalidate the two new `910C-023` 70/70 measurements. The A3 explicit profile
is now functionally qualified through target concurrency with an actual bucket-
70 replay. This result does not qualify the compile-enabled repository default,
the encoder or prefill graph paths, the original 910B device class, hard-target
performance, or realtime.

The `910C-023` p95 of 4.49 s is approximately nine times the 0.50 s target, but
SeedTTS clips are not the frozen exact-10-second manifest and the run retained
diagnostic settings. Treat that ratio only as a direction-of-travel signal,
not as the hard-gate gap.

### Exact-10-second performance harness: local implementation complete

Local commits `37f598f3` and `63f235fa` implement and unit-test the NPU-aware
exact-manifest
harness required by the
[performance task](qwen3_asr_ascend_910b_performance_task.md). It provides the
strict JSONL/RIFF manifest loader, full content fingerprint, disjoint warm-up
and measured partitions, all-outcome request accounting, post-upload latency
timestamp, server-local raw JSONL, fail-closed `npu-smi` monitoring, NPU
environment fingerprint, and one-repeat-per-service hard-gate enforcement.

The focused local suites pass 43 tests: 18 manifest, 10 NPU-monitor, and 15
benchmark orchestration/accounting tests. The complete benchmark directory was
also attempted but could not collect in the current Windows interpreter because
pre-existing optional `torchaudio` and `scipy` dependencies are absent. This is
recorded as unexecuted coverage, not a pass. `python -m compileall` and
`git diff --check` pass for the new files. The local environment does not
provide `ruff`, Black, or isort, so those commands were not run.

Local commits `2cb63b9e` and `8d46ddec` add the deterministic,
revision-pinned SeedTTS corpus transform and seven tests. It creates 70
disjoint warm-up plus 700 measured
clips by concatenating only complete source utterances, inserting a fixed
100 ms silence, and padding the tail; it never crops speech and requires at
least 80% speech occupancy. The pinned snapshot was subsequently observed to
contain 1088 mono PCM16 sources at 24 kHz. Commit `8d46ddec` therefore adds a
single generator-owned ffmpeg/swresample conversion to 16 kHz, with fixed
filter arguments, no dithering, no fallback backend, and complete backend
identity in provenance. Together with the 43 harness tests, the locally
maintained exact10 toolchain now has 50 focused passes.

### Next isolated task: `910C-024A` harness and corpus qualification

The first authorization at `441b6db4` stopped correctly during corpus
preflight because the source-rate contract rejected all 1088 pinned 24 kHz
WAVs. It is superseded and must not be resumed from that commit. This section
is the only newly authorized server task. The isolated operator must check out
the handoff commit containing local parents `37f598f3`, `2cb63b9e`,
`63f235fa`, and `8d46ddec`, keep both repositories clean, and keep SGLang at
the previously
qualified dependency commit unless this handoff names a replacement. The
server may execute these files; it may not edit them, patch installed packages,
hand-select or manually preprocess data, install ffmpeg, or continue to a
performance ladder.

Before running code, repeat the established process/port/NPU-HBM health,
OpenCV-headless/libGL, serving-pyarrow 25.0.0, benchmark-client pyarrow 25.0.1,
`openai-whisper`, model, pinned English Parquet snapshot, editable checkout,
and exact repository-HEAD preflight. Stop and report any material difference.
Additionally require `command -v ffmpeg` and `ffmpeg -version` to succeed.
Capture the resolved executable, the first version line, and the SHA-256 of the
complete version output. Do not install, replace, or relink ffmpeg on the
server. Stop if it is absent or if its fixed `aresample` invocation is not
supported.
Run these local tests from the benchmark-client environment:

```bash
python -m pytest -q \
  tests/unit_test/benchmarks/test_exact10s_manifest.py \
  tests/unit_test/benchmarks/test_prepare_seedtts_exact10s.py \
  tests/unit_test/benchmarks/test_npu_monitor.py \
  tests/unit_test/benchmarks/test_benchmark_asr_exact10s.py
```

Require exactly 50 passes. Stop at the first collection or test failure. Then
run the corpus command from the performance task against the already approved
pinned local snapshot into a new server-local `910C-024A` evidence directory.
Require exactly 770 manifest rows, 70 warm-up/700 measured in provenance,
770 distinct PCM hashes, duration min/max within `10.000 +/- 1/16000` seconds,
the pinned dataset revision, `source_sample_rate_counts={"24000": 1088}` as
observed, `resampled_source_count=1088`, resampler backend
`ffmpeg-swresample`, a 64-character ffmpeg version-output SHA-256, and no
source-format, resampling, occupancy, duplicate, or overwrite failure. Every
derived WAV must be PCM16 mono 16 kHz. Preserve manifest, audio, source
membership, transcripts, paths, and full resampler provenance on the server;
return only counts, duration range, exclusions count, source-rate counts,
resampled count, ffmpeg first version line/version-output SHA-256,
source-Parquet-set SHA-256, and derived manifest SHA-256.

Start one fresh service with the exact accepted `910C-023` guarded
compatibility profile: encoder graph and prefill graph disabled, decode graph
enabled through bucket 70, torch compile disabled, eight request-build workers,
maximum running requests 70, the accepted device-execution guard, and no
diagnostic logging unless already required for positive graph evidence. Do not
change model, precision, memory fraction, cache, batching, admission, graph,
worker, stream, or timeout settings. Verify startup, decode capture through
bucket 70, zero unexpected fallback/errors, and health before requests.

Run two harness smokes in order, each once and without retry, writing separate
result/raw directories:

```bash
python -m benchmarks.eval.benchmark_asr_exact10s \
  --meta "${EXACT10_ROOT}/manifest.jsonl" \
  --host 127.0.0.1 --port "${QWEN3_ASR_PORT}" \
  --model Qwen/Qwen3-ASR-1.7B --lang en \
  --concurrencies 1 --repeats 1 \
  --warmup-samples 1 --max-samples 1 --min-distinct-audio 770 \
  --npu-id 0 --npu-chip-id 0 --monitor-interval-s 1 \
  --request-timeout-s 120 --launch-command "${DECLARED_SERVER_LAUNCH}" \
  --output "${EVIDENCE}/batch1/result.json" \
  --save-raw-dir "${EVIDENCE}/batch1/raw"

python -m benchmarks.eval.benchmark_asr_exact10s \
  --meta "${EXACT10_ROOT}/manifest.jsonl" \
  --host 127.0.0.1 --port "${QWEN3_ASR_PORT}" \
  --model Qwen/Qwen3-ASR-1.7B --lang en \
  --concurrencies 2 --repeats 1 \
  --warmup-samples 2 --max-samples 2 --min-distinct-audio 770 \
  --npu-id 0 --npu-chip-id 0 --monitor-interval-s 1 \
  --request-timeout-s 120 --launch-command "${DECLARED_SERVER_LAUNCH}" \
  --output "${EVIDENCE}/conc2/result.json" \
  --save-raw-dir "${EVIDENCE}/conc2/raw"
```

Both results must be valid with all requests present, no failure, timeout,
empty response, missing/duplicate/unexpected result, or unscoreable output.
Require full 64-character manifest hashes; raw record counts exactly one and
two; all expected latency fields including p90/p95/p99; NPU monitor
`available=true`, `error=null`, HBM samples, and AI Core or NPU utilization;
service graph replay evidence; zero unexpected eager fallback; final request,
scheduler, service, device, HBM, process, and port cleanup.

Stop on the first discrepancy and return the first complete sanitized failure.
Even if both smokes pass, stop after cleanup. Do not run 100 sequential, 700 at
concurrency 70, soak, fresh-process repetitions, acceleration experiments, or
realtime. `910C-024B` requires review of this evidence and a new committed
handoff update.

After `910C-024A` passes, a separately committed handoff may authorize
`910C-024B`: 100 sequential requests and one 700-request concurrency-70
baseline against the explicit qualified profile. That run exists to identify
the dominant stage and freeze a before-state; it is not the final performance
candidate and cannot close the hard target.

The project requires every currently failing acceleration path to be repaired;
disabling it is not an acceptable close condition. Qualify these changes
separately and then in combination:

1. re-enable prefill graph with the execution guard present and verify whether
   the guard fixes its former cold-input overlap failure;
2. replace or repair the incompatible NPU encoder-graph capture path so host-
   device copies and synchronization do not occur illegally inside capture;
3. repair the compile-enabled generation path across the SGLang/triton-ascend
   boundary and requalify every compiled and non-compiled decode bucket;
4. profile the coarse encoder/generation execution guard and narrow or replace
   its critical section if serialization limits throughput;
5. run the fully combined profile with compile, encoder graph, prefill graph,
   and decode graph enabled, positive execution markers, zero unexpected eager
   fallback, and the exact-10-second gate.

Repair qualification and final performance selection are separate decisions.
Each path above must first become correct, stable, and observable. Only after
that may a controlled A/B show whether an implementation should be replaced or
tuned; a negative performance result does not waive the compatibility defect.
The current evidence is far from supporting a disabled-feature final
candidate: 70 SeedTTS requests in 4.58 s is about 15.3 requests/s versus the
hard gate's derived 140 requests/s, while preliminary p95 is 4.49 s versus
0.50 s. Because the corpora and duration distributions differ, these are only
gap indicators, not a hard-gate comparison.

Do not start realtime in parallel. First establish the exact-10-second offline
baseline, repair and combine the viable acceleration paths, and determine
whether the single-card target is feasible. Realtime remains a separate
protocol and implementation gate after the offline path is trustworthy.

#### Isolated-server harness draft: not yet accepted locally

The isolated operator reported a server-only draft containing an exact-10-
second manifest loader, an `npu-smi` resource monitor, a benchmark entry point,
and 26 focused tests, plus 147 benchmark tests passing with 7 skipped. None of
the reported source or tests exists in the local reviewable checkout at
`0c5a311a`; the local worktree is clean. Under the collaboration contract,
those counts are a server report, not proof of a locally maintained
implementation. Do not commit the server draft as the project source of truth
or run `910C-024` from it.

A follow-up server report claimed that the acceptance items were implemented
and changed the local rule to allow an isolated-server developer to generate
and modify code. That rule change is rejected: it directly conflicts with the
project owner's hard constraint above. At local HEAD `c39dc5e9`, none of the
reported exact-10-second source or test files exists and the worktree is clean.
Reported counts of 147 benchmark tests, 19 manifest tests, and 9 NPU-monitor
tests therefore remain unverified server-local observations. The isolated
operator must not commit those changes or continue developing them. It may
return sanitized design points and failure evidence; the local owner will
rebuild the implementation in the reviewable repository.

The local equivalent must resolve and test these acceptance details before a
hardware task is issued:

- validate RIFF/WAVE structure, PCM encoding, mono channel count, 16 kHz sample
  rate, 16-bit sample width, and effective frame count; a raw `data_size`
  calculation alone is insufficient;
- require at least 700 distinct measured audio-content hashes per hard-gate
  repeat, plus a disjoint warm-up set, so neither repeated inputs nor warm-up
  cache hits can satisfy the timed workload;
- retain every request outcome. A timeout or failed request must invalidate the
  repeat and remain represented in machine-readable request records and the
  aggregate latency/error accounting; adding only failure counters while
  calculating percentiles from successful `SampleOutput` objects is not
  sufficient;
- keep one benchmark repeat within one service lifetime, but orchestrate the
  three final repeats with three fresh server processes. An in-client
  `--repeats 3` loop cannot attest the fresh-process requirement;
- test the exact `npu-smi` command/output variants on the frozen server stack,
  identify the selected physical device, record command/parser failures, and
  make missing required HBM/utilization evidence fail the performance run
  rather than silently degrading to an available-looking result;
- document the manifest schema, full aggregate SHA-256, duration/language/count
  summary, raw JSONL schema, timeout treatment, warm-up partition, and server-
  local artifact layout without returning paths, transcripts, or audio.

Once the implementation and tests exist in the local repository, review and
commit them separately from the executable `910C-024` handoff. The server must
then check out that exact commit, rerun the declared focused and benchmark
tests, execute only a manifest preflight and batch-one/two harness smoke first,
and stop on the first schema, duration, uniqueness, monitor, or request-
accounting discrepancy. The full ladder, soak, and three fresh-process repeats
require later gates; they are not authorized by the initial harness smoke.

The exact-10-second corpus is also locally specified, not improvised by the
operator. The local implementation must pin the upstream dataset revision and
a deterministic transform that produces at least 700 distinct, representative
10-second measured clips plus a disjoint warm-up partition, with transcript
and content fingerprints. The server may execute that committed preparation
and validation procedure against its approved, pre-staged source snapshot;
it may not hand-select files, substitute private data, or invent a manifest.

The next performance phase is still a baseline phase, but it is deliberately
bounded. After `910C-024A` qualifies the locally committed harness with
manifest preflight, batch one, two concurrent requests, failure accounting,
and NPU monitoring, `910C-024B` may measure one exact-10-second diagnostic
baseline on the qualified compatibility profile. It should collect 100
sequential requests and one 700-request concurrency-70 repeat with stage and
NPU metrics. Its purpose is to freeze the before-state and identify the
dominant stage before acceleration work; it cannot satisfy the hard target.
Do not spend a ten-minute soak or three fresh-process measured repeats on this
known disabled-feature baseline. Reserve those expensive gates for the fully
accelerated candidate after prefill graph, encoder graph, compile, and guard-
scope work is complete.

## Qualification sequence

1. Run the [first hardware validation task](qwen3_asr_ascend_910b_validation_task.md)
   on the unchanged base commit and classify the earliest complete failure.
2. Reproduce or isolate the failure locally where possible, make one bounded
   repository change, add focused tests, and map its local commit to the exact
   server commit.
3. Repeat minimal import, startup, batch 1, two-request, sequential, bounded
   concurrency, health, and memory gates in new processes.
4. After eager correctness is stable, qualify the declared generation graph
   mode and record every deviation from repository defaults before running the
   [performance and realtime task](qwen3_asr_ascend_910b_performance_task.md).
5. Compare one variable at a time against the frozen eager-NPU baseline. Retain
   only changes that pass correctness, stability, and repeated performance
   gates.

## Evidence record

For each remote run, add a row here after reviewing its redacted result:

| Run | Server commit | Local equivalent | Stack fingerprint | Gate | Result | First failure or key metric |
|---|---|---|---|---|---|---|
| 910B-000 | pending | `e7d876b2` | pending | target-hardware baseline | pending | Original 910B target has not been run |
| 910C-000 | `e7d876b2` | `e7d876b2` | A3; CANN 9.0.1; torch 2.10.0; torch_npu 2.10.0.post2; SGLang 0.5.18 (Git HEAD missing); triton-ascend 3.2.1 | default startup, then eager batch 1 | failed | Default decode graph: `PagedAttentionOperation`; eager request: GE Manager EC0009 -> error 500001 |
| 910C-001 | no diagnostic commit; edits reverted; final HEAD `fa27495d` | not applicable | Same A3 stack as 910C-000 | daemon process A/B, eager batch 1 | failed; daemon hypothesis rejected | Confirmed stage `daemon=False`; unchanged Manager EC0009 -> GE failure -> error 500001 after stale-process cleanup |
| 910C-002 | no diagnostic commit; final HEAD `aba09fe3` | not applicable | Same A3 stack as 910C-000 | capture raw Manager failure | passed; root cause found | Manager parent `EOFError`; spawn child failed importing `cv2` because `libGL.so.1` was absent |
| 910C-003 | `aba09fe3`; environment-only repair | not applicable | Same A3 stack; `opencv-python` removed; `opencv-python-headless` 5.0.0 reinstalled | eager functional gates, restart, default graph retry | eager passed; graph failed | Eager gates passed; compile-enabled decode capture failed at batch 64 in ATB `PagedAttentionOperation`; later absent with compile disabled |
| 910C-004 | `bb456255`; no runtime edit | not applicable | Repaired A3 stack; SGLang `71de97b2`; headless OpenCV 5.0.0.93 | capacity-1 decode graph capture with compile enabled | failed; ladder stopped as required | Batch 1 failed before ATB: Dynamo rejected skipped triton-ascend `NPUUtils.get_device_properties`; 16/32/64 not run |
| 910C-005 | `18c4e6c4`; no runtime edit | not applicable | Repaired A3 stack; SGLang `71de97b2`; headless OpenCV 5.0.0.93; torch compile disabled | capacity-1 generation graph capture and replay | passed | Prefill/decode captured; one request HTTP 200 with `npu graph: True`; output hash matched eager; encoder graph independently failed with `aclrtMemcpy` 107030 and stayed eager |
| 910C-006 | `2e37bcc6`; no runtime edit | not applicable | Repaired A3 stack; SGLang `71de97b2`; headless OpenCV 5.0.0.93; torch compile disabled | generation decode capacity 16/32/64 | passed | All capture ladders and one-smoke replay checks passed; output hash frozen and `npu graph: True`; compile-enabled batch-64 ATB failure absent; encoder six-bucket fallback unchanged |
| 910C-007 | `330db6d0`; no runtime edit | not applicable | Repaired A3 stack; SGLang `71de97b2`; headless OpenCV 5.0.0.93; torch compile disabled | generation decode capacity 70 capture and one replay smoke | passed | Bucket list ended exactly at 70; all 13 buckets captured in 2.13 s; smoke HTTP 200 with frozen hash and `npu graph: True`; encoder six-bucket fallback unchanged |
| 910C-008 | `4e5befe6`; no runtime edit | not applicable | Repaired A3 stack; SGLang `71de97b2`; compile disabled; encoder graph explicitly disabled | named candidate functional and bounded-concurrency ladder | failed at two concurrent | Startup and one request passed; A+B wave timed out at 120 s with two coordinator-running requests, one prefill, no decode, stable HBM, and no error/fallback |
| 910C-009 | `0452d9de`; no runtime edit | not applicable | Exact `910C-008` named-candidate stack | sequentially warm A/B, then one A+B wave with state polling | passed | Both-warm wave completed in 1.87 s; 2/2 HTTP 200 with frozen hashes, prefill `#new-seq: 2` and `npu graph: True`, zero fallback, and drained state; uncached encoder work is necessary for the `910C-008` hang under the tested ordering |
| 910C-010 | `2336ccff`; no runtime edit | not applicable | Exact `910C-008` named-candidate stack; only generation graph enablement varied | warm-A/cold-B two-request A/B with generation graph enabled versus explicitly disabled | Arm A reproduced hang; Arm B passed | Graph on timed out at 120 s with two running requests and no decode; graph off completed 2/2 in 0.44 s with frozen hashes and `npu graph: False`; generation graph execution is necessary for the cold-encoder hang |
| 910C-011 | `a67b2859`; no runtime edit | not applicable | Exact graph-enabled `910C-010` Arm A stack except `request_build_max_workers=1` | warm-A/cold-B two-request wave with synchronous request building | passed | Completed 2/2 in 1.88 s with frozen hashes and `npu graph: True`; one worker with zero build pending/backlog; asynchronous request-building overlap is also necessary for the hang |
| 910C-012 | `97769286`; no runtime edit | not applicable | Exact graph-enabled `910C-010` Arm A stack except prefill graph explicitly disabled | warm-A/cold-B two-request wave with prefill eager and decode graph retained | wave passed; decode replay not attested | Completed 2/2 in 0.19 s with frozen hashes, prefill `npu graph: False`, zero fallback, and drained state; prefill graph is necessary for the hang; short outputs and absent decode counter left decode replay unproven |
| 910C-013 | `9bae2619`; no runtime edit | not applicable | Exact `910C-012` candidate plus decode log interval 1; serving pyarrow 25.0.0; isolated benchmark client pyarrow 25.0.1 | SeedTTS EN 70 content-distinct inputs at concurrency 8, no benchmark warm-up | failed; hung at first level | Preflight passed; eight requests remained outstanding beyond 90 s and none completed; HBM stayed stable with no error/fallback; 66 decode records with `npu graph: True` attest replay but do not qualify stability; levels 16/32/64/70 did not run |
| 910C-014 | `d69c5d3f`; no runtime edit | not applicable | Exact `910C-013` stack; only decode graph disabled; serving pyarrow 25.0.0; isolated client pyarrow 25.0.1 | Arm B: same 70 content-distinct inputs at concurrency 8 | stability-isolation arm passed; benchmark post-processing incomplete | Warm A plus 70 measured requests returned HTTP 200 with `npu graph: False` and state drained; pending peaked at 8 and running batch at 7; missing declared `openai-whisper` caused WER post-processing failure and no result JSON; decode graph is necessary for the graph-enabled Arm A hang |
| 910C-015 | `544b8cd9`; server-only diagnostic draft | `b64b16d6`; locally reviewed replacement | No hardware run; local Windows environment lacks runnable SGLang/Linux dependencies | Env-gated encoder/build/admission timeline plus decode graph counters | local diagnostic change ready; server verification pending | Corrected Qwen `_enqueue()` override gap, added request correlation and exact encode-return boundary; 10 local diagnostic/encoder tests passed; server must run full focused and Qwen3-ASR suites |
| 910C-016 | `6057bdb3`; includes faulty `b64b16d6` instrumentation | `b64b16d6`; superseded by `144316fe` | Exact repaired A3 stack; SGLang `71de97b2`; clean worktree; serving/client dependency split verified | Focused tests before instrumented cold-input run | failed at preflight; no service run | Decode usage test expected one replay and one eager decode but both counters stayed zero because the local guard excluded `ForwardMode.DECODE`; operator stopped before startup as required |
| 910C-017 | `bd4dca13`; includes `144316fe` | `144316fe` plus prior diagnostic change | Exact repaired A3 stack; SGLang `71de97b2`; `910C-013` graph-enabled profile | Instrumented SeedTTS EN cold-input run at concurrency 8 | completed; hang reproduced and coarse boundaries classified | Focused 13 and Qwen3-ASR 588 passed; eight outstanding: one inside encoder encode, five admitted before prefill, two after prefill; decode completed-replay count stayed at warm value 15; this does not exclude a replay entered but not returned |
| 910C-018 | `39ec921b`; includes `4c25482e` | `4c25482e` plus handoff commit | Exact `910C-017` stack and profile | Repeat concurrency-8 cold-input diagnostic with standard generation forward start/return events | completed; first blocker inside standard decode forward | Focused 25 and Qwen3-ASR 588 passed; one unmatched decode `generation_forward_start` after a completed eager prefill for the same request; completed replay count stayed at warm value 15 |
| 910C-019 | `a949960c`; no runtime edit | SGLang `f86279db9` based on `71de97b2` | Exact `910C-018` stack/profile plus generic SGLang graph-stage logging; repaired headless OpenCV invariant | Repeat concurrency-8 cold-input diagnostic with inner decode graph dispatch markers | completed; selected NPU runner did not return | SGLang 13, omni focused 25, and Qwen3-ASR 588 passed; final batch-size 2 decode emitted `execute_begin` without return; generic inner markers were bypassed by `NPUGraphRunner.execute()` override; editable install had reintroduced non-headless OpenCV and was repaired before the accepted run |
| 910C-020 | `45bbd120`; no runtime edit | SGLang `9dbc4f89c` on `f86279db9` | Exact `910C-019` stack/profile; NPU-specific stage logging only | Repeat concurrency-8 cold-input diagnostic with NPU load, host-copy, graph-update, replay, and join markers | completed; update lane blocked | Tests passed; final raw batch 6/bucket 8 had `graph_update` 34/33, `graph_replay` 34/34, and update-thread join 34/33; host replay returned but update never returned, so the main thread remained in join and the wave made no completion progress |
| 910C-021 | `e923d70c` plus server compatibility edit (hash not reported) | guard `29ca236f`; compatibility equivalent `d9df3a74`; SGLang `9dbc4f89c` | Exact `910C-020` stack/profile plus Qwen3-ASR NPU FIFO execution guard | Authorized concurrency-8 treatment; exploratory 16/32 extension | partial: functional 8 passed; diagnostic contract incomplete; exploratory 16 passed and 32 hung | Concurrency 8: 70/70, p95 0.61 s, WER 0.77%, replay 211/eager 0; concurrency 16: 70/70, p95 3.58 s, WER 0.77%, replay 118/eager 0; concurrency 32: ten-minute timeout, 64 pending and only two measured completions. Required guard events were absent, so the 32 failure boundary is unclassified |
| 910C-022 | `81177bea`; no runtime edit | guard `29ca236f` + compatibility `d9df3a74`; SGLang `9dbc4f89c` | Exact guarded `910C-021` stack/profile after restoring a clean target-device baseline | One concurrency-32 cold-input diagnostic only | functional/guard stability passed; historical scoring denominator anomaly retained | Reported 70 HTTP completions but benchmark evaluated 65/70; WER 0.77%, p95 3.18 s, RTFx 57.3; decode replay 68/eager 0 with bucket 32 hit 27 times; guard wait/acquire/release each 1,303 and state drained. The earlier concurrency-32 hang is invalidated as environment-contaminated, not an intrinsic guard deadlock |
| 910C-023 | `81177bea`; no runtime edit | guard `29ca236f` + compatibility `d9df3a74`; SGLang `9dbc4f89c` | Exact clean guarded candidate; compile, encoder graph, and prefill graph disabled; decode graph through 70 | Fresh functional capacities 64 and 70 | passed on A3 explicit profile | Both levels evaluated 70/70 with WER 0.77%, zero eager decode/fallback, balanced guard events and drain; concurrency 64 p95 4.94 s with bucket 64 replayed 12 times; concurrency 70 p95 4.49 s with bucket 70 replayed 11 times. Functional capacity passed; exact-10-second performance and realtime remain unstarted |
| 910C-024A | pending; original `441b6db4` authorization stopped on pinned 24 kHz source discovery and is superseded | exact10 harness `37f598f3` + corpus transform `2cb63b9e` + accounting hardening `63f235fa` + fixed 24-to-16 kHz transform `8d46ddec` | Must retain the exact accepted `910C-023` stack/profile, pass the fresh environment and ffmpeg preflight, and perform no manual preprocessing | Deterministic resampled 770-clip corpus, NPU parser, batch-one and concurrency-two harness qualification | reauthorized; pending | Require 50 focused passes and complete resampler provenance; stop after the first discrepancy or after the two smokes and cleanup; no baseline ladder, acceleration experiment, soak, process repeats, or realtime is authorized |

The returned evidence may contain commit IDs, package versions, command lines,
test names, tensor shapes/dtypes, aggregate latency/throughput/accuracy, peak
memory, and sanitized traceback categories. It must not contain model paths,
hostnames, usernames, IP addresses, tokens, raw audio, transcripts, full logs,
or proprietary profiler captures.

## References

- [Ascend NPU installation](../get_started/installation_npu.md)
- [Qwen3-ASR usage and current SSE behavior](../cookbook/qwen3_asr.md)
- [Qwen3-ASR concurrency profile](qwen3_asr_concurrency_profile.md)
- [SGLang Ascend NPU guide](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu)
- [SGLang compile-bucket selection at server commit](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py)
- [PyTorch fine-grained compiler controls](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_fine_grain_apis.html)
- [Qwen3-ASR-1.7B model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Python multiprocessing daemon-process contract](https://docs.python.org/3/library/multiprocessing.html)
- [`torch_npu` compatibility matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
- [CANN 9.0.1 release notes](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/releasenote/9.0.1release-notes.md)
- [Public `cann_kb_init` Manager/GE failure report](https://gitee.com/ascend/pytorch/issues/ICVT2X)
- [Public CANN Manager spawn-bootstrapping report](https://gitee.com/ascend/pytorch/issues/I9KIW7)
- [torch_npu NPUGraph update-stream implementation](https://gitee.com/ascend/pytorch/blob/master/torch_npu/npu/graphs.py)
- [vLLM Ascend device-family mapping](https://github.com/vllm-project/vllm-ascend/blob/main/setup.py)
- [SGLang A3 installation examples](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/getting-started/installation.mdx)
- [CANN 9.0 process-log path](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0120.html)
- [CANN 9.0 application log level](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0122.html)
- [SGLang Ascend graph-capacity controls](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/mimo_v2_flash.mdx)
- [Ascend ATB and ASDOPS diagnostic logging example](https://gitee.com/ascend/MindSpeed-LLM/blob/59408f7f7520266976599912f8e35b97fb0c74d/mindie_ref/mindie_llm/atb_models/README.md)
