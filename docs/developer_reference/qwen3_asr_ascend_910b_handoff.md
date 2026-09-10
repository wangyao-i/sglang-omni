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
is primarily an execution and evidence environment. Its operator may check out
the exact repository commits named by the current handoff, run authorized
commands, retain raw evidence locally, and perform declared cleanup. To reduce
diagnostic round trips, the operator may also make one bounded, non-semantic
diagnostic repair when the current task expressly permits it: test-only fixes,
launch/bootstrap or import-provenance fixes, evidence-script corrections, and
additive observability. Each such repair must be its own commit and the text
return must name its commit, files/symbols, concise diff summary, focused-test
result, and the first complete failure that justified it. The server must not
alter model semantics, graph/compile execution, accuracy/performance settings,
dependencies, benchmark inputs, configuration policy, or documentation unless
the handoff explicitly authorizes that exact change. It must not apply an
unreviewed patch or silently repair a failed gate. Raw patches, wheels, logs,
audio, and other files remain server-local: the operator returns only sanitized
text evidence. The local Codex owner reconstructs any accepted server repair,
reviews/tests/commits it, updates this handoff, and then authorizes a
fresh-process verification. Historical server-side diagnostic commits remain
evidence records, not automatic precedent for future edits.

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
`910C-022` 65/70 scoring anomaly. Exact-10-second harness qualification and a
compatibility-profile before-state have now run: 100 sequential requests
passed at 0.289-second p95, while 700 requests at concurrency 70 passed
functionally but missed the hard target at 3.516-second p95 and 39.31
requests/s. The latter service left chip-0 HBM at 87% after shutdown. The
follow-up read-only attribution found 53,966 MB owned by stale NPU context PID
2043369 after the Arm C70 service had been terminated with `SIGKILL`; no
corresponding manageable user process remained. The operator removed the stale
context and `910C-024D` subsequently observed three healthy 4% HBM snapshots,
closing that cleanup exception. `910C-025A` then qualified guarded prefill plus
decode graph as the best compatible profile at 1.771-second p95 and 48.39
requests/s. `910C-026` improved the diagnostic C70 result to 1.652-second p95
and 52.55 requests/s with Encoder Graph enabled, but 64 counted encoder
signature-mismatch eager fallbacks failed feature qualification. `910C-027`
then removed those fallbacks with bounded multi-signature capture, but its
signature count grew from five to eight during measurement, so deterministic
pre-measurement saturation remains required. `910C-029` then exposed that the
decode-only TC profile lacked attention-layer metadata; its later server-side
context bypass and compile-disabled arm violated the source-authority and stop
contracts, so neither arm qualified. Local SGLang commit `93d312480` added the
missing metadata initialization on top of the compile-safe fused-op and NPU
decode-attention boundaries. `910C-030` then captured decode buckets 70 through
2 with torch compile enabled, proving that the metadata and ATB PagedAttention
defects are repaired. Its sole remaining capture failure was compiled batch 1:
Dynamo traced into the out-of-tree `split_qkv_rmsnorm_rope` Triton launcher and
rejected its live `get_device_properties()` call. Local SGLang commit
`44f9e40b5` places that unchanged external kernel behind an opaque custom-op
boundary with fake-tensor shape propagation; it does not modify or require
rebuilding `sgl-kernel-npu`. `910C-031` proved that this boundary removes every
historical capture failure: TC isolation captured all 13 buckets and drained
concurrency 1/8/32/70. The combined `ALL` profile also completed 700/700, but
its output was numerically invalid (WER 1.4464 with corrupted decoded text,
versus the established approximately 0.016 baseline). Thus capture support is
closed but Torch Compile correctness is not. A server-local experiment that
removed `fullgraph=True` changed the corruption pattern without restoring
accuracy and left the SGLang checkout dirty; it is diagnostic evidence, not an
accepted repair. Local SGLang commit `5403d1f7d` adds value-level comparison of
the installed kernel, opaque op, and TorchAir-compiled op for batches 1 and 2.
Combined task `910C-032` first restores a clean Git state, runs that parity
gate, then uses a bounded multi-arm accuracy matrix to locate the corruption
without another one-variable-per-conversation loop. Fully accelerated
performance and realtime remain unqualified**.

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
| Generation graph | Enabled by the Qwen3-ASR defaults and delegated to SGLang; local SGLang repairs `d7e0d517e`, `93d312480`, and `44f9e40b5` add compile-safe attention metadata and opaque boundaries for NPU attention and the external fused QKV/RMSNorm/RoPE Triton kernel | In the explicit compile-disabled profile, A3 decode capacities 1 through 70 passed. `910C-031` captured all 13 compile-enabled buckets and drained concurrency through 70, but the fully enabled C70 output had WER 1.4464 and corrupted text. Torch Compile is capture-capable but numerically unqualified; `910C-032` isolates value parity and batch-transition effects |
| Encoder graph | Local repairs `fa5b8852` and `9080b901` keep NPU sequence-boundary metadata host-side, lazily capture real layouts, and retain a bounded heterogeneous-signature cache | `910C-031` saturated the cache at 8/8 and completed the ALL C70 arm with positive encoder replay and zero eager fallback/capture failure. Encoder Graph is functionally qualified; the combined result remains blocked by Torch Compile accuracy |
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

### Two-goal project completion policy

The owner has fixed two independent acceptance goals, in this order:

1. **Complete support for every currently identified acceleration path.** The
   NPU encoder graph, prefill graph, decode graph, and torch-compile generation
   path must each pass correctness, capture/replay, cold-input concurrency,
   stability, cleanup, and observability gates. They must then pass together in
   one `ALL` profile with positive execution evidence and zero unexpected eager
   fallback. An explicit feature-disable profile remains useful as a regression
   control, but cannot satisfy this goal.
2. **Meet the final offline performance target with that fully enabled `ALL`
   profile.** On the frozen exact-10-second workload, every one of three fresh-
   process concurrency-70 repeats must have p95 latency below 500 ms, all 700
   measured requests must succeed, and measured throughput/RTFx, accuracy,
   health, memory, and cleanup must satisfy the contracts below.

These goals are conjunctive: the project is not complete if the fully enabled
profile is correct but too slow, or if a disabled-feature profile meets the
latency target. Before goal 1 closes, performance measurements are diagnostic
before/after evidence only; they do not constitute the final performance gate.
The execution guard is a correctness mechanism rather than a separately named
acceleration feature. Its current implementation remains qualified, but its
critical-section scope may be narrowed later if profiling shows that it prevents
goal 2.

The original realtime requirement remains a separate protocol/product gate
after the offline `ALL` path is correct and its dominant latency is understood.
SSE token deltas after a complete upload do not count as realtime support.

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

Local commits `2cb63b9e`, `8d46ddec`, and `30b21522` add the deterministic,
revision-pinned SeedTTS corpus transform and nine tests. It creates 70
disjoint warm-up plus 700 measured
clips by concatenating only complete, distinct-audio source utterances,
inserting a fixed 100 ms silence, and padding the tail; it never crops speech
and requires at least 80% speech occupancy. The pinned snapshot was subsequently observed to
contain 1088 mono PCM16 sources at 24 kHz. Commit `8d46ddec` therefore adds a
single generator-owned ffmpeg/swresample conversion to 16 kHz, with fixed
filter arguments, no dithering, no fallback backend, and complete backend
identity in provenance. The first server corpus attempt then exposed a 78.1%
occupancy result from the original cyclic greedy packer. Commit `30b21522`
keeps the 80% gate, exact 10-second duration, English split, and 100 ms
separator unchanged while replacing that packer with deterministic best-fit
whole-utterance plans. It skips infeasible anchors, retries an alternate plan
when duplicate source rows would reproduce an emitted PCM hash, and fails
unless it obtains all 770 distinct outputs. Together with the 43 harness
tests, the locally maintained exact10 toolchain now has 52 focused passes.

### Next isolated task: `910C-024A` harness and corpus qualification

The first authorization at `441b6db4` stopped correctly during corpus
preflight because the source-rate contract rejected all 1088 pinned 24 kHz
WAVs. The next authorization at `b4c4dc39` also stopped correctly when the
original cyclic packer produced only 78.1% speech occupancy for clip zero.
Both are superseded and must not be resumed from those commits. This section
is the only newly authorized server task. The isolated operator must check out
the handoff commit containing local parents `37f598f3`, `2cb63b9e`,
`63f235fa`, `8d46ddec`, and `30b21522`, keep both repositories clean, and keep SGLang at
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

Require exactly 52 passes. Stop at the first collection or test failure. Then
run the corpus command from the performance task against the already approved
pinned local snapshot into a new, empty server-local `910C-024A` evidence
directory; do not reuse or overwrite either failed corpus directory.
Require exactly 770 manifest rows, 70 warm-up/700 measured in provenance,
770 distinct PCM hashes, duration min/max within `10.000 +/- 1/16000` seconds,
the pinned dataset revision, `source_sample_rate_counts={"24000": 1088}` as
observed, `distinct_source_audio_count=666`, `resampled_source_count=1088`, resampler backend
`ffmpeg-swresample`, a 64-character ffmpeg version-output SHA-256, and no
source-format, resampling, occupancy, duplicate, or overwrite failure. Every
derived WAV must be PCM16 mono 16 kHz. Preserve manifest, audio, source
membership, transcripts, paths, and full resampler provenance on the server;
return only counts, duration range, exclusions count, source-rate and distinct
source-audio counts, resampled count, low-occupancy/duplicate-anchor skip
counts, minimum/maximum speech occupancy, ffmpeg first version
line/version-output SHA-256,
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

`910C-024A` passed at `94dec6e0`. The isolated run reported 52/52 focused
tests, a 770-row corpus with 70 warm-up and 700 measured clips, 770 distinct
PCM hashes, exact 10.0-second duration, the pinned manifest SHA-256
`25314d13e3922de7f91525dec5e2f562fed133ee7530355d094bc30c4bd3000b`,
1088 resampled 24 kHz sources, 666 distinct source audios, 10 low-occupancy
anchors skipped, and two duplicate-derived anchors skipped. Batch one was
valid at 0.250-second p95 with WER 0.0000. The two-request smoke was also
valid, but p95 increased to 2.082 seconds with WER 0.0152. Both had complete
raw outcome accounting, valid NPU monitoring, positive decode-graph evidence,
zero unexpected fallback, and clean teardown. Raw audio, transcripts, paths,
request identifiers, and full logs remain server-local.

### Next isolated task: `910C-024B` exact10 compatibility baseline

This section is the only newly authorized server task. It freezes the
before-state of the already qualified compatibility profile; it is not a
fully accelerated candidate and cannot close the 500 ms hard target. The
isolated operator must check out the handoff commit containing this section,
keep sglang-omni and SGLang clean at their declared commits, and use the exact
`910C-024A` corpus without regeneration or mutation. The server must not edit
source, tests, configuration, benchmark code, documentation, installed
packages, or the manifest.

Run two independent arms, each with a new service process and a new evidence
directory. Do not let either arm warm the other. Before each arm, repeat the
established repository identity, editable mapping, OpenCV-headless/libGL,
pyarrow split, ffmpeg identity, model, process/port, device-health, and idle-HBM
preflight. Stop on any mismatch, residual worker or NPU device process,
`SetDevice` error, or unhealthy device rather than applying an ad-hoc server
repair. Start the exact accepted guarded compatibility profile: encoder graph
and prefill graph disabled, decode graph enabled through bucket 70, torch
compile disabled, eight asynchronous request-build workers, maximum running
requests 70, and the accepted `FairDeviceExecutionGuard`. Do not enable
diagnostic event logging for measured passes. Require healthy startup, bucket
70 capture, zero unexpected graph fallback, and empty request state before
launching the client.

Arm S measures 100 distinct requests sequentially:

```bash
python -m benchmarks.eval.benchmark_asr_exact10s \
  --meta "${EXACT10_ROOT}/manifest.jsonl" \
  --host 127.0.0.1 --port "${QWEN3_ASR_PORT}" \
  --model Qwen/Qwen3-ASR-1.7B --lang en \
  --concurrencies 1 --repeats 1 \
  --warmup-samples 70 --max-samples 100 --min-distinct-audio 770 \
  --npu-id 0 --npu-chip-id 0 --monitor-interval-s 1 \
  --request-timeout-s 120 --launch-command "${DECLARED_SERVER_LAUNCH}" \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --output "${EVIDENCE}/sequential100/result.json" \
  --save-raw-dir "${EVIDENCE}/sequential100/raw"
```

Require exactly 100 measured raw records and a valid result with no failed,
timed-out, empty, missing, duplicate, unexpected, or unscoreable outcome. If
Arm S is invalid, preserve evidence, clean up, and stop without Arm C70.
Otherwise save final health and rich model-info snapshots, stop the service,
and attest port, process, device, and HBM cleanup before starting Arm C70.

Arm C70 measures the complete 700-clip partition at concurrency 70 in a second
fresh process:

```bash
python -m benchmarks.eval.benchmark_asr_exact10s \
  --meta "${EXACT10_ROOT}/manifest.jsonl" \
  --host 127.0.0.1 --port "${QWEN3_ASR_PORT}" \
  --model Qwen/Qwen3-ASR-1.7B --lang en \
  --concurrencies 70 --repeats 1 \
  --warmup-samples 70 --max-samples 700 \
  --min-distinct-audio 770 --hard-gate \
  --npu-id 0 --npu-chip-id 0 --monitor-interval-s 1 \
  --request-timeout-s 120 --launch-command "${DECLARED_SERVER_LAUNCH}" \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --output "${EVIDENCE}/concurrency70/result.json" \
  --save-raw-dir "${EVIDENCE}/concurrency70/raw"
```

Require exactly 700 measured raw records. Any correctness, request-accounting,
monitoring, OOM, device, process, or stability failure invalidates the arm.
Latency p95 above 0.500 seconds is an expected possible baseline result: record
it as a hard-target miss after all outcomes drain; do not hide it, retune the
profile, or rerun. Capture rich model-info before requests and after drain so
encoder cache/items/batches/queue wait/encoder time and decode graph
replay/eager/bucket counters can be compared. Require positive bucket-70
decode replay, zero standard-eager decode, zero unexpected graph fallback,
valid NPU HBM and AI Core or NPU-utilization samples, and final cleanup.

Return only sanitized aggregates: validity and failure categories; manifest
and measured SHA-256; evaluated/raw counts; WER; wall time; throughput; RTFx;
latency mean/p50/p90/p95/p99/max; RTF mean/p95; NPU utilization, HBM, power and
temperature summaries; pre/post encoder counters; decode replay/eager counts
and replay buckets; server startup/capture/error/fallback summary; and cleanup
status. Keep raw records, transcripts, paths, model-info payloads, resource
samples, and logs server-local.

Stop after Arm C70 cleanup regardless of whether it meets 500 ms. Do not run a
soak, repeat either arm, start the three fresh-process hard-gate repetitions,
change acceleration settings, begin a repair experiment, or start realtime.
The next committed local task will use this before-state to order the required
prefill-graph, encoder-graph, compile, and guard-scope repairs.

### `910C-024B` result and cleanup exception

The isolated run at `45535923` completed both independent arms with valid
request and monitor accounting. Arm S evaluated 100/100 sequential requests
with zero failures, 24.19-second wall time, 4.13 requests/s, 0.289-second p95,
and WER 0.0167. Arm C70 evaluated 700/700 with zero failures, 17.81-second wall
time, 39.31 requests/s, 393.11 RTFx, 3.516-second p95, 4.959-second p99,
5.408-second maximum latency, and WER 0.0164. Mean/max AI Core utilization was
22%/42%, steady HBM was 87%, and the run reported no unexpected graph
fallback. The compatibility before-state therefore misses the 0.500-second
p95 target by 7.03 times and delivers only 28.1% of the implied 140-request/s
rate. These are measured exact-10-second gaps, not estimates.

The returned summary did not include the required pre/post rich model-info
encoder counters or decode replay/eager/bucket deltas. It therefore does not
yet establish whether encoder execution, decode execution, or the coarse
execution guard dominates the exact workload. Do not promote the operator's
"encoder or decode" suggestion to a root-cause conclusion. `910C-024C` may
read and sanitize those already preserved snapshots if they exist, but their
absence does not authorize a rerun.

Arm C70 did not, however, satisfy cleanup: after service shutdown chip 0 still
reported 87% HBM while chip 1 was at 4%. Port 8000 was free and the operator
reported no residual process, but neither observation proves that the NPU
context or allocation was released. Do not label the complete `910C-024B`
task a clean pass. Retain Arm S and Arm C70 performance evidence as valid
diagnostic measurements, and classify the task as performance-complete with
cleanup unresolved. No acceleration experiment may start on that device state.

### Isolated task: `910C-024C` post-run HBM attribution

This section is the only newly authorized server task. It is read-only and
must not start Qwen3-ASR, rerun a benchmark, load a model, allocate NPU memory,
reset a device, restart a driver/service, kill `hdc`/`tsd` or another process,
install a tool, or edit any source, configuration, documentation, package, or
evidence from `910C-024B`.

Check out the handoff commit containing this section and first record the
current wall-clock time, repository identities, port 8000 state, and a complete
`npu-smi info` snapshot. Record chip 0 and chip 1 HBM from the same snapshot.
Then collect five additional HBM/device-health snapshots at 60-second intervals
for a five-minute observation window. Existing tools may be queried read-only:

```bash
npu-smi info
npu-smi info -t usages -i 0
ps -eo pid,ppid,user,stat,etimes,cmd
fuser -v /dev/davinci0 /dev/davinci_manager /dev/hisi_hdc
lsof /dev/davinci0 /dev/davinci_manager /dev/hisi_hdc
```

`fuser` or `lsof` being absent is not permission to install it; record the
missing tool and continue with the other read-only evidence. Do not return the
full process command lines. Keep raw process, device-node, and NPU output
server-local and return only sanitized process categories, PIDs where policy
allows, ownership type, HBM percentages, device health, and timestamps.
Also inspect only the already preserved `910C-024B` model-info/evidence files.
If the required pre/post rich model-info snapshots exist, return sanitized
encoder items/batches/queue-wait/encoder-time deltas plus decode
replay/eager/bucket deltas. If they do not exist, report that evidence gap and
do not recreate it by starting a service.

Classify exactly one outcome:

1. **delayed release:** chip 0 returns to the established approximately 4%
   baseline during observation, with the first baseline timestamp and no
   device holder remaining;
2. **identified holder:** a live non-baseline process or device-node owner is
   found; report its sanitized category and do not terminate it;
3. **unattributed retained HBM:** chip 0 remains materially above baseline for
   all snapshots and no holder is visible; preserve the evidence and stop for
   operator-approved runtime recovery;
4. **sampling discrepancy:** repeated same-time `npu-smi` views disagree on
   chip identity or HBM; retain the raw outputs locally and report the mismatch.

Stop after the observation and classification. Even if HBM returns to 4%, do
not start an acceleration run in the same task. A new committed handoff must
close this cleanup exception and authorize the first acceleration repair gate.

`910C-024C` completed with outcome 2, **identified holder**. Chip 0 remained at
87% HBM, and the NPU runtime attributed 53,966 MB to context PID 2043369. The
operator found no corresponding user-space process that could be terminated
through normal process management. The context is attributed to the Arm C70
sgl-omni process being terminated with `SIGKILL`, which bypassed normal NPU
teardown. Do not turn that attribution into permission to kill a daemon,
reset a device, restart the driver, or reboot the host.

The existing Arm C70 server log also yielded one terminal/coarse stats view:

- encoder: 251 batches, 754 items, 1.24-second average queue wait,
  13.26-second maximum queue wait, and 22.6 seconds cumulative encoder time;
- decode graph: 100% NPU graph replay, with all 13 configured buckets including
  bucket 70 observed;
- no pre/post rich model-info snapshots exist, so exact measured-only deltas
  and a complete encoder/decode/guard time decomposition cannot be recovered.

The queue-wait measurements are large relative to request latency and make
encoder/guard scheduling the first performance-analysis priority, while 100%
decode replay rules out eager decode fallback as the explanation for this
baseline. This is prioritization evidence, not proof that the guard alone is
the root cause: the encoder counters are cumulative/coarse and there is no
paired pre/post snapshot.

### `910C-024D` recovery result and next campaign `910C-025A`

`910C-024D` passed on handoff commit `db19be76`. Three snapshots at zero, ten,
and twenty seconds reported both chips healthy and stable at 4% HBM (chip 0:
3,174/65,536 MB; chip 1: 2,890/65,536 MB at the first snapshot). PID 2043369,
other context holders, sgl-omni workers, and residual service processes were
absent; port 8000 was free. The server did not start an acceleration run. The
retained-context cleanup exception from `910C-024B/C` is therefore closed.

The project owner has requested a faster grouped qualification campaign, but
the known encoder-graph and torch-compile failures require local code changes.
Do not rerun those unchanged failing paths. `910C-025A` groups only the two
current-code arms that add new information; repaired EG, TC, and ALL arms will
be authorized together after their local reviewable commits exist. Grouping
changes turnaround time, not evidence or correctness requirements. The
isolated server still must not edit source, tests, configuration,
documentation, packages, or repository state.

#### Phase 0: accepted recovery and exact-checkout gate

The `910C-024D` three-snapshot recovery evidence above satisfies the hardware
portion of phase 0 and need not be repeated before `910C-025A`. Check out the
handoff commit containing this authorization and reconfirm immediately before
the first arm that:

- chip 0 and chip 1 are healthy and each is at no more than 5% HBM;
- PID 2043369 and every unexpected compute/context holder are absent;
- no sgl-omni, SGLang, model worker, benchmark client, or orphan Python process
  remains;
- port 8000 is free and both repositories are at their exact declared clean
  commits.

Any discrepancy stops the entire campaign. Do not repair, kill, reset,
restart, install, or mutate anything in place.

#### Common measured contract

If phase 0 passes, execute only E0 and P below, in order. Each arm must use a new
service process, a distinct evidence directory, the frozen `910C-024A` corpus
and manifest hash, and the same model, BF16 precision, memory fraction, cache,
admission settings, eight request-build workers, maximum-running-requests 70,
request timeout, WER stack, NPU monitor, and exact10 harness as `910C-024B`.
Record the resolved profile rather than trusting command-line intent.

For every arm that starts and positively attests its requested acceleration
path, run batch one and then exactly one 700-request concurrency-70 exact10
hard-gate measurement. This campaign deliberately omits the sequential arm,
soak, and three-repeat confirmation. It is a feature-screening and performance-
direction campaign, not final acceptance.

After every arm, require graceful service termination, port release, no worker
or context holder, and three healthy HBM snapshots returning to at most 5%.
`SIGKILL`, retained context, OOM, device/ACL failure, unexplained HBM retention,
or failed cleanup stops the entire campaign. A clean startup/capture/feature
failure ends only that arm: preserve its first complete failure, cleanly tear
down, and continue to the next independent arm. A valid arm that misses the
500 ms or 140-request/s target is still measured and does not stop later arms.
Request-accounting, correctness, WER, monitor, unexpected fallback, and cleanup
failures remain disqualifying.

#### Arms

| Arm | Encoder graph | Prefill graph | Decode graph | Torch compile | Guard | Purpose and positive gate |
|---|---:|---:|---:|---:|---:|---|
| E0 | off | off | off | off | inactive | Exact all-eager/no-guard control; batch one plus C70 measurement |
| P | off | on | on through 70 | off | active | Prefill capture and replay must be positive with zero unexpected fallback, then measure C70 |

EG, TC, and ALL are explicitly not authorized on the current code. Do not use
the isolated server to diagnose, patch, monkeypatch, or reconfirm their already
known failures. Local development must first repair encoder capture and the
compile boundary with regression tests and committed handoff mappings. Do not
invent another configuration or tune parameters between E0 and P. Do not run
realtime in this campaign.

Return one sanitized matrix row for E0 and P containing: exact repository and
dependency HEADs; resolved profile; startup/capture status or first failure;
batch-one result; C70 validity and request-accounting counts; WER; wall time;
throughput; RTFx; latency mean/p50/p90/p95/p99/max; NPU utilization, HBM, power,
and temperature; encoder batch/item/queue/elapsed counters; graph capture and
replay buckets; guard wait/acquire/release counts; forbidden signatures; and
cleanup/HBM status. Preserve raw logs, JSONL, transcripts, paths, request IDs,
audio and profiler data only on the isolated server.

`910C-025A` completed both arms. The all-eager/no-guard E0 arm measured
2.641-second p95 and 31.52 requests/s. The guarded prefill-plus-decode graph P
arm measured 1.771-second p95 and 48.39 requests/s. Relative to the accepted
`910C-024B` decode-only guarded baseline (3.516-second p95 and 39.31 requests/s),
P reduced p95 by 49.6% and increased throughput by 23.1%. It is the best
measured configuration so far, but its p95 remains 3.54 times the 0.500-second
target and its throughput is only 34.6% of the implied 140-request/s target.
Prefill graph with the existing execution guard is therefore functionally and
directionally qualified; it is no longer an open compatibility repair.

The returned summary did not state whether a context remained after teardown
or provide the full requested request-accounting, WER, graph-counter, NPU, and
cleanup matrix. Do not infer a cleanup failure from the operator's conditional
warning. Before any later hardware run, require the normal no-holder, healthy
HBM, free-port, and clean-checkout preflight; if a retained context is actually
observed, stop and return its identity rather than modifying the host under a
code-validation task.

### `910C-026` NPU encoder-graph repair result

Local commit `fa5b8852` owns the first encoder-graph repair. On Ascend it keeps
the fused-attention sequence-boundary metadata on the host before capture,
eliminating the captured-stream device-to-host synchronization that previously
raised `aclrtMemcpy` error 107030. Because Ascend consumes those boundaries as
host-side operator parameters, the runner lazily captures the first real
window signature for each token bucket and admits at most one signature per
bucket. A different signature uses an explicit, counted eager fallback rather
than replaying stale boundaries or growing graph memory without a bound.
`model_info.encoder_cuda_graph` now reports capture, replay bucket, and eager-
fallback counts. The isolated run proved that this removes the illegal capture-
stream synchronization and that identical signatures capture and replay, but
it also proved that the one-signature-per-bucket policy is insufficient for the
real exact10 corpus.

The focused encoder-graph suite passed 15 tests with 1 skipped and the model-
info suite passed 4 tests. The complete Qwen3-ASR suite then reported 583
passed, 11 failed, and 3 skipped. Those failures were attributed by the
operator to pre-existing `SimpleNamespace.audio_tower` mocks, but the task's
declared stop-on-test-failure rule still applied. The server continued through
hardware measurement after that first failed gate; retain the later numbers as
diagnostic evidence only, not as a valid feature qualification.

Startup deferred synthetic NPU capture as designed and reported no error
107030, ACL/GE/OOM, capture failure, or `bucket stays eager` signature. Two
identical batch-one requests captured bucket 256 with 11 windows, replayed it,
returned identical normalized output, and kept both capture-failure and eager-
fallback counts at zero. The 700-request C70 arm then completed 700/700 with
WER 0.0164, 1.652-second p95, 52.55 requests/s, and RTFx 525.46. This is the
best directional profile so far, but it missed the 0.500-second target and,
critically, recorded 64 `npu_signature_mismatch` eager fallbacks. Four encoder
graphs were captured for token buckets 256, 512, 1024, and 2048; 218 requests
replayed while 64 fell back. Prefill and decode replay remained positive,
including decode bucket 70.

Therefore `910C-026` **fails encoder-graph feature qualification**. It proves a
useful implementation direction and directional gain, not complete support.
The next work is local and no new isolated run is authorized yet: replace the
single-signature-per-token-bucket policy with a bounded strategy that covers
the real window-layout diversity without stale host metadata, unbounded graph
growth, or silent eager fallback; fix or explicitly rebase the full-suite mock
contract; add coverage for heterogeneous signatures; then issue a new handoff
commit for a fresh-process hardware gate.

The original execution instructions are retained below as the audit record for
what `910C-026` was authorized to run.

Check out local code commit `fa5b8852` plus the handoff commit containing this
authorization. Keep the exact SGLang dependency commit and all environment
invariants from accepted `910C-025A`; report and stop on any mismatch. The
isolated server must not edit source, tests, configuration, documentation,
packages, site-packages, or environment state.

Run, in order:

1. the focused encoder-graph and model-info tests
   `tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py` and
   `tests/unit_test/model_runner/test_prefill_cuda_graph_usage.py`, followed by
   the complete `tests/unit_test/qwen3_asr` suite; stop on collection or test
   failure;
2. the normal clean-device preflight: both chips healthy at no more than 5%
   HBM, no holder/worker/client, port 8000 free, headless OpenCV invariant, and
   exact clean repository/dependency HEADs;
3. start one fresh service with the exact accepted `910C-025A` P profile,
   changing only `enable_encoder_cuda_graph` from false to true; keep prefill
   and decode graphs enabled, decode capacity 70, torch compile disabled, the
   execution guard active, and eight request-build workers;
4. require the NPU lazy-capture deferral marker at startup and zero startup
   capture failure, error 107030, `bucket stays eager`, ACL, GE, OOM, or graph
   fallback signatures;
5. send the same frozen exact10 batch-one sample twice. The first successful
   request must produce both the bounded encoder capture marker and encoder
   replay marker; the second must return the same normalized output/hash,
   increase `encoder_cuda_graph.replay_count`, and not capture a second graph
   for the same key. Require `capture_failure_count=0` and
   `eager_fallback_count=0` after the pair;
6. if the pair passes, run one 70-sample concurrency-70 warmup excluded from
   measurement, then exactly one 700-request concurrency-70 exact10 hard-gate
   measurement using the frozen `910C-024A` corpus and the same harness,
   timeout, WER, request-accounting, and NPU-monitor contract as `910C-025A`;
7. capture rich model-info immediately before warmup, after warmup, and after
   measurement. Report deltas for encoder, prefill, and decode graph capture,
   replay and fallback counters. Any measured encoder eager fallback makes the
   feature qualification fail, but allow the already launched 700-request arm
   to drain so its diagnostic performance result is preserved;
8. stop gracefully and require port release, no residual process/context, and
   three healthy HBM snapshots at no more than 5%. Do not use `SIGKILL`; a
   device error, OOM, retained context, forced cleanup, or HBM recovery failure
   stops the task and remains a blocking exception.

Return the focused and full test counts; resolved profile; capture/replay log
markers; the three sanitized graph-counter snapshots/deltas; batch-one hashes;
700-request validity and accounting; WER; wall time; throughput; RTFx; latency
mean/p50/p90/p95/p99/max; NPU utilization/HBM/power/temperature; forbidden
signature counts; and cleanup state. Keep raw logs, JSONL, paths, transcripts,
request IDs, audio, and profiler data server-local. Do not start torch-compile
or realtime work in this task.

### Next isolated task: `910C-027` bounded multi-signature encoder graph

Local commit `9080b901` is the follow-up to the failed `910C-026` feature gate.
It replaces one-signature-per-token-bucket admission with an exact-signature
cache bounded globally by `pre_lm_max_batch_size` (8 in this profile). Multiple
host-side window layouts may therefore own independent graphs even when they
share a token bucket, while graph memory cannot grow beyond eight signatures.
`model_info.encoder_cuda_graph` now exposes `npu_signature_capacity` and
`npu_signature_count`; capacity exhaustion is an explicit
`npu_signature_capacity` eager-fallback reason. The same commit also makes guard
creation tolerate legacy/test model objects without `audio_tower`; a real
Qwen3-ASR model still creates the guard under the existing NPU graph conditions.

Check out code commit `9080b901` plus the handoff commit containing this
authorization. Preserve the exact model, SGLang dependency commit, packages,
headless OpenCV invariant, exact10 corpus, service profile, and benchmark-client
environment recorded by `910C-026`. The isolated operator must not edit code,
tests, configuration, packages, site-packages, or documentation.

Run the following accelerated gate in one fresh service process:

1. Run `tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py`,
   `tests/unit_test/model_runner/test_prefill_cuda_graph_usage.py`, and the
   complete `tests/unit_test/qwen3_asr` suite. Stop on the first collection or
   test failure; unlike `910C-026`, do not start hardware after a failed suite.
2. Require exact clean repository/dependency HEADs, both chips healthy at no
   more than 5% HBM, no holder/worker/client, port 8000 free, and headless OpenCV
   with no `libGL` dependency. Stop and report any mismatch.
3. Start the same `EG` profile as `910C-026`: encoder, prefill, and decode graphs
   enabled; torch compile disabled; decode capacity 70; execution guard active;
   eight request-build workers. Require startup success, NPU lazy encoder-
   capture deferral, and zero ACL/GE/OOM/error-107030/capture-failure or graph-
   fallback signature.
4. Capture rich model-info before warm-up. Use the frozen exact10 corpus to
   issue successful warm-up waves at client concurrency 1, 2, 4, and 8,
   followed by the standard 70-sample concurrency-70 warm-up. Drain state
   between the small waves. This phase deliberately populates normal encoder
   batch-size signatures and is excluded from all measurements.
5. Capture rich model-info after the final warm-up. Require encoder capture and
   replay to be positive, `npu_signature_capacity=8`,
   `1 <= npu_signature_count <= 8`, `capture_failure_count=0`, and
   `eager_fallback_count=0`. Require positive prefill/decode replay and zero
   unexpected fallback. If any condition fails, stop before C70.
6. Run exactly one 700-request concurrency-70 exact10 hard-gate arm. Preserve
   all request outcomes and let an already-launched arm drain. Capture rich
   model-info immediately after measurement. Encoder signature count and
   captured graph count must not increase during measurement, proving warm-up
   coverage; encoder/prefill/decode replay counts must increase; every eager-
   fallback and capture-failure count must remain zero. Any fallback, timeout,
   request-accounting error, output/accuracy regression, OOM, or device error
   fails the feature gate even if aggregate performance improves.
7. Stop gracefully without `SIGKILL`. Require port release, no residual user or
   device context, and three healthy HBM snapshots at no more than 5%.

Return exact test counts; resolved profile and repository/dependency HEADs;
sanitized pre-warmup/post-warmup/post-measurement graph counters and deltas;
signature capacity/count and captured buckets; request-accounting/accuracy;
wall time, throughput, RTFx, latency mean/p50/p90/p95/p99/max; NPU utilization,
HBM, power, and temperature; forbidden-signature counts; and cleanup state.
Keep raw logs, JSONL, paths, transcripts, request IDs, audio, and profiler data
server-local. This is the only authorized server task. Do not enable torch
compile or start realtime work in `910C-027`.

### `910C-027` result and next isolated task `910C-028`

`910C-027` proved that the bounded multi-signature implementation removed the
functional fallback seen in `910C-026`: the C70 arm completed with
`capture_failure_count=0` and `eager_fallback_count=0`, and graceful `SIGTERM`
cleanup returned both chips to 4% HBM with no retained context. The signature
cache contained five graphs after the declared warm-up, however, and grew to
its hard capacity of eight during measurement. This violated the predeclared
no-measurement-capture condition, so the run is not a clean Encoder Graph
performance qualification. Its 700-request C70 result (p95 2.685 seconds,
50.60 requests/s, RTFx 506) is retained as capture-contaminated diagnostic
evidence. Exact focused/full test counts were not included in the returned
summary and must be returned by the next run.

Do not increase the signature capacity yet. `910C-027` reached exactly 8/8
without fallback or capture failure, so there is no evidence that capacity is
insufficient. Canonicalizing Ascend host operator parameters would be a larger
semantic change and is not justified while deterministic warm-up can exercise
the bounded signature set. Accepting graph capture during the measured arm is
also not permitted for the final performance profile.

`910C-028` is an accelerated, no-code-change warm-up qualification of the same
commit `9080b901`. Check out code `9080b901` plus the handoff commit containing
this authorization and preserve the exact SGLang dependency, model, packages,
headless OpenCV invariant, exact10 corpus, benchmark client, EG service profile,
timeouts, accuracy rules, and cleanup contract from `910C-027`. The server may
not edit source, tests, configuration, packages, site-packages, or docs.

Run in order:

1. Repeat the focused encoder-graph/model-info tests and complete Qwen3-ASR
   suite. Return exact pass/fail/skip counts and stop before hardware on any
   failure.
2. Require exact clean HEADs, port 8000 free, no holder/worker/client, headless
   OpenCV with no `libGL`, and both chips healthy at no more than 5% HBM.
3. Start one fresh `910C-027` EG service: encoder, prefill, and decode graphs on;
   torch compile off; decode capacity 70; guard active; encoder batch size and
   request-build workers both 8. Require zero startup forbidden signatures.
4. Capture pre-warm-up rich model-info. Issue separate exact10 warm-up waves at
   client concurrency 1, 2, 3, 4, 5, 6, 7, and 8, draining after each wave and
   recording encoder signature/capture/replay/fallback counters. These levels
   target every possible encoder batch size rather than relying on larger
   client concurrency to produce a particular encoder batch.
5. Then issue up to three excluded 70-sample C70 warm-up waves, draining and
   recording rich model-info after each. Stop warm-up successfully as soon as
   `npu_signature_count=8` and two consecutive drained snapshots show unchanged
   signature and captured-graph counts. If the cache does not reach a stable
   8/8 within three C70 waves, or any capture failure/eager fallback/device
   error appears, stop without measurement and return the counter timeline.
6. After saturation, run one 700-request C70 exact10 arm. Require 700/700 valid,
   stable signature/captured-graph counts, positive encoder/prefill/decode
   replay deltas, zero eager fallback and capture failure, unchanged normalized
   correctness/WER, and no timeout/OOM/device error. This closes Encoder Graph
   feature qualification if it passes; its latency remains diagnostic rather
   than the final project hard gate.
7. Stop gracefully without `SIGKILL`; require port release, no retained process
   or context, and three healthy HBM snapshots at no more than 5%.

Return the complete counter timeline for every warm-up wave, exact test counts,
resolved profile/HEADs, C70 request accounting/WER/latency/throughput/RTFx/NPU
metrics, forbidden signatures, and cleanup state. Raw data remains server-local.
Do not enable torch compile or realtime in `910C-028`.

### Combined code handoff: `910C-029` Torch Compile and `ALL`

`910C-028` is superseded before execution by this combined task. The project
owner has requested one complete, locally reviewed code delivery followed by
one server qualification sequence, rather than a code-edit/run round trip for
each acceleration feature. This changes task batching, not source authority:
the isolated-server agent still must not modify source, tests, configuration,
documentation, packages, or site-packages, and must not commit. On any code
defect it must return evidence to the local owner instead of repairing it.

Use the established Git collaboration path only. Do not create or transfer an
offline archive. Fetch and check out these remote branches and exact source
snapshots together; if either existing server worktree is dirty, stop and
report it instead of overwriting local state:

- SGLang-Omni `origin/qwen3-asr-910b-opt`, code `f10067cf` plus the handoff
  commit containing this task;
- SGLang `origin/codex/qwen3-asr-torch-compile`, code
  `d7e0d517e9c6f3537078c53a504bcf8530e4cebf`, based on the
  previously qualified diagnostic lineage `9dbc4f89c` / upstream base
  `71de97b264`.

The SGLang change keeps fused operators in their compile-safe form while
`torch.compile` traces, establishes an NPU capture context for decode, and
routes decode attention through the registered graph-safe custom-op boundary.
It is intended to remove both historical compile-enabled failures: the batch-1
Dynamo trace into Triton-Ascend `NPUUtils.get_device_properties`, and the
non-compiled large-bucket ATB `PagedAttentionOperation setup failed`. The Omni
change adds positive model-info evidence for `torch_compile_enabled` and
`compile_bs`; absence of the historical errors alone is not acceptance.

Run the following as one authorized task, using a fresh service process for
each service arm. Stop the sequence at the first failed gate, retain raw
evidence server-local, and return one sanitized summary. Do not improvise a
patch or dependency change.

1. Preflight both exact HEADs and clean tracked worktrees. Verify that the
   editable SGLang import resolves to the transferred SGLang tree without a
   dependency-resolving reinstall. Recheck the hard OpenCV invariant:
   `opencv-python` absent, headless OpenCV present, fresh-process `cv2` import
   succeeds, and the loaded extension has no `libGL` dependency. Require port
   8000 free, no worker/client/context holder, both chips healthy, and HBM no
   more than 5%.
2. Before hardware, run the SGLang decode-graph-runner tests containing the new
   compile-boundary cases, the Omni model-info/encoder-graph focused tests, and
   the complete Qwen3-ASR suite. Return exact pass/fail/skip counts. Stop before
   service startup on any collection or test failure.
3. Run a **TC isolation arm** with encoder graph disabled, prefill graph
   disabled, decode graph enabled through bucket 70, torch compile enabled,
   `torch_compile_max_bs=32`, the execution guard active, encoder batch size 8,
   request-build workers 8, and `max_running_requests=70`. Require the resolved
   profile and rich model-info to report `torch_compile_enabled=true`,
   `compile_bs=[1,2,4,8,12,16,24,32]` (subject only to an explicitly reported
   alignment filter), and capture buckets through 70. Run drained exact10 waves
   at client concurrency 1, 32, 64, and 70. Require HTTP success and unchanged
   accuracy, positive replay deltas at a compiled bucket and at buckets 64 and
   70, zero eager fallback, and zero Dynamo skipped-function, Triton device-
   property, ATB PagedAttention, graph-capture, ACL/GE, OOM, or device-error
   signature. This closes Torch Compile qualification only if both compiled and
   non-compiled decode buckets execute successfully.
4. Gracefully stop the TC arm and establish a clean device baseline. Start one
   fresh **ALL arm** with encoder graph, prefill graph, decode graph, and torch
   compile all enabled; retain the same guard, batch/workers/capacity/memory,
   exact10 corpus, timeout, and accuracy settings. Require positive startup
   evidence for all four features, not merely configuration text.
5. Before measuring ALL, deterministically saturate the bounded encoder graph:
   issue separate drained exact10 waves at concurrency 1 through 8, then up to
   three excluded 70-sample C70 waves. Continue only after
   `npu_signature_count=8` and two consecutive drained snapshots have unchanged
   signature and captured-graph counts. Any encoder eager fallback, capture
   failure, device error, or failure to stabilize aborts the arm.
6. Run batch 1 and concurrency 2 correctness smokes, then one 700-request
   exact10 C70 go/no-go measurement. Require 700/700 valid, unchanged WER,
   positive encoder/prefill/decode replay, positive compiled-bucket execution,
   stable encoder graph counts, zero unexpected eager fallback/capture growth,
   no timeout/OOM/device error, and complete request/state drain. Return p50,
   p90, p95, p99, max, throughput, input-audio-seconds/s, RTFx, stage timings,
   guard wait/hold statistics, NPU utilization/HBM/power, and every graph/
   compile counter delta.
7. If and only if that ALL C70 arm meets p95 below 0.500 seconds, at least 140
   requests/s, and every correctness/feature condition, continue in this same
   authorization with the remaining formal campaign: 100 sequential requests;
   concurrency 8, 16, 32, 64, and 70; a ten-minute C70 soak; and two additional
   fresh-process 700-request C70 measurements so that three fresh-process C70
   results exist. Each C70 repeat must independently meet the hard target. If
   the first ALL C70 arm misses, stop and return its bottleneck evidence rather
   than spending time on redundant repeats.
8. Use graceful shutdown for every arm. Do not use `SIGKILL` as normal cleanup.
   Require port/process release and three healthy HBM snapshots at or below 5%.
   If bounded forced cleanup is unavoidable, attribute any retained NPU context
   and report it instead of starting the next arm.

Realtime remains outside `910C-029`. It begins only after the offline ALL path
is functionally qualified and its remaining performance gap is understood.

### `910C-029` result and corrected combined task `910C-030`

`910C-029` did not produce a qualification result. The TC isolation arm first
exposed a real local defect: with prefill graph disabled, SGLang had never
populated `model_runner.attention_layers`, so the new NPU compile context raised
`AttributeError`. The server then made an unauthorized uncommitted edit that
returned an empty context when the metadata was absent. That edit removed the
graph-safe attention boundary under test, after which capture predictably fell
back to the ATB path and failed at `PagedAttentionOperation`. Therefore this
ATB failure does not disprove the intended local boundary; it demonstrates
that bypassing it recreates the historical failure.

The returned pre-hardware counts were encoder/model-info 15 passed and 1
skipped, complete Qwen3-ASR 594 passed and 3 skipped, and no reported SGLang
decode-runner test failure. The final arm used an otherwise idle NPU 7 at its
4% HBM baseline. These facts are retained, but they do not cure the source and
stop-contract deviations below.

The server also continued after the TC first failure and ran a compile-disabled
encoder+prefill+decode arm. This violated the stop rule and was not the declared
`ALL` profile. Its 140 outstanding requests and unresponsive shutdown are
retained as diagnostic evidence only. They do not establish a new execution-
guard defect: the existing encoder service holds the shared guard around its
entire `_execute_batch`, including encoder-graph capture/replay, and earlier
`910C-027` completed the encoder+prefill+decode C70 measurement. Because 140
pending requests exceed one declared 70-request wave, the next run must prove
drain before submitting another wave and must never exceed 70 outstanding.

Local SGLang commit `93d312480` fixes the actual TC defect. Before decode graph
capture begins, NPU torch compile now derives the same attention/MoE layer
metadata normally published by prefill-graph setup. It preserves already
initialized metadata and fails explicitly if the decoder attention map is
incomplete; it never silently returns an empty compile context. Three CPU unit
tests cover decode-only initialization, preservation, and incomplete-map
failure. Local compileall, fatal Ruff checks, and `git diff --check` pass;
Windows cannot collect the Linux SGLang suite because the standard `resource`
module is unavailable, so the isolated Linux test gate remains mandatory.

`910C-030` replaces `910C-029` and remains one combined server interaction.
Use Git only: SGLang-Omni `origin/qwen3-asr-910b-opt` at the handoff commit
containing this task, and SGLang `origin/codex/qwen3-asr-torch-compile` at exact
commit `93d312480`. The server must discard its uncommitted 029 edit by using a
fresh clean worktree or by stopping and asking the operator to provide one; the
agent itself must not reset, overwrite, stash, or repair a dirty checkout.

Run the same preflight and test gates from `910C-029`, including the three new
SGLang metadata tests. Then:

1. Repeat the TC isolation arm exactly as specified in `910C-029`. Require no
   `AttributeError`, require non-empty attention metadata through the compile
   context, require model-info `torch_compile_enabled=true`, and exercise
   drained concurrency 1, 32, 64, and 70 waves. The old Dynamo, Triton device-
   property, and ATB PagedAttention signatures must all be zero. Stop on the
   first failure; do not edit code and do not start ALL.
2. Only if TC passes, gracefully stop it, prove the device returned to baseline,
   and run the declared fully enabled ALL arm. Compile must remain enabled; a
   compile-disabled arm is not ALL.
3. For encoder saturation, submit concurrency 1 through 8 one drained wave at
   a time. Submit at most one C70 saturation wave, wait for all request and
   scheduler counters to drain to zero, and capture model-info before deciding
   whether another is needed. Never allow more than 70 outstanding requests.
   If a wave has no completion progress for 90 seconds, capture guard owner/
   ticket/wait/hold, encoder signature counters, request states, and graph
   counters, then stop; do not submit another wave.
4. If ALL stabilizes at 8/8, continue with the batch1/conc2 and one 700-request
   C70 go/no-go measurement from `910C-029`. The conditional full performance
   campaign and cleanup rules remain unchanged.

Return the exact checked-out HEADs, confirmation that both worktrees stayed
clean, test counts, resolved profiles, every target-path counter, per-wave
drain timeline, first complete failure if any, and cleanup state. The isolated
agent must make zero source, test, config, documentation, package, or commit
changes during `910C-030`.

### `910C-030` result and combined qualification task `910C-031`

`910C-030` used clean SGLang-Omni `d0d55e8c` and SGLang `93d312480` on
NPU 15. SGLang tests passed 20/20, the focused Omni suites passed 15 with one
skip, and the full Qwen3-ASR suite passed 594 with three skips. The TC-isolation
arm no longer raised the missing-`attention_layers` error or ATB
`PagedAttentionOperation` setup failure. It captured 11 of 13 decode buckets,
from batch 70 through batch 2. Batch 1, the only compiled bucket in this
profile, failed while Dynamo traced
`split_qkv_rmsnorm_rope -> get_device_properties()` in the installed
`sgl_kernel_npu` Triton launcher. The operator correctly made no code change
and did not start `ALL`.

The attempted `cuda_graph_bs_decode` override is not a repair and must not be
used in the next run. Besides encountering a string/list conflict, excluding
batch 1 would waive the only compiled bucket and violate the all-feature gate.
Local SGLang commit `44f9e40b5` instead registers the existing external fused
kernel as `sglang::npu_split_qkv_rmsnorm_rope`, supplies fake q/k/v metadata to
Dynamo, and calls the unchanged installed kernel at runtime. No
`sgl-kernel-npu` source, wheel, package, or dependency change is required or
authorized.

`910C-031` is one combined, conditional server task. Use Git only:

- SGLang-Omni: fetch `origin/qwen3-asr-910b-opt` and check out the exact handoff
  commit containing this task;
- SGLang: fetch `origin/codex/qwen3-asr-torch-compile` and check out exact
  commit `44f9e40b5`;
- keep the existing installed `sgl-kernel-npu` package unchanged. Record its
  version and import path only; do not rebuild or reinstall it.

Before startup, require both tracked worktrees clean, the established headless-
OpenCV invariant, no service/holder process, a free port, healthy selected NPU,
and baseline HBM. Run the SGLang decode-runner tests plus
`test_npu_fused_ops.py`; then run the established Omni encoder/model-info
focused suites and the complete Qwen3-ASR suite. Stop at the first collection
or test failure. The new NPU test must prove the custom op is registered, its
fake outputs have the declared shapes/dtype, and symbolic tracing retains one
opaque `sglang::npu_split_qkv_rmsnorm_rope` node.

If tests pass, execute these phases in the same task, always using a fresh
service between TC and `ALL` and draining every wave before the next:

1. **TC isolation.** Preserve the exact `910C-030` TC configuration and its
   original full bucket list `[1,2,4,8,12,16,24,32,40,48,56,64,70]`; do not set
   `cuda_graph_bs_decode` or exclude batch 1. Require all 13 buckets to capture,
   `torch_compile_enabled=true`, compile bucket 1 present, and zero
   `get_device_properties`, Dynamo skipped-function, ATB PagedAttention,
   graph-fallback, ACL/GE, OOM, or request-error signatures. Run drained waves
   at concurrency 1, 8, 32, and 70 with frozen correctness/accuracy checks,
   positive decode graph replay, zero eager decode, and full state drain.
2. **Fully enabled `ALL`, conditional on TC passing.** Gracefully stop TC,
   prove HBM returned to baseline, then enable torch compile, encoder graph,
   prefill graph, and decode graph together with the existing execution guard
   and asynchronous request builders. Saturate encoder signatures with drained
   batches 1 through 8 before measurement. Require signature capacity/count
   stable at the declared bound, zero encoder eager fallback/capture failure,
   positive encoder/prefill/decode graph execution, positive compiled-batch
   evidence, balanced guard events, and no more than 70 outstanding requests.
   Run batch 1, concurrency 2, 8, 32, and 70, stopping at the first 90-second
   no-progress interval and returning the first complete boundary.
3. **Exact10 C70 diagnostic, conditional on `ALL` passing.** Without restarting
   or changing the `ALL` profile, run one 700-measured-request, concurrency-70
   repeat on the frozen `910C-024A` manifest. Return validity, WER, p50/p95/p99/
   max latency, throughput, RTFx, stage timing, NPU utilization/HBM, every graph
   and compile counter delta, encoder signature delta, guard wait/hold metrics,
   and drain/cleanup state. Label this a single diagnostic before the final
   three-fresh-process campaign; it does not by itself close the hard target.

On any failure, stop the remaining phases, capture the first full failure and
current counters, shut down gracefully, and verify two stable baseline HBM
snapshots. Do not use `SIGKILL` unless bounded cleanup has failed and the report
explicitly records the resulting context risk. Return sanitized aggregates and
exact commits only; retain paths, request IDs, transcripts, audio, and full logs
server-local. The isolated agent must make zero source, test, configuration,
documentation, package, site-package, or commit changes during `910C-031`.

### `910C-031` result and accelerated correctness-localization task `910C-032`

`910C-031` used clean SGLang-Omni `9c27cca9` and SGLang `44f9e40b5`.
The declared SGLang suites passed 3 fused-op and 20 decode-runner tests; Omni
passed 15 focused tests with one skip and 594 Qwen3-ASR tests with three skips.
TC isolation captured every decode bucket `[1,2,4,8,12,16,24,32,40,48,56,
64,70]`, reported compile buckets `[1,2]`, drained concurrency 1/8/32/70, and
contained none of the historical Dynamo, device-property, PagedAttention,
ATB, ACL, GE, OOM, fallback, or request-error signatures. That is decisive
capture and scheduling evidence, but the returned TC waves did not include a
corpus-WER result and therefore do not establish numerical correctness.

The fully enabled arm captured and exercised encoder, prefill, and decode
graphs with Torch Compile enabled and zero encoder fallback/capture failure.
It completed 700/700 exact10 requests, but corpus WER was 1.4464 rather than
the established approximately 0.016 baseline; decoded hypotheses contained
invalid mixed-script/replacement-character output. Its p95 of 2.142 seconds
and 37.89 requests/s are invalid as performance evidence because correctness
failed. A subsequent server-local source edit removed `fullgraph=True` and set
`mode="reduce-overhead"`; outputs became variable but remained corrupt. The
claim that an opaque custom op inherently forces a graph break under
`fullgraph=True` is not established: a registered custom op is intended to be
an opaque legal graph node. The edit is rejected as a repair and must not be
committed or used by another run.

`910C-032` deliberately combines the remaining high-information correctness
checks into one server interaction. It is diagnostic, not a waiver of Torch
Compile. Use Git only:

- SGLang-Omni: fetch `origin/qwen3-asr-910b-opt` and check out the exact
  handoff commit containing this task;
- SGLang: discard the explicitly identified server-local
  `npu_graph_runner.py` experiment, fetch
  `origin/codex/qwen3-asr-torch-compile`, and check out exact commit
  `5403d1f7dad0692b5fdd26c1dafc4c8192935303` in a clean worktree;
- keep the installed `sgl-kernel-npu` unchanged and record its version/import
  path. Do not rebuild or reinstall it.

The operator has explicitly authorized discarding only the reported dirty
`npu_graph_runner.py` experiment. If any other tracked or untracked change is
present, stop and report it; do not reset, stash, delete, or overwrite it.
Recheck the headless-OpenCV invariant, process/port/device health, and baseline
HBM before each service. Run the prior focused/full suites plus both new
parameterized value tests in `test_npu_fused_ops.py`. The new tests compare the
unchanged external fused kernel with the opaque custom op after
`torch.compile(..., backend=npugraph_ex, fullgraph=True, dynamic=False)` at
batch 1 and 2. Require bit-identical q/k/v tensors. If either test fails,
retain its max/mean absolute difference and first mismatching output, stop
before service startup, and return the parity failure as the first blocker.

If the kernel-level parity tests pass, run the following fresh-service arms on
the same deterministic 70-warmup/140-measured exact10 subset. Accuracy and
pairwise-output comparison, not hard-target performance, are the gate. Finish
all six arms unless a service has a device error, no completion progress for
90 seconds, or cannot cleanly return HBM to baseline:

| Arm | Encoder graph | Prefill graph | Decode graph | Torch Compile | Compile max batch |
|---|---:|---:|---:|---:|---:|
| `T1` | off | off | on through 70 | on | 1 |
| `T2` | off | off | on through 70 | on | 2 |
| `T70` | off | off | on through 70 | on | 70 |
| `A0` | on | on | on through 70 | off | not applicable |
| `A1` | on | on | on through 70 | on | 1 |
| `A2` | on | on | on through 70 | on | 2 |

For every arm, verify the resolved compile/capture bucket lists, positive
execution for the enabled paths, zero unexpected fallback/capture growth, and
request accounting for all 140 measured requests. `A0` is the accuracy control
and cannot satisfy the all-feature goal. Saturate the bounded encoder signature
cache before measuring `A0/A1/A2`, and require stable 8/8 counts with zero
eager fallback. Run each arm with identical sample ordering, generation
parameters, tokenizer, and seed. Preserve per-sample results server-local and
return only sanitized aggregates.

Report, for each arm: valid/evaluated counts, corpus WER, Unicode replacement
or non-expected-script count, latency and throughput as diagnostic context,
compiled and non-compiled bucket replay deltas, and cleanup. Compute pairwise
normalized-hypothesis equality against `A0`: exact-match count, first-divergent
token-position histogram, and whether divergence begins while the active raw
decode batch is 1, 2, or above 2. Do not return transcripts or request IDs.

Interpret the matrix without another server edit:

- a kernel parity failure assigns the next local repair to the opaque fused-op
  boundary or TorchAir lowering of that op;
- bad `T1` assigns the defect to the common batch-1 compile path, independent
  of encoder/prefill graphs;
- good `T1` but bad `T2` assigns it to compiled batch 2;
- bad `T1/T2` but good `T70` indicates a compiled/non-compiled bucket
  transition or mixed-artifact cache defect;
- good `T1/T2/T70` but bad `A1/A2` proves an ALL-only state/buffer interaction;
- good `A0` with bad `A1/A2` reconfirms Torch Compile as the only accuracy
  variable, while a bad `A0` invalidates this matrix and reopens the combined
  graph baseline.

Use graceful shutdown for every arm and require two stable HBM snapshots at or
below 5%. The isolated agent must make zero source, test, configuration-file,
documentation, package, site-package, or commit changes during `910C-032`.
Runtime command-line values declared by the table are authorized; editing a
profile file is not.

### `910C-032` result and Torch Compile stage-localization task `910C-033`

`910C-032` used clean SGLang-Omni `9c27cca9` and SGLang
`5403d1f7dad0692b5fdd26c1dafc4c8192935303`. The two new NPU value-parity
cases passed, along with the decode-runner and declared Omni suites. The
external `sgl-kernel-npu` split-QKV/RMSNorm/RoPE implementation, its registered
opaque custom-op boundary, and the `npugraph_ex`-compiled invocation were
bit-identical for both compiled batch sizes 1 and 2. Rebuilding or replacing
`sgl-kernel-npu` is therefore neither required nor authorized.

The subsequent deterministic exact10 accuracy matrix completed five meaningful
arms. `A0` (encoder, prefill, and decode graphs on; Torch Compile off) was
correct with WER 0.0183 and no garbled output. Every compile-enabled arm was
incorrect: `T1` WER 1.3923 with 70/70 garbled outputs, `T2` 1.3929 with 70/70,
`T70` 1.4198 with 70/70, and `A1` 1.4106 with 70/70. `A2` was not run because
`A1` already established the all-feature compile failure and `T1/T2/T70`
already excluded batch-size and encoder/prefill interactions. This proves the
fault lies beyond the split-QKV opaque op, is independent of compile batch size,
and is independent of encoder/prefill graph enablement. It does not yet prove
whether the remaining fault is the compile-safe fused-op state, Dynamo graph
semantics, or TorchAir lowering.

Local SGLang commit `a7b279da6b0e858cb074e5ba6e8e5f144c1f84d4` introduces an
explicit diagnostic-only environment selector in `patch_model_npu`, with CPU
tests for its normal and diagnostic dispatch. It preserves production behavior
when unset:

- `SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC=prepared-eager` enters the identical
  compile-safe fused-op context but calls the uncompiled `model.forward`;
- `SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC=dynamo-eager` keeps
  `torch.compile(fullgraph=True, dynamic=False)` but uses PyTorch's `eager`
  backend instead of TorchAir `npugraph_ex`.

`910C-033` is a two-arm numerical-localization task, not a performance run.
Use Git only: fetch and check out the exact handoff commit in SGLang-Omni and
SGLang commit `a7b279da6b0e858cb074e5ba6e8e5f144c1f84d4`; leave the installed
`sgl-kernel-npu` unchanged. The server must make no source, test, configuration,
documentation, package, site-package, or commit changes. Before startup, require
clean tracked worktrees, the existing headless OpenCV invariant, no holder or
service process, port availability, and two stable HBM snapshots at or below
5%. Run the prior SGLang fused-op/decode-runner suites, the new patch-model
dispatch tests, the Omni focused suites, and the complete Qwen3-ASR suite. Stop
before hardware on the first test collection or test failure.

For each arm, use a fresh service process and the `T1` profile from `910C-032`:
encoder graph off, prefill graph off, decode graph enabled through bucket 70,
Torch Compile enabled with compile maximum batch 1, all other model, corpus,
seed, sampling, memory, worker, timeout, and exact10 settings unchanged. Use
the same deterministic 70 warmup and 140 measured exact10 inputs, require all
140 completions, positive decode-graph replay, zero unexpected eager fallback,
and graceful cleanup. Do not start `ALL`, C70, realtime, package actions, or
any third arm.

| Arm | Sole additional environment value | Required positive marker |
|---|---|---|
| `D0` | `SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC=prepared-eager` | warning that compile-safe dispatch ran without `torch.compile` |
| `D1` | `SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC=dynamo-eager` | warning that `fullgraph` Dynamo ran with backend `eager` |

For both arms return only sanitized aggregate evidence: completion/accounting,
WER, garbled-output count, normalized equality against the recorded correct
`A0` control, decode replay/eager counts, forbidden-signature counts, the
diagnostic marker, and cleanup state. Retain raw audio, text, request IDs, and
logs server-local. Interpret outcomes mechanically:

- bad `D0`: the defect is in the compile-safe fused-op/context state before
  Dynamo or TorchAir; the next local repair must inventory that state;
- good `D0`, bad `D1`: the defect is in Dynamo's full-graph handling of the
  remaining model path or custom-op alias/state contract;
- good `D0` and `D1`, with the recorded bad `T1`: the defect is isolated to
  TorchAir `npugraph_ex` lowering/runtime; retain the exact compiler config and
  first divergence evidence for the framework-owned repair;
- a clean `D0`/`D1` result does not qualify Torch Compile or permit a
  performance claim. The normal `npugraph_ex` path must be repaired, then
  requalified through TC and `ALL`.

### `910C-033` result and NPU TopK dispatch-repair task `910C-034`

`910C-033` completed both requested numerical-localization arms with the
declared suites passing. `D0` (prepared-eager) and `D1` (Dynamo with the eager
backend) each completed 140/140 requests but produced garbled output for every
request, with WER 1.3929. This excludes both Dynamo and TorchAir as the first
cause: the fault is active while compile-safe fused-op dispatch is selected.
It does **not** prove that `leave_torch_compile()` fails, because the wrong
output was generated while the context was still active. The installed
`sgl-kernel-npu` remains value-parity qualified and must not be rebuilt.

Local SGLang commit `3c389d2f1` changes only `TopK`'s NPU compile policy. The
generic batch-one compile policy previously replaced NPU `TopK.forward_npu`
with `forward_native`. On NPU, the existing route is composed of
`torch.ops.npu` custom-op boundaries and is valid for the outer Dynamo trace;
it also preserves the eager routing contract. The native replacement mutates
`TopKConfig.torch_native` and is the first identified compile-safe state change
that can explain the `D0` failure. The new focused unit test asserts that NPU
keeps its existing dispatch for compile batch sizes one and two while the
non-NPU policy is unchanged. This is a bounded hypothesis repair, not a claim
that all fused-op state has been exhaustively cleared.

`910C-034` is a conditional, correctness-only task. Use Git only: fetch the
SGLang-Omni branch at the handoff commit containing this task and fetch
`origin/codex/qwen3-asr-torch-compile` at exact commit `3c389d2f1`. Do not
modify source, tests, configuration files, documentation, packages,
site-packages, commits, or the installed `sgl-kernel-npu` package. Before each
fresh service process, require clean tracked worktrees, the headless-OpenCV
invariant, no holder/service process, port availability, and two stable HBM
snapshots at or below 5%. If an NPU driver holder remains from `910C-033`, stop
and request operator cleanup; do not use `SIGKILL`, reboot, or restart a driver.

1. Run the SGLang TopK dispatch test, prior fused-op/decode-runner tests, Omni
   focused encoder/model-info tests, and the complete Qwen3-ASR suite. Stop
   before service startup on any collection or test failure.
2. In a fresh `D0-R` service, repeat the exact `910C-033` `D0` profile:
   encoder and prefill graphs disabled, decode graph through bucket 70, compile
   maximum batch one, and
   `SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC=prepared-eager`. Run the fixed 70-item
   warm-up and 140 exact10 measured requests. Require all completions, zero
   garbled outputs, WER no worse than the accepted compile-off control plus
   0.01 absolute, positive decode replay, zero unexpected eager fallback, and
   the diagnostic marker. On any failure, stop; do not start later arms.
3. Conditional on `D0-R` passing, use a fresh `T1-R` service with the same
   profile but without the diagnostic environment value, thus exercising normal
   `npugraph_ex` compile. Require the same accuracy/accounting criteria plus
   all 13 decode buckets captured and zero historical Dynamo, ATB, ACL/GE, OOM,
   or graph-fallback signatures. Stop on failure.
4. Conditional on `T1-R` passing, use a fresh `A1-R` service with encoder,
   prefill, decode, and Torch Compile all enabled; retain the execution guard
   and the deterministic encoder-signature warm-up already required by
   `910C-031`. Run the same 140-request correctness set. Require zero encoder
   capture failures/fallbacks, positive replay on all enabled graph paths,
   normal output quality, and clean drain. This is not a C70 performance run.

For every completed arm return sanitized completion/error accounting, WER,
garbled-output count, graph capture/replay/eager counters, TopK focused-test
result, forbidden-signature counts, and teardown state. Retain transcripts,
audio, paths, request IDs, and logs server-local. A successful `A1-R` closes
the Torch Compile functional-correctness blocker only; exact10 C70 performance
is separately authorized after it.

### `910C-034` result and NPU norm/activation dispatch-repair task `910C-035`

`910C-034` passed all declared tests but `D0-R` remained garbled 70/70 with
WER 1.4220. The conditional normal-compile and ALL arms did not start. This
rejects the `TopK` dispatch hypothesis and leaves the first fault in another
compile-safe fused-op switch. It does not weaken the earlier conclusion that
the fault predates Dynamo and TorchAir.

The local audit of the Qwen3-ASR decoder is now complete. Its active generic
`BaseFusedOp` instances are decoder RMSNorms and MLP `SiluAndMul`; Q/K norm and
RoPE are already bypassed by the separately value-qualified opaque packed-QKV
custom-op boundary. Local SGLang commit `9438420a6` makes the smallest remaining
NPU-specific repair: `RMSNorm` retains `torch_npu.npu_rms_norm` /
`npu_add_rms_norm`, and `SiluAndMul` retains `torch_npu.npu_swiglu`, while the
outer compiler traces their `torch.ops` boundaries. Non-NPU behavior remains
the generic native-reference policy. The commit includes CPU policy tests for
both NPU retention and non-NPU preservation. This is deliberately scoped to
the real Qwen3-ASR decoder inventory, not a global exemption for all fused ops.

`910C-035` is another conditional correctness task, not a performance task.
Use Git only: fetch the SGLang-Omni handoff commit containing this task and
fetch `origin/codex/qwen3-asr-torch-compile` at exact commit `9438420a6`. Keep
the installed `sgl-kernel-npu` unchanged. The server must make zero source,
test, configuration, documentation, package, site-package, or commit changes.
Require clean tracked worktrees, headless OpenCV, no holder/service process,
port availability, and two stable HBM snapshots at or below 5% before every
service. If a residual driver holder exists, stop and request operator cleanup.

1. Run both new NPU dispatch-test files, the prior fused-op/decode-runner
   suites, Omni focused encoder/model-info suites, and the complete Qwen3-ASR
   suite. Stop before startup on the first collection or test failure.
2. In fresh `D0-R2`, repeat `D0-R` exactly: encoder/prefill graphs disabled,
   decode graph through bucket 70, compile maximum batch one, and
   `SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC=prepared-eager`; run its fixed warm-up
   and 140 exact10 correctness requests. Require all completions, no garbled
   output, WER within 0.01 absolute of `A0`, positive decode replay, and zero
   unexpected eager fallback. Stop on failure.
3. Conditional on `D0-R2` passing, run fresh normal `T1-R2` with the same
   profile but no diagnostic variable. Require all 13 capture buckets, normal
   accuracy, positive replay, and zero historical Dynamo/ATB/ACL/GE/OOM/fallback
   signatures. Stop on failure.
4. Conditional on `T1-R2` passing, run fresh fully enabled `A1-R2` with the
   existing guard and deterministic encoder-signature warm-up. Run the same
   140-request correctness set; require all graph-path replay markers, zero
   encoder capture failures/fallbacks, normal accuracy, and drain. Do not run
   C70, realtime, or a performance campaign.

Return sanitized test counts and per-arm accounting, WER, garbled-output count,
graph counters, forbidden signatures, and cleanup state. Keep logs, paths,
audio, transcripts, request IDs, and model details server-local.

### `910C-035` result, superseded compile-context task `910C-036`, and overnight scoped-state matrix `910C-037`

`910C-035` passed its declared tests but `D0-R2` remained garbled 70/70 at WER
1.4140. The conditional `T1-R2` and `A1-R2` arms correctly did not start. This
rejects the Qwen3-ASR decoder's remaining generic fused-op dispatch switches as
the direct cause. More importantly, the prior D0 interpretation was too broad:
the NPU runner applies `set_tc_piecewise_forward_context` independently from
`prepare_model_for_torch_compile`. That context sets
`use_decode_graph_attention=True`, changing decode attention to the graph-safe
path even in prepared-eager mode. D0 therefore did not isolate fused-op state
from graph-safe attention dispatch.

`910C-036` is superseded before server execution. The initial local selector
`context-eager` was useful, but a single arm would still leave an unnecessary
round trip: the outer Qwen3-ASR model prepares both `audio_tower` and
`language_model` fused-op instances. Server agents must use the expanded,
local-reviewable SGLang commit `3295b12d3`, not `e45d64c9f`.

`3295b12d3` adds diagnostic-only selectors, all inert when the environment
variable is unset:

| selector | compile-safe fused-op preparation | graph-safe decode-attention context | `torch.compile` |
| --- | --- | --- | --- |
| `context-eager` | none | retained | no |
| `audio-prepared-eager` | `audio_tower` only | retained | no |
| `language-prepared-eager` | `language_model` only | retained | no |

The scoped selectors apply and later reverse only matching `BaseFusedOp`
instances, using their stable module paths. They do not alter production
behavior, package versions, or the installed `sgl-kernel-npu`.

#### `910C-037`: overnight Torch-Compile correctness localization matrix

This is an explicitly authorized diagnostic matrix, designed to avoid another
human round trip. Use Git only: fetch this SGLang-Omni handoff commit and
SGLang `origin/codex/qwen3-asr-torch-compile` at exact commit `3295b12d3`.
The server must make zero source, test, configuration, documentation, package,
or site-packages changes. Do not rebuild or reinstall `sgl-kernel-npu`.

Before any service startup, require clean tracked worktrees, the headless
OpenCV invariant, no device holder or service process, an available port, and
two HBM snapshots at or below 5%. A failed infrastructure preflight, test
collection failure, or test failure stops the entire matrix and is returned as
the first blocker; do not use `kill -9`, restart a driver, or reboot. Run the
NPU dispatch tests, fused-op tests, decode-runner tests, Omni focused
encoder/model-info tests, and the full Qwen3-ASR suite. Return exact test
counts and the first failure, if any.

For every matrix arm, use a fresh service process and the exact `D0-R2`
correctness profile: encoder graph disabled, prefill graph disabled, decode
graph enabled through bucket 70, the Torch-Compile feature flag enabled so
the NPU runner supplies its normal graph-safe attention context, 70-item
warm-up, then 140 exact10 measured requests. The sole arm-specific change is
`SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC`. Require its positive startup marker,
complete request accounting, normal drain, normal graceful shutdown, and two
post-stop HBM snapshots at or below 5%.

Run in this exact order:

1. `C0-context-eager` with `context-eager`.
2. If and only if C0 is correct, run `A0-audio-prepared-eager` with
   `audio-prepared-eager`.
3. If and only if C0 is correct, also run `L0-language-prepared-eager` with
   `language-prepared-eager`, even if A0 is garbled. A0 and L0 are independent
   diagnosis arms, so an accuracy failure in either must be recorded but does
   not suppress the other.

Do not run normal Torch Compile, ALL, C70 performance, realtime, a package
change, or an unlisted experiment. For every started arm return only sanitized
aggregate evidence: WER, garbled-output count, response accounting, capture
and replay/eager counters, forbidden signatures, selector marker, and cleanup
state. Keep transcripts, audio, raw request IDs, and logs server-local.

Interpret the completed matrix mechanically:

- garbled C0: the graph-safe decode-attention context is necessary for the
  corruption. The next local repair is an NPU numerical-parity investigation
  of the normal attention path versus `forward_decode_graph` /
  `unified_attention_with_output`; do not audit more generic fused ops.
- correct C0 plus garbled A0 only: audit actual audio-tower fused-op state
  transitions and parity (starting with encoder convolution/norm paths).
- correct C0 plus garbled L0 only: audit remaining language-model fused-op
  state transitions and parity, including rotary/q-k normalization context;
  existing split-QKV kernel parity is not enough.
- correct C0 plus both A0 and L0 correct while all-prepared D0 remains bad:
  audit global or ordering interaction in `_to_torch`, including backend
  initialization and enter/leave ordering, before another numerical fix.
- any capture/startup failure: report its first signature as a separate capture
  blocker and make no accuracy or performance claim.

`910C-037` returned before A0/L0: `C0-context-eager` was garbled. It proves
that the full graph-safe decode-attention route is necessary for the observed
corruption, but does **not** yet prove that `AscendAttnBackend.forward_decode_graph`
is itself numerically wrong. The compile-off decode graph route already reaches
that backend successfully; C0 adds the `unified_attention_with_output` custom-op
wrapper, including temporary ForwardBatch narrowing and an output-buffer copy.
A0/L0 must not start because their conditional entry criterion (a correct C0)
was not met.

#### `910C-038`: direct graph-attention wrapper-isolation arm

Local SGLang commit `8ec282120` adds the opt-in diagnostic selector
`SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC=direct-graph-eager`. It retains the NPU
graph backend and Torch-Compile runner feature flag, but makes the TC context
skip the `unified_attention_with_output` custom-op route. The NPU backend then
selects its existing `forward_decode_graph` path directly, just as the known-
correct compile-disabled decode graph does. It invokes neither fused-op prepare
nor `torch.compile`; the selector is process-scoped and inert when unset.

Use Git only: fetch this handoff commit and SGLang
`origin/codex/qwen3-asr-torch-compile` at exact commit `8ec282120`. The server
must make zero source, test, configuration, package, or site-packages changes;
do not rebuild or reinstall `sgl-kernel-npu`. Repeat the same clean-worktree,
headless-OpenCV, port, device-holder, and two-HBM-snapshot preflight. Run the
new focused runner test, the preceding NPU dispatch/fused-op/decode-runner
suites, Omni focused suites, and the full Qwen3-ASR suite. A preflight,
collection, or test failure stops the arm and is returned as the first blocker.

Run one fresh `W0-direct-graph-eager` service using the exact C0/T1 correctness
profile: encoder and prefill graphs disabled; decode graph enabled through
bucket 70; the Torch-Compile feature flag enabled; 70-item warm-up; then 140
exact10 measured requests. Set only the new diagnostic selector. Require its
positive marker, capture success, complete request accounting, normal drain,
normal graceful shutdown, and two post-stop HBM snapshots at or below 5%.
Return sanitized WER, garbled-output count, graph capture/replay/eager counts,
forbidden signatures, selector marker, and cleanup evidence. Do not start
normal compile, ALL, C70 performance, realtime, package changes, or another
unlisted experiment.

Interpret W0 mechanically:

- correct W0: the custom-op wrapper is necessary for corruption. The next local
  repair must compare and correct its ForwardBatch slicing, output ownership,
  cache-location mutation, and padded-tail behavior; `direct-graph-eager` is a
  diagnostic only and must not become the production workaround.
- garbled W0: the failure remains after bypassing the wrapper. The next local
  diagnosis must compare direct NPU graph attention versus normal NPU attention
  with isolated KV cache state, focusing on BSH/BSND layout, sequence-length
  metadata, and cache writes.
- capture/startup failure: report the first signature as a distinct capture
  blocker and make no attention-correctness or performance claim.

`910C-038` completed with a correct W0 result: WER 0.0183 and 0/70 garbled
outputs. This localizes the defect to the custom-op wrapper, not the terminal
NPU `forward_decode_graph` backend. The direct diagnostic remains forbidden as
a production workaround because it removes the custom-op boundary required by
normal Torch Compile.

#### `910C-039`: NPU decode-attention wrapper repair qualification

Local SGLang commit `54a8d042d` repairs the NPU TC decode branch inside
`_unified_attention_with_output_impl`. It retains the registered custom-op
boundary, but passes the original static graph Q/K/V tensors and ForwardBatch
state straight to `forward_decode_graph`. It no longer applies generic PCG
query/KV slicing, temporary cache-location or position replacement,
`_attn_output` replacement, or padded-tail clearing to that branch. The
special Prefix-MHA/LSE cases retain the prior generic behavior pending their
own parity qualification. A CPU-focused test asserts that NPU decode-graph
dispatch preserves the static tensors and all ForwardBatch identities.

Use Git only: fetch this handoff commit and SGLang
`origin/codex/qwen3-asr-torch-compile` at exact commit `54a8d042d`. The server
must make zero source, test, configuration, package, or site-packages changes;
do not rebuild or reinstall `sgl-kernel-npu`. Before running, require clean
tracked worktrees, headless OpenCV, no holder/service process, a free port, and
two HBM snapshots at or below 5%. A failed preflight, collection, or test
failure stops all arms and returns the first blocker. Run the new Radix
attention focused test, decode-runner/NPU dispatch/fused-op tests, Omni focused
suites, and the full Qwen3-ASR suite.

Use a fresh service process for each arm and leave
`SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC` unset. This is a real `torch.compile`
repair qualification, not a diagnostic selector:

1. Run `T1-fixed`: encoder and prefill graphs disabled, decode graph enabled
   through bucket 70, normal Torch Compile enabled, then 70-item warm-up and
   140 exact10 measured requests. Require the actual compile marker, successful
   decode capture, complete accounting, zero garbled output, normal accuracy,
   no forbidden signature, and normal cleanup.
2. Only if T1-fixed is correct, run fresh `A1-fixed`: encoder, prefill, decode
   graph, execution guard, and normal Torch Compile all enabled, with the same
   warm-up and 140-request exact10 correctness workload. Require positive
   markers for all enabled graphs, zero encoder capture failure/fallback, zero
   garbled output, normal accuracy, complete drain, and cleanup. Record encoder
   signature count/capacity and any measurement-period growth, but do not make
   a final performance claim from this arm.

For every started arm return sanitized test counts, WER, garbled-output count,
response accounting, compile/capture/replay/eager counters, encoder signature
state, forbidden signatures, and cleanup. Do not run C70 performance, final
three-repeat performance, realtime, a package change, or another unlisted
experiment. If T1 fails, do not start A1. If T1 passes and A1 fails, report A1
as the first ALL-combination blocker.

`910C-039` passed `T1-fixed`: real normal Torch Compile with encoder and
prefill graphs disabled returned WER 0.0167 with 0/70 garbled outputs. The
wrapper repair is therefore correct for the TC-only path. `A1-fixed` remained
garbled at approximately WER 1.39. Because A1 simultaneously enables encoder
graph, prefill graph, and the encoder/generation guard, it does **not** by
itself prove an encoder-only interaction.

#### `910C-040`: real-compile feature-combination split matrix

No new source change is required. Use the exact reviewed code pair from
`910C-039`: this handoff commit plus SGLang
`origin/codex/qwen3-asr-torch-compile` at `54a8d042d`; leave
`SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC` unset. The server must make zero source,
test, configuration, package, or site-packages changes. Do not rebuild or
reinstall `sgl-kernel-npu`.

After the same clean-worktree, headless-OpenCV, no-holder, free-port, and two
HBM-snapshot preflight, run the already-authorized focused and full suites. A
preflight, collection, or test failure stops the whole matrix. Each accuracy
arm uses a fresh service, normal `torch.compile`, decode graph through bucket
70, a 70-item warm-up, and 140 exact10 measured requests. Return WER,
garbled-output count, response accounting, compile/capture/replay/eager
counters, encoder signature state, forbidden signatures, and cleanup.

Run both independent arms even when the first has a garbled accuracy result:

1. `E1-encoder-compile`: encoder graph and execution guard enabled; prefill
   graph disabled; decode graph and normal Torch Compile enabled.
2. `P1-prefill-compile`: encoder graph disabled; prefill and decode graphs plus
   normal Torch Compile enabled. The encoder guard is absent by construction.

Do not run ALL again, C70 performance, final three-repeat performance,
realtime, a package change, or another unlisted experiment. Interpret results
mechanically:

- only E1 garbled: the blocker is the encoder-graph/guard plus compile
  interaction; next local work instruments encoder graph replay/update and
  compile-forward ordering.
- only P1 garbled: the blocker is prefill-graph plus compile interaction; next
  local work instruments the prefill PCG context and its transition to decode.
- both garbled: there are at least two independent graph-plus-compile
  compatibility defects; return both signatures before choosing a repair.
- both correct while A1 remains garbled: the fault requires the combined
  encoder and prefill state; next local work instruments their ordering and
  shared ForwardBatch/cache state.

#### `910C-041`: graph-to-compiled-decode transition matrix

`910C-040` completed both independent arms.  `E1-encoder-compile` and
`P1-prefill-compile` both produced garbled output, while the already accepted
T1 profile (compile plus decode graph only) remains correct.  This is evidence
of two feature combinations, but it is not yet evidence of two unrelated
defects: both combinations add a graph-produced state that is consumed by the
first compiled decode step.

Run this small, correctness-only transition matrix before making another
source change.  Use this handoff commit and SGLang
`origin/codex/qwen3-asr-torch-compile` at `54a8d042d`; the server must make no
source, test, configuration-file, package, site-package, or commit change and
must not rebuild `sgl-kernel-npu`.  The standard clean-worktree, headless
OpenCV, holder-free, free-port, and two stable-HBM-snapshot preflight applies.
Run the previously declared focused and full suites once before any service;
their first collection or test failure stops the entire task.

For each of the four rows below, start a fresh service, retain decode graph
through bucket 70, the fixed exact10 corpus, deterministic temperature/seed,
and the normal execution guard whenever encoder graph is enabled.  Submit the
same fixed, content-distinct 20-item probe set serially twice: first with the
documented transcription request field `max_new_tokens=1`, then with
`max_new_tokens=2`.  Do not score WER for the truncated probes.  Instead,
compare each normalized response with the matching compile-off control from
the same feature profile and return only aggregate equality counts,
garbled-output counts, request accounting, and graph replay/eager counters.
The server may use runtime request fields or an existing benchmark-client
option, but must not create or edit a helper source file.

| Arm | Encoder graph | Prefill graph | Torch Compile | Purpose |
|---|---:|---:|---:|---|
| `E0-1/2` | on | off | off | Encoder graph reference at one and two generated tokens |
| `E1-1/2` | on | off | on | Isolate encoder-graph to compiled-decode hand-off |
| `P0-1/2` | off | on | off | Prefill graph reference at one and two generated tokens |
| `P1-1/2` | off | on | on | Isolate prefill-graph to compiled-decode hand-off |

Run all four arms even if an earlier compile-on arm diverges; they are
independent and the goal is to avoid another server round trip.  Do not run
ALL, C70, performance, realtime, package actions, or any unlisted experiment.
Always use graceful shutdown and require two post-stop HBM snapshots at or
below 5% before the next service.

When multiple idle NPUs are available, the four service arms may run in
parallel, with exactly one service process assigned to each physical NPU.  The
operator must record the physical card/chip mapping, use disjoint ports and
evidence directories, and perform the complete preflight and two post-stop HBM
snapshots independently for every assigned NPU.  Do not place two arms on one
NPU, share a service process across arms, or compare raw latency across cards;
this task compares only each compile-on response with its same-profile,
same-card compile-off control.  A card-level health, holder, startup, request,
or cleanup failure invalidates only its arm, but its matching control and
treatment must be rerun together on another clean card before drawing a
numerical conclusion.

Interpret the results mechanically:

- a compile-on mismatch already at one token locates that feature's defect at
  encoder/prefill output production, input-embedding injection, or the
  prefill logits/KV write itself, before a completed decode transition;
- one-token equality followed by a two-token mismatch locates it at the
  feature's state hand-off to the first compiled decode forward (KV/cache,
  static input/output ownership, or ForwardBatch mutation);
- equality at both limits contradicts the reported 140-item failure and
  requires retaining the per-request result categories before widening the
  probe; it does not authorize a performance run;
- independent E and P outcomes select separate minimal local repairs.  A
  common one-token or two-token boundary is a shared-lifecycle hypothesis to
  audit before duplicating fixes.

#### `910C-042`: serial stage-performance attribution before accuracy closure

Accuracy remains a hard gate for acceptance, but it does not prevent
diagnostic performance attribution.  No result from a garbled compile-on arm
is an acceptance, hard-target, or final-candidate measurement.  This task is
still authorized, but all future server services and arms must run serially;
the earlier multi-NPU parallel authorization is superseded because it proved
operationally unreliable on the isolated server.

Use this handoff commit and SGLang `54a8d042d`, with the same immutable source,
package, kernel, corpus, and server constraints as `910C-041`.  Enable the
existing request/encoder/guard recorder, but do not enable per-decode INFO
logging because its volume would perturb the measurement.  Freeze one
content-distinct 70-item warm-up and the 700-item exact10 C70 workload.  For an
encoder-graph arm, finish deterministic signature saturation before timed
measurement and require signature count/capture count not to grow while
timing.  Capture rich model-info immediately before and after measurement and
sample NPU resources at 0.5 seconds.  Each arm gets one fresh service; this is
a screening pass, not the final three-repeat campaign.

Use one verified-clean physical NPU.  Run every control and treatment as a
fresh service, one at a time, on that same NPU.  Between services require
graceful teardown, a free port, no residual holder, and two stable HBM
snapshots at or below 5%.  Keep disjoint event directories and benchmark
outputs.  Do not overlap service startup, warm-up, measurement, teardown, or
resource monitoring across arms.

| Same-card lane | Control | Treatment | Marginal question |
|---|---|---|---|
| `G` | decode graph, compile off | decode graph, compile on (`T1`) | cost/benefit of the now-correct TC-only path |
| `E` | encoder+decode graphs, compile off | encoder+decode graphs, compile on (`E1`) | encoder graph/guard interaction with compile |
| `P` | prefill+decode graphs, compile off | prefill+decode graphs, compile on (`P1`) | prefill graph interaction with compile |
| `A` | encoder+prefill+decode graphs, compile off (`A0`) | all graphs+compile (`A1`) | complete-stack contention and utilization |

Return sanitized aggregates for every arm, even when compile-on accuracy is
bad:

- request completion/error accounting, WER, garbled count, generated/decode
  step count, and end-to-end latency/throughput;
- encoder item count, batch count and histogram, average/max batch occupancy,
  queue-wait p50/p95/max, execute p50/p95/max, graph capture/replay/fallback,
  and signature counts before/after;
- prefill forward count and elapsed p50/p95/max, prefill graph replay/eager
  counts and bucket histogram;
- decode forward count and elapsed p50/p95/max, elapsed per completed decode
  step, decode graph replay/eager counts and bucket histogram;
- execution-guard wait and hold p50/p95/max, acquisition count and balanced
  wait/acquire/release counts;
- NPU utilization, AI Core utilization, HBM and power mean/max, plus service
  drain and cleanup state.

Compare end-to-end latency or request throughput across an arm pair only when
their completion and generated/decode-step distributions are sufficiently
similar.  Otherwise compare prefill time, encoder time, guard time, and
per-decode-step time; explicitly label end-to-end figures output-length
contaminated.  Never use the bad-output arm to claim the 500 ms target.

The required optimization decisions from this screening are:

1. quantify how much of p95 is encoder queue/execute versus guard wait versus
   prefill versus repeated decode;
2. calculate encoder batch occupancy (historically only about three items per
   batch against capacity eight) and decide whether a bounded batch-wait sweep
   is the first throughput experiment;
3. determine whether the coarse guard dominates and whether the next local
   design should protect only NPU graph input update/device submission with
   explicit stream/event ordering;
4. report compiled-bucket coverage, since compiling only batch sizes 1 and 2
   cannot establish the performance value of compile at C70;
5. use measured stage lower bounds to assess whether 140 requests/s and p95
   below 500 ms are feasible on one card before spending time on soak/repeats.

Do not tune parameters during these arms, run realtime, perform a soak, or
change source/packages.  Parameter optimization is a subsequent, bounded
serial sweep selected from these stage results.

#### `910C-043`: NPU guard completion-fence transition probe

`910C-041` found the same token boundary in both failing combinations: E1 and
P1 match their compile-off controls at `max_new_tokens=1` and diverge at
`max_new_tokens=2`.  Treat this as a shared graph-to-first-compiled-decode
lifecycle hypothesis, not yet as two unrelated numerical defects.  Local code
commit `b6966d4d` adds an opt-in completion callback to
`FairDeviceExecutionGuard`.  On NPU, setting
`SGLANG_OMNI_NPU_GUARD_COMPLETION_FENCE=1` makes the guard synchronize device
work before handing its FIFO ticket to the next encoder or generation owner.
The default remains unchanged.  This is a diagnostic correctness fence, not a
performance implementation: it intentionally destroys asynchronous overlap.

Use Omni `b6966d4d` and SGLang `54a8d042d`.  The isolated operator must make no
source, test, configuration-file, package, site-package, kernel, or commit
change.  Re-run the declared focused suites plus the complete Qwen3-ASR suite;
the first collection or test failure stops this task.  Require the startup
warning `NPU guard completion fence enabled` before sending a request.  A
missing marker invalidates the arm.

Run these two treatment arms independently and in parallel on two clean NPUs,
using disjoint ports and evidence directories:

| Arm | Encoder graph | Prefill graph | Decode graph | Compile | Fence |
|---|---:|---:|---:|---:|---:|
| `E1-F` | on | off | on | on | on |
| `P1-F` | off | on | on | on | on |

Reuse the exact `910C-041` deterministic 20-item probe, seed and request
settings.  It is sufficient to run `max_new_tokens=2`; compare normalized
outputs against the already preserved matching E0/P0 results.  Return request
accounting, equality and garbled counts, WER only if the untruncated request is
also run, encoder/prefill/decode graph counters, guard event balance, the fence
startup marker, and clean drain/teardown evidence.  Do not run C70, report
performance from a fenced arm, or substitute a global environment setting for
the per-service variable above.  `910C-042` may continue concurrently on other
NPUs, but its services and artifacts must remain independent.

Interpret the pair mechanically:

- both arms correct: missing NPU completion/visibility at the guard hand-off is
  the shared cause; replace the coarse synchronization with producer-event to
  consumer-stream ordering before performance qualification;
- only E1-F correct: encoder graph completion ordering is confirmed, while the
  prefill-to-decode transition needs a separate state/stream repair;
- only P1-F correct: prefill graph completion ordering is confirmed, while the
  encoder-to-generation transition needs a separate state/stream repair;
- neither correct: reject the completion-only hypothesis and next compare
  graph-produced KV/static-buffer ownership at the first decode call.

#### `910C-044`: explicit KV-state compile boundary and conditional ALL screen

The explicit KV-state change in SGLang `634303cdf` did not repair correctness.
The TC-only control regressed, and the new path also introduced a warm-up hang,
so the remaining arms and conditional performance screen were not valid.
Reject that implementation rather than building further diagnostics on it.
SGLang `ca17cd413` reverts `634303cdf` and restores the accepted
`54a8d042d` behavior.  No server-side source change was made.

#### `910C-045`: serial graph-capture-only transition isolation

The token boundary remains precise: TC-only is correct, encoder+TC and
prefill+TC match their controls for one generated token, and both diverge on
the first compiled decode transition.  Completion fencing and explicit KV
operands are rejected.  The next discriminator is whether graph
initialization/capture alone contaminates compiled decode state, or whether an
actual encoder/prefill graph replay is required.

Use Omni `8dab0b8f` and SGLang `5cb571995`.  The latter includes revert
`ca17cd413`; do not test `634303cdf`.  Run on one clean physical NPU, with one
fresh service at a time.  All arms from `910C-045` onward are serial: do not
start a second service, benchmark client, resource monitor, or cleanup on
another NPU concurrently.  Between arms require graceful shutdown, a free
port, no residual holder, and two HBM snapshots at or below 5%.

Before the arms, run the SGLang radix-attention/decode-runner suites, the Omni
encoder/model-info suites, and the complete Qwen3-ASR suite once.  Stop on the
first collection or test failure.  Keep server packages, kernels, exact10
corpus, deterministic request settings, and all unrelated profile values
unchanged.  The isolated operator must not edit source, tests, configuration
files, packages, site-packages, kernels, or Git history.

Run these arms sequentially, each with the exact `910C-041` 20-item
`max_new_tokens=2` probe followed by the content-distinct 70-item untruncated
accuracy set:

| Order | Arm | Encoder graph | Prefill graph | Decode graph | Compile | Diagnostic |
|---:|---|---:|---:|---:|---:|---|
| 1 | `E-CAP` | capture only | off | on | on | `SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_ONLY=1` |
| 2 | `P-CAP` | off | capture only | on | on | `SGLANG_NPU_PREFILL_GRAPH_CAPTURE_ONLY=1` |

For `E-CAP`, require positive encoder captured-graph/signature counts,
`replay_count=0`, and a positive `diagnostic_capture_only` eager-fallback
reason.  The diagnostic intentionally captures each real NPU encoder
signature and then executes the full encoder eagerly.  For `P-CAP`, require
positive prefill graph capture/startup attestation, zero request-time prefill
replay delta, and a positive standard-eager delta.  The diagnostic leaves
startup capture enabled but routes request prefill through eager execution.
Both arms retain normal decode graph replay and Torch Compile markers.

Return request accounting, normalized equality/garbled counts, WER, compile
buckets, encoder capture/signature/replay/fallback reasons, prefill
replay/eager deltas, decode replay/eager/buckets, guard balance, forbidden
signatures, drain, and cleanup evidence.  A missing capture-only marker or a
positive replay delta for the bypassed graph invalidates its arm.

Interpret each arm independently:

- correct capture-only output means actual graph replay is necessary for that
  feature's corruption; next compare the replay-produced output and first
  compiled-decode inputs without changing capture;
- garbled capture-only output means graph initialization/capture is sufficient
  to contaminate later compiled decode state; next audit capture ordering,
  persistent module/context mutation, and static buffer ownership;
- different E/P outcomes demonstrate different mechanisms; matching outcomes
  support one shared graph-lifecycle repair, but do not by themselves prove
  two independent defects.

Do not run ALL, C70, performance, soak, realtime, or any unlisted experiment
under these diagnostic bypasses.  The still-authorized `910C-042` performance
attribution is a separate serial task; do not overlap or mix its evidence with
the capture-lifecycle diagnostics below.

`910C-045` completed with both capture-only arms still garbled.  Encoder and
prefill replay counters remained zero for their bypassed paths, so an actual
auxiliary graph replay is not necessary for corruption.  This establishes
capture/init plus retention as sufficient, but it does not yet distinguish an
irreversible capture-time mutation from interference caused by the still-live
captured graph, graph pool, or static tensors.

#### `910C-046`: serial capture-release ownership isolation

The first attempt at handoff `e20cc11c` stopped before hardware because the
existing exact model-info assertion omitted the new release counter's default
value.  Test-only commit `b28013f0` adds the expected
`diagnostic_capture_release_count: 0`; runtime behavior is unchanged.  Use the
current handoff commit, which includes Omni code `0948859a`, test fix
`b28013f0`, and SGLang `1cd6be1b5`.  Run on one verified-clean NPU, one fresh
service at a time, with the same packages, kernels, exact10 corpus,
deterministic settings, and profile controls as `910C-045`.  Explicitly unset
the `910C-045` capture-only variables and keep the rejected completion fence
disabled.  The isolated operator must not edit source, tests, configuration
files, packages, site-packages, kernels, or Git history.

Run the SGLang prefill/decode-runner focused tests, the Omni encoder/model-info
focused tests, and the complete Qwen3-ASR suite once.  Stop on the first
collection or test failure.  Then run these two arms sequentially:

| Order | Arm | Encoder graph | Prefill graph | Decode graph | Compile | Diagnostic |
|---:|---|---:|---:|---:|---:|---|
| 1 | `E-REL` | capture then release | off | on | on | `SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_RELEASE=1` |
| 2 | `P-REL` | off | capture then release | on | on | `SGLANG_NPU_PREFILL_GRAPH_CAPTURE_RELEASE=1` |

Each arm runs the exact `910C-041` 20-item `max_new_tokens=2` probe followed by
the content-distinct 70-item untruncated accuracy set.  For `E-REL`, require a
positive `diagnostic_capture_release_count`, zero live captured graphs after
release, zero encoder replay, and a positive
`diagnostic_capture_released` fallback count.  Each real signature is captured
and released at most once; subsequent matching requests stay eager without
recapture.  For `P-REL`, require the startup warning `NPU prefill graph
capture-release diagnostic active`, normal prefill capture attestation before
that warning, eager request prefill, and normal decode capture/compile/replay.

Between arms require graceful shutdown, a free port, no residual holder, and
two HBM snapshots at or below 5%.  Return request accounting,
equality/garbled counts, WER, all diagnostic markers and counters, compile and
decode graph counters, forbidden signatures, guard balance, drain, and cleanup
evidence.

Interpret each arm independently:

- correct after release: a live auxiliary graph, pool, or retained static
  tensor is necessary for corruption; next isolate graph-pool identity and
  static output/input ownership while keeping the capture alive;
- still garbled after release: capture performs an irreversible process/module
  state transition; next snapshot fused-op dispatch, graph context globals,
  streams, allocators, module buffers, and compiler state immediately before
  and after capture;
- one arm recovers and the other does not: split the fixes by owner rather
  than assuming one shared lifecycle defect.

Do not run ALL, C70, performance, soak, realtime, or additional arms under the
release diagnostics.  `910C-042` remains separately authorized and serial;
its performance data must not be mixed with `910C-046`.

`910C-046 E-REL` completed with 2 of 20 probe outputs garbled after encoder
capture and immediate release.  The returned evidence does not include a
completed P-REL arm, so P-REL remains unclassified and must not be inferred
from E-REL.  This result also does **not** prove that
`prepare_model_for_torch_compile` mutated the encoder: the Qwen3-ASR encoder
capture path does not call that helper.  The reduction from the broadly
corrupt capture-retained result to 2/20 after release instead makes correlation
with the requests that actually triggered the two lazy signature captures the
next required check.

#### `910C-047` result and `910C-048` capture-state snapshot gate

`910C-047` completed: the parity diagnostic returned `allclose=True` (encoder
output is numerically identical immediately before and after capture+release),
yet the probe outputs were still garbled. This **disproves** the hypothesis
that `prepare_model_for_torch_compile` mutated the encoder (the encoder
capture path never calls it) and **disproves** encoder-output corruption.
The conclusion is narrower than the operator's summary: capture changes
downstream device/compile runtime state that only becomes visible in the
subsequent compiled decode, not in the encoder itself. The next gate must
attribute *which* runtime state transitions during capture.

`910C-048` instruments exactly that attribution.  It adds an opt-in
capture-state snapshot diagnostic (local commit `cf79b353` on the
`qwen3-asr-910b-opt` branch, files
`sglang_omni/models/qwen3_asr/encoder_cuda_graph.py`,
`sglang_omni/models/qwen3_asr/sglang_model.py`, and
`tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py`).  The diagnostic
splits the capture lifecycle into four state points — pre-warmup,
pre-graph-capture, post-graph-capture (after a `torch.cuda.synchronize`), and
post-release — and snapshots, on the first lazy capture only, under both
`SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_RELEASE=1` and
`SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_RELEASE_STATE_SNAPSHOT=1`:

- `fused_ops`: per `BaseFusedOp` module — type, `is_torch_compile`, and the
  resolved `forward` / `original_forward` / `compiled_native` callable labels;
- `tensor_metadata`: per parameter and buffer — shape, dtype, device,
  `data_ptr`, `_version`, `requires_grad`;
- `module_training`: per-module training flag;
- `runtime`: grad mode, diagnostic gates, `pynccl_allocator` graph-pool id,
  `memory_allocated`/`memory_reserved`, current stream type/handle, and
  `torch.cuda.is_current_stream_capturing()`; every NPU-runtime introspection
  degrades to an `"unavailable:<ErrorType>"` string rather than failing.

It logs the per-section bounded delta (`count` plus first 8 changed keys) for
the warmup, capture, and release transitions and exposes them in
`model_info.encoder_cuda_graph.diagnostic_capture_release_state`.
`memory_allocated`/`memory_reserved` deltas are dominated by the graph's
static-buffer allocation and are expected; the diagnostic value is in
fused-op dispatch flips, tensor `_version`/`data_ptr` changes, module
training-flag flips, and stream/pool identity changes.  The diagnostic is
inert unless both gates are set and runs at most once per service.

Local checks run: `git diff --check` clean; `py_compile` clean.  Local CPU
unit tests were **not** run because `sglang` is not installed in this local
environment; the focused suites are delegated to the server.

`910C-048` execution: use the same accepted Omni/SGLang stack and clean-NPU
environment as `910C-047`.  Run on one verified-clean NPU, one fresh service,
serial only.  Set `SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_RELEASE=1`,
`SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_RELEASE_PARITY=1`, and
`SGLANG_OMNI_ENCODER_GRAPH_CAPTURE_RELEASE_STATE_SNAPSHOT=1`; unset every
other capture bypass/release diagnostic and keep the rejected completion
fence disabled.  Before hardware, run
`tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py`, the encoder/model-info
focused set, and the complete Qwen3-ASR suite; stop on the first collection
or test failure.  Then run only the exact `910C-041` 20-item
`max_new_tokens=2` E1 probe.

Return the full
`model_info.encoder_cuda_graph.diagnostic_capture_release_state` object (the
three warmup/capture/release transition deltas), every
`encoder capture-state deltas` log line, the 20-request equality/garbled
count and WER, the parity object, capture-release/released-fallback/live-graph
counts, compile/decode counters, forbidden signatures, drain, graceful
cleanup, and two post-stop HBM snapshots.  Interpretation: whichever section
flips during the `capture` transition (but not `warmup`) and persists through
`release` is the state that the next repair must isolate or restore — fused-op
dispatch flip means a compiled forward was swapped permanently; tensor
`_version`/`data_ptr` change means an in-place mutation or rebind of a weight
or buffer; module training flip means a mode leak; stream/pool identity
change means capture installed a persistent stream or allocator context.  A
`capture`-only flip that resolves by `release` is not the persistent
contaminant.

#### `910C-048` result and `910C-049` graph-pool isolation repair gate

`910C-048` completed.  The four-point capture-state snapshot reported
`count=0` for every non-memory section (fused_ops, tensor_metadata,
module_training, runtime) across all three transitions (warmup, capture,
release).  Only `memory_allocated`/`memory_reserved` changed, as expected
from the graph's own static-buffer allocation.  This is a decisive negative
result: **no Python-observable state is mutated by encoder graph capture**.
The contamination therefore lives below the Python layer, in NPU driver
state.  Two earlier observations constrain what that driver state can be:

- `910C-045`: capture-only (never replayed) still garbles, so an actual
  encoder graph replay is not required; capture/init alone is sufficient.
- `910C-046 E-REL`: capture-then-release still garbles 2/20, so the
  contamination is not removed by dropping the live graph object; it is a
  persistent driver-side transition.
- `910C-047`: encoder output is numerically identical before and after
  capture+release (`allclose=True`), so the encoder is exonerated; the
  corrupted consumer is compiled decode.

The operator's handoff summary attributes this to
`prepare_model_for_torch_compile`.  That is **incorrect**: the function is
SGLang-internal, is never called by the omni encoder capture path (no
reference exists anywhere under `sglang_omni/`), and runs only during decode
graph capture.  The encoder capture path and the decode capture path share
exactly one mutable NPU driver resource: the **default graph memory pool**.
`encoder_cuda_graph.py` captures with
`torch.cuda.graph(graph, capture_error_mode="thread_local")` and passes no
`pool=`, so every encoder graph lands in the device's default graph pool,
the same pool the SGLang decode graphs use.  NPU graph capture installs
driver-side allocator and stream state scoped to that pool; capturing a
second graph into the shared pool can leave the pool's allocator bookkeeping
in a state that corrupts subsequent replay of the decode graphs already
resident in it.

`910C-049` repairs by isolating the encoder graphs into a dedicated private
graph pool.  The repair is a minimal local change to
`sglang_omni/models/qwen3_asr/encoder_cuda_graph.py`: allocate one
`torch.cuda.graph_pool_handle()` per runner, retain it for the runner's
lifetime, and pass it as `pool=` to `torch.cuda.graph(...)` in `_capture`.
Non-NPU platforms keep the existing default-pool behavior (the defect is
NPU-driver-specific).  This keeps every code path, bucket layout, and
diagnostic identical; the only variable is pool ownership.  Local CPU unit
tests remain delegated to the server (`sglang` absent locally).

Execution: use the same accepted Omni/SGLang stack and clean-NPU environment
as `910C-048`.  Run on one verified-clean NPU, one fresh service, serial
only.  **Unset** every capture bypass/release/state-snapshot diagnostic and
keep the rejected completion fence disabled; this is a normal-replay
correctness gate, not a bypass probe.  Before hardware, run
`tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py`, the encoder/model-info
focused set, and the complete Qwen3-ASR suite; stop on the first collection
or test failure.  Then run the exact `910C-041` 20-item `max_new_tokens=2`
E1 probe (encoder graph **on**, prefill off, decode graph on, compile on).

Return the 20-request equality/garbled count and WER, encoder
capture/signature/replay/fallback reasons, compile/decode graph counters,
forbidden signatures, guard balance, drain, graceful cleanup, and two
post-stop HBM snapshots.  Interpretation:

- all 20 correct at the historical WER level: shared-pool contamination
  confirmed and repaired; promote pool isolation to a reviewable repair,
  then requalify `910C-040` E1 and the `910C-042` performance attribution;
- still garbled with an isolated pool: the contamination is process-global
  NPU driver state, not pool-scoped; escalate to a CANN/torch_npu minimal
  reproduction capturing an encoder graph then replaying a decode graph;
- a capture failure or any forbidden signature invalidates the arm.

#### `910C-049` result and `910C-050` runtime interaction probe

`910C-049` completed: capturing encoder graphs into a dedicated private graph
pool did **not** repair the corruption — the E1 probe remained garbled 2/20.
It rejects this specific pool-isolation repair; it does **not** prove that the
entire failure is a process-global CANN/torch_npu defect. Combined with the
prior gates, the evidence supports this bounded working hypothesis:

- `910C-039 T1-fixed` (encoder/prefill graphs off, decode graph + compile on)
  returns WER 0.0167 with 0/70 garbled — the captured decode graphs replay
  correctly when no encoder graph is ever captured.
- `910C-040 E1` differs from T1-fixed **only** by enabling the encoder graph,
  which lazily captures once on the first real encoder request — long after
  the decode graphs were captured at startup and have been replaying
  correctly.  The output then garbles.
- `910C-045` capture-only (never replayed) still garbles, and `910C-046
  E-REL` capture-then-release still garbles, so neither replay nor a live
  retained graph is required; the capture action alone is sufficient.
- `910C-048` reports count=0 for every Python-observable section across
  warmup/capture/release, so no Python-layer state is mutated.

The leading hypothesis is that **one encoder-graph capture interacts with
state held by already-captured, correctly-replaying decode graphs in the same
process**. Private-pool isolation did not remove it and the current Python
snapshot did not expose it. A lower-layer runtime interaction is plausible,
but requires a reproducible result before it is attributed to CANN/torch_npu.
Do not treat `e9032edc` as a production repair or include it in a final
performance candidate merely because it changes graph-pool ownership.

`910C-050` builds a two-graph *synthetic* probe to determine whether a
vendor-actionable lower-layer reproduction can be produced. It is a single
self-contained script using only `torch`/`torch_npu` — no model weights,
SGLang service, HTTP, benchmark client, audio, or private data. The script
runs one arm per clean process and performs, in order:

1. capture a decode-like synthetic graph on a fixed static input, then replay
   it N times and record a reference output hash;
2. confirm the replayed output matches the eager reference (baseline healthy);
3. capture one distinct encoder-like synthetic graph on its own static input;
4. replay the decode graph again on the identical static input and re-hash;
5. report whether the post-encoder-capture decode replay still matches the
   step-2 reference.

A mismatch at step 5 with a match at step 2 and stable repeated hashes before
and after capture is vendor-actionable evidence of a runtime-level interaction:
a second capture breaks a resident graph in one process. The independently run
control captures the encoder-like graph **first** and later captures two
decode-like graphs, classifying whether a later capture can alter a resident
graph regardless of family. All inputs are fixed synthetic tensors, so the
script and sanitized output can leave the isolated environment in full.

A negative synthetic result is **inconclusive** for Qwen3-ASR: this probe does
not include compiled decode, Qwen3 attention/KV-cache behavior, or the
production stream/pool lifecycle. It must not be used to clear the runtime or
to dismiss the observed graph-plus-compile failure.

Local implementation note: the reproduction script is authored at
`benchmarks/diagnostics/npu_two_graph_capture_corruption.py` and committed
alongside this handoff, per the source-authority rule.  It is self-contained
and uses only `torch`/`torch_npu` — no SGLang service, no HTTP, no benchmark
client, no model weights, no audio.  The server runs it read-only
(`python benchmarks/diagnostics/npu_two_graph_capture_corruption.py --arm
decode-first`) and separately runs the control in a new process with `--arm
encoder-first`. Return the two stdout JSON documents, including per-arm
baseline/stability booleans, hashes, verdict, and torch/torch_npu/CANN/device
versions. The decode-first script exits 0 only when it reproduces the
interaction, 1 when it does not (inconclusive), and 2 on import/device
preflight failure. The control exits 0 after reporting either classification.
Local CPU execution is not possible (no NPU, `sglang` absent); only
`py_compile` and `git diff --check` are run locally.

##### vLLM-Omni reference points (design input, not a correctness claim)

The local `vllm-omni` checkout's
`vllm_omni/platforms/npu/graph_tools.py` uses native `torch.npu.NPUGraph`,
explicit synchronization before and after capture, and
`current_platform.get_global_graph_pool()` for its NPU captures. Its replay
copies each request input into a static tensor and clones persistent static
outputs before those outputs become request-owned. Its NPU tests also exercise
the lifecycle of two live graphs sharing the **same global pool**.

This is useful contrast, but not proof that Qwen3-ASR is wrong or that
vLLM-Omni is immune: the models, graph callables, compile path, cache layout,
and NPU runtime build may differ. It does reject treating a private pool as an
obvious correctness requirement. The next local design comparison must record
the production Qwen3-ASR capture API, pool identity, warmup stream, input-copy
ownership, and output lifetime against these vLLM-Omni invariants before
changing production graph ownership again.

#### `910C-051` real Qwen3 graph-to-compile transition probe

The synthetic `910C-050` result is inconclusive. `910C-051` therefore tests
the real service lifecycle with the existing Qwen3 encoder graph runner and
the actual compiled SGLang decode path, while keeping the experiment bounded.
Local commit `a7fda80f` adds the opt-in environment control
`SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES=N` (default `0`). On NPU only, it
returns `None` for the first N *unseen encoder signatures* before capture; the
normal audio-tower eager path then runs. This is an intentional diagnostic
control, not an eager fallback. When the counter reaches zero, the next
unseen signature follows the normal real encoder graph capture and replay
path. `model_info.encoder_cuda_graph.diagnostic_capture_defer` reports the
deferred count and remaining count.

Run one fresh process on one verified-clean NPU with the exact E1 profile:
encoder graph enabled, prefill graph disabled, decode graph enabled, torch
compile enabled, and the existing guard/settings otherwise unchanged. Set
`SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES=1`. Do not alter source, packages,
or server configuration files.

Use three distinct exact-10-second clips with the same Qwen3 encoder window
signature. They must be distinct because replaying the same bytes can be
satisfied by the pre-LM cache and would not call the encoder runner again.

1. Send clip A. It must be a pre-LM cache miss and return correct normalized
   text/expected WER through compiled decode. Immediately record model-info:
   `deferred_count=1`, `remaining=0`, encoder `captured_graph_count=0`, and
   encoder `replay_count=0` are required.
2. Send clip B. It must also be a cache miss with the same encoder signature.
   It must trigger normal encoder capture plus replay, return correct text,
   and advance encoder `captured_graph_count` and `replay_count` to at least
   one with no capture failure.
3. Send clip C. It must be a cache miss with that signature, replay the same
   encoder graph, return correct text, and increment encoder replay again.

Return only sanitized per-step correctness/latency, the two model-info
snapshots, encoder capture/replay/fallback counters, compile/decode graph
counters, and server revisions. A correct A followed by garbled B or C proves
the real encoder-capture-to-compiled-decode transition in one process. A
correct A/B/C disproves that narrow transition hypothesis for the tested
signature; it does not qualify the ALL configuration or close the prefill plus
compile defect. Any cache hit, signature mismatch, capture failure, fallback,
or teardown failure invalidates the run rather than supporting either result.

`910C-051` did not establish that sequence: the first measured clip was
already garbled, while its returned model-info showed `deferred_count=0`,
`remaining=0`, and one captured/replayed encoder graph. NPU `capture_all()`
does not call `_capture()`, and the Qwen3 runner has no other production
`_capture()` caller than `run()`. Therefore the result is ambiguous: either
the serving model process did not inherit the environment variable, or an
earlier real runner invocation occurred before clip A. Do not characterize it
as startup capture without evidence.

`910C-051A` adds runner provenance only: constructor log with PID/configured
defer count plus model-info fields `configured`, `run_count`, and
`first_capture` (run index, bucket, window count). It does not alter capture,
replay, graph pools, or compiled decode. Repeat only the startup-to-first-clip
precheck with `SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES=1`; do not send audio
until both import provenance and model-info satisfy the following gate.

1. With the exact interpreter and environment used to launch the service,
   print the resolved `__file__` for
   `sglang_omni.models.qwen3_asr.encoder_cuda_graph`, the checkout HEAD, and
   the tracked blob ID for that file. Return these three sanitized values with
   the startup log line. The resolved file must belong to the declared clean
   checkout at the handoff HEAD; an installed wheel, another worktree, or an
   untracked copy invalidates the run.
2. After the service is healthy but before any audio request, query
   `model_info.encoder_cuda_graph.diagnostic_capture_defer`. Its dictionary
   must contain `configured`, `deferred_count`, `remaining`, `run_count`, and
   `first_capture`. Require `configured=1` and `run_count=0`. The earlier
   response that omitted these fields is an import/response-provenance failure,
   not evidence of an NPU startup capture or stale bytecode by itself.
3. If the import or model-info gate fails, stop without sending clip A and
   return the resolved module path, checkout/blob identity, startup line, and
   complete key set of the encoder graph info. Do not delete bytecode caches,
   reinstall packages, edit source, or infer the cause from `.pyc` files.

If `configured=0` after the provenance gate passes, fix the service-process
environment propagation before proceeding. If `configured=1` and `run_count>0`
before clip A, identify that pre-request runner invocation from the logged PID
and execution timeline. Only if `configured=1` and `run_count=0` may the
three-clip transition procedure be retried.

`910C-051A` confirmed the source checkout path, HEAD, and tracked blob, but
the running service still omitted the three fields introduced by `811ff448`.
That is an invalid precheck. It does not prove a different source file was
imported: a stale bytecode code object or a preloaded parent process can retain
the source file name while executing older definitions.

#### `910C-051B` source-only runtime import gate

Before any further A/B/C request, start one new service process from the
declared clean checkout with a task-local **empty** Python cache prefix and no
bytecode writes. The task-local prefix must be outside the repository and may
be removed at normal task cleanup; do not delete or modify any repository
`__pycache__`, package, source, or configuration file. Do not use the
installed `sgl-omni` console script for this gate: its interpreter has already
started before it can demonstrate that the Python initialization options were
accepted. Replace only the launch prefix of the existing service command with:

```text
PYTHONPYCACHEPREFIX=<new-empty-task-local-directory> \
PYTHONDONTWRITEBYTECODE=1 \
SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES=1 \
python -B -X pycache_prefix=<new-empty-task-local-directory> \
    -m sglang_omni.cli serve <the-existing-unchanged-serve-arguments>
```

The environment variables are retained for child-process inheritance; the
`-X pycache_prefix` option is required on the parent interpreter command line.

Before the service starts, return the launch interpreter's resolved module
path and `__cached__` path for `encoder_cuda_graph`, plus checkout HEAD and
tracked blob ID. The cached path must resolve under the task-local prefix, not
an existing checkout `__pycache__`. After the service is healthy and before
audio, require the full five-key `diagnostic_capture_defer` dictionary with
`configured=1` and `run_count=0`.

If that dictionary is still incomplete, stop without audio and classify the
failure as a service worker/bootstrap or endpoint-routing mismatch. Return the
launch PID, worker PIDs, port owner, and module/cache paths; do not claim an
NPU capture effect. If it passes, continue immediately in the same fresh
process with the existing three-clip A/B/C transition procedure.

`910C-051B` remained invalid, but its API response must not be interpreted as
a top-level encoder-graph object: `/model_info` preserves each worker response
under `stages[*].data`. Query and return
`stages[*].data.encoder_cuda_graph`, including its complete key set, rather
than a flattened or inferred `model_info.encoder_cuda_graph` path.

#### `910C-051C` worker-owned runtime identity gate

Before another capture or audio probe, use the handoff commit that adds
`encoder_cuda_graph.runtime_identity`. This value is produced in
`ModelWorker._encoder_cuda_graph_info()` by the stage worker immediately after
calling the live runner's `model_info()`; it neither reloads modules nor
changes graph capture, replay, pools, streams, or compile behavior. It
contains only path-free fields:

```text
runner_type
model_info_has_defer_provenance
model_info_keys
diagnostic_capture_defer_keys
```

Run one fresh, no-audio process with the existing source-only command and
`SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES=1`. After health is ready, return
the complete `stages[*].data.encoder_cuda_graph` object and require all of:

1. `runtime_identity.model_info_has_defer_provenance=true`;
2. `runtime_identity.diagnostic_capture_defer_keys` contains exactly
   `configured`, `deferred_count`, `remaining`, `run_count`, and
   `first_capture`;
3. `diagnostic_capture_defer.configured=1`, `run_count=0`,
   `deferred_count=0`, `remaining=1`, and `first_capture=null`; and
4. `captured_graph_count=0` before any audio.

If any field is absent or disagrees, stop without audio and return the raw
stage-data shape plus launch and worker PIDs. This distinguishes an old runner
method, a stale model-worker implementation, and API response routing without
using `.pyc` paths as causal evidence. Only a successful gate may continue
immediately with the existing A/B/C procedure in that same process.

#### `910C-051D` spawn import-boundary probe

`910C-051C` reported an empty `runtime_identity`, which cannot be emitted by
the `f063a4f6` worker implementation. Before changing module reload behavior,
run the dedicated no-NPU probe from the declared checkout with the exact
service interpreter and the existing defer environment:

```text
python -B -m benchmarks.diagnostics.spawn_import_provenance \
    --expected-root <declared-clean-checkout>
```

It starts one plain `multiprocessing.spawn` child and reports only path-free
booleans for both parent and child: whether the encoder and model-worker
modules resolve under the expected checkout, whether the live runner method
contains all five defer keys, whether the model-worker method contains the
`runtime_identity` wrapper, cache/bytecode mode, and inherited defer value.
It creates no model, graph, NPU context, service listener, or audio request.

If parent and child both pass, the remaining fault is service-stage routing or
the reported response extraction, and the next local change must instrument
that stage boundary. If the child differs from the parent, classify it as a
reproducible spawn bootstrap/import problem and repair the executable or
`sys.path` handoff based on the returned boolean that differs. Do not reload a
module or make another graph claim before this probe has a result.

#### `910C-051E` hash-validated source-cache gate

`910C-051D` found that both its parent and child runner methods lacked the
five-key implementation. This is consistent with an accepted stale timestamp
`.pyc`, but `python -B` and `importlib.reload()` do not guarantee a source
recompile: the former only suppresses bytecode writes and the latter may read
the same valid cache entry. Do not delete checkout caches or edit source.

Instead create one new task-local cache directory outside the checkout and,
using the exact service interpreter, compile only the two affected source files
with checked-hash invalidation. Then run the provenance probe with that same
prefix on its parent command line:

```text
PYTHONPYCACHEPREFIX=<new-empty-task-cache> \
python -X pycache_prefix=<new-empty-task-cache> -m compileall -q -f \
    --invalidation-mode checked-hash \
    sglang_omni/models/qwen3_asr/encoder_cuda_graph.py \
    sglang_omni/model_runner/model_worker.py

PYTHONPYCACHEPREFIX=<same-task-cache> \
PYTHONDONTWRITEBYTECODE=1 \
SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES=1 \
python -B -X pycache_prefix=<same-task-cache> \
    -m benchmarks.diagnostics.spawn_import_provenance \
    --expected-root <declared-clean-checkout>
```

The returned parent and child objects must both report true for
`encoder_cache_under_pycache_prefix`,
`model_worker_cache_under_pycache_prefix`,
`runner_model_info_has_defer_provenance`, and
`model_worker_has_runtime_identity`. A failure is an interpreter/bootstrap
launch defect, not a reason to reload modules or begin a service. If every
field passes, launch the no-audio service with the identical task cache prefix
and re-run `910C-051C`'s nested stage-data gate. The temporary cache is normal
task output and may be removed after clean shutdown.

#### `910C-052` immutable wheel runtime gate

`910C-051B` through `910C-051E` are historical import diagnostics and are
superseded. Do not run another cache-prefix, reload, editable-install, or
checkout-launched service probe. The standard isolated-server runtime is an
immutable wheel installed into a new, task-scoped virtual environment; the
checkout is a build input only and must not appear on the runtime import path.

The server operator is authorized to create this new venv and install the
locally built wheel with `--no-deps`. This is environment provisioning, not a
source/package edit: it must not alter the existing serving environment,
checkout contents, or dependencies. Use the exact current handoff commit and
the existing serving interpreter, then run the following conceptual sequence
with server-local temporary directories outside the checkout:

```text
# 1. Build exactly one wheel from the declared clean checkout.
<serving-python> -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir <task-wheel-dir> <declared-clean-checkout>
sha256sum <task-wheel-dir>/sglang_omni-*.whl

# 2. Create a runtime-only environment that inherits the already-qualified
#    NPU stack but installs the Omni wheel ahead of it.
<serving-python> -m venv --system-site-packages <task-runtime-venv>
<task-runtime-venv>/bin/python -m pip install --no-deps --force-reinstall \
    <task-wheel-dir>/sglang_omni-*.whl

# 3. From outside the checkout, attest the installed artifact before service.
cd <task-runtime-directory-outside-checkout>
<task-runtime-venv>/bin/python -I -m sglang_omni.diagnostics.runtime_artifact \
    --forbid-root <declared-clean-checkout>
```

The returned JSON must be `valid=true`, with all of the following true:

```text
isolated_venv
encoder_module_under_install_root
model_worker_module_under_install_root
encoder_record_hash_matches
model_worker_record_hash_matches
runner_model_info_has_defer_provenance
model_worker_has_runtime_identity
encoder_module_outside_forbidden_root
model_worker_module_outside_forbidden_root
```

Return the wheel SHA-256, venv-local distribution version, and boolean report;
do not return private paths. On any false field, stop before service startup.
On success, start the service only with `<task-runtime-venv>/bin/python -m
sglang_omni.cli`, from outside the checkout, using the unchanged approved E1
arguments. Then repeat the no-audio nested-stage model-info gate once. This is
the final runtime-identity gate; a pass immediately authorizes the A/B/C
transition procedure in the same fresh service. The task venv and wheel cache
may be removed after graceful shutdown.

`910C-052` found one attestation implementation defect: CPython may fold the
five nested defer-field strings into a tuple inside `co_consts`, so a direct
top-level membership test reports a false negative even for the correct wheel.
The repaired gate recursively checks folded literal containers and has a CPU
regression test. Rebuild the wheel from the repair handoff commit and repeat
only the immutable-wheel attestation as `910C-052B`; do not start a service or
touch an NPU until its JSON is `valid=true`. This is a correction to the gate,
not a new graph experiment.

`910C-052B` passed: the immutable wheel runtime reported `valid=true` with
wheel SHA-256 prefix `a40d12f`. This closes runtime provenance. Earlier
checkout-path, console-script, pycache-prefix, and spawn-probe gates are
historical diagnostics only and must not be repeated or used to qualify the
current runtime.

#### `910C-053B` isolated wheel launch gate

`910C-053` did not start a valid wheel service: although the wheel attestation
passed, the service was launched from the checkout and an inherited editable
path remained eligible for imports. This is a bootstrap-contract failure, not
graph, capture, compile, or NPU evidence. Do not reload modules, delete caches,
or modify the existing serving environment.

Build one wheel from this handoff commit and install it into a fresh task venv
as in `910C-052B`. From a task directory outside the checkout, use only this
standard launch chain for both its no-service preflight and service process.
First install a venv-local startup guard. It is not an editable install and
does not modify the checkout or existing serving environment; it is written
only to the newly created task venv and refuses to overwrite a pre-existing
`sitecustomize.py`:

```text
<task-runtime-venv>/bin/python -I -m sglang_omni.diagnostics.isolated_launch \
    --forbid-root <declared-clean-checkout> --install-site-guard

export SGLANG_OMNI_FORBID_IMPORT_ROOT=<declared-clean-checkout>

<task-runtime-venv>/bin/python -I -m sglang_omni.diagnostics.isolated_launch \
    --forbid-root <declared-clean-checkout> --attest

<task-runtime-venv>/bin/python -I -m sglang_omni.diagnostics.isolated_launch \
    --forbid-root <declared-clean-checkout> -- \
    serve <the unchanged approved E1 arguments>
```

Do not add `-S`: it can suppress the virtual-environment site-packages needed
for the installed wheel. The `--attest` JSON must be `valid=true` with
`isolated_interpreter=true`, `cwd_outside_forbidden_root=true`,
`checkout_absent_from_sys_path=true`, and
`launcher_outside_forbidden_root=true`, `site_guard_installed=true`, and
`forbid_root_environment_matches=true`. The guard removes the declared
checkout during Python site initialization, before either the parent or a
spawned stage worker imports Omni. The wrapper repeats the same parent-path
check before importing the serving CLI. If the gate fails, stop before service
startup and return only its booleans plus the wheel SHA prefix.

The operator is authorized to make a server-local, commit-backed correction
only if this wrapper itself has an unambiguous bootstrap/test defect. It may
not change serving, model, graph, compile, dependency, or benchmark behavior.
Return the allowed text-only repair record defined at the top of this handoff.

#### `910C-053E` standard spawn-child import gate

`910C-053D` established the parent interpreter gate but the later service
response did not contain the expected worker provenance. That response alone
does not prove that a child loaded an editable checkout: standard-library
`spawn` normally replaces its child path with the parent's preparation data.
Before another service launch, test that exact process boundary without model
construction, weights, graph, audio, or NPU initialization:

```text
export SGLANG_OMNI_FORBID_IMPORT_ROOT=<declared-clean-checkout>
<task-runtime-venv>/bin/python -I -m sglang_omni.diagnostics.isolated_launch \
    --forbid-root <declared-clean-checkout> --attest-spawn-child
```

The combined JSON must be `valid=true`, including all parent `910C-053D`
booleans and the child booleans: no checkout on `sys.path`, both encoder and
ModelWorker modules outside the checkout, all five defer provenance fields,
and worker `runtime_identity` support. A failure is definitive import-runtime
evidence: stop before service and replace the system-site-packages task venv
with a dependency-complete pure wheel runtime. A pass means the old
`/model_info` shape is a stage/runner response issue, not evidence that spawn
loaded stale source; then repair that response contract locally before audio.

#### `910C-053C` real encoder-capture transition probe

Use the `910C-053B`-attested task venv and its wheel as the only runtime. Run
one fresh E1 service from outside the checkout with the exact existing E1
arguments: encoder graph enabled, prefill graph disabled, decode graph enabled,
torch compile enabled, guard enabled, and
`SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES=1`. Do not modify source, packages,
or service configuration; do not run a new wheel or venv setup during this
task.

After health is ready and before audio, query `/model_info` and read the raw
stage result at `stages[*].data.encoder_cuda_graph`. Require the worker-owned
object to report a nonempty `runtime_identity`, the exact five-key
`diagnostic_capture_defer`, `configured=1`, `run_count=0`,
`deferred_count=0`, `remaining=1`, `first_capture=null`, and
`captured_graph_count=0`. Any mismatch stops the task without audio.

If the no-audio gate passes, immediately run the existing three distinct,
same-signature exact-10-second clips A/B/C in that same process:

1. A must be a cache miss and correct through compiled decode while deferring
   capture (`deferred_count=1`, no capture or replay);
2. B must be a cache miss, trigger one normal encoder capture/replay, and be
   correct; and
3. C must be a cache miss, replay that graph, and be correct.

Any incorrect normalized output, cache hit, signature mismatch, capture
failure, eager fallback, endpoint/device error, non-drained service, or
teardown failure ends the task at its first occurrence. Return only the wheel
SHA prefix, runtime identity/key sets, per-step success/correctness summary,
counter deltas, and health/cleanup status. A correct A then garbled B/C is now
valid evidence of the real graph-capture-to-compiled-decode transition; a
correct A/B/C rejects that narrow hypothesis and permits the prefill-plus-
compile transition probe next.

The project requires every currently failing acceleration path to be repaired;
disabling it is not an acceptable close condition. Qualify these changes
separately and then in combination:

1. prefill graph with the execution guard passed `910C-025A` and is retained;
2. qualify local commit `fa5b8852`, which repairs the incompatible NPU encoder-
   graph capture path so host-
   device copies and synchronization do not occur illegally inside capture;
3. repair the compile-enabled generation path across the SGLang/triton-ascend
   boundary and requalify every compiled and non-compiled decode bucket;
4. profile the coarse encoder/generation execution guard and narrow or replace
   its critical section if serialization limits throughput;
5. run the fully combined profile with compile, encoder graph, prefill graph,
   and decode graph enabled, positive execution markers, zero unexpected eager
   fallback, and the exact-10-second gate.

This list is the current feature-completion backlog for goal 1. The first
encoder-graph implementation did not close item 2 because `910C-026` observed
64 signature-mismatch eager fallbacks. Do not start the final three-repeat
hard-target campaign until items 1 through 5 are complete.
Single-arm C70 measurements collected while qualifying an item are authorized
only as directional before/after evidence and must be labelled diagnostic.

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
4. Qualify each identified acceleration feature independently: prefill graph,
   decode graph, NPU encoder graph, and torch compile. Record positive execution
   markers, zero unexpected fallback, and every deviation from repository
   defaults.
5. Qualify the combined `ALL` profile with every identified acceleration
   feature enabled. A disabled-feature compatibility profile cannot close this
   step.
6. Only after the `ALL` feature gate passes, run the full
   [performance task](qwen3_asr_ascend_910b_performance_task.md), including the
   soak and three fresh-process concurrency-70 repeats. Compare controlled
   variants as needed, but the final accepted result must use the fully enabled
   profile.
7. Start the separate realtime implementation and qualification gate after the
   offline path is correct and its remaining latency bottleneck is understood.

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
| 910C-024A | `94dec6e0`; clean | exact10 harness `37f598f3` + corpus transform `2cb63b9e` + accounting hardening `63f235fa` + fixed 24-to-16 kHz transform `8d46ddec` + deterministic best-fit packing `30b21522` | Exact accepted `910C-023` profile; pinned local SeedTTS snapshot; ffmpeg 6.1.1 aarch64 | Deterministic resampled corpus, NPU parser, batch-one and concurrency-two harness qualification | passed | 52/52 tests; 770 distinct exact-10-second clips; manifest `25314d13...3000b`; batch-one p95 0.250 s/WER 0; concurrency-two p95 2.082 s/WER 0.0152; zero request failures/fallback and clean teardown |
| 910C-024B | `45535923`; no runtime edit | Exact10 compatibility profile and frozen `910C-024A` corpus | Two independent fresh services; compile, encoder graph, and prefill graph disabled; guarded decode graph enabled through 70 | Arm S: 100 sequential; Arm C70: 700 measured at concurrency 70 | performance measurement valid; hard target missed; cleanup unresolved; required model-info deltas not returned | Arm S 100/100, p95 0.289 s, WER 0.0167. Arm C70 700/700, p95 3.516 s, 39.31 req/s, WER 0.0164, zero request failures; post-stop chip0 HBM remained 87% rather than 4%; encoder/decode/guard dominance is not yet established |
| 910C-024C | handoff `0ce137dc`; no runtime edit | Quiescent post-`910C-024B` host; no service/model/benchmark/device mutation | Read-only five-minute HBM, process, device-node, health, and preserved-evidence attribution | Post-run HBM attribution and missing model-info audit | completed: outcome 2, identified holder | NPU context PID 2043369 retained 53,966 MB after Arm C70 was killed with `SIGKILL`; no manageable user process remained. Coarse log stats: encoder 251 batches/754 items, queue wait avg/max 1.24/13.26 s, encoder time 22.6 s; decode 100% graph replay across all 13 buckets; no pre/post rich model-info snapshots |
| 910C-024D | `db19be76`; no runtime edit | Quiescent post-`910C-024C` host | Operator had terminated PID 2043369; no reboot or driver restart | Three-snapshot read-only recovery verification | passed | Both chips healthy at stable 4% HBM across t=0/10/20 s; PID 2043369 and other holders/workers absent; port 8000 free; no acceleration run started |
| 910C-025A | `3ced6537`; no runtime edit | Frozen `910C-024A` exact10 corpus and `910C-024B` common workload | Two independent fresh-process current-code arms: all-eager E0 and guarded prefill+decode graph P | Acceleration screening and one C70 measurement per qualified arm | completed; P is the best measured configuration; hard target missed | E0 p95 2.641 s/31.52 req/s; P p95 1.771 s/48.39 req/s, a 49.6% p95 reduction and 23.1% throughput gain versus `910C-024B`; P remains 3.54x over the latency target and at 34.6% of required throughput; full cleanup and metric matrix were not included in the returned summary |
| 910C-026 | handoff `92fcf514`; code `fa5b8852` | Exact accepted `910C-025A` P stack plus NPU host-side encoder attention metadata and bounded lazy signature capture | Encoder, prefill and decode graphs enabled; compile disabled; guard active | Focused/full tests, repeated batch one, 70-sample warmup, and one exact10 C70 measurement | feature qualification failed; later measurement diagnostic only | Focused 15 passed/1 skipped and model-info 4 passed; full Qwen suite had 11 failures but execution incorrectly continued. Batch-one capture/replay passed. C70 completed 700/700 at p95 1.652 s and 52.55 req/s, but 64 `npu_signature_mismatch` eager fallbacks violated the zero-fallback gate; local multi-signature repair required |
| 910C-027 | handoff `65273c98`; code `9080b901` | Exact `910C-026` stack plus globally bounded NPU encoder multi-signature cache and legacy-mock guard compatibility | Encoder, prefill and decode graphs enabled; compile disabled; guard active | Signature warm-up at 1/2/4/8 and C70; one 700-request C70 measurement | partial: functional fallback removed; warm-up/performance qualification failed | C70 had zero encoder fallback/capture failure and clean 4% HBM recovery, but signatures grew 5 to 8 during measurement. Diagnostic p95 2.685 s, 50.60 req/s, RTFx 506; exact test counts were omitted from the returned summary |
| 910C-028 | handoff `a148f8c6`; code `9080b901` | Exact `910C-027` stack and profile; no code change | Encoder, prefill and decode graphs enabled; compile disabled; guard active | Deterministic encoder-batch warm-up 1 through 8, bounded repeated C70 saturation, then one C70 measurement | superseded before execution | Folded into `910C-029` so the server receives the complete Torch Compile and Encoder Graph code together |
| 910C-029 | handoff `a389a000`; SGLang `d7e0d517e` plus an uncommitted server edit | Complete local Torch Compile boundary repair, bounded encoder signatures, graph guard, and positive compile model-info | TC isolation, then fully enabled ALL profile | Compile/non-compile buckets 1/32/64/70; deterministic encoder saturation; ALL batch1/conc2 and 700-request C70; conditional full campaign | failed protocol and feature gate; not qualified | Prefill-disabled TC exposed missing `attention_layers`; server then bypassed the compile context and saw ATB PagedAttention failure. It continued contrary to the stop rule with compile off, so the 140-request hang is diagnostic only and was not an ALL result |
| 910C-030 | `d0d55e8c`; SGLang `93d312480`; no server edit | Decode-only attention/MoE metadata initialization on the prior compile-safe attention boundary | TC isolation before conditional ALL | Full decode capture list 1 through 70 | partial progress; TC failed at compiled batch 1; ALL correctly not run | SGLang 20, Omni focused 15/1 skipped, and Qwen3-ASR 594/3 skipped passed. Buckets 70 through 2 captured; batch 1 failed when Dynamo traced external `split_qkv_rmsnorm_rope -> get_device_properties`. Missing metadata and ATB PagedAttention signatures were absent |
| 910C-031 | `9c27cca9`; SGLang `44f9e40b5`; no authorized server edit | Opaque custom-op boundary around the unchanged installed fused QKV/RMSNorm/RoPE kernel | TC isolation, fully enabled ALL, exact10 C70 diagnostic | capture/scheduling passed; numerical correctness failed | TC captured all 13 buckets and drained 1/8/32/70 with no forbidden signature. ALL completed 700/700 with all graph paths positive and zero encoder fallback, but WER was 1.4464 with corrupted output; p95 2.142 s is invalid performance evidence. A later dirty removal of `fullgraph=True` did not restore accuracy and is rejected |
| 910C-032 | handoff `f56cc244`; SGLang `5403d1f7d`; no server edit | Compiled external-kernel/custom-op value parity plus bounded TC/ALL matrix | NPU op parity, then T1/T2/T70/A0/A1 exact10 accuracy arms | completed; Torch Compile incorrect | Both parity cases passed, but every compile-on arm garbled 70/70 with WER 1.3923--1.4198; A0 compile-off was correct at WER 0.0183. Compile defect is independent of batch size and encoder/prefill graphs |
| 910C-033 | handoff `f62030d3`; SGLang `a7b279da6` | `910C-032` plus diagnostic compile-stage selector | D0 prepared-eager and D1 Dynamo-eager T1-profile exact10 accuracy arms | completed; compile-safe fused-op state is faulty | Both arms completed 140/140 but were garbled 140/140 with WER 1.3929; the first fault is active before Dynamo and TorchAir, while the compile-safe context is enabled |
| 910C-034 | handoff `fc021329`; SGLang `3c389d2f1` | NPU TopK preserves its eager `torch.ops.npu` dispatch inside the compile-safe context | Conditional D0-R, normal T1-R, then fully enabled A1-R exact10 correctness arms | completed; TopK hypothesis rejected | Declared tests passed, but D0-R remained garbled 70/70 with WER 1.4220; T1-R and A1-R correctly did not start |
| 910C-035 | handoff `9041c9a0`; SGLang `9438420a6` | NPU RMSNorm and SiLU preserve their existing `torch_npu` dispatch inside the compile-safe context | Conditional D0-R2, normal T1-R2, then fully enabled A1-R2 exact10 correctness arms | completed; generic decoder dispatch hypothesis rejected | Declared tests passed, but D0-R2 remained garbled 70/70 with WER 1.4140; T1-R2 and A1-R2 correctly did not start |
| 910C-036 | handoff `7b07596a`; SGLang `e45d64c9f` | Raw model forward with graph-safe decode-attention context retained but compile-safe fused-op preparation omitted | One `C0-context-eager` exact10 correctness arm | superseded before server execution | Replaced by the broader, independently scoped `910C-037` matrix to avoid a human round trip |
| 910C-037 | handoff `6c7f1756`; SGLang `3295b12d3` | Separate graph-safe-attention, audio-tower-only prepared, and language-model-only prepared eager diagnostics | C0 first; if correct, independent A0 and L0 exact10 correctness arms | completed; C0 corruption reproduced | C0 remained garbled; A0/L0 correctly did not start; the graph-safe route is necessary, but its custom-op wrapper remains unisolated from the terminal backend |
| 910C-038 | handoff `a2c91ead`; SGLang `8ec282120` | Direct NPU graph attention with TC custom-op wrapper bypassed | One W0 direct-graph eager exact10 correctness arm | completed; wrapper localized | W0 was correct at WER 0.0183 with 0/70 garbled outputs; direct backend is not the first fault, and the bypass remains diagnostic only |
| 910C-039 | handoff `e8b80db9`; SGLang `54a8d042d` | Preserve static NPU graph state within the registered decode-attention custom op | Real normal-compile T1, then conditional real normal-compile ALL A1 correctness arms | partial; T1 passed, A1 garbled | T1 returned WER 0.0167 with 0/70 garbled; A1 remained garbled around WER 1.39, proving only an unresolved full-combination interaction |
| 910C-040 | handoff `02cc6166`; SGLang `54a8d042d`; no server edit | Split real normal-compile encoder+graph and prefill+graph interactions | Independent E1 encoder-compile and P1 prefill-compile exact10 correctness arms | completed; both feature combinations garbled | T1 compile-only remained correct, but both encoder+compile E1 and prefill+compile P1 produced garbled output; this proves two failing combinations but may still reflect a shared graph-state hand-off defect |
| 910C-041 | handoff `28e297c3`; SGLang `54a8d042d`; no server edit | Token-boundary comparison of graph-produced state entering compiled decode | Four independent encoder/prefill on/off controls and compile-on probes at `max_new_tokens=1` and `2` | completed; shared transition boundary found | Both E1 and P1 matched their controls at one generated token and diverged at two; the first compiled decode transition is the common failure boundary |
| 910C-042 | handoff commit containing this row; SGLang `54a8d042d` | Serial stage attribution while compile-combination accuracy remains open | Same-NPU control/treatment arms run one fresh service at a time at exact10 C70 with existing structured events | authorized; pending; serial policy supersedes multi-NPU plan | Collect encoder, prefill, decode-step, guard, graph-bucket and NPU utilization costs; garbled arms are diagnostic only and cannot satisfy the hard target |
| 910C-043 | handoff `970e9560`; code `b6966d4d`; SGLang `54a8d042d`; no server edit | Opt-in NPU device completion before each execution-guard hand-off | Parallel E1-F/P1-F two-token correctness probes | completed; hypothesis rejected | Both fenced arms remained garbled, so device completion alone does not repair the first compiled decode transition; fenced performance is invalid |
| 910C-044 | handoff `53c1eccd`; SGLang `634303cdf`; no server edit | Make paged KV storage and cache locations explicit custom-op state | TC regression control and conditional combination arms | completed; rejected and reverted by SGLang `ca17cd413` | The explicit operands did not repair accuracy, regressed T1, and introduced a warm-up hang; do not reuse this implementation |
| 910C-045 | handoff `af793d17`; Omni `8dab0b8f`; SGLang `5cb571995`; no server edit | Distinguish graph capture/init contamination from actual encoder/prefill replay | Serial E-CAP then P-CAP capture-only correctness arms on one clean NPU | completed; both arms garbled | Encoder and prefill replay were bypassed, but both combinations still corrupted the first compiled decode transition; capture/init plus retained graph state is sufficient |
| 910C-046 | handoff `06f6043d`; Omni code `0948859a`; test fix `b28013f0`; SGLang `1cd6be1b5`; no server edit | Distinguish irreversible capture mutation from live graph/pool/static-buffer ownership | Serial E-REL then conditional P-REL capture-release correctness arms on one clean NPU | E-REL completed and remained garbled 2/20; P-REL not evidenced | Releasing the encoder graph prevented broad retained-graph corruption but did not protect two probe outputs. Encoder capture does not call `prepare_model_for_torch_compile`; correlate the remaining failures with lazy captures before claiming persistent global mutation |
| 910C-047 | handoff `deb3a680`; Omni code `d3f71fb7`; SGLang `1cd6be1b5`; no server edit | Compare real encoder output immediately before capture with normal full-eager output after capture and release | One serial E1 capture-release parity probe on one clean NPU | completed; encoder exonerated, downstream state implicated | Parity `allclose=True` (encoder numerically identical across capture+release) yet outputs still garbled; disproves both the `prepare_model_for_torch_compile` mutation hypothesis and encoder-output corruption; contamination is in downstream device/compile runtime state visible only in compiled decode |
| 910C-048 | handoff `daf5c354`; Omni code `cf79b353`; SGLang as `910C-047`; no server edit | Attribute which runtime state transitions during capture persist through release | One serial E1 capture-state snapshot probe on one clean NPU | completed; all Python-visible sections clean, driver layer implicated | Every non-memory section reported count=0 across warmup/capture/release; no Python-observable state is mutated by encoder capture, so contamination is below the Python layer in NPU driver state; only memory counters moved (graph static-buffer allocation, expected) |
| 910C-049 | handoff `22d22ef1`; Omni code `e9032edc`; SGLang as `910C-048`; no server edit | Repair shared default graph-pool contamination by capturing encoder graphs into a dedicated private pool | One serial E1 normal-replay correctness probe on one clean NPU (no bypass diagnostics) | completed; hypothesis rejected, driver layer confirmed process-global | Private pool did not repair; E1 remained garbled 2/20, so contamination is not pool-scoped but process-global NPU driver state; combined with T1-fixed/E1/045/046/048 this isolates one encoder capture retroactively corrupting already-captured correctly-replaying decode graphs |
| 910C-050 | server script locally patched (not source-authoritative); source replacement `a7fda80f`; SGLang not required | Determine whether a synthetic two-graph torch_npu probe reproduces a runtime-level interaction | Run decode-first and encoder-first separately, each in a fresh verified-clean NPU process; no service/HTTP/benchmark | completed; no interaction reproduced, inconclusive for Qwen3-ASR | The initial script required fixes for torch scope and encoder-like conv/MLP shapes; its negative result cannot clear the real failure because it omits compiled decode, attention/KV, and production stream/pool lifecycle |
| 910C-051 | Omni `606252b6`; SGLang as current exact E1 environment; no server edit | Isolate the real first encoder capture-to-compiled-decode transition in one service process | One fresh serial E1 process with `SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES=1`, then three cache-miss clips with one exact encoder signature | invalidated before comparison | Clip A was already garbled; returned model-info had defer configured state unavailable (`deferred=0`, `remaining=0`) and an existing graph/replays, so the mandatory A-before-capture condition was not met |
| 910C-051A | `b76e8166`; Omni code with constructor/run/capture provenance; SGLang as exact `910C-051` environment | Determine whether the defer environment reached the serving model process or a runner invocation preceded clip A | One serial precheck only: startup log plus model-info before any audio under `SGLANG_OMNI_ENCODER_GRAPH_DEFER_CAPTURES=1` | invalidated: source identity passed, runtime code identity did not | The on-disk module path, HEAD, and tracked blob matched `811ff448`, but runtime model-info omitted the new `configured`, `run_count`, and `first_capture` fields. No audio was sent; do not infer source mismatch, stale bytecode, or NPU capture without the source-only gate. |
| 910C-051B | `2229e61f`; same source/SGLang as `910C-051A` | Prove the service executes the declared encoder graph module before any graph/capture conclusion | One fresh no-audio source-only service precheck with an empty task-local cache prefix and defer=1 | invalidated: response-shape/runtime identity unresolved | Source path/HEAD/blob matched but the reported defer dictionary was incomplete. No audio was sent. `/model_info` stage nesting was not inspected, so this is neither bytecode nor NPU evidence. |
| 910C-051C | handoff commit containing this row; same source/SGLang as `910C-051B` | Identify the live stage runner method and correctly read nested encoder graph data before capture attribution | One fresh no-audio source-only service precheck with defer=1 and worker-owned runtime identity | authorized; pending server run | Require `stages[*].data.encoder_cuda_graph.runtime_identity` plus the five defer fields, zero run/capture count, then immediately execute A/B/C only if all gates pass. |
| 910C-051D | handoff commit containing this row; no SGLang/NPU dependency | Reproduce or reject a Python spawn import mismatch without model construction | Parent/child source-provenance probe under the exact service interpreter and environment | authorized; pending server run | Compare checkout ownership and method capabilities on both sides of one plain spawn boundary; no service, graph, or audio is allowed. |
| 910C-051E | handoff commit containing this row; no SGLang/NPU dependency | Force and attest checked-hash bytecode for the two source-defined provenance methods | Task-local checked-hash cache, then parent/child provenance probe using the exact service interpreter | authorized; pending server run | Both processes must load encoder and ModelWorker cache entries from the temporary prefix and report the new method capabilities before any service starts. |
| 910C-052 | `715fddf1`; no NPU action before artifact attestation | Replace checkout/editable runtime with a wheel installed to a new task-scoped venv | Immutable wheel build, SHA record, isolated-venv install, and installed-file RECORD validation | initial implementation false-negative | CPython folded nested literal keys into a tuple, so a correct wheel was incorrectly reported invalid; no service/NPU ran. |
| 910C-052B | `f3ca69d6`; no service/NPU action | Correct the immutable-wheel attestation's CPython folded-constant false negative | Rebuild one wheel and repeat only the installed-artifact JSON gate | passed | `valid=true`, wheel SHA-256 prefix `a40d12f`; code, wheel RECORD, venv isolation, and provenance checks all passed. |
| 910C-053 | `20964b79`; exact attested `910C-052B` wheel venv | Execute the first valid encoder-capture-to-compiled-decode transition experiment | One fresh E1 service: no-audio nested-stage gate, then distinct same-signature A/B/C | invalidated before service | The service was started from the checkout, so its runtime path contract was breached. This is not graph or compile evidence and must not be used for attribution. |
| 910C-053B | `bcd04cbd`; new wheel from that commit | Establish the only allowed wheel-service launcher | Wheel install, task-venv `sitecustomize` guard, `python -I` isolated-launch attestation, then unchanged E1 service through the same wrapper | superseded before valid service | Parent attestation alone passed, but the child-runtime gate was not hard enough. Do not attribute this failure to graph/compile or spawn semantics. |
| 910C-053D | handoff commit containing this row; new wheel from that commit | Establish checkout-free imports before every parent and child module load | Wheel install, guarded task venv, `python -I` attestation, then unchanged E1 service through the same wrapper | authorized; pending server run | `valid=true` requires the prior fields plus `site_guard_installed` and an exact forbid-root environment match. Stop before service on any false value. |
| 910C-053E | handoff commit containing this row; exact guarded task venv | Directly attest standard-library spawn child imports before service | One parent/one child module-origin and method-capability JSON gate | authorized; pending server run | A failed child requires a pure wheel dependency runtime; a passed child rejects stale spawn import as the explanation for old model-info shape. |
| 910C-053C | exact attested guarded task venv | Execute the first valid encoder-capture-to-compiled-decode transition experiment | One fresh E1 service: no-audio nested-stage gate, then distinct same-signature A/B/C | blocked on 910C-053E | A correct/deferred, B correct/capture, C correct/replay sequence rejects the narrow encoder transition hypothesis; a correct A then garbled B/C confirms it. |
| 910C-054 | Omni `dccb36d4`; SGLang `1d5aa4b8e`; no server edit | Correct graph-break implementation for stateful NPU decode attention | Exact10 C70 A0/T1/A1 attribution | completed; correctness passed, compile regressed performance | WER remained about 0.0164--0.0165. A0 p95 4.875 s at 35.71/s; T1 p95 5.294 s at 30.00/s; A1 p95 5.984 s at 35.26/s. A1 reported 1,636 graph-break log records and 3,336 recompiles; low compile coverage and guard/queue serialization remain performance blockers. |
| 910C-055 | same qualified correctness stack; prefix compile max changed to 8 | Test whether compiling all decode buckets through 8 improves coverage | Fresh-process startup/capture before performance | failed at startup OOM | The legacy prefix policy attempted cumulative compile buckets `[1, 2, 4, 8]` with about 9 GiB available and exhausted HBM. This is an allocation-policy failure, not proof that batch sizes above 2 cannot compile. |
| 910C-056 | Omni `fcc2fb64`; SGLang `edc504ff6` on `1d5aa4b8e` | Allocate compile memory to observed high-benefit decode buckets without compiling every smaller bucket | Retained-artifact memory accounting, then serial M1/M2 sparse-bucket correctness and conditional C70 performance | authorized; pending server run | Preserve at least 1.25x observed peak live KV-token capacity, select H1/H2 from the recorded C70 histogram, require exact compile bucket attestation, and stop on test failure, OOM, correctness/fallback failure, or insufficient cleanup/headroom. |
| 910C-056A | correction handoff containing this row; Omni `fcc2fb64`; SGLang `edc504ff6` | Replace ambiguous fractional KV tuning with an explicit token-cap budget and test only one high-value compile bucket | Fresh-process corrected M1: `[1,2,H1]`, 140-request correctness, conditional exact10 C70 | authorized; pending server run | The first M1 used both 64 and 70, did not reconcile requested 0.80 with reported 0.837/46.5-GiB KV allocation, and did not measure its claimed compile increment. Set `max_total_tokens=max(32768,ceil(1.5*peak_live_kv_tokens))`, attest resolved allocation before capture, and do not retry another bucket/fraction after failure. |
| 910C-056A M1c | Omni `8c95ad32`; SGLang `edc504ff6`; no server edit | Explicit 32,768-token KV cap plus sparse compile buckets `[1,2,70]` | 140-request correctness and exact10 700-request C70 | passed; best performance so far | Peak live KV was 17,423 tokens, so 32,768 retained more than 1.5x headroom. WER was 0.0161/0.0164 with zero garbled output; C70 p95 1.442 s, throughput 59.88/s, RTFx 598.8, HBM 22%. Hard target remains missed by 2.88x latency and 2.34x throughput. |
| 910C-056B | Omni `8c95ad32`; SGLang `edc504ff6`; no server edit | Separate KV right-sizing from compile coverage, then spend recovered HBM on all decode buckets | Serial K0 `[1,2]`, F13 all 13 buckets, and conditional reproducibility repeats | completed; M1c remains the only reproducible winner | K0 was correct but slower: p95 4.873 s, 33.18 req/s, WER 0.0165. F13 first measurement was correct at p95 1.504 s and 57.59 req/s, but its required second fresh-process repeat hung in warm-up. Thus F13 is not eligible; retain M1c `[1,2,70]`, 32,768 tokens, 0.80 static fraction, p95 1.442 s, 59.88 req/s, WER 0.0164 as the fixed `910C-057` baseline. |
| 910C-057 | Omni `b50f82f8`; SGLang `27b246532`; no server edit | Independently remove the stateful decode-attention compile break and narrow the FIFO guard from the whole generation forward to exact graph dispatch | Serial AC explicit-state attention, GS graph-scoped guard, and conditional AG composition | failed; neither candidate promotable | Preflight passed. AC ran 140/140 but all outputs were garbled with WER about 1.44, rejecting the explicit-state custom-op design. GS reported a runtime abnormality without the required first exception/boundary evidence. AG did not run. Retain M1c `[1,2,70]` at p95 1.442 s and 59.88 req/s as the qualified baseline. |
| 910C-057R | no code change; retained `910C-057` artifacts only | Classify the GS runtime abnormality before any guard code revision | Read-only extraction of first exception/boundary, guard/graph/request counters, and cleanup from retained logs/events | authorized; no service or NPU run | Return the first project-owned traceback or, if none exists, the last bounded event timeline and no-progress duration. Do not call the GS issue a deadlock, edit code, or rerun hardware until ownership is established. |
| 910C-057R GS | Omni `b50f82f8`; SGLang `27b246532`; no server edit | Retained graph-only guard evidence | Cold concurrency 8, 140 correctness, and C70 | failed at C70 | Cold concurrency 8 passed 8/8; correctness passed with WER 0.0161 and zero garbled output; C70 made no progress for at least 90 seconds with 70/70 pending. This confirms a graph-only-scope liveness regression, not by itself a defect in the FIFO primitive. |
| 910C-058 | Omni implementation `f9aeabad` plus handoff `05bbd106`; SGLang `4db662590`; no server edit | Guard the complete SGLang raw model execution region instead of graph execute alone | Focused/full tests followed by the model-scope hardware gate | failed; all requests stalled | `scope=model` produced an immediate liveness failure. The returned summary did not include ticket-correlated acquire/release counts or the active holder, so it does not establish FIFO contention as the mechanism. Do not promote or rerun this scope before retained-evidence classification. |
| 910C-058R | no code change; retained `910C-058` artifacts only | Identify the first unreturned ticket and distinguish reentrant self-wait, holder-side device stall, guard state-machine failure, or an observation gap | Read-only ticket/event/model-info join | completed; holder-side NPU graph-update stall | Cold conc8 completed 70/70 with final next=serving=440 and outstanding=0. A later warmup stalled at conc3: the scheduler held a generation model ticket inside `replay_with_input_update`, graph replay returned, and its background update thread remained blocked in `graph.update()` while request builders waited for later encoder tickets. This is not a ticket state-machine failure. Reject graph/model scopes, retain qualified forward scope, and fix update/replay ordering in the SGLang NPU backend before revisiting guard narrowing. |
| 910C-059 | handoff commit containing this row; SGLang `e4d18390a`; no server edit | Replace the NPU backend's Python update-thread/join overlap with opt-in same-thread ordered `NPUGraph.update()` then `replay()` | Serial focused/full tests; Arm O with qualified forward guard; conditional fresh graph-scope arm under the fixed M1c profile | authorized; pending server run | Set `SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE=ordered` and require complete ordered update/replay marker pairs with zero update-thread markers. Arm O must independently pass batch-one, cold conc8, 140 correctness, and exact10 C70. Only then test graph scope in a fresh process; never retry model scope or change memory/token/compile settings. CUDA backends are outside this change. |

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
