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

## Scope and status

Target topology: one Ascend 910B, one Qwen3-ASR-1.7B stage, BF16, no model
quantization, and no tensor or data parallelism.

Status at base commit `e7d876b28326c55d777ae62e1c3650b816785d8c`:
**A3 eager functional baseline and generation graph capacities 1 through 70
with torch compile disabled passed capture and single-request replay; bounded
concurrency, compiled generation, and encoder graph remain unqualified**.

The first remote run used `Ascend910_9382`, which the current Ascend ecosystem
identifies as A3 hardware. It is therefore recorded as the derived run
`910C-000`, not as proof for the original 910B target. The 910B qualification
remains unstarted until the same gates run on the intended device class.

| Area | Repository state | 910B evidence |
|---|---|---|
| Ascend installation | NPU manifest, precheck, and installation guide are implemented | Precheck passed on A3; not yet run on the target 910B |
| Qwen3-ASR model path | Single-stage model, batching, pre-LM encoder, SSE output, and long-audio upload chunking are implemented | A3 eager batch 1, two-concurrent, ten-sequential, health, shutdown, and restart gates passed after repairing the OpenCV environment; not yet started on 910B |
| Generation graph | Enabled by the Qwen3-ASR defaults and delegated to SGLang | With torch compile disabled, A3 capacities 1/16/32/64/70 captured and each passed one decode replay smoke; with compile enabled, batch 1 failed at Dynamo/triton-ascend and batch 64 failed at ATB `PagedAttentionOperation`; maximum-bucket concurrent replay remains untested |
| Encoder graph | Implemented with `torch.cuda.Stream`, `torch.cuda.CUDAGraph`, and `torch.cuda.graph`; enabled by default | A3 capture failed for every attempted bucket because captured-stream synchronization memcpy is unsupported; each bucket explicitly stayed eager |
| Pre-LM encoder service | NPU tensors use the default device stream; the dedicated stream path is CUDA-only | A3 eager functionality and restart stability passed; graph mode remains blocked before serving |
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

Run identifier: `910C-008`. Establish the named A3 candidate mode
**generation graph enabled at capacity 70, torch compile disabled, and encoder
graph explicitly disabled**. The only configuration change from `910C-007` is
`--asr.factory.enable_encoder_cuda_graph false`; the progressive request loads
below are validation levels, not configuration A/Bs.

1. Use a fresh process and keep the exact repaired A3 stack, model, SGLang
   `71de97b2`, `mem_fraction_static=0.837`, prefill settings,
   `--asr.engine.max_running_requests 70`,
   `--asr.engine.cuda_graph_max_bs 70`, and
   `--asr.engine.enable_torch_compile false` from `910C-007`. Add only
   `--asr.factory.enable_encoder_cuda_graph false`.
2. Before startup, require a clean tracked worktree, no service process or port
   owner, healthy device state, and idle-baseline HBM. Record exact revisions,
   the resolved runtime profile, and the server-local input-set fingerprint.
3. Require generation graph capture through bucket 70. Also require zero
   encoder graph capture attempts, zero `bucket stays eager` warnings, and zero
   encoder `aclrtMemcpy` 107030 errors. The encoder must be eager by explicit
   configuration, not by capture failure or runtime fallback.
4. Use at least two distinct frozen server-local clips with different expected
   output hashes, alternating them within every multi-request wave. Do not
   return paths, audio, or transcripts. Record pre-LM cache hit, miss, in-flight
   deduplication, queue, and batch counters so repeated-input effects remain
   visible. Do not enable a cache that was disabled in `910C-007` or change its
   configured size.
5. Run these functional levels in order with no retries: one request, one
   synchronized two-request wave, ten sequential requests, then one
   synchronized wave each at concurrency 8, 16, 32, and 64, followed by three
   synchronized waves at concurrency 70. Stop at the first HTTP failure,
   timeout, output-hash mismatch, cross-input contamination, graph fallback,
   forbidden error, device-health failure, or monotonic HBM growth.
6. At every concurrency level, report requested concurrency, successes and
   failures, client wall time, diagnostic latency distribution, peak scheduled
   and running requests, generation graph markers, encoder queue/batch/cache
   counters, and peak/settled HBM. These are functional diagnostics, not the
   exact-10-second performance result.
7. Bucket-70 replay passes only if a decode runtime event from a concurrency-70
   wave shows an actual running decode batch of 70 selecting generation NPU
   graph execution (`npu graph: True`) with no fallback. Seventy successful
   client requests without that same-event evidence are **inconclusive**, not a
   pass. Do not increase offered concurrency above 70 to force the condition.
8. After the final wave, verify service health, allow queues to drain, record
   settled HBM, stop normally, and confirm process/port release, device health,
   idle-baseline HBM, and no orphan process.

Do not run the formal exact-10-second benchmark, realtime workload, long soak,
compile-enabled A/B, encoder-graph implementation experiment, version change,
memory-fraction change, or source patch in `910C-008`. If the service cannot
form an actual decode batch of 70, preserve scheduler/encoder timing evidence
and stop; the next task will isolate admission, coalescing, or request-duration
limits without redefining this gate after the fact.

Formal performance prerequisites are not yet met. Maximum-bucket concurrent
replay is unproven, the named encoder-eager mode is not yet functionally
qualified, and the
repository performance task records that the NPU-aware exact-10-second manifest
harness has not yet been implemented and tested. Compile-disabled generation is
a valid explicit candidate mode, but it is not evidence that the repository's
compile-enabled default is supported.

Do not run performance or realtime measurements until the functional gates are
green. A server-side diagnostic edit is not a deliverable fix; any confirmed
change must be rebuilt locally, tested, committed in this repository, and
mapped in the evidence table.

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
| 910C-008 | pending | pending | Exact 910C-007 stack; compile disabled; encoder graph explicitly disabled | named candidate functional and bounded-concurrency ladder through 70 | pending | pending |

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
- [vLLM Ascend device-family mapping](https://github.com/vllm-project/vllm-ascend/blob/main/setup.py)
- [SGLang A3 installation examples](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/getting-started/installation.mdx)
- [CANN 9.0 process-log path](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0120.html)
- [CANN 9.0 application log level](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0122.html)
- [SGLang Ascend graph-capacity controls](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/mimo_v2_flash.mdx)
- [Ascend ATB and ASDOPS diagnostic logging example](https://gitee.com/ascend/MindSpeed-LLM/blob/59408f7f7520266976599912f8e35b97fb0c74d/mindie_ref/mindie_llm/atb_models/README.md)
