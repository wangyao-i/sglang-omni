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
with torch compile disabled passed capture and single-request replay; the named
encoder-eager candidate passes two concurrent requests after both inputs are
warmed but hangs only when uncached encoder work overlaps enabled generation
graph execution; disabling generation graph removes the hang, while compiled
generation and encoder graph remain unqualified. Synchronous request building
also removes the hang; disabling only prefill graph removes it while decode
graph remains captured, making prefill-eager/decode-graph the bounded-
concurrency candidate. That ladder is currently blocked before service startup
because the isolated server cannot stage the pinned SeedTTS dataset through its
enterprise proxy**.

The first remote run used `Ascend910_9382`, which the current Ascend ecosystem
identifies as A3 hardware. It is therefore recorded as the derived run
`910C-000`, not as proof for the original 910B target. The 910B qualification
remains unstarted until the same gates run on the intended device class.

| Area | Repository state | 910B evidence |
|---|---|---|
| Ascend installation | NPU manifest, precheck, and installation guide are implemented | Precheck passed on A3; not yet run on the target 910B |
| Qwen3-ASR model path | Single-stage model, batching, pre-LM encoder, SSE output, and long-audio upload chunking are implemented | A3 eager batch 1, two-concurrent, ten-sequential, health, shutdown, and restart gates passed after repairing the OpenCV environment; not yet started on 910B |
| Generation graph | Enabled by the Qwen3-ASR defaults and delegated to SGLang | With torch compile disabled, A3 capacities 1/16/32/64/70 captured and each passed one decode replay smoke; prefill graph was isolated as necessary for the cold-encoder hang and is disabled in the current candidate; with compile enabled, batch 1 failed at Dynamo/triton-ascend and batch 64 failed at ATB `PagedAttentionOperation`; maximum-bucket concurrent replay remains untested |
| Encoder graph | Implemented with `torch.cuda.Stream`, `torch.cuda.CUDAGraph`, and `torch.cuda.graph`; enabled by default | A3 capture failed for every attempted bucket because captured-stream synchronization memcpy is unsupported; each bucket explicitly stayed eager |
| Pre-LM encoder service | NPU tensors use the default device stream; the dedicated stream path is CUDA-only | A3 eager functionality and restart stability passed; cold work hangs only when it overlaps asynchronous prefill-graph execution in the tested matrix; prefill eager with decode graph retained passed the two-request wave |
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

Formal performance prerequisites are not yet met. The prefill-eager/decode-
graph candidate has passed only one two-request cold-overlap wave; bounded
concurrency and maximum-bucket replay remain unproven. The performance contract
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
| 910C-013 | `eb5bd9fd`; no runtime service started | not applicable | Exact `910C-012` prefill-eager/decode-graph stack planned; pinned EN Parquet transferred; local-Parquet and content-unique benchmark support prepared locally | fresh-process SeedTTS EN cold-input ladder at concurrency 8/16/32/64/70 | blocked before startup; repository/client-environment sync pending | Network staging failed first; verified local Parquet then exposed the Arrow 25.0.0 aarch64 reader failure; no ladder level ran; reject monkeypatch/fake-cache/meta export; resume only after syncing the loader commit, restoring the server package set, and passing the isolated pyarrow 25.0.1 full scan plus `--unique-audio` preflight |

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
