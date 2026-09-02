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
**A3 eager functional baseline passed; decode graph qualification blocked at
ATB `PagedAttentionOperation` setup**.

The first remote run used `Ascend910_9382`, which the current Ascend ecosystem
identifies as A3 hardware. It is therefore recorded as the derived run
`910C-000`, not as proof for the original 910B target. The 910B qualification
remains unstarted until the same gates run on the intended device class.

| Area | Repository state | 910B evidence |
|---|---|---|
| Ascend installation | NPU manifest, precheck, and installation guide are implemented | Precheck passed on A3; not yet run on the target 910B |
| Qwen3-ASR model path | Single-stage model, batching, pre-LM encoder, SSE output, and long-audio upload chunking are implemented | A3 eager batch 1, two-concurrent, ten-sequential, health, shutdown, and restart gates passed after repairing the OpenCV environment; not yet started on 910B |
| Generation graph | Enabled by the Qwen3-ASR defaults and delegated to SGLang | A3 prefill capture succeeded and decode capture failed independently at ATB `PagedAttentionOperation`; not yet verified on 910B |
| Encoder graph | Implemented with `torch.cuda.Stream`, `torch.cuda.CUDAGraph`, and `torch.cuda.graph`; enabled by default | Incompatible by inspection; the first baseline run must preserve the complete failure evidence |
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
  `triton-ascend` 3.2.1, and SGLang-Omni 0.1.3 at `e7d876b2`. The exact SGLang
  Git HEAD was not returned and remains required for a reproducible final
  runtime matrix.
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

### Independent decode graph failure

Run `910C-003` repeated default startup after the OpenCV repair. Prefill graph
capture succeeded, but decode graph capture failed on its first, largest bucket
at batch size 64 with approximately 9.28 GiB available. The complete log had
zero occurrences of EC0009, Manager instantiation, `GEInitializeV2`,
`EOFError`, `AclSetCompileopt`, error 500001, `libGL`, or `cv2`. The first
failure is instead ATB `PagedAttentionOperation setup failed` from
`OpParamMaker.cpp` and `AtbCommon.cpp` during SGLang's decode NPU graph
capture.

This is a separate graph qualification item. SGLang owns the NPU graph runner,
attention backend, capture buckets, and the call into the ATB operation;
ATB/op-plugin owns the native operation setup result. Ownership between those
components remains unresolved until the native diagnostic identifies whether
the rejected input is a batch shape, memory/workspace budget, graph-capture
constraint, or invalid operation parameter.

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

Run identifier: `910C-004`. The only variable is **decode graph capacity**.
Keep the repaired OpenCV environment, A3 hardware, runtime versions, model,
prefill graph mode, torch-compile mode, memory fraction, and all other settings
fixed. Use a fresh process and evidence subdirectory for each capacity.

Before the first arm, record the exact SGLang Git HEAD, resolved ATB and
op-plugin versions/library paths, `sgl-kernel-npu` version, confirmation that
`opencv-python` is absent, the installed headless OpenCV version, and a
successful fresh-process `cv2` import. Confirm no stale process, port owner, or
non-baseline HBM allocation.

1. Run paired capacities `1`, `16`, `32`, and `64` in that order. For capacity
   `N`, set both `--asr.engine.max_running_requests N` and
   `--asr.engine.cuda_graph_max_bs N`; Omni requires the graph cap to cover the
   admission limit. Do not use the upstream `--cuda-graph-max-bs-decode` name
   directly through the Omni CLI.
2. For every arm, record the resolved capture-bucket list, first bucket
   attempted, last bucket completed, available HBM immediately before capture,
   capture duration, and the first complete failure. SGLang captures decode
   buckets from largest to smallest, so a 0-progress failure identifies the
   configured maximum bucket.
3. Stop the ladder immediately if capacity 1 fails. If a larger capacity fails
   after a smaller one passes, retain both logs and stop; that boundary is the
   input for the next shape-versus-memory experiment. Do not compensate by
   changing memory fraction in this run.
4. On the first failing arm, rerun that same capacity once with ATB and ASDOPS
   diagnostic logging enabled (`ATB_LOG_TO_STDOUT=1`,
   `ATB_LOG_LEVEL=DEBUG`, `ASDOPS_LOG_TO_STDOUT=1`, and
   `ASDOPS_LOG_LEVEL=DEBUG`) plus a writable CANN process-log directory. Do not
   enable per-operation synchronization, alter task-queue policy, or patch
   native libraries. Preserve the first native ATB/op-plugin error code and
   message below `OperationSetup`, if any.
5. If native logging still reports only the two wrapper frames, stop and return
   the sanitized shapes, dtypes, strides, capture bucket, available HBM, exact
   ATB/op-plugin versions, and full native stack. The next change then belongs
   in the SGLang NPU backend as a minimal diagnostic/reproducer; do not edit
   installed SGLang or CANN files in place.
6. After every arm, stop normally and verify service processes, ports, device
   health, and HBM return to baseline. A new orphan process is a gate failure.

Do not run concurrency, performance, realtime, version-stack A/B, or memory
fraction A/B in `910C-004`. Passing capacity 64 restores the default graph gate
only; capacity 70 and the performance workload remain later qualification
steps.

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
4. After eager correctness is stable, qualify the default generation graph
   before running the
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
| 910C-003 | `aba09fe3`; environment-only repair | not applicable | Same A3 stack; `opencv-python` removed; `opencv-python-headless` 5.0.0 reinstalled | eager functional gates, restart, default graph retry | eager passed; graph failed | Eager gates and cleanup passed; independent decode ATB `PagedAttentionOperation` setup failure at batch 64 |
| 910C-004 | pending | pending | Same repaired A3 stack required | decode graph capacity ladder and native ATB diagnostics | pending | pending |

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
