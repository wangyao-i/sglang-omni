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
**first failure captured; CANN Manager child failure detail pending**.

The first remote run used `Ascend910_9382`, which the current Ascend ecosystem
identifies as A3 hardware. It is therefore recorded as the derived run
`910C-000`, not as proof for the original 910B target. The 910B qualification
remains unstarted until the same gates run on the intended device class.

| Area | Repository state | 910B evidence |
|---|---|---|
| Ascend installation | NPU manifest, precheck, and installation guide are implemented | Precheck passed on A3; not yet run on the target 910B |
| Qwen3-ASR model path | Single-stage model, batching, pre-LM encoder, SSE output, and long-audio upload chunking are implemented | A3 eager service became ready, but its first request failed; not yet started on 910B |
| Generation graph | Enabled by the Qwen3-ASR defaults and delegated to SGLang | A3 prefill capture succeeded and decode capture failed at `PagedAttentionOperation`; not yet verified on 910B |
| Encoder graph | Implemented with `torch.cuda.Stream`, `torch.cuda.CUDAGraph`, and `torch.cuda.graph`; enabled by default | Incompatible by inspection; the first baseline run must preserve the complete failure evidence |
| Pre-LM encoder service | NPU tensors use the default device stream; the dedicated stream path is CUDA-only | A3 first request failed while GE initialized for the first audio-tower `conv2d`; functionality is not established |
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

## Derived A3 baseline: run 910C-000

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

### Current diagnosis and ownership

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

Therefore the current classification is **CANN GE/TBE knowledge-bank
initialization failure in a full service process; owning component unresolved**.
The failure may be a CANN defect, or a CANN compatibility defect triggered by
process state established by SGLang/SGLang-Omni. It is not currently evidence
for an Omni daemon-lifecycle fix, a CANN version mismatch, or an unsupported
`conv2d`.

The EC0009 text is not a root cause. CANN wraps exceptions from
`multiprocessing.Manager()` with that message. In the closely matching public
issue ICVT2X, the underlying exception is an `EOFError` while the Manager
parent waits for the server child to return its address, meaning the short-lived
Manager child exited before completing its startup handshake. A separate
public case shows the same CANN wrapper around Python's spawn bootstrapping
`RuntimeError`. The next gate must capture this run's original Python exception
and Manager-child exit reason before assigning ownership or changing versions.

The operator also found and force-stopped a stage process orphaned from the
initial validation for more than 12 hours, with PPID 1 and approximately 55 GiB
of chip-0 HBM. It contaminated earlier retry observations but was not the batch-1
cause: after removal, HBM returned to the approximately 3 GiB idle baseline and
a fresh non-daemon run reproduced the same failure. The orphan is retained as
a separate shutdown/reap defect relevant to the later stability gate.

### Runtime and workaround conclusions

- A3 does not imply a CANN 8.x requirement. CANN 9.0.x is used by documented
  A3 stacks, and CANN 9.0.1 with `torch_npu` 2.10.0.post2 is a plausible matched
  ecosystem stack. The present evidence does not identify version mismatch as
  the cause.
- A CANN KB/GE integration defect is now the leading candidate, but is not yet
  confirmed as a CANN 9.0.1-specific bug. The public ICVT2X report exhibits the
  same Manager/EC0009/GE failure class on a different CANN and PyTorch stack and
  has no public patch, so it supports the failure class rather than an exact
  version conclusion.
- Do not downgrade or upgrade CANN/`torch_npu` as the next variable. If a later
  version experiment is needed, replace the complete vendor-supported stack
  and record every component, rather than changing only one package.
- No verified environment-variable workaround is accepted. `TASK_QUEUE_ENABLE`
  changes task dispatch and is disabled by synchronous launch; HCCL controls do
  not address this pre-communication single-device failure; no authoritative
  `GE_NO_NEED_*` bypass has been established. Changing
  `jit_compile=True` is also not the next experiment because it still requires
  GE initialization.
- Installing a headless OpenCV package merely to unlock direct SGLang serving
  is lower priority and must not mutate the frozen environment. Use a cloned
  environment only if separation remains necessary after the Manager failure
  is classified.

## Next bounded diagnostic task

Run identifier: `910C-002`. Keep the A3 hardware, runtime stack, model, eager
flags, input, and batch-1 request fixed. Begin only after confirming no stale
service/stage process, port owner, or non-baseline HBM allocation. Use a new
server-local evidence directory and a fresh service process for every arm.

### Gate A: capture the CANN Manager exception without pre-initialization

1. Set `ASCEND_PROCESS_LOG_PATH` to a writable directory inside this run's
   evidence directory, `ASCEND_GLOBAL_LOG_LEVEL=0`,
   `PYTHONFAULTHANDLER=1`, and `PYTHONUNBUFFERED=1` before service startup.
2. Use a diagnostic-only non-daemon ASR stage so Python permits a Manager
   child. In that stage, enable `multiprocessing.util` DEBUG logging and wrap
   `multiprocessing.Manager` only to log its full original exception before
   re-raising it. Do not call Manager early, change its context, suppress the
   exception, or edit CANN/`site-packages`.
3. Immediately before the failing `conv2d`, from the same request-builder
   thread, log the sanitized values of current-process daemon state,
   `multiprocessing.get_start_method(allow_none=True)`, default-context start
   method, thread name/main-thread status, `sys.executable`, `sys.argv[0]`,
   current directory, `tempfile.gettempdir()`, and the SIGCHLD handler.
4. Send exactly one smoke request. Preserve the service stderr, the complete
   Python traceback intercepted around Manager, CANN plog for the stage PID and
   any short-lived child PID, and any permitted kernel OOM/segfault record.
   Record cgroup `pids.current` and `pids.max` in addition to shell ulimits.
5. Stop after this request even if it succeeds. Restore all diagnostic source
   edits and verify the checkout, processes, ports, and HBM baseline.

If the wrapper does not expose an exception more specific than EC0009, use one
additional fresh process with `strace -ff` limited to process, signal, file,
and IPC events to identify the Manager child exit/exec/permission failure. Do
not return the full trace; retain it server-side and report only the relevant
syscall, errno or terminating signal.

### Gate B: plain Manager control in the exact service thread

Use another fresh non-daemon diagnostic process. At the same request-thread
location, run one ordinary `multiprocessing.Manager()` creation and a trivial
proxy operation, record its original result, then deliberately stop before
executing `conv2d`. Do not use the process for a subsequent functional claim,
because the control can initialize or mutate global multiprocessing state.

- Plain Manager fails: preserve its raw traceback and child exit reason. The
  service process state, rather than CANN alone, is sufficient to reproduce the
  failure.
- Plain Manager passes while Gate A's CANN Manager fails: classify the boundary
  as CANN KB integration/embedded-Python state and prepare a vendor reproducer.
- Either arm reports a new first failure: stop and return that failure; do not
  proceed to model loading, version changes, concurrency, or performance.

Only if these two gates cannot locate the boundary, perform fresh-process
Manager probes at one initialization checkpoint per run: stage entry,
accelerator-environment preparation, SGLang NPU backend initialization, model
load before ready, and request-builder entry. A probe process ends immediately
after its result. Load the complete model in a standalone reproducer only if
this bisection first changes from pass to fail at model loading.

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
4. Only after eager correctness is stable, run the
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
| 910C-002 | pending | pending | Same A3 stack required | capture raw Manager failure; exact-thread plain Manager control | pending | pending |

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
