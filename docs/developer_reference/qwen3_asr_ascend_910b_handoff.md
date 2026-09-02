# Qwen3-ASR 1.7B on Ascend 910B: hardware handoff

This page is the source of truth for the local-development and isolated-Ascend
qualification effort. It is written for the local SGLang-Omni developer and
the operator of the isolated 910B server. Update this page when a gate changes;
the companion task pages contain executable procedures, not competing status
or support claims.

## Scope and status

Target topology: one Ascend 910B, one Qwen3-ASR-1.7B stage, BF16, no model
quantization, and no tensor or data parallelism.

Status at base commit `e7d876b28326c55d777ae62e1c3650b816785d8c`:
**first failure captured; bounded process-topology A/B pending**.

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
the operator. Those probes rule out a general A3 `conv2d` failure, but they did
not reproduce the daemon property of the actual Omni stage process.

### Current diagnosis and ownership

The strongest diagnosis is a SGLang-Omni process-lifecycle defect, pending a
one-variable A/B confirmation:

1. `StageGroup.spawn()` currently constructs every stage process with
   `daemon=True` in `sglang_omni/pipeline/stage_workers.py`.
2. Python daemon processes are not permitted to create child processes.
3. CANN knowledge-base initialization attempts to create a
   `multiprocessing.Manager`, which itself creates a child process.
4. The likely failure chain is therefore: daemonized ASR stage -> first
   GE/TBE-compiled operator -> Manager child creation rejected -> CANN EC0009
   -> `GEInitializeV2` failure -> outer `AclSetCompileopt` error 500001.

Items 1 and 2 are established implementation/runtime facts; items 3 and 4 are
supported by the captured nested error and the missing condition in the five
passing probes, but remain a high-confidence diagnosis until the next A/B run.
The first `conv2d` is most likely only the first operator to trigger compiler
initialization. Do not classify it as an unsupported convolution without new
operator-level evidence.

If the daemon A/B confirms this chain, ownership returns to this repository:
stage-process lifecycle must allow a backend to create managed child
processes. The existing `StageGroup.shutdown()` already owns explicit
join/terminate/kill cleanup, but a production change still requires focused
tests for normal shutdown, startup failure, parent termination, and stuck-child
cleanup. If the A/B does not remove EC0009, record the next daemon ancestor and
reclassify before changing versions or compiler controls.

### Runtime and workaround conclusions

- A3 does not imply a CANN 8.x requirement. CANN 9.0.x is used by documented
  A3 stacks, and CANN 9.0.1 with `torch_npu` 2.10.0.post2 is a plausible matched
  ecosystem stack. The present evidence does not identify version mismatch as
  the cause.
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
  environment only if separation is still needed after the daemon A/B.

## Next bounded diagnostic task

Keep the A3 environment, base commits, eager flags, input, and single-request
gate fixed. Change only the process daemon property, and start from a fresh
process for each arm:

1. Extend the passing spawn-child probe with `daemon=True`; print the child
   daemon state and attempt `multiprocessing.Manager()` before running the same
   `conv2d`. Preserve the direct Python exception and the operator-wrapped
   exception separately.
2. In a diagnostic-only SGLang-Omni checkout, change the stage process from
   `daemon=True` to `daemon=False`, start the same eager profile, and run only
   batch 1. Do not continue to larger gates in that process.
3. If batch 1 passes, implement the repository fix with CPU process-lifecycle
   regression tests, then repeat eager batch 1 from a fresh server process. If
   EC0009 persists, locate the next daemonized parent. If a different failure
   appears, preserve it as the new first failure and classify it independently.
4. Return to the default graph gate only after eager batch 1, two-request,
   sequential, restart, health, and memory gates pass. The earlier
   `PagedAttentionOperation` failure remains a separate graph qualification
   item unless its full nested log contains the same Manager/EC0009 signature.

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
| 910C-000 | diagnostic edit pending | `e7d876b2` | A3; CANN 9.0.1; torch 2.10.0; torch_npu 2.10.0.post2; SGLang 0.5.18 (Git HEAD missing); triton-ascend 3.2.1 | default startup, then eager batch 1 | failed | Default decode graph: `PagedAttentionOperation`; eager request: GE Manager EC0009 -> error 500001 |

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
- [vLLM Ascend device-family mapping](https://github.com/vllm-project/vllm-ascend/blob/main/setup.py)
- [SGLang A3 installation examples](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/getting-started/installation.mdx)
