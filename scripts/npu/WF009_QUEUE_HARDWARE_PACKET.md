# WF-009 Q1: one targeted queue-stall snapshot

Revision Q1. Diagnosis, not qualification. Supersedes historical A/B multi-round
instructions for this operation. ONE fresh service and ONE cold conc8/140 run;
no automatic retry, no fix variant. Operator owns hardware baseline recovery.

## Question

Which operation prevents the Host Repository from draining: task callback,
release-queue backpressure, or empty-notification failure? Establish the actual
Repository identity first. Two wait frames alone do not establish a cycle.

## Frozen identities

- Serving SGLang: b950878e03f07c63bdca050e8a854503ebfa7058.
- Serving Omni B: 424ea1c4b822ebb63b12a3ff7261e9e438096887.
- Collector ONLY: queue_snapshot_gdb.py from diagnostic commit
  7fdabe9567611ec8915df7adf0766dc1391fc5db. Invoke by absolute path from a
  separate diagnostic checkout; do NOT serve the diagnostic branch by accident.
- Instructions: this file at its published commit; record file SHA256 separately.
- torch_npu 2.10.0.post2, torch 2.10.0+cpu, Python 3.11.10, CANN 9.0.1.
- libtorch_npu.so SHA256:
  6f000ee3be32c15d062c9fc7a5c0889e9033e2b1959800608b7a3819249010f6.
- Reported CPU preflight REPORT.md SHA256:
  01e90877019222cc388fd2e740960273c692e54cddc8ce04456d0c7e040cce10.
  Require tested helper identical to frozen helper; retain the preflight scripts
  and command logs. 49 assertions do not qualify AArch64 decoding.

This is a recurrence of the unchanged failing B arm, not an exact repetition of
D1's diagnostic hooks. No hook repairs or new in-process tracing are included.

## Preconditions / workload

Operator confirms idle allocated device and ownership; record baseline HBM,
health, driver version if available, namespace/PID mapping and TracerPid.
Do not touch old unrelated GDB PID 3421810 or fusion_result.json. If any actual
service target is already traced, stop. No device reset or host-wide kill.

Require clean tracked serving worktrees and actual import paths/hashes for
backend, submission module, encoder graph/service and model files. Preserve
the retained B1 launcher, model, cache policy, generation settings and logging.
Do not upgrade dependencies or change TASK_QUEUE_ENABLE, PER_STREAM_QUEUE or
ASCEND_LAUNCH_BLOCKING. Save only these allowlisted raw values; effective values
remain unknown unless supported by actual runtime evidence. Record inherited
values even when launcher did not set them. Do not silently select defaults.

SeedTTS EN 140, ordered evaluation_input_sha256:
9f631ab78d8bf3e19ab82a9c099826a850a0f103ae08d96db62561a71ec14815.
Cold conc8/140, no warmup. compile=false; prefill=breakable; decode=full;
encoder graph=true; disable_cuda_graph=false; mrr=64; graph_max_bs=64;
pre-LM encoder worker active. No exact10s corpus, eager fallback switch,
ordered variant, cache-transfer mitigation or in-process graph trace hooks.
Record executable launcher/harness and input hashes; discrepancy stops launch.

## One run and trigger

Create a NEW explicit evidence directory. Save owned PID/start-time identities
and unbuffered server/client output there. Independent external watchdog:
readiness limit 15 minutes; workload wall limit 5 minutes; first 60 seconds
without request/batch progress while work is outstanding triggers snapshot.
Watchdog must allow capture before terminating service. Readiness failure is
reported, not automatically retried. 140/140 means non-reproduction, not repair.

On stall, collect /proc status/stat/maps and per-thread comm/wchan/syscall plus
a bounded py-spy dump (10 seconds). Resolve current scheduler and blocked-copy
LWPs in the correct namespace; never reuse D1 numeric IDs. If copy is not blocked,
say so; collect scheduler and all consumer candidates without inventing a peer.

Set collector-process variables WF_QUEUE_TIDS to fresh target LWPs,
WF_QUEUE_SNAPSHOT to a NEW absolute JSONL filename, WF_QUEUE_SECONDS=15.
Use the CPU-preflight-validated GDB invocation: disable user/local/auto-load
scripts and debuginfod, pagination off, attach exact owned PID then source the
absolute frozen helper. External timeout 25 seconds. Do not call inferior
functions, write registers/memory, or use gcore/tensor dumps.

After detach, immediately record TracerPid and process state. If target remains
traced/stopped, stop collection and notify operator; do not attach again or claim
clean recovery. External GDB termination safety remains runtime-specific.

If first capture is incomplete, permit ONE bounded supplemental attach on the
SAME stalled process, not a new workload (25 seconds max). Prioritize omitted
acl/release candidates or newly identified dependency threads. Obtain bt60,
frame PCs and readable unwound registers. Record unknown when unavailable.
Do not begin another whole-process bt40. Save mappings so unnamed PCs can be
resolved offline. No ad hoc field offsets: extra queue/object memory decoding
requires binary/source/ABI justification, otherwise defer as unknown.

## Required observations / decisions

- Both wait PCs hit? Repository candidates and mutex addresses equal or unequal?
  Report unwind errors; same function alone is insufficient.
- Main read/write indices, need_empty/status and eventfd identifiers, if decoded.
- Actual consumer stack/current callback. Names merely select candidates.
- If in PushToReleaseQueue: release-worker stack and justified queue state, or
  unknown. The helper does NOT automatically decode ReleaseQueue or mutex owner.
- If main queue is empty while waiter sleeps: retain notification/eventfd evidence;
  a single field snapshot alone does not prove a lost wakeup.
- If callback blocks: preserve its dependency stack/PC; do not infer from API name.
- If consumers absent/truncated: diagnostic incomplete, not a proven hypothesis.
- If 107033 occurs instead: preserve first error and identities; classify a
  different failure signature. Do not continue to manufacture a silent stall.

No absence-of-log claim becomes a zero replay/fallback counter. Partial output
is not corpus correctness qualification. Do not apply a root-cause patch here.

## Cleanup and return

After bounded evidence capture, SIGTERM owned client/server; allow 60 seconds,
then SIGKILL only precisely verified owned processes if necessary. Record forced
shutdown separately. Enumerate owned children using saved PID/start-time lineage;
do not match broad command strings. Residual device resources go to operator.
No second run after cleanup under this packet.

Return identities/packet hash, preflight binding, raw vs effective queue settings,
completed count and trigger, both wait/object results, consumer/release stacks,
coverage gaps, TracerPid/state after each attach, cleanup and evidence hashes.
Keep full logs/private data on server. Outcome is mechanism evidence obtained,
incomplete diagnosis, or non-reproduction, never an automatic stability PASS.
