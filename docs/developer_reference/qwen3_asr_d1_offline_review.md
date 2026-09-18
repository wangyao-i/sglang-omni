# WF-009 D1-OFFLINE revision 1

Scope: existing artifact analysis and CPU-only checks. ZERO NPU workloads,
service launches, device resets, runtime environment changes or live attaches.
Do not import torch_npu merely to inspect source; do not modify site-packages.
Second hardware opportunity remains reserved for an evidence-directed fix.

## Source findings (not installed-binary attestation)

Public Ascend/pytorch v2.10.0 NPUStream.cpp NPUStream::stream() has a per-stream
queue branch and a default-repository branch. In the latter, different streams
can drain the same default Host queue before returning their own stream handles.
This can explain why private encoder stream alone did not resolve the stall.
NPUQueue.cpp MakeSureQueueEmpty holds mu_empty across an eventfd_read; Dequeue
signals efd_empty after a successful queue read and emptiness check. This matches
the reported waiting primitives, but installed library/source/offset alignment
is necessary before claiming the exact lock or repository is identified.

Also determine whether scheduler is blocked converting NPUStream to aclrtStream
BEFORE invoking the actual CANN End API. A Python End frame does not establish
that CANN End was entered. Do not patch CANN on that assumption.

Sources inspected 2026-09-18 (moving branch, not post2 build identity):
https://raw.githubusercontent.com/Ascend/pytorch/v2.10.0/torch_npu/csrc/core/npu/NPUStream.cpp
https://raw.githubusercontent.com/Ascend/pytorch/v2.10.0/torch_npu/csrc/core/npu/NPUQueue.cpp

## Server artifact processing

1. Preserve original D1 report/log/trace/maps/tasks/commands files and hashes.
   Record whether GDB timed out and which requested commands completed. 5521
   entries in tasks.json do not imply all 5521 native backtraces were captured.
2. Run on the existing full GDB output (substitute its real retained path):

```bash
python scripts/npu/summarize_stall_stacks.py "$NATIVE_LOG" --out "$EV/offline-stacks.json"
```

   This reads text only. Review ALL blocks selected as queue_consumer or gil,
   plus scheduler LWP 3446125 and request-builder LWP 3446411. Return full selected
   stacks, not just matching lines. Regex categories are not root-cause verdicts.
   If consumer stacks were omitted by timeout, say missing; do not invent idle.
3. Read existing maps and identify installed libtorch_npu / runtime shared
   objects. Save sha256sum and readelf -n build IDs. Use existing objdump or
   addr2line with saved module-relative offsets where possible; do not install
   symbols or treat current public source as installed source. Disassembly is
   read-only and does not execute NPU operations. Keep raw paths on server.
4. Recover effective TASK_QUEUE_ENABLE / per-stream queue configuration ONLY
   from saved launch commands/environment/config and matching installed source.
   Unset is not the same as zero. No new imports/context initialization to query it.
5. Inspect saved consumer stacks for queue callbacks blocked on stream/event
   synchronization, GIL, allocator locks or runtime calls. Distinguish which
   mutex/function is observed from hypotheses about ownership. Without core or
   retained registers/memory, mutex addresses cannot be recovered after exit.

## Hook coverage gap: CPU/static review

D1 patched graphs.NPUGraph class methods and graphs module Begin/End globals.
Existing instances of the SAME class ordinarily see replaced class methods;
creation before install alone is not sufficient to explain missing logs.
Local CPU wrapper tests pass, but do not reproduce the full server import path.
No replacement/reload was found in the narrowly searched local NPU backend and
Qwen3-ASR source. This is not a whole-process absence proof.

Inspect installed graphs.py and package exports AS TEXT: do the NPUGraph methods
resolve the patched globals, or retain another function/class reference? Search
recorded imported source trees for reload, alias-module loading, class replacement,
instance-bound replay/update overrides and process-specific installation timing.
Compare all PID/TID trace files and timestamps: hooks.installed must precede the
observed graph calls; a single-PID match alone is insufficient. Check log location
and hashes against the actual D1 launch identity. Do not reimport the NPU package
to claim historical in-process identity. A CPU surrogate can test an identified
mechanism but cannot prove that mechanism occurred in D1.

The local diagnostic wrapper is not changed speculatively. Do not spend another
run merely to repair logging. Preserve this gap independently of useful native
and transfer evidence.

## Return and decision boundary

Return: log completeness; selected consumer/GIL stacks; installed library IDs and
matching code/offset evidence; queue-mode evidence; hook findings with explicit
observed/inferred/unknown labels. No need for a new acceptance workload.
If the queue consumer waits on an operation dependent on the unsignaled graph
event, map that cycle before choosing transfer/queue ownership changes. If it waits
for GIL, map the holder and blocked call before proposing a binding fix. If stacks
are absent, state the irrecoverable gap and stop rather than auto-consuming budget.
No new fix, queue-disable flag or ordering variant is authorized by this packet.
