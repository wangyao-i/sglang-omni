# Minimal graph-update / H2D cycle: design, not an executable hardware packet

Question: can an H2D operation that implicitly synchronizes compute prevent the
Host queue from draining, thereby preventing the graph update from signaling
the ExternalEvent needed by that same compute replay?

Scope: isolate this mechanism, not qualify ASR or retrospectively prove every
historical stall. Keep 107033/shared-update-stream collision separate.

## Reuse and source evidence

Use Ascend/pytorch 8751b36d5d6959e499e6bf6530c1928060ced030
test/npu/test_aclgraph_update.py:test_ifa_update as the task-group/capture API
reference. It already constructs one attention operator, explicit ExternalEvent,
task-group handle, workspace and out buffers. Its update-before-replay ordering
is a baseline, not evidence that the replay-first variant below is qualified.
Choose valid small tensor dimensions/lengths and compare against eager; do not
copy the test's length constants without checking the actual operator contract.

CopyKernelOpApi.cpp:copy_between_host_and_device_opapi submits nonblocking H2D
before process_non_blocking_copy. CachingHostAllocator.cpp may synchronize for
unregistered host memory; the runtime memcpy path can independently synchronize.
Thus Python non_blocking=True does not prove a non-synchronizing path.

## Minimal objects

One NPU, one captured operator, one ExternalEvent, one task-group handle,
separate compute/update streams, one preallocated H2D destination, and two Host
workers (copy and update). No encoder, cache, SGLang, service or corpus. The H2D
buffer must NOT be an input to the captured operator: avoid introducing a new
data hazard as the explanation for failure. Hold every buffer alive until exit.

## Stages and sole experimental variable

1. Gate: legal capture and ordinary update/record-before-replay work correctly
   against eager. Then replay-before-update with NO injected copy must complete
   correctly. A gate failure stops the dependent experiment, not a hypothesis win.
2. Control: preallocated pinned CPU bf16 source, 143360 elements, preallocated
   matching NPU destination. Replay on compute; copy worker submits H2D on that
   same compute stream; update worker performs Begin/op/End/record on update.
3. Suspect: identical setup, values, shape, streams and Host scheduling protocol,
   but source is ordinary CPU memory. Do not vary stream ownership or queue flags.

Run each arm in a fresh process. Allocate/pin/warm up before measurement. Do not
join the copy worker, synchronize compute, or wait for the replay to finish before
allowing update: that would manufacture a Python-side deadlock. Use CPU-only
phase signals. copy.enter is NOT proof of queue insertion. Any scheduling delay
must be identical in both arms, bounded and explicitly labelled a forced
interleaving window, not the measured production schedule. No unbounded retries.

## Required discriminator

Completion alone is insufficient. Suspect failure must show consumer synchronizing
this compute stream, updater waiting for the same Repository, and the captured
wait whose matching record is still downstream of update. Here the event/handle
is deliberately unique, making identity tractable. A pinned control must show
the intended registration/non-synchronizing branch; is_pinned alone is not CANN
classification proof. Both output and copied data are checked after completion.

If pinned also fails, do not add locks/order variants. If both pass, record not
reproduced; do not exonerate production. If suspect fails before enqueue, or only
the Python coordination blocks, reject that run as a different mechanism.
Even matching differential results establish the reduced mechanism first; link
it back to Q2's installed paths before calling the production root cause proven.

## Safety and execution boundary

This is design only: no workload authorized by this file. An executable script
must first receive CPU policy checks for worker independence, phase reporting,
external-process watchdog and owned-process cleanup. Runtime gate and trial
budgets must be frozen before launch; no multi-hour wait or all-thread dump.
Use existing targeted collection, preserve full maps, and never print tensors
or query stream handles inside capture. Do not modify installed dependencies,
reset devices, or kill unrelated processes. Stop after the first discriminating
failure and return evidence; no automatic expansion of the test matrix.
