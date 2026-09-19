# Qwen3-ASR graph stalls: causal findings and repair scope

This note combines locally inspected versioned source with operator-reported
hardware results. Raw server snapshots have not been independently imported.
It does not qualify arbitrary workloads or identify every historical failure.

## Two different failure mechanisms

### Stream task-group conflict (107033)

CANN runtime 02522b7ab84c12baab996317128079ac9970027d,
src/runtime/feature/src/context/context.cc:4674 StreamBeginTaskUpdate takes the
stream's task-group mutex and rejects a status other than NONE with the exact
reported message, "Unable to start update tasks because the stream is busy."
Begin sets UPDATE; End restores NONE. The mutex protects each API call, not an
entire caller-side Begin/operator/End transaction. Another check separately
rejects updating a task-group handle already being updated.

Thus different graph handles do not make interleaved update transactions on
one stream legal. Nor does reducing the number of Host threads guarantee that
transactions cannot overlap. Historical reports show shared update streams and
this error, but the first failing run did not capture both transaction identities.
Do not claim its exact counterparty was directly observed. Do not describe the
runtime as having no locks, or identify 107033 as GRAPH_TASK_UPDATE_BEGIN based
only on the Python call site; errcode_manage.cc maps this condition to
ACL_ERROR_STREAM_TASK_GROUP_STATUS.

### Host queue and graph handshake cycle (silent stall)

torch_npu source 8751b36d5d6959e499e6bf6530c1928060ced030:

- NPUStream.cpp:601 preserves each task's raw stream in paramStream but, with
  per-stream queues disabled, enqueues into the device's default Repository.
- NPUStream.cpp:364 drains this Repository on ordinary stream-handle access.
  NPUGraph.cpp Begin/End use that conversion before entering the CANN API.
- NPUQueue.cpp ReadQueue runs the queued callback before advancing its read
  index; Dequeue signals efd_empty only after the necessary queue progress.
- graphs.py capture inserts ExternalEvent wait/reset. Update executes
  Begin/operator/End before recording the corresponding event.

Runtime api_impl/api_error.cc:1637 MemcpyAsync can call
StreamSynchronize(stm,-1) for the classified unregistered-memory path before
copying. non_blocking=True at the Python layer does not forbid this behavior.

The resulting invalid dependency is:

    compute replay needs update's signal
      -> update needs Host queue drain
      -> queue consumer's H2D needs compute completion
      -> compute replay needs update's signal

The library's valid stream-local synchronization becomes a liveness problem
when combined with the application's outstanding replay/update handshake and
the framework's shared Host queue. This does not establish a broken mutex,
lost eventfd wakeup, or defective CANN synchronization implementation.

## What the minimal probe established

Workload at 8637b77c, collector repaired at 7f1dd0a0. AST comparison confirmed
child/overlap/stop_owned unchanged. Gate and pinned arms completed with output
checks; ordinary CPU source stalled twice after copy.enter/update.begin.enter.
The first capture failed due to the diagnostic GDB quoting defect and contributes
no native evidence. The second reported 722/722 stacks and clean teardown:

- consumer in runtime memcpy's implicit StreamSynchronize chain;
- copy thread waiting on efd_empty while holding the Repository mutex;
- updater waiting on that same mutex before graph_task_update_begin;
- shared Repository read=14/write=15, need_empty=1.

The native snapshot did not retain the consumer's runtime stream pointer or
pending device event/value address. Their correspondence is code-derived in
the reduced program: one graph/event, an already synchronized initialization
and warmup, one overlapping copy submitted to compute, and this iteration's
sole event record downstream of the blocked update. This is a constructive
dependency argument under the APIs' intended semantics, not a direct device
task inspection. Source pinning's differential supports it but is not itself
proof of the runtime classifier's branch. Production Q2 separately reported
the synchronized stream matching decoder compute.

## Why encoder-owned cached transfer addresses this path

Inspected Omni 7dbecfc57f8b1e74bd779b9b3a02137dfc9ca1ca. Its encoder_service.py
and request_builders.py are identical to delivery 40822af1 for this comparison.
Previously cache hits called attach_embedding from request-building threads.
The revised builder submits a transfer entry and returns DeferredAdmission.
The encoder worker copies under its private stream context, synchronizes that
stream, and only then resolves the admission future. The entry retains the CPU
source. Late follower callbacks publish metadata without initiating device work.

For this path, the copy no longer implicitly synchronizes decoder compute.
Transfers are processed before the same worker starts a new encoder graph
submission. Prior encoder submissions have already issued their event records.
record_stream(default) protects allocation lifetime; it does not introduce a
stream wait back to decoder. This removes a specific edge, not merely a thread.

The six successful 140-input runs remain bounded, operator-reported validation.
Existing CPU tests cover worker ownership and future completion after sync,
not NPU execution. This analysis does not certify other decoder H2D operations,
all encoder operations, or the separate 107033 condition. A production regression
must bind this transfer contract to the relevant backend/runtime configuration.

## Development invariant

While replay awaits a future Host update/signal, an operation needed to submit
that signal must not synchronously wait for the replay, directly or through a
shared submission queue. Account for implicit waits in transfers and stream
handle conversion; Python non_blocking flags and private device streams alone
do not establish this invariant. Transfer completion must precede admission,
and source/destination lifetimes must cover asynchronous use.
