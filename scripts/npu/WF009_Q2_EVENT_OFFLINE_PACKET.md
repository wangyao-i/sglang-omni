# WF-009 Q2 event-backend and retained-evidence audit

Operation: read-only offline discovery, revision 1. Budget: 30 minutes. This is
not a Q2 rerun or a hardware qualification packet. Use a new evidence directory;
never overwrite Q1/Q2 or CPU-preflight artifacts.

## Question

Does the installed ExternalEvent implementation use ordinary event waits or
ValueWait/ValueWrite, and do retained Q2 artifacts identify the signal awaited
by decoder compute and the update record responsible for satisfying it?

Q2 reports persistent implicit StreamSynchronize on decoder compute, while the
scheduler waits for Repository drain. It does NOT yet prove that compute waits
for the signal that this scheduler update would record. Do not call this a
lost wakeup or a completed dependency cycle.

## Boundaries

Read saved artifacts and installed files with text tools/readelf/nm/objdump.
Do not import torch/torch_npu, start a service/workload, attach a process, call
an inferior function, change configuration/dependencies, reset a device, or
edit serving code. No new runtime instrumentation in this task.

Record packet commit and SHA256, library identities and exact commands. Verify
the Q2 library identity against its own saved maps/identity; do not reuse an old
load bias. Keep private paths and raw logs on the server.

## Source anchors and bounded checks

Reference torch_npu: Ascend/pytorch commit
8751b36d5d6959e499e6bf6530c1928060ced030 (v26.0.1-pytorch2.10.0).
Reference runtime: cann/runtime commit
02522b7ab84c12baab996317128079ac9970027d (v9.0.1).
These are source references, not claims of whole-binary equivalence.

1. Inspect the installed library's ExternalEvent backend selection. The source
   interface/AclInterface.cpp IsExistValueWaitAndWrite checks resolution of BOTH
   aclrtValueWait and aclrtValueWrite. AclrtCreateEventWithFlag uses device memory
   for EXTERNAL when both exist. AsyncTaskQueueInterface.cpp selects WAIT_VALUE
   and WRITE_VALUE; OpParamMaker.cpp ValueWaitResetFunc waits for 1 then writes 0;
   record writes 1. Otherwise ordinary event APIs are used. Establish installed
   control flow and resolver/provider evidence if available. Export presence
   alone is capability evidence, NOT proof of Q2's actual branch. If cached
   function pointers or branch data were not retained, explicitly say UNKNOWN.

2. Inspect existing Q2 artifacts only for graph/update-record identity and the
   pending compute task. Ordinary backend needs event identity; value backend
   needs wait/write device-address identity, value and comparison flag. Runtime
   memory_task.cc MemWaitValueTaskInit stores devAddr/value/flag in TaskInfo;
   ConstructLastSqeForMemWaitValueTask logs stream_id/task_id/devAddr/value/
   sqHeadPre/flag and emits CONDS_SUB_TYPE_MEM_WAIT_VALUE. Do not assume those
   logs were enabled. A compute MODEL_EXECUTE may require following an internal
   captured stream; a host cursor difference of one is not one event wait.

3. If retained records permit it, correlate the pending wait identity with the
   event belonging to the scheduler's blocked update record. graphs.py orders
   operator -> graph_task_update_end -> record.event.record(update_stream).
   Absence of a record log is not proof it never ran unless coverage is proven.
   A matching address alone without iteration/lifetime and pending-task context
   is not a complete dependency proof. Never dereference a device address as
   host memory. Do not invent TaskInfo offsets from a different build.

## Decision and return

- Backend confirmed but no pending-task/event data: report the exact missing
  identity fields. Stop; do not spend hardware to fill them automatically.
- Both identities and dependency ordering retained: provide the complete chain
  with artifact locations and competing explanations; no automatic fix.
- Binary/source or resolver mismatch: retain both identities and stop the
  dependent inference, without modifying the environment.

Return a short table: claim / observed evidence / inference / unknown. Include
whether Q2's raw files actually contain the necessary metadata, not a proposed
command presented as an executed result. Preserve original hashes. Success for
this task is a truthful evidence inventory, not a mandatory backend verdict.
