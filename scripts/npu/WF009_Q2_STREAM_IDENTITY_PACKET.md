# WF-009 Q2: identify the copy stream causing consumer synchronization

Revision 1. One fresh B service / one cold conc8/140 workload; no fix variant.
User permits evidence-directed hardware diagnosis after code analysis. This
packet is NEW; Q1 and its artifacts remain immutable. No claim of root cause.

## Source findings motivating the run

torch_npu source v26.0.1-pytorch2.10.0:
- NPUStream.cpp:enCurrentNPUStream records current stream in paramStream.
- OpParamMaker.cpp copies paramStream; AsncExecFunc passes it to MemcopyAsyncFunc.
- graphs.py captures ExternalEvent wait/reset. Update runs operator, End, then
  event.record. NPUGraph.cpp End implicitly converts NPUStream to ACL stream;
  that conversion can drain the shared Host Repository.

CANN runtime v9.0.1 02522b7a:
- ApiErrorDecorator::MemcpyAsync may synchronize a stream before copying when
  its memory-location classification requires it.
- F7's saved Q1 return PC 0xf01c8 already supports the synchronization branch
  under the candidate mapping. Do not spend a run just re-proving this branch.
- A nested object loaded from this+8 can have a DIFFERENT vtable. Slot identity
  alone does not establish recursion or identical sync/async copy semantics.

Missing fact needing live state: the synchronized stream/copy identity and its
progress over time. Does it match decoder compute, decoder update, encoder, or
another stream? Source cannot recover this historical object's value.

## Frozen identities and preflight

Serving: SGLang b950878e03f07c63bdca050e8a854503ebfa7058;
Omni B 424ea1c4b822ebb63b12a3ff7261e9e438096887. Tracked clean; explicitly set
PYTHONPATH to intended roots without empty entries and verify imported paths and
hashes from the actual serving launch. Do not serve the diagnostics checkout.

Collector: queue_snapshot_gdb.py from 4932ffb7d6a90ba56e7b9c67b5010241d7aaa68a,
SHA b11568b91b954b25801c6c09fe0370c6550bdcc68c4e0cf377a73cb8dcf6ef9e.
Reported delta CPU preflight 66/66 qualifies maps persistence, not runtime field
decoding. Instructions: this file's published commit and Git-blob SHA separately.

Require installed binary SHA:
- torch_npu: 6f000ee3be32c15d062c9fc7a5c0889e9033e2b1959800608b7a3819249010f6
- runtime_v100: e728be132988845e0294637d1bbe927df21aadcf8a03d87e4e96661e0a857583
Record actual maps/load bias; NEVER reuse Q1 addresses or infer uniform ASLR.
Same Python/torch/torch_npu/CANN profile as Q1. No package/queue-mode changes,
in-process tracing, monkeypatches, device reset or sudo installation.

Before workload, prepare the extra GDB command sequence and verify its syntax
offline. Stop before launch if collector cannot read selected unwound registers
without inferior calls. Registers/memory unavailable in optimized frames must
remain unknown; no assumption that volatile argument registers survived.

New exclusive evidence directory. Owned PID/start-time lineage, independent
watchdog. Fix previous harness variable-name and stage.pid/scheduler.pid errors
BEFORE launch; review cleanup against all owned spawn children. No broad pgrep
kill. Unrelated gdb/processes untouched. Require operator-confirmed idle baseline.

## Workload / bounded trigger

Exactly Q1 model/launch/cache/generation configuration: compile=false,
prefill=breakable, decode=full, encoder_graph=true, disable_cuda_graph=false,
mrr=64, graph_max_bs=64, pre-LM worker active. No warmup, SeedTTS EN140 conc8,
evaluation SHA 9f631ab78d8bf3e19ab82a9c099826a850a0f103ae08d96db62561a71ec14815.
Readiness 15 minutes, workload 5 minutes; 60s without progress while outstanding
requests triggers capture. No automatic retries. If 107033/other failure appears,
preserve first failure and stop, not manufacture the desired signature.

## Two snapshots of the SAME stalled process

Reserve 120s AFTER the stall trigger for collection, separately from workload
timeout. At T1 save full maps, status, tasks, allowlisted raw queue env, py-spy,
and existing source markers (compute/update streams). Resolve fresh LWP IDs.
Use existing tested GDB attach settings (auto-load/debuginfod disabled).

Highest priority: consumer inside rtMemcpyAsync, scheduler, encoder and blocked
builders. BEFORE any all-thread traversal, save consumer bt60 and unwound
registers of each candidate runtime frame. Confirm PC membership from actual
maps and exact SHA first. Save raw register provenance and errors.

At F7 return PC base+0xf01c8 (only for the matching binary):
- frame.read_register x19..x29, SP, PC (unwound, not innermost registers).
- binary-derived values x24=stream argument, x26=decorator receiver,
  x21=dst, x22=src, x23=dstMax. Treat as candidates until prologue/dataflow
  confirms preservation at this call; do not infer ABI argument values blindly.
- bounded reads at frame x29+0x90 (1 byte flag), +0x98 (4 byte corrected kind),
  +0x78 (8 byte count); this+8 and each receiver vptr (8 byte each).
- Do NOT read src/dst tensor contents or arbitrary pointer chains. Follow only
  the proven impl_/vptr links, mapping checked; retain private pointers locally.
- Source/destination classification temporaries belong to an earlier returned
  classifier frame: do not read it as a still-live frame.

At F1/F2: preserve x19..x29 and raw frame/PC. Only decode this+0xeba (u16),
+0xd8c/+0xd90 (u32), +3816 (u32) if this-register recovery is independently
justified by that function's prologue and matched binary dataflow. Otherwise
unknown, NOT guessed x19. Same rule for task target, retry counter and timeout.
Do not call methods (including IsProcessTimeout) from GDB.

Then run frozen maps-preserving helper. Hard external GDB cap 35s each attach;
helper internal budget 15s. Flush targeted register records before broad stacks.
Record TracerPid=0 and non-stopped state after detach; otherwise stop and report.

Wait 15–20s with process resumed, then T2 targeted capture of the SAME consumer,
scheduler and runtime objects. T2 is required, not conditional on elapsed<30s.
No second workload. If signature changes, capture the change instead of forcing
the same frames. Do not start an attach that cannot fit remaining collection
budget; report incomplete second capture explicitly.

## Decisions

- Mapping differs: reject historical fixed-PC decode, preserve raw maps/stacks.
- Copy stream matches a source marker: record exact handle type and provenance;
  do not compare Python object id to runtime Stream* without a proven conversion.
- Stream/target unchanged and completion fields static T1/T2: persistent stream
  wait supported, NOT yet proof of a graph-event cycle or lost notification.
- Consumer changes/completes: do not call F1 a permanent spin.
- Corrected kind/source path differs from embedding H2D: do not attribute queued
  copy to request-builder. It may be operator metadata or another copy.
- Full event-cycle proof additionally needs pending task/event identity and its
  record owner. If absent, report the next precise missing link; no dependency fix.
- 140/140 without stall: non-reproduction, no extra run or stability qualification.

## Cleanup / return

After capture, SIGTERM owned processes, 60s grace then exact owned PID/start-time
SIGKILL if needed. Enumerate owned spawn children from recorded lineage even if
reparented. Verify port, no traced/stopped target, owned process inventory and HBM
baseline. No device reset/host-wide kill; unresolved residue goes to operator.

Return both timestamps, raw/decoded identities, stream-marker comparison, copy
kind/count (not tensor data), loop-state deltas, bytecode/source alignment limits,
coverage, cleanup and artifact SHA manifest. No partial WER qualification.
