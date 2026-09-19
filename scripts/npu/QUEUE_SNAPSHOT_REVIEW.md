# Targeted queue snapshot: preflight required

Post-Q1 correction: Q1 used helper 7fdabe95, which read full maps but retained
only the torch_npu mapping. The updated helper flushes a full maps JSONL row
BEFORE library hashing, and records gdb.solib_name per frame where available.
Keep the new raw mappings on-server. Q1 cannot be repaired retroactively and
its frozen packet must not be silently rerun with this changed helper. Local
policy tests now total 5, including retained mappings on library-inspection
failure. The reported 49-check CPU preflight covers the OLD helper; a delta
preflight of the new maps row/shared-library fields is still required.

Diagnostic only. No production changes. This helper is NOT yet qualified for
the next NPU run: real Linux GDB attach/unwinding/detach preflight is pending.
Local WSL has Python but no GDB; no packages were installed.

## What changed

The legacy collector is unchanged. Source queue_snapshot_gdb.py in an already
attached GDB. It writes JSONL incrementally, before any full-thread traversal:

1. Library hash/base and allowlisted raw queue/blocking environment.
2. Fresh LWP inventory and explicitly requested missing LWPs.
3. Requested scheduler/copy LWPs, then ALL acl/release-named candidates.
4. Bounded shallow stacks of remaining threads and explicit missing coverage.

Only exact D1 libtorch_npu SHA + AArch64 + the two known wait return PCs allow
reading the unwound x19 and selected Repository fields. Unknown layouts are
not decoded. Mutex owner is unknown; matching mutex addresses can establish
same-object correlation without guessing the libc owner field. Repository
values remain candidates if unwinding is unreliable. No tensor contents are
read. Consumer names are candidates, never an identity proof.

## CPU-only acceptance before hardware

Use only an explicitly owned disposable Linux CPU child. Do not attach a
service, use sudo, change ptrace policy or install tools implicitly.

- Child should have multiple native LWPs including acl/release-named workers
  and ordinary workers; return fresh native IDs, not historical D1 IDs.
- Set WF_QUEUE_TIDS to at least two owned child LWPs and WF_QUEUE_SNAPSHOT to
  a new absolute JSONL output path. WF_QUEUE_SECONDS defaults to 15 (max 30).
- Start GDB with local/user/auto-load scripts and debuginfod disabled, pagination
  off. Attach the exact owned child, then source the helper. Use an external
  bounded subprocess timeout (e.g. 25 seconds); don't rely on GDB's internal
  between-thread deadline to interrupt a stuck debugger operation.
- Verify requested-first ordering, every acl/release candidate, frame PC/name
  output, inventory/coverage consistency, and graceful unknown for absent NPU
  library. Repeat with a small internal budget to verify missing coverage.
- Verify TracerPid=0 and the child is not stopped after normal and timeout
  paths. On failure, stop and report; never promote failed detach as a pass.
  Cleanup only this owned CPU child. The helper's finally-detach does not
  guarantee detach after external termination of GDB.
- This preflight does NOT qualify AArch64 register recovery or NPU Repository
  decoding; those remain checks on the future matching failure snapshot.

Do not start an NPU workload automatically after CPU preflight. Return its
evidence and freeze the hardware packet first. No stream/queue flag change,
dependency patch, service launch, push or device reset is included here.

## Local tests actually run

`python scripts/npu/test_queue_snapshot_gdb.py`: 5 passed. Tests cover ordering,
fresh positive ID validation, library identity guard, selected-field decoding
with mocks, and maps persistence before a failed library inspection. They do not
execute GDB or prove debugger compatibility.

## Reported post-Q1 CPU delta preflight

Operator reports 66/66 checks on helper SHA256
b11568b91b954b25801c6c09fe0370c6550bdcc68c4e0cf377a73cb8dcf6ef9e,
including maps-before-identity, frame-to-library consistency and repeated
timeout/detach checks. New artifacts are in post-q1-delta/ on the server.
These are reported results; raw artifacts were not independently inspected here.
Real GDB library-inspection failure injection and NPU decoding remain untested
by this CPU preflight. Do not infer Q1 rerun authorization.

The operator reports overwriting the OLD 49-check preflight's raw artifacts;
only its report survives. Keep that loss explicit, not a reproducible historical
raw-evidence claim. The newer tests do not reconstruct the lost old artifacts.
Future runs must create a new evidence directory exclusively and stop if it
already exists. Do not delete or reconstruct prior outputs to obtain a clean run.
