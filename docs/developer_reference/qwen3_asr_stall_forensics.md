# WF-009 D1: one high-information stall reproduction

Revision D1. Diagnosis only. Budget: ONE NPU workload in this packet; retain the
remaining opportunity for an evidence-directed fix. No A/O/B repetitions, no new
lock/order variants. User owns container/host cleanup and restoration of baseline.

## Identity and question

- SGLang unchanged: b950878e03f07c63bdca050e8a854503ebfa7058.
- Omni base B: 424ea1c4b822ebb63b12a3ff7261e9e438096887.
- Diagnostic code through e052becb06babf0714dca88e1aaad891945b1d65; subsequent
  packet/test-only commits are permitted only as the published frozen tip.
- Branch: qwen3-asr/update-stall-forensics. Record exact HEAD, diff (tracked clean),
  packet SHA256 and imported source hashes. Do not modify site-packages.

Question: what are decoder update_end and request-builder attach_embedding
actually waiting on in native code? Source request_builders.py:418 calls attach
on cached_embedding; service cache_device is CPU. Confirm actual source device
and native transfer behavior, not merely the Python .to frame. Two blocked calls
do not prove they wait on each other. Independent encoder stream has failed B1.

Diagnostic hooks are process-local Python wrappers installed by NPU encoder
service construction only when SGLANG_ASR_STALL_TRACE_DIR is set. They wrap
installed graphs.py Begin/End and NPUGraph replay/update. Encoder manually bound
update aliases are NOT claimed to be traced. Metadata brackets attach.to, including
current device/stream queries separately. IDs named *_pyid are Python object IDs,
NOT native task-group pointers. No tensors/repr/operator arguments are serialized.
No synchronization, device/stream switch or changed call order is added. Disk writes
and metadata queries DO perturb timing: a non-reproduction is not a fix/pass.

## Preflight WITHOUT consuming the NPU opportunity

1. Operator confirms task-owned container/host residue cleared and idle baseline
   restored. Record allocated device, health, driver version, host/container PID
   mapping and /proc status NSpid. Do not reset or kill unrelated processes.
2. Use clean worktrees and verify imports from the serving process. Preserve
   Python 3.11.10, torch 2.10.0, torch_npu 2.10.0.post2, CANN 9.0.1; unexpected
   identity requires reconciliation, not silent substitution.
3. Run CPU tests and native focused suites before loading the model:

```bash
cd "$OMNI_WORKTREE"
python scripts/npu/test_stall_trace.py
python scripts/npu/test_private_encoder_update_stream.py
python -m pytest tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py tests/unit_test/qwen3_asr/test_encoder_service.py -q
cd "$SGLANG_WORKTREE"
python scripts/npu/test_async_submission.py --omni-root "$OMNI_WORKTREE"
python -m pytest test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py -q
```

4. Set EV to a new explicit evidence directory on local server disk (not NFS).
   Ensure free space and readable/writable paths. Do not leave EV empty.
   In the SAME namespace/user/capabilities as the future scheduler collector:

```bash
cd "$OMNI_WORKTREE"
python scripts/npu/collect_stall_native.py --preflight --out "$EV/ptrace-preflight"
```

   This creates a disposable CPU-only Python child and uses installed py-spy and
   GDB to attach/detach. Require exit 0, Python frames, native #0 backtrace and
   TracerPid=0. Review native names/module addresses. If tools/ptrace are unavailable,
   stop BEFORE serving; ask operator to arrange access separately. Do not install
   dependencies, change ptrace policy or elevate privileges under this packet.
   Local Windows validation cannot establish Linux GDB functionality.

## One frozen workload and armed collector

Use exact prior launcher/evaluator and log full commands/hashes. SeedTTS EN 140,
evaluation_input_sha256:
9f631ab78d8bf3e19ab82a9c099826a850a0f103ae08d96db62561a71ec14815.
Cold conc8/140, no warmup; compile=false, prefill=breakable, decode=full, encoder
graph=true, disable_cuda_graph=false, mrr=64, graph_max_bs=64, pre-LM worker active.
No change to cache behavior, input order or decoding parameters. Unset WF-008
trace/serialization flags. Do not enable additional CANN debug modes without a
separate frozen instruction; preserve already available runtime logs and actual
log settings (absence of custom env vars alone does not prove logs do not exist).

Before service launch export SGLANG_ASR_STALL_TRACE_DIR="$EV/trace". Log hooks.installed
must appear in the actual scheduler PID's trace before sending load. Verify import
paths/hashes in that process; metadata alone in a launcher PID is insufficient.
Service readiness timeout 15 min. Source=encoder/decoder one-time markers retain
their original meaning; they are not progress counters.

After readiness, determine the scheduler PID in the SAME /proc namespace as the
collector using the hooks marker and saved process identity, not only the port
owner. Before load start the collector as a managed background process:

```bash
python scripts/npu/collect_stall_native.py --pid "$SCHED_PID" \
  --watch-trace "$EV/trace" --out "$EV/stall-native"
```

Do not run that command synchronously ahead of load: it watches up to 600 seconds.
It triggers once a recorded .enter phase has no subsequent record on its thread
for 20 seconds, then writes /proc thread wchan/syscall, Python stacks and all native
thread stacks (40 frames), loaded library paths and maps. Each external tool has
a 25-second timeout. GDB temporarily stops threads; it does not call inferior
functions or dump tensor data. Identify tracer-induced pauses separately.

Keep the independent response watchdog: 60 sec no completions stops NEW load;
absolute workload limit 10 min. It MUST NOT kill the server before the armed
collector finishes. Allow at most 60 seconds for collection after the watchdog
fires, then proceed to cleanup even on collection failure. If an exception occurs
before the auto-trigger, stop load and invoke collector directly with --pid/--out
before shutdown; never attach two collectors simultaneously. Save native error
logs immediately; preserve partial outputs in finally paths. Client must append
each completed response to JSONL and flush it, not wait for final summary.

If the workload finishes, stop/reap the task-owned idle collector; do not repeat
the workload. If still live, never terminate service until snapshot files and
commands.json are preserved. Check status-after: TracerPid must be 0; a stopped
or still-traced target needs operator handling. SIGTERM service with 60-sec grace,
then operator-owned forced cleanup if required, recorded separately. No host-wide
kill/reset. Keep model/input text, native addresses and private paths on server.

## Return and second-opportunity decision

Return exact identities/profile/input hash, complete/evaluated counts, diagnostic
hook coverage, trigger.json and the last 30 metadata records for scheduler and
request-builder, native stacks mapped by TID/NSpid, shared-library build IDs if
available (readelf -n on already mapped installed libraries, no package installs),
existing CANN log references, time bounds and cleanup. Do not call missing evidence
success. attach.stream_query stall is distinct from attach.to stall.

- CPU->NPU transfer plus native stream-sync/queue wait: map which stream and native
  dependency blocks; design a targeted ownership/transfer fix only if supported.
- Native allocator/host lock wait: compare mutex/queue addresses and owner threads;
  do not blame stream sharing from co-occurrence.
- update_end waits in runtime/event processing while request-builder is downstream:
  follow the graph event/queue dependency; don't delete .to just because it blocks.
- No recurrence or unusable native evidence: report inconclusive, preserve the
  remaining run; no automatic variant, longer stress, or claim that tracing fixes it.

Second run is NOT scheduled by this packet. Local review first selects ONE minimal
fix or justified runtime probe from observed native evidence. PRs remain drafts;
diagnostic wrappers must not enter shipping PRs.

References: GDB attach/detach and thread backtraces:
https://sourceware.org/gdb/current/onlinedocs/gdb.html/Attach.html
https://sourceware.org/gdb/current/onlinedocs/gdb.html/Threads.html
