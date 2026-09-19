# Minimal cycle: repair collector and capture pageable once

Revision 1. Supersedes only the no-retry execution boundary of the original
minimal-cycle packet for the one attempt below. Existing gate/pinned results
at 8637b77c remain reported passes; the old pageable run remains
timeout_unclassified because its quoted GDB source command never loaded the
helper. Preserve every original artifact.

The serving workload functions child/overlap are unchanged. Collector now loads
the helper with GDB's Python runpy.run_path(..., run_name='__main__'), records its
exact argv, checks snapshot phases and post-detach state, and reports collection
failure explicitly. Helper already detaches in finally: remove the redundant
outer detach. The loader supports spaces and quotes as Python path literals.

## CPU gate (no torch import, no NPU work)

    python scripts/npu/test_probe_graph_h2d_cycle.py
    python scripts/npu/probe_graph_h2d_cycle.py --preflight --out evidence/wf009-cycle-loader-preflight

On the server, real-GDB loader test must run, not skip. Preflight must exit 0;
collector.json ok=true, rc=0, detached=true; queue.jsonl has maps/identity/
inventory/nonempty stack/coverage. Confirm the owned CPU child is reaped.
Inspect coverage for its requested PID and no missing threads. If anything
fails, stop before NPU and return the evidence. Do not silently change ptrace
policy/install tools. No automatic preflight/workload retries.

## One pageable run after that gate

Confirm previous device/process baseline and unchanged dependency identities,
queue flags and CANN environment. Preserve 8637b77c's gate/pinned artifacts and
record the new instruction/script/helper hashes separately.

    python scripts/npu/probe_graph_h2d_cycle.py --arm pageable --out evidence/wf009-cycle-pageable-recapture

One fresh child only. Same capture, warmup, 50 ms delay and 120-second parent
timeout as before. No gate/pinned reruns, extra arms or repeated trials.
The ordered initialization inside pageable is part of its unchanged workload.
Reuse the original packet's classification and device cleanup requirements.

Return the timeout snapshot's consumer stack, update/copy stacks and Repository
objects, maps, raw compute/update handles, coverage and collector.json. If native
code is synchronizing compute and updater waits for its Repository, describe
that evidence. The sole graph event and update-after-Begin code constrain the
hypothesis, but a pending-task-to-event link still needs evidence; do not claim
that existence of queue.jsonl alone proves the production root cause.

If this run completes, report non-reproduction and output checks; stop. If the
collector fails again, stop and retain logs rather than spend another workload.
