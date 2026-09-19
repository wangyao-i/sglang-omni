# Minimal graph-update / H2D cycle probe, revision 1

Purpose: test the proposed cycle without ASR, encoder, cache or HTTP serving.
This is a mechanism probe until native evidence reproduces the failed path.
Source/design: WF009_MINIMAL_CYCLE_DESIGN.md; upstream attention test at
Ascend/pytorch 8751b36d5d6959e499e6bf6530c1928060ced030.

## Preparation

Use the previously identified CANN 9.0.1, torch_npu 2.10.0.post2 and device.
Record current package/library identities, imported paths, instruction commit,
script/helper SHA256, device baseline and actual environment before execution.
Preserve the working CANN/TBE environment. Do not modify queue or launch flags.
The script uses process-visible npu:0; operator verifies its physical mapping.
Run the five CPU coordination/cleanup checks first:

    python scripts/npu/test_probe_graph_h2d_cycle.py

Ensure GDB is available and existing attach policy permits the owned child to be
collected. No installs or ptrace-policy changes in this packet. The inherited
helper already has a Linux CPU attach preflight. The new wrapper's checks include
a real CPU child timed out and reaped, plus a mocked TERM-to-KILL escalation.
They do not qualify NPU teardown or the complete Linux debugger integration.

## Bounded sequence

Fresh child per command, fresh evidence directory (script refuses reuse). Run
commands sequentially and inspect each result before the next:

    python scripts/npu/probe_graph_h2d_cycle.py --arm gate --out evidence/wf009-cycle-gate
    python scripts/npu/probe_graph_h2d_cycle.py --arm pinned --out evidence/wf009-cycle-pinned
    python scripts/npu/probe_graph_h2d_cycle.py --arm pageable --out evidence/wf009-cycle-pageable

At most three child workloads, one per arm, no retries. Gate must finish with
ordered_gate.pass AND probe.pass, correct graph output and copy contents. A gate
failure stops all dependent work. Pinned failure stops the pageable arm. Report
operator or environment errors separately; do not repair/retry automatically.

The graph has one task group and ExternalEvent, two heads, query length 1,
KV capacity 16, head dimension 64; captured KV length 8 is updated to 16 and
checked against eager. Copy source has 143360 bf16 elements; its destination is
independent of graph inputs. Pinned and pageable differ only in CPU pinning.
After replay, copy thread signals entry; the updater waits a fixed 50 ms then
proceeds independently of copy completion. This is a forced scheduling window,
not native-enqueue proof. Each arm warms the copy path before this interval.

The parent allows 120 seconds total including import/capture. Timeout triggers
one targeted snapshot (25-second external bound), saves full maps and target
status, then terminates its exact child (TERM, 5 seconds, KILL if needed).
This parent is independent of the child's Python GIL. Timeout is classified as
timeout_unclassified, never automatically as deadlock or hypothesis success.
No all-thread legacy collector is used. Inspect TracerPid and device baseline
after cleanup; failure to reclaim owned resources stops the sequence.

## Interpretation

- Pinned passes, pageable stalls: useful only if consumer native stack shows
  implicit compute-stream synchronization and updater waits on its Repository.
  One deliberately unique captured event simplifies, but does not by itself
  prove, that the blocked compute task is waiting for this event's record.
- is_pinned is PyTorch metadata, not proof of runtime memory classification.
  If registration/non-sync branch is unobserved, report the differential as
  consistent with the hypothesis, not a proven branch or production root cause.
- Both pass: not reproduced within this schedule. Do not expand repetitions.
- Pinned fails: stop; examine gate, branch and failure location. No lock/order
  variants or feature disabling are authorized as substitutes.
- Failure in import/capture/operator or Python coordination: different mechanism,
  not a reproduction of the production queue cycle.

Return exact identities, arm result/last phase, output correctness when completed,
snapshot path/coverage, error codes, normal versus forced termination and baseline.
Preserve all original evidence. No production code, dependency or device reset.
