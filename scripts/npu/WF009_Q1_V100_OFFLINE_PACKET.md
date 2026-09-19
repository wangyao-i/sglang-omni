# WF-009 Q1 V100-OFFLINE — revision 1

## Question and boundary

Can the installed CANN 9.0.1 libruntime_v100.so explain Q1 consumer frames
1–7 under load bias `0xfffc763c0000`, and what condition controls the yield loop?

This is **offline binary/source analysis**, not a workload packet. No service,
NPU import/workload, attach, reset, dependency installation, library changes or
process cleanup. The user permits evidence-directed hardware work, but first
exhaust this offline question. Do not rerun frozen Q1 with a different helper.
Use a NEW evidence directory; never overwrite the existing Q1/preflight reports.
Stop after one bounded analysis pass (30 minutes); return explicit gaps rather
than expanding to a new hardware run. No commands below execute the library.

## Frozen inputs and corrections

- Library: /usr/local/Ascend/cann-9.0.1/aarch64-linux/lib64/libruntime_v100.so
- Required SHA256:
  e728be132988845e0294637d1bbe927df21aadcf8a03d87e4e96661e0a857583
- No Build ID reported; use SHA, not nearest exported symbol, for identity.
- Existing Q1 consumer: tid 3721273, saved stack only; PID is no longer live.
- Public source reference: cann/runtime v9.0.1,
  02522b7ab84c12baab996317128079ac9970027d. This is not binary-build proof.

Independent arithmetic from reported addresses:

```
Q1 frame 1 PC                 0xfffc7667c9fc
candidate yield BL offset          0x2bc9f8
candidate return offset            0x2bc9fc
candidate ELF load bias       0xfffc763c0000
```

Only this one of the 16 reported direct yield calls gives a 4 KiB-aligned bias;
it is also 64 KiB aligned. This is a candidate, NOT recovered Q1 maps.
Assumptions: correct binary, valid unwind, direct yield call among the list.
Tail calls/interposition or a different loaded binary can invalidate the match.
Do not infer that frames 1–7 share one library from their proximity alone.

The prior report's 0x15c9fc uses Q1-PC **minus** 0x33c0000 minus D1-base.
With the stated D1-minus-Q1 delta, the plus calculation yields 0x68dc9fc.
Neither cross-process translation is evidence for this library's mapping.

| Frame | Q1 PC | Candidate ELF virtual address |
|---|---|---|
| 1 | 0xfffc7667c9fc | 0x2bc9fc |
| 2 | 0xfffc766846d4 | 0x2c46d4 |
| 3 | 0xfffc7667e920 | 0x2be920 |
| 4 | 0xfffc76696340 | 0x2d6340 |
| 5 | 0xfffc76454140 | 0x94140 |
| 6 | 0xfffc764a2ac4 | 0xe2ac4 |
| 7 | 0xfffc764b01c8 | 0xf01c8 |

## Read-only collection

Use installed readelf/objdump/nm. Missing tools are a stop, not install approval.
Save stdout/stderr and exit status in the new directory, retaining raw outputs.

```bash
LIB=/usr/local/Ascend/cann-9.0.1/aarch64-linux/lib64/libruntime_v100.so
sha256sum "$LIB"
readelf -lW "$LIB"
readelf -nW "$LIB"
readelf --debug-dump=frames "$LIB"
nm -D -C "$LIB"
objdump -d --no-show-raw-insn --start-address=0x2bc838 --stop-address=0x2bca50 "$LIB"
```

First check SHA. If different, stop the fixed-offset analysis. From PT_LOAD,
verify each candidate PC is within PF_X, using p_vaddr/p_memsz (not file size).
For string bytes, translate ELF vaddr through the containing segment's
p_offset + (vaddr - p_vaddr); do not assume vaddr equals file offset.

Disassemble bounded windows at the remaining return sites:

```bash
for pc in 0x2c46d4 0x2be920 0x2d6340 0x94140 0xe2ac4 0xf01c8; do
    objdump -d --no-show-raw-insn \
      --start-address="$((pc - 32))" --stop-address="$((pc + 32))" "$LIB"
done
```

Then use FDEs to obtain complete containing functions for frames 1–4 first.
Decode each caller PC-4: a direct BL target should enter the callee's containing
function or a demonstrated veneer; an indirect BLR does not recover its target
without additional evidence. Inlining/tail calls must be explicit exceptions,
not reasons to manufacture a missing edge. Only frame 1 should call sched_yield;
frames 2–7 should call the preceding function, not each call sched_yield.

## Required analysis / return

1. A row per frame: PC, candidate vaddr, executable segment, FDE range, PC-4
   instruction, direct target/indirect register, edge status and literal strings.
2. In [0x2bc838,0x2bca50): transcribe the loop, registers/field offsets it reads,
   retry threshold, exit/error/timeout paths, and state the condition required
   to exit. Preserve the whole function disassembly, not only the yield line.
3. Identify the caller functions using their own strings/control flow; do not
   identify them from a nearest exported std:: symbol.
4. Match to public source only after binary shape/strings support it. Locate the
   writer/operation that releases the wait. Separate code-level possibility from
   observed field values: Q1 did not save registers for this runtime loop.
5. If online state is necessary, list the smallest missing set (object register,
   field offsets, owner thread/task/stream identities) and which outcomes would
   distinguish the remaining hypotheses. Do not run that experiment in this task.

Local source scouting found these DIFFERENT yield mechanisms, not conclusions:

- feature/src/pool/buffer_allocator.cc:GetItemById waits while allocFuncState_
  is true and pool_[poolIdx] is null; allocation publishes that pool slot in
  AllocIdWithoutRetry. This is not simply 'pool exhausted'.
- feature/inc/reference.hpp:RefObject::IncRef retries REF_UPDATING/CAS.
- feature/src/stream/stream.cc:WaitForTask/StarsWaitForTask poll task reclaim;
  SynchronizeDelayTime and WaitConcernedTaskRecycled have different conditions.
- feature/src/device/raw_device.cc also yields during event cleanup.

The source MemcopyAsync path in feature/src/launch/memcpy_stars.cc allocates a
task, initializes it, then submits it. Therefore 'inside rtMemcpyAsync' alone
does not prove DMA execution has started, resource exhaustion, or device fault.
Do not force the candidate function to match any item in this list.

## Decision and stop rules

- Failed mapping/edge: reject that mapping or individual edge, not all v100
  hypotheses by fiat. Record which assumption failed.
- Consistent chain, recognized loop: report binary-supported function/condition
  and source writer; root cause still needs actual state/causal linkage.
- Indirect edges or missing symbols: return partial proof; no guessed owners.
- Nonempty Repository + need_empty=1 is NOT a lost-wakeup verdict. If a callback
  has not returned, notification of queue emptiness may not yet be due.

Return: report, binary identity, full function snippets, seven-edge table,
source alignment, remaining dynamic-state requirements, hashes of new artifacts.
Keep raw server paths/artifacts on-server. No runtime fix or hardware verdict.
