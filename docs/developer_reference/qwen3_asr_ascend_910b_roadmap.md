# Qwen3-ASR Ascend 910B roadmap

This roadmap is the decision layer above the detailed hardware handoff and
performance task. It keeps two outcomes moving in parallel:

1. land maintainable Qwen3-ASR Ascend support in the owning community
   repositories; and
2. reach the exact-10-second C70 performance target without weakening
   correctness, graph coverage, or stability.

The historical `910C-*` records remain evidence. They are not the development
backlog. New work is tracked below by `UP-*` (upstream) and `OPT-*`
(optimization) work packages.

## North-star acceptance

The project is complete only when both tracks pass.

| Track | Required result |
|---|---|
| Feature support | Encoder graph, prefill graph, decode graph, Torch Compile, and the NPU execution guard are enabled together; exact10 output remains correct; target paths show positive capture/replay or compile evidence; forbidden eager fallback, graph fallback, capture failure, device error, hang, and residual NPU holder counts are zero. |
| Performance | On one Ascend 910B/910C-class card with the pinned 700-sample exact10 measured set at concurrency 70: latency p95 `< 0.500 s`, throughput `>= 140 requests/s`, WER within the qualified approximately `0.016` band, zero failed/missing/duplicate/unexpected requests, and repeatable clean teardown. |

Functionality, stability, and performance are separate gates. A fast but
incorrect result is invalid; a correct diagnostic run is not a performance
qualification; a server report does not prove that a patch is reviewable or
present in the local repository.

## Current frozen checkpoint

Snapshot date: 2026-09-10. These values are retained until a newer valid run
supersedes them.

| Item | Current state | Decision |
|---|---|---|
| All acceleration features | Correct and stable with the broad `forward` execution guard | Qualified functional baseline |
| Best qualified profile | `max_total_tokens=32768`, `mem_fraction_static=0.80`, `torch_compile_bs=[1,2,70]` | Freeze as M1c control |
| Exact10 C70 | p95 `1.442 s`, throughput `59.88/s`, RTFx `598.8`, WER `0.0164` | Best reproducible performance |
| Hard-target gap | p95 is `2.88x` target; throughput is `2.34x` below target | Performance incomplete |
| Full 13-bucket compile | p95 `1.504 s`, throughput `57.59/s`; later fresh-process warm-up hung | Not promotable; compiling every bucket is not the answer by itself |
| Guard narrowing | `graph` and `model` candidates produced liveness failures | Rejected until NPU update/replay ownership is fixed |
| `910C-058R` root cause | Generation holder waited for an update helper stuck inside `NPUGraph.update()` after replay returned | Holder-side device/update stall, not FIFO corruption |
| Ordered NPU update candidate | SGLang `e4d18390a`; `910C-059` authorized | Pending hardware qualification |

The only performance control is M1c with the broad `forward` guard. Do not use
capture-contaminated Encoder Graph measurements, the non-reproducible F13 run,
or rejected guard scopes as baselines.

## Branch and evidence policy

### Development sources

| Repository | Branch or commit | Role |
|---|---|---|
| SGLang-Omni aggregate history | `qwen3-asr-910b-opt` | Complete experiment history; not directly PR-ready |
| SGLang-Omni current optimization | `qwen3-asr-guard-scope` | Current roadmap and ordered-update/guard qualification handoff |
| SGLang-Omni consolidated code snapshot | `qwen3-asr-npu-graph-fixes` / `0c6f3be4` | Starting inventory for clean PR reconstruction, not an automatic merge candidate |
| SGLang current optimization | `qwen3-asr-perf-next` / `e4d18390a` | Current NPU graph/compile candidate stack |

Before opening a community PR, reconstruct the selected change on the current
upstream base. Do not open a PR directly from either aggregate experimental
branch. Each PR must omit private evidence paths, `910C` authorization prose,
temporary import/wheel probes, rejected diagnostics, and unrelated upstream
cherry-picks.

### Change disposition

| Category | Examples | Disposition |
|---|---|---|
| Product code with positive hardware evidence | FIFO device-execution guard, host-side Encoder Graph metadata, bounded encoder signatures, graph/compile counters, sparse compile buckets | Reconstruct as small upstream PRs |
| Benchmark infrastructure | exact10 manifest, deterministic SeedTTS packer, NPU resource monitor, strict per-request accounting | Keep, but submit separately from runtime changes |
| Temporary observability useful in production | bounded counters, fallback reasons, model-info execution mode | Retain only if low-overhead and generally useful |
| Diagnostic-only code | capture defer/release probes, state snapshots, spawn/wheel provenance tools, two-graph synthetic reproducer | Preserve on the development/evidence branch; do not put in runtime PRs |
| Rejected semantic candidates | explicit-state attention, graph/model guard implementations as defaults, private graph-pool experiments | Drop from upstream patches unless a later qualified design explicitly needs a minimal part |
| Historical documents | detailed handoff and performance task | Keep on the development branch; upstream only concise user-facing support/configuration docs |

## Track A: upstream code roadmap

Every work package starts from the current upstream base, has its own tests,
and can be reviewed without reading the 910C history.

### SGLang ownership

| ID | Deliverable | Dependencies | Exit gate |
|---|---|---|---|
| `UP-S1` | Minimal NPU stateful decode-attention graph-break correctness fix derived from the qualified `1d5aa4b8e` behavior | None | 140-sample TC-only and graph+compile correctness: WER in band, zero garbled output; focused attention/compile tests pass |
| `UP-S2` | Explicit sparse Torch Compile bucket selection | `UP-S1` | CLI/config normalization tests; exact resolved bucket attestation; `[1,2,70]` works without legacy prefix expansion |
| `UP-S3` | Backend-neutral external execution-context hook plus NPU integrator usage | None; may land before `UP-S1` | No CUDA behavior change; context nesting and exception restoration tests; used by Omni guard without importing Omni code |
| `UP-S4` | NPU-only graph input update/replay ownership repair | `910C-059` | Ordered mode passes correctness and C70 liveness; no helper-thread/join markers; CUDA graph tests unchanged |
| `UP-S5` | Promote the chosen update mode and remove experiment-only switch or retain a documented compatibility fallback | `UP-S4` | Two fresh-process C70 runs and no regression on other NPU graph users such as draft/EAGLE runners |

`UP-S4` must stay inside
`hardware_backend/npu/graph_runner`; it must not alter CUDA graph scheduling.
If ordered high-level `NPUGraph.update()` blocks before replay, stop adding
ordering variants. Replace the wrapper with explicit NPU graph-task update
stream/event ownership, following the device-backend pattern already used by
vLLM Ascend, and validate it as a new implementation of the same work package.

### SGLang-Omni ownership

| ID | Deliverable | Dependencies | Exit gate |
|---|---|---|---|
| `UP-O1` | Qwen3-ASR NPU execution guard and minimal encoder/generation integration | `UP-S3` | Cold concurrency ladder through 70, balanced tickets, no deadlock, no behavior change on non-NPU platforms |
| `UP-O2` | NPU Encoder Graph support: host-side metadata, lazy real-signature capture, bounded multi-signature cache | `UP-O1` | Signature capacity stable before measurement, capture/replay positive, zero capture failure and eager fallback, output parity |
| `UP-O3` | Runtime observability for encoder/prefill/decode graph and compile coverage | Can accompany owning runtime PR or land separately | Counters have defined completion semantics, bounded overhead, and model-info regression tests |
| `UP-O4` | Sparse compile bucket pass-through | `UP-S2` | Schema and propagation tests; resolved server configuration equals requested sparse list |
| `UP-O5` | Exact10 manifest, corpus preparation, strict benchmark accounting, and NPU monitor | Independent of runtime PRs | CPU tests, deterministic corpus hash, explicit invalidation on request or monitor failure, documented private-data boundary |
| `UP-O6` | Concise Ascend Qwen3-ASR operator documentation | Accepted runtime PRs | Lists supported features, exact flags, limitations, validation commands, and cleanup; contains no internal 910C chronology |

### Recommended PR order

```text
SGLang:      UP-S1 -> UP-S2
                 \-> UP-S3 -> UP-S4 -> UP-S5

SGLang-Omni:          UP-O1 -> UP-O2 -> UP-O3
              UP-S2 -> UP-O4
                         UP-O5 (independent)
              accepted runtime stack -> UP-O6
```

Keep no more than two community PRs under active review per repository. While
one runtime PR is in review, prepare the next patch and continue performance
work against the aggregate development branch. Upstream review must not block
hardware optimization, and experimental commits must not leak into PR diffs.

## Track B: performance roadmap

### Fixed experiment contract

Unless a work package explicitly studies one of these values, every candidate
uses:

- one device and one fresh service process;
- the pinned 70 warm-up plus 700 measured exact10 corpus;
- concurrency 70;
- all acceleration features enabled;
- `max_total_tokens=32768`;
- `mem_fraction_static=0.80`;
- `torch_compile_bs=[1,2,70]`;
- the same model, precision, client, request timeout, WER normalizer, and NPU
  monitor as M1c.

Every performance result must first pass correctness, target-path, fallback,
health, and cleanup gates. Compare against M1c and the immediately preceding
accepted candidate. A change is retained only if it improves p95 or throughput
materially without regressing the other metric beyond run noise. Confirmed
winners require fresh-process reproduction; failed candidates are closed, not
retuned repeatedly in the same process.

### Optimization packages

| Priority | ID | Question and implementation direction | Required evidence | Promotion rule |
|---|---|---|---|---|
| P0 | `OPT-1` | Can NPU input update/replay use one owned transaction without the helper-thread stall? Implemented by SGLang `e4d18390a`. | `910C-059` ordered markers, correctness, C70 liveness, guard counters | Arm O must pass before any narrowing result is considered |
| P0 | `OPT-2` | After `OPT-1`, can the guard shrink from whole generation forward to the exact graph transaction? | Conditional graph-scope arm in `910C-059`; encoder and generation wait/hold distributions | Accept only if C70 drains and improves p95/throughput; otherwise retain forward scope |
| P1 | `OPT-3` | Reduce attention/compiler discontinuity without restoring incorrect custom-op behavior. Keep stateful decode attention correct while reducing graph breaks and recompiles. | Per-phase forward histogram, compile/eager coverage, unique graph-break reasons, recompile keys, NPU timeline | Must reduce steady-state host gaps and improve end-to-end metrics; graph-break count alone is not success |
| P1 | `OPT-4` | Reduce encoder queueing and long guard holds. Split host preparation from the minimal device transaction, improve batch formation, and stabilize signatures before measurement. | Encoder queue wait p50/p95/max, batch-size histogram, guard hold p50/p95/max, capture count during measured wave | Zero measured-wave capture/fallback and lower encoder critical-path time |
| P2 | `OPT-5` | Rebalance prefill/decode admission and batch formation after serialization is reduced. | Request state timeline, running-batch histogram, time in build/admit/prefill/decode/output stages | Lower tail latency without reducing throughput or starving any phase |
| P2 | `OPT-6` | Optimize remaining NPU kernels only after stage attribution identifies a device-bound operator. | Operator-level NPU trace, kernel duration and utilization, numerical parity | No speculative `sgl-kernel-npu` rebuild; require a named hot operator and measurable gain |

### Why compile coverage is no longer the first lever

M1c proved that compiling bucket 70 matters, but F13 did not beat sparse
`[1,2,70]` and was not reproducible. Therefore the next compile work is not
"compile more buckets." It is to reduce steady-state graph breaks,
recompilations, and host/device discontinuities on the buckets that dominate
the real workload. Bucket additions are allowed only when a fresh histogram
shows meaningful uncovered traffic and the memory/performance budget predicts
a gain.

### Performance milestones

These are decision checkpoints, not claims that one patch will reach them.

| Milestone | Target | Purpose |
|---|---|---|
| `PERF-A` | Reproducible p95 `<= 1.20 s`, throughput `>= 70/s` | Prove update/guard and host-gap work beats M1c materially |
| `PERF-B` | Reproducible p95 `<= 0.80 s`, throughput `>= 100/s` | Prove encoder and scheduler critical paths are no longer dominant |
| `PERF-C` | p95 `< 0.500 s`, throughput `>= 140/s` with all final gates | Customer acceptance |

Intermediate milestones do not relax the final target. If `PERF-A` cannot be
reached after `OPT-1` through `OPT-4`, perform an architecture review before
more parameter search: quantify the theoretical encoder, prefill, and decode
lower bounds and decide whether the target requires a different batching,
streaming, or model-serving architecture.

## Immediate execution plan

### Now: close the graph-update chapter

1. Run `910C-059` exactly once against SGLang `e4d18390a` and the handoff
   commit containing this roadmap.
2. Arm O validates ordered update under the qualified broad guard.
3. Only if Arm O passes, the same task runs a fresh graph-scope arm.
4. Record the result here as accepted, rejected, or requiring the explicit
   graph-task update-stream/event implementation. Do not add a third guard
   scope or another update-order flag.

### In parallel locally: prepare upstream slices

1. Create clean worktrees from current upstream for `UP-S1` and `UP-O1`.
2. Reconstruct only the minimal qualified code and focused tests; do not
   cherry-pick the aggregate history wholesale.
3. Produce a commit dependency map from experimental hashes to clean commits.
4. Review non-NPU behavior, CUDA boundaries, public configuration names, and
   removal of diagnostic environment variables before opening PRs.

### Immediately after `910C-059`

- If ordered update and graph scope pass: promote the backend design into
  `UP-S4`, freeze that guard boundary, and begin `OPT-3` plus `OPT-4`.
- If ordered update passes but graph scope fails: keep broad forward scope,
  upstream the correctness/backend fix only after cross-runner validation, and
  prioritize `OPT-4` before another narrowing design.
- If ordered update blocks before replay: replace the high-level update wrapper
  with explicit NPU task update stream/event ownership; do not return to FIFO
  tuning or Python thread experiments.

## Operating cadence

Each development cycle produces exactly four artifacts:

1. one bounded code diff in the owning repository;
2. focused offline tests and a review note;
3. one committed server task with fixed stop/pass rules;
4. one result update that either promotes or closes the candidate.

The current-state table in this roadmap is updated only for accepted facts.
Detailed commands and evidence remain in the companion documents:

- [Ascend 910B hardware handoff](qwen3_asr_ascend_910b_handoff.md)
- [Ascend 910B validation task](qwen3_asr_ascend_910b_validation_task.md)
- [Ascend 910B performance task](qwen3_asr_ascend_910b_performance_task.md)

## Definition of done

- Clean, reviewable upstream commits exist for every retained runtime feature.
- Community PRs contain no internal diagnostics or historical experiment log.
- Cross-repository dependency order is documented and reproducible.
- ALL-feature correctness and capacity qualification pass on the final rebased
  commits, not only on aggregate development branches.
- Three fresh-process final performance runs satisfy `PERF-C`; no run uses
  warm-up or capture-contaminated measurements.
- Graceful shutdown returns HBM to baseline with no holder or service process.
- The handoff records the final runtime matrix and remaining limitations, and
  this roadmap has no unresolved P0 item.
