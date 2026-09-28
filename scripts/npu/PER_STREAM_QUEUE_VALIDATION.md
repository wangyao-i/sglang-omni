# GPU-style cache transfer with per-stream Host queues

Experimental handoff PSQ-02, revision 1. Not production qualification.

## Frozen comparison

- Control: Omni d776cf5ab38916359c8512fb11066755d71aa9b1, PER_STREAM_QUEUE=0.
- Candidate: the commit containing this document, based directly on the control,
  PER_STREAM_QUEUE=1. Record its full SHA and verify the parent before execution.
- SGLang: 0bcd822377da7b5718e674eaf9c870d349424dd1, unchanged.
- Candidate cache hits follow the existing GPU caller-thread attach path.
  There is no cache-transfer worker entry, explicit cache-transfer synchronization,
  or new event protocol. Normal encoder batch synchronization, allocator stream
  registration, and graph replay/update remain unchanged.
- The hypothesis is that independent Host queues let graph update progress even
  when a pageable copy blocks another queue. This is not proof that copies become
  nonblocking, nor that arbitrary cross-stream consumption is safe.

## Environment and isolation

Use the previous validated environment, model, 140-input corpus, reference,
startup and measurement scripts. Record their identities, actual imported Omni,
SGLang and torch_npu paths/builds, effective configuration and clean tracked state.
Do not resolve imports to a diagnostic checkout.

Reserve physical device 0 exclusively; record visible-to-physical mapping.
Do not start if other tasks own the device. If contention appears during a run,
stop subsequent runs and mark the affected pair contaminated, not passed.
Do not install packages, edit dependencies, reset devices or kill foreign processes.

Set PER_STREAM_QUEUE before starting each fresh service process. Keep the same
effective TASK_QUEUE_ENABLE=1 or 2 in both arms; ASCEND_LAUNCH_BLOCKING must be off.
Verify the installed runtime supports per-stream queues using matching source/build
and existing runtime diagnostics. An environment assignment alone is not evidence
of activation. Unsupported or unconfirmed activation blocks the comparison.
Do not set this globally or enable fine-grained CPU binding for this experiment.

## Checks and workload

1. Run the existing native encoder-service and request-builder CPU tests against
   the candidate. Stop on failure; do not reproduce the old test-count target.
2. Before acceptance, establish that caller-thread cache H2D and actual LM
   embedding consumption use the same device stream, using version-matched call
   paths and available stream evidence. record_stream is allocator tracking, not
   an execution dependency. If the stream relationship cannot be established,
   report UNKNOWN; do not silently add events or change the experiment.
3. Run four pairs of independent cold starts: A/B, B/A, A/B, B/A.
   Preserve the existing full-graph, single-encoder-worker configuration.
   Each process runs conc8/140 without warmup for the comparable measurement.
4. Then repeat the same 140 inputs within that process as a separate cache-hit
   correctness phase. Require a positive cache-hit counter delta and actual
   CPU-cache-to-device transfer coverage; report mixed hit/miss coverage from the
   cold phase separately. Do not combine the repeat phase with cold performance.
5. If all pairs pass, run the candidate for 20 further independent cold starts,
   each with the same cold and cache-hit phases. Total budget: two hours from the
   first operation, including setup and aborted attempts. Report incomplete rounds
   honestly; do not expand the budget or silently rerun failed rounds.

For each 140-request phase require: 140 HTTP successes, no failures/empty outputs/
sample WER above 0.5, no per-sample disagreement with the frozen reference, and
matching corpus WER (previously reported 0.011373). Require prefill attestation,
decode graph execution without fallback, encoder capture/reuse evidence, and no
runtime errors. Missing target-path evidence is UNKNOWN, not PASS.

Retain the existing 30-second no-progress watchdog and 600-second round ceiling.
Stop on the first test, correctness, runtime, liveness or cleanup failure. Preserve
first-failure logs; do not retry over them. Stop only owned processes using the
existing bounded cleanup procedure; any forced stop must be reported as failure.
Each successful round must use normal SIGTERM, release its port/processes, and
return device memory to its recorded idle baseline before the next round.

## Report

Return exact identities, activation/stream evidence, phase-level hit/miss counts,
correctness, graph fallback, watchdog and cleanup results; keep full logs on server.
Report cold throughput, p50/p95/p99, maximum latency and service-tree CPU/thread
usage per arm. Compute per-pair changes and their median. Performance benefit means
median throughput gain >=5% with median p95 worsening <=5%; absence of this benefit
does not itself fail functional feasibility.

Distinguish correctness/liveness PASS, FAIL or INCOMPLETE from performance benefit.
This compares two complete designs, not the isolated effect of the environment
variable. It does not qualify an untested runtime, disabled per-stream queues, or
all possible stream topologies. Return the evidence path and report SHA256.
Rollback means selecting the unchanged control for a fresh process, not modifying
either checkout. Do not update PR #2160 from these results automatically.
