# Qwen3-ASR Ascend 910B: performance and realtime task

This task begins only after the eager functional and restart gates in the
[first validation task](qwen3_asr_ascend_910b_validation_task.md) pass. The
[hardware handoff](qwen3_asr_ascend_910b_handoff.md) owns the thresholds and
status; this page owns the measurement procedure.

## Prerequisites

- Use a clean server process per variant and record the exact repository and
  dependency revisions.
- Freeze a server-local 10 s evaluation manifest meeting the handoff contract.
  Record its aggregate hash, number of clips, language counts, duration range,
  and reference-token counts without returning paths, audio, or transcripts.
- Ensure every timed request has unique audio bytes or disables the repeated
  audio embedding cache. Cache hits are a separate experiment and cannot prove
  the primary target.
- Freeze the eager-NPU correctness result before measuring an optimized
  variant. Change only one setting or patch between A and B.
- Install benchmark dependencies through the repository-supported `eval`
  extra. Do not install ad-hoc packages into `site-packages` to repair a run.
- On an isolated host without Hugging Face access, pre-stage the standard
  `HF_HOME` cache in a connected environment using the repository's documented
  dataset command and pinned revision. Verify that cache there with
  `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`, archive it while preserving
  layout and links, record the archive SHA-256 and dependency versions, and
  transfer it through the approved channel. The isolated run must verify the
  hash and repeat the prepare command fully offline before starting a server.
  A smaller local Parquet snapshot is acceptable only when the handoff fixes
  its upstream revision, file layout, byte size, and content SHA-256 and the
  benchmark's local-directory loader validates the expected split and sample
  count fully offline. Do not replace the pinned corpus with repeated test
  clips or an unmanifested local dataset.
- Keep benchmark-only data-reader repairs out of the serving environment. When
  the handoff identifies a compatible `pyarrow` patch wheel, install it offline
  in a separate benchmark-client virtual environment, scan the complete pinned
  Parquet and validate the requested samples there, and run only the HTTP client
  from that environment. Record both client and server dependency fingerprints;
  launch the server with its original interpreter and verify it is unchanged.
- For a local snapshot, pass the snapshot root through `--meta` and use
  `--unique-audio` before `--max-samples`. The local loader resolves only
  `data/<lang>-*.parquet`, remains offline, and fails if the requested number
  of distinct audio byte sequences is unavailable.

The current SeedTTS benchmark is useful for public accuracy and concurrency
comparisons, but it does not enforce exact 10 s clips and its resource monitor
is GPU/NVML-oriented. Until an NPU-aware exact-manifest harness is implemented
and tested in this repository, the hard 10 s/70-session result is **not
measurable by the checked-in benchmark**. Do not relabel SeedTTS or SSE results
as the hard gate.

## Progressive offline ladder

For each server variant, run these levels in order with no request retries:

1. batch 1 and two concurrent requests;
2. 100 sequential requests;
3. concurrency 8, 16, 32, 64, and 70;
4. ten minutes at concurrency 70;
5. three fresh-process measured repeats at concurrency 70.

At every level collect request count, success/failure categories, latency
distribution, request throughput, input-audio-seconds/s, RTFx, corpus WER/CER,
normalized-output hashes for the smoke subset, peak/steady NPU memory, device
utilization, host CPU and memory, and encoder queue/batch/cache statistics.
Stop at the first correctness, stability, or memory failure and preserve the
complete server-local evidence.

The exact-manifest harness must use the latency timestamps and minimum sample
size declared in the handoff, emit per-request machine-readable records, and
aggregate failed requests rather than dropping them from percentiles. Add its
unit tests before accepting its numbers.

## Performance-candidate contract

The compile-disabled, encoder-eager, prefill-eager, decode-graph profile is the
qualified correctness and measurement baseline. It is not the intended final
performance profile. Compile, encoder graph, and prefill graph are mandatory
repair items; explicit disablement cannot close them. After the harness is
validated, restore one acceleration at a time and compare it against that
frozen baseline before combining changes:

1. prefill graph with the Qwen3-ASR NPU execution guard enabled;
2. an NPU-compatible encoder graph or a faster replacement that preserves the
   same encoder outputs;
3. torch compile after repairing the SGLang/triton-ascend capture boundary;
4. a reduced execution-guard critical section or stream/event protocol if the
   current whole-forward serialization is a measured bottleneck;
5. the combined candidate with compile, encoder graph, prefill graph, and
   decode graph enabled and all execution markers attested.

Every currently failing path must pass its functional and stability gate even
if a later A/B motivates a different optimized implementation. Do not call the
disabled-feature baseline the final candidate merely because it is stable.
The primary hard-target attempt uses the fully enabled combined profile, with
positive execution evidence and zero unexpected fallback.

## Public regression run

Prepare the pinned SeedTTS dataset and run the existing benchmark separately.
On NPU, disable its NVML resource monitor until NPU sampling support exists:

```bash
python -m benchmarks.dataset.prepare --dataset seedtts
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port "${QWEN3_ASR_PORT}" \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --concurrencies 1,8,16,32,64,70 \
  --repeats 3 \
  --warmup \
  --fingerprint \
  --disable-resource-monitor \
  --output "${QWEN3_ASR_EVIDENCE}/seedtts.json"
```

Record the pinned dataset revision used by the command. Apply the repository's
existing language-specific accuracy thresholds; do not introduce different
numbers in this task page.

## Realtime implementation gate

Before performance measurement, a protocol test must prove that the server
emits transcript progress while audio is still arriving. The existing
`stream=true` transcription endpoint and the current VAD auto-commit flow are
negative controls: both operate on a complete utterance.

The realtime harness must:

- open 70 WebSocket sessions and wait for each `session.created` event;
- append exact 500 ms PCM16 chunks according to wall-clock pacing;
- timestamp every completed append, partial transcript/revision, commit, final
  transcript, error, and disconnect event with a monotonic clock;
- prove at least one transcript event arrives before final commit for every
  non-empty utterance;
- validate session isolation by using distinct server-local utterances and
  comparing only normalized-output hashes in returned evidence;
- report partial and final latency using the definitions in the handoff;
- run a 10-minute bounded soak and verify buffers and device memory do not grow
  without bound.

If the protocol cannot represent transcript revision unambiguously, implement
and test that contract before optimizing inference. Do not infer revisions by
diffing arbitrary text fragments in the benchmark client.

## A/B decision rule

Each candidate is compared with the frozen eager-NPU baseline on the same
server, manifest, process lifecycle, and harness revision. Accept it only when:

- all correctness and stability gates pass;
- every one of the three repeats meets the relevant latency target;
- throughput improvement is repeatable and not caused by failed requests,
  cache hits, shorter audio, or a changed output budget; and
- logs show the intended backend or graph path, with no unexpected fallback.

Return only aggregate results, exact revisions and sanitized failure classes.
Keep raw request records, logs, audio, transcripts, and profiler captures on
the isolated server.
