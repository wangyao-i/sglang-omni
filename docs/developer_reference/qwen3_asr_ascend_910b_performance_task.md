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

The current SeedTTS benchmark remains useful for public accuracy and
concurrency comparisons, but it does not enforce exact 10 s clips. Local
commits `37f598f3` and `63f235fa` add the strict exact-manifest client and NPU
monitor described below. This makes the workload measurable after its
deterministic corpus has
been prepared and the server-specific `npu-smi` parser preflight has passed;
it does not turn historical SeedTTS or SSE results into hard-gate evidence.

## Exact-10-second harness contract

The checked-in entry point is
`python -m benchmarks.eval.benchmark_asr_exact10s`. It consumes a JSONL
manifest, not a Hugging Face repository ID or an ad-hoc directory scan. Each
non-empty record has this schema:

```json
{"sample_id":"stable-id","wav_path":"audio/clip.wav","ref_text":"reference","language":"en"}
```

Relative paths resolve below the manifest directory and may not escape it.
Every row is mandatory: invalid JSON, a missing file/reference, a duplicate
sample ID, or a bad WAV fails preflight. WAV validation walks RIFF chunks and
requires uncompressed PCM, one channel, 16 kHz, 16-bit samples, an untruncated
payload, and `160000 +/- 1` effective frames. Audio uniqueness hashes decoded
PCM frame bytes. The aggregate manifest fingerprint is a full SHA-256 over a
versioned namespace plus length-prefixed sample ID, language, reference,
effective frame count, and audio hash; local paths and WAV container metadata
do not affect it.

`--warmup-samples N` reserves the first N records as a deterministic warm-up
partition and removes them from measurement. Warm-up and measured PCM hashes
must be disjoint. `--hard-gate` additionally requires at least 70 warm-up
records, exactly 700 measured records, at least 770 distinct inputs in the
complete manifest, and `--repeats 1`. The last constraint is intentional: the
operator must restart and re-attest the service between the three final
repeats; a client loop against one process is rejected.

For every expected request, the raw JSONL contains schema version, record kind,
sample ID, language, PCM SHA-256, duration, success, latency, RTF, timeout,
error, reference, and hypothesis. Duplicate and unexpected returned IDs are
also retained as separate record kinds. Missing results receive an explicit
record and a conservative wall-clock latency. Any timeout, transport/HTTP
error, empty response, missing/duplicate/unexpected result, unscoreable output,
or NPU-monitor failure marks the repeat invalid. Latency mean, p50, p90, p95,
p99, and max include every expected request outcome, not only successes.

The hard latency timestamp is taken when the complete multipart body has been
accepted by the aiohttp stream writer, followed by receipt of the final
response body. It does not use aiohttp's pre-write chunk trace callback. Raw
JSONL, transcripts, paths, server logs, and NPU samples remain server-local;
only sanitized aggregates and fingerprints are returned.

`NpuResourceMonitor` samples the explicitly selected `--npu-id` and
`--npu-chip-id` with `npu-smi info -t usages`, `memory`, and `power`. It records
HBM capacity/use, AI Core or NPU utilization, temperature, power, and host CPU.
Command failures, parser failures, missing HBM use, or missing both AI Core and
NPU utilization invalidate the repeat. Because vendor output varies by stack,
`910C-024A` proved these commands and labels on the frozen A3 environment with
valid batch-one and concurrency-two smokes. The local tests remain fixture
coverage rather than hardware proof; the accepted hardware identity and
sanitized evidence are recorded in the handoff.

The artifact layout for one service lifetime is:

```text
<evidence>/exact10/<service-run-id>/
  result.json
  raw/conc<N>-repeat1.jsonl
  server.log
  preflight.txt
```

The summary JSON records the full manifest and measured fingerprints, counts,
concurrencies, per-repeat validity/reasons, all-outcome latency/RTF/throughput,
WER, NPU summary and raw samples, NPU environment fingerprint, dependency
inventory, declared
launch command, and repository state. Do not return the raw directory outside
the isolated environment.

Example client invocation after the corpus and service have been separately
attested:

```bash
python -m benchmarks.eval.benchmark_asr_exact10s \
  --meta "${EXACT10_MANIFEST}" \
  --host 127.0.0.1 --port "${QWEN3_ASR_PORT}" \
  --model Qwen/Qwen3-ASR-1.7B --lang en \
  --concurrencies 70 --repeats 1 \
  --warmup-samples 70 --max-samples 700 \
  --min-distinct-audio 770 --hard-gate \
  --npu-id 0 --npu-chip-id 0 --monitor-interval-s 1 \
  --request-timeout-s 120 \
  --launch-command "${DECLARED_SERVER_LAUNCH}" \
  --output "${EVIDENCE}/result.json" \
  --save-raw-dir "${EVIDENCE}/raw"
```

Local commits `2cb63b9e`, `8d46ddec`, and `30b21522` provide that deterministic transform at
`python -m benchmarks.manifest.prepare_seedtts_exact10s`. It accepts only the
pinned SeedTTS English snapshot revision already named in this handoff, sorts
stable source IDs, accepts only uncompressed mono PCM16 source WAVs at 16 kHz
or 24 kHz, and converts the pinned 24 kHz sources to 16 kHz with one fixed
ffmpeg/swresample contract before duration filtering or composition. It
excludes and records sources longer than 10 seconds after conversion and
requires at least 770 usable source rows. Each derived clip starts from a
stable anchor, excludes other rows with the same resampled PCM hash from that
clip, and evaluates deterministic whole-utterance plans containing the anchor
plus one or two distinct-audio partners. Plans are ranked by retained speech
frames with stable tie-breaking. If a repeated source row would reproduce an
already emitted whole-clip PCM hash, the generator tries its next qualifying
plan. Anchors that cannot reach 80% speech or exhaust distinct alternatives
are recorded and skipped; corpus creation succeeds only after 770 distinct
outputs have been built. A fixed 100 ms silence separates utterances and only
the remainder is padded with PCM silence. Speech is never cropped. The joined
references therefore describe all retained speech. Every output must contain
at least 80% speech frames, exactly 160000 frames, and a distinct whole-clip
PCM hash. It writes
`manifest.jsonl` plus server-local `provenance.json` containing source
membership, exclusions, the pinned source identity, source-Parquet-set hash,
source-rate counts, distinct source-audio count, resampled-source count,
skipped-anchor categories, ffmpeg version-output SHA-256, fixed command
template, and derived manifest hash.

The only authorized 24 kHz conversion is the generator-owned command below.
It uses ffmpeg's built-in swresample backend, disables dithering, and emits raw
mono PCM16 for composition:

```text
ffmpeg -nostdin -hide_banner -loglevel error -i <input-wav> \
  -map_metadata -1 -vn -sn -dn -ac 1 \
  -af aresample=16000:filter_size=32:phase_shift=10:linear_interp=0:exact_rational=1:dither_method=none \
  -c:a pcm_s16le -f s16le pipe:1
```

There is no torchaudio/scipy fallback and no operator-selected resampling
backend. The generator resolves `ffmpeg` from `PATH`, requires `ffmpeg
-version` to succeed, and fails closed on timeout, nonzero exit, empty output,
or invalid PCM16 byte length. A missing or incompatible ffmpeg installation
requires a new local decision; it does not authorize an isolated-server
package install or manual preprocessing.

The generator refuses an existing output directory rather than merging or
overwriting prior evidence. The standard corpus command is:

```bash
python -m benchmarks.manifest.prepare_seedtts_exact10s \
  --snapshot-root "${SEEDTTS_SNAPSHOT}" \
  --revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --output-root "${EVIDENCE}/exact10-corpus" \
  --total-clips 770 --warmup-clips 70 --silence-ms 100
```

Do not replace this command with operator selection, external clipping,
external resampling, or padding. If the pinned snapshot has fewer than 770
valid sources, an unsupported source
format mismatch, insufficient speech occupancy, or duplicate derived audio,
stop and return the first failure. Such a result requires a new locally
reviewed transform; it does not authorize a server edit.

## Progressive offline ladder

The compatibility-profile baseline uses a bounded subset of this ladder.
`910C-024A` qualified the harness with manifest preflight, batch one, and two
concurrent requests. `910C-024B` then collects 100 sequential requests and one
700-request concurrency-70 repeat in two independent fresh services so one arm
cannot warm the other. This establishes the before-state and dominant stage
only. Skip the ten-minute soak and three fresh-process repeats for that known
disabled-feature baseline.

The accepted `910C-024B` measurement evaluated 100/100 sequential requests at
0.289-second p95, then 700/700 requests at concurrency 70 at 3.516-second p95
and 39.31 requests/s. The latter misses the latency target by 7.03 times and
reaches 28.1% of the implied 140-request/s rate. Its post-service chip-0 HBM
remained at 87% instead of the approximately 4% idle baseline, so the
performance aggregates are retained as the compatibility before-state while
cleanup remains a blocking qualification exception. The handoff owns the
read-only attribution task and the decision to resume acceleration work.

The read-only `910C-024C` attribution identified 53,966 MB retained by stale
NPU context PID 2043369 after the concurrency-70 service was terminated with
`SIGKILL`; no normally manageable user process remained. Existing logs showed
251 encoder batches for 754 items, queue-wait average/maximum of 1.24/13.26
seconds, 22.6 seconds cumulative encoder time, and 100% decode graph replay
across all 13 buckets. No paired rich model-info snapshots exist. Prioritize
encoder/guard scheduling analysis after recovery, but do not claim a final
stage attribution from these coarse counters. Hardware work remains blocked
until operator-approved runtime recovery and a separately authorized clean
post-recovery preflight.

The project owner subsequently reported terminating PID 2043369 without a
driver restart or host reboot. `910C-024D` then verified three healthy snapshots
at stable 4% HBM with no retained holder or worker, closing the cleanup
exception. To shorten turnaround without repeating known failures,
`910C-025A` groups only independent fresh-process all-eager E0 and guarded
prefill+decode graph P arms. A clean arm failure does not block the other arm;
any device, OOM, retained-context, forced-cleanup, or HBM-recovery failure stops
the campaign. Each qualified arm receives batch one plus one 700-request C70
exact10 measurement. Encoder graph, torch compile, and the fully accelerated
combination remain unauthorized until local repair commits and regression
tests exist. This screening campaign intentionally omits sequential, soak, and
three-repeat acceptance work.

The full ladder below applies to the fully accelerated candidate and to later
candidates that are being considered for the hard target.

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

For the hard gate, each measured repeat requires at least 700 distinct audio-
content hashes and a separate warm-up partition. Validate PCM16 mono 16 kHz
format and effective WAV frame count, not only a RIFF data-chunk byte count.
All request outcomes must be present in raw records. Any timeout, transport or
HTTP failure, empty response, missing result, or duplicate result ID fails the
repeat; the summary must not publish a successful-only percentile as if it
described the requested workload.

The three final repeats are process-level repeats: stop, cleanly drain, verify
the device baseline, and start a fresh service before each one. A client-side
loop against one service process does not satisfy this contract. NPU monitoring
is required for the hard gate. A command/parser failure or absence of the
selected device's HBM and utilization samples invalidates the run and must be
reported explicitly.

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
