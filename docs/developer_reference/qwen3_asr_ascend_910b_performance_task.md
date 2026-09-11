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

`910C-025A` completed both current-code arms. E0 (all eager, guard inactive)
measured 2.641-second p95 at 31.52 requests/s. P (guarded prefill and decode
graphs) measured 1.771-second p95 at 48.39 requests/s, improving p95 by 49.6%
and throughput by 23.1% over the `910C-024B` guarded decode-only baseline.
Prefill graph is therefore retained in the accelerated candidate. The result
still misses the hard target by 3.54 times on p95 and reaches only 34.6% of the
implied 140-request/s throughput. The next performance-bearing work is local
encoder-graph repair, torch-compile repair, and reduction of guard/encoder
serialization before a combined EG/TC/ALL hardware campaign.

Local encoder-graph repair `fa5b8852` is the next performance-bearing change.
It removes Ascend capture-time D2H sequence-boundary synchronization, uses one
bounded real host-side window signature per token bucket, and exposes encoder
capture/replay/fallback counters in rich model-info. `910C-026` compares this
single change against the accepted P profile: repeated batch one first proves
capture and replay correctness, then a separate 70-sample warmup and one
700-request C70 measurement determine graph coverage and directional benefit.
Any measured encoder eager fallback fails full feature qualification even when
the diagnostic performance measurement completes.

`910C-026` has now supplied that diagnostic result. Repeated batch one captured
and replayed successfully with zero fallback, but the heterogeneous 700-request
C70 corpus produced 64 counted `npu_signature_mismatch` eager fallbacks. The
arm completed 700/700 at p95 1.652 seconds and 52.55 requests/s, the best
directional result so far, but Encoder Graph remains unqualified. In addition,
the full Qwen3-ASR suite reported 11 failures before the server continued, so
the C70 result cannot override the failed stop condition. No final performance
campaign is authorized from this result. A local bounded multi-signature or
canonical-layout repair and a clean full-suite gate must precede the next
hardware task.

Local follow-up `9080b901` changes the NPU encoder graph to retain up to eight
exact real signatures globally, allowing several layouts in the same token
bucket while bounding graph memory by the configured encoder batch size. The
`910C-027` feature gate warms concurrency shapes 1, 2, 4, and 8 plus one C70
wave before measuring. Qualification requires no signature/capture growth in
the measured arm and zero eager fallback. Its C70 metric remains diagnostic
until Encoder Graph feature qualification closes; torch compile remains a
separate later repair.

`910C-027` reached zero eager fallback and zero capture failure, but its graph
count grew from five after warm-up to the eight-signature cap during the
measured arm. The resulting p95 2.685 seconds and 50.60 requests/s are therefore
capture-contaminated diagnostics. `910C-028` keeps the code and capacity fixed,
warms every encoder batch size 1 through 8, then repeats excluded C70 waves
until two drained snapshots are stable at 8/8 before measuring. Do not raise
the memory bound or accept measured capture growth unless new evidence shows
that this deterministic saturation cannot cover the frozen workload.

The standalone `910C-028` run is superseded before execution by combined task
`910C-029`. That task transfers the complete local SGLang Torch Compile repair
and the SGLang-Omni encoder/model-info changes in one delivery. It first
qualifies Torch Compile in isolation, then saturates Encoder Graph and runs an
`ALL` arm with encoder, prefill, decode graph, and Torch Compile enabled. A
single 700-request C70 ALL measurement is the fast go/no-go point. Only when it
already meets the latency, throughput, correctness, and feature-evidence gates
does the same authorization continue into the sequential/ladder/soak and two
additional fresh-process repeats. This combines independent checks without
allowing the server operator to edit code or dependencies.

`910C-029` stopped being valid when its TC arm exposed missing decode-only
`attention_layers` metadata and the server bypassed the compile context with an
uncommitted edit. The resulting ATB failure tested the bypass, not the intended
custom-op path. A later compile-disabled arm was neither authorized after the
first failure nor an `ALL` profile. Corrected task `910C-030` uses local SGLang
commit `93d312480`, which initializes the required layer metadata even when
prefill graph is disabled. It repeats TC first and proceeds to ALL only after TC
passes. Encoder saturation is strictly drained between waves and may never
exceed 70 outstanding requests.

The full ladder below applies to the fully accelerated candidate and to later
candidates that are being considered for the hard target.

The project has two conjunctive goals. First, every currently identified
acceleration path must be supported and the combined `ALL` profile must pass its
functional/stability qualification. Second, that same fully enabled profile
must meet the exact-10-second performance target. Until the first goal closes,
any latency or throughput result is a diagnostic before/after measurement only;
it must not be reported as the final hard-gate result. A profile that reaches a
performance threshold by disabling encoder graph, prefill graph, decode graph,
or torch compile does not satisfy project completion.

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

The mandatory order is therefore:

1. qualify encoder graph (the first `910C-026` attempt failed on 64 real-corpus
   signature-mismatch fallbacks; bounded multi-signature follow-up `9080b901`
   is authorized as `910C-027`);
2. repair and qualify torch compile, including the Dynamo/triton-ascend boundary
   and every compiled/non-compiled decode bucket;
3. requalify prefill and decode graph together with encoder graph and compile in
   one `ALL` profile under cold-input concurrency 70;
4. profile and, if necessary, narrow the execution guard without weakening the
   `ALL` feature evidence;
5. run the final sequential, concurrency ladder, ten-minute soak, and three
   fresh-process C70 hard-target repeats on the accepted `ALL` profile.

Current gate `910C-031` implements step 2 without changing or rebuilding the
installed `sgl-kernel-npu`: SGLang commit `44f9e40b5` wraps the external fused
QKV/RMSNorm/RoPE Triton launcher in an opaque custom op so the compiled batch-
one bucket cannot trace its eager device-property query. The gate must retain
batch 1 and the complete 13-bucket decode list. If TC passes, the same server
task proceeds directly through a drained fully enabled `ALL` ladder and one
exact10 C70 diagnostic; the final soak and three fresh-process repeats remain
reserved for a qualified `ALL` profile.

An item-level C70 run may accompany steps 1 through 4 to expose its performance
direction and bottleneck. It is intentionally not a substitute for step 5.

## `910C-056`: memory-budgeted sparse compile buckets

`910C-055` established an allocation limit, not a hardware batch-size limit.
With the existing KV-cache reservation and prefix compile policy, requesting
compile max batch 8 compiled `[1, 2, 4, 8]` cumulatively and exhausted the
remaining HBM during startup. Do not describe `[1, 2]` as the maximum supported
compile shape.

Use Omni `fcc2fb64` with SGLang `edc504ff6` (based on the correctness-qualified
`1d5aa4b8e`). The new `asr.engine.torch_compile_bs` list selects a sparse subset
of the already captured decode buckets. Omitting it preserves the legacy
`torch_compile_max_bs` prefix behavior.

Run this task serially on one clean NPU. The server must not edit source or
packages. First run the new SGLang bucket-selection tests, the existing NPU
decode/radix-attention tests, the Omni runtime-schema test, and the existing
Qwen3-ASR focused correctness suites. Stop on the first failure.

Before starting a service, read the retained `910C-054` and `910C-055` artifacts
and report:

- the exact padded decode-bucket histogram for the A1 C70 workload;
- current `mem_fraction_static`, KV-pool bytes/token capacity, and maximum
  observed live KV tokens;
- available HBM before capture and, where logged, after each compiled bucket in
  the failed prefix `[1, 2, 4, 8]` capture.

Choose one reduced `mem_fraction_static` for all following arms. It must still
provide at least 1.25 times the maximum observed live KV tokens; if the retained
artifacts cannot prove that capacity, stop rather than guessing. Rank buckets
above 2 by observed `bucket_hits * bucket_size`. Let `H1` and `H2` be the first
and second ranked buckets.

Run fresh-process arm M1 with compile buckets `[1, 2, H1]`. Require model-info
to report exactly those compile buckets, every requested graph path to replay,
zero eager fallback/capture failure, normal WER, 140/140 correctness, and clean
drain. If startup OOMs, stop and return the memory budget. If M1 passes, run the
700-request exact10 C70 measurement and retain latency, throughput, bucket,
guard, NPU utilization, and HBM evidence.

Run fresh-process arm M2 with `[1, 2, H1, H2]` only when M1 leaves at least the
larger of 4 GiB or 1.25 times the observed M1 capture increment free after
capture. Apply the same correctness and C70 gates. Never proceed to a third
bucket, soak, or realtime in this task. Gracefully stop each service and require
HBM to return to the pre-run baseline before the next arm.

Return one sanitized table containing the memory-budget calculation, histogram,
selected buckets, startup/capture result, compile/replay/fallback counters,
WER, p95, throughput, utilization, post-capture free HBM, and cleanup state for
each attempted arm. Raw logs, paths, transcripts, and profiler files remain on
the isolated server.

### `910C-056A` correction after invalid M1

The first returned M1 is not a valid execution of the gate: it used two new
high buckets (`[1, 2, 64, 70]`) instead of `[1, 2, H1]`, reported the requested
`mem_fraction_static=0.80` while attributing 46.5 GiB of KV cache to `0.837`,
and estimated the additional compile requirement without a measured capture
peak. Preserve that startup failure as evidence, but do not use it to declare
`[1, 2]` a device limit or choose another memory fraction by trial and error.

Repeat only corrected M1 in a fresh process. From the retained successful A1
run, compute:

```text
required_tokens = max(32768, ceil(1.5 * peak_live_kv_tokens))
```

Set `asr.engine.max_total_tokens=required_tokens` explicitly and retain
`mem_fraction_static=0.80`. This caps the oversized KV pool directly while
preserving at least one native maximum-context request and 50% headroom over
the observed C70 live-token peak. Before graph capture, require the worker's
resolved configuration/startup accounting to report the requested static
fraction, the explicit token cap, and the resulting KV-pool bytes. Any missing
or different value stops the run before capture.

Use exactly `[1, 2, H1]`, where H1 is the single highest
`bucket_hits * bucket_size` bucket from the retained A1 histogram. Do not add
H2, do not substitute both 64 and 70, and do not use `torch_compile_max_bs`.
Require startup logs and model-info to attest that exact compile list. If graph
capture passes, continue in the same service through 140-request correctness
and one 700-request exact10 C70 measurement. If it OOMs, return the last
completed capture bucket, free HBM immediately before it, the first complete
OOM traceback, resolved KV token/byte capacity, and cleanup state. Do not retry
with another memory fraction or bucket in this task.

### `910C-056B`: overnight KV attribution and full compile-coverage qualification

Corrected M1c passed with `max_total_tokens=32768`,
`mem_fraction_static=0.80`, and compile buckets `[1,2,70]`. Its 140-request
correctness gate had WER 0.0161 with no garbled output. The 700-request C70 run
had WER 0.0164, p95 1.442 seconds, throughput 59.88 requests/s, RTFx 598.8,
and 22% HBM use. This is the best result so far, but it still misses the
0.500-second p95 and 140-request/s hard targets.

The improvement is not yet attributable solely to bucket 70 because the same
arm reduced the KV pool from 435,584 to 32,768 tokens. Run the following arms
serially. Every arm must use a fresh service process, the pinned exact10 corpus,
the fully enabled ALL feature profile, `max_total_tokens=32768`,
`mem_fraction_static=0.80`, graceful shutdown, and confirmed HBM recovery before
the next arm. Do not edit source, tests, packages, configuration files, or
documentation on the isolated server.

1. **K0 control:** use compile buckets `[1,2]`. Run one 700-request C70
   measurement. Require normal WER, zero garbled output/failure/fallback,
   exact compile-list attestation, graph replay evidence, drain, and clean HBM
   recovery. K0 isolates the gain from KV-pool right-sizing.
2. **F13 treatment:** use all captured decode buckets
   `[1,2,4,8,12,16,24,32,40,48,56,64,70]`. Require startup and model-info to
   attest the exact list. Run 140-request correctness first, then one
   700-request C70 measurement only if correctness passes. Apply the same
   accuracy, fallback, health, and cleanup gates.
3. If F13 OOMs during capture, preserve the first complete OOM and memory
   evidence, clean up, and run one pre-authorized **F8 fallback** in a fresh
   process with `[1,2,32,40,48,56,64,70]`. Apply the F13 gates. Do not try any
   other subset.

If F13, or F8 when used, passes correctness and its first C70 run, perform two
additional fresh-process C70 repeats of that same winning profile. These three
C70 measurements are the overnight reproducibility set; do not use an in-process
`--repeats 3` loop. Stop the set on the first accuracy, request-accounting,
fallback, OOM, health, or cleanup failure.

For the first successful C70 run of the winning full-coverage profile, retain
one bounded performance-attribution capture using existing observability only:
raw and padded decode-bucket histograms, compiled-versus-eager forward counts,
Dynamo graph-break/recompile counts, encoder batch and queue-wait time, guard
wait/hold time, prefill/decode timing, AI Core utilization, power, HBM peak,
and wall-clock/latency distribution. Profiling must not be enabled for the
other two C70 repeats. Do not run soak or realtime.

Return one sanitized comparison table for old A1, M1c, K0, F13, and attempted
F8, plus the three-repeat table for the winning profile. Include resolved KV
tokens/bytes, compile buckets, capture time/peak HBM, correctness, WER,
garbled/failure/fallback counts, decode coverage, p50/p95/p99/max, throughput,
RTFx, utilization, queue/guard timing, and cleanup. Explicitly calculate the
KV-sizing contribution (K0 versus old A1), sparse bucket-70 contribution (M1c
versus K0), and full-coverage contribution (F13 or F8 versus K0 and M1c).

### `910C-057`: attention compile continuity and execution-guard scope

`910C-056B` selected the qualified M1c profile `[1,2,70]` as the only
reproducible baseline. K0 was valid but slower (p95 4.873 seconds).
F13 first completed at p95 1.504 seconds, but its required second fresh-process
repeat hung during warm-up; it is not a reproducible candidate. Keep
`torch_compile_bs=[1,2,70]`, `max_total_tokens=32768`,
`mem_fraction_static=0.80`, the exact10 manifest and all other
service/benchmark settings fixed. Do not invent another bucket set.

Candidate revisions are:

- SGLang branch `qwen3-asr-perf-next` at `27b246532`, based on `edc504ff6`,
  with `caa1d2916` (explicit-state attention), `b03400e5a` (integrator graph
  dispatch context), and the production graph sequence-length correction;
- sglang-omni branch `qwen3-asr-guard-scope` at the handoff commit containing
  this reauthorization (code through `fb5110a9`), based on `c50996e7`. This
  includes the graph-scope implementation, labeled timing, and the legacy
  runner/test compatibility correction required by the first `910C-057`
  preflight.

The new paths are opt-in. Their absence must preserve the already qualified
behavior. Execute the following arms serially on one clean NPU. Every arm uses
a fresh service, graceful shutdown, port release, and verified HBM recovery.
Do not run arms in parallel and do not edit source, tests, packages, or the
bucket list on the isolated server.

Before the first service, check out clean worktrees at the Omni handoff commit
containing this reauthorization and SGLang `27b246532`. Run the
candidate-focused suites before starting any arm:
SGLang `test/registered/unit/layers/test_radix_attention.py`,
`test/registered/unit/runner/test_decode_cuda_graph_runner.py`, and
`test/registered/unit/model_executor/test_external_graph_execution_context.py`,
plus `test/registered/unit/npu/attention/test_npu_ascend_backend.py`;
Omni `tests/unit_test/utils/test_execution_guard.py`,
`tests/unit_test/qwen3_asr/test_pipeline.py`, and
`tests/unit_test/qwen3_asr/test_encoder_service.py`. Then run the existing full
`tests/unit_test/qwen3_asr/` suite. Stop before service startup on the first
collection or test failure. Record the actual pass/skip counts rather than
assuming the historic totals.

1. **AC -- attention compile continuity.** Use the candidate SGLang and the
   baseline Omni behavior. Set
   `SGLANG_NPU_TORCH_COMPILE_DIAGNOSTIC=explicit-state-attention` and leave
   `SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE` unset (resolved scope must be
   `forward`). The NPU custom-op ABI explicitly carries output cache location,
   sequence lengths, block tables, and key/value cache storage; unsupported
   MLA/SWA variants must fail rather than silently take this path. Run the
   focused SGLang attention/decode-runner tests, then the 140-request exact10
   correctness gate. Continue to one 700-request C70 measurement only when WER
   is in the established range, garbled/failed/timeout/missing/duplicate/
   unexpected counts are zero, and all graph/fallback gates pass. Compare
   graph-break and recompile counts with the selected `910C-056B` baseline.
2. **GS -- guard scope.** Unset the attention diagnostic so the established
   stateful-attention graph break is the control. Set
   `SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=graph`; model-info must report
   `device_execution_guard.scope=graph`. First run the historical cold-input
   70-sample concurrency-8 no-warmup deadlock gate. Then run the 140-request
   exact10 correctness gate and one 700-request C70 measurement. Require zero
   encoder capture failure/fallback, zero prefill/decode standard-eager or
   unexpected fallback, balanced guard tickets, and a final outstanding count
   of zero. Returned guard stats must separately contain `encoder_batch`,
   `generation_prefill_graph`, and `generation_decode_graph`; use their wait
   and hold totals to quantify time removed from the old full-forward critical
   section.
3. **AG -- combined candidate.** Run only if AC and GS independently pass all
   correctness, fallback, deadlock, and cleanup gates. Enable both environment
   values, repeat the cold concurrency-8 gate and 140-request correctness, then
   run one C70 measurement. This arm determines whether the two improvements
   compose; it is not a substitute for either independent arm.

Stop the current arm immediately on a focused-test failure, startup/capture
error, WER regression, garbled output, request-accounting error, eager/fallback
increment, 90 seconds without a completion, OOM, unbalanced guard state, or
failed HBM cleanup. A failed AC must not block independent GS. A failed GS may
still return AC performance, but AG must not run. No soak, realtime, second
bucket search, or source patch is authorized by this task.

Return one table containing the selected `910C-056B` baseline and AC/GS/AG:
exact revisions and environment, compile buckets, validity and WER, all error
counts, graph replays/eager/fallbacks, graph-breaks/recompiles, guard labeled
wait/hold statistics, encoder queue timing, NPU utilization/HBM/power, p50/p95/
p99/max, throughput, RTFx, and cleanup. Preserve full logs and raw records only
on the server.

### `910C-057R`: classify the retained GS failure without another run

`910C-057` did not produce a promotable candidate. AC completed its
140-request correctness gate but produced WER about 1.44 with 140/140 garbled
outputs; the explicit-state custom-op ABI is rejected and must remain opt-in.
GS reported a runtime abnormality, but the returned summary omitted the first
exception and execution boundary, so its mechanism is not yet classified. AG
correctly did not run.

Do not start a service or rerun a benchmark for this task. Read only the
retained GS server log, event JSONL, model-info snapshots, and cleanup record.
Return:

- the exact Omni/SGLang HEADs, complete effective environment switches, and
  whether the attention diagnostic was absent;
- the first exception type, message, and sanitized traceback from the first
  project-owned frame through the failing call;
- whether failure occurred during startup, graph capture, batch one, cold
  concurrency 8, 140 correctness, C70, drain, or cleanup;
- the last `device_execution_guard` model-info object, including scope,
  tickets, outstanding count, and every labeled wait/hold aggregate;
- the last encoder/prefill/decode graph replay, standard-eager, and fallback
  counters, plus pending/running request counts;
- final process, port, NPU health, and HBM state.

If there is no Python exception, return the last 40 sanitized project-owned log
events before progress stopped, the no-completion duration, thread/phase
markers, and the same counters. Do not infer `runtime abnormality` as a guard
deadlock without this evidence. No source, test, package, configuration, or
documentation edit is authorized.

### `910C-058`: replace graph-only guard scope with the complete model boundary

`910C-057R` proved that graph-only scope passes cold concurrency 8 and the
140-request correctness gate, then makes no progress with 70/70 requests
pending at C70 for at least 90 seconds.  This is a liveness regression, but it
does not prove that the FIFO primitive itself is defective.  The graph-only
scope leaves graph eligibility, device-input preparation, and eager/compiled
fallback outside the shared encoder/generation guard.

Check out the exact commits named by the corresponding handoff row.  Set
`SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=model`; do not enable the rejected
explicit-state-attention diagnostic.  The new scope holds the existing FIFO
guard around SGLang `_forward_raw`, covering the complete model device region
while excluding Omni scheduling, sampling, event emission, and graph-counter
bookkeeping.

Run serially from fresh processes:

1. the focused Omni guard/pipeline tests, focused SGLang external-context
   tests, and the full Qwen3-ASR unit suite;
2. cold concurrency 8 and the 140-request correctness gate;
3. only if both pass, the fixed M1c exact10 C70 workload with
   `max_total_tokens=32768`, `mem_fraction_static=0.80`, and
   `torch_compile_bs=[1,2,70]`.

Require WER within the qualified band, zero garbled output, zero forbidden
fallback/capture/device errors, positive encoder/prefill/decode replay, and a
drained service.  Model info must report `scope=model`; labeled
`generation_prefill_model`, `generation_decode_model`, and `encoder_batch`
acquires/releases must balance with final guard `outstanding=0`.  Stop on the
first failure and return its complete sanitized boundary.  Do not run the AC
or AG candidates and do not edit server source, tests, packages, configuration,
or documentation.

### `910C-058R`: identify the unreturned guard holder from retained evidence

Do not rerun hardware and do not describe 70 waiters as a FIFO deadlock without
identifying the holder. Read only the retained `910C-058` server log, request
events, poller snapshots, and model-info payloads. For each label
(`encoder_batch`, `generation_prefill_model`, and
`generation_decode_model`), return:

- wait, acquired, and released event counts;
- minimum and maximum acquired ticket, duplicate tickets, and gaps;
- every acquired ticket without a matching release, including thread, phase,
  request/batch identity, acquisition time, and last event inside its body;
- every release without a matching acquire;
- final `next_ticket`, `serving_ticket`, and `outstanding`, plus their last
  change timestamps.

Then classify exactly one outcome: reentrant self-wait when one host thread
requests a second ticket before releasing its first; holder-side device stall
when one guarded encoder/model call does not return; guard state-machine defect
when all acquired tickets were released but serving did not advance or the
next waiter did not wake; or observation gap when retained evidence lacks the
required ticket-correlated events. For a holder-side stall, include the last
paired begin/return markers for encoder, model forward, graph update/replay,
and compile dispatch. Return sanitized text only. No source, test, package,
environment, or documentation change is authorized.

### `910C-059`: order NPU graph input update and replay before guard narrowing

`910C-058R` found a holder-side device stall, not a FIFO ticket-lock defect.
The scheduler held a generation ticket while `replay_with_input_update` joined
its Python update thread; `graph.replay()` had returned, but that helper thread
had not returned from `graph.update()`. Request builders waiting for later
encoder tickets were downstream victims. Releasing the guard around the join,
using separate encoder/generation guards, or reordering FIFO tickets would
reintroduce the encoder/update overlap that the qualified forward-scope guard
exists to prevent and is not authorized.

Check out Omni at the commit containing this section and SGLang at exact commit
`e4d18390a`. Use a clean process and the qualified ALL M1c profile:
`max_total_tokens=32768`, `mem_fraction_static=0.80`, and
`torch_compile_bs=[1,2,70]`. Set:

```bash
export SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE=ordered
export SGLANG_LOG_DECODE_GRAPH_KEY=1
export SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=forward
```

The SGLang change is NPU-backend-only. In `ordered` mode it performs
`NPUGraph.update()` and `NPUGraph.replay()` on the model execution thread in
that order; it creates no Python update thread and has no update-thread join.
The historical `threaded` mode remains the default for this first hardware
gate. CUDA graph backends and their control flow are unchanged.

Run serially:

1. focused SGLang NPU decode graph/backend tests, focused Omni guard/pipeline
   tests, and the full Qwen3-ASR suite;
2. one fresh-process Arm O with the qualified `forward` guard: batch-one,
   cold concurrency 8, the 140-request correctness workload, then the exact10
   700-request C70 workload;
3. only if Arm O is correct, live, and drained, start a second fresh process
   with the sole additional change
   `SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=graph`; repeat the 140-request
   correctness workload and exact10 C70 workload. Do not run `scope=model`.

Before each service, require healthy chips, baseline HBM, a free port, no NPU
holder, clean tracked worktrees, and exact import/commit identity. After each
service, use graceful shutdown and require the same cleanup invariants. Stop
on the first unit failure, `ordered_update_begin` without
`ordered_update_return`, `ordered_replay_begin` without
`ordered_replay_return`, 90 seconds without a completion, accuracy/fallback
failure, device error, or cleanup failure. Do not retry with another ordering,
guard scope, token cap, bucket set, or memory fraction.

For each arm return sanitized text containing request validity and counts, WER,
garbled count, wall time, p95/p99, throughput and RTFx; encoder/prefill/decode
capture/replay/fallback counters; guard wait/acquire/release counts by label,
maximum wait and hold time, final tickets/outstanding; and ordered update/replay
begin/return counts. In ordered mode every update/replay pair must be complete
and no `update_thread_*` or `update_thread_join_*` event may occur during
replay. Arm O establishes the backend repair independently. The conditional
graph-scope arm determines whether that repair also makes guard narrowing live
and beneficial. No server source, test, dependency, benchmark, configuration
policy, or documentation edit is authorized.

#### `910C-059` result and retained-artifact completion

The ordered backend passed the liveness portion of both arms. Arm O completed
cold conc8 70/70, correctness 140/140, and C70 700/700. Its ordered
update/return and replay/return counts were all 5,083, update-thread markers
were zero, and final guard state was `next=serving=5731, outstanding=0`. Arm G
then completed correctness 140/140 and C70 700/700 with 4,706 complete ordered
pairs, zero update-thread markers, and final
`next=serving=5325, outstanding=0`. This demonstrates that ordered input update
removes the previously observed helper-thread/join liveness failure and makes
graph scope live at C70. It does not establish correctness or performance
benefit.

The text return omitted metrics required to judge correctness and performance.
Without starting a service, read the retained Arm O/G result JSON, raw JSONL,
event JSONL, model-info snapshots, poller, and logs and return for each arm:

- valid/total/evaluated plus failed, timeout, missing, duplicate, and unexpected;
- WER and garbled count;
- wall time, latency p95/p99/max, throughput, and RTFx;
- encoder, prefill, and decode capture/replay/fallback counters;
- per-label guard wait/acquire/release counts and wait/hold maxima;
- graceful shutdown, port, device health, HBM baseline, and residual-holder state.

The retained result rejects graph scope: its 140-request WER was 0.0778 versus
the qualified approximately 0.016 band, while C70 p95 was 1.81 seconds and
throughput 51.66/s versus M1c 1.442 seconds and 59.88/s. Encoder hold max also
remained 12.641 seconds, and shutdown left HBM at 22%. Ordered update fixes the
join-based liveness mechanism, but graph scope is inaccurate, slower, and not
cleanly torn down.

### `910C-060`: ordered update with complete model-device scope

The graph-only guard excludes eager/compiled model device work and failed
accuracy. The broad forward guard is correct but includes more Omni work than
the device boundary. The remaining coherent candidate is the SGLang
`_forward_raw` model scope. Its earlier stall occurred in the update-thread
join now removed by ordered mode.

First require a clean idle NPU baseline and no holder; do not start while the
22% residual state remains. Use exact SGLang `e4d18390a`, set
`SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE=ordered` and
`SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=model`, and keep every M1c setting fixed.
Use one fresh service process:

1. run the 140-request correctness workload;
2. stop immediately unless all requests complete, garbled/failure/fallback
   counts are zero, and WER is in the qualified band;
3. only after correctness passes, run the 700-request exact10 C70 workload;
4. gracefully stop and require idle HBM, a free port, and no holder.

Return complete WER/request validity, latency p95/p99/max, throughput, RTFx,
encoder/prefill/decode capture/replay/fallback counters, ordered marker pairs,
and per-label guard wait/acquire/release plus wait/hold maxima. Compare with
M1c and both `910C-059` arms. Accept model scope only if it remains correct and
live, cleans up fully, and materially improves M1c end-to-end performance.
Otherwise retain broad forward scope and close guard narrowing. Do not edit
the server or test another guard scope.

`910C-060` ran and was live: correctness completed 140/140 with zero garbled
output, 4,707 ordered update/replay pairs, zero update-thread markers, and a
final guard state of `next=serving=5324, outstanding=0`. Its teardown HBM was
not confirmed, and its C70 result did not materially beat M1c, at p95 1.75
seconds and 52.79 requests/s against 1.442 seconds and 59.88 requests/s.

Its 140-request WER of `0.0778`, which `910C-059` also used to reject graph
scope, is withdrawn as an accuracy basis. `910C-062` measured the same gate on
the qualified harness with two arms and returned `0.0176` (threaded) and
`0.0179` (ordered), retaining raw per-sample output, and `910C-063` returned
`0.0179`. The `0.0778` readings share the era and protocol of `910C-061`, whose
run used a bespoke client that persisted no raw output. Until those earlier runs
are re-measured on the qualified harness, neither `910C-059`'s nor `910C-060`'s
accuracy rejection stands, and their performance comparisons are equally
inadmissible. Treat `graph` and `model` scope as not adjudicated on evidence and
closed only for absence of benefit: `910C-059` arm G was slower than M1c, and
`910C-063` shows the entire guard is worth only a few percent, so narrowing it
cannot be the performance lever. Retain the broad `forward` scope.

The 140-request gate value on this stack is `0.0176`-`0.0179` across three
independent configurations (`910C-062` arms A and B, and `910C-063`), while
`910C-056B` recorded `0.0161` for M1c. That difference is roughly four word
errors and is a stack-level property rather than an effect of any update mode.
Restate the band as approximately `0.018` for the current stack, or account for
the difference, before `0.0161` is used as a gate value again.

### `910C-061`: ordered update with the qualified forward guard

`910C-059` established that ordered input update removes the helper-thread/join
liveness failure, but Arm O ran under the qualified `forward` guard and never
returned accuracy, latency, or graph-counter fields. Ordered execution is the
change intended for upstream, so its behavior under the qualified guard scope
is still unqualified: every accuracy-bearing `910C-059`/`060` measurement used
a narrowed scope, and both narrowed scopes returned the same 140-request WER
`0.0778`.

The product target is a 0.500-second p95 at 140 requests/s. The currently
qualified exact10 reference is `910C-056B` M1c: 140-request WER `0.0161`,
700-request WER `0.0164`, p95 1.442 seconds, 59.88 requests/s, RTFx 598.8.

Use exactly SGLang `e4d18390a` for code. SGLang-Omni code must equal
`21e755be`; documentation-only commits may sit on top of it on the same branch,
so verify that identity with `git diff --name-only 21e755be HEAD` and require
every listed path to be under `docs/` before starting the service. Set
`SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE=ordered` and
`SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=forward` (or that variable unset; the
resolved scope must be `forward`). Keep every M1c setting fixed:
`max_total_tokens=32768`, `mem_fraction_static=0.80`, compile buckets
`[1,2,70]`, and all acceleration paths enabled.

Before starting any service, read the retained `910C-059` Arm O artifacts and
state in one line whether they contain a WER field and a latency field. This
settles whether Arm O's accuracy can be recovered instead of rerun; it does not
replace the run below.

Then start one fresh process, and only after healthy chips, idle baseline HBM
with no holder, a free port, clean tracked worktrees, and attested commit
identity. Require startup logs and model-info to attest the resolved guard
scope, the exact compile list, and the ordered update mode. Run, in order:

1. batch one and the cold concurrency-8 gate;
2. the 140-request correctness workload, stopping immediately unless every
   request completes with WER inside the qualified band, zero garbled output,
   zero failed/timeout/missing/duplicate/unexpected results, and zero eager
   fallbacks;
3. only after correctness passes, one 700-request exact10 C70 measurement;
4. graceful shutdown, then idle HBM, a free port, and no residual holder.

Stop on the first unit failure, `ordered_update_begin` without
`ordered_update_return`, `ordered_replay_begin` without `ordered_replay_return`,
90 seconds without a completion, accuracy or fallback failure, device error, or
cleanup failure. Do not retry with another scope, ordering, token cap, bucket
set, or memory fraction.

Return request validity and counts, WER, garbled count, wall time, latency
mean/p50/p90/p95/p99/max, throughput, RTFx, and RTF; encoder, prefill, and
decode capture/replay/fallback counters with replay buckets; the attested
compile list and guard scope; per-label guard wait/acquire/release counts with
wait and hold maxima and the final ticket/outstanding state; ordered
update/replay begin and return counts; whether any `update_thread_*` marker
occurred; and graceful cleanup, port, and HBM state.

Ordered mode with the `forward` guard qualifies for upstream only if the
140-request WER stays inside the qualified band, C70 accuracy matches M1c, C70
p95 and throughput do not regress against M1c's 1.442 seconds and 59.88
requests/s, and cleanup returns the device to baseline. If ordered mode cannot
hold the qualified accuracy under the qualified guard, ordered execution must
be re-examined before it is proposed upstream, and the retained candidate
remains the configuration that produced the M1c result. No server source, test,
dependency, benchmark, configuration policy, or documentation edit is
authorized.

`910C-061` ran and closed the liveness question only. Ordered update under the
retained `forward` guard was live at C70: 700/700 completed with 4,709 complete
ordered update/replay pairs, zero `update_thread_*` markers, a final guard state
of `next=serving=5334, outstanding=0`, 140/140 correctness with zero garbled
output, and cleanup to a 5% HBM baseline. Its reported 140-request WER was
`0.0778`, the same value that rejected graph scope in `910C-059`, and its C70 WER
could not be computed because the run used a bespoke aiohttp client that
persisted no hypotheses.

Its C70 comparison against M1c is not admissible: the run differed from the
qualified protocol in four ways at once - a bespoke client instead of the
packaged exact10 harness, no warm-up pool, the 140-request gate at concurrency 1
instead of the recorded sequence, and a single measurement instead of the
reproducibility set. `910C-062` replaces that comparison with a single-variable
A/B.

### `910C-062`: ordered versus threaded graph input update under one protocol

`910C-061` showed ordered update is live, but neither its accuracy nor its
performance is attributable yet. The 140-request WER `0.0778` both rejects graph
scope in `910C-059` and appears in `910C-061`, so it cannot be treated as corpus
noise in one place and as a failure in another until a single-variable
comparison says which variable produces it.

The recorded references this task adjudicates:

- M1c 140-request correctness gate: WER `0.0161`, zero garbled output;
- M1c 700-request C70: WER `0.0164`, p95 1.442 seconds, 59.88 requests/s,
  RTFx 598.8.

The 140-request gate is accuracy-bearing: it is the gate `910C-059` used to
reject graph scope, so a run that fails it cannot be promoted on C70 evidence
alone.

#### Read-only prechecks

Do not start a service for these. Report as sanitized text:

1. From the retained `910C-056A`/`056B` artifacts, the exact invocation of the
   M1c 140-request correctness gate and of its 700-request C70 run:
   concurrency, client or script used, whether the cold concurrency-8 gate ran
   first, and whether a warm-up preceded the measurement. This establishes
   whether the `910C-061` "concurrency-1 cold path" explanation has any basis.
2. From the retained `910C-061` artifacts, whether any raw per-sample output was
   written, and if so why its hypotheses were unavailable for scoring.
3. Code identity: for SGLang-Omni, `git diff --name-only 21e755be HEAD` must
   list only paths under `docs/`; for SGLang report `community/main..HEAD`.

#### Arms

Exactly two arms, differing in one variable only:

| Arm | `SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE` | Guard scope |
|---|---|---|
| A | `threaded` (default) | `forward` |
| B | `ordered` | `forward` |

Both arms set `SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=forward` or leave it unset;
the resolved scope must be `forward`. Do not combine `threaded` with a narrowed
scope: the join-based liveness failure appeared only under narrowed scopes, and
`forward` is the configuration M1c qualified.

Run arm A first, then arm B. Both arms use:

- `max_total_tokens=32768`, `mem_fraction_static=0.80`, compile buckets
  `[1,2,70]`, and every acceleration path enabled;
- the pinned exact10 corpus, with its manifest SHA-256 reported so the arms are
  provably scored on the same sample set;
- the packaged `benchmarks/eval/benchmark_asr_exact10s.py` harness. A bespoke
  request script is not admissible evidence;
- one fresh service process per repeat, graceful shutdown, and confirmed idle
  HBM before the next process starts.

Each arm runs, in order: batch one, the cold concurrency-8 gate, the
140-request correctness workload, then the 700-request exact10 C70 workload.
Both the 140 and the C70 phase must persist raw per-sample JSONL carrying the
hypothesis, the reference, and per-sample error counts; a run whose raw output
is missing cannot be scored and must be repeated.

After the first C70 of each arm, repeat that arm's C70 in two further fresh
processes with identical settings, so each arm has a three-measurement
reproducibility set. Do not use an in-process repeat loop.

#### Instrumentation

Always-on counters are required for every C70 measurement in both arms:
per-label guard wait/acquire/release counts with wait and hold maxima and the
final ticket/outstanding state, encoder batch and queue-wait timing,
prefill/decode timing, raw and padded decode-bucket histograms, and
compiled-versus-eager forward counts. Enable Dynamo graph-break and recompile
logging for the first C70 of each arm only, never for the remaining repeats and
never for the 140-request gate. Also record AI Core utilization, power, HBM
peak, wall clock, and the latency distribution.

#### Stop conditions

Stop on the first unit failure, `ordered_update_begin` without
`ordered_update_return`, `ordered_replay_begin` without `ordered_replay_return`,
an `update_thread_*` marker inside arm B, 90 seconds without a completion,
missing or unscorable raw output, accuracy or fallback failure, device error, or
cleanup failure. Do not retry with another guard scope, ordering, token cap,
bucket set, or memory fraction.

#### Adjudication

The 140-request WER comparison has three mutually exclusive outcomes:

| Arm A (threaded) | Arm B (ordered) | Conclusion |
|---|---|---|
| `0.0778` | `0.0778` | Not an ordered-mode effect. The discrepancy is a protocol, client, or sample-subset effect and M1c's `0.0161` must be explained on the same basis; `910C-059`'s rejection of graph scope then rests on performance alone and its record must be corrected. |
| `0.0161` | `0.0778` | Ordered mode changes decode numerics. Ordered execution must not be proposed upstream until the cause is found. |
| `0.0161` | `0.0161` | The `910C-061` WER came from the bespoke client, which is discarded. |

Performance is adjudicated only against each arm's own three-measurement band,
never against a single historical number. Ordered mode may be proposed upstream
only if its C70 accuracy stays inside the qualified band and its p95 and
throughput bands overlap arm A's. If the arms differ beyond their measured
spreads, report the direction plus the instrumentation evidence that explains
it instead of a verdict.

### `910C-063`: is the execution guard load-bearing?

The pull-request review asked whether the execution guard adds serialization
without benefit, and whether it should be conditioned on the encoder graph
being enabled. A separate precedent bears directly on the same question:
`vllm-ascend` PR `#4233` records that under ACL graphs the CPU's record event for
one step can complete before the previous step's graph execution has finished,
so "if cpu is fast enough, device will hang on `event_wait` in iteration i+1".
PR `#6432` then narrowed that repair from a full device synchronize to
`torch.npu.current_stream().synchronize()`, and noted the wider form was
unnecessary and costly when background work runs concurrently.

Two mechanisms remain undistinguished on this stack:

- **ordering**: the encoder and generation merely interleave and need ordering,
  in which case a stream event between them could replace the host lock; or
- **runtime**: the NPU graph runtime does not accept concurrent submission from
  two host threads, in which case no amount of ordering helps and the host lock
  is load-bearing.

`910C-063` adds a diagnostic `off` guard scope to discriminate them. `off` is
not one of the rejected narrowed scopes (`graph`, `model`); those were rejected
for accuracy, while `off` removes the guard entirely, which is the boundary the
review question and the `vllm-ascend` precedent both point at.

This arm is complementary to `910C-062`, which varies the graph input-update
mode under the qualified `forward` guard. `910C-063` removes the guard under the
ordered mode that `910C-062` arm B measures. Run `910C-063` only after
`910C-062` has completed, so neither result is attributed to the other.

Verify the code identity by requiring `off` in the accepted scope set:

```
rg -n "_NPU_GUARD_SCOPES" sglang_omni/models/qwen3_asr/engine_builder.py
```

and require `git diff --name-only <that commit> HEAD` to list only paths under
`docs/`. Use exactly SGLang `e4d18390a`. Set
`SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE=ordered` and
`SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=off`, and keep every M1c setting fixed
(`max_total_tokens=32768`, `mem_fraction_static=0.80`, compile buckets
`[1,2,70]`) with the encoder graph, prefill graph, decode graph, and compile all
enabled.

Require the startup log to carry the guard-disabled warning and model-info to
attest that no guard is installed. Then run, in order, in one fresh process:

1. batch one and the cold concurrency-8 gate;
2. the 140-request correctness workload, stopping immediately unless every
   request completes with WER inside the qualified band, zero garbled output,
   zero failed/timeout/missing/duplicate/unexpected results, and zero eager
   fallbacks;
3. only after correctness passes, one 700-request exact10 C70 measurement;
4. graceful shutdown, then idle HBM, a free port, and no residual holder.

Stop on the first unit failure, 90 seconds without a completion, accuracy or
fallback failure, device error, or cleanup failure. If the service cannot be
stopped gracefully, use bounded forced cleanup and report the residual HBM and
any holder. Do not retry with another scope, ordering, token cap, bucket set, or
memory fraction.

Return the same fields as `910C-062`, plus the guard-disabled attestation and,
if the run hangs, the last completed request counts, the poller state, and the
per-label guard statistics, which must be empty while the guard is off.

Predeclared interpretation:

- If the run completes cold concurrency 8, 140 correctness, and C70 without a
  hang and inside the qualified accuracy band, the guard is not load-bearing on
  the fully accelerated path. A later task may then replace it with a stream
  event between the encoder stream and the generation graph, and the review
  request to drop or condition the guard can be answered in code.
- If the run hangs or fails, concurrent NPU graph submission from two host
  threads is unsafe regardless of ordering, the guard is load-bearing, and the
  review request must be declined with this evidence. Retain the `forward`
  scope and change nothing in the retained candidate.

No server source, test, dependency, benchmark, configuration policy, or
documentation edit is authorized.

`910C-063` then ran once and completed: 700/700 at C70 with WER `0.0161`,
p95 1.478 seconds, throughput 57.14 requests/s, graceful shutdown, and idle
HBM. No stop condition fired. Two corrections are required before that is read
as a verdict on the guard.

First, `910C-063` ran the fully accelerated profile with the encoder graph,
prefill graph, decode graph, and compile all enabled. The configuration that
originally hung is a different point in that space. `910C-013` hung with
**encoder graph disabled**, prefill graph disabled, compile disabled, decode
graph enabled through bucket 70, eight request-build workers, and concurrency 8;
its server commit `9bae2619` predates the guard commit `ef6db56c`, so that hang
occurred with no guard installed. `910C-024B` later passed the same
encoder-eager/decode-graph family with the guard enabled. `910C-063` therefore
does not cover the configuration where the guard was introduced, and it cannot
support dropping the guard in general.

Second, that failure mode is intermittent: `910C-014` arm B passed all eager,
`910C-024B` passed with the guard, and `910C-013` hung without it. A passing run
is weak evidence and a hang is strong evidence. `910C-063` is one run, and its
cold concurrency-8 gate was recorded as implicitly passed rather than executed
as its own step, which is the one gate a liveness arm must not elide. The
`910C-062` arms also show the guard costs only a few percent: guarded ranges
were p95 1.539-1.765 s at 50.66-55.95 requests/s (threaded) and 1.559-1.739 s at
51.34-54.51 requests/s (ordered), against the single guard-off reading of
1.478 s at 57.14 requests/s.

### `910C-064`: guard-off liveness at the historical failure profile

The review question is whether the execution guard is load-bearing or merely
orders two device producers. `910C-063` removed the guard at the fully
accelerated profile and passed once, but that profile has never produced the
hang. The configuration that did produce it is the encoder-eager/decode-graph
family from `910C-013`, reproduced below exactly as `910C-014` specified it.

Use exactly SGLang `e4d18390a` and the SGLang-Omni commit containing this
section, with `SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE=ordered` and
`SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=off`. Attest the guard-disabled warning
and, from model-info, that no guard is installed.

Profile, fixed at the `910C-013` point:

- compile disabled;
- encoder graph disabled;
- prefill graph disabled;
- decode graph captured through bucket 70;
- eight request-build workers;
- the pinned 70-input selection;
- concurrency 8 with no benchmark warm-up;
- a bounded request timeout of 120 seconds.

Run three fresh service processes. Each process must execute the cold
concurrency-8 gate as its own explicit step, with the 70 distinct requests
driven closed-loop, and must be stopped gracefully with idle HBM and a free port
before the next process starts. Do not treat an implicit pass as a gate.

Only if all three concurrency-8 gates complete, run a fourth fresh process at
the same profile through the 140-request correctness workload and one
700-request exact10 C70 measurement, to confirm the profile is still usable
end-to-end without the guard.

Stop on the first request that does not complete within its timeout, 90 seconds
without a completion, an accuracy or output failure, an OOM, a device error, a
state leak, an orphan, or a cleanup failure. Do not retry with the guard
re-enabled inside the same task, and do not change the scope, ordering, token
cap, bucket set, memory fraction, compile setting, or graph setting.

Return, per process: the guard-disabled attestation, completion counts and
timeouts, wall clock, request accounting, the poller state at the last
snapshot, whether decode replay executed, encoder batch and queue-wait timing,
and cleanup state. When the guard is disabled the per-label guard statistics
must be empty; report that they are.

Predeclared interpretation:

- Any hang, timeout, or failure in any of the three concurrency-8 gates means
  concurrent NPU graph submission from two host threads is unsafe regardless of
  ordering. The guard is load-bearing, the review request to drop it or
  condition it on the encoder graph must be declined with this evidence, and
  the retained candidate keeps the `forward` scope and its current condition.
- All three gates plus the fourth-process C70 completing means the trigger that
  hung `910C-013` no longer reproduces on this stack. The guard's necessity then
  has to be re-justified rather than assumed, and the next local task is a
  narrow-barrier variant, mirroring `vllm-ascend` PR `#6432`, to decide whether
  a per-handoff `current_stream().synchronize()` can replace the FIFO lock. The
  `off` scope must not be shipped either way: if the guard is load-bearing it
  stays unconditional, and if it is not it is deleted rather than exposed as a
  switch.

No server source, test, dependency, benchmark, configuration policy, or
documentation edit is authorized.

`910C-064` then ran and passed. Three independent cold concurrency-8 gates at
the `910C-013` profile, each in its own fresh process with no guard installed,
completed 70/70 with zero failures and zero timeouts at p95 13.26-14.87 seconds
and 3.6-3.9 requests/s. The fourth fresh process completed the 140-request
correctness workload at WER `0.0179`, then 700/700 at C70 with WER `0.0162`,
p95 1.716 seconds, and 50.5 requests/s. Every process emitted the guard-disabled
warning, model-info attested that no guard was installed, the per-label guard
statistics were empty, and shutdown left the port free with HBM at baseline.

Together with `910C-063` that is four fresh processes across two profiles with
no hang: the trigger that hung `910C-013` does not reproduce on this stack.

One variable nevertheless separates `910C-064` from `910C-013`. `910C-013`
predates ordered input update, so it ran the threaded path, in which the decode
graph itself submits from two host threads: `replay()` on the forward thread and
`graph.update()` on a helper thread. Adding the encoder thread makes three
concurrent host submitters; ordered mode removes the helper and leaves two. That
is consistent with `910C-020`, where `graph_update_begin` never returned,
`replay_begin`/`replay_return` were complete, and the main thread was parked at
`update_thread_join`. The earlier attribution named encoder-versus-decode graph
stream interaction; the sharper form is that the threaded path submits from two
host threads by itself, `replay()` on the forward thread and `graph.update()` on
a helper thread, and the encoder thread makes three. `910C-067` later isolated
that: with the encoder silent, two submitters are live and the profile passes.

So `910C-064` does not show the guard is unnecessary. It shows the guard is
unnecessary *while ordered mode is in force*, which is a different claim and one
that would let the pull request ship ordered mode without the guard.

### `910C-065`: guard-off liveness under the threaded update path

This is the counterfactual that explains the original hang. It changes exactly
one variable from `910C-064`: the graph input-update mode.

Use exactly SGLang `e4d18390a` and the SGLang-Omni commit containing this
section, with `SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE=threaded` (or unset, which is
equivalent) and `SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=off`. Attest the
guard-disabled warning and, from model-info, that no guard is installed.

Keep the `910C-013` profile fixed exactly as `910C-064` ran it: compile
disabled, encoder graph disabled, prefill graph disabled, decode graph captured
through bucket 70, eight request-build workers, the pinned 70-input selection,
concurrency 8 with no warm-up, and a 120-second request timeout.

Run three fresh service processes, each executing the cold concurrency-8 gate as
its own explicit step over the 70 distinct requests, closed-loop. Only if all
three gates complete, run a fourth fresh process through the 140-request
correctness workload and one 700-request exact10 C70 measurement.

The predeclared hypothesis is that this arm hangs. A hang is the result here,
not a fault. Because a hung service may not stop gracefully, this task
explicitly authorizes bounded forced cleanup: after the request timeout plus a
bounded grace period, stop the service, and if it does not exit, terminate it,
then report residual HBM occupancy per chip, whether any holder remains and
whether it is a user-space process, the port state, and the exact cleanup
sequence used. Do not leave a retained driver context unreported, and do not
start the next process until the device is back at its idle baseline or the
residual state is explicitly recorded.

Stop on the first request that does not complete within its timeout, 90 seconds
without a completion, accuracy or output failure, OOM, device error, state leak,
orphan, or cleanup failure. Do not retry with the guard re-enabled, with ordered
mode, or with any other scope, token cap, bucket set, memory fraction, compile
setting, or graph setting.

Return, per process: the graph input-update mode attestation, the guard-disabled
attestation, completion and timeout counts, wall clock, the poller state at the
last snapshot, whether any `update_thread_*` or `ordered_*` marker appeared,
encoder batch and queue-wait timing, and the cleanup record described above.

Predeclared interpretation:

- A hang in any gate, or in the fourth process, confirms the mechanism: under
  the threaded path the decode graph's own update helper is a third concurrent
  host submitter, and adding the encoder thread is what hung `910C-013`. Ordered
  mode removes that helper, so the guard becomes redundant once ordered update
  is adopted and the accelerated candidate can ship ordered update without the
  guard. Record that the guard's original justification is superseded rather
  than wrong.
- Three clean gates plus a clean C70 mean the hazard is not the threaded
  helper. Liveness then requires neither the guard nor ordered mode, the guard's
  necessity is purely historical, and the decision reduces to whether to keep a
  cheap barrier as insurance. In that case the next task is the narrow-barrier
  variant mirroring `vllm-ascend` PR `#6432`, not the `off` switch.

Either way the `off` scope is not shipped: if the guard is retained it stays
unconditional, and if it is dropped it is deleted rather than exposed as a
switch.

`910C-065` then hung at Gate 1, as predicted, and produced the same marker
signature as `910C-020`:

| marker | count |
|---|---|
| `update_thread_enter` | 37 |
| `update_thread_join_begin` | 37 |
| `update_thread_join_return` | 36 |
| `graph_update_begin` | 37 |
| `graph_update_return` | 36 |

One iteration entered `update_thread_join_begin` and never returned, and its
`graph.update()` never returned either; the last event was
`update_thread_join_begin key_size=4`. The encoder thread was present under the
`910C-013` profile. The mechanism is therefore confirmed: under the threaded
path the forward thread waits on the join, the helper thread is stuck inside
`graph.update()`, and the encoder thread is the third concurrent submitter.
Ordered update removes the helper, and `910C-064` shows that with two submitters
the same profile is live with no guard at all. The guard's original
justification is superseded rather than wrong: it was the only correct repair
available before ordered input update existed.

### `910C-066`: does a narrow barrier replace the mutex?

With the guard redundant under ordered mode, the remaining question is whether a
cheap device barrier is worth keeping as insurance. That question can only be
answered at the configuration where the hazard exists. A barrier arm run at the
fully accelerated profile would pass trivially, because that profile has never
hung; it would prove nothing. The test bench must be the hanging configuration:
threaded update with no mutex. Gate 1 of `910C-065` hangs reliably enough to
serve as that bench.

The barrier-only mode this arm needs already exists: `scope=off` together with
`SGLANG_OMNI_NPU_GUARD_COMPLETION_FENCE` builds a `FairDeviceExecutionGuard`
with `serialize=False`, so callers never wait for each other while the
completion fence still runs at every handoff and the per-label statistics stay
empty by construction. That fence is the existing full-device
`device_module.synchronize()`.

Run the strong barrier first, as a falsification step rather than the full
ladder. A handoff fence cannot order the two calls that actually race: the fence
runs when `hold()` exits, while `graph.update()` and `graph.replay()` are issued
concurrently *inside* one generation hold, because the `forward` scope's
`hold()` wraps `forward_batch_generation`, which starts and joins the update
helper. A full-device synchronize at the handoff is a strict superset of the
narrow `torch.npu.current_stream().synchronize()` in `vllm-ascend` PR `#6432`,
so if the strong barrier leaves the hang in place then the narrow one cannot
repair it either, and no narrow-fence implementation is warranted.

So run one fresh process at the `910C-013` profile with
`SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE=threaded`,
`SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=off`, and the fence enabled, executing
the explicit cold concurrency-8 gate once. Attest from the startup logs and
model-info that the mutex is disabled, the barrier is enabled, and the
per-label guard statistics are empty. Bounded forced cleanup is authorized
exactly as in `910C-065`, with the same cleanup record required.

Only if that single gate completes, extend to the full ladder: three fresh
processes each executing the gate as their own explicit step, then one fourth
process through the 140-request correctness workload and one 700-request
exact10 C70 measurement.

Bounded forced cleanup is authorized exactly as in `910C-065`, and the same
cleanup record is required.

Predeclared interpretation:

- The single strong-barrier gate hanging means the barrier does not address the
  mechanism and cannot be justified as insurance, and the narrow-fence variant
  is closed without an implementation. The mutex remains the only proven
  serialization, and the decision becomes ordered alone (`910C-064` evidence)
  versus ordered plus the mutex.
- The single gate completing, and then all three gates plus the fourth process
  completing, means the barrier suppresses the known mechanism even though it
  cannot order the racing pair, which points at device-wide serialization
  rather than at handoff ordering. That reading authorizes implementing the
  narrow fence and measuring its cost against the `910C-064` guard-off
  reference; it does not authorize shipping the full-device fence.

Neither outcome authorizes shipping the `off` scope.

`910C-066` then hung at its single gate, before the ladder was extended. The
barrier-only process reported `update_thread_join_begin` 29 with 28 returns and
`graph_update_begin` 29 with 28 returns, attested both the guard-disabled and
the barrier-only warnings, and completed no gate. The barrier route is therefore
closed without an implementation of the narrow fence: the full-device fence is a
strict superset of the narrow one and still does not address the mechanism,
because a fence that runs at hold exit cannot order a pair issued inside the
hold. Ordered update remains the only mechanism that removes the third
concurrent submitter.

One integration constraint follows from these results and must travel with both
pull requests: removing the guard from the Ascend candidate makes the SGLang
ordered-input-update commit a hard prerequisite. No guard together with the
threaded update path is exactly the configuration that hangs in `910C-013` and
`910C-065`. If the SGLang change has not landed, the guard must remain;
otherwise the default combined state is the hanging one.

No server source, test, dependency, benchmark, configuration policy, or
documentation edit is authorized.

### `910C-067`: is the encoder a necessary participant in the hang?

`910C-065` hung with the encoder submitting, and the confirmed mechanism names
the encoder as the third concurrent host submitter, but no arm has ever run the
hanging configuration with the encoder silent. That attribution is therefore an
inference, not a controlled result. This arm removes the encoder's device
submissions without changing the code path that produces them.

Keep the `910C-013` profile exactly as `910C-064` ran it, with
`SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE=threaded` and
`SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=off`. In each fresh process, first drive
the 70 pinned inputs once at concurrency 1 so the encoder stage cache is
populated, then run the explicit cold concurrency-8 gate over the same 70
inputs. Cache keys are `namespace:audio_fingerprint`, so the second pass
attaches cached embeddings and `submit_item` never queues encoder work.

Run three fresh processes, each performing its own warm pass followed by its own
gate, with graceful shutdown and an idle baseline between them. Bounded forced
cleanup is authorized exactly as in `910C-065`.

Each process must attest from the encoder diagnostics that zero encoder batches
were started during its gate phase; a process whose gate phase shows any encoder
batch is invalid and does not count toward the three. Report the warm pass
separately from the gate; if the warm pass itself hangs, that is that process's
result and the profile is not established.

Predeclared interpretation:

- All three gates completing means the encoder's device submissions are
  necessary for the hang. The mechanism statement then stands as written, and
  the mitigation must keep excluding the encoder from concurrent submission for
  as long as the threaded update helper exists.
- Any gate hanging with zero encoder batches means the encoder is not a
  necessary participant and the encoder-versus-decode-graph attribution is
  withdrawn: the threaded path's own update helper is sufficient on its own, and
  no encoder-side repair can address the hazard.

`910C-067` then ran and passed. Three fresh processes at the `910C-013` profile
with `threaded` update and no guard each warmed the cache with one
concurrency-1 pass over the 70 pinned inputs and then completed the explicit
cold concurrency-8 gate 70/70, at p95 0.52-0.53 seconds, with zero encoder
batches started in every gate phase. `910C-065` hung at the same profile with
the encoder active. The encoder's device submissions are therefore a necessary
participant in the hang, and the attribution is a controlled result rather than
an inference.

`910C-067` sharpens the mechanism in one direction and softens it in another.
With the encoder silent, the threaded path still submits from two host threads,
the forward thread's `replay()` and the helper thread's `graph.update()`, and it
is live; so two concurrent submitters are not sufficient to hang, and the
encoder is the third. That also means the unsynchronized update/replay pair is
not by itself the cause, which is a correction to the reading that followed
`910C-065`. The load-bearing variable is the number of concurrent host
submitters rather than encoder device volume: `910C-024B` ran an encoder that
submitted, serialized against generation by the guard, and passed.

### `910C-068`: sample the hang while it is live

The mechanism is established at the host-marker level: the forward thread waits
at `update_thread_join` while the helper never returns from `graph.update()`.
What is not established is what the helper is blocked *in*. Every hang arm so
far force-cleaned the process, so no runtime or device log was ever captured at
the moment of the stall. This arm reproduces `910C-065` and samples the live
process before any cleanup.

This is the only remaining path to a root cause below our own code, and it is
what would make an upstream report to CANN or `torch_npu` credible rather than
circumstantial. It is also the arm most at risk of not reproducing: host logging
changes timing, and this profile is intermittent.

Profile: identical to `910C-065`, that is the `910C-013` profile with
`SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE=threaded`,
`SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=off`, a cold encoder cache, and the
encoder active. Do not carry the cache warm-up from `910C-067` into this arm.

Step 0, before any new run: check whether plog from the `910C-065` or `910C-066`
hangs is still on disk (`find "$HOME/ascend/log" -name "plog-*"`). If it is,
record the paths and copy the tail; those runs then need no new hardware time.

Before starting the service, prepare CANN host logging:

- `ASCEND_GLOBAL_LOG_TO_FILE=1`, `ASCEND_GLOBAL_LOG_TO_STDOUT=0`,
  `ASCEND_GLOBAL_LOG_LEVEL=1` (INFO). These are the three variables verified in
  this environment. Do not set level 0 in this arm: the volume perturbs timing
  and the hang is intermittent.
- Logs land under `$HOME/ascend/log`. Locate the plog for the server pid by name
  rather than assuming a fixed subdirectory, and record the path found.

When the request gate stops making progress, treat that as the measurement
rather than the end of the run. Then, in this order and inside a bounded
ten-minute window:

1. Do not kill the server. Record when progress stopped and the wall clock then.
2. Capture the native backtrace of the hung process with
   `gdb -p <server pid> -batch -ex "thread apply all bt"`. This is the primary
   artifact: the Python frames identify `graph.update`, and the native frames
   below them identify the call that is actually blocking. If `gdb` is not
   installed, use `py-spy dump --pid <pid>` and record explicitly that native
   frames are missing.
3. Locate the plog for that pid, record its path, size, and last timestamp, and
   copy the final 200 lines. A plog that stops at an API entry with no matching
   return is itself the finding.
4. Record `cat /proc/<pid>/task/*/wchan`, one `npu-smi info`, and one
   `npu-smi info -t usages` snapshot.
5. If `msnpureport` is present, run `msnpureport -h` first and use only the
   syntax that version documents, recording the exact command. Do not guess
   flags.
6. Only after all of the above, perform the bounded forced cleanup authorized
   in `910C-065` and return the same cleanup record.

If an artifact cannot be collected inside the window, record that and continue
with the rest instead of extending the window.

Return, text only and redacted: the plog path and tail excerpt, the native
backtrace excerpt around the blocking frames, the `/proc` wait channels, the
device snapshots, whether the hang reproduced, and the cleanup record. Keep raw
logs, core files, audio, and transcripts on the isolated server.

Predeclared interpretation:

- Native frames naming a blocking ACL or runtime call inside the graph update
  path, with plog stopping at the matching entry, localize the stall to graph
  input update and make the evidence upstream-fileable.
- A wait channel or plog entry tied to another stream's outstanding work means
  the multi-stream interaction is the mechanism rather than host-side locking.
- Native frames that stop in process-local state with no device wait mean the
  mechanism is a host-side interlock inside the graph backend.
- No reproduction in this arm means the logging changed the timing. Record that
  and do not raise the log level to retry inside this arm; the decision is local.

`910C-068` then reproduced the hang and sampled the live process. `gdb` is not
installed on the host, so the capture is `py-spy` plus `/proc`: the update
thread sits in `graph_task_update_end`, and its kernel wait channel is
`eventfd_read`, a wait for a device completion signal rather than a host lock.
`npu-smi` reported AI Core and Aicore utilisation at 0 percent while the process
was stalled, so no kernel was executing. plog's last visible entry is the
encoder's `AllocTaskAndSendStars` for a `KERNEL_AICORE` task, so device work was
still being submitted as the stall began. `msnpureport` is not installed, so no
device-side log exists for this run.

Three qualifications travel with that result:

- The predeclared branch that fires is the one about a blocking call inside the
  graph update path. We have the call and its wait channel, but a Python frame
  plus `wchan` is not a native backtrace, because `py-spy` substituted for
  `gdb`. A later arm must produce native frames before this is quoted as one.
- The stall is a silent indefinite wait on a device completion signal with the
  device idle. That shape fits a submitted task whose dependency is never
  satisfied, the same shape as `vllm-ascend` PR `#4233`, more closely than it
  fits a task the hardware rejected, which would normally raise or time out. The
  reading that the hardware scheduler refused the task is not established here.
- Under INFO the plog ring buffer rotates in roughly four minutes, so the file's
  tail is not guaranteed to be the stall instant. A repeat should stream-copy
  plog while the run proceeds rather than read it after the stall.

The cleanup left a residual holder from an earlier experiment taking about
9.6 GiB. Confirm the device is back at its idle baseline before the next arm.

### `910C-069`: is the shared submission stream the mechanism?

On CUDA the encoder owns a private stream; on NPU it does not, because
`encoder_service.py` creates that stream only when `device.type == "cuda"`. The
consequence is that encoder kernels, `graph.replay()` and `graph.update()` all
submit to the same default stream, and the GPU layout never does. Nothing in the
NPU graph runner or backend switches streams, so every submission in the failing
profile is queued on one lane.

Sharing a lane cannot produce a data race; a shared lane is FIFO ordered, which
is the safer arrangement. What sharing does is let the host run far ahead of the
device: at the `910C-013` profile the shared lane carries roughly thirteen
seconds of encoder work per gate, against 0.53 seconds with the encoder silent
in `910C-067`. If the graph update's event ordering assumes the host does not
outrun the device, that is the precondition it violates.

Two diagnostic switches exist locally for this arm and neither ships:

- `SGLANG_OMNI_NPU_ENCODER_PRIVATE_STREAM=1` gives the NPU encoder the private
  stream CUDA already has. This is more than one change: `synchronize_batch()`
  stops being a no-op and the embedding handoff registers its consumer, which is
  the CUDA handoff contract.
- `SGLANG_OMNI_NPU_ENCODER_BATCH_SYNC=1` leaves the encoder on the shared
  default stream and blocks the worker until that stream drains after every
  batch. This is the host-throttle control, and it is what makes the private
  stream arm interpretable: it holds the "host does not run ahead" property
  constant while the stream layout changes.

Setting both selects the private-stream arm and logs that the fence was ignored.

Run both arms at the `910C-013` profile with
`SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE=threaded`,
`SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=off`, and a cold encoder cache, so the
encoder is active. Run the explicit cold concurrency-8 gate in fresh processes,
stop an arm at its first hang, and treat three completed gates as live. Bounded
forced cleanup is authorized exactly as in `910C-065`.

Attest, per arm, the matching warning line from the startup log, plus code
identity with
`rg -n "SGLANG_OMNI_NPU_ENCODER_PRIVATE_STREAM|SGLANG_OMNI_NPU_ENCODER_BATCH_SYNC" sglang_omni/models/qwen3_asr/encoder_service.py`.

Predeclared interpretation, a two-by-two over the arms:

| shared stream + batch fence | private stream | reading |
|---|---|---|
| live | live | Stream sharing is not the mechanism and keeping the host from running ahead is sufficient, which matches the guard result in `910C-024B`. Close the stream question without a code change. |
| hang | hang | Neither throttling nor stream separation addresses it, so the hazard sits below the host submission layer, consistent with the never-signalled completion in `910C-068`. Close the stream question as well. |
| hang | live | Stream sharing is the mechanism and the CUDA-parity layout removes it. That is a real defect with a real fix, and it still needs its own validation before it is proposed for shipping. |
| live | hang | Not expected; report it and stop. The arm is not interpretable and must not be read as either outcome above. |

No outcome here authorises shipping either switch, and no outcome changes the
retained decision: the guard stays unconditional and ordered input update
remains the only mechanism that removes the third concurrent submitter. A "live"
result for the private-stream arm would additionally need the allocator
registration and the per-batch synchronize to be examined on their own, because
that arm changes three things at once.

`910C-069` then ran and landed on the third row of that table: arm P hung at
Gate 1, and arm Q completed three gates. Both arms attested their warning line.
The prediction recorded here before the run was that arm P would be live, on the
reading that a per-batch drain is a strong serialization comparable to the
guard; that prediction was wrong, and the reason is worth keeping.
`current_stream().synchronize()` blocks the encoder worker only *after* a batch,
so the encoder still submits concurrently with the generator's replay and update
inside every batch. P therefore never removes the concurrency that the guard
removes; it adds a drain to a hazard that has already been entered. Arm Q, by
contrast, keeps the same concurrency and changes only which queue the encoder's
work occupies, and it is live.

That makes the mechanism statement sharper, and it also corrects the earlier
wording. Two concurrent host submitters on one stream are safe: `910C-067`
removed the encoder and the threaded `replay()` plus `graph.update()` pair stayed
live. Three submitters on one stream hang: `910C-065`, `910C-066` and arm P, each
with the encoder active and sharing that stream. Moving the encoder off the
shared stream while keeping all three submitters running removes the hang: arm
Q. So the encoder is the third submitter, and the shared submission lane is what
makes the third one fatal. The plausible reason is the one already recorded
above, that `torch_npu`'s graph update orders itself against the caller's stream
through its own update stream and events, and a third producer on that stream
can invert that ordering; this run did not capture a device-side log, so that
stays an explanation rather than a proof.

Two consequences follow, and neither is yet a shipping decision.

- The whole handoff-barrier family is now closed with evidence from two
  directions: a full-device fence at hold exit (`910C-066`) and a per-batch
  stream drain (arm P) both leave the hang in place.
- The private stream is no longer only a CUDA-parity tidy-up. It is a candidate
  replacement for the guard itself, because it removes the hazard while keeping
  the encoder and generation overlapping. Validating that is the next arm, and
  it must cover correctness at the accelerated profile, not just liveness at
  the `910C-013` profile.

The `SGLANG_OMNI_NPU_ENCODER_PRIVATE_STREAM` switch as it stands is still not
shippable: it changes three things at once (the stream, an active
`synchronize_batch()`, and the embedding consumer registration), and only
liveness has been measured. The run also left HBM at 63 percent, so the device
was not returned to its idle baseline and that must be resolved before the next
arm.

### `910C-070`: can the private stream replace the guard?

`910C-069` established that the hazard is the encoder sharing the generation's
submission lane, and that moving the encoder to its own stream removes the hang
while keeping all three host submitters running. That makes the private stream a
candidate replacement for the guard rather than a CUDA-parity tidy-up, and the
question this arm answers is whether it survives the shipping profile.
`910C-069` measured liveness only, at the `910C-013` profile, with compile and
the encoder graph disabled. Neither correctness nor performance at the
accelerated profile has been measured with the private stream.

Two arms, both at the M1c profile (`max_total_tokens=32768`,
`mem_fraction_static=0.80`, compile buckets `[1,2,70]`, encoder, prefill and
decode graphs enabled) and both with `SGLANG_NPU_GRAPH_INPUT_UPDATE_MODE=threaded`,
which is the SGLang default and the configuration the guard exists for:

- Arm A, the candidate: `SGLANG_OMNI_NPU_ENCODER_PRIVATE_STREAM=1` and
  `SGLANG_OMNI_NPU_EXECUTION_GUARD_SCOPE=off`. Leave
  `SGLANG_OMNI_NPU_ENCODER_BATCH_SYNC` unset.
- Arm B, the retained reference: guard at its default `forward` scope with the
  private-stream switch unset.

Each arm runs one fresh process through the 140-request correctness workload,
then three fresh processes each running one `exact10` C70 measurement with the
packaged harness and its warm-up. Stop an arm at its first hang, timeout,
accuracy failure, OOM, device error, or cleanup failure. Bounded forced cleanup
is authorized as in `910C-065`. Compare only A against B on this session; do not
compare against historical M1c numbers, which come from a different day and
process lifecycle.

Attest per arm: the private-stream warning line and `device_execution_guard: null`
from model-info in arm A; a non-null `device_execution_guard` and the absence of
that warning in arm B; code identity with
`rg -n "SGLANG_OMNI_NPU_ENCODER_PRIVATE_STREAM" sglang_omni/models/qwen3_asr/encoder_service.py`;
per-request WER and garbled counts for the correctness gate; p95, throughput and
RetFx for each C70 repeat; and the graceful-shutdown and idle-baseline record.

Predeclared interpretation:

- Arm A passes the correctness gate in band with zero garbled output, and its
  three C70 repeats are not materially worse than arm B's, means the private
  stream is a viable replacement for the guard. The guard's justification then
  becomes historical in the same sense ordered update made it historical: it was
  the only repair available while the encoder shared the generation lane. The
  follow-up is a real change rather than a switch, making the NPU stream
  unconditional, resolving the allocator registration question below, and
  restating both pull requests.
- Arm A failing the correctness gate, in particular producing garbled output,
  means the private stream is not sufficient at the accelerated profile and the
  guard stays. It would also point at the encoder handoff rather than at device
  scheduling, because the embedding is produced on one stream and consumed on
  another.
- Arm A passing correctness but measuring materially worse than arm B means the
  private stream costs more than the guard buys and the guard stays.
- A hang in either arm means the result contradicts `910C-069`; stop and report
  rather than reading it either way.

Two code-hygiene items must be closed before the private stream is proposed for
shipping, independently of how this arm reads. The embedding consumer
registration in `attach_embedding` is guarded by `hasattr` and would skip
silently if `torch_npu` does not expose `record_stream` on tensors, which would
leave the allocator side of a cross-stream handoff unverified. And the switch is
an environment diagnostic, so shipping it means making the behaviour
unconditional and removing the env gate rather than keeping a second switch.

No server source, test, dependency, benchmark, configuration policy, or
documentation edit is authorized.

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
