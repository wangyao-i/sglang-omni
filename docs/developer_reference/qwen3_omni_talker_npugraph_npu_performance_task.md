# Qwen3-Omni Talker NPUGraph NPU Performance Task

> **Status: pending isolated-server execution.** Correctness, graph replay,
> concurrency, and stability have already passed. This task adds the numerical
> A/B evidence requested by the pull-request template's Benchmark & Profiling
> section.

## Objective

Measure the steady-state serving effect of enabling the Qwen3-Omni Talker
decode NPUGraph on Ascend 910 A3. Compare graph off and graph on with every
other variable held fixed, then return a small table suitable for the stacked
Talker pull request.

This is a performance measurement task, not a new correctness gate. Report the
numbers honestly even when the result is parity or a regression. Do not tune
launch parameters separately for either mode.

## Scope and Non-Goals

Measure:

- non-streaming end-to-end latency at concurrency one;
- request and generated-audio throughput at concurrency four;
- per-request real-time factor (RTF) and output-token rate;
- Talker graph replay/fallback markers and request/WAV validity;
- NPU 8 memory after warmup and after the measured run.
- launch-to-health time and, when timestamps make it separable, Talker graph
  capture time as secondary startup-cost context.

Do not:

- rerun WER, UTMOS, or speaker-similarity scoring; this patch changes the NPU
  execution path, not model weights or sampling semantics;
- compare different SGLang, CANN, PyTorch, torch_npu, model, prompt, sampling,
  topology, memory-budget, or Code2Wav configurations;
- include server startup/model-loading time in steady-state latency;
- use `/v1/audio/speech`; use `/v1/chat/completions` through the repository's
  Qwen3-Omni SeedTTS benchmark;
- modify code, site-packages, or launch parameters to improve one result.

## Fixed Branch and Runtime Contract

Record exact HEADs before the first run:

```bash
git -C /path/to/sglang-omni rev-parse --abbrev-ref HEAD
git -C /path/to/sglang-omni rev-parse HEAD
git -C /path/to/sglang rev-parse --abbrev-ref HEAD
git -C /path/to/sglang rev-parse HEAD
python - <<'PY'
import torch
import torch_npu

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
PY
```

Expected source stack:

- sglang-omni Talker branch containing the three PR commits based on
  `npu-code2wav-npugraph`;
- SGLang v0.5.16 plus the capture-safe Ascend sampler and `top_ps` dtype fixes;
- Thinker TP=8 on NPU 0-7;
- Talker and Code2Wav on NPU 8;
- Thinker graph off and Code2Wav graph on in both A/B modes;
- `max-running-requests=4` and `cuda-graph-max-bs=4` in both modes.

The only allowed A/B difference is:

| Mode | Launch argument |
|---|---|
| `graph_off` | `--talker-cuda-graph off` |
| `graph_on` | `--talker-cuda-graph on` |

Use the same patched SGLang checkout for both modes. NPU must continue to use
the Ascend sampling backend when the Talker graph is off; otherwise the run
would compare both the sampling backend and the graph at once.

## Dataset and Measurement Matrix

Use the same locally available SeedTTS metadata file for every run. Prefer the
cached `seed-tts-eval` English split. Select the first 32 samples in their
stable dataset order, use speaker `Ethan`, no reference audio, non-streaming
WAV output, `max_new_tokens=256`, and `temperature=0.7`.

Run three complete repetitions for each cell:

| Cell | Measured samples | Concurrency | Warmup |
|---|---:|---:|---:|
| latency | 32 | 1 | 4 |
| throughput | 32 | 4 | 4 |

Alternate the fresh-process order to reduce time drift:

```text
repetition 1: graph_off -> graph_on
repetition 2: graph_on  -> graph_off
repetition 3: graph_off -> graph_on
```

If the isolated environment cannot access the full SeedTTS dataset, reuse the
exact fixed prompt corpus from the completed 32-request stability run. Record
that substitution and retain the same prompt order in all twelve measurements.
Do not mix two corpora in one comparison.

## Server Procedure

For every A/B mode change, stop the old service and launch a fresh process with
a new log. Start from the Phase 2 command in
`qwen3_omni_talker_npugraph_npu_task.md`, keep all its topology and memory
arguments, and change only `--talker-cuda-graph off|on`. Wait for the health
endpoint before benchmarking.

Before each measured cell:

1. verify the mode and effective launch arguments in the new log;
2. capture `npu-smi info` after service readiness;
3. let the benchmark perform four warmups, which are excluded from metrics;
4. run one measured 32-sample cell;
5. wait for the scheduler to drain, then capture `npu-smi info` again;
6. call the health endpoint and scan the complete server log;
7. stop the process before changing graph mode or repetition.

Resolve `<MODEL_ID>` once from `/v1/models` and use it unchanged. With `<META>`
set to the same local metadata source and `<PORT>` set to the active service
port, run:

```bash
python -m benchmarks.eval.benchmark_omni_seedtts \
  --base-url http://127.0.0.1:<PORT> \
  --model <MODEL_ID> \
  --meta <META> \
  --lang en \
  --speaker Ethan \
  --no-ref-audio \
  --max-samples 32 \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --warmup 4 \
  --max-concurrency 1 \
  --generate-only \
  --disable-tqdm \
  --output-dir /tmp/talker_npugraph_perf/<MODE>/r<REPEAT>/c1
```

Then rerun the same command with:

```bash
--max-concurrency 4 \
--output-dir /tmp/talker_npugraph_perf/<MODE>/r<REPEAT>/c4
```

Do not reuse an output directory. Preserve each generated `results.json`, raw
per-request result, WAV directory, server log, and before/after device snapshot
inside the isolated environment.

## Required Metrics

For each mode and concurrency, compute the median of the three repetition-level
values. Do not pool requests across repetitions. Return:

- completed and failed requests;
- `latency_mean_s`, `latency_median_s`, `latency_p95_s`, and `latency_p99_s`;
- `rtf_mean`, `rtf_median`, and `rtf_p95`;
- `throughput_qps`;
- `audio_duration_mean_s` and `audio_throughput_s_per_s`;
- `output_tokens_mean`, `output_throughput`, and `output_tok_per_req_s` when
  present in the API response;
- NPU 8 used-memory snapshots after ready/warmup and after drain;
- process launch-to-health seconds for both modes and Talker capture seconds
  for graph on when the log exposes unambiguous start/end timestamps;
- Talker `npu_graph` and `eager` runtime marker counts for graph-on runs;
- graph capture/replay failure, ACL `107027`, `161002`/`EZ1001`, NonZero, and
  device-error counts.

Calculate relative change as:

```text
higher-is-better delta = (graph_on / graph_off - 1) * 100%
lower-is-better improvement = (graph_off / graph_on - 1) * 100%
```

Use lower-is-better for latency and RTF. Use higher-is-better for QPS, audio
throughput, and token throughput.

## Validity and Investigation Rules

A measurement set is valid only when:

- all 384 measured requests pass (`2 modes * 2 cells * 3 repeats * 32`);
- every response is a non-empty valid WAV;
- each graph-on log contains Talker `execution_mode=npu_graph` replay evidence;
- no graph-on decode batch within sizes 1-4 reports eager execution;
- no capture/replay/ACL/device failure occurs;
- graph-off and graph-on use identical request corpus and launch settings apart
  from the one graph flag;
- health and participating NPUs remain healthy after every cell.

There is no preselected speedup threshold. If the graph-on median regresses by
more than 5% in any primary metric (`c1` latency median/p95, `c4` QPS, or `c4`
audio throughput), stop after preserving the valid measurements and return the
result for profiling. Do not discard the run or tune only graph-on. Changes
within +/-5% may be reported as parity; do not claim a speedup unsupported by
the table.

If a regression needs diagnosis, a second bounded task may enable
`SGLANG_RECORD_STEP_TIME=1` and inspect the Talker worker's `step_time_dict` or
capture one representative NPU profile. Do not add profiling overhead to the
primary A/B measurements.

## Result to Return

Return this redacted structure. Keep private paths, host names, full logs,
weights, and generated audio inside the isolated environment.

```text
Omni branch / HEAD:
SGLang branch / HEAD:
CANN / torch / torch_npu:
Hardware topology:
Dataset identity and selection:
Common effective launch arguments:
Only A/B launch difference confirmed: yes/no

Per-repetition results:
mode,repeat,concurrency,passed,failed,latency_median_s,latency_p95_s,
rtf_median,throughput_qps,audio_throughput_s_per_s,output_throughput

Median summary:
| concurrency | metric | graph_off | graph_on | relative change |
| 1 | latency_median_s | | | |
| 1 | latency_p95_s | | | |
| 1 | rtf_median | | | |
| 4 | latency_median_s | | | |
| 4 | latency_p95_s | | | |
| 4 | throughput_qps | | | |
| 4 | audio_throughput_s_per_s | | | |
| 4 | output_throughput | | | |

Measured requests passed/failed:
WAV validation:
Graph-on Talker graph/eager markers:
Forbidden marker counts:
NPU 8 memory observations:
Launch-to-health and graph-capture observations:
Post-run endpoint/NPU health:
Artifacts retained:
Anomalies or substitutions:
Working trees unchanged:
```

After the result returns, update the Talker handoff with the compact median
table and use the same table in the pull request's Benchmark & Profiling
section.
