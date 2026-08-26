# Fun-CosyVoice3 NPU Performance Handoff

## Objective

Produce reviewable Ascend performance and accuracy evidence for community PR
[#1694](https://github.com/sgl-project/sglang-omni/pull/1694), `[NPU]
Cosyvoice3 support and docs`. The server Agent owns hardware execution and
returns redacted results; the local Agent owns repository-quality instructions,
result review, and any later PR-description update.

The exact gate is in
[`fun_cosyvoice3_npu_performance_task.md`](fun_cosyvoice3_npu_performance_task.md).
Use both documents together.

## Scope and non-goals

This qualification covers:

- FunAudioLLM/Fun-CosyVoice3-0.5B-2512 serving on Ascend 910 A3;
- non-streaming SeedTTS English latency, RTF, and throughput at concurrency 1,
  4, and 10;
- a full 1,088-sample concurrency-10 generation run and corpus WER;
- zero request failures and post-run service/device health.

It does not benchmark Qwen3-Omni, Talker NPUGraph, Code2Wav NPUGraph, CUDA, or
another TTS model. It does not compare different hardware types. Do not add an
eager fallback, edit site-packages, or weaken output validity to obtain a pass.

## Branch and PR contract

The local branch `br_omni_cosyvoice3_0824` is an internal NPU integration and
validation branch. It is **not** the head of PR #1694. The similarly named PR
#1682 is closed and its frozen head is `5f31e73e0963314869c0ba5c7c8e857a7dbb2921`.

At the time this handoff was written:

- PR #1694 head: `f9f681883599cd04b64b627baf610208af615c2a`;
- PR #1694 base recorded by GitHub: `c1146aeea2ebc638d0f869d681e9ef6bb6cd5285`;
- current upstream main observed locally:
  `c2d193fc62e22052105fbb6341b5535af156ce95`;
- internal handoff branch base stack: upstream main, community Code2Wav and
  Talker work, then the NPU installer/stability utilities.

The server Agent must fetch `refs/pull/1694/head`, verify its exact HEAD, and
benchmark that checkout. If the PR head changes, stop and return the new hash;
do not silently benchmark a different revision. The internal branch may be used
to read this task but must not be reported as the tested PR revision.

## Dependency and environment record

Before running, record redacted values for:

- `sglang-omni` exact HEAD and dirty status;
- the actual `sglang` HEAD, not only its release label;
- Python, PyTorch, `torch_npu`, CANN, driver, and firmware versions;
- model identifier/revision and config path;
- NPU model, card count, and logical device mapping;
- effective launch command and all relevant environment variables.

The expected environment is Ascend 910 A3 with CANN 9.0.1 and the NPU package
set used for this PR. If the server uses SGLang v0.5.16 plus vendor commits,
record both the v0.5.16 relationship and actual HEAD.

Keep weights, generated WAVs, full logs, host names, addresses, credentials,
and internal filesystem paths on the isolated server.

## Current finding to replace

PR #1694 currently shows a 1,088-request, concurrency-10 run with zero failures,
but its command omits `--model`. Consequently the speed table labels the model
as the benchmark default `fishaudio/s2-pro`, and its WER table labels it
`qwen3-omni`. Those numbers are not accepted as final evidence unless the raw
artifacts prove the intended model and request contract. The new run must pass
the exact CosyVoice3 model identifier explicitly.

## Ordered qualification

Run these phases in order, using a fresh server process and fresh log whenever
the checkout, server arguments, or runtime mode changes:

1. record repository, dependency, and device baselines;
2. run static checks and a 16-sample concurrency-1 smoke;
3. run a 200-sample concurrency sweep at 1, 4, and 10, three independent runs
   per point;
4. run the full 1,088-sample English corpus once at concurrency 10;
5. stop the TTS service and calculate WER from the full-run WAVs;
6. check device and service health and preserve redacted evidence.

Do not claim stability from startup alone. Every performance run must have
valid generated audio and zero failed requests.

## Pass criteria

The gate passes only when:

- the tested checkout equals the recorded PR head and is clean;
- smoke is 16/16 and each sweep run is 200/200 with zero failures;
- the full run is 1,088/1,088 with zero failures;
- every accepted row has non-null latency, RTF, audio throughput, and QPS;
- full-run WER evaluates 1,088/1,088 with zero skipped samples;
- all generated outputs needed by the evaluator are valid non-empty WAVs;
- no traceback, process death, ACL error, OOM, graph/capture/replay failure, or
  unexpected eager fallback appears in the fresh log;
- the health endpoint responds after generation and all participating cards
  remain healthy.

Report run-level values, then use the median of the three run-level values for
each sweep point. Do not pool samples across repeats or select only the best run.

## First-failure and ownership policy

Stop on the first complete failure and retain its first full traceback. Classify
it as capture-time operator compatibility, replay ordering/address failure,
graph-ineligible input, correctness mismatch, memory budget/growth, topology
failure, dependency/API incompatibility, device/ACL recovery, or unrelated
pre-existing failure.

Fix code in the repository that owns the failure. Framework sampling/runtime
changes belong in `sglang`; CosyVoice3 integration, tests, docs, and benchmark
instructions belong in `sglang-omni`. Commit fixes separately and restart the
entire affected phase from a fresh worker. A server-only commit must be mapped
later as `server commit <hash> -> local equivalent <hash>`.

## Completion update

After a pass, update this handoff with the tested runtime matrix, redacted
evidence summary, exact PR/server commit mapping, and remaining non-blocking
limitations. Preserve the procedure for future dependency qualification. Do
not paste private logs or generated media into the repository.

## Redacted result template

Return exactly this structure:

```text
status: pass | fail
tested_pr: 1694
sglang_omni_head: <40-char hash>
sglang_omni_clean: true | false
sglang_release_relation: <tag/version>
sglang_actual_head: <40-char hash>
runtime: python=<x> torch=<x> torch_npu=<x> cann=<x> driver=<x>
hardware: <redacted model/count/topology>
model: FunAudioLLM/Fun-CosyVoice3-0.5B-2512@<revision>
launch_args: <redacted effective args>
smoke: completed=<n> failed=<n>
sweep:
  c1:  [<run 1 summary>, <run 2 summary>, <run 3 summary>]
  c4:  [<run 1 summary>, <run 2 summary>, <run 3 summary>]
  c10: [<run 1 summary>, <run 2 summary>, <run 3 summary>]
sweep_medians: <latency/RTF/audio-throughput/QPS by concurrency>
full_c10: completed=<n> failed=<n> <speed summary>
wer: evaluated=<n>/<n> skipped=<n> corpus=<value>
health_after: http=<status> devices=<healthy|unhealthy>
forbidden_markers: {}
artifacts: <redacted filenames only>
first_failure: <none or classification + first traceback tail>
server_commits: <none or repo/hash/subject>
notes: <bounded redacted notes>
```
