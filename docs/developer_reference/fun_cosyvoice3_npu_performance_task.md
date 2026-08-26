# Fun-CosyVoice3 NPU Performance Gate

This task is the executable companion to
[`fun_cosyvoice3_npu_performance_handoff.md`](fun_cosyvoice3_npu_performance_handoff.md).
Read the handoff first and return its exact redacted result template.

## 1. Pin and record the PR checkout

Use a dedicated worktree. Replace `<repo>` and `<worktree>` only with server-local
paths; do not return those paths.

```bash
set -euo pipefail
git -C <repo> fetch upstream refs/pull/1694/head
git -C <repo> worktree add --detach <worktree> FETCH_HEAD
cd <worktree>

test "$(git rev-parse HEAD)" = \
  "f9f681883599cd04b64b627baf610208af615c2a"
test -z "$(git status --porcelain)"
git rev-parse HEAD
git -C /path/to/sglang describe --tags --always --dirty
git -C /path/to/sglang rev-parse HEAD
python - <<'PY'
import platform
import torch
import torch_npu

print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
PY
npu-smi info
```

If PR #1694 no longer has the pinned head, stop and return the new head. The
local Agent will update this document before qualification resumes.

## 2. Focused checks

Run checks that cover the PR-owned files before hardware startup:

```bash
git diff --check "$(git merge-base HEAD upstream/main)"..HEAD
python -m compileall -q sglang_omni
python -m pytest -q tests/unit_test/benchmarks/test_tts_seedtts_benchmark_config.py
```

If a command is unavailable because the pinned PR predates a test or lacks a
local dependency, report that exact limitation. Do not install an unrecorded
package merely to turn the check green.

## 3. Start a fresh CosyVoice3 server

Use the exact served model identifier below. Keep a fresh server log and record
the model revision separately if the local weight checkout exposes one.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SGLANG_OMNI_STARTUP_TIMEOUT=1800
export COSY_MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
export COSY_PORT=8764
export COSY_LOG=/tmp/cosyvoice3_pr1694_server.log

npu-smi info > /tmp/cosyvoice3_pr1694_npu_before.txt
sgl-omni serve \
  --model-path "$COSY_MODEL" \
  --port "$COSY_PORT" \
  --config examples/configs/fun_cosyvoice3_0_5b.yaml \
  2>&1 | tee "$COSY_LOG"
```

Wait for readiness from another shell:

```bash
curl -f "http://127.0.0.1:${COSY_PORT}/v1/models"
```

## 4. Smoke gate

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model "$COSY_MODEL" \
  --port "$COSY_PORT" \
  --output-dir /tmp/cosyvoice3_pr1694/smoke_c1 \
  --lang en --max-samples 16 --warmup 1 --concurrency 1 \
  --generate-only --use-existing-server
```

Require 16 completed and zero failed requests before continuing.

## 5. Three-repeat performance sweep

Use the same warm server for all nine runs. Each output directory must be new;
do not overwrite or reuse WAVs from another point.

```bash
for concurrency in 1 4 10; do
  for repeat in 1 2 3; do
    out="/tmp/cosyvoice3_pr1694/sweep_c${concurrency}_r${repeat}"
    python -m benchmarks.eval.benchmark_tts_seedtts \
      --meta zhaochenyang20/seed-tts-eval-arrow \
      --model "$COSY_MODEL" \
      --port "$COSY_PORT" \
      --output-dir "$out" \
      --lang en --max-samples 200 --warmup 1 \
      --concurrency "$concurrency" \
      --generate-only --use-existing-server
  done
done
```

For every `speed_results.json`, preserve and report:

- completed and failed request counts;
- latency mean, median, p95, and p99;
- RTF mean, median, p95, and p99;
- audio duration mean and audio throughput seconds/second;
- output throughput tokens/second, output tokens/request-second, and QPS.

Report all three run-level rows and the median of each run-level metric. Do not
merge per-request samples across repeats.

## 6. Full corpus generation and WER

Generate the full English set once at concurrency 10:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model "$COSY_MODEL" \
  --port "$COSY_PORT" \
  --output-dir /tmp/cosyvoice3_pr1694/full_c10 \
  --lang en --warmup 1 --concurrency 10 \
  --generate-only --use-existing-server

curl -f "http://127.0.0.1:${COSY_PORT}/v1/models"
npu-smi info > /tmp/cosyvoice3_pr1694_npu_after_generation.txt
```

Require 1,088 completed, zero failed, and 1,088 usable generated WAV entries.
Stop the CosyVoice3 service cleanly before starting ASR so it does not compete
for the same NPU memory. Then transcribe the exact full-run directory:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model "$COSY_MODEL" \
  --output-dir /tmp/cosyvoice3_pr1694/full_c10 \
  --lang en --device npu:0 \
  --transcribe-only
npu-smi info > /tmp/cosyvoice3_pr1694_npu_final.txt
```

If the approved ASR environment uses a different device or service topology,
record it and keep generation and transcription sequential. WER must report
1,088 evaluated, zero skipped, and the corpus micro-average.

## 7. Forbidden markers and evidence

Scan only the fresh server log:

```bash
grep -Ein \
  'traceback|process .*died|capture_failed|runtime_replay_failed|ACL.*107027|out of memory|eager fallback' \
  "$COSY_LOG" || true
```

Any matching runtime failure is a gate failure unless the full line proves it is
an unrelated benign startup message; include that explanation in `notes`.

Preserve inside the isolated environment:

- fresh server log;
- every `speed_results.json`, `generated.json`, and full-run WER result;
- generated WAVs used by WER;
- before/after/final `npu-smi` snapshots;
- exact commands and version output.

Return only counts, metrics, hashes, minimal traceback fragments, runtime
metadata, and artifact filenames through the handoff template. Do not return
weights, WAVs, full logs, hosts, addresses, credentials, or internal paths.
