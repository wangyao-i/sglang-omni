# Qwen3-ASR Ascend 910B: first validation task

Run this task on the isolated server before applying a Qwen3-ASR NPU code
change. Its purpose is to establish the environment, unchanged-main failure,
and eager functional baseline. It is not a performance qualification.

Read the [hardware handoff](qwen3_asr_ascend_910b_handoff.md) first. Keep full
logs in a server-local evidence directory and return only the redacted summary
template at the end.

## Inputs and safety

The operator supplies these values locally. Do not paste their values into the
returned report:

```bash
export QWEN3_ASR_REPO=/server/local/sglang-omni
export QWEN3_ASR_MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export QWEN3_ASR_SMOKE_WAV=/server/local/approved-smoke.wav
export QWEN3_ASR_PORT=8000
export QWEN3_ASR_EVIDENCE=/server/local/evidence/qwen3-asr-910b-000
```

Use an approved speech clip between 1 s and 30 s for the smoke request. Do not
use silence, private speech that may not be retained, or a repeated file for
later performance measurements.

## 1. Freeze the baseline

```bash
cd "${QWEN3_ASR_REPO}"
mkdir -p "${QWEN3_ASR_EVIDENCE}"
git status --short
git fetch --all --tags --prune
git switch --detach e7d876b28326c55d777ae62e1c3650b816785d8c
git status --short
git rev-parse HEAD
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/npu/install_npu.sh --check \
  >"${QWEN3_ASR_EVIDENCE}/precheck.log" 2>&1
```

Stop if the checkout is dirty, the commit differs, or the precheck fails.
Classify environment failures separately from model failures. Record only
package versions and sanitized device model/count in the returned summary.

Install the editable package only if the precheck passed:

```bash
bash scripts/npu/install_npu.sh \
  >"${QWEN3_ASR_EVIDENCE}/install.log" 2>&1
python -m pip check \
  >"${QWEN3_ASR_EVIDENCE}/pip-check.log" 2>&1
```

## 2. Run static and unit gates

```bash
python -m pytest -q tests/unit_test/npu/test_npu_install_script.py \
  >"${QWEN3_ASR_EVIDENCE}/test-install-npu.log" 2>&1
python -m pytest -q tests/unit_test/qwen3_asr \
  >"${QWEN3_ASR_EVIDENCE}/test-qwen3-asr.log" 2>&1
```

If the second test path is absent on the exact checkout, record that fact and
run `find tests/unit_test -maxdepth 2 -type f -iname '*qwen3*asr*'` to select
the repository's actual focused test files. Do not substitute the full suite
without recording what changed.

## 3. Preserve the unchanged-default startup result

Start a new process with the repository defaults. This is expected to expose
the CUDA-only encoder graph boundary; preserve the entire local log even if the
failure appears obvious.

```bash
cd "${QWEN3_ASR_REPO}"
sgl-omni serve \
  --model-path "${QWEN3_ASR_MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${QWEN3_ASR_PORT}" \
  >"${QWEN3_ASR_EVIDENCE}/default-startup.log" 2>&1
```

Stop the process after a terminal error or after the health endpoint is ready.
Record the first complete exception, including the first repository frame and
the lowest-level exception category. Do not return model paths or the raw log.

## 4. Start an explicit eager diagnostic profile

Use a new process. These switches are diagnostic controls, not a supported
910B profile: they disable both generation and encoder graph capture and
`torch.compile`, and admit only one request.

```bash
sgl-omni serve \
  --model-path "${QWEN3_ASR_MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --asr.factory.enable_encoder_cuda_graph false \
  --asr.engine.disable_cuda_graph true \
  --asr.engine.enable_torch_compile false \
  --asr.engine.max_running_requests 1 \
  --port "${QWEN3_ASR_PORT}" \
  >"${QWEN3_ASR_EVIDENCE}/eager-startup.log" 2>&1
```

After the server is ready, run one request, then two concurrent requests, then
ten sequential requests. Use fresh request IDs and keep responses only in the
server-local evidence directory. Abort immediately on non-finite tensors,
empty output, device OOM, device reset, or a changed transcript for identical
deterministic inputs.

```bash
curl --fail --silent --show-error \
  -X POST "http://127.0.0.1:${QWEN3_ASR_PORT}/v1/audio/transcriptions" \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F language=en \
  -F response_format=json \
  -F "file=@${QWEN3_ASR_SMOKE_WAV}" \
  >"${QWEN3_ASR_EVIDENCE}/smoke-response.json"
```

Use the same request shape in a short server-local script for the two-request
and sequential gates. Record count, status, latency summary, normalized-output
hash, and peak NPU memory; do not copy audio or transcript text into the
repository or returned report.

## 5. Inspect health and shutdown

Before stopping the eager process, capture a sanitized device-memory summary
and search the full local logs for `ERROR`, `Traceback`, `OOM`, `NaN`, device
reset, graph fallback, and unsupported-operator messages. Stop the server
normally, confirm no worker remains, then start it once more and repeat the
single request. A restart failure is a stability failure.

Do not proceed to concurrency or performance if any earlier gate fails.

## Return template

Return this Markdown block only; keep attachments on the isolated server:

```text
Run: 910B-000
Server commit: <sha>
Local equivalent: e7d876b2
Working tree clean: yes/no
Hardware: Ascend 910B, device count <n>, sanitized memory capacity <value>
Stack: Python <v>; CANN <v>; torch <v>; torch_npu <v>; SGLang <v>;
       triton-ascend <v>; sgl-kernel-npu <v>; sglang-omni <sha>
NPU precheck: pass/fail
Focused tests: <passed>/<failed>/<skipped>; exact test selectors <list>
Default startup: ready/fail
Default first failure: <category; first repository frame; operator/kernel name>
Eager startup: ready/fail
Batch-1: pass/fail; latency <aggregate>; normalized output hash <hash>
Two concurrent: pass/fail; latency <aggregate>; normalized output hashes <hashes>
Ten sequential: pass/fail; latency <aggregate>; output stable yes/no
Peak NPU memory: <aggregate>
Restart: pass/fail
Unexpected fallback: none/<sanitized description>
Next smallest failing gate: <one sentence>
```
