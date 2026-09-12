# Qwen3-ASR Ascend v0.5.19 validation task

This is the active validation task after moving the runtime baseline from
SGLang main to the `v0.5.19` release line. It replaces the historical
main-based classification and retry tasks.

This compile+decode-graph validation produced the first complete startup
failure. The current next action is the single graph-only run in
[`qwen3_asr_ascend_v0519_graph_only_task.md`](qwen3_asr_ascend_v0519_graph_only_task.md).

## Inputs

```bash
export SGLANG_REPO=/server/local/sglang
export OMNI_REPO=/server/local/sglang-omni
export EXPECTED_SGLANG_HEAD=e0011e30fbdb9690f01fa2083d452c93b37bb214
export MODEL_PATH=/server/local/Qwen3-ASR-1.7B
export SMOKE_WAV=/server/local/approved-smoke.wav
export PORT=8000
export EVIDENCE=/server/local/evidence/qwen3-asr-v0519-validation
mkdir -p "${EVIDENCE}"
```

Do not return the values of these variables.

`SGLANG_REPO` and `OMNI_REPO` must be the actual checkout roots imported by
Python. A legacy directory name is not a code identity and is not a Gate 0
failure when the exact HEAD, clean worktree, and imported module path all
agree.

## Exact Identity

| Repository | Branch | Exact runtime head | Base |
|---|---|---|---|
| SGLang | `codex/qwen3-asr-v0519-fused-op` | `e0011e30fbdb9690f01fa2083d452c93b37bb214` | `v0.5.19` commit `0bcd822377da7b5718e674eaf9c870d349424dd1` |
| SGLang-Omni | `codex/qwen3-asr-npu-encoder-stream-v0519` | `5190678c463f6c2b01e4ee0007cf788c3fdc2287` | `6ff46426469a1af2746cef71a9fdcfc09613966d` |

The SGLang candidate is the same three-file fused-op boundary cherry-picked
onto the release-line commit. It is not based on SGLang main.

## Gate 0

- Fetch both branches and check out the detached runtime heads above in the
  actual checkout roots. Do not create or switch to a second checkout merely
  to make a directory name match the logical repository name.
- Require clean tracked worktrees. Preserve any unrelated untracked artifact
  outside the checkout; do not use a dirty tree for this gate.
- Verify the repository identity, installed distribution, and imported module
  path in one fresh process. Run the following commands before pytest or serve:

```bash
cd "${SGLANG_REPO}"
test "$(git rev-parse HEAD)" = "${EXPECTED_SGLANG_HEAD}"
test -z "$(git status --porcelain)"

python - <<'PY'
import json
from importlib.metadata import distribution
from pathlib import Path
from urllib.parse import unquote, urlparse

import sglang

repo = Path.cwd().resolve()
expected_python = repo / "python"
module = Path(sglang.__file__).resolve()
print("sglang version:", sglang.__version__)
print("imported module:", module)
print("expected module root:", expected_python)
assert module.is_relative_to(expected_python), (
    f"wrong SGLang checkout: {module}"
)
assert sglang.__version__.startswith(("0.5.19", "0.5.v19")), (
    f"wrong SGLang release line: {sglang.__version__}"
)

dist = distribution("sglang")
direct_url_raw = dist.read_text("direct_url.json")
assert direct_url_raw, "sglang has no editable direct_url metadata"
direct_url = json.loads(direct_url_raw)
assert direct_url.get("dir_info", {}).get("editable") is True, (
    f"sglang is not an editable checkout: {direct_url}"
)
parsed = urlparse(direct_url["url"])
assert parsed.scheme == "file", f"unexpected editable URL: {direct_url['url']}"
editable_root = Path(unquote(parsed.path)).resolve()
print("editable root:", editable_root)
assert expected_python.is_relative_to(editable_root), (
    f"editable install points to another checkout: {editable_root}"
)
PY

cd "${OMNI_REPO}"
test "$(git rev-parse HEAD)" = \
  "5190678c463f6c2b01e4ee0007cf788c3fdc2287"
test -z "$(git status --porcelain)"

python - <<'PY'
from pathlib import Path

import sglang_omni

repo = Path.cwd().resolve()
module = Path(sglang_omni.__file__).resolve()
print("imported Omni module:", module)
print("expected Omni root:", repo)
assert module.is_relative_to(repo), f"wrong SGLang-Omni checkout: {module}"
PY
```

- If any assertion fails, stop immediately and return `blocked`; do not run
  pytest or the model server in that environment.
- Do not repair this by editing `site-packages` or by relying on whichever
  editable checkout happens to appear first on `sys.path`. Remove or
  deactivate the conflicting install through its owning environment, then
  restart from a fresh process.
- Record Python, CANN, torch, torch_npu, triton-ascend, and `sgl_kernel_npu`
  versions.
- Record `LD_PRELOAD`, `LD_LIBRARY_PATH`, `PYTORCH_NPU_ALLOC_CONF`,
  `STREAMS_PER_DEVICE`, `HCCL_BUFFSIZE`, `ASCEND_LAUNCH_BLOCKING`, and
  `ASCEND_USE_FIA` without changing them.
- Confirm no residual process, port `8000` free, and the selected NPU healthy.

Stop and return `blocked` if the code does not resolve to the requested
release-line checkout, `sglang.__version__` does not belong to the `0.5.19`
release line, or the environment is not identifiable.

## Gate 1: Focused SGLang Test

```bash
cd "${SGLANG_REPO}"
python -m pytest -q \
  test/registered/unit/hardware_backend/npu/test_fused_ops.py \
  >"${EVIDENCE}/test-sglang-fused-ops.log" 2>&1
```

## Gate 2: Focused Omni Test

```bash
cd "${OMNI_REPO}"
python -m pytest -q \
  tests/unit_test/qwen3_asr/test_encoder_service.py \
  >"${EVIDENCE}/test-omni-encoder-service.log" 2>&1
```

## Gate 3: Default Startup And Smoke

Use the default Qwen3-ASR profile. Keep `torch.compile` enabled and keep the
decode graph enabled; do not add a fallback switch.

```bash
cd "${OMNI_REPO}"
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port "${PORT}" \
  >"${EVIDENCE}/server.log" 2>&1
```

After readiness, send one transcription request:

```bash
curl --fail --silent --show-error \
  -X POST "http://127.0.0.1:${PORT}/v1/audio/transcriptions" \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F language=en \
  -F response_format=json \
  -F "file=@${SMOKE_WAV}" \
  >"${EVIDENCE}/smoke-response.json"
```

Require a non-empty transcript and clean normal shutdown. Record the actual
compile/graph mode evidence from the startup log or `/server_info`.

## Stop Conditions

Stop at the first complete failure:

- focused test failure or collection error;
- startup failure, graph capture failure, or Scheduler `AttributeError`;
- request failure, empty transcript, hang, or timeout;
- device, ACL, OOM, allocator, or stream error;
- unexpected eager fallback; or
- non-clean shutdown.

Keep the first complete traceback and the first repository frame that owns it.
Do not edit code or environment variables on the server.

## Non-Goals

- No SGLang main validation.
- No scheduler main-compatibility branch.
- No performance, concurrency-70, realtime, timestamp, or forced-alignment
  work.
- No source, package, or configuration edits on the server.

## Return

Return the exact repository heads and worktree state, imported SGLang module
path, editable root, runtime versions, environment values, focused test counts,
default startup result, smoke result, target-path evidence, fallback/error
counts, device health, cleanup state, and the first complete failure if any.
Keep full logs and response bodies server-local.
