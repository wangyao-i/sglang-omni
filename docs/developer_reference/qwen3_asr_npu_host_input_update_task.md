# Qwen3-ASR NPU host-input update: 910C task

This task implements the gate defined in
[the handoff](qwen3_asr_npu_host_input_update_handoff.md). It is deliberately a
single-device mechanism test, not an all-graph or realtime validation run.

## 0. Preconditions and checkout

Use one idle 910C device and an isolated worktree. Do not reuse the running
#2016 validation process or its checkout. The local owner must push
`codex/qwen3-asr-npu-host-input-update-probe` before this task can start.

```bash
export OMNI_REPO=/server/path/to/sglang-omni
export PROBE_TREE=/server/path/to/sglang-omni-host-update-probe
export PROBE_BRANCH=codex/qwen3-asr-npu-host-input-update-probe
export PROBE_CODE_HEAD=a851d1f886c3ef6e72de215000d71c13d2ecf278
export SGLANG_REPO=/server/path/to/sglang
export SGLANG_HEAD=0bcd822377da7b5718e674eaf9c870d349424dd1

git -C "$OMNI_REPO" fetch origin "$PROBE_BRANCH"
git -C "$OMNI_REPO" worktree add --detach "$PROBE_TREE" "origin/$PROBE_BRANCH"
git -C "$PROBE_TREE" merge-base --is-ancestor "$PROBE_CODE_HEAD" HEAD
test -z "$(git -C "$PROBE_TREE" status --porcelain)"
test "$(git -C "$SGLANG_REPO" rev-parse HEAD)" = "$SGLANG_HEAD"
test -z "$(git -C "$SGLANG_REPO" status --porcelain)"
```

If the worktree already exists, do not delete, reset, or clean it. Stop and ask
the server owner to provide another empty path. If the branch is unavailable,
the candidate is not signed out correctly, either worktree is dirty, or the
SGLang HEAD differs, return `blocked` without running tests.

Record the effective environment before the experiment. Do not include private
paths or host identity in the returned summary.

```bash
cd "$PROBE_TREE"
python - <<'PY'
import platform
import torch
import torch_npu

print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("npu_available", torch.npu.is_available())
print("npu_count", torch.npu.device_count())
print("current_device", torch.npu.current_device())
print("device_name", torch.npu.get_device_name(torch.npu.current_device()))
PY
```

Stop if NPU is unavailable, the selected device is not 910C, another workload
occupies it, or the `torch`/`torch_npu` pair is inconsistent with the node's
supported CANN stack.

## 1. Focused contract tests

Run from a fresh shell with the same environment used for the hardware probe:

```bash
cd "$PROBE_TREE"
python -m pytest -q tests/unit_test/platforms/test_device_graph.py
python -m pytest -q tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py \
  -k "npu_replay_updates_window_boundaries_for_bucket_graph or npu_captures_share_one_graph_pool"
git diff --check
```

All selected tests must pass and both repositories must remain clean. At the
first failure, save the complete output in the server-only artifact directory,
classify it as collection/dependency or control-flow failure, and stop.

## 2. Single hardware mechanism gate

Select exactly one 910C device through the server's normal device-isolation
mechanism. Do not add compile, backend, fallback, performance, or graph flags.
Use a fresh Python process and retain its complete output on the server.

```bash
cd "$PROBE_TREE"
python -m pytest -q -s \
  tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py::test_graph_matches_eager_tower
```

The test fixes the model seed and uses a synthetic two-layer encoder. Its first
two cases, `[500]` and `[450]`, produce distinct cumulative window boundaries
inside one token bucket. It asserts eager parity for each case and asserts that
the second case does not add another captured graph.

Pass only when the process exits zero and the following evidence can be read
from the test/assertions and capture log:

- the first case captures a graph;
- the second case reuses the same bucket/registry entry;
- both cases have `max_abs_diff < 3e-2` versus eager;
- update and replay complete without a hang or exception;
- no eager fallback, graph-capture failure, ACL/device error, or process crash.

Do not relax the tolerance or change the input sequence after seeing results.
Do not run the service, more layouts, concurrency, or performance after a pass;
those are separate decisions. On any failure, stop at the first complete
failure and classify it as capture, host-input update, replay/ordering, parity,
dependency/API, memory, or device/ACL.

## 3. Cleanup and return

Synchronize the selected device, confirm it remains healthy, and record the two
repository states. Do not remove the worktree or raw evidence until the local
owner accepts the result.

Return the filled result template from the handoff. Raw logs, environment dumps,
internal paths, addresses, hostnames, and device identifiers stay on the
server. A server-side repair is not part of this gate: if one appears necessary,
return the failure first and wait for a new bounded task.

## 4. Focused-test closure after the `fcaa07b9` mechanism result

The 910C run at `fcaa07b9` exposed one repository test issue: the installed NPU
PyTorch distribution does not provide `torch.xpu.graph`. After the local branch
contains the test-only correction, verify that the new HEAD is a descendant of
the hardware-tested commit and inspect the intervening paths before rerunning:

```bash
cd "$PROBE_TREE"
git fetch origin "$PROBE_BRANCH"
git switch --detach "origin/$PROBE_BRANCH"
git merge-base --is-ancestor fcaa07b9b51975ece8ec65dfde0a5c6d9c61a8c4 HEAD
git diff --name-only fcaa07b9b51975ece8ec65dfde0a5c6d9c61a8c4..HEAD
python -m pytest -q tests/unit_test/platforms/test_device_graph.py
python -m pytest -q tests/unit_test/qwen3_asr/test_encoder_cuda_graph.py \
  -k "npu_replay_updates_window_boundaries_for_bucket_graph or npu_captures_share_one_graph_pool"
git diff --check
```

Only tests and handoff/task documentation may differ from `fcaa07b9`; otherwise
stop and require a new hardware gate. Expected on this NPU distribution is all
available platform tests passing with exactly the unavailable XPU introspection
test skipped, plus both focused encoder tests passing. This closure does not
claim HBM, service, stability, concurrency, all-graph, realtime, or performance
qualification.
