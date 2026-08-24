# Qwen3-Omni Code2Wav NPUGraph NPU Server Handoff

## Objective and Scope

Validate the new Qwen3-Omni Code2Wav NPUGraph path on real Ascend hardware,
fix hardware-only compatibility issues, and enable the NPU default only after
the hardware gate passes. CI and Docker are intentionally out of scope because
the community currently has no NPU CI capacity. Commit each independent stage
locally; do not push unless the user requests it.

## Starting Point

Work from branch `br_omni_cosyvoice3_0824` with these commits present:

- `7ee052c4 [NPU] Add Code2Wav NPUGraph runner`
- `ce9fe507 [NPU] Add Code2Wav NPUGraph validation probe`

The implementation is in
`sglang_omni/models/qwen3_omni/components/code2wav_cuda_graph.py` and scheduler
selection is in `code2wav_scheduler.py`. The public `enable_cuda_graph` name is
retained for configuration compatibility; on NPU it selects
`Code2WavNpuGraphRunner` and reports `execution_mode: npu_graph`.

Do not change `NPUOmniPlatform.enable_code2wav_graph()` yet. It deliberately
returns `False` so graph capture remains explicit opt-in during bring-up.

## Preflight

Run from the repository root and record the complete output:

```bash
git status --short
git log -3 --oneline
python - <<'PY'
import torch, torch_npu
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("available", torch.npu.is_available())
print("count", torch.npu.device_count())
print("NPUGraph", hasattr(torch.npu, "NPUGraph"))
print("graph", hasattr(torch.npu, "graph"))
print("graph_pool_handle", hasattr(torch.npu, "graph_pool_handle"))
PY
npu-smi info
```

Stop if the worktree contains unrelated changes that overlap the files above.
Preserve all user changes.

## Hardware Gate

Replace the model path and run in a fresh process:

```bash
python scripts/npu/validate_qwen3_omni_code2wav_graph.py \
  --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --device npu:0 --memory-fraction 0.02 --iterations 20 \
  2>&1 | tee /tmp/code2wav_npugraph_validation.log
```

Mandatory pass conditions:

- top-level `status` is `pass`;
- all `T={10,20,30,35}` entries report `npu_graph` and `exact_match: true`;
- runner stats show `enabled: true`, four published graphs, and no disable reason;
- no device fault or replay failure occurs.

Record eager latency, graph latency, speedup, memory statistics, NPU model,
driver/CANN, PyTorch, and torch_npu versions. Correctness is mandatory;
speedup greater than 1 is desirable but is not a correctness gate.

## Failure Triage

If capture is disabled, preserve the complete JSON and first exception. Use
`ASCEND_LAUNCH_BLOCKING=1` in a new process to localize asynchronous operator
failures. For memory-budget failures, first confirm the card is otherwise idle,
then retry `--memory-fraction 0.03` and `0.05`; do not weaken the runner's
rollback or equivalence checks. For missing graph APIs, report the exact
torch_npu version and API inventory instead of adding compatibility guesses.
After a device fault, terminate only the failed validation process and verify
card health before retrying.

Classify the failure as one of: missing API, unsupported captured operator,
dynamic allocation/shape during capture, memory budget, eager/graph mismatch,
or replay failure. Make the smallest backend-specific fix and add a fake-backend
unit test before committing it.

## Verification and Commit Policy

After every fix run:

```bash
pytest tests/unit_test/qwen3_omni/test_code2wav_cuda_graph.py -q
pytest tests/unit_test/qwen3_omni/test_code2wav_batching.py -q
git diff --check
```

Then rerun the hardware gate in a fresh process. Commit each passing stage with
a message such as `[NPU] Fix Code2Wav NPUGraph <issue>`. Do not mix formatting,
CI, Docker, CosyVoice, or unrelated platform work into these commits.

## Enabling the Default

Only after the hardware gate and one eager full-speech request both pass, test
the full pipeline with this YAML override:

```yaml
runtime_overrides:
  code2wav:
    enable_cuda_graph: true
```

Confirm audio completion, absence of graph fallback/replay errors, and at least
two consecutive requests in the same worker. Then change only
`sglang_omni/platforms/npu.py` so `enable_code2wav_graph()` returns `True`, add
or update the platform unit test, rerun the suites above, and create a separate
commit: `[NPU] Enable Code2Wav NPUGraph by default`.

## Report Back

Return the validation JSON or full log, exact environment versions, commits
created, test results, and remaining blockers. Never report the adaptation as
complete if execution fell back to eager mode.
