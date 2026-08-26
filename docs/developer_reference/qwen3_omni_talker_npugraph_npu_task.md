# Qwen3-Omni Talker NPUGraph on Ascend NPU

> **Status: complete.** Batch-one and 16-sequential plus 16-concurrent serving
> gates passed on Ascend 910 A3. The validated runtime matrix, commit mapping,
> failure resolutions, and evidence summary are preserved in the handoff.
>
> The isolated-server execution contract, resolved sampler blockers, required
> cross-repository commits, and result template are recorded in
> `qwen3_omni_talker_npugraph_npu_handoff.md`.

## Dependency and Branch Contract

This is stacked work. The branch `npu-talker-npugraph` is based on
`npu-code2wav-npugraph`, the head branch of
[sglang-omni PR #1710](https://github.com/sgl-project/sglang-omni/pull/1710).
Until #1710 merges, a Talker pull request must target
`wangyao-i/sglang-omni:npu-code2wav-npugraph`, not `main`. After #1710 merges,
rebase this branch onto the new upstream `main` and retarget the pull request.

The stack is intentional:

- #1710 provides the validated Code2Wav NPUGraph backend used by the same
  full-speech pipeline;
- this branch changes and qualifies only the Qwen3-Omni Talker decode graph;
- Thinker decode graphs stay disabled because the current MoE `NonZero` route
  is not safe during NPU graph capture;
- Code2Wav must not be modified while triaging a Talker failure. Its completed
  16-sequential plus 16-concurrent stability gate is the downstream baseline.

Do not merge the historical `br_omni_cosyvoice3_0824` validation branch into
this branch. It has a different history and contains unrelated installation,
configuration, and bring-up commits.

## Goal

Qualify SGLang's existing full decode graph runner for the Qwen3-Omni Talker on
Ascend 910 A3. The Talker does not need a second model-local graph runner:
SGLang already selects `NPUGraphRunner` for device type `npu`. This task owns
the Omni integration boundary, runtime replay evidence, hardware-only fixes,
and the serving stability gate.

## Required Environment

- Ascend 910 A3
- Qwen3-Omni-30B-A3B-Instruct
- CANN, PyTorch, torch_npu, and SGLang versions recorded in the result
- thinker TP=8 on NPU 0-7
- Talker and Code2Wav on NPU 8
- a fresh server process and log for every graph-mode change

Run the standard preflight before every hardware attempt:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SGLANG_OMNI_STARTUP_TIMEOUT=1800
python - <<'PY'
import torch
import torch_npu

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("npu_available", torch.npu.is_available())
print("npu_count", torch.npu.device_count())
print("npugraph", hasattr(torch.npu, "NPUGraph"))
PY
npu-smi info
```

## Phase 1: Batch-One Bring-up

Start with only batch size one captured. Keep the Thinker eager and explicitly
enable the Talker graph:

```bash
sgl-omni serve \
  --model-path /home/weights/Qwen3-Omni-30B-A3B-Instruct \
  --thinker-tp-size 8 \
  --thinker-gpus 0,1,2,3,4,5,6,7 \
  --talker-gpu 8 \
  --code2wav-gpu 8 \
  --stages.thinker.process thinker \
  --stages.image_encoder.runtime.resources.total_gpu_memory_fraction 0.03 \
  --stages.audio_encoder.runtime.resources.total_gpu_memory_fraction 0.03 \
  --stages.thinker.runtime.resources.total_gpu_memory_fraction 0.80 \
  --stages.talker_ar.runtime.resources.total_gpu_memory_fraction 0.20 \
  --stages.code2wav.runtime.resources.total_gpu_memory_fraction 0.02 \
  --stages.code2wav.factory_args.enable_cuda_graph true \
  --thinker-cuda-graph off \
  --talker-cuda-graph on \
  --talker-partial-start off \
  --max-running-requests 1 \
  --cuda-graph-max-bs 1 \
  --host 0.0.0.0 --port 8008 \
  2>&1 | tee /tmp/qwen3_omni_talker_npugraph_b1.log
```

The startup log must report:

```text
sglang_ar_startup stage=talker_ar ... disable_cuda_graph=False ...
disable_decode_cuda_graph=False ... decode_cuda_graph_backend=full
```

SGLang must also complete target decode NPU graph capture. Then send two
identical non-streaming speech requests. Both must return non-empty valid WAV,
and the service log must contain the runtime marker:

```text
Qwen3-Omni talker decode execution active: execution_mode=npu_graph batch_size=1
```

That marker is emitted from SGLang's per-forward `can_run_cuda_graph` result;
startup capture alone is not accepted as evidence of replay.

## Phase 2: Serving Stability Gate

After batch-one passes, raise `max_running_requests` and
`cuda_graph_max_bs` together to four and rerun in a fresh process. Run 16
sequential requests followed by 16 requests at concurrency four.

Pass criteria:

- 32 successful requests and zero failures;
- every response contains non-empty valid WAV audio;
- measured peak client concurrency is at least two;
- Talker startup resolves `decode_cuda_graph_backend=full`;
- at least one Talker runtime marker reports `execution_mode=npu_graph`;
- no Talker runtime marker reports `execution_mode=eager` for a decode batch
  whose size is within the captured 1-4 contract;
- no capture failure, replay failure, graph disable, ACL `107027`, device
  fault, stage crash, or post-run health failure;
- participating NPUs remain healthy after the run;
- repeated-request live tensor memory does not grow monotonically after the
  scheduler drains and the device synchronizes.

Preserve the result JSON, complete server log, generated WAV files, and
before/after `npu-smi` snapshots.

## Failure Policy

Stop at the first complete traceback. Classify it as capture-time operator
compatibility, replay-time address/stream ordering, graph-ineligible batch,
memory budget, eager/graph correctness mismatch, or device failure. Make the
smallest NPU-specific fix and add a CPU/fake-backend regression test. Do not
disable Talker functionality, weaken Code2Wav's validated graph path, edit
site-packages, or claim completion when serving fell back to eager execution.

## Local Verification

```bash
pytest tests/unit_test/qwen3_omni/test_sglang_ar_budget.py -q
pytest tests/unit_test/qwen3_omni/test_talker_token_readback.py -q
pytest tests/unit_test/qwen3_omni/test_talker.py -q
git diff --check
```

Local development environments without the pinned SGLang dependency cannot
collect these tests. Report that as an environment limitation; hardware
qualification remains mandatory and cannot be replaced by unit tests.
