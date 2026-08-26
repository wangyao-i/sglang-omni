# Qwen3-Omni Talker NPUGraph NPU Handoff

## Completion Status

> **Status: complete.** The batch-one and 16-sequential plus 16-concurrent
> stability gates passed on Ascend 910 A3. Talker replayed through SGLang's full
> decode NPUGraph runner, Thinker decode graphs remained disabled, and
> Code2Wav NPUGraph remained enabled.

Validated runtime matrix:

- Ascend 910 A3 with CANN 9.0.1;
- PyTorch 2.10.0+cpu and torch_npu 2.10.0.post2;
- SGLang v0.5.16 base `fdebc938f7`;
- Thinker TP=8 on NPU 0-7;
- Talker and Code2Wav colocated on NPU 8.

The isolated server report supplied commit messages but not its three commit
hashes. The verified changes were reconstructed locally as:

| Repository | Server change | Local equivalent |
|---|---|---|
| sglang-omni | `[NPU] Route Talker sampling through Ascend backend on NPU` | `71bd777d` |
| sglang | `[NPU] Make Ascend top-k/top-p sampling guard capture-safe` | `f46251ce8` |
| sglang | `[NPU] Cast top_ps to logits dtype for fused npu_top_k_top_p` | `ec43c1f20` |

The two SGLang equivalents are on branch
`npu/talker-npugraph-v0.5.16` in the dedicated v0.5.16 worktree. Add the server
hashes to this table if they become available; their absence does not weaken
the supplied hardware evidence or the locally reviewable equivalents.

Completion evidence:

- focused Omni suite: 17 passed on the isolated server;
- wider Omni suite: 91 passed, 6 skipped;
- focused SGLang capture-safety suite: 4 passed;
- batch one: two valid WAV responses, Talker
  `execution_mode=npu_graph batch_size=1`, and Code2Wav NPUGraph replay;
- stability: 32/32 valid WAV responses, zero request failures, peak concurrency
  four, and zero in-contract Talker eager markers;
- zero `aclnnNonzeroV2`, ACL `107027`, `161002`/`EZ1001`, capture failure, or
  replay failure markers;
- Code2Wav reported zero replay failures; only its expected final-tail
  `reason=ineligible` fallback was observed;
- the post-run health endpoint and participating NPUs remained healthy;
- live tensor memory stabilized after scheduler drain and device synchronize.

Use `/v1/chat/completions` with `modalities=["audio"]` and WAV audio options for
this qualification. `/v1/audio/speech` currently has an unrelated text-to-chat
message wrapping failure and is not part of this gate.

## Purpose

This file is the execution contract between the public-repository Agent and the
Agent running inside the isolated Ascend environment. The server Agent applies
or reconstructs the requested commits, runs the hardware gates, and returns the
structured summary at the end of this document. Do not transfer model weights,
generated audio, complete private logs, host names, addresses, or credentials
out of the isolated environment.

The detailed serving procedure and pass criteria live in
`qwen3_omni_talker_npugraph_npu_task.md`. This handoff records the exact branch
stack, cross-repository dependency, current blockers, execution order, and the
evidence that must be returned.

## Branch and Dependency Contract

The Omni work is intentionally stacked:

```text
sglang-omni upstream/main
  -> npu-code2wav-npugraph
    -> npu-talker-npugraph
```

Use the `npu-talker-npugraph` branch in `sglang-omni`. It currently contains:

- `de4d2dd9 [NPU] Log talker decode graph startup settings`;
- `6e421116 [NPU] Log Qwen3-Omni talker graph replay mode`;
- `e752cef0 docs: add Qwen3-Omni Talker NPUGraph NPU gate`.

The runtime dependency is a separate SGLang checkout based on `v0.5.16`. Do
not copy the framework patch into `sglang-omni` or edit an installed
site-package. Record the exact SGLang HEAD before testing; `v0.5.16` points to
`fdebc93`, but an internal checkout may contain additional vendor commits and
must be patched from its actual HEAD.

Until the Code2Wav branch merges, a Talker pull request must target
`npu-code2wav-npugraph`, not `main`. Thinker decode graphs remain disabled and
the already-qualified Code2Wav NPUGraph path must remain enabled.

## Current Hardware Findings

The first Talker capture failure has been identified and fixed in the isolated
environment: `stages.py` selected `sampling_backend="pytorch"` unconditionally.
On NPU, the PyTorch sampling path lowers boolean advanced indexing to
`aclnnNonzeroV2`, which is not capture-safe. The server-side fix selects the
`ascend` backend on NPU and preserves `pytorch` on other platforms. Before
hardware qualification, reconstruct that change on the Omni branch with its
two new regression tests and the corrected environment-dependent assertion.

The second blocker was in SGLang `v0.5.16`:

```text
python/sglang/srt/layers/sampler.py
top_k_top_p_min_p_sampling_from_logits_ascend()
```

Its fused-kernel guard calls `torch.all()` on the device `top_ks` tensor. That
causes a device-to-host scalar synchronization during capture and fails with
ACL `107027` (`Not allow to synchronize captured-stream`).

After that guard was made capture-safe, the eligible eager fused path exposed
a third issue: `torch_npu.npu_top_k_top_p` requires `top_ps` to have the same
dtype as `logits`. `SamplingBatchInfo` supplied FP32 `top_ps` with BF16 Talker
logits, causing `AclNN_Parameter_Error` `161002`/`EZ1001`. Casting `top_ps` to
`logits.dtype` immediately before the fused call resolved it.

## Phase 1: Reconstruct and Verify the Omni Fix

In the Omni checkout:

1. Confirm the branch and preserve any unrelated local work.
2. Change Talker construction so NPU uses `sampling_backend="ascend"` while
   every other platform continues to use `sampling_backend="pytorch"`.
3. Add tests for both platform outcomes and update the existing test that
   incorrectly assumes one backend on every platform.
4. Keep this as an Omni-only commit; report its server commit hash.

Required local checks:

```bash
pytest tests/unit_test/qwen3_omni/test_sglang_ar_budget.py -q
git diff --check
```

The previously reported isolated-environment baseline was 17 focused tests and
91 passed with 6 skipped for the wider local suite. If the counts differ,
report the collected test set and reason; do not silently weaken assertions.

## Phase 2: Patch SGLang v0.5.16

Create a separate SGLang branch from the server's actual HEAD. In
`python/sglang/srt/layers/sampler.py`, make the fused Ascend guard capture-safe.
The conservative fix is:

- query NPU stream-capture state without reading a device tensor;
- outside capture, retain the current fused `npu_top_k_top_p` eligibility check;
- during capture, do not execute `torch.all()`, `.item()`, or
  `bool(device_tensor)`;
- during capture, use the existing tensor-only fallback so replay remains valid
  when request `top_k` values change;
- preserve the `v0.5.16` fused call argument order
  `torch_npu.npu_top_k_top_p(logits, top_ps, top_ks)`.

The intended control flow is:

```python
use_fused_top_k_top_p = False
is_capturing = bool(torch.npu.is_current_stream_capturing())

if hasattr(torch_npu, "npu_top_k_top_p") and not is_capturing:
    use_fused_top_k_top_p = bool(
        torch.all((top_ks >= 1) & (top_ks <= 1024))
    )

if use_fused_top_k_top_p:
    # Existing v0.5.16 fused implementation.
    ...
else:
    # Existing v0.5.16 tensor-only fallback.
    ...
```

Prefer a small helper for the capture-state query so a unit test can mock it.
Add regression coverage proving that capture mode neither calls `torch.all`
nor enters `npu_top_k_top_p`; retain coverage for the eligible eager fused path
and the out-of-range eager fallback. Commit this change only in SGLang and
report its hash.

Commit the capture-safe guard first. In a second commit, cast `top_ps` to
`logits.dtype` inside the `if use_fused_top_k_top_p:` block immediately before
calling `npu_top_k_top_p`. Strengthen the eligible eager regression test with
BF16 logits and FP32 `top_ps`, and assert that the fused stub receives BF16
`top_ps`. Keep the v0.5.16 argument order unchanged.

Before the full service run:

```bash
python -m py_compile python/sglang/srt/layers/sampler.py
git diff --check
```

If the installed torch_npu version does not expose
`torch.npu.is_current_stream_capturing()`, stop and report the available stream
capture API. Do not replace it with a device-tensor probe.

## Phase 3: Batch-One Hardware Gate

Use a fresh worker and log. Follow Phase 1 of
`qwen3_omni_talker_npugraph_npu_task.md` with Thinker eager, Talker graph on,
Code2Wav graph on, `max_running_requests=1`, and `cuda_graph_max_bs=1`.

The run passes only when:

- Talker startup reports decode graph backend `full`;
- graph capture completes without `aclnnNonzeroV2`, host synchronization, or
  ACL `107027`;
- two consecutive speech requests return non-empty valid WAV;
- the same Talker worker reports
  `execution_mode=npu_graph batch_size=1`;
- Code2Wav continues to replay through `execution_mode=npu_graph`;
- the health endpoint still responds and all participating NPUs remain healthy.

Stop on the first complete failure. Preserve the first traceback and classify
it before modifying code. Do not accept eager Talker fallback as a pass.

## Phase 4: Stability Gate

Only after batch one passes, start another fresh worker with the Talker graph
contract raised to batch sizes 1 through 4. Run 16 sequential requests followed
by 16 requests at concurrency four, as specified in the task document.

The final result must contain 32 valid WAV responses, zero request failures,
peak client concurrency of at least two, at least one Talker replay marker, no
in-contract eager marker, no capture/replay/ACL/device failure, a healthy
post-run endpoint, healthy cards, and no monotonic live-tensor memory growth
after the scheduler drains and the device synchronizes.

## Failure and Commit Policy

Keep Omni integration changes and SGLang framework changes in separate
commits. For every new hardware failure:

1. retain the first complete traceback and the effective launch arguments;
2. identify whether it occurred during capture, replay, sampling, stage
   handoff, Code2Wav, or device recovery;
3. make the smallest NPU-specific correction in the repository that owns the
   failing code;
4. add a CPU/fake-backend unit regression when possible;
5. rerun the focused tests and restart the hardware gate from a fresh process.

Do not disable Talker graphs, weaken WAV or replay evidence, edit site-packages,
merge the historical internal bring-up branch, or include unrelated cleanup.

## Result to Return

Return a redacted summary using this exact structure:

```text
Omni branch:
Omni base/head before patch:
Omni sampling-backend fix commit:
SGLang version and base/head before patch:
SGLang capture-safe sampler commit:
CANN / torch / torch_npu versions:
Hardware topology:

Focused tests:
Wider tests:
git diff --check:

Batch-one startup graph backend:
Batch-one capture result:
Talker replay markers:
Code2Wav replay markers:
Speech requests passed/failed:
WAV validation:
Health after run:
NPU health after run:

Stability requests passed/failed:
Peak concurrency:
Talker graph/eager marker counts:
Capture/replay failure counts:
ACL 107027 count:
Memory observation:

First failure traceback, if any:
Failure classification:
Files changed:
Commits created:
Working tree status:
```

This handoff is complete. Preserve the procedure for future CANN, PyTorch, or
torch_npu qualification and rerun both gates when the runtime matrix changes.
