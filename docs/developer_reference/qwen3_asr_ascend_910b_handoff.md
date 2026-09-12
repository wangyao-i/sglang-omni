# Qwen3-ASR Ascend branch-validation handoff

This is the Gate 0 handoff for validating the two branch deltas listed below.
It replaces the older, unrelated handoff state for this restart. Keep raw
logs, model files, audio, and host details on the isolated server.

## Status

The active baseline is now the SGLang `v0.5.19` release line, not SGLang main.
The previous main-based startup failures and the main-only Scheduler
compatibility fix are historical and must not be mixed into this candidate.

The current candidate keeps the minimal three-file fused-op boundary on top of
the `v0.5.19` commit, together with the Omni encoder private-stream change.
Hardware validation on this release-line stack reached a first complete
failure during default startup after the SGLang checkout identity was verified.

## Latest Server Report

The latest report is a valid exact-head run with an unresolved first failure.

- The server worktree uses the legacy directory name
  `sglang_qwen3-asr-decode-graph-diag`, but its actual HEAD is
  `e0011e30fbdb9690f01fa2083d452c93b37bb214`, its worktree is clean, and
  `python -c "import sglang"` resolves to that checkout's `python/sglang`.
- A directory name is not a code identity. The checkout is valid because the
  exact clean commit and the imported module path agree.
- SGLang `test_fused_ops.py` passed `5` tests and Omni
  `test_encoder_service.py` passed `30` tests with `1` skip.
- The run stopped at the first complete startup failure during decode graph
  capture: `PagedAttentionOperation setup failed` followed by
  `Capture cuda graph failed`.
- The failure is valid exact-head evidence, but its owner and root cause remain
  unproven. `PagedAttentionOperation` is an asynchronous operator report, and
  no single-variable experiment has connected the failure to `fused_ops.py`.
- Do not modify `fused_ops.py`, PagedAttention, or the compile boundary based on
  the operator name alone. Classify the first failure before selecting a fix.

## Scope

- Validate that Qwen3's NPU path uses the external fused kernel through an
  opaque custom-op boundary.
- Validate that the Qwen3-ASR encoder uses a private device stream and records
  the consuming default stream.
- Do not run performance, realtime, timestamp, or forced-alignment gates.

## Exact Branch Stack

| Repository | Branch | Exact head | Base |
|---|---|---|---|
| SGLang | `codex/qwen3-asr-v0519-fused-op` | `e0011e30fbdb9690f01fa2083d452c93b37bb214` | `v0.5.19` commit `0bcd822377da7b5718e674eaf9c870d349424dd1` |
| SGLang-Omni | `codex/qwen3-asr-npu-encoder-stream-v0519` | `5190678c463f6c2b01e4ee0007cf788c3fdc2287` | `6ff46426469a1af2746cef71a9fdcfc09613966d` |

The Omni branch tip may advance for handoff, task, and roadmap documents. The
validation task checks out the code commit above, so documentation commits
cannot change the tested runtime.

## Local Evidence

- SGLang: `fused_ops.py` registers the external kernel with
  `register_custom_op_from_extern`; `qwen3.py` imports that wrapper.
- SGLang: the same patch was cherry-picked cleanly onto the `v0.5.19` commit;
  the helper and Qwen3 call site already exist on that release line.
- SGLang: `python -m compileall` and `git diff --check` pass on the
  release-line candidate. The focused pytest cannot execute on this Windows
  host because `sgl_kernel_npu` is unavailable.
- SGLang: an independent stub loaded the implementation, validated output
  metadata, passed the call through to the external-kernel stub, and confirmed
  `torch.library.infer_schema` accepts the function signature.
- Omni: the branch is rebased onto current Omni main and contains focused
  stream-selection, stream-context, and default-stream recording tests.
- Omni: local `compileall`, AST parsing, and an isolated stub of
  `attach_embedding` and `_batch_context` passed. The repository pytest suite
  could not collect because this Windows host has no installed SGLang.

## Historical Main-Baseline Evidence

These results belong to the abandoned SGLang main baseline and are not
current-head evidence.

The first server report observed the requested SGLang main code hash
`85e8933dc3ae4ecd44a1e0ebf4595f0fd9011595`, but labeled its branch
`codex/910c-072-chain23-off`; the local branch is
`codex/qwen3-asr-compile-safe-fused-op`. This is an identity-reporting
discrepancy only. Future runs must use the detached hash.

Observed results:

- SGLang `test_fused_ops.py`: `5 passed`;
- Omni `test_encoder_service.py`: `30 passed, 1 skipped`;
- model smoke: failed before readiness during decode graph capture.

The first complete failure sequence was:

```text
RuntimeError: The Inner error is reported as above. The process exits for this inner
error, and the current working operator name is PagedAttentionOperation.
Exception: Capture cuda graph failed
corrupted size vs. prev_size
```

Do not classify this as a PagedAttention tensor-layout mismatch without a
single-variable comparison. `corrupted size vs. prev_size` is a native heap
allocator signal and may originate in CANN, torch_npu, ATB, graph capture, or
another runtime path. Public Ascend reports also show that required allocator
or dynamic-library settings can be absent from non-interactive launches. The
classification task records those values before changing any of them.
`PagedAttentionOperation` is also an asynchronous operator report, so its name
may not identify the true first fault until a blocking diagnostic confirms it.
The later compile-disabled main run also exposed a separate main-only
`scheduler_stage_metrics` interface drift. Neither result is part of the active
`v0.5.19` candidate.

## Server Facts To Confirm

- Exact SGLang-Omni and SGLang checkout paths and working-tree state.
- Actual Python, CANN, torch, torch_npu, triton-ascend, and `sgl_kernel_npu`
  versions.
- The actual imported `sglang.__file__`, its editable distribution root, and
  whether they resolve to the requested release-line checkout and `0.5.19`
  version line.
- Ascend device model, count, selected device, and idle memory.
- Model path and whether the approved smoke WAV is available.

## Hard Constraints

- Check out the exact heads above; do not substitute branch tips from an older
  run.
- Do not edit `site-packages`, model semantics, graph/compile behavior, or
  configuration policy from the server.
- Do not silently fall back to eager mode.
- Do not restore GraphKey, execution guards, selectors, or unrelated dispatch
  changes.
- Stop at the first complete failure and return its category plus the first
  repository frame.
- Performance, realtime ASR, timestamps, and forced alignment are out of scope.

## Ownership

- SGLang compilation and NPU operator-boundary failures belong in SGLang.
- Encoder stream ownership and embedding handoff failures belong in
  SGLang-Omni.
- CANN, torch_npu, triton-ascend, and sgl_kernel_npu defects must be reported
  with versions first; do not patch them inside either checkout.

## First Task

Run the exact-head release-line validation in
[`qwen3_asr_ascend_v0519_validation_task.md`](qwen3_asr_ascend_v0519_validation_task.md).
Do not run the historical main-based task.

## Return Contract

Return only:

```text
Task status: passed / failed / blocked
Omni branch / observed HEAD / worktree clean:
SGLang branch / observed HEAD / worktree clean:
SGLang imported module / editable root:
Runtime versions:
Hardware model / device count / selected device:
Focused SGLang test:
Focused Omni test:
Model smoke:
Target-path evidence:
Fallback evidence:
Resolved server args:
Default profile verification:
First complete failure and owner repository:
Server-local artifacts retained:
Suggested next bounded task:
```

Do not return full logs, model paths, audio, transcripts, hostnames, IP
addresses, credentials, or proprietary profiler output.

## Roadmap

The live development state is tracked in
[`qwen3_asr_ascend_910b_roadmap.md`](qwen3_asr_ascend_910b_roadmap.md).
