# Qwen3-ASR Ascend branch-validation handoff

This is the Gate 0 handoff for validating the two branch deltas listed below.
It replaces the older, unrelated handoff state for this restart. Keep raw
logs, model files, audio, and host details on the isolated server.

## Status

Ready for server sign-off and the first bounded branch-validation task.

## Scope

- Validate that Qwen3's NPU path uses the external fused kernel through an
  opaque custom-op boundary.
- Validate that the Qwen3-ASR encoder uses a private device stream and records
  the consuming default stream.
- Do not run performance, realtime, timestamp, or forced-alignment gates.

## Exact Branch Stack

| Repository | Branch | Exact head | Base |
|---|---|---|---|
| SGLang | `codex/qwen3-asr-compile-safe-fused-op` | `85e8933dc3ae4ecd44a1e0ebf4595f0fd9011595` | `e91c94805747b60d0b8e8012f3647ae82d4ae862` |
| SGLang-Omni | `codex/qwen3-asr-npu-encoder-stream` | `5190678c463f6c2b01e4ee0007cf788c3fdc2287` | `6ff46426469a1af2746cef71a9fdcfc09613966d` |

The Omni branch tip may advance for handoff, task, and roadmap documents. The
validation task checks out the code commit above, so documentation commits
cannot change the tested runtime.

## Local Evidence

- SGLang: `fused_ops.py` registers the external kernel with
  `register_custom_op_from_extern`; `qwen3.py` imports that wrapper.
- SGLang: `python -m compileall`, `git diff --check`, and the focused pytest
  file were run locally. The pytest command skipped because this host has no
  `sgl_kernel_npu`.
- SGLang: an independent stub loaded the implementation, validated output
  metadata, passed the call through to the external-kernel stub, and confirmed
  `torch.library.infer_schema` accepts the function signature.
- Omni: the branch is rebased onto current Omni main and contains focused
  stream-selection, stream-context, and default-stream recording tests.
- Omni: local `compileall`, AST parsing, and an isolated stub of
  `attach_embedding` and `_batch_context` passed. The repository pytest suite
  could not collect because this Windows host has no installed SGLang.

## Server Facts To Confirm

- Exact SGLang-Omni and SGLang checkout paths and working-tree state.
- Actual Python, CANN, torch, torch_npu, triton-ascend, and `sgl_kernel_npu`
  versions.
- Whether the installed SGLang package resolves to the requested branch.
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

Run the focused branch tests and the minimal model smoke in
[`qwen3_asr_ascend_910b_branch_validation_task.md`](qwen3_asr_ascend_910b_branch_validation_task.md).
The server must return the Gate 0 sign-off fields before running the smoke.

## Return Contract

Return only:

```text
Task status: passed / failed / blocked
Omni branch / observed HEAD / worktree clean:
SGLang branch / observed HEAD / worktree clean:
Runtime versions:
Hardware model / device count / selected device:
Focused SGLang test:
Focused Omni test:
Model smoke:
Target-path evidence:
Fallback evidence:
First complete failure and owner repository:
Server-local artifacts retained:
Suggested next bounded task:
```

Do not return full logs, model paths, audio, transcripts, hostnames, IP
addresses, credentials, or proprietary profiler output.

## Roadmap

The live development state is tracked in
[`qwen3_asr_ascend_910b_roadmap.md`](qwen3_asr_ascend_910b_roadmap.md).
