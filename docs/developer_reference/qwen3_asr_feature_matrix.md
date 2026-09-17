# Qwen3-ASR feature matrix

This matrix separates four different questions that must not be collapsed into
one support claim:

1. what the official Qwen3-ASR model and related model cards declare;
2. what SGLang-Omni implements and documents;
3. what currently has CUDA/runtime evidence; and
4. what is implemented or qualified on Ascend NPU.

There is no single official CUDA feature table for Qwen3-ASR. This document is
an evidence-backed synthesis of the official model documentation, the
SGLang-Omni implementation, and repository tests.

## Baselines

- Official Qwen3-ASR model:
  <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
- Official Qwen3-ForcedAligner model:
  <https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B>
- SGLang-Omni source baseline: `upstream/main` contains merged #2084
- SGLang-Omni PR #2160 candidate:
  `302cf932fcf17ce2f1e836b44a06a6a8d9979451`
- SGLang-Omni PR #2016 server-validation base:
  `18c8cfd2eeeb495569426875a2e2bf4114133caf`
- SGLang-Omni all-graph and realtime integration code:
  `63bc33e05de798d754e9953ebfcee9abd08a11ca`, which combines PR #2016,
  the validated PR #2160 bucket-key update path, the NPU graph-only profile,
  and the bounded Qwen3-ASR final-prefix change, then aligns PR #2016 final
  decode with its stability gate
- SGLang dependency: `sglang==0.5.19`; the active runtime baseline is the
  `v0.5.19` release line, not SGLang main
- Ascend validation candidates:
  - SGLang pure `v0.5.19` tag commit `0bcd82237`, with no fused-op patch
  - SGLang-Omni combined all-graph/realtime validation code at `63bc33e0`

The SGLang `e0011e30` fused-op candidate is archived and deferred. It is not
part of the active functional baseline.

The NPU implementation and qualification columns describe those candidates.
The pure-base graph-only smoke is qualified; all other NPU cells remain
provisional until their stated gates complete.

## Matrix

| Feature | Official model declaration | SGLang-Omni behavior | CUDA/runtime evidence | NPU implementation | NPU qualification |
|---|---|---|---|---|---|
| Checkpoints | Qwen3-ASR-1.7B and 0.6B | Generic Qwen3-ASR pipeline accepts the architecture | Cookbook and MLX paths exercise 1.7B and 0.6B | Not checkpoint-specific | Pending; current target is 1.7B |
| Offline transcription | Supported | `/v1/audio/transcriptions` | Cookbook transcription request; ASR CI | Uses the same Omni path | Exact-head graph-only smoke passed on pure `v0.5.19`; full correctness pending |
| SSE token streaming | Streaming inference supported by the model | `stream=true` emits incremental transcript deltas while decoding a complete upload | Cookbook and stream-output tests | Depends on the same LM/compile path | Pending |
| Realtime PCM input | Streaming/offline unified model | `/v1/realtime?intent=transcription`, PCM16 chunks, VAD or explicit commit, replacement partial segments | `test_qwen3_asr_realtime.py` | Existing implementation is device-agnostic and is reused unchanged on the graph-only candidate | Implemented, exact-head NPU qualification pending in `qwen3_asr_ascend_npu_realtime_task.md` |
| Language identification | 30 languages | Automatic detection when `language` is omitted | Result adapter parses the model language prefix | Same model path | Pending |
| Forced language hints | Canonical language names used in the prompt | Accepts 30 canonical names/codes and rejects unsupported hints | `languages.py` and request-builder tests | Same model path | Pending |
| Chinese dialect coverage | 22 Chinese dialects | Covered through `Chinese`/`zh`; dialect names are not independent hint values | Cookbook language section | Same model path | Pending |
| Audio types | Speech, singing voice, songs with BGM | Accepts the corresponding audio through the generic upload path | No model-specific filter; no separate singing/BGM qualification found | Same preprocessing path | Not qualified |
| Long audio | Long-audio transcription supported | Native request limit 1,200s; uploads up to 3,600s via chunking | Cookbook and chunking tests | Same pipeline; graph behavior differs by platform | Pending |
| Chunking | Not a model API detail | 30s default chunks, up to 8 concurrent chunks, minimum tail 0.5s; SSE streaming does not chunk | Cookbook and chunking tests | Same pipeline | Pending |
| Prompt biasing | Model uses its system/context input | `prompt` supplies vocabulary-biasing context; it does not force terms | Request-builder and cookbook behavior | Same model path | Pending |
| Response formats | Not a model-level contract | `json`, `verbose_json`, and `text` | Shared speech-to-text endpoint tests | Same API path | Pending |
| Translation endpoint | Qwen3-ASR is an ASR model | `/v1/audio/translations` returns HTTP 400 | Cookbook and endpoint tests | Same API behavior | Not applicable |
| Timestamps | Separate Qwen3-ForcedAligner-0.6B, up to 5 minutes, 11 languages | No forced-aligner integration; `verbose_json` exposes chunk boundaries only | Cookbook long-audio notes | Planned as a separate alignment stage; see `qwen3_asr_forced_aligner_integration.md` | Not implemented |
| Batch/concurrency | Official vLLM path supports batch inference | Batched stage with `max_running_requests`, pre-LM batching, and chunk concurrency | Config, engine builder, ASR CI | Same scheduler; device execution differs | Pending |
| BF16/FP16 | BF16 checkpoints; FlashAttention requires BF16/FP16 | `auto` follows checkpoint dtype; FP16 can be forced | Cookbook dtype notes | Same dtype policy | Pending |
| Encoder graph | Not a model contract | Optional platform-owned encoder layer-stack graph | CUDA graph tests | PR #2160 adds lazy token-bucket NPU graphs, replay-time host-boundary updates, a shared graph pool, and fail-fast capture/update errors | Current-head all-graph qualification pending |
| Prefill/decode graph | Not a model contract | Prefill uses the breakable backend; decode graph is enabled by default | Engine builder and graph tests | The all-graph candidate pins `breakable` prefill and full decode with the NPU encoder graph active | Encoder, prefill, and decode graph qualification is pending as one exact-head all-graph task |
| Torch compile | Not a model contract | Enabled by default in the CUDA performance profile with `torch_compile_max_bs=2`; the stage default is disabled | Engine builder/default config | The NPU profile forces `enable_torch_compile=false` even when typed pipeline defaults request compilation | Compile is not part of the NPU graph-only support claim |
| NPU encoder stream isolation | Not applicable | `encoder_service.py` uses a private device stream and records the default consuming stream | Focused unit tests | Merged in #2084 through `886ced95` | Merged code is the active baseline; the all-graph task re-attests the integrated path on an exact head |
| NPU fused-op compile boundary | Not applicable | Qwen3 imports the external kernel through an opaque custom op | Focused SGLang NPU tests | Archived on `e0011e30`; pure `v0.5.19` has no diff against the tag | Not required by the pure-base graph-only path; deferred until compile is proven necessary |

## Current Gaps

- The 22 Chinese dialects are model coverage, not 22 selectable language hints
  in the serving API.
- Singing voice and BGM transcription are officially declared model
  capabilities, but this repository has no dedicated qualification set for
  those audio classes.
- Word/character timestamps require the separate forced-aligner model and are
  not part of the Qwen3-ASR serving path described here.
- Realtime has no NPU-specific production adaptation, but it still requires
  exact-head hardware evidence before it is marked qualified.
- The all-graph candidate changed after earlier hardware runs; the previous
  results must not be treated as current-head qualification.
- The NPU implementation columns do not imply qualification. The current
  repository claim is limited to branch implementation and focused tests.
- CUDA performance numbers are intentionally excluded. This matrix records
  capabilities and execution paths, not throughput or latency claims.

## Evidence Sources

- SGLang-Omni cookbook: [`docs/cookbook/qwen3_asr.md`](../cookbook/qwen3_asr.md)
- Default pipeline configuration:
  [`config.py`](../../sglang_omni/models/qwen3_asr/config.py)
- Engine construction and runtime defaults:
  [`engine_builder.py`](../../sglang_omni/models/qwen3_asr/engine_builder.py)
- Language normalization:
  [`languages.py`](../../sglang_omni/models/qwen3_asr/languages.py)
- Request construction and response parsing:
  [`request_builders.py`](../../sglang_omni/models/qwen3_asr/request_builders.py)
- Realtime streaming strategy:
  [`streaming.py`](../../sglang_omni/models/qwen3_asr/streaming.py)
- Realtime integration test:
  [`test_qwen3_asr_realtime.py`](../../tests/test_model/test_qwen3_asr_realtime.py)
- ASR CI thresholds:
  [`asr_ci_config.py`](../../tests/test_model/asr_ci_config.py)

Update this matrix only from repository evidence or official model
documentation. Mark a capability as qualified only when the exact runtime
matrix and hardware evidence are recorded.
