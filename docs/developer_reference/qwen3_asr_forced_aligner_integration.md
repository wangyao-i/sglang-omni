# Qwen3-ASR forced-aligner integration

Status: planned separate feature; not implemented in the realtime or all-graph
branches.

## Decision

Qwen3-ASR does not emit word or character timestamps. The official
[Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
is a separate non-autoregressive model. Timestamp support must therefore be a
separate alignment stage with its own checkpoint, dependency, resource
budget, and validation contract.

Do not reuse chunk boundaries as word timestamps. Do not infer timestamps from
token positions or decoder output. Do not silently omit timestamps after a
request asks for them.

## Official Contract

The official `qwen-asr` package exposes:

```python
from qwen_asr import Qwen3ForcedAligner

model = Qwen3ForcedAligner.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    device_map="cuda:0",
)

units = model.align(
    audio=audio,
    text="transcript",
    language="English",
)
```

The documented output is a list of aligned units with:

- `text`;
- `start_time`;
- `end_time`.

Official model-card facts:

- the aligner is a separate 0.6B checkpoint;
- it supports speech, not singing or songs with background music;
- it covers 11 languages: Chinese, English, Cantonese, French, German,
  Italian, Japanese, Korean, Portuguese, Russian, and Spanish;
- it supports audio up to 5 minutes;
- streaming inference does not support timestamps.

## Required Interface

The integration should define one explicit alignment boundary, conceptually:

```python
def align(
    *,
    audio: bytes,
    transcript: str,
    language: str,
    audio_offset_s: float = 0.0,
) -> list[AlignedUnit]:
    ...
```

The implementation should return absolute media offsets. If long audio is
chunked, the parent caller must add each chunk's absolute audio offset rather
than treating chunk-local timestamps as global.

The serving response contract must be decided before implementation. A
reasonable API is an opt-in `words` field in `verbose_json`, while `json` and
`text` remain unchanged. That choice requires an OpenAPI schema, docs, and
compatibility tests; it is not part of the current Qwen3-ASR decoder.

## Ownership And Lifecycle

- The aligner checkpoint must be owned by a separate stage or explicitly
  bounded sidecar; it must not be loaded into the ASR engine process by
  accident.
- Device placement, memory budget, startup order, and shutdown must be declared
  in the pipeline configuration.
- The aligner must not share mutable decoder state or graph registries with
  Qwen3-ASR.
- Unsupported language, unsupported audio type, or audio over 5 minutes must
  produce a declared error rather than silent degradation.
- The ASR response path must remain correct when the aligner is absent.

## NPU Status

The official package documents CUDA-oriented examples. NPU support requires a
separate investigation of:

- available `qwen-asr` operators and attention implementation on Ascend;
- checkpoint loading and dtype behavior;
- whether the aligner can share a card with the ASR engine under the declared
  memory budget;
- numerical parity against the official CUDA or CPU output;
- whether chunked alignment preserves absolute offsets and boundary quality.

Until those checks exist, the NPU timestamp feature remains unqualified.

## Non-Goals

- No forced-alignment implementation in the realtime PR.
- No forced-alignment implementation in the all-graph PR.
- No claim that streaming, realtime, or all-graph execution produces word
  timestamps.
- No fake timestamps derived from ASR chunk boundaries.

## Next Bounded Task

Before writing serving code, run one CPU or CUDA design probe with the official
`qwen-asr` package:

1. align one short English clip with a known transcript;
2. align one short Chinese clip with a known transcript;
3. verify unit ordering, monotonicity, and boundary values;
4. serialize and reload the output through a proposed response schema;
5. record dependency and license costs.

Only after that probe should the repository define a new pipeline stage and
choose NPU qualification gates.
