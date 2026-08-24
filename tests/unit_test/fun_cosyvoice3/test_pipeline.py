# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import torch

import sglang_omni.platforms as platforms
from sglang_omni.models.fun_cosyvoice3 import CAPABILITIES
from sglang_omni.models.fun_cosyvoice3.config import FunCosyVoice3PipelineConfig
from sglang_omni.models.fun_cosyvoice3.engine_builder import (
    FunCosyVoice3EngineBuilder,
)
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.stages import (
    create_sglang_tts_engine_executor,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from tests.unit_test.pipeline.helpers import build_compiled_process_topology


def test_fun_cosyvoice3_config_and_registry_contract() -> None:
    config = FunCosyVoice3PipelineConfig(model_path="model")

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert [stage.process for stage in config.stages] == [
        "pipeline",
        "pipeline",
        "pipeline",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.required_speech_reference_count == 1
    assert config.speech_reference_text_excludes_instructions is True
    assert config.gpu_placement == {"tts_engine": 0, "vocoder": 0}
    assert type(config).stage_config_cls("tts_engine").engine_stage
    assert config.process_local_edges() == frozenset({("preprocessing", "tts_engine")})
    assert CAPABILITIES.supports_streaming_vocoder is False

    build_compiled_process_topology(config)
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("FunCosyVoice3SGLangModel")
        is FunCosyVoice3PipelineConfig
    )


def test_fun_cosyvoice3_generation_factory_resolves_host_platform_by_default() -> None:
    signature = inspect.signature(create_sglang_tts_engine_executor)

    assert signature.parameters["device"].default is None


def test_fun_cosyvoice3_npu_forces_eager_generation(monkeypatch) -> None:
    monkeypatch.setattr(platforms.current_platform, "device_type", "npu")
    overrides = {
        **FunCosyVoice3EngineBuilder().generation_defaults(dtype="bfloat16"),
        "disable_cuda_graph": False,
        "disable_prefill_cuda_graph": False,
        "cuda_graph_backend_prefill": "piecewise",
        "cuda_graph_bs_prefill": [1, 2],
        "cuda_graph_max_bs_prefill": 2,
        "cuda_graph_config": {"prefill": {"backend": "piecewise"}},
        "enable_torch_compile": True,
    }

    FunCosyVoice3EngineBuilder().adjust_overrides(overrides)

    assert overrides["disable_cuda_graph"] is True
    assert overrides["disable_prefill_cuda_graph"] is True
    assert overrides["cuda_graph_backend_prefill"] == "disabled"
    assert overrides["enable_torch_compile"] is False
    assert "cuda_graph_config" not in overrides
    assert "cuda_graph_bs_prefill" not in overrides
    assert "cuda_graph_max_bs_prefill" not in overrides


def test_fun_cosyvoice3_cuda_keeps_generation_defaults(monkeypatch) -> None:
    monkeypatch.setattr(platforms.current_platform, "device_type", "cuda")
    overrides = FunCosyVoice3EngineBuilder().generation_defaults(dtype="bfloat16")

    FunCosyVoice3EngineBuilder().adjust_overrides(overrides)

    assert overrides["disable_cuda_graph"] is False


def test_fun_cosyvoice3_state_round_trip_preserves_wire_contract() -> None:
    state = FunCosyVoice3State(
        text="hello",
        language="en",
        instructions="speak brightly",
        ref_text="reference",
        stream=True,
        speed=1.25,
        seed=7,
        generation_kwargs={"max_new_tokens": 32},
        flow_embedding=torch.tensor([[1.0, 2.0]]),
        flow_prompt_speech_token=torch.tensor([[10, 11]], dtype=torch.int32),
        flow_prompt_speech_feat=torch.ones(1, 2, 80),
        audio_codes=torch.tensor([[20], [21]], dtype=torch.long),
    )

    wire = state.to_dict()
    restored = FunCosyVoice3State.from_dict(wire)

    assert wire["flow_embedding"] == [[1.0, 2.0]]
    assert restored.text == state.text
    assert restored.language == state.language
    assert restored.instructions == state.instructions
    assert restored.stream is True
    assert restored.speed == 1.25
    assert restored.seed == 7
    assert restored.generation_kwargs == {"max_new_tokens": 32}
    assert restored.flow_prompt_speech_token == [[10, 11]]
    assert restored.flow_prompt_speech_feat[0][0] == [1.0] * 80
    assert restored.audio_codes == [[20], [21]]
