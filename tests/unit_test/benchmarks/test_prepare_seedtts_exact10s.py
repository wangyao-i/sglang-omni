# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.manifest.exact10s import (
    TARGET_FRAMES,
    fingerprint_manifest,
    load_exact10s_manifest,
)
from benchmarks.manifest.prepare_seedtts_exact10s import build_exact10s_corpus


def _source(tmp_path: Path, sample_id: str, frames: int, fill: int) -> SampleInput:
    path = tmp_path / f"{sample_id}.wav"
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(bytes([fill]) * frames * 2)
    return SampleInput(
        sample_id=sample_id,
        ref_text=f"text {sample_id}",
        ref_audio=str(path),
        target_text="unused",
    )


def test_builder_creates_exact_distinct_manifest_and_provenance(
    tmp_path: Path,
) -> None:
    sources = [
        _source(tmp_path, f"s{index}", 70_000, index + 1)
        for index in range(3)
    ]
    output = tmp_path / "derived"
    provenance = build_exact10s_corpus(
        sources,
        output,
        total_clips=2,
        warmup_clips=1,
    )
    samples = load_exact10s_manifest(output / "manifest.jsonl")
    info = fingerprint_manifest(samples, min_distinct_count=2)

    assert len(samples) == 2
    assert all(sample.num_samples == TARGET_FRAMES for sample in samples)
    assert info.sha256 == provenance["manifest_sha256"]
    assert provenance["warmup_count"] == 1
    assert provenance["measured_count"] == 1
    assert provenance["minimum_speech_fraction"] == 0.8
    stored = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert stored["source_membership"][0]["source_sample_ids"] == ["s0", "s1"]


def test_builder_is_deterministic_across_output_directories(tmp_path: Path) -> None:
    sources = [
        _source(tmp_path, f"s{index}", 70_000, index + 1)
        for index in range(3)
    ]
    first = build_exact10s_corpus(
        list(reversed(sources)),
        tmp_path / "first",
        total_clips=2,
        warmup_clips=1,
    )
    second = build_exact10s_corpus(
        sources,
        tmp_path / "second",
        total_clips=2,
        warmup_clips=1,
    )
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_builder_refuses_existing_output(tmp_path: Path) -> None:
    sources = [
        _source(tmp_path, f"s{index}", 70_000, index + 1)
        for index in range(3)
    ]
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_exact10s_corpus(
            sources,
            output,
            total_clips=2,
            warmup_clips=1,
        )


def test_builder_rejects_low_speech_occupancy(tmp_path: Path) -> None:
    sources = [
        _source(tmp_path, f"s{index}", 10_000, index + 1)
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="speech occupancy"):
        build_exact10s_corpus(
            sources,
            tmp_path / "derived",
            total_clips=2,
            warmup_clips=1,
        )


def test_builder_excludes_overlong_sources_before_count_gate(
    tmp_path: Path,
) -> None:
    sources = [
        _source(tmp_path, "short", 140_000, 1),
        _source(tmp_path, "long", TARGET_FRAMES + 1, 2),
    ]
    with pytest.raises(ValueError, match="usable"):
        build_exact10s_corpus(
            sources,
            tmp_path / "derived",
            total_clips=2,
            warmup_clips=1,
        )
