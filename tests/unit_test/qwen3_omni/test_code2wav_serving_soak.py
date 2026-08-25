# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import wave

import pytest

from scripts.npu.validate_qwen3_omni_code2wav_serving import (
    RequestResult,
    _decode_wav,
    _peak_concurrency,
    _percentile,
    _phase_summary,
    _scan_server_log,
)


def _wav_base64(*, frames: int = 160, sample_rate: int = 16000) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x01\x00" * frames)
    return base64.b64encode(buffer.getvalue()).decode()


def _result(index: int, started: float, ended: float) -> RequestResult:
    return RequestResult(
        phase="concurrent",
        index=index,
        prompt_index=0,
        started_s=started,
        ended_s=ended,
        latency_s=ended - started,
        status="pass",
        audio_bytes=2048,
        audio_seconds=1.0,
    )


def test_decode_wav_validates_and_reports_duration() -> None:
    audio, duration = _decode_wav(_wav_base64(), min_audio_bytes=100)

    assert len(audio) > 100
    assert duration == pytest.approx(0.01)


@pytest.mark.parametrize(
    "payload", ["not base64", base64.b64encode(b"x" * 200).decode()]
)
def test_decode_wav_rejects_invalid_audio(payload: str) -> None:
    with pytest.raises(ValueError):
        _decode_wav(payload, min_audio_bytes=100)


def test_phase_summary_reports_overlap_and_interpolated_latency() -> None:
    results = [_result(0, 0.0, 2.0), _result(1, 1.0, 4.0), _result(2, 4.0, 5.0)]

    summary = _phase_summary(results)

    assert _peak_concurrency(results) == 2
    assert _percentile([1.0, 2.0, 3.0], 0.95) == pytest.approx(2.9)
    assert summary["passed"] == 3
    assert summary["peak_concurrency"] == 2
    assert summary["latency_s"]["p50"] == 2.0
    assert summary["audio_bytes"] == 6144


def test_server_log_scan_accepts_clean_replay_and_runtime_stats(tmp_path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        'Code2Wav NPU graph startup stats={"disable_reason":null,"enabled":true}\n'
        "Code2Wav NPU graph replay active: execution_mode=npu_graph\n"
        "Code2Wav NPU graph runtime stats: graph_replays=100 "
        "replay_failures=0 fallback_counts={}\n"
    )

    report = _scan_server_log(log_path, require_runtime_stats=True)

    assert report["status"] == "pass"
    assert report["healthy_startup"] is True
    assert report["replay_markers"] == 1
    assert report["runtime_stats_markers"] == 1
    assert report["failure_markers"] == {}


def test_server_log_scan_rejects_fallback_and_replay_failure(tmp_path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        'Code2Wav NPU graph startup stats={"disable_reason":null,"enabled":true}\n'
        "Code2Wav NPU graph replay active: execution_mode=npu_graph\n"
        "Code2Wav NPU graph eager fallback: reason=key_miss\n"
        "runtime_replay_failed: RuntimeError\n"
    )

    report = _scan_server_log(log_path, require_runtime_stats=True)

    assert report["status"] == "fail"
    assert report["failure_markers"] == {
        "runtime_replay_failed": 1,
    }
    assert report["unexpected_eager_fallback_markers"] == 1
    assert "missing periodic graph runtime stats marker" in report["issues"]


def test_server_log_scan_accepts_bounded_final_window_fallback(tmp_path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        'Code2Wav NPU graph startup stats={"disable_reason":null,"enabled":true}\n'
        "Code2Wav NPU graph replay active: execution_mode=npu_graph\n"
        "Code2Wav NPU graph eager fallback: reason=ineligible key=None\n"
        "Code2Wav NPU graph runtime stats: graph_replays=100 "
        "replay_failures=0 fallback_counts={'ineligible': 26}\n"
    )

    report = _scan_server_log(
        log_path,
        require_runtime_stats=True,
        max_final_ineligible=32,
    )

    assert report["status"] == "pass"
    assert report["allowed_final_ineligible_fallbacks"] == 26
    assert report["allowed_final_ineligible_markers"] == 1
    assert report["unexpected_fallback_counts"] == {}


def test_server_log_scan_rejects_ineligible_count_above_request_budget(
    tmp_path,
) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        'Code2Wav NPU graph startup stats={"disable_reason":null,"enabled":true}\n'
        "Code2Wav NPU graph replay active: execution_mode=npu_graph\n"
        "Code2Wav NPU graph eager fallback: reason=ineligible key=None\n"
        "Code2Wav NPU graph runtime stats: graph_replays=100 "
        "replay_failures=0 fallback_counts={'ineligible': 33}\n"
    )

    report = _scan_server_log(
        log_path,
        require_runtime_stats=True,
        max_final_ineligible=32,
    )

    assert report["status"] == "fail"
    assert any("exceeds request budget" in issue for issue in report["issues"])


def test_server_log_scan_rejects_non_final_or_key_miss_fallback(tmp_path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        'Code2Wav NPU graph startup stats={"disable_reason":null,"enabled":true}\n'
        "Code2Wav NPU graph replay active: execution_mode=npu_graph\n"
        "Code2Wav NPU graph eager fallback: reason=ineligible key=GraphKey(1, 7)\n"
        "Code2Wav NPU graph runtime stats: graph_replays=100 "
        "replay_failures=0 fallback_counts={'ineligible': 1, 'key_miss': 1}\n"
    )

    report = _scan_server_log(
        log_path,
        require_runtime_stats=True,
        max_final_ineligible=32,
    )

    assert report["status"] == "fail"
    assert report["unexpected_eager_fallback_markers"] == 1
    assert report["unexpected_fallback_counts"] == {"key_miss": 1}
