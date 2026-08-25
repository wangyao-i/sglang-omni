# SPDX-License-Identifier: Apache-2.0
"""Exercise the full Qwen3-Omni speech path after enabling Code2Wav NPUGraph."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

_PROMPTS = (
    "请用两句话介绍昇腾 NPU，并说明它适合哪类人工智能任务。",
    "Explain in three short sentences why graph replay can reduce inference overhead.",
    "请给初学者解释 eager execution 与 graph replay 的区别，回答四句话。",
    "Describe one benefit and one limitation of accelerator graph capture in four sentences.",
)


@dataclass(slots=True)
class RequestResult:
    phase: str
    index: int
    prompt_index: int
    started_s: float
    ended_s: float
    latency_s: float
    status: str
    response_id: str | None = None
    text: str | None = None
    audio_bytes: int | None = None
    audio_seconds: float | None = None
    audio_sha256: str | None = None
    error: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument(
        "--model-path",
        default="/home/weights/Qwen3-Omni-30B-A3B-Instruct",
    )
    parser.add_argument("--sequential-requests", type=int, default=12)
    parser.add_argument("--concurrent-requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--min-audio-bytes", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--server-log",
        type=Path,
        help="Fresh server log to scan for graph replay, fallback, and device errors.",
    )
    parser.add_argument(
        "--require-runtime-stats",
        action="store_true",
        help="Require the periodic Code2Wav graph runtime stats marker.",
    )
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _decode_wav(audio_b64: str, *, min_audio_bytes: int) -> tuple[bytes, float]:
    try:
        audio = base64.b64decode(audio_b64, validate=True)
    except Exception as exc:
        raise ValueError(f"invalid base64 audio: {exc}") from exc
    if len(audio) < min_audio_bytes:
        raise ValueError(
            f"audio payload is too small: {len(audio)} < {min_audio_bytes} bytes"
        )
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"audio payload is not a valid WAV file: {exc}") from exc
    if frame_count <= 0 or sample_rate <= 0:
        raise ValueError(
            f"WAV contains no playable audio: frames={frame_count}, rate={sample_rate}"
        )
    return audio, frame_count / sample_rate


def _request_payload(model_path: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model_path,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }


def _run_request(
    *,
    base_url: str,
    model_path: str,
    phase: str,
    index: int,
    timeout: float,
    max_tokens: int,
    min_audio_bytes: int,
    output_dir: Path | None,
    epoch: float,
) -> RequestResult:
    prompt_index = index % len(_PROMPTS)
    started = time.perf_counter()
    common = {
        "phase": phase,
        "index": index,
        "prompt_index": prompt_index,
        "started_s": started - epoch,
    }
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=_request_payload(model_path, _PROMPTS[prompt_index], max_tokens),
            timeout=timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"HTTP {response.status_code}: {response.text[:500].strip()}"
            )
        body = response.json()
        message = body["choices"][0]["message"]
        audio_b64 = (message.get("audio") or {}).get("data")
        if not audio_b64:
            raise ValueError("response contains no choices[0].message.audio.data")
        audio, audio_seconds = _decode_wav(
            audio_b64,
            min_audio_bytes=min_audio_bytes,
        )
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{phase}_{index:03d}.wav").write_bytes(audio)
        ended = time.perf_counter()
        return RequestResult(
            **common,
            ended_s=ended - epoch,
            latency_s=ended - started,
            status="pass",
            response_id=body.get("id"),
            text=message.get("content"),
            audio_bytes=len(audio),
            audio_seconds=audio_seconds,
            audio_sha256=hashlib.sha256(audio).hexdigest(),
        )
    except Exception as exc:
        ended = time.perf_counter()
        return RequestResult(
            **common,
            ended_s=ended - epoch,
            latency_s=ended - started,
            status="fail",
            error=f"{type(exc).__name__}: {exc}",
        )


def _run_phase(
    *,
    phase: str,
    request_count: int,
    concurrency: int,
    request_kwargs: dict[str, Any],
) -> list[RequestResult]:
    if request_count == 0:
        return []
    if concurrency == 1:
        return [
            _run_request(phase=phase, index=index, **request_kwargs)
            for index in range(request_count)
        ]
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _run_request,
                phase=phase,
                index=index,
                **request_kwargs,
            ): index
            for index in range(request_count)
        }
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda result: result.index)


def _peak_concurrency(results: list[RequestResult]) -> int:
    events = [
        event
        for result in results
        for event in ((result.started_s, 1), (result.ended_s, -1))
    ]
    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        peak = max(peak, active)
    return peak


def _phase_summary(results: list[RequestResult]) -> dict[str, Any]:
    latencies = [result.latency_s for result in results if result.status == "pass"]
    started = min((result.started_s for result in results), default=0.0)
    ended = max((result.ended_s for result in results), default=started)
    wall_s = ended - started
    return {
        "requests": len(results),
        "passed": sum(result.status == "pass" for result in results),
        "failed": sum(result.status != "pass" for result in results),
        "wall_s": wall_s,
        "requests_per_second": len(results) / wall_s if wall_s > 0 else None,
        "peak_concurrency": _peak_concurrency(results),
        "latency_s": {
            "min": min(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "audio_bytes": sum(result.audio_bytes or 0 for result in results),
        "audio_seconds": sum(result.audio_seconds or 0.0 for result in results),
    }


def _scan_server_log(path: Path, *, require_runtime_stats: bool) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    startup_marker = "Code2Wav NPU graph startup stats="
    replay_marker = "Code2Wav NPU graph replay active"
    runtime_marker = "Code2Wav NPU graph runtime stats"
    failure_markers = {
        "eager_fallback": "Code2Wav NPU graph eager fallback",
        "capture_failed": "capture_failed",
        "runtime_replay_failed": "runtime_replay_failed",
        "runner_disabled": "Code2Wav NPU graph replay disabled the runner",
        "acl_stream_error": "107027",
    }
    failures = {
        name: text.count(marker)
        for name, marker in failure_markers.items()
        if marker in text
    }
    startup_lines = [line for line in text.splitlines() if startup_marker in line]
    healthy_startup = any(
        '"enabled":true' in line and '"disable_reason":null' in line
        for line in startup_lines
    )
    replay_markers = text.count(replay_marker)
    runtime_lines = [line for line in text.splitlines() if runtime_marker in line]
    unhealthy_runtime_lines = [
        line
        for line in runtime_lines
        if "replay_failures=0" not in line or "fallback_counts={}" not in line
    ]
    issues = []
    if not healthy_startup:
        issues.append("missing healthy graph startup stats")
    if replay_markers == 0:
        issues.append("missing graph replay marker")
    if require_runtime_stats and not runtime_lines:
        issues.append("missing periodic graph runtime stats marker")
    if unhealthy_runtime_lines:
        issues.append("periodic graph runtime stats report failures or fallback")
    if failures:
        issues.append(f"graph/device failure markers present: {failures}")
    return {
        "path": str(path),
        "status": "pass" if not issues else "fail",
        "startup_markers": len(startup_lines),
        "healthy_startup": healthy_startup,
        "replay_markers": replay_markers,
        "runtime_stats_markers": len(runtime_lines),
        "unhealthy_runtime_stats": len(unhealthy_runtime_lines),
        "failure_markers": failures,
        "issues": issues,
    }


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("sequential_requests", "concurrent_requests"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.sequential_requests + args.concurrent_requests == 0:
        raise ValueError("at least one request must be selected")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if args.timeout <= 0 or args.max_tokens <= 0 or args.min_audio_bytes <= 0:
        raise ValueError("timeout, max-tokens, and min-audio-bytes must be positive")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    base_url = args.base_url.rstrip("/")
    health = requests.get(f"{base_url}/v1/models", timeout=min(args.timeout, 30.0))
    health.raise_for_status()

    epoch = time.perf_counter()
    request_kwargs = {
        "base_url": base_url,
        "model_path": args.model_path,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
        "min_audio_bytes": args.min_audio_bytes,
        "output_dir": args.output_dir,
        "epoch": epoch,
    }
    sequential = _run_phase(
        phase="sequential",
        request_count=args.sequential_requests,
        concurrency=1,
        request_kwargs=request_kwargs,
    )
    concurrent = _run_phase(
        phase="concurrent",
        request_count=args.concurrent_requests,
        concurrency=min(args.concurrency, max(args.concurrent_requests, 1)),
        request_kwargs=request_kwargs,
    )
    results = sequential + concurrent
    issues = [result.error for result in results if result.status != "pass"]
    concurrent_peak = _peak_concurrency(concurrent)
    if args.concurrent_requests >= 2 and args.concurrency >= 2 and concurrent_peak < 2:
        issues.append("concurrent phase did not demonstrate overlapping requests")

    log_report = None
    if args.server_log is not None:
        log_report = _scan_server_log(
            args.server_log,
            require_runtime_stats=args.require_runtime_stats,
        )
        issues.extend(log_report["issues"])
    elif args.require_runtime_stats:
        issues.append("--require-runtime-stats requires --server-log")

    report = {
        "status": "pass" if not issues else "fail",
        "base_url": base_url,
        "model_path": args.model_path,
        "configuration": {
            "sequential_requests": args.sequential_requests,
            "concurrent_requests": args.concurrent_requests,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "min_audio_bytes": args.min_audio_bytes,
        },
        "summary": {
            "sequential": _phase_summary(sequential),
            "concurrent": _phase_summary(concurrent),
            "total": _phase_summary(results),
        },
        "server_log": log_report,
        "issues": issues,
        "requests": [asdict(result) for result in results],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
