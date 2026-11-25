"""Unit tests for Ollama timing conversion and psutil Ollama RSS filtering."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.ollama_client import parse_generate_telemetry
from app.services.telemetry import (
    capture_hardware_snapshot,
    get_ollama_runner_rss_mb,
    is_ollama_process_name,
)


def test_parse_generate_telemetry_nanosecond_conversions() -> None:
    payload = {
        "response": "hello",
        "load_duration": 1_500_000_000,  # 1500 ms
        "prompt_eval_duration": 250_000_000,  # 250 ms TTFT proxy
        "eval_duration": 2_000_000_000,  # 2 s decode
        "total_duration": 4_000_000_000,  # 4000 ms
        "eval_count": 40,
        "prompt_eval_count": 10,
    }
    result = parse_generate_telemetry(payload)
    assert result.response == "hello"
    assert result.load_time_ms == 1500.0
    assert result.ttft_ms == 250.0
    assert result.decode_time_ms == 2000.0
    assert result.total_time_ms == 4000.0
    assert result.eval_count == 40
    assert result.tokens_per_sec == 20.0


def test_parse_generate_telemetry_zero_eval_duration() -> None:
    payload = {
        "response": "",
        "load_duration": 0,
        "prompt_eval_duration": 0,
        "eval_duration": 0,
        "total_duration": 0,
        "eval_count": 5,
    }
    result = parse_generate_telemetry(payload)
    assert result.tokens_per_sec == 0.0
    assert result.load_time_ms == 0.0


def test_is_ollama_process_name() -> None:
    assert is_ollama_process_name("ollama") is True
    assert is_ollama_process_name("ollama.exe") is True
    assert is_ollama_process_name("ollama_runner") is True
    assert is_ollama_process_name("ollama_runner.exe") is True
    assert is_ollama_process_name("chrome") is False
    assert is_ollama_process_name("python") is False


def test_get_ollama_runner_rss_sums_matching_processes() -> None:
    procs = [
        MagicMock(
            info={
                "name": "ollama.exe",
                "memory_info": SimpleNamespace(rss=100 * 1024 * 1024),
            }
        ),
        MagicMock(
            info={
                "name": "ollama_runner.exe",
                "memory_info": SimpleNamespace(rss=50 * 1024 * 1024),
            }
        ),
        MagicMock(
            info={
                "name": "notepad.exe",
                "memory_info": SimpleNamespace(rss=10 * 1024 * 1024),
            }
        ),
    ]
    with patch("app.services.telemetry.psutil.process_iter", return_value=procs):
        rss_mb, count = get_ollama_runner_rss_mb()
    assert count == 2
    assert abs(rss_mb - 150.0) < 1e-6


def test_capture_hardware_snapshot_shape() -> None:
    snap = capture_hardware_snapshot(cpu_interval=0.0)
    assert snap.total_ram_mb > 0
    assert snap.available_ram_mb >= 0
    assert snap.used_ram_mb >= 0
    assert snap.runner_rss_mb >= 0
    assert snap.ollama_process_count >= 0
    assert 0.0 <= snap.cpu_percent <= 100.0 or snap.cpu_percent >= 0.0
