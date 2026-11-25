"""psutil-based CPU, system RAM, and Ollama runner RSS tracking."""

from __future__ import annotations

from dataclasses import dataclass

import psutil


OLLAMA_PROCESS_NAMES = frozenset(
    {
        "ollama",
        "ollama.exe",
        "ollama_runner",
        "ollama_runner.exe",
    }
)


@dataclass(frozen=True)
class HardwareSnapshotData:
    """Live hardware + runner memory snapshot."""

    cpu_percent: float
    total_ram_mb: float
    available_ram_mb: float
    used_ram_mb: float
    runner_rss_mb: float
    ollama_process_count: int


def _bytes_to_mb(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def is_ollama_process_name(name: str) -> bool:
    """Return True if a process name matches known Ollama runners."""
    normalized = name.strip().lower()
    if normalized in OLLAMA_PROCESS_NAMES:
        return True
    # Cover names like "ollama runner" on some platforms.
    return "ollama" in normalized and (
        "runner" in normalized or normalized.startswith("ollama")
    )


def get_ollama_runner_rss_mb() -> tuple[float, int]:
    """
    Sum RSS of processes whose names match ollama / ollama_runner.

    Returns (total_rss_mb, process_count).
    """
    total_rss = 0
    count = 0
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            name = proc.info.get("name") or ""
            if not is_ollama_process_name(name):
                continue
            mem = proc.info.get("memory_info")
            if mem is None:
                mem = proc.memory_info()
            total_rss += int(mem.rss)
            count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return _bytes_to_mb(total_rss), count


def capture_hardware_snapshot(cpu_interval: float = 0.0) -> HardwareSnapshotData:
    """
    Capture CPU %, system RAM, and current Ollama runner RSS.

    cpu_interval=0.0 uses a non-blocking sample relative to the previous call.
    """
    vm = psutil.virtual_memory()
    runner_rss_mb, proc_count = get_ollama_runner_rss_mb()
    return HardwareSnapshotData(
        cpu_percent=float(psutil.cpu_percent(interval=cpu_interval)),
        total_ram_mb=_bytes_to_mb(vm.total),
        available_ram_mb=_bytes_to_mb(vm.available),
        used_ram_mb=_bytes_to_mb(vm.used),
        runner_rss_mb=runner_rss_mb,
        ollama_process_count=proc_count,
    )


class RunnerRssTracker:
    """Track runner RSS before and peak during an inference window."""

    def __init__(self) -> None:
        self.before_mb: float = 0.0
        self.peak_mb: float = 0.0

    def mark_before(self) -> float:
        rss_mb, _ = get_ollama_runner_rss_mb()
        self.before_mb = rss_mb
        self.peak_mb = rss_mb
        return self.before_mb

    def sample(self) -> float:
        rss_mb, _ = get_ollama_runner_rss_mb()
        if rss_mb > self.peak_mb:
            self.peak_mb = rss_mb
        return rss_mb

    def mark_after(self) -> float:
        return self.sample()
