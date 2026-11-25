"""Batch benchmark sweep orchestrator with thermal cooldowns."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import (
    BENCHMARK_SUITE_PATH,
    DEFAULT_COOLDOWN_SECONDS,
    MODEL_CATALOG,
    get_model_spec,
)
from app.models.db_models import BenchmarkRun
from app.services.evaluator import score_prediction
from app.services.ollama_client import OllamaClient
from app.services.router import AdaptiveRouter
from app.services.telemetry import RunnerRssTracker, capture_hardware_snapshot


def load_benchmark_suite(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the verified benchmark suite JSON."""
    suite_path = path or BENCHMARK_SUITE_PATH
    with suite_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        items = data.get("items") or data.get("benchmarks") or []
    else:
        items = data
    if not isinstance(items, list):
        raise ValueError("Benchmark suite must be a list or contain an items list")
    return items


class BenchmarkRunner:
    """Run suite items against catalog models (or adaptive routing) and persist results."""

    def __init__(
        self,
        db: Session,
        ollama: OllamaClient | None = None,
        router: AdaptiveRouter | None = None,
        suite_path: Path | None = None,
    ) -> None:
        self.db = db
        self.ollama = ollama or OllamaClient()
        self.router = router or AdaptiveRouter()
        self.suite_path = suite_path or BENCHMARK_SUITE_PATH

    async def run(
        self,
        model_tags: list[str] | None = None,
        use_adaptive_routing: bool = False,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        limit: int | None = None,
    ) -> list[BenchmarkRun]:
        items = load_benchmark_suite(self.suite_path)
        if limit is not None:
            items = items[:limit]

        if use_adaptive_routing:
            targets: list[str | None] = [None]
        elif model_tags:
            targets = list(model_tags)
        else:
            targets = [m.ollama_tag for m in MODEL_CATALOG]

        persisted: list[BenchmarkRun] = []
        first = True

        for item in items:
            suite_item_id = str(item["id"])
            task_type = str(item["task_type"])
            prompt = str(item["prompt"])
            ground_truth = item.get("ground_truth")

            for target in targets:
                if not first and cooldown_seconds > 0:
                    await asyncio.sleep(cooldown_seconds)
                first = False

                hw = capture_hardware_snapshot()
                if use_adaptive_routing or target is None:
                    installed: set[str] = set()
                    try:
                        local = await self.ollama.list_local_models()
                        for entry in local:
                            name = entry.get("name") or entry.get("model")
                            if name:
                                installed.add(str(name))
                    except Exception:
                        installed = set()
                    decision = self.router.route(
                        prompt=prompt,
                        available_ram_mb=hw.available_ram_mb,
                        task_type=task_type,
                        installed_tags=installed or None,
                    )
                    model_tag = decision.selected_model
                else:
                    model_tag = target

                spec = get_model_spec(model_tag)
                quantization = spec.quantization if spec else "unknown"

                rss = RunnerRssTracker()
                before = rss.mark_before()
                telemetry = await self.ollama.generate_with_telemetry(
                    model=model_tag,
                    prompt=prompt,
                )
                peak = rss.mark_after()
                hw_after = capture_hardware_snapshot()

                quality = score_prediction(
                    task_type=task_type,
                    prediction=telemetry.response,
                    ground_truth=ground_truth,
                )

                row = BenchmarkRun(
                    suite_item_id=suite_item_id,
                    task_type=task_type,
                    model_tag=model_tag,
                    quantization=quantization,
                    quality_score=quality,
                    load_time_ms=telemetry.load_time_ms,
                    ttft_ms=telemetry.ttft_ms,
                    decode_time_ms=telemetry.decode_time_ms,
                    tokens_per_sec=telemetry.tokens_per_sec,
                    total_time_ms=telemetry.total_time_ms,
                    runner_rss_before_mb=before,
                    runner_rss_peak_mb=peak,
                    available_ram_mb=hw_after.available_ram_mb,
                    cpu_percent=hw_after.cpu_percent,
                    response_text=telemetry.response,
                )
                self.db.add(row)
                self.db.commit()
                self.db.refresh(row)
                persisted.append(row)

        return persisted
