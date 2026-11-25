"""Pydantic request/response contracts for the EdgeSLM Lab API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

TaskType = Literal["classification", "extraction", "summarization", "reasoning"]


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    detail: str | None = None


class HardwareSnapshot(BaseModel):
    cpu_percent: float
    total_ram_mb: float
    available_ram_mb: float
    used_ram_mb: float
    runner_rss_mb: float
    ollama_process_count: int


class ModelInfo(BaseModel):
    ollama_tag: str
    family: str
    params_b: float
    quantization: str
    estimated_ram_mb: int
    complexity_tier: str
    installed_locally: bool = False


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class AdaptiveInferRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    task_type: TaskType | None = None
    options: dict[str, Any] | None = None


class RoutingDecisionOut(BaseModel):
    selected_model: str
    quantization: str
    complexity_score: float
    hardware_constrained: bool
    rationale: str
    available_ram_mb: float


class InferenceTelemetryOut(BaseModel):
    load_time_ms: float
    ttft_ms: float
    decode_time_ms: float
    tokens_per_sec: float
    total_time_ms: float
    eval_count: int
    runner_rss_before_mb: float | None = None
    runner_rss_peak_mb: float | None = None
    available_ram_mb: float | None = None
    cpu_percent: float | None = None


class AdaptiveInferResponse(BaseModel):
    routing: RoutingDecisionOut
    response: str
    telemetry: InferenceTelemetryOut


class BenchmarkRunRequest(BaseModel):
    model_tags: list[str] | None = Field(
        default=None,
        description="Optional subset of catalog tags; default runs all catalog models.",
    )
    use_adaptive_routing: bool = False
    cooldown_seconds: float = Field(default=3.0, ge=0.0)
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap on suite items for smoke runs.",
    )


class BenchmarkRunResultItem(BaseModel):
    suite_item_id: str
    task_type: str
    model_tag: str
    quantization: str
    quality_score: float
    load_time_ms: float
    ttft_ms: float
    decode_time_ms: float
    tokens_per_sec: float
    total_time_ms: float
    runner_rss_before_mb: float | None = None
    runner_rss_peak_mb: float | None = None


class BenchmarkRunResponse(BaseModel):
    runs_completed: int
    cooldown_seconds: float
    results: list[BenchmarkRunResultItem]


class BenchmarkResultRow(BaseModel):
    id: int
    suite_item_id: str
    task_type: str
    model_tag: str
    quantization: str
    quality_score: float
    load_time_ms: float
    ttft_ms: float
    decode_time_ms: float
    tokens_per_sec: float
    total_time_ms: float
    runner_rss_before_mb: float | None
    runner_rss_peak_mb: float | None
    available_ram_mb: float | None
    cpu_percent: float | None
    created_at: datetime | None

    model_config = {"from_attributes": True}


class BenchmarkResultsResponse(BaseModel):
    count: int
    results: list[BenchmarkResultRow]
