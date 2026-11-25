"""SQLAlchemy tables for benchmark runs, inference metrics, and routing decisions."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class BenchmarkRun(Base):
    """One scored inference against a benchmark suite item."""

    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    suite_item_id: Mapped[str] = mapped_column(String(64), index=True)
    task_type: Mapped[str] = mapped_column(String(32), index=True)
    model_tag: Mapped[str] = mapped_column(String(128), index=True)
    quantization: Mapped[str] = mapped_column(String(32))
    quality_score: Mapped[float] = mapped_column(Float)
    load_time_ms: Mapped[float] = mapped_column(Float)
    ttft_ms: Mapped[float] = mapped_column(Float)
    decode_time_ms: Mapped[float] = mapped_column(Float)
    tokens_per_sec: Mapped[float] = mapped_column(Float)
    total_time_ms: Mapped[float] = mapped_column(Float)
    runner_rss_before_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    runner_rss_peak_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    available_ram_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class InferenceMetric(Base):
    """Per adaptive/infer call telemetry row."""

    __tablename__ = "inference_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt: Mapped[str] = mapped_column(Text)
    task_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_tag: Mapped[str] = mapped_column(String(128), index=True)
    quantization: Mapped[str] = mapped_column(String(32))
    response_text: Mapped[str] = mapped_column(Text)
    load_time_ms: Mapped[float] = mapped_column(Float)
    ttft_ms: Mapped[float] = mapped_column(Float)
    decode_time_ms: Mapped[float] = mapped_column(Float)
    tokens_per_sec: Mapped[float] = mapped_column(Float)
    total_time_ms: Mapped[float] = mapped_column(Float)
    runner_rss_before_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    runner_rss_peak_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    available_ram_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class RoutingDecision(Base):
    """Explainable adaptive router decision log."""

    __tablename__ = "routing_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt: Mapped[str] = mapped_column(Text)
    task_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    complexity_score: Mapped[float] = mapped_column(Float)
    selected_model: Mapped[str] = mapped_column(String(128))
    quantization: Mapped[str] = mapped_column(String(32))
    hardware_constrained: Mapped[bool] = mapped_column(Boolean, default=False)
    available_ram_mb: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
