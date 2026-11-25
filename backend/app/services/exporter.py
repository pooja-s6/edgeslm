"""Export benchmark runs as CSV/JSON experiment artifacts."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.db_models import BenchmarkRun, InferenceMetric, RoutingDecision

CSV_COLUMNS = [
    "id",
    "suite_item_id",
    "task_type",
    "model_tag",
    "quantization",
    "quality_score",
    "load_time_ms",
    "ttft_ms",
    "decode_time_ms",
    "tokens_per_sec",
    "total_time_ms",
    "runner_rss_before_mb",
    "runner_rss_peak_mb",
    "available_ram_mb",
    "cpu_percent",
    "created_at",
]


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).isoformat()
    return value.isoformat()


def fetch_benchmark_rows(db: Session, limit: int = 10_000) -> list[BenchmarkRun]:
    stmt = (
        select(BenchmarkRun)
        .order_by(BenchmarkRun.id.asc())
        .limit(max(1, min(limit, 50_000)))
    )
    return list(db.scalars(stmt).all())


def row_to_dict(row: BenchmarkRun) -> dict[str, Any]:
    return {
        "id": row.id,
        "suite_item_id": row.suite_item_id,
        "task_type": row.task_type,
        "model_tag": row.model_tag,
        "quantization": row.quantization,
        "quality_score": row.quality_score,
        "load_time_ms": row.load_time_ms,
        "ttft_ms": row.ttft_ms,
        "decode_time_ms": row.decode_time_ms,
        "tokens_per_sec": row.tokens_per_sec,
        "total_time_ms": row.total_time_ms,
        "runner_rss_before_mb": row.runner_rss_before_mb,
        "runner_rss_peak_mb": row.runner_rss_peak_mb,
        "available_ram_mb": row.available_ram_mb,
        "cpu_percent": row.cpu_percent,
        "created_at": _iso(row.created_at),
    }


def build_csv(rows: list[BenchmarkRun]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        payload = row_to_dict(row)
        writer.writerow(payload)
    return buffer.getvalue()


def build_json_artifact(db: Session, rows: list[BenchmarkRun]) -> dict[str, Any]:
    """Academic-friendly experiment bundle for paper figures."""
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row.model_tag, []).append(row_to_dict(row))

    aggregates: list[dict[str, Any]] = []
    for model_tag, items in sorted(by_model.items()):
        n = len(items)

        def mean(key: str) -> float | None:
            vals = [float(i[key]) for i in items if i.get(key) is not None]
            return (sum(vals) / len(vals)) if vals else None

        aggregates.append(
            {
                "model_tag": model_tag,
                "n": n,
                "mean_quality": mean("quality_score"),
                "mean_ttft_ms": mean("ttft_ms"),
                "mean_tokens_per_sec": mean("tokens_per_sec"),
                "mean_total_time_ms": mean("total_time_ms"),
                "mean_runner_rss_peak_mb": mean("runner_rss_peak_mb"),
            }
        )

    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "artifact_version": 1,
        "counts": {
            "benchmark_runs": len(rows),
            "inference_metrics": db.query(InferenceMetric).count(),
            "routing_decisions": db.query(RoutingDecision).count(),
        },
        "aggregates_by_model": aggregates,
        "runs": [row_to_dict(r) for r in rows],
    }


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)
