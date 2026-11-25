"""FastAPI routes for health, hardware, models, adaptive inference, and benchmarks."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import MODEL_CATALOG, get_model_spec
from app.core.database import get_db
from app.models.db_models import BenchmarkRun, InferenceMetric, RoutingDecision
from app.schemas.api_schemas import (
    AdaptiveInferRequest,
    AdaptiveInferResponse,
    BenchmarkResultRow,
    BenchmarkResultsResponse,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    BenchmarkRunResultItem,
    HardwareSnapshot,
    HealthResponse,
    InferenceTelemetryOut,
    ModelInfo,
    ModelsResponse,
    RoutingDecisionOut,
)
from app.services.benchmark_runner import BenchmarkRunner
from app.services.exporter import (
    build_csv,
    build_json_artifact,
    dumps_json,
    fetch_benchmark_rows,
)
from app.services.ollama_client import OllamaClient
from app.services.router import AdaptiveRouter
from app.services.telemetry import RunnerRssTracker, capture_hardware_snapshot

router = APIRouter(prefix="/api")


def get_ollama_client() -> OllamaClient:
    return OllamaClient()


@router.get("/health", response_model=HealthResponse)
async def health(ollama: OllamaClient = Depends(get_ollama_client)) -> HealthResponse:
    reachable = await ollama.health_check()
    await ollama.aclose()
    return HealthResponse(
        status="ok" if reachable else "degraded",
        ollama_reachable=reachable,
        detail=None if reachable else "Ollama is not reachable at localhost:11434",
    )


@router.get("/hardware", response_model=HardwareSnapshot)
def hardware() -> HardwareSnapshot:
    snap = capture_hardware_snapshot(cpu_interval=0.1)
    return HardwareSnapshot(
        cpu_percent=snap.cpu_percent,
        total_ram_mb=snap.total_ram_mb,
        available_ram_mb=snap.available_ram_mb,
        used_ram_mb=snap.used_ram_mb,
        runner_rss_mb=snap.runner_rss_mb,
        ollama_process_count=snap.ollama_process_count,
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    ollama: OllamaClient = Depends(get_ollama_client),
) -> ModelsResponse:
    installed: set[str] = set()
    try:
        local = await ollama.list_local_models()
        for entry in local:
            name = entry.get("name") or entry.get("model")
            if name:
                installed.add(str(name))
    except Exception:
        installed = set()
    finally:
        await ollama.aclose()

    models = [
        ModelInfo(
            ollama_tag=spec.ollama_tag,
            family=spec.family,
            params_b=spec.params_b,
            quantization=spec.quantization,
            estimated_ram_mb=spec.estimated_ram_mb,
            complexity_tier=spec.complexity_tier,
            installed_locally=spec.ollama_tag in installed
            or any(spec.ollama_tag in tag for tag in installed),
        )
        for spec in MODEL_CATALOG
    ]
    return ModelsResponse(models=models)


async def _installed_model_tags(ollama: OllamaClient) -> set[str]:
    try:
        local = await ollama.list_local_models()
    except Exception:
        return set()
    tags: set[str] = set()
    for entry in local:
        name = entry.get("name") or entry.get("model")
        if name:
            tags.add(str(name))
    return tags


@router.post("/infer/adaptive", response_model=AdaptiveInferResponse)
async def infer_adaptive(
    body: AdaptiveInferRequest,
    db: Session = Depends(get_db),
    ollama: OllamaClient = Depends(get_ollama_client),
) -> AdaptiveInferResponse:
    hw = capture_hardware_snapshot()
    installed = await _installed_model_tags(ollama)
    adaptive = AdaptiveRouter()
    decision = adaptive.route(
        prompt=body.prompt,
        available_ram_mb=hw.available_ram_mb,
        task_type=body.task_type,
        installed_tags=installed or None,
    )

    routing_row = RoutingDecision(
        prompt=body.prompt,
        task_type=body.task_type,
        complexity_score=decision.complexity_score,
        selected_model=decision.selected_model,
        quantization=decision.quantization,
        hardware_constrained=decision.hardware_constrained,
        available_ram_mb=decision.available_ram_mb,
        rationale=decision.rationale,
    )
    db.add(routing_row)
    db.commit()

    rss = RunnerRssTracker()
    before = rss.mark_before()
    try:
        telemetry = await ollama.generate_with_telemetry(
            model=decision.selected_model,
            prompt=body.prompt,
            options=body.options,
        )
    except Exception as exc:
        await ollama.aclose()
        raise HTTPException(status_code=502, detail=f"Ollama generate failed: {exc}") from exc
    finally:
        await ollama.aclose()

    peak = rss.mark_after()
    hw_after = capture_hardware_snapshot()

    metric = InferenceMetric(
        prompt=body.prompt,
        task_type=body.task_type,
        model_tag=decision.selected_model,
        quantization=decision.quantization,
        response_text=telemetry.response,
        load_time_ms=telemetry.load_time_ms,
        ttft_ms=telemetry.ttft_ms,
        decode_time_ms=telemetry.decode_time_ms,
        tokens_per_sec=telemetry.tokens_per_sec,
        total_time_ms=telemetry.total_time_ms,
        runner_rss_before_mb=before,
        runner_rss_peak_mb=peak,
        available_ram_mb=hw_after.available_ram_mb,
        cpu_percent=hw_after.cpu_percent,
    )
    db.add(metric)
    db.commit()

    return AdaptiveInferResponse(
        routing=RoutingDecisionOut(
            selected_model=decision.selected_model,
            quantization=decision.quantization,
            complexity_score=decision.complexity_score,
            hardware_constrained=decision.hardware_constrained,
            rationale=decision.rationale,
            available_ram_mb=decision.available_ram_mb,
        ),
        response=telemetry.response,
        telemetry=InferenceTelemetryOut(
            load_time_ms=telemetry.load_time_ms,
            ttft_ms=telemetry.ttft_ms,
            decode_time_ms=telemetry.decode_time_ms,
            tokens_per_sec=telemetry.tokens_per_sec,
            total_time_ms=telemetry.total_time_ms,
            eval_count=telemetry.eval_count,
            runner_rss_before_mb=before,
            runner_rss_peak_mb=peak,
            available_ram_mb=hw_after.available_ram_mb,
            cpu_percent=hw_after.cpu_percent,
        ),
    )


@router.post("/benchmarks/run", response_model=BenchmarkRunResponse)
async def run_benchmarks(
    body: BenchmarkRunRequest,
    db: Session = Depends(get_db),
) -> BenchmarkRunResponse:
    if body.model_tags:
        unknown = [t for t in body.model_tags if get_model_spec(t) is None]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model tags: {unknown}",
            )

    runner = BenchmarkRunner(db=db)
    try:
        rows = await runner.run(
            model_tags=body.model_tags,
            use_adaptive_routing=body.use_adaptive_routing,
            cooldown_seconds=body.cooldown_seconds,
            limit=body.limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Benchmark run failed: {exc}") from exc
    finally:
        await runner.ollama.aclose()

    results = [
        BenchmarkRunResultItem(
            suite_item_id=row.suite_item_id,
            task_type=row.task_type,
            model_tag=row.model_tag,
            quantization=row.quantization,
            quality_score=row.quality_score,
            load_time_ms=row.load_time_ms,
            ttft_ms=row.ttft_ms,
            decode_time_ms=row.decode_time_ms,
            tokens_per_sec=row.tokens_per_sec,
            total_time_ms=row.total_time_ms,
            runner_rss_before_mb=row.runner_rss_before_mb,
            runner_rss_peak_mb=row.runner_rss_peak_mb,
        )
        for row in rows
    ]
    return BenchmarkRunResponse(
        runs_completed=len(results),
        cooldown_seconds=body.cooldown_seconds,
        results=results,
    )


@router.get("/benchmarks/results", response_model=BenchmarkResultsResponse)
def benchmark_results(
    limit: int = 100,
    db: Session = Depends(get_db),
) -> BenchmarkResultsResponse:
    stmt = (
        select(BenchmarkRun)
        .order_by(BenchmarkRun.id.desc())
        .limit(max(1, min(limit, 1000)))
    )
    rows = list(db.scalars(stmt).all())
    return BenchmarkResultsResponse(
        count=len(rows),
        results=[BenchmarkResultRow.model_validate(row) for row in rows],
    )


@router.get("/metrics/summary")
def metrics_summary(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Aggregate helper for the research dashboard."""
    bench_count = db.query(BenchmarkRun).count()
    infer_count = db.query(InferenceMetric).count()
    route_count = db.query(RoutingDecision).count()
    return {
        "benchmark_runs": bench_count,
        "inference_metrics": infer_count,
        "routing_decisions": route_count,
    }


@router.get("/export/csv")
def export_benchmarks_csv(
    limit: int = Query(default=10_000, ge=1, le=50_000),
    db: Session = Depends(get_db),
) -> Response:
    """Download benchmark runs as CSV for paper figures / spreadsheets."""
    rows = fetch_benchmark_rows(db, limit=limit)
    csv_body = build_csv(rows)
    return Response(
        content=csv_body,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="edgeslm_benchmark_runs.csv"'
        },
    )


@router.get("/export/json")
def export_benchmarks_json(
    limit: int = Query(default=10_000, ge=1, le=50_000),
    db: Session = Depends(get_db),
) -> Response:
    """Download a JSON experiment artifact with per-model aggregates."""
    rows = fetch_benchmark_rows(db, limit=limit)
    payload = build_json_artifact(db, rows)
    return Response(
        content=dumps_json(payload),
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="edgeslm_experiment_artifact.json"'
        },
    )
