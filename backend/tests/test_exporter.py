"""Unit tests for CSV/JSON experiment export."""

from __future__ import annotations

from app.models.db_models import BenchmarkRun
from app.services.exporter import build_csv, build_json_artifact, row_to_dict


def _sample_row(**overrides: object) -> BenchmarkRun:
    base = dict(
        id=1,
        suite_item_id="cls_01",
        task_type="classification",
        model_tag="llama3.2:1b-instruct-q4_K_M",
        quantization="Q4_K_M",
        quality_score=1.0,
        load_time_ms=100.0,
        ttft_ms=50.0,
        decode_time_ms=200.0,
        tokens_per_sec=20.0,
        total_time_ms=350.0,
        runner_rss_before_mb=400.0,
        runner_rss_peak_mb=450.0,
        available_ram_mb=12000.0,
        cpu_percent=35.0,
        created_at=None,
    )
    base.update(overrides)
    return BenchmarkRun(**base)  # type: ignore[arg-type]


def test_build_csv_includes_header_and_row() -> None:
    csv_text = build_csv([_sample_row()])
    lines = csv_text.strip().splitlines()
    assert "suite_item_id" in lines[0]
    assert "cls_01" in lines[1]
    assert "llama3.2:1b-instruct-q4_K_M" in lines[1]


def test_row_to_dict_keys() -> None:
    payload = row_to_dict(_sample_row())
    assert payload["quality_score"] == 1.0
    assert payload["tokens_per_sec"] == 20.0


class _FakeQuery:
    def count(self) -> int:
        return 0


class _FakeDB:
    def query(self, _model: object) -> _FakeQuery:
        return _FakeQuery()


def test_json_artifact_aggregates() -> None:
    rows = [
        _sample_row(id=1, quality_score=1.0, ttft_ms=40.0),
        _sample_row(id=2, quality_score=0.5, ttft_ms=60.0),
    ]
    artifact = build_json_artifact(_FakeDB(), rows)  # type: ignore[arg-type]
    assert artifact["counts"]["benchmark_runs"] == 2
    assert len(artifact["aggregates_by_model"]) == 1
    agg = artifact["aggregates_by_model"][0]
    assert agg["n"] == 2
    assert abs(agg["mean_quality"] - 0.75) < 1e-9
    assert abs(agg["mean_ttft_ms"] - 50.0) < 1e-9
