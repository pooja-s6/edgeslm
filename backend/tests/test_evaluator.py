"""Unit tests for deterministic task scoring."""

from __future__ import annotations

import pytest

from app.services.evaluator import (
    rouge_l_f1,
    score_classification,
    score_extraction,
    score_prediction,
    score_reasoning,
    score_summarization,
)


def test_classification_exact_match() -> None:
    assert score_classification("positive", "positive") == 1.0
    assert score_classification("POSITIVE", "positive") == 1.0


def test_classification_mismatch_is_zero() -> None:
    assert score_classification("negative", "positive") == 0.0


def test_extraction_perfect_json_f1() -> None:
    pred = '{"name": "Alice", "age": "29", "city": "Austin"}'
    truth = {"name": "Alice", "age": "29", "city": "Austin"}
    assert score_extraction(pred, truth) == 1.0


def test_extraction_partial_overlap() -> None:
    pred = '{"name": "Alice", "age": "30", "city": "Austin"}'
    truth = {"name": "Alice", "age": "29", "city": "Austin"}
    score = score_extraction(pred, truth)
    assert 0.0 < score < 1.0


def test_extraction_invalid_json_scores_zero() -> None:
    assert score_extraction("not json at all", {"a": "1"}) == 0.0


def test_summarization_rouge_bounds() -> None:
    score = score_summarization(
        "Local models protect privacy with quantization under memory limits.",
        {
            "summary": "Local small language models protect privacy but need quantization under memory limits.",
            "key_concepts": ["privacy", "quantization", "memory"],
        },
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.3


def test_rouge_l_identical_is_one() -> None:
    assert rouge_l_f1("alpha beta gamma", "alpha beta gamma") == 1.0


def test_reasoning_numeric_final_answer() -> None:
    pred = "17 + 25 = 42\nFinal answer: 42"
    assert score_reasoning(pred, {"answer": 42}) == 1.0


def test_reasoning_boolean_and_mismatch() -> None:
    assert score_reasoning("Final answer: false", {"answer": False}) == 1.0
    assert score_reasoning("Final answer: true", {"answer": False}) == 0.0


def test_score_prediction_dispatch_and_unknown() -> None:
    assert score_prediction("classification", "account", "account") == 1.0
    with pytest.raises(ValueError):
        score_prediction("unknown_task", "x", "y")
