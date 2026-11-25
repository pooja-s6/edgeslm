"""Deterministic task scorers (no LLM-as-judge)."""

from __future__ import annotations

import json
import re
from typing import Any


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(text))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length for ROUGE-L."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 between prediction and reference."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def score_classification(prediction: str, ground_truth: Any) -> float:
    """Normalized exact / label match for classification tasks."""
    pred = _normalize_text(str(prediction))
    truth = _normalize_text(str(ground_truth))
    if not truth:
        return 0.0
    if pred == truth:
        return 1.0
    # Allow label appearing as a standalone token in a short answer.
    pred_tokens = set(_tokenize(pred))
    truth_tokens = set(_tokenize(truth))
    if truth_tokens and truth_tokens.issubset(pred_tokens) and len(pred_tokens) <= 8:
        return 1.0
    if truth in pred and len(pred) <= len(truth) + 16:
        return 0.75
    return 0.0


def _parse_json_blob(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _flatten_items(obj: dict[str, Any], prefix: str = "") -> set[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            items |= _flatten_items(value, path)
        else:
            items.add((path.lower(), _normalize_text(str(value))))
    return items


def score_extraction(prediction: str, ground_truth: Any) -> float:
    """Valid JSON parsing + key/value F1 against ground-truth object."""
    if isinstance(ground_truth, str):
        truth_obj = _parse_json_blob(ground_truth)
    elif isinstance(ground_truth, dict):
        truth_obj = ground_truth
    else:
        return 0.0

    pred_obj = _parse_json_blob(prediction)
    if truth_obj is None or pred_obj is None:
        return 0.0

    truth_items = _flatten_items(truth_obj)
    pred_items = _flatten_items(pred_obj)
    if not truth_items and not pred_items:
        return 1.0
    if not truth_items or not pred_items:
        return 0.0

    overlap = len(truth_items & pred_items)
    precision = overlap / len(pred_items)
    recall = overlap / len(truth_items)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def score_summarization(prediction: str, ground_truth: Any) -> float:
    """ROUGE-L / key-concept coverage for summarization."""
    if isinstance(ground_truth, dict):
        reference = str(ground_truth.get("summary") or ground_truth.get("text") or "")
        concepts = ground_truth.get("key_concepts") or []
    else:
        reference = str(ground_truth)
        concepts = []

    rouge = rouge_l_f1(prediction, reference)
    if not concepts:
        return rouge

    pred_norm = _normalize_text(prediction)
    hits = sum(1 for c in concepts if _normalize_text(str(c)) in pred_norm)
    coverage = hits / len(concepts)
    return 0.6 * rouge + 0.4 * coverage


def _extract_final_answer(text: str) -> str | None:
    """Extract a trailing numeric/boolean/final-answer style value."""
    patterns = [
        r"(?:final\s*answer|answer)\s*[:=]\s*([-+]?\d+(?:\.\d+)?|true|false|yes|no)\b",
        r"\b([-+]?\d+(?:\.\d+)?)\s*$",
        r"\b(true|false|yes|no)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)
    # Fallback: last number anywhere.
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def score_reasoning(prediction: str, ground_truth: Any) -> float:
    """Final numeric/boolean answer extraction + match."""
    truth_raw = ground_truth
    if isinstance(ground_truth, dict):
        truth_raw = ground_truth.get("answer", ground_truth)

    truth = _normalize_text(str(truth_raw))
    extracted = _extract_final_answer(prediction)
    if extracted is None:
        return 0.0

    pred = _normalize_text(extracted)
    # Boolean aliases
    bool_map = {"yes": "true", "no": "false"}
    pred = bool_map.get(pred, pred)
    truth = bool_map.get(truth, truth)

    try:
        if abs(float(pred) - float(truth)) < 1e-6:
            return 1.0
    except ValueError:
        pass

    return 1.0 if pred == truth else 0.0


def score_prediction(task_type: str, prediction: str, ground_truth: Any) -> float:
    """Dispatch to the appropriate deterministic scorer."""
    scorers = {
        "classification": score_classification,
        "extraction": score_extraction,
        "summarization": score_summarization,
        "reasoning": score_reasoning,
    }
    scorer = scorers.get(task_type)
    if scorer is None:
        raise ValueError(f"Unsupported task_type: {task_type}")
    score = scorer(prediction, ground_truth)
    return max(0.0, min(1.0, float(score)))
