"""Unit tests for complexity scoring and RAM-constraint fallback."""

from __future__ import annotations

from app.core.config import MODEL_CATALOG, OS_SAFETY_BUFFER_MB, ModelSpec
from app.services.router import AdaptiveRouter, compute_complexity_score


def test_complexity_reasoning_higher_than_classification() -> None:
    cls_score = compute_complexity_score(
        "Classify sentiment as positive or negative.",
        task_type="classification",
    )
    rsn_score = compute_complexity_score(
        "Solve this multi-step reasoning problem and explain each step carefully "
        "while you calculate the final numeric answer.",
        task_type="reasoning",
    )
    assert 0.0 <= cls_score <= 1.0
    assert 0.0 <= rsn_score <= 1.0
    assert rsn_score > cls_score


def test_route_selects_low_tier_for_simple_prompt() -> None:
    router = AdaptiveRouter()
    # Plenty of RAM so hardware does not constrain.
    available = OS_SAFETY_BUFFER_MB + 20_000
    result = router.route(
        prompt="Is this positive or negative? Great product.",
        available_ram_mb=available,
        task_type="classification",
    )
    assert result.hardware_constrained is False
    assert result.complexity_score < 0.5
    assert result.selected_model == "llama3.2:1b-instruct-q4_K_M"
    assert "selected" in result.rationale.lower() or "maps" in result.rationale.lower()


def test_difficulty_ladder_routes_distinct_models() -> None:
    router = AdaptiveRouter()
    available = OS_SAFETY_BUFFER_MB + 20_000
    installed = {
        "llama3.2:1b-instruct-q4_K_M",
        "llama3.2:1b-instruct-q8_0",
        "qwen2.5:1.5b-instruct-q4_K_M",
        "qwen2.5:3b-instruct-q4_K_M",
    }
    simple = router.route(
        prompt=(
            "Classify the sentiment of this review as Positive or Negative: "
            "The delivery was fast and the packaging was great."
        ),
        available_ram_mb=available,
        task_type="classification",
        installed_tags=installed,
    )
    medium = router.route(
        prompt=(
            "Extract the company name, role, and location as valid JSON from: "
            "Hexaware is hiring a Backend Intern in Chennai."
        ),
        available_ram_mb=available,
        task_type="extraction",
        installed_tags=installed,
    )
    hard = router.route(
        prompt=(
            "A server processes 45 requests per second at Q4 precision and 28 "
            "requests per second at Q8. If a batch of 1,000 requests splits 60% "
            "to Q4 and 40% to Q8, what is the total processing time? "
            "Show step-by-step math."
        ),
        available_ram_mb=available,
        task_type="reasoning",
        installed_tags=installed,
    )
    assert simple.selected_model == "llama3.2:1b-instruct-q4_K_M"
    assert medium.selected_model in {
        "qwen2.5:1.5b-instruct-q4_K_M",
        "llama3.2:1b-instruct-q8_0",
    }
    assert hard.selected_model == "qwen2.5:3b-instruct-q4_K_M"
    assert len({simple.selected_model, medium.selected_model, hard.selected_model}) >= 2


def test_low_ram_forces_smaller_or_q4_fallback() -> None:
    router = AdaptiveRouter()
    # Usable budget only enough for the smallest Q4 1B (~1200MB).
    available = OS_SAFETY_BUFFER_MB + 1300
    result = router.route(
        prompt="Perform multi-step reasoning: derive and calculate a proof sketch "
        "with careful explanation of each logical step.",
        available_ram_mb=available,
        task_type="reasoning",
    )
    assert result.hardware_constrained is True
    assert result.selected_model == "llama3.2:1b-instruct-q4_K_M"
    assert result.quantization == "Q4_K_M"
    assert "downgraded" in result.rationale.lower() or "fallback" in result.rationale.lower()


def test_no_feasible_model_uses_smallest_forced_fallback() -> None:
    tiny_catalog = (
        ModelSpec(
            ollama_tag="huge:q8",
            family="huge",
            params_b=7.0,
            quantization="Q8_0",
            estimated_ram_mb=9000,
            complexity_tier="high",
        ),
        ModelSpec(
            ollama_tag="mid:q4",
            family="mid",
            params_b=3.0,
            quantization="Q4_K_M",
            estimated_ram_mb=5000,
            complexity_tier="mid",
        ),
    )
    router = AdaptiveRouter(catalog=tiny_catalog, safety_buffer_mb=3072)
    result = router.route(
        prompt="simple",
        available_ram_mb=3500,  # usable = 428 MB — nothing fits
        task_type="classification",
    )
    assert result.hardware_constrained is True
    assert result.selected_model == "mid:q4"
    assert "forced fallback" in result.rationale.lower()


def test_rationale_includes_budget_numbers() -> None:
    router = AdaptiveRouter()
    available = 8000.0
    result = router.route(
        prompt="Extract JSON fields from text",
        available_ram_mb=available,
        task_type="extraction",
    )
    assert "budget" in result.rationale.lower() or "usable" in result.rationale.lower()
    assert result.available_ram_mb == available
    assert result.usable_ram_mb == available - OS_SAFETY_BUFFER_MB


def test_route_prefers_installed_tags() -> None:
    router = AdaptiveRouter()
    available = OS_SAFETY_BUFFER_MB + 20_000
    installed = {"llama3.2:1b-instruct-q4_K_M"}
    result = router.route(
        prompt="Perform multi-step reasoning and calculate carefully.",
        available_ram_mb=available,
        task_type="reasoning",
        installed_tags=installed,
    )
    assert result.selected_model == "llama3.2:1b-instruct-q4_K_M"
    assert "not installed" in result.rationale.lower() or result.hardware_constrained
