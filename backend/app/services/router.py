"""Explainable feature + resource-aware adaptive model router."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.core.config import (
    MODEL_CATALOG,
    OS_SAFETY_BUFFER_MB,
    ModelSpec,
)

TaskType = Literal["classification", "extraction", "summarization", "reasoning"]

REASONING_KEYWORDS = frozenset(
    {
        "reason",
        "reasoning",
        "step",
        "steps",
        "explain",
        "why",
        "because",
        "calculate",
        "compute",
        "solve",
        "multi-step",
        "chain",
        "logic",
        "prove",
        "derive",
        "processing",
        "batch",
        "precision",
    }
)

STRUCTURAL_KEYWORDS = frozenset(
    {
        "json",
        "extract",
        "schema",
        "fields",
        "format",
        "structured",
        "key",
        "value",
        "object",
        "company",
        "role",
        "location",
    }
)

TASK_BASE_COMPLEXITY: dict[str, float] = {
    "classification": 0.12,
    "extraction": 0.42,
    "summarization": 0.55,
    "reasoning": 0.88,
}


@dataclass(frozen=True)
class RoutingResult:
    selected_model: str
    quantization: str
    complexity_score: float
    hardware_constrained: bool
    rationale: str
    available_ram_mb: float
    usable_ram_mb: float


def estimate_token_count(prompt: str) -> int:
    """Rough token estimate (~4 chars/token) without a tokenizer dependency."""
    return max(1, len(prompt) // 4)


def compute_complexity_score(
    prompt: str,
    task_type: TaskType | str | None = None,
) -> float:
    """
    Deterministic complexity score in [0, 1] from length, keywords, and task type.
    """
    tokens = estimate_token_count(prompt)
    words = re.findall(r"[A-Za-z0-9_]+", prompt.lower())
    unique_ratio = (len(set(words)) / len(words)) if words else 0.0

    length_component = min(1.0, tokens / 220.0)
    density_component = min(1.0, unique_ratio)

    word_set = set(words)
    reasoning_hits = len(word_set & REASONING_KEYWORDS)
    structural_hits = len(word_set & STRUCTURAL_KEYWORDS)
    keyword_component = min(1.0, 0.18 * reasoning_hits + 0.10 * structural_hits)

    inferred = task_type
    if not inferred:
        if reasoning_hits >= 2 or any(w in word_set for w in ("calculate", "compute", "solve")):
            inferred = "reasoning"
        elif structural_hits >= 2 or "json" in word_set:
            inferred = "extraction"
        elif "classify" in word_set or "sentiment" in word_set:
            inferred = "classification"

    task_component = TASK_BASE_COMPLEXITY.get(str(inferred or ""), 0.35)

    score = (
        0.30 * length_component
        + 0.10 * density_component
        + 0.25 * keyword_component
        + 0.35 * task_component
    )
    return round(max(0.0, min(1.0, score)), 4)


def _tier_for_complexity(complexity: float) -> str:
    if complexity < 0.34:
        return "low"
    if complexity < 0.58:
        return "mid"
    return "high"


def _preference_key(spec: ModelSpec, target_tier: str) -> tuple:
    """
    Higher is better.

    Policy:
    - low  → smallest efficient model (prefer Q4 / 1B)
    - mid  → mid-tier or higher-precision small models (1.5B / 1B-Q8)
    - high → largest available capacity (prefer 3B-Q4 on edge)
    """
    tier_rank = {"low": 0, "mid": 1, "high": 2}
    tier_match = 1 if spec.complexity_tier == target_tier else 0
    tier_distance = -abs(tier_rank[spec.complexity_tier] - tier_rank[target_tier])

    if target_tier == "low":
        quant_eff = 1 if spec.quantization == "Q4_K_M" else 0
        return (tier_match, tier_distance, quant_eff, -spec.params_b)

    if target_tier == "mid":
        # Prefer true mid-tier; otherwise lean toward Q8 small models.
        quant_quality = 1 if spec.quantization == "Q8_0" else 0
        return (tier_match, tier_distance, spec.params_b, quant_quality)

    quant_eff = 1 if spec.quantization == "Q4_K_M" else 0
    return (tier_match, tier_distance, spec.params_b, quant_eff)


def _fits_budget(spec: ModelSpec, usable_ram_mb: float) -> bool:
    return spec.estimated_ram_mb <= usable_ram_mb


class AdaptiveRouter:
    """Select an SLM + quantization pair from complexity and live RAM budget."""

    def __init__(
        self,
        catalog: tuple[ModelSpec, ...] | list[ModelSpec] = MODEL_CATALOG,
        safety_buffer_mb: int = OS_SAFETY_BUFFER_MB,
    ) -> None:
        self.catalog = tuple(catalog)
        self.safety_buffer_mb = safety_buffer_mb

    def route(
        self,
        prompt: str,
        available_ram_mb: float,
        task_type: TaskType | str | None = None,
        installed_tags: set[str] | frozenset[str] | None = None,
    ) -> RoutingResult:
        complexity = compute_complexity_score(prompt, task_type)
        target_tier = _tier_for_complexity(complexity)
        usable_ram_mb = max(0.0, float(available_ram_mb) - float(self.safety_buffer_mb))

        full_catalog = sorted(
            self.catalog,
            key=lambda s: _preference_key(s, target_tier),
            reverse=True,
        )
        ideal = full_catalog[0] if full_catalog else None

        if installed_tags:
            pool = [s for s in full_catalog if s.ollama_tag in installed_tags]
            install_note = ""
            if not pool:
                pool = list(full_catalog)
                install_note = (
                    " No catalog models are installed locally; "
                    "routing against the full registry."
                )
        else:
            pool = list(full_catalog)
            install_note = ""

        feasible = [s for s in pool if _fits_budget(s, usable_ram_mb)]
        hardware_constrained = False

        if feasible:
            selected = feasible[0]
            if ideal is not None and selected.ollama_tag != ideal.ollama_tag:
                hardware_constrained = True
                if installed_tags and selected.ollama_tag in installed_tags and (
                    ideal.ollama_tag not in installed_tags
                ):
                    rationale = (
                        f"Complexity {complexity:.2f} targets {target_tier}-tier "
                        f"(ideal {ideal.ollama_tag}), but that tag is not installed. "
                        f"Using installed {selected.ollama_tag} within usable "
                        f"{usable_ram_mb:.0f}MB budget."
                    )
                else:
                    rationale = (
                        f"Complexity {complexity:.2f} targets {target_tier}-tier; "
                        f"ideal {ideal.ollama_tag} needs ~{ideal.estimated_ram_mb}MB but "
                        f"usable RAM is {usable_ram_mb:.0f}MB "
                        f"(available {available_ram_mb:.0f}MB minus "
                        f"{self.safety_buffer_mb}MB OS buffer). "
                        f"Downgraded to {selected.ollama_tag}."
                    )
            else:
                policy = {
                    "low": "efficiency-first (1B-Q4)",
                    "mid": "balanced structured quality (1.5B / 1B-Q8)",
                    "high": "capacity-first reasoning (3B-Q4)",
                }[target_tier]
                rationale = (
                    f"Complexity {complexity:.2f} maps to {target_tier}-tier "
                    f"({policy}); selected {selected.ollama_tag} "
                    f"(est. {selected.estimated_ram_mb}MB) within usable "
                    f"{usable_ram_mb:.0f}MB budget."
                )
            rationale = rationale + install_note
        else:
            fallback_pool = pool or list(self.catalog)
            selected = min(fallback_pool, key=lambda s: (s.estimated_ram_mb, s.params_b))
            hardware_constrained = True
            rationale = (
                f"No catalog model fits usable RAM {usable_ram_mb:.0f}MB "
                f"(available {available_ram_mb:.0f}MB minus {self.safety_buffer_mb}MB buffer). "
                f"Forced fallback to smallest model {selected.ollama_tag}."
                + install_note
            )

        return RoutingResult(
            selected_model=selected.ollama_tag,
            quantization=selected.quantization,
            complexity_score=complexity,
            hardware_constrained=hardware_constrained,
            rationale=rationale,
            available_ram_mb=float(available_ram_mb),
            usable_ram_mb=usable_ram_mb,
        )
