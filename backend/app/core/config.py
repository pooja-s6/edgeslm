"""Model registry, quantization tags, and hardware thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ComplexityTier = Literal["low", "mid", "high"]
QuantizationType = Literal["Q4_K_M", "Q8_0"]


@dataclass(frozen=True)
class ModelSpec:
    """Catalog entry for a supported SLM + quantization pair."""

    ollama_tag: str
    family: str
    params_b: float
    quantization: QuantizationType
    estimated_ram_mb: int
    complexity_tier: ComplexityTier


# Paths resolve relative to the repository root (parent of backend/).
BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
DATA_DIR = REPO_ROOT / "data"
BENCHMARK_SUITE_PATH = DATA_DIR / "benchmark_suite.json"
DATABASE_PATH = BACKEND_DIR / "edgeslm.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

OLLAMA_BASE_URL = "http://localhost:11434"
OS_SAFETY_BUFFER_MB = 3072
DEFAULT_COOLDOWN_SECONDS = 3.0
OLLAMA_REQUEST_TIMEOUT_S = 300.0

# Estimated RSS footprints for CPU GGUF runners (conservative).
MODEL_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        ollama_tag="llama3.2:1b-instruct-q4_K_M",
        family="llama3.2",
        params_b=1.0,
        quantization="Q4_K_M",
        estimated_ram_mb=1200,
        complexity_tier="low",
    ),
    ModelSpec(
        ollama_tag="llama3.2:1b-instruct-q8_0",
        family="llama3.2",
        params_b=1.0,
        quantization="Q8_0",
        estimated_ram_mb=1800,
        complexity_tier="low",
    ),
    ModelSpec(
        ollama_tag="qwen2.5:1.5b-instruct-q4_K_M",
        family="qwen2.5",
        params_b=1.5,
        quantization="Q4_K_M",
        estimated_ram_mb=1600,
        complexity_tier="mid",
    ),
    ModelSpec(
        ollama_tag="qwen2.5:1.5b-instruct-q8_0",
        family="qwen2.5",
        params_b=1.5,
        quantization="Q8_0",
        estimated_ram_mb=2400,
        complexity_tier="mid",
    ),
    ModelSpec(
        ollama_tag="qwen2.5:3b-instruct-q4_K_M",
        family="qwen2.5",
        params_b=3.0,
        quantization="Q4_K_M",
        estimated_ram_mb=2800,
        complexity_tier="high",
    ),
    ModelSpec(
        ollama_tag="qwen2.5:3b-instruct-q8_0",
        family="qwen2.5",
        params_b=3.0,
        quantization="Q8_0",
        estimated_ram_mb=4200,
        complexity_tier="high",
    ),
)

MODEL_BY_TAG: dict[str, ModelSpec] = {m.ollama_tag: m for m in MODEL_CATALOG}


def get_model_spec(ollama_tag: str) -> ModelSpec | None:
    """Return a catalog entry by Ollama tag, or None if unknown."""
    return MODEL_BY_TAG.get(ollama_tag)
