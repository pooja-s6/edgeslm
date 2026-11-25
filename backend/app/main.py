"""FastAPI application entrypoint for EdgeSLM Lab."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.database import init_db
from app.services.ollama_client import OllamaClient

logger = logging.getLogger("edgeslm")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    client = OllamaClient()
    try:
        reachable = await client.health_check()
        if reachable:
            logger.info("Ollama health check succeeded")
        else:
            logger.warning(
                "Ollama is not reachable at startup; API will start in degraded mode"
            )
    finally:
        await client.aclose()
    yield


app = FastAPI(
    title="EdgeSLM Lab",
    description=(
        "Resource-aware SLM benchmarking and adaptive inference "
        "on CPU-constrained edge hardware."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "EdgeSLM Lab",
        "docs": "/docs",
        "health": "/api/health",
    }
