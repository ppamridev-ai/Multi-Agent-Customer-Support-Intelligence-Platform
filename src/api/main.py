"""FastAPI application for the support platform."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.tickets import router as tickets_router
from src.classification.service import get_classification_service
from src.rag.service import get_faq_retrieval_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load reusable ML and RAG services at startup."""

    get_classification_service()
    get_faq_retrieval_service()

    yield


app = FastAPI(
    title="Multi-Agent Customer Support Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(tickets_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return application health status."""

    return {
        "status": "healthy",
    }