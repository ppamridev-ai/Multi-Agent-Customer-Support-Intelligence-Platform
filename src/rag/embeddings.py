"""Embedding-model configuration for RAG."""

from __future__ import annotations

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and cache the local embedding model."""

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )