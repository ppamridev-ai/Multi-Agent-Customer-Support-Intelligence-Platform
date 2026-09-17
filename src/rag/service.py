"""FAQ retrieval service backed by ChromaDB."""

from __future__ import annotations
from functools import lru_cache
from langchain_chroma import Chroma

from src.config import (
    CHROMA_PERSIST_DIR,
    FAQ_COLLECTION_NAME,
    settings,
)
from src.rag.embeddings import get_embedding_model
from src.schemas import RetrievedFAQ


class FAQRetrievalService:
    """Retrieve relevant FAQ documents from ChromaDB."""

    def __init__(self) -> None:
        if not CHROMA_PERSIST_DIR.exists():
            raise FileNotFoundError(
                "Chroma database was not found at "
                f"{CHROMA_PERSIST_DIR}. Run "
                "`python -m src.rag.build_vector_store` first."
            )

        self.vector_store = Chroma(
            collection_name=FAQ_COLLECTION_NAME,
            embedding_function=get_embedding_model(),
            persist_directory=str(CHROMA_PERSIST_DIR),
        )

    def search(self,query: str,category: str | None = None,top_k: int | None = None) -> list[RetrievedFAQ]:
        """Return FAQs most relevant to a ticket."""

        normalized_query = " ".join(query.split())

        if len(normalized_query) < 5:
            raise ValueError("Retrieval query must contain at least 5 characters.")

        number_of_results = top_k or settings.rag_top_k

        metadata_filter = {"category": category} if category else None

        matches = (
            self.vector_store
            .similarity_search_with_relevance_scores(
                query=normalized_query,
                k=number_of_results,
                filter=metadata_filter,
            )
        )

        results: list[RetrievedFAQ] = []

        for document, score in matches:
            results.append(
                RetrievedFAQ(
                    faq_id=document.metadata["faq_id"],
                    question=document.metadata["question"],
                    answer=document.metadata["answer"],
                    category=document.metadata["category"],
                    similarity_score=float(score),
                )
            )

        return results


@lru_cache(maxsize=1)
def get_faq_retrieval_service() -> FAQRetrievalService:
    """Return one cached retrieval-service instance."""

    return FAQRetrievalService()