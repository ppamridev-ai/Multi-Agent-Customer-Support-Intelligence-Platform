"""Build and persist the FAQ Chroma vector store."""

from __future__ import annotations

from langchain_chroma import Chroma

from src.config import (
    CHROMA_PERSIST_DIR,
    FAQ_COLLECTION_NAME,
)
from src.rag.documents import build_faq_documents
from src.rag.embeddings import get_embedding_model


def main() -> None:
    """Embed FAQ documents and save them in ChromaDB."""

    documents = build_faq_documents()
    if not documents:
        raise ValueError("No FAQ documents were available for indexing.")

    embedding_model = get_embedding_model()

    CHROMA_PERSIST_DIR.mkdir(parents=True,exist_ok=True)

    vector_store = Chroma(
        collection_name=FAQ_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_PERSIST_DIR),
        collection_metadata={
            "description": "Customer-support FAQ knowledge base",
            "hnsw:space": "cosine",
        },
    )

    document_ids = [document.metadata["faq_id"] for document in documents]

    vector_store.add_documents(documents=documents,ids=document_ids)

    print(f"FAQ documents indexed: {len(documents)}")
    print(f"Chroma collection: {FAQ_COLLECTION_NAME}")
    print(f"Chroma persistence directory: {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()