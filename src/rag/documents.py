"""Build LangChain documents from processed FAQs."""

from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document

from src.config import FAQS_PROCESSED_PATH


def build_faq_documents(faqs: pd.DataFrame | None = None) -> list[Document]:
    """Convert each FAQ row into one LangChain document."""

    df = faqs.copy() if faqs is not None else pd.read_csv(FAQS_PROCESSED_PATH)
    required_columns = {"faq_id", "question", "answer", "category", "document_text"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "FAQ dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    documents: list[Document] = []

    for row in df.to_dict(orient="records"):
        document = Document(
            page_content=row["document_text"],
            metadata={
                "faq_id": row["faq_id"],
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
                "source": "faq",
            },
        )

        documents.append(document)

    return documents