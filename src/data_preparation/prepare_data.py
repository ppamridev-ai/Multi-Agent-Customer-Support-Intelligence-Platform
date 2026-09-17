"""Prepare ticket and FAQ datasets without modifying raw files."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    FAQS_PROCESSED_PATH,
    FAQS_RAW_PATH,
    PROCESSED_DATA_DIR,
    TICKETS_PROCESSED_PATH,
    TICKETS_RAW_PATH,
)


TEXT_COLUMNS = [
    "ticket_text",
    "product_name",
    "product_segment",
    "ticket_category",
    "priority",
    "sentiment",
    "escalated",
    "resolution_text",
    "auto_resolved",
]

FAQ_TEXT_COLUMNS = [
    "faq_id",
    "question",
    "answer",
    "category",
]

ORDER_ID_PATTERN = r"^#[A-Z0-9]{10}$" #Example 

def normalize_whitespace(value: object) -> str:
    """Trim text and replace repeated whitespace with one space."""

    if pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()

def extract_order_id(ticket_text: str) -> str | None:
    """Extract the first order ID from a ticket."""

    match = re.search(ORDER_ID_PATTERN, ticket_text, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(0).upper()

def prepare_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Clean tickets and create a processed training dataset."""

    prepared = df.copy()

    for column in TEXT_COLUMNS:
        prepared[column] = prepared[column].map(normalize_whitespace)
    prepared["order_id"] = prepared["ticket_text"].map(extract_order_id)
    prepared["ticket_id"] = prepared["ticket_id"].astype(str).str.strip().str.upper()
    prepared["created_date"] = pd.to_datetime(prepared["created_date"], errors="raise")
    prepared["resolved_date"] = pd.to_datetime(prepared["resolved_date"], errors="raise")
    prepared["confidence_score"] = pd.to_numeric(prepared["confidence_score"], errors="raise")
    prepared["resolution_days"] = pd.to_numeric(prepared["resolution_days"], errors="raise").astype(int)
    prepared["customer_satisfaction_score"] = pd.to_numeric(prepared["customer_satisfaction_score"], errors="raise").astype(int)
    prepared = prepared.drop_duplicates(subset=["ticket_id"], keep="first")
    prepared = prepared.sort_values(by=["created_date", "ticket_id"]).reset_index(drop=True)
    return prepared


def prepare_faqs(df: pd.DataFrame) -> pd.DataFrame:
    """Clean FAQs and create text used for embedding."""

    prepared = df.copy()

    for column in FAQ_TEXT_COLUMNS:
        prepared[column] = prepared[column].map(normalize_whitespace)

    prepared["faq_id"] = prepared["faq_id"].str.upper()
    prepared = prepared.drop_duplicates(subset=["faq_id"], keep="first")
    prepared = prepared.drop_duplicates(subset=["question"], keep="first")
    prepared["document_text"] = "Question: " + prepared["question"] + "\nAnswer: " + prepared["answer"]
    prepared = prepared.sort_values(by="faq_id").reset_index(drop=True)
    return prepared


def build_quality_report(
    raw_tickets: pd.DataFrame,
    processed_tickets: pd.DataFrame,
    raw_faqs: pd.DataFrame,
    processed_faqs: pd.DataFrame,
) -> dict:
    """Create a summary of preprocessing results."""

    calculated_resolution_days = (
        processed_tickets["resolved_date"]
        - processed_tickets["created_date"]
    ).dt.days

    resolution_day_mismatches = int(
        (
            calculated_resolution_days
            != processed_tickets["resolution_days"]
        ).sum()
    )

    return {
        "tickets": {
            "raw_rows": len(raw_tickets),
            "processed_rows": len(processed_tickets),
            "removed_rows": (
                len(raw_tickets) - len(processed_tickets)
            ),
            "missing_values": int(
                processed_tickets.isna().sum().sum()
            ),
            "unique_ticket_ids": int(
                processed_tickets["ticket_id"].nunique()
            ),
            "resolution_day_mismatches": (
                resolution_day_mismatches
            ),
            "category_distribution": (
                processed_tickets["ticket_category"]
                .value_counts()
                .to_dict()
            ),
        },
        "faqs": {
            "raw_rows": len(raw_faqs),
            "processed_rows": len(processed_faqs),
            "removed_rows": len(raw_faqs) - len(processed_faqs),
            "missing_values": int(
                processed_faqs.isna().sum().sum()
            ),
            "unique_faq_ids": int(
                processed_faqs["faq_id"].nunique()
            ),
            "category_distribution": (
                processed_faqs["category"]
                .value_counts()
                .to_dict()
            ),
        },
    }


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary as formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_tickets = pd.read_csv(TICKETS_RAW_PATH)
    raw_faqs = pd.read_csv(FAQS_RAW_PATH)

    processed_tickets = prepare_tickets(raw_tickets)
    processed_faqs = prepare_faqs(raw_faqs)

    processed_tickets.to_csv(TICKETS_PROCESSED_PATH,index=False,date_format="%Y-%m-%d")
    processed_faqs.to_csv(FAQS_PROCESSED_PATH,index=False)

    report = build_quality_report(
        raw_tickets=raw_tickets,
        processed_tickets=processed_tickets,
        raw_faqs=raw_faqs,
        processed_faqs=processed_faqs,
    )

    report_path = ARTIFACTS_DIR / "data_quality_report.json"
    save_json(report, report_path)

    print(
        f"Processed tickets: {len(processed_tickets):,} "
        f"-> {TICKETS_PROCESSED_PATH}"
    )
    print(
        f"Processed FAQs: {len(processed_faqs):,} "
        f"-> {FAQS_PROCESSED_PATH}"
    )
    print(f"Quality report: {report_path}")
    print(
        "Resolution-day mismatches found: "
        f"{report['tickets']['resolution_day_mismatches']}"
    )


if __name__ == "__main__":
    main()