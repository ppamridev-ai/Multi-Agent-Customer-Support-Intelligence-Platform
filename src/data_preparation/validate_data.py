"""Validate raw ticket and FAQ records using Pydantic schemas."""

import pandas as pd
from pydantic import ValidationError

from src.config import FAQS_RAW_PATH, TICKETS_RAW_PATH
from src.schemas import FAQRecord, TicketRecord


def validate_frame(df: pd.DataFrame,schema: type[TicketRecord] | type[FAQRecord],dataset_name: str) -> list[dict]:
    """Validate every row and return details about invalid records."""

    errors: list[dict] = []

    for row_number, record in enumerate(df.to_dict(orient="records"), start=2):
        try:
            schema.model_validate(record)
        except ValidationError as error:
            errors.append(
                {
                    "dataset": dataset_name,
                    "csv_row": row_number,
                    "record_id": (
                        record.get("ticket_id")
                        or record.get("faq_id")
                    ),
                    "errors": error.errors(),
                }
            )

    return errors


def main() -> None:
    tickets = pd.read_csv(TICKETS_RAW_PATH)
    faqs = pd.read_csv(FAQS_RAW_PATH)

    ticket_errors = validate_frame(
        df=tickets,
        schema=TicketRecord,
        dataset_name="support_tickets",
    )

    faq_errors = validate_frame(
        df=faqs,
        schema=FAQRecord,
        dataset_name="faqs",
    )

    errors = ticket_errors + faq_errors

    print(f"Ticket records checked: {len(tickets):,}")
    print(f"FAQ records checked: {len(faqs):,}")
    print(f"Invalid records: {len(errors):,}")

    if errors:
        print("\nFirst five validation errors:")

        for error in errors[:5]:
            print(error)
    else:
        print("\nAll records passed schema validation.")


if __name__ == "__main__":
    main()