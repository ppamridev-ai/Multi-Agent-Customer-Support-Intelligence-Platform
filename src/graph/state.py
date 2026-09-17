"""Shared state for the customer-support LangGraph."""

from __future__ import annotations
from typing import Any
from typing_extensions import NotRequired, TypedDict


class SupportGraphState(TypedDict):
    """State passed between customer-support graph nodes."""

    ticket_text: str

    classification: NotRequired[dict[str, Any]]
    retrieved_faqs: NotRequired[list[dict[str, Any]]]

    current_step: NotRequired[str]
    requires_review: NotRequired[bool]
    errors: NotRequired[list[str]]

    ticket_understanding: NotRequired[dict[str, Any]]

    specialist_name: NotRequired[str]
    specialist_resolution: NotRequired[dict[str, Any]]
    draft_response: NotRequired[dict[str, Any]]

    validation: NotRequired[dict[str, Any]]
    final_response: NotRequired[dict[str, Any]]

    human_review: NotRequired[dict[str, Any]]
    review_status: NotRequired[str]