"""Streamlit UI for the customer-support platform."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL","http://127.0.0.1:8000")


def call_api(method: str, endpoint: str, payload: dict | None = None) -> dict[str, Any]:
    """Call the FastAPI backend."""

    url = f"{API_BASE_URL}{endpoint}"

    response = requests.request(
        method=method,
        url=url,
        json=payload,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"API request failed ({response.status_code}): "
            f"{response.text}"
        )

    return response.json()


def display_classification(classification: dict) -> None:
    """Display ML predictions and confidence scores."""

    st.subheader("Classification")

    first, second, third = st.columns(3)

    first.metric(
        "Category",
        classification.get("category", "Unknown"),
        f"{classification.get('category_confidence', 0):.1%}",
    )

    second.metric(
        "Priority",
        classification.get("priority", "Unknown"),
        f"{classification.get('priority_confidence', 0):.1%}",
    )

    third.metric(
        "Sentiment",
        classification.get("sentiment", "Unknown"),
        f"{classification.get('sentiment_confidence', 0):.1%}",
    )

    low_confidence = classification.get(
        "low_confidence_targets",
        [],
    )

    if low_confidence:
        st.warning(
            "Low-confidence predictions: "
            + ", ".join(low_confidence)
        )


def display_faqs(faqs: list[dict]) -> None:
    """Display retrieved FAQ evidence."""

    st.subheader("Retrieved FAQs")

    for faq in faqs:
        with st.expander(
            f"{faq['faq_id']} — {faq['question']}"
        ):
            st.write(faq["answer"])
            st.caption(
                f"Category: {faq['category']} | "
                f"Similarity: "
                f"{faq.get('similarity_score', 0):.3f}"
            )


def display_workflow(data: dict[str, Any]) -> None:
    """Display the current workflow state."""

    state = data.get("state", {})

    st.info(f"Workflow status: {data.get('status')}")
    st.caption(f"Thread ID: {data.get('thread_id')}")

    understanding = state.get("ticket_understanding")

    if understanding:
        st.subheader("Ticket Understanding")
        st.json(understanding)

    classification = state.get("classification")

    if classification:
        display_classification(classification)

    faqs = state.get("retrieved_faqs", [])

    if faqs:
        display_faqs(faqs)

    specialist = state.get("specialist_resolution")

    if specialist:
        st.subheader("Specialist Resolution")
        st.json(specialist)

    draft = state.get("draft_response")

    if draft:
        st.subheader("Draft Response")
        st.markdown(f"**{draft['subject']}**")
        st.write(draft["message"])

    validation = state.get("validation")

    if validation:
        st.subheader("Validation")
        st.json(validation)

    final_response = state.get("final_response")

    if final_response:
        st.success("Final response")

        st.markdown(
            f"**{final_response.get('subject', '')}**"
        )
        st.write(
            final_response.get("message", "")
        )


def submit_review(
    action: str,
    subject: str | None = None,
    message: str | None = None,
    comment: str | None = None,
) -> None:
    """Submit a human-review decision."""

    workflow = st.session_state.get("workflow")

    if not workflow:
        st.error("No workflow is available.")
        return

    thread_id = workflow["thread_id"]

    payload: dict[str, Any] = {
        "action": action,
        "reviewer_comment": comment,
    }

    if action == "edit":
        payload["subject"] = subject
        payload["message"] = message

    with st.spinner("Submitting review decision..."):
        result = call_api(
            method="POST",
            endpoint=f"/tickets/{thread_id}/review",
            payload=payload,
        )

    st.session_state.workflow = result
    st.rerun()


def main() -> None:
    """Render the Streamlit application."""

    st.set_page_config(
        page_title="Customer Support Intelligence",
        page_icon="🎧",
        layout="wide",
    )

    st.title("Multi-Agent Customer Support")
    st.caption(
        "LangGraph · ML Classification · ChromaDB · "
        "Human-in-the-Loop"
    )

    ticket_text = st.text_area(
        "Customer ticket",
        height=150,
        placeholder=(
            "Describe the customer's issue..."
        ),
    )

    if st.button(
        "Process Ticket",
        type="primary",
        use_container_width=True,
    ):
        if len(ticket_text.strip()) < 5:
            st.warning(
                "Please enter at least five characters."
            )
        else:
            try:
                with st.spinner(
                    "Running the multi-agent workflow..."
                ):
                    result = call_api(
                        method="POST",
                        endpoint="/tickets/process",
                        payload={
                            "ticket_text": ticket_text,
                        },
                    )

                st.session_state.workflow = result

            except Exception as error:
                st.error(str(error))

    workflow = st.session_state.get("workflow")

    if not workflow:
        return

    st.divider()
    display_workflow(workflow)

    if workflow.get("status") != (
        "human_review_required"
    ):
        return

    state = workflow.get("state", {})
    draft = state.get("draft_response", {})

    st.divider()
    st.header("Human Review")

    reviewer_comment = st.text_input(
        "Reviewer comment",
    )

    first, second, third = st.columns(3)

    with first:
        if st.button(
            "Approve",
            use_container_width=True,
        ):
            try:
                submit_review(
                    action="approve",
                    comment=reviewer_comment,
                )
            except Exception as error:
                st.error(str(error))

    with second:
        if st.button(
            "Reject",
            use_container_width=True,
        ):
            try:
                submit_review(
                    action="reject",
                    comment=reviewer_comment,
                )
            except Exception as error:
                st.error(str(error))

    with third:
        edit_enabled = st.checkbox(
            "Edit response",
        )

    if edit_enabled:
        edited_subject = st.text_input(
            "Edited subject",
            value=draft.get("subject", ""),
        )

        edited_message = st.text_area(
            "Edited message",
            value=draft.get("message", ""),
            height=200,
        )

        if st.button(
            "Save Edited Response",
            use_container_width=True,
        ):
            try:
                submit_review(
                    action="edit",
                    subject=edited_subject,
                    message=edited_message,
                    comment=reviewer_comment,
                )
            except Exception as error:
                st.error(str(error))


if __name__ == "__main__":
    main()