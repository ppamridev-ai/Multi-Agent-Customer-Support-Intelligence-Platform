"""Nodes used by the customer-support LangGraph."""

from __future__ import annotations

from src.classification.service import (
    get_classification_service,
)
from src.graph.state import SupportGraphState
from src.rag.service import get_faq_retrieval_service
from src.agents.ticket_understanding import understand_ticket
from src.agents.specialists import run_specialist
from src.agents.response_generator import generate_response
from src.agents.validator import validate_response
from langgraph.types import interrupt


def classification_node(state: SupportGraphState) -> dict:
    """Predict ticket category, priority, and sentiment."""

    ticket_text = state["ticket_text"]
    service = get_classification_service()
    prediction = service.predict(ticket_text)

    return {
        "classification": prediction.model_dump(mode="json"),
        "requires_review": prediction.requires_review,
        "current_step": "classification_completed",
    }


def retrieval_node(state: SupportGraphState) -> dict:
    """Retrieve category-specific FAQs for the ticket."""

    ticket_text = state["ticket_text"]
    classification = state["classification"]
    predicted_category = classification["category"]

    service = get_faq_retrieval_service()
    results = service.search(query=ticket_text,category=predicted_category)

    return {
        "retrieved_faqs": [
            result.model_dump(mode="json")
            for result in results
        ],
        "current_step": "retrieval_completed",
    }

def ticket_understanding_node(state: SupportGraphState) -> dict:
    """Extract structured meaning from the ticket."""

    understanding = understand_ticket(state["ticket_text"])
    return {
        "ticket_understanding": understanding.model_dump(mode="json"),
        "current_step": "ticket_understanding_completed",
    }   

def _run_specialist_node(state: SupportGraphState, category: str) -> dict:
    """Run one category-specific specialist."""

    resolution = run_specialist(
        category=category,
        ticket_text=state["ticket_text"],
        ticket_understanding=state["ticket_understanding"],
        classification=state["classification"],
        retrieved_faqs=state.get("retrieved_faqs", []),
    )

    requires_review = state.get("requires_review", False) or resolution.requires_human_review

    return {
        "specialist_name": category,
        "specialist_resolution": resolution.model_dump(mode="json"),
        "requires_review": requires_review,
        "current_step": "specialist_completed",
    }


def delivery_specialist_node(state: SupportGraphState) -> dict:
    return _run_specialist_node(state, "Delivery Issue")


def refund_specialist_node(state: SupportGraphState) -> dict:
    return _run_specialist_node(state, "Refund & Return")


def payment_specialist_node(state: SupportGraphState) -> dict:
    return _run_specialist_node(state, "Payment Issue")


def product_specialist_node(state: SupportGraphState) -> dict:
    return _run_specialist_node(state, "Product Issue")


def account_specialist_node(state: SupportGraphState) -> dict:
    return _run_specialist_node(state, "Account & Login")


def app_specialist_node(state: SupportGraphState) -> dict:
    return _run_specialist_node(state, "App & Website Issue")


def seller_specialist_node(state: SupportGraphState) -> dict:
    return _run_specialist_node(state, "Seller & Product Listing")

def route_to_specialist(state: SupportGraphState) -> str:
    """Select a specialist using the predicted category."""

    category = state["classification"]["category"]

    routes = {
        "Delivery Issue": "delivery_specialist",
        "Refund & Return": "refund_specialist",
        "Payment Issue": "payment_specialist",
        "Product Issue": "product_specialist",
        "Account & Login": "account_specialist",
        "App & Website Issue": "app_specialist",
        "Seller & Product Listing": "seller_specialist",
    }

    try:
        return routes[category]
    except KeyError as error:
        raise ValueError(
            f"No specialist route for category: {category}"
        ) from error

def response_generation_node(state: SupportGraphState) -> dict:
    """Create a customer-facing response draft."""

    response = generate_response(
        ticket_text=state["ticket_text"],
        ticket_understanding=state["ticket_understanding"],
        classification=state["classification"],
        specialist_resolution=state["specialist_resolution"],
        retrieved_faqs=state.get("retrieved_faqs", []),
    )

    requires_review = state.get("requires_review", False) or response.contains_unverified_action

    return {
        "draft_response": response.model_dump(mode="json"),
        "requires_review": requires_review,
        "current_step": "response_generated",
    }

def validation_node(state: SupportGraphState) -> dict:
    """Validate the generated customer response."""

    validation = validate_response(
        ticket_text=state["ticket_text"],
        ticket_understanding=state["ticket_understanding"],
        classification=state["classification"],
        retrieved_faqs=state.get("retrieved_faqs", []),
        specialist_resolution=state["specialist_resolution"],
        draft_response=state["draft_response"],
    )

    requires_review = state.get("requires_review", False) or not validation.safe_to_send

    return {
        "validation": validation.model_dump(mode="json"),
        "requires_review": requires_review,
        "current_step": "validation_completed",
    }


def finalize_response_node(state: SupportGraphState) -> dict:
    """Mark an approved draft as the final response."""

    return {
        "final_response": state["draft_response"],
        "current_step": "response_finalized",
    }

def route_after_validation(state: SupportGraphState) -> str:
    """Route the response to automatic or human approval."""

    validation = state["validation"]

    if state.get("requires_review", False):
        return "human_review"

    if not validation["safe_to_send"]:
        return "human_review"

    return "finalize_response"

def human_review_node(state: SupportGraphState) -> dict:
    """Pause execution until a human reviews the response."""

    review_decision = interrupt(
        {
            "type": "customer_support_review",
            "ticket_text": state["ticket_text"],
            "ticket_understanding": state["ticket_understanding"],
            "classification": state["classification"],
            "specialist_resolution": state["specialist_resolution"],
            "draft_response": state["draft_response"],
            "validation": state["validation"],
            "review_reasons": {
                "low_confidence_targets": state["classification"].get("low_confidence_targets", []),
                "validation_issues": state["validation"].get("issues", []),
                "escalation_reason": state["specialist_resolution"].get("escalation_reason"),
            },
            "allowed_actions": ["approve", "edit", "reject"],
        }
    )

    if not isinstance(review_decision, dict):
        raise ValueError("Human-review decision must be a dictionary.")

    action = review_decision.get("action")

    if action == "approve":
        return {
            "human_review": review_decision,
            "review_status": "approved",
            "final_response": state["draft_response"],
            "current_step": "human_approved",
        }

    if action == "edit":
        edited_subject = review_decision.get(
            "subject",
            state["draft_response"]["subject"],
        )

        edited_message = review_decision.get("message")

        if not edited_message:
            raise ValueError(
                "An edited message is required when "
                "action is 'edit'."
            )

        edited_response = {
            **state["draft_response"],
            "subject": edited_subject,
            "message": edited_message,
        }

        return {
            "human_review": review_decision,
            "review_status": "edited",
            "final_response": edited_response,
            "current_step": "human_edited",
        }

    if action == "reject":
        return {
            "human_review": review_decision,
            "review_status": "rejected",
            "final_response": {},
            "current_step": "human_rejected",
        }

    raise ValueError(
        "Human-review action must be "
        "'approve', 'edit', or 'reject'."
    )