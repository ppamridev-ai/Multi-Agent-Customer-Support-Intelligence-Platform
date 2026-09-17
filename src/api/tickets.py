"""FastAPI routes for ticket-processing workflows."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from langgraph.types import Command

from src.graph.workflow import support_graph
from src.schemas import (
    HumanReviewRequest,
    TicketProcessRequest,
    WorkflowResponse,
)


router = APIRouter(prefix="/tickets",tags=["tickets"])


def create_graph_config(thread_id: str) -> dict:
    """Create LangGraph configuration for one workflow."""

    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def extract_interrupt(result: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the JSON-safe interrupt payload."""

    interrupts = result.get("__interrupt__", ())

    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    return first_interrupt.value


def remove_internal_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Remove LangGraph internal fields from API output."""

    return {
        key: value
        for key, value in result.items()
        if not key.startswith("__")
    }


@router.post("/process",response_model=WorkflowResponse)
def process_ticket(request: TicketProcessRequest) -> WorkflowResponse:
    """Start a new customer-support workflow."""

    thread_id = str(uuid.uuid4())
    config = create_graph_config(thread_id)

    initial_state = {
        "ticket_text": request.ticket_text,
        "errors": [],
    }

    try:
        result = support_graph.invoke(
            initial_state,
            config=config,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Ticket processing failed: {error}",
        ) from error

    review_request = extract_interrupt(result)
    state = remove_internal_fields(result)

    if review_request is not None:
        status = "human_review_required"
    else:
        status = "completed"

    return WorkflowResponse(
        thread_id=thread_id,
        status=status,
        state=state,
        review_request=review_request,
    )


@router.post("/{thread_id}/review",response_model=WorkflowResponse)
def review_ticket(thread_id: str,request: HumanReviewRequest) -> WorkflowResponse:
    """Resume a workflow with a human-review decision."""

    if request.action == "edit" and not request.message:
        raise HTTPException(
            status_code=422,
            detail=(
                "A message is required when the review "
                "action is 'edit'."
            ),
        )

    config = create_graph_config(thread_id)
    snapshot = support_graph.get_state(config)

    if not snapshot.values:
        raise HTTPException(
            status_code=404,
            detail="Workflow thread was not found.",
        )

    has_pending_interrupt = any(
        task.interrupts
        for task in snapshot.tasks
    )

    if not has_pending_interrupt:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "This workflow is not waiting for "
                    "human review."
                ),
                "current_step": snapshot.values.get("current_step"),
                "review_status": snapshot.values.get("review_status"),
                "next_nodes": list(snapshot.next),
            },
        )

    decision = request.model_dump(
        exclude_none=True
    )

    try:
        result = support_graph.invoke(
            Command(resume=decision),
            config=config,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow resume failed: {error}",
        ) from error

    state = remove_internal_fields(result)

    review_status = state.get(
        "review_status",
        request.action,
    )

    return WorkflowResponse(
        thread_id=thread_id,
        status=review_status,
        state=state,
        review_request=None,
    )


@router.get(
    "/{thread_id}/state",
    response_model=WorkflowResponse,
)
def get_ticket_state(
    thread_id: str,
) -> WorkflowResponse:
    """Return the current state of a workflow."""

    config = create_graph_config(thread_id)
    snapshot = support_graph.get_state(config)

    if not snapshot.values:
        raise HTTPException(
            status_code=404,
            detail="Workflow thread was not found.",
        )

    state = dict(snapshot.values)

    has_pending_interrupt = any(
        task.interrupts
        for task in snapshot.tasks
    )

    if has_pending_interrupt:
        status = "human_review_required"
    else:
        status = state.get("review_status","completed")

    state["next_nodes"] = list(snapshot.next)
    state["has_pending_interrupt"] = has_pending_interrupt

    return WorkflowResponse(
        thread_id=thread_id,
        status=status,
        state=state,
        review_request=None,
    )