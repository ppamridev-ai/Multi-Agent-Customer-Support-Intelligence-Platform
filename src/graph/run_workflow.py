"""Run and review the customer-support LangGraph locally."""

from __future__ import annotations

import json
import uuid

from langgraph.types import Command

from src.graph.workflow import support_graph


def print_result(result: dict) -> None:
    """Print a graph result as formatted JSON."""

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )


def collect_review_decision() -> dict:
    """Collect a reviewer decision from the terminal."""

    print("\nHuman review required.")
    print("Available actions: approve, edit, reject")

    action = input("Action: ").strip().lower()

    if action == "approve":
        return {
            "action": "approve",
            "reviewer_comment": (
                "Approved through the local review CLI."
            ),
        }

    if action == "edit":
        subject = input(
            "Edited subject (leave blank to keep existing): "
        ).strip()

        message = input("Edited message: ").strip()

        decision = {
            "action": "edit",
            "message": message,
            "reviewer_comment": (
                "Edited through the local review CLI."
            ),
        }

        if subject:
            decision["subject"] = subject

        return decision

    if action == "reject":
        reason = input("Reason for rejection: ").strip()

        return {
            "action": "reject",
            "reviewer_comment": reason,
        }

    raise ValueError(
        "Action must be approve, edit, or reject."
    )


def main() -> None:
    """Run one ticket and handle a possible interruption."""

    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    initial_state = {
        "ticket_text": (
            "My order has not arrived and the tracking "
            "has not updated for five days."
        ),
        "errors": [],
    }

    result = support_graph.invoke(
        initial_state,
        config=config,
    )

    print_result(result)

    if "__interrupt__" not in result:
        print("\nResponse was finalized automatically.")
        return

    decision = collect_review_decision()

    resumed_result = support_graph.invoke(
        Command(resume=decision),
        config=config,
    )

    print("\nResumed workflow result:")
    print_result(resumed_result)


if __name__ == "__main__":
    main()