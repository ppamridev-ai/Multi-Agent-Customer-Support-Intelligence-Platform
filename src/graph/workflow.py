"""LangGraph workflow for customer-support tickets."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    classification_node,
    retrieval_node,
    ticket_understanding_node,
    account_specialist_node,
    app_specialist_node,
    classification_node,
    delivery_specialist_node,
    payment_specialist_node,
    product_specialist_node,
    refund_specialist_node,
    retrieval_node,
    route_to_specialist,
    seller_specialist_node,
    ticket_understanding_node,
    response_generation_node,
    finalize_response_node,
    validation_node,
    human_review_node,
    route_after_validation
)
from src.graph.state import SupportGraphState
from langgraph.checkpoint.memory import InMemorySaver


def build_support_graph():
    """Build and compile the customer-support graph."""

    builder = StateGraph(SupportGraphState)

    builder.add_node("classify_ticket",classification_node)
    builder.add_node("retrieve_faqs",retrieval_node)
    builder.add_node("understand_ticket",ticket_understanding_node)
    builder.add_node("delivery_specialist",delivery_specialist_node)
    builder.add_node("refund_specialist",refund_specialist_node)
    builder.add_node("payment_specialist",payment_specialist_node)
    builder.add_node("product_specialist",product_specialist_node)
    builder.add_node("account_specialist",account_specialist_node)
    builder.add_node("app_specialist",app_specialist_node)
    builder.add_node("seller_specialist",seller_specialist_node)
    builder.add_node("generate_response",response_generation_node)
    builder.add_node("validate_response",validation_node)
    builder.add_node("finalize_response",finalize_response_node)
    builder.add_node("human_review",human_review_node)

    builder.add_edge(START,"understand_ticket")
    builder.add_edge("understand_ticket","classify_ticket")
    builder.add_edge("classify_ticket","retrieve_faqs")
    builder.add_conditional_edges("retrieve_faqs",
        route_to_specialist,
        {
            "delivery_specialist": "delivery_specialist",
            "refund_specialist": "refund_specialist",
            "payment_specialist": "payment_specialist",
            "product_specialist": "product_specialist",
            "account_specialist": "account_specialist",
            "app_specialist": "app_specialist",
            "seller_specialist": "seller_specialist",
        },
    )
    for specialist_node in [
        "delivery_specialist",
        "refund_specialist",
        "payment_specialist",
        "product_specialist",
        "account_specialist",
        "app_specialist",
        "seller_specialist",
    ]:
        builder.add_edge(specialist_node, "generate_response")
    builder.add_edge("generate_response", "validate_response")
    builder.add_conditional_edges("validate_response", route_after_validation, {
        "human_review": "human_review",
        "finalize_response": "finalize_response",
    })
    builder.add_edge("human_review", END)
    builder.add_edge("finalize_response", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


support_graph = build_support_graph()