"""Ticket Understanding Agent."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings
from src.schemas import TicketUnderstanding


SYSTEM_PROMPT = """
You are a customer-support ticket understanding agent for an
e-commerce platform.

Analyze the customer's ticket and extract structured facts.

Rules:
1. Do not invent order IDs, products, dates, or customer details.
2. Preserve an order ID exactly as written.
3. Summarize the issue without proposing a resolution.
4. Determine urgency from the customer's wording and situation.
5. Use Critical only for immediate safety, security, fraud, or
   severe account-compromise situations.
6. Set needs_clarification to true only when essential information
   is missing.
7. If clarification is needed, provide one concise question.
8. Do not classify the ticket into a support department. A separate
   ML classifier handles category routing.
"""


@lru_cache(maxsize=1)
def get_ticket_understanding_agent():
    """Create and cache the structured-output agent."""

    model = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )

    structured_model = model.with_structured_output(TicketUnderstanding)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human","Customer ticket:\n\n{ticket_text}"),
        ]
    )

    return prompt | structured_model


def understand_ticket(ticket_text: str) -> TicketUnderstanding:
    """Analyze one customer-support ticket."""

    normalized_text = " ".join(ticket_text.split())

    if len(normalized_text) < 5:
        raise ValueError("Ticket text must contain at least 5 characters.")

    agent = get_ticket_understanding_agent()
    result = agent.invoke({"ticket_text": normalized_text})

    return result