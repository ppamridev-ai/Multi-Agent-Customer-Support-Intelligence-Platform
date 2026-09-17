"""Validation Agent for customer-support responses."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings
from src.schemas import ResponseValidation


SYSTEM_PROMPT = """
You are a strict customer-support response validator.

Evaluate the generated response against the original ticket,
ticket understanding, retrieved FAQs, and specialist resolution.

Validation requirements:

1. Grounding
   Every policy statement and recommended action must be supported
   by the ticket, retrieved FAQs, or specialist resolution.

2. No fabricated actions
   The response must not claim that a refund, replacement, shipment,
   investigation, complaint, escalation, or account change has
   already happened unless the supplied context proves it.

3. Policy compliance
   The response must not promise unsupported exceptions, timelines,
   reimbursements, credits, or outcomes.

4. Completeness
   The response must address the customer's problem or request the
   information needed to continue.

5. Tone
   The response must be clear, respectful, empathetic, and concise.

6. Safety
   Fraud, security, account compromise, safety issues, unsupported
   policy exceptions, and uncertain high-risk actions require human
   review.

Set safe_to_send to true only when every important validation
requirement passes.

Do not rewrite the response. Return specific revision instructions
when something needs correction.
"""


@lru_cache(maxsize=1)
def get_validation_agent():
    """Create and cache the Validation Agent."""

    model = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )

    structured_model = model.with_structured_output(ResponseValidation)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human",
                """
Original ticket:
{ticket_text}

Ticket understanding:
{ticket_understanding}

Classification:
{classification}

Retrieved FAQs:
{retrieved_faqs}

Specialist resolution:
{specialist_resolution}

Generated response:
{draft_response}
""",
            ),
        ]
    )

    return prompt | structured_model


def validate_response(
    ticket_text: str,
    ticket_understanding: dict,
    classification: dict,
    retrieved_faqs: list[dict],
    specialist_resolution: dict,
    draft_response: dict,
) -> ResponseValidation:
    """Validate a generated customer response."""

    agent = get_validation_agent()

    return agent.invoke(
        {
            "ticket_text": ticket_text,
            "ticket_understanding": ticket_understanding,
            "classification": classification,
            "retrieved_faqs": retrieved_faqs,
            "specialist_resolution": specialist_resolution,
            "draft_response": draft_response,
        }
    )