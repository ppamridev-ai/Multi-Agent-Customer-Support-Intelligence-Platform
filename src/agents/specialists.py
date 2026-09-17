"""Category-specific customer-support agents."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.config import settings
from src.schemas import SpecialistResolution


SPECIALIST_GUIDANCE = {
    "Delivery Issue": """
Handle tracking, delays, missed delivery, damaged packages,
false delivery attempts, address changes, and delivery failures.
Do not promise a refund or replacement unless supported by an FAQ.
""",
    "Refund & Return": """
Handle return eligibility, refund timelines, exchanges, and
non-returnable items. Do not approve exceptions that are not
supported by the provided FAQs.
""",
    "Payment Issue": """
Handle failed payments, duplicate charges, payment verification,
COD, wallets, cards, and transaction problems. Escalate suspected
fraud or unauthorized charges.
""",
    "Product Issue": """
Handle defective, damaged, incorrect, incomplete, or poor-quality
products. Do not make warranty or refund promises unless they are
supported by the provided FAQs.
""",
    "Account & Login": """
Handle password, login, account access, identity verification,
security, and account-compromise issues. Escalate suspected account
takeover or security risks.
""",
    "App & Website Issue": """
Handle application errors, website failures, checkout problems,
performance issues, and missing interface options. Request useful
technical details when required.
""",
    "Seller & Product Listing": """
Handle seller behavior, listing accuracy, counterfeit concerns,
pricing information, and incorrect product descriptions. Escalate
fraudulent or prohibited listings.
""",
}


BASE_SYSTEM_PROMPT = """
You are the {category} specialist for an e-commerce customer-support
platform.

Your responsibility is to recommend a safe, policy-grounded
resolution for the ticket.

Specialist guidance:
{specialist_guidance}

Rules:
1. Use only facts from the ticket, ticket understanding,
   classification result, and retrieved FAQs.
2. Do not invent company policies, refund approvals, delivery dates,
   credits, discounts, or completed actions.
3. A retrieved FAQ is supporting information, not proof that an
   action has already occurred.
4. Include only FAQ IDs that directly support the recommendation.
5. Escalate cases involving fraud, account compromise, safety,
   unsupported policy exceptions, or insufficient information.
6. If the FAQs do not support a safe resolution, require human review.
7. Recommend actions only. Do not write the final customer response.
"""


@lru_cache(maxsize=7)
def get_specialist_agent(category: str):
    """Create and cache one agent for each specialist category."""

    if category not in SPECIALIST_GUIDANCE:
        raise ValueError(f"Unsupported specialist category: {category}")

    model = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )

    structured_model = model.with_structured_output(SpecialistResolution)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",BASE_SYSTEM_PROMPT),
            ("human",
                """
                Customer ticket:
                {ticket_text}

                Ticket understanding:
                {ticket_understanding}

                ML classification:
                {classification}

                Retrieved FAQ context:
                {faq_context}
                """,
            ),
        ]
    ).partial(
        category=category,
        specialist_guidance=(
            SPECIALIST_GUIDANCE[category]
        ),
    )

    return prompt | structured_model


def run_specialist(
    category: str,
    ticket_text: str,
    ticket_understanding: dict,
    classification: dict,
    retrieved_faqs: list[dict],
) -> SpecialistResolution:
    """Run the appropriate category specialist."""

    faq_context = "\n\n".join(
        (
            f"FAQ ID: {faq['faq_id']}\n"
            f"Question: {faq['question']}\n"
            f"Answer: {faq['answer']}\n"
            f"Similarity: {faq['similarity_score']}"
        )
        for faq in retrieved_faqs
    )

    if not faq_context:
        faq_context = "No relevant FAQs were retrieved."

    agent = get_specialist_agent(category)

    return agent.invoke(
        {
            "ticket_text": ticket_text,
            "ticket_understanding": ticket_understanding,
            "classification": classification,
            "faq_context": faq_context,
        }
    )