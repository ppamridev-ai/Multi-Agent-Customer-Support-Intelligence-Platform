from datetime import date
from enum import Enum
from typing import Literal, Any

from pydantic import BaseModel, ConfigDict, Field

class TicketCategory(str, Enum):
    ACCOUNT_LOGIN = "Account & Login"
    APP_WEBSITE = "App & Website Issue"
    DELIVERY = "Delivery Issue"
    PAYMENT = "Payment Issue"
    PRODUCT = "Product Issue"
    REFUND_RETURN = "Refund & Return"
    SELLER_LISTING = "Seller & Product Listing"

class TicketPriority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class TicketSentiment(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    SLIGHTLY_NEGATIVE = "Slightly Negative"
    NEGATIVE = "Negative"
    VERY_NEGATIVE = "Very Negative"

class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"

class TicketRecord(BaseModel):
    """Validated representation of one historical support ticket."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ticket_id: str = Field(min_length=1, pattern=r"^TKT\d+$", description="Unique identifier for the support ticket")
    created_date: date
    ticket_text: str = Field(min_length=5, description="The text of the support ticket")
    product_name: str = Field(min_length=1)
    product_segment: str = Field(min_length=1)
    ticket_category: TicketCategory
    priority: TicketPriority
    sentiment: TicketSentiment
    confidence_score: float = Field(ge=0.0, le=1.0)
    escalated: YesNo
    resolution_text: str = Field(min_length=1)
    resolved_date: date
    resolution_days: int = Field(ge=0)
    auto_resolved: YesNo
    customer_satisfaction_score: int = Field(ge=1, le=5)


class FAQRecord(BaseModel):
    """Validated representation of one FAQ entry."""

    model_config = ConfigDict(extra="forbid",str_strip_whitespace=True)
    faq_id: str = Field(min_length=1, pattern=r"^FAQ\d+$", description="Unique identifier for the FAQ entry")
    question: str = Field(min_length=5)
    answer: str = Field(min_length=5)
    category: TicketCategory


class ClassificationPrediction(BaseModel):
    """Output returned by the ML classification component."""

    category: TicketCategory
    priority: TicketPriority
    sentiment: TicketSentiment
    category_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    priority_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sentiment_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    requires_review: bool = False
    low_confidence_targets: list[str] = Field(default_factory=list)

class RetrievedFAQ(BaseModel):
    """FAQ result returned by the RAG retriever."""

    faq_id: str
    question: str
    answer: str
    category: TicketCategory
    similarity_score: float | None = None

class TicketUnderstanding(BaseModel):
    """Structured understanding of a customer ticket."""

    intent: str = Field(description="Short name for the customer's intent.")
    issue_summary: str = Field(description="A concise summary of the problem.")
    requested_action: str | None = Field(default=None,description="What the customer wants the company to do.")
    urgency: Literal["Low","Medium","High","Critical"]
    order_id: str | None = Field(default=None,description="Order ID stated in the ticket, if present.")
    product_name: str | None = None
    key_details: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None

class SpecialistResolution(BaseModel):
    """Resolution recommended by a category specialist."""

    specialist: str
    issue_assessment: str
    recommended_action: str
    resolution_steps: list[str] = Field(default_factory=list)
    supporting_faq_ids: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    escalation_reason: str | None = None

class GeneratedResponse(BaseModel):
    """Customer-facing response generated from a resolution plan."""

    subject: str = Field(min_length=3)
    message: str = Field(min_length=10)
    supporting_faq_ids: list[str] = Field(default_factory=list)
    contains_unverified_action: bool = False

class ResponseValidation(BaseModel):
    """Validation result for a generated response."""

    grounded: bool
    policy_compliant: bool
    complete: bool
    tone_appropriate: bool
    safe_to_send: bool

    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)

class TicketProcessRequest(BaseModel):
    """Request for processing a new support ticket."""
    ticket_text: str = Field(min_length=5, max_length=5_000)


class HumanReviewRequest(BaseModel):
    """Human decision for an interrupted workflow."""

    action: Literal["approve", "edit", "reject"]
    subject: str | None = None
    message: str | None = None
    reviewer_comment: str | None = None


class WorkflowResponse(BaseModel):
    """API representation of a LangGraph workflow."""

    thread_id: str
    status: Literal["completed", "human_review_required", "approved", "edited", "rejected"]
    state: dict[str, Any]
    review_request: dict[str, Any] | None = None