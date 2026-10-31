"""Example schema: triaging customer support tickets."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Category(str, Enum):
    billing = "billing"
    bug = "bug"
    feature_request = "feature_request"
    account = "account"
    shipping = "shipping"
    other = "other"


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class Sentiment(str, Enum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"


class SupportTicket(BaseModel):
    category: Category
    priority: Priority = Field(description="urgent = service down or money lost right now")
    sentiment: Sentiment
    product: str | None = Field(None, description="product or feature the ticket is about")
    order_id: str | None = Field(None, pattern=r"^[A-Z]{2}-\d{4,8}$", description="e.g. AB-12345")
    refund_requested: bool = False
    summary: str = Field(max_length=140, description="one sentence, neutral tone")

    @field_validator("category", "priority", "sentiment", mode="before")
    @classmethod
    def _lower(cls, v):
        # Models often return "Billing" or "HIGH"; normalise before enum validation.
        return v.lower().strip().replace(" ", "_") if isinstance(v, str) else v
