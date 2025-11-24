"""
Transaction data models using Pydantic.

Pydantic models serve two purposes:
1. Define the structure of data we expect
2. Automatically validate and type-check incoming data
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from decimal import Decimal


class TransactionRequest(BaseModel):
    """
    Input model for fraud prediction requests.

    When a client sends data to /predict, FastAPI will:
    1. Parse the JSON
    2. Validate it against this model
    3. Return a 422 error if validation fails
    """

    # Transaction ID (required)
    transaction_id: str = Field(
        ..., description="Unique transaction identifier", examples=["TXN-2024-001234"]
    )

    # User information
    user_id: str = Field(
        ..., description="User account identifier", examples=["USER-12345"]
    )

    # Transaction amount (using Decimal for precise financial calculations)
    amount: float = Field(
        ...,
        gt=0,  # Must be greater than 0
        description="Transaction amount in USD",
        examples=[99.99],
    )

    # Payment method
    payment_method: str = Field(
        ...,
        description="Payment method used",
        examples=["credit_card", "debit_card", "paypal", "bank_transfer"],
    )

    # Location data
    ip_address: str = Field(
        ..., description="IP address of the transaction", examples=["192.168.1.1"]
    )

    country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
        examples=["US", "GB", "CA"],
    )

    # Optional fields
    merchant_id: Optional[str] = Field(
        None, description="Merchant/seller identifier", examples=["MERCHANT-789"]
    )

    device_id: Optional[str] = Field(
        None, description="Device fingerprint or ID", examples=["DEVICE-ABC123"]
    )

    # Timestamp (defaults to current time if not provided)
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Transaction timestamp"
    )

    # Custom validator example
    @field_validator("country")
    @classmethod
    def country_must_be_uppercase(cls, v: str) -> str:
        """Ensure country code is uppercase."""
        return v.upper()

    # Enable example generation for API docs
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transaction_id": "TXN-2024-001234",
                    "user_id": "USER-12345",
                    "amount": 99.99,
                    "payment_method": "credit_card",
                    "ip_address": "192.168.1.1",
                    "country": "US",
                    "merchant_id": "MERCHANT-789",
                    "device_id": "DEVICE-ABC123",
                }
            ]
        }
    }


class FraudPredictionResponse(BaseModel):
    """
    Output model for fraud prediction results.

    This defines what structure we send back to clients.
    """

    transaction_id: str = Field(..., description="Original transaction ID")

    is_fraud: bool = Field(
        ..., description="Whether transaction is predicted as fraudulent"
    )

    fraud_score: float = Field(
        ...,
        ge=0.0,  # Greater than or equal to 0
        le=1.0,  # Less than or equal to 1
        description="Fraud probability score (0.0 = legitimate, 1.0 = fraudulent)",
    )

    risk_level: str = Field(
        ..., description="Risk category", examples=["low", "medium", "high", "critical"]
    )

    flags: list[str] = Field(
        default_factory=list, description="List of triggered fraud indicators"
    )

    ai_reasoning: Optional[str] = Field(
        None, description="AI-generated explanation of the fraud prediction"
    )

    detection_method: str = Field(
        default="ensemble",
        description="Detection method used",
        examples=["rules_only", "llm_only", "ensemble"],
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Prediction timestamp"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transaction_id": "TXN-2024-001234",
                    "is_fraud": True,
                    "fraud_score": 0.87,
                    "risk_level": "high",
                    "flags": ["high_amount", "unusual_location", "new_device"],
                    "ai_reasoning": "High-risk combination: large amount + suspicious location + unusual hour + unknown device",
                    "detection_method": "ensemble",
                    "timestamp": "2024-01-15T10:30:00Z",
                }
            ]
        }
    }
