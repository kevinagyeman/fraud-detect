"""
API route handlers for fraud detection endpoints.

This module defines the HTTP endpoints that clients interact with.
FastAPI handles:
- Request parsing
- Validation (via Pydantic models)
- Response serialization
- API documentation
"""

from fastapi import APIRouter, HTTPException, status, Query
from app.models.transaction import TransactionRequest, FraudPredictionResponse
from app.services.ensemble_detector import ensemble_detector
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Create a router for fraud-related endpoints
# This can be included in the main app
router = APIRouter(
    prefix="/api/v1",
    tags=["fraud-detection"],  # Groups endpoints in API docs
)


@router.post(
    "/predict",
    response_model=FraudPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict fraud probability for a transaction",
    description="""
    Analyzes a financial transaction and returns a fraud prediction.

    The endpoint:
    1. Validates the incoming transaction data
    2. Runs fraud detection rules
    3. Returns a fraud score and risk level

    **Use this endpoint when**:
    - A new transaction is initiated
    - You need real-time fraud assessment
    - Before processing a payment
    """,
)
async def predict_fraud(
    transaction: TransactionRequest,
    strategy: str = Query(
        default="weighted",
        description="Detection strategy: 'weighted', 'cascade', 'max', 'llm_only', 'rules_only'",
        pattern="^(weighted|cascade|max|llm_only|rules_only)$",
    ),
) -> FraudPredictionResponse:
    """
    Predict whether a transaction is fraudulent.

    This endpoint now supports multiple detection strategies:
    - **weighted**: Combines rule-based and LLM scores (default)
    - **cascade**: Uses rules first, LLM if uncertain
    - **max**: Takes the highest risk score (conservative)
    - **llm_only**: Uses only LLM analysis
    - **rules_only**: Uses only rule-based detection

    Args:
        transaction: Transaction details to analyze
        strategy: Detection strategy to use

    Returns:
        FraudPredictionResponse with fraud prediction and AI reasoning

    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        logger.info(
            f"Processing transaction {transaction.transaction_id} with strategy: {strategy}"
        )

        # Run ensemble fraud detection
        fraud_score, flags, reasoning, metadata = ensemble_detector.predict(
            transaction, strategy=strategy
        )

        # Determine risk level
        risk_level = ensemble_detector.get_risk_level(fraud_score)

        # Determine if it's fraud (using configured threshold)
        is_fraud = fraud_score >= 0.5

        # Build response
        response = FraudPredictionResponse(
            transaction_id=transaction.transaction_id,
            is_fraud=is_fraud,
            fraud_score=round(fraud_score, 3),  # Round to 3 decimals
            risk_level=risk_level,
            flags=flags,
            ai_reasoning=reasoning,
            detection_method=strategy,
            timestamp=datetime.utcnow(),
        )

        logger.info(
            f"Transaction {transaction.transaction_id} analyzed: "
            f"score={fraud_score:.3f}, is_fraud={is_fraud}, strategy={strategy}"
        )

        return response

    except Exception as e:
        # Log the error
        logger.error(
            f"Error processing transaction {transaction.transaction_id}: {str(e)}",
            exc_info=True,
        )

        # Return a 500 error to the client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the transaction: {str(e)}",
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Check if the API is running and healthy",
)
async def health_check():
    """
    Simple health check endpoint.

    Used by:
    - Load balancers to check if service is up
    - Monitoring systems
    - CI/CD pipelines

    Returns:
        Status information
    """
    return {
        "status": "healthy",
        "service": "fraud-detection-api",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get(
    "/stats",
    status_code=status.HTTP_200_OK,
    summary="Get detection statistics",
    description="Returns statistics about fraud detection (placeholder)",
)
async def get_stats():
    """
    Get fraud detection statistics.

    In a real system, this would return:
    - Total transactions processed
    - Fraud rate
    - Most common fraud flags
    - Performance metrics
    """
    return {
        "message": "Statistics endpoint - to be implemented",
        "note": "In production, this would query a database or cache for metrics",
    }
