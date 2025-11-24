"""
Fraud detection service using rule-based heuristics.
"""

from datetime import datetime
from typing import List, Tuple
from app.models.transaction import TransactionRequest


class FraudDetector:
    """Rule-based fraud detection engine."""

    HIGH_AMOUNT_THRESHOLD = 1000.0
    VERY_HIGH_AMOUNT_THRESHOLD = 5000.0
    SUSPICIOUS_COUNTRIES = {"NG", "RU", "CN", "BR"}
    NIGHT_HOUR_START = 23
    NIGHT_HOUR_END = 5

    def __init__(self):
        """Initialize the fraud detector."""
        pass

    def predict(self, transaction: TransactionRequest) -> Tuple[float, List[str]]:
        """
        Analyze a transaction and return fraud score and flags.

        Args:
            transaction: The transaction to analyze

        Returns:
            Tuple of (fraud_score, list_of_flags)
        """
        score = 0.0
        flags = []

        amount_score, amount_flag = self._check_amount(transaction.amount)
        score += amount_score
        if amount_flag:
            flags.append(amount_flag)

        location_score, location_flag = self._check_location(transaction.country)
        score += location_score
        if location_flag:
            flags.append(location_flag)

        time_score, time_flag = self._check_timing(transaction.timestamp)
        score += time_score
        if time_flag:
            flags.append(time_flag)

        payment_score, payment_flag = self._check_payment_method(
            transaction.payment_method, transaction.amount
        )
        score += payment_score
        if payment_flag:
            flags.append(payment_flag)

        if not transaction.device_id:
            score += 0.15
            flags.append("missing_device_info")

        final_score = min(score, 1.0)

        return final_score, flags

    def _check_amount(self, amount: float) -> Tuple[float, str]:
        """Check if transaction amount is suspicious."""
        if amount >= self.VERY_HIGH_AMOUNT_THRESHOLD:
            return 0.4, "very_high_amount"
        elif amount >= self.HIGH_AMOUNT_THRESHOLD:
            return 0.2, "high_amount"
        return 0.0, ""

    def _check_location(self, country: str) -> Tuple[float, str]:
        """Check if transaction originates from a high-risk location."""
        if country in self.SUSPICIOUS_COUNTRIES:
            return 0.3, "high_risk_country"
        return 0.0, ""

    def _check_timing(self, timestamp: datetime) -> Tuple[float, str]:
        """Check if transaction occurs at unusual times."""
        hour = timestamp.hour

        if hour >= self.NIGHT_HOUR_START or hour <= self.NIGHT_HOUR_END:
            return 0.15, "unusual_hours"

        return 0.0, ""

    def _check_payment_method(
        self, payment_method: str, amount: float
    ) -> Tuple[float, str]:
        """Check for suspicious payment method patterns."""
        if payment_method == "credit_card" and amount > 2000:
            return 0.2, "high_amount_credit_card"

        if payment_method == "bank_transfer" and amount < 50:
            return 0.1, "unusual_payment_method_for_amount"

        return 0.0, ""

    def get_risk_level(self, fraud_score: float) -> str:
        """Convert numerical score to human-readable risk level."""
        if fraud_score >= 0.75:
            return "critical"
        elif fraud_score >= 0.5:
            return "high"
        elif fraud_score >= 0.25:
            return "medium"
        else:
            return "low"

    def should_block(self, fraud_score: float) -> bool:
        """Decide whether to automatically block a transaction."""
        return fraud_score >= 0.75


fraud_detector = FraudDetector()
