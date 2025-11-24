"""
Fraud detection service using rule-based heuristics.

This demonstrates how fraud detection works at a fundamental level.
In production, you'd combine rules with machine learning models.

Rule-based detection is:
- Easy to understand and debug
- Fast to execute
- Transparent (you know why something was flagged)
- Good for known fraud patterns

Machine learning is better for:
- Unknown patterns
- Complex relationships
- Adapting to new fraud techniques
"""

from datetime import datetime, timedelta
from typing import List, Tuple
from app.models.transaction import TransactionRequest


class FraudDetector:
    """
    Rule-based fraud detection engine.

    Each rule checks for a specific fraud indicator.
    Rules return a score (0.0 to 1.0) and optionally a flag name.
    """

    # Thresholds (you can tune these based on your data)
    HIGH_AMOUNT_THRESHOLD = 1000.0
    VERY_HIGH_AMOUNT_THRESHOLD = 5000.0
    SUSPICIOUS_COUNTRIES = {"NG", "RU", "CN", "BR"}  # Example, not real data
    NIGHT_HOUR_START = 23  # 11 PM
    NIGHT_HOUR_END = 5  # 5 AM

    def __init__(self):
        """Initialize the fraud detector."""
        # In a real system, you might load ML models here
        # or connect to a feature store
        pass

    def predict(self, transaction: TransactionRequest) -> Tuple[float, List[str]]:
        """
        Analyze a transaction and return fraud score and flags.

        Args:
            transaction: The transaction to analyze

        Returns:
            Tuple of (fraud_score, list_of_flags)
            - fraud_score: 0.0 (legitimate) to 1.0 (definitely fraud)
            - flags: List of triggered fraud indicators
        """
        score = 0.0
        flags = []

        # Rule 1: Check transaction amount
        amount_score, amount_flag = self._check_amount(transaction.amount)
        score += amount_score
        if amount_flag:
            flags.append(amount_flag)

        # Rule 2: Check location/country
        location_score, location_flag = self._check_location(transaction.country)
        score += location_score
        if location_flag:
            flags.append(location_flag)

        # Rule 3: Check transaction timing
        time_score, time_flag = self._check_timing(transaction.timestamp)
        score += time_score
        if time_flag:
            flags.append(time_flag)

        # Rule 4: Check payment method
        payment_score, payment_flag = self._check_payment_method(
            transaction.payment_method, transaction.amount
        )
        score += payment_score
        if payment_flag:
            flags.append(payment_flag)

        # Rule 5: Check for new device (if device_id is missing, it's suspicious)
        if not transaction.device_id:
            score += 0.15
            flags.append("missing_device_info")

        # Normalize score to 0-1 range (we have 5 rules, max score ~= 1.0)
        # Using min() to cap at 1.0
        final_score = min(score, 1.0)

        return final_score, flags

    def _check_amount(self, amount: float) -> Tuple[float, str]:
        """
        Check if transaction amount is suspicious.

        Higher amounts are riskier, especially if unusual for the user.
        In a real system, you'd compare against user's historical behavior.
        """
        if amount >= self.VERY_HIGH_AMOUNT_THRESHOLD:
            return 0.4, "very_high_amount"
        elif amount >= self.HIGH_AMOUNT_THRESHOLD:
            return 0.2, "high_amount"
        return 0.0, ""

    def _check_location(self, country: str) -> Tuple[float, str]:
        """
        Check if transaction originates from a high-risk location.

        In reality, you'd use:
        - IP geolocation services
        - Historical user location patterns
        - Velocity checks (user in two countries in short time)
        """
        if country in self.SUSPICIOUS_COUNTRIES:
            return 0.3, "high_risk_country"
        return 0.0, ""

    def _check_timing(self, timestamp: datetime) -> Tuple[float, str]:
        """
        Check if transaction occurs at unusual times.

        Fraudulent transactions often happen at night when
        the real user is asleep and won't notice immediately.
        """
        hour = timestamp.hour

        # Check if it's late night / early morning
        if hour >= self.NIGHT_HOUR_START or hour <= self.NIGHT_HOUR_END:
            return 0.15, "unusual_hours"

        return 0.0, ""

    def _check_payment_method(
        self, payment_method: str, amount: float
    ) -> Tuple[float, str]:
        """
        Check for suspicious payment method patterns.

        Different payment methods have different fraud rates:
        - Credit cards: Higher fraud (can dispute charges)
        - Bank transfers: Lower fraud (harder to reverse)
        """
        # Large amounts via certain payment methods are riskier
        if payment_method == "credit_card" and amount > 2000:
            return 0.2, "high_amount_credit_card"

        # Wire transfers of small amounts are unusual
        if payment_method == "bank_transfer" and amount < 50:
            return 0.1, "unusual_payment_method_for_amount"

        return 0.0, ""

    def get_risk_level(self, fraud_score: float) -> str:
        """
        Convert numerical score to human-readable risk level.

        This helps stakeholders understand the results.
        """
        if fraud_score >= 0.75:
            return "critical"
        elif fraud_score >= 0.5:
            return "high"
        elif fraud_score >= 0.25:
            return "medium"
        else:
            return "low"

    def should_block(self, fraud_score: float) -> bool:
        """
        Decide whether to automatically block a transaction.

        In production, you'd have different actions:
        - Low risk: Allow
        - Medium risk: Flag for review
        - High risk: Require additional verification (2FA, etc.)
        - Critical: Block immediately
        """
        return fraud_score >= 0.75


# Singleton instance
fraud_detector = FraudDetector()
