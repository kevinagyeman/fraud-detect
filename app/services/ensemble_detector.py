"""
Ensemble Fraud Detection - Combining Rules + LLM.

This demonstrates a KEY CONCEPT in production ML systems:
ENSEMBLE METHODS - combining multiple models/approaches for better accuracy.

Why Ensemble?
1. Rules are fast and explainable
2. LLMs are smart but slow and less predictable
3. Together they're more accurate than either alone

Real-world examples:
- Netflix: Combines multiple recommendation algorithms
- Fraud systems: Rules + ML + manual review
- Self-driving cars: Multiple sensor fusion
"""

import logging
from typing import Tuple, List, Dict, Any
from app.models.transaction import TransactionRequest
from app.services.fraud_detector import fraud_detector
from app.services.llm_fraud_detector import llm_fraud_detector
from app.core.config import settings

logger = logging.getLogger(__name__)


class EnsembleFraudDetector:
    """
    Ensemble fraud detector combining rule-based and LLM analysis.

    Strategy Options:
    1. Weighted Average: Combine scores with weights
    2. Maximum: Take the highest risk score (conservative)
    3. Voting: Majority vote from multiple models
    4. Cascading: Rules first, LLM only if uncertain

    We implement Weighted Average and Cascading.
    """

    def __init__(self):
        """Initialize the ensemble detector."""
        self.rule_detector = fraud_detector
        self.llm_detector = llm_fraud_detector

    def predict(
        self, transaction: TransactionRequest, strategy: str = "weighted"
    ) -> Tuple[float, List[str], str, Dict[str, Any]]:
        """
        Predict fraud using ensemble of rule-based and LLM detectors.

        Args:
            transaction: The transaction to analyze
            strategy: Ensemble strategy ("weighted", "cascade", "max", "llm_only", "rules_only")

        Returns:
            Tuple of (final_score, combined_flags, explanation, metadata)
            - final_score: Combined fraud score (0.0-1.0)
            - combined_flags: All risk factors from both systems
            - explanation: Human-readable reasoning
            - metadata: Individual scores and timing info

        """
        logger.info(
            f"Analyzing transaction {transaction.transaction_id} "
            f"with strategy: {strategy}"
        )

        metadata = {
            "strategy": strategy,
            "rule_score": None,
            "llm_score": None,
            "rule_flags": [],
            "llm_factors": [],
            "llm_reasoning": "",
        }

        # Strategy 1: LLM Only (for comparison)
        if strategy == "llm_only":
            if not settings.use_llm:
                logger.warning("LLM disabled but llm_only strategy requested")
                return 0.5, ["llm_disabled"], "LLM is disabled", metadata

            llm_score, llm_factors, llm_reasoning = self.llm_detector.predict(
                transaction
            )

            metadata["llm_score"] = llm_score
            metadata["llm_factors"] = llm_factors
            metadata["llm_reasoning"] = llm_reasoning

            return llm_score, llm_factors, llm_reasoning, metadata

        # Strategy 2: Rules Only (for comparison)
        if strategy == "rules_only":
            rule_score, rule_flags = self.rule_detector.predict(transaction)

            metadata["rule_score"] = rule_score
            metadata["rule_flags"] = rule_flags

            reasoning = (
                f"Rule-based detection: {', '.join(rule_flags)}"
                if rule_flags
                else "No red flags detected"
            )

            return rule_score, rule_flags, reasoning, metadata

        # For all other strategies, get both predictions
        # STEP 1: Run rule-based detection (always fast)
        logger.info("Running rule-based detection...")
        rule_score, rule_flags = self.rule_detector.predict(transaction)

        metadata["rule_score"] = rule_score
        metadata["rule_flags"] = rule_flags

        # STEP 2: Run LLM detection (if enabled)
        llm_score = None
        llm_factors = []
        llm_reasoning = ""

        if settings.use_llm:
            logger.info("Running LLM detection...")
            try:
                llm_score, llm_factors, llm_reasoning = self.llm_detector.predict(
                    transaction, rule_score=rule_score, rule_flags=rule_flags
                )

                metadata["llm_score"] = llm_score
                metadata["llm_factors"] = llm_factors
                metadata["llm_reasoning"] = llm_reasoning

            except Exception as e:
                logger.error(f"LLM detection failed: {e}")
                llm_score = rule_score  # Fallback to rule score

        # STEP 3: Combine results based on strategy
        if strategy == "weighted":
            final_score, combined_flags, explanation = self._weighted_combination(
                rule_score, rule_flags, llm_score, llm_factors, llm_reasoning
            )

        elif strategy == "cascade":
            final_score, combined_flags, explanation = self._cascade_combination(
                rule_score,
                rule_flags,
                llm_score,
                llm_factors,
                llm_reasoning,
                transaction,
            )

        elif strategy == "max":
            final_score, combined_flags, explanation = self._max_combination(
                rule_score, rule_flags, llm_score, llm_factors, llm_reasoning
            )

        else:
            logger.warning(f"Unknown strategy: {strategy}, using weighted")
            final_score, combined_flags, explanation = self._weighted_combination(
                rule_score, rule_flags, llm_score, llm_factors, llm_reasoning
            )

        return final_score, combined_flags, explanation, metadata

    def _weighted_combination(
        self,
        rule_score: float,
        rule_flags: List[str],
        llm_score: float,
        llm_factors: List[str],
        llm_reasoning: str,
    ) -> Tuple[float, List[str], str]:
        """
        Weighted average ensemble.

        Formula: final_score = (rule_score * w1) + (llm_score * w2)
        where w1 + w2 = 1.0

        This is the most common ensemble technique.
        """
        if llm_score is None:
            # No LLM score available, use rules only
            return rule_score, rule_flags, "Rule-based detection only"

        # Weighted average
        final_score = (
            rule_score * settings.rules_weight + llm_score * settings.llm_weight
        )

        # Combine flags
        combined_flags = list(set(rule_flags + llm_factors))

        # Create explanation
        explanation = (
            f"Ensemble analysis (Rules: {rule_score:.2f}, LLM: {llm_score:.2f}). "
            f"{llm_reasoning}"
        )

        return final_score, combined_flags, explanation

    def _cascade_combination(
        self,
        rule_score: float,
        rule_flags: List[str],
        llm_score: float,
        llm_factors: List[str],
        llm_reasoning: str,
        transaction: TransactionRequest,
    ) -> Tuple[float, List[str], str]:
        """
        Cascading ensemble.

        Logic:
        - If rules are confident (very high or very low), trust them
        - If rules are uncertain (middle range), use LLM

        This saves LLM calls for obvious cases.
        """
        # Define confidence thresholds
        HIGH_CONFIDENCE_THRESHOLD = 0.8
        LOW_CONFIDENCE_THRESHOLD = 0.2

        if rule_score >= HIGH_CONFIDENCE_THRESHOLD:
            # Rules are confident it's fraud
            return rule_score, rule_flags, "High-confidence rule-based detection"

        elif rule_score <= LOW_CONFIDENCE_THRESHOLD:
            # Rules are confident it's legitimate
            return rule_score, rule_flags, "High-confidence legitimate transaction"

        else:
            # Rules are uncertain, use LLM
            if llm_score is None:
                return rule_score, rule_flags, "Uncertain, LLM unavailable"

            return (
                llm_score,
                llm_factors,
                f"Uncertain rules, LLM analysis: {llm_reasoning}",
            )

    def _max_combination(
        self,
        rule_score: float,
        rule_flags: List[str],
        llm_score: float,
        llm_factors: List[str],
        llm_reasoning: str,
    ) -> Tuple[float, List[str], str]:
        """
        Maximum ensemble (conservative approach).

        Take the highest risk score from any model.
        Use this when false negatives (missing fraud) are very costly.
        """
        if llm_score is None:
            return rule_score, rule_flags, "Rule-based detection only"

        if llm_score > rule_score:
            return llm_score, llm_factors, f"LLM detected higher risk: {llm_reasoning}"
        else:
            return rule_score, rule_flags, "Rule-based detected higher risk"

    def get_risk_level(self, fraud_score: float) -> str:
        """Convert score to risk level."""
        if fraud_score >= settings.high_risk_threshold:
            return "critical"
        elif fraud_score >= settings.fraud_threshold:
            return "high"
        elif fraud_score >= 0.25:
            return "medium"
        else:
            return "low"


# Singleton instance
ensemble_detector = EnsembleFraudDetector()


# Usage example:
# from app.services.ensemble_detector import ensemble_detector
#
# score, flags, explanation, metadata = ensemble_detector.predict(
#     transaction,
#     strategy="weighted"
# )
