"""
Ensemble Fraud Detection - Combining Rules + LLM.
"""

import logging
from typing import Tuple, List, Dict, Any
from app.models.transaction import TransactionRequest
from app.services.fraud_detector import fraud_detector
from app.services.llm_fraud_detector import llm_fraud_detector
from app.core.config import settings

logger = logging.getLogger(__name__)


class EnsembleFraudDetector:
    """Ensemble fraud detector combining rule-based and LLM analysis."""

    def __init__(self):
        """Initialize the ensemble detector."""
        self.rule_detector = fraud_detector
        self.llm_detector = llm_fraud_detector

    def predict(
        self, transaction: TransactionRequest, strategy: str = "weighted"
    ) -> Tuple[float, List[str], str, Dict[str, Any]]:
        """Predict fraud using ensemble of rule-based and LLM detectors."""
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

        logger.info("Running rule-based detection...")
        rule_score, rule_flags = self.rule_detector.predict(transaction)

        metadata["rule_score"] = rule_score
        metadata["rule_flags"] = rule_flags

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
                llm_score = rule_score

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
        """Weighted average ensemble combining rules and LLM scores."""
        if llm_score is None:
            return rule_score, rule_flags, "Rule-based detection only"

        final_score = (
            rule_score * settings.rules_weight + llm_score * settings.llm_weight
        )

        combined_flags = list(set(rule_flags + llm_factors))

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
        """Cascading ensemble: use rules if confident, otherwise call LLM."""
        HIGH_CONFIDENCE_THRESHOLD = 0.8
        LOW_CONFIDENCE_THRESHOLD = 0.2

        if rule_score >= HIGH_CONFIDENCE_THRESHOLD:
            return rule_score, rule_flags, "High-confidence rule-based detection"

        elif rule_score <= LOW_CONFIDENCE_THRESHOLD:
            return rule_score, rule_flags, "High-confidence legitimate transaction"

        else:
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
        """Maximum ensemble: take the highest risk score (conservative)."""
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


ensemble_detector = EnsembleFraudDetector()
