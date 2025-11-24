"""
LLM-Based Fraud Detection Service.

This module brings together:
1. The LLM Client (HOW to talk to LLM)
2. The Prompt Builder (WHAT to say to LLM)
3. Response parsing and validation

Key Learning: This is the ORCHESTRATOR - it coordinates
the different components to perform fraud detection.
"""

import json
import logging
from typing import Tuple, List, Dict, Any, Optional
from app.models.transaction import TransactionRequest
from app.core.llm_client import ollama_client
from app.services.prompt_builder import FraudPromptBuilder
from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)


class LLMFraudDetector:
    """
    LLM-based fraud detection using local Llama 3.2.

    This class demonstrates the complete LLM integration flow:
    1. Receive transaction
    2. Build prompt
    3. Call LLM
    4. Parse response
    5. Handle errors
    6. Return structured result
    """

    def __init__(self):
        """Initialize the LLM fraud detector."""
        self.client = ollama_client
        self.prompt_builder = FraudPromptBuilder()

    def predict(
        self,
        transaction: TransactionRequest,
        rule_score: float = None,
        rule_flags: List[str] = None,
        use_few_shot: bool = False,
    ) -> Tuple[float, List[str], str]:
        """
        Analyze transaction using LLM and return fraud prediction.

        This is the MAIN method that orchestrates the entire process.

        Args:
            transaction: The transaction to analyze
            rule_score: Optional score from rule-based system
            rule_flags: Optional flags from rule-based system
            use_few_shot: Whether to include examples in prompt

        Returns:
            Tuple of (fraud_score, risk_factors, reasoning)
            - fraud_score: 0.0 to 1.0
            - risk_factors: List of identified risk factors
            - reasoning: Human-readable explanation

        Raises:
            Exception: If LLM call fails or response is invalid
        """
        try:
            # STEP 1: Build the prompt
            logger.info(f"Building prompt for transaction {transaction.transaction_id}")

            if use_few_shot:
                prompt = self.prompt_builder.build_few_shot_prompt(
                    transaction, rule_score, rule_flags
                )
            else:
                prompt = self.prompt_builder.build_analysis_prompt(
                    transaction, rule_score, rule_flags
                )

            # Log the prompt (useful for debugging/learning)
            logger.debug(f"Prompt length: {len(prompt)} characters")

            # STEP 2: Call the LLM
            logger.info(f"Calling LLM (model: {settings.ollama_model})")

            llm_response = self.client.generate(
                prompt=prompt,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

            response_text = llm_response["response"]
            logger.debug(
                f"LLM response: {response_text[:200]}..."
            )  # Log first 200 chars

            # STEP 3: Parse the response
            logger.info("Parsing LLM response")
            analysis = self._parse_response(response_text)

            if not analysis:
                # Fallback if parsing fails
                logger.warning("Failed to parse LLM response, using fallback")
                return 0.5, ["llm_parse_error"], "LLM response could not be parsed"

            # STEP 4: Extract structured data
            fraud_score = float(analysis.get("confidence", 0.5))
            risk_factors = analysis.get("risk_factors", [])
            reasoning = analysis.get("reasoning", "No explanation provided")

            logger.info(
                f"LLM analysis complete: score={fraud_score:.2f}, "
                f"factors={len(risk_factors)}"
            )

            return fraud_score, risk_factors, reasoning

        except Exception as e:
            # Log the error but don't crash - return a safe default
            logger.error(f"LLM fraud detection error: {str(e)}", exc_info=True)

            # Return medium risk as fallback
            return 0.5, ["llm_error"], f"Error during LLM analysis: {str(e)}"

    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response into structured data.

        The LLM should return JSON, but might add extra text.
        This method tries multiple strategies to extract valid JSON.

        Args:
            response_text: Raw text from LLM

        Returns:
            Parsed dict, or None if parsing fails
        """
        # Try using the client's JSON parser
        parsed = self.client.parse_json_response(response_text)

        if parsed:
            # Validate required fields
            if self._validate_analysis(parsed):
                return parsed
            else:
                logger.warning(f"Parsed JSON missing required fields: {parsed}")

        # If parsing failed, try manual extraction
        logger.warning("JSON parsing failed, attempting manual extraction")
        return self._fallback_parse(response_text)

    def _validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """
        Validate that LLM response has required fields.

        Args:
            analysis: Parsed JSON from LLM

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["is_fraud", "confidence", "reasoning"]

        for field in required_fields:
            if field not in analysis:
                return False

        # Validate data types
        if not isinstance(analysis["is_fraud"], bool):
            return False

        if not isinstance(analysis["confidence"], (int, float)):
            return False

        if not (0.0 <= analysis["confidence"] <= 1.0):
            return False

        return True

    def _fallback_parse(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Fallback parser when JSON extraction fails.

        Uses heuristics to extract information from unstructured text.

        Args:
            response_text: Raw LLM response

        Returns:
            Best-effort parsed dict, or None
        """
        # Look for keywords to determine if fraud
        fraud_keywords = [
            "fraud",
            "suspicious",
            "risky",
            "high risk",
            "account takeover",
        ]
        safe_keywords = ["legitimate", "normal", "safe", "low risk"]

        text_lower = response_text.lower()

        # Count fraud vs safe indicators
        fraud_count = sum(1 for kw in fraud_keywords if kw in text_lower)
        safe_count = sum(1 for kw in safe_keywords if kw in text_lower)

        if fraud_count > safe_count:
            is_fraud = True
            confidence = min(0.5 + (fraud_count * 0.1), 0.9)
        else:
            is_fraud = False
            confidence = max(0.5 - (safe_count * 0.1), 0.1)

        return {
            "is_fraud": is_fraud,
            "confidence": confidence,
            "reasoning": response_text[:200],  # Use first 200 chars
            "risk_factors": ["fallback_parse_used"],
        }

    def get_risk_level(self, fraud_score: float) -> str:
        """
        Convert numerical score to risk level.

        Args:
            fraud_score: 0.0 to 1.0

        Returns:
            Risk level string
        """
        if fraud_score >= 0.75:
            return "critical"
        elif fraud_score >= 0.5:
            return "high"
        elif fraud_score >= 0.25:
            return "medium"
        else:
            return "low"


# Singleton instance
llm_fraud_detector = LLMFraudDetector()


# Usage example:
# from app.services.llm_fraud_detector import llm_fraud_detector
#
# score, factors, reasoning = llm_fraud_detector.predict(transaction)
# print(f"Score: {score}, Reasoning: {reasoning}")
