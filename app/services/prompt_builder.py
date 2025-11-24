"""
Prompt Builder for Fraud Detection.

This module demonstrates PROMPT ENGINEERING - the art of asking
LLMs effective questions to get reliable, structured answers.

Key Concepts:
1. Clear Role Definition
2. Structured Context
3. Specific Instructions
4. Output Format Specification
5. Few-Shot Examples (optional)
"""

from datetime import datetime
from typing import List, Dict, Any
from app.models.transaction import TransactionRequest


class FraudPromptBuilder:
    """
    Builds prompts for fraud detection LLM analysis.

    This class knows HOW to ask questions about fraud,
    but doesn't know how to call the LLM (that's the client's job).
    """

    # System prompt - defines the LLM's role and behavior
    SYSTEM_ROLE = """You are an expert fraud detection analyst with 20 years of experience in financial security. Your job is to analyze transaction data and identify potential fraud using pattern recognition, behavioral analysis, and risk assessment."""

    # Few-shot examples - teach the LLM by example
    FEW_SHOT_EXAMPLES = [
        {
            "transaction": {
                "amount": 45.99,
                "country": "US",
                "time": "14:30",
                "device": "known",
                "payment": "debit_card",
            },
            "analysis": {
                "is_fraud": False,
                "confidence": 0.05,
                "reasoning": "Normal transaction: typical amount, familiar location, known device, reasonable hour.",
                "risk_factors": [],
            },
        },
        {
            "transaction": {
                "amount": 4500.00,
                "country": "NG",
                "time": "03:15",
                "device": "unknown",
                "payment": "credit_card",
            },
            "analysis": {
                "is_fraud": True,
                "confidence": 0.92,
                "reasoning": "High-risk combination: large amount + suspicious location + unusual hour + unknown device. Classic account takeover pattern.",
                "risk_factors": [
                    "high_amount",
                    "high_risk_location",
                    "unusual_hours",
                    "unknown_device",
                ],
            },
        },
    ]

    @staticmethod
    def build_analysis_prompt(
        transaction: TransactionRequest,
        rule_score: float = None,
        rule_flags: List[str] = None,
    ) -> str:
        """
        Build a comprehensive fraud analysis prompt.

        This is the MAIN prompt engineering function. It combines:
        - Role definition
        - Context (transaction data)
        - Prior analysis (from rules)
        - Specific instructions
        - Output format

        Args:
            transaction: The transaction to analyze
            rule_score: Score from rule-based system (optional)
            rule_flags: Flags from rule-based system (optional)

        Returns:
            Formatted prompt string
        """
        # Extract and format transaction details
        hour = transaction.timestamp.hour
        day_name = transaction.timestamp.strftime("%A")
        time_str = transaction.timestamp.strftime("%H:%M UTC")

        # Determine time period
        if 5 <= hour < 12:
            time_period = "morning"
        elif 12 <= hour < 17:
            time_period = "afternoon"
        elif 17 <= hour < 21:
            time_period = "evening"
        else:
            time_period = "late night / early morning"

        # Build the prompt
        prompt = f"""{FraudPromptBuilder.SYSTEM_ROLE}

=== TRANSACTION TO ANALYZE ===

Transaction ID: {transaction.transaction_id}
User ID: {transaction.user_id}

Financial Details:
- Amount: ${transaction.amount:,.2f} USD
- Payment Method: {transaction.payment_method}
- Merchant: {transaction.merchant_id or "Not specified"}

Location & Device:
- Country: {transaction.country}
- IP Address: {transaction.ip_address}
- Device ID: {transaction.device_id or "Unknown/Missing"}

Timing:
- Timestamp: {time_str}
- Day: {day_name}
- Time Period: {time_period}
"""

        # Add rule-based analysis if available
        if rule_score is not None or rule_flags:
            prompt += f"""
=== RULE-BASED SYSTEM ANALYSIS ===

The rule-based fraud detection system has analyzed this transaction:
- Risk Score: {rule_score:.2f} (0.0 = safe, 1.0 = definite fraud)
- Triggered Flags: {", ".join(rule_flags) if rule_flags else "None"}

Consider this information, but use your own judgment.
"""

        # Add analysis instructions
        prompt += """
=== YOUR TASK ===

Analyze this transaction for potential fraud. Consider:

1. **Amount Analysis**: Is this amount unusual or suspiciously large/small?
2. **Location Risk**: Is the country/IP location suspicious?
3. **Timing Patterns**: Is the transaction time unusual (e.g., late night)?
4. **Device Security**: Is the device known or missing?
5. **Payment Method**: Does the payment method match the amount?
6. **Combined Patterns**: Do multiple factors together suggest fraud?

Think step-by-step about:
- What's normal vs. abnormal?
- What fraud patterns does this match?
- What's the likelihood this is account takeover, stolen card, etc.?

=== OUTPUT FORMAT ===

Respond with ONLY a JSON object (no additional text):

{
  "is_fraud": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief 1-2 sentence explanation",
  "risk_factors": ["factor1", "factor2", ...]
}

Examples of risk factors: "high_amount", "suspicious_location", "unusual_hours",
"unknown_device", "risky_payment_method", "velocity_anomaly", "account_takeover_pattern"

Respond now with your analysis:"""

        return prompt

    @staticmethod
    def build_simple_prompt(transaction: TransactionRequest) -> str:
        """
        Build a simpler prompt without rule-based context.
        Useful for comparing LLM-only vs hybrid approaches.

        Args:
            transaction: The transaction to analyze

        Returns:
            Formatted prompt string
        """
        return f"""Analyze this financial transaction for fraud:

Amount: ${transaction.amount}
Country: {transaction.country}
Time: {transaction.timestamp.strftime("%H:%M %A")}
Device: {transaction.device_id or "Unknown"}
Payment: {transaction.payment_method}

Is this likely fraud? Respond in JSON:
{{"is_fraud": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    @staticmethod
    def build_few_shot_prompt(
        transaction: TransactionRequest,
        rule_score: float = None,
        rule_flags: List[str] = None,
    ) -> str:
        """
        Build a prompt with few-shot examples.

        Few-shot learning: Show the LLM examples before asking
        it to analyze. This often improves accuracy.

        Args:
            transaction: The transaction to analyze
            rule_score: Score from rule-based system
            rule_flags: Flags from rule-based system

        Returns:
            Formatted prompt with examples
        """
        prompt = f"""{FraudPromptBuilder.SYSTEM_ROLE}

Before analyzing your transaction, here are examples:

=== EXAMPLE 1: LEGITIMATE TRANSACTION ===
Amount: $45.99, Country: US, Time: 14:30, Device: Known
Analysis: {{"is_fraud": false, "confidence": 0.05, "reasoning": "Normal transaction"}}

=== EXAMPLE 2: FRAUDULENT TRANSACTION ===
Amount: $4500, Country: NG, Time: 03:15, Device: Unknown
Analysis: {{"is_fraud": true, "confidence": 0.92, "reasoning": "High-risk combination: large amount + suspicious location + unusual hour + unknown device"}}

=== NOW ANALYZE THIS TRANSACTION ===
"""
        # Add the actual transaction
        prompt += FraudPromptBuilder.build_analysis_prompt(
            transaction, rule_score, rule_flags
        )

        return prompt


# Usage example:
# from app.services.prompt_builder import FraudPromptBuilder
#
# prompt = FraudPromptBuilder.build_analysis_prompt(transaction)
# print(prompt)
