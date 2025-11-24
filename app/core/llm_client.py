"""
LLM Client for interacting with Ollama.

This module handles:
1. HTTP communication with Ollama API
2. Error handling and retries
3. Response parsing
4. Timeout management

Key Learning: This is a CLIENT - it knows HOW to talk to the LLM,
but it doesn't know WHAT to say (that's the prompt builder's job).
"""

import httpx
import json
from typing import Dict, Any, Optional
from app.core.config import settings


class OllamaClient:
    """
    Client for communicating with local Ollama LLM.

    Ollama API Documentation:
    - POST /api/generate: Generate completion (non-streaming)
    - POST /api/chat: Chat completion (conversational)

    We use /api/generate for structured fraud analysis.
    """

    def __init__(self, base_url: str = None, model: str = None, timeout: int = None):
        """
        Initialize the Ollama client.

        Args:
            base_url: Ollama server URL (default from config)
            model: Model name (default from config)
            timeout: Request timeout in seconds (default from config)
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout

        # Create HTTP client (reusable connection)
        self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def generate(
        self, prompt: str, temperature: float = None, max_tokens: int = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a completion from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional Ollama parameters

        Returns:
            Dict containing the response and metadata

        Raises:
            httpx.HTTPError: If the request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        # Use config defaults if not provided
        temperature = (
            temperature if temperature is not None else settings.llm_temperature
        )
        max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

        # Build request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # We want the full response at once
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,  # Ollama's name for max_tokens
                "top_p": settings.llm_top_p,
            },
        }

        # Add any extra parameters
        if kwargs:
            payload["options"].update(kwargs)

        try:
            # Make the HTTP request
            response = self.client.post("/api/generate", json=payload)

            # Raise exception for HTTP errors (4xx, 5xx)
            response.raise_for_status()

            # Parse JSON response
            result = response.json()

            return {
                "response": result.get("response", ""),
                "model": result.get("model", ""),
                "total_duration": result.get("total_duration", 0),
                "load_duration": result.get("load_duration", 0),
                "prompt_eval_count": result.get("prompt_eval_count", 0),
                "eval_count": result.get("eval_count", 0),
            }

        except httpx.TimeoutException as e:
            raise Exception(f"LLM request timed out after {self.timeout}s: {str(e)}")

        except httpx.HTTPError as e:
            raise Exception(f"LLM HTTP error: {str(e)}")

        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response from LLM: {str(e)}")

    def parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Try to extract JSON from LLM response.

        LLMs sometimes wrap JSON in markdown or add extra text.
        This tries to find and extract valid JSON.

        Args:
            response_text: Raw text from LLM

        Returns:
            Parsed JSON dict, or None if no valid JSON found
        """
        # Try parsing the whole response first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try finding JSON in markdown code block
        if "```json" in response_text:
            try:
                start = response_text.index("```json") + 7
                end = response_text.index("```", start)
                json_str = response_text[start:end].strip()
                return json.loads(json_str)
            except (ValueError, json.JSONDecodeError):
                pass

        # Try finding any JSON object
        try:
            start = response_text.index("{")
            end = response_text.rindex("}") + 1
            json_str = response_text[start:end]
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            pass

        return None

    def close(self):
        """Close the HTTP client connection."""
        self.client.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()


# Create a singleton client instance
ollama_client = OllamaClient()


# Example usage:
# from app.core.llm_client import ollama_client
#
# result = ollama_client.generate("What is 2+2?")
# print(result["response"])
