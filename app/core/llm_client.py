"""
LLM Client for interacting with Ollama.
"""

import httpx
import json
from typing import Dict, Any, Optional
from app.core.config import settings


class OllamaClient:
    """Client for communicating with local Ollama LLM."""

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
        temperature = (
            temperature if temperature is not None else settings.llm_temperature
        )
        max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": settings.llm_top_p,
            },
        }

        if kwargs:
            payload["options"].update(kwargs)

        try:
            response = self.client.post("/api/generate", json=payload)
            response.raise_for_status()
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
        """Extract JSON from LLM response that may contain extra text."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        if "```json" in response_text:
            try:
                start = response_text.index("```json") + 7
                end = response_text.index("```", start)
                json_str = response_text[start:end].strip()
                return json.loads(json_str)
            except (ValueError, json.JSONDecodeError):
                pass

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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


ollama_client = OllamaClient()
