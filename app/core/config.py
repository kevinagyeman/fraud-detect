"""
Configuration for the Fraud Detection API.

Why we need this:
- Centralize all settings in one place
- Easy to change without touching code
- Can load from environment variables
- Type-safe with Pydantic
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    You can create a .env file with these values:
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=llama3.2
    """

    # API Settings
    api_title: str = "Fraud Detection API"
    api_version: str = "1.0.0"
    debug: bool = True

    # Ollama LLM Settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout: int = 30  # seconds

    # LLM Generation Parameters
    llm_temperature: float = 0.1  # Low temperature for consistent fraud detection
    llm_max_tokens: int = 500  # Max response length
    llm_top_p: float = 0.9  # Nucleus sampling

    # Fraud Detection Settings
    use_llm: bool = True  # Enable/disable LLM analysis
    llm_weight: float = 0.6  # Weight for LLM score (0.0-1.0)
    rules_weight: float = 0.4  # Weight for rule-based score (0.0-1.0)

    # Thresholds
    fraud_threshold: float = 0.5  # Above this = fraud
    high_risk_threshold: float = 0.75  # Above this = critical

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # OLLAMA_BASE_URL or ollama_base_url both work


# Singleton instance - import this in other files
settings = Settings()


# Example of how to use:
# from app.core.config import settings
# print(settings.ollama_base_url)
