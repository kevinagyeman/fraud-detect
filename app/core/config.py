"""
Configuration for the Fraud Detection API.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    api_title: str = "Fraud Detection API"
    api_version: str = "1.0.0"
    api_description: str = "Production-ready fraud detection API with ensemble methods"
    debug: bool = True

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout: int = 30

    # Low temperature for consistent fraud detection
    llm_temperature: float = 0.1
    llm_max_tokens: int = 500
    llm_top_p: float = 0.9

    use_llm: bool = True
    llm_weight: float = 0.6
    rules_weight: float = 0.4

    fraud_threshold: float = 0.5
    high_risk_threshold: float = 0.75

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
