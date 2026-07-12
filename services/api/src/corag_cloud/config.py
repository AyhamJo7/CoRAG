"""Application settings with production-safety validation."""

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Committed defaults that must never survive into a non-development
# environment; the validator below refuses startup if they do.
INSECURE_INTERNAL_TOKEN = "dev-insecure-internal-token"
INSECURE_API_KEY_SALT = "dev-insecure-api-key-salt"


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"

    # Shared secret authenticating the Next.js BFF to this service.
    internal_service_token: str = INSECURE_INTERNAL_TOKEN

    # Restricted app role (RLS enforced). Set from Phase 2 onward.
    database_url: str = ""
    # Admin role for migrations/provisioning only; never used on the hot path.
    database_admin_url: str = ""

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    llm_model: str = "gpt-4o-mini"
    ask_max_steps: int = 4
    ask_k: int = 6
    ask_max_tokens: int = 1024
    ask_rate_limit_per_minute: int = 6

    # Salts developer API keys before hashing (see api_keys.py).
    api_key_salt: str = INSECURE_API_KEY_SALT

    # Real-card billing stays off until the company-registration gate lifts.
    allow_live_billing: bool = False

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        if self.environment == "development":
            return self
        problems = []
        if self.internal_service_token == INSECURE_INTERNAL_TOKEN:
            problems.append("INTERNAL_SERVICE_TOKEN is still the insecure default")
        if not self.internal_service_token or len(self.internal_service_token) < 32:
            problems.append("INTERNAL_SERVICE_TOKEN must be at least 32 characters")
        if self.api_key_salt == INSECURE_API_KEY_SALT or len(self.api_key_salt) < 32:
            problems.append(
                "API_KEY_SALT must be set to a unique value of at least 32 characters"
            )
        if problems:
            raise ValueError(
                "Refusing to start in non-development environment: "
                + "; ".join(problems)
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
