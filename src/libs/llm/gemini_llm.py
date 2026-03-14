"""Gemini LLM implementation via OpenAI-compatible endpoint.

Google Gemini exposes an OpenAI-compatible REST API, so this provider
simply subclasses OpenAILLM and overrides the base URL and env-var name.

Base URL: https://generativelanguage.googleapis.com/v1beta/openai/
Auth:     Bearer <GEMINI_API_KEY>
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.libs.llm.openai_llm import OpenAILLM


class GeminiLLMError(RuntimeError):
    """Raised when Gemini API call fails."""


class GeminiLLM(OpenAILLM):
    """Gemini LLM provider via OpenAI-compatible endpoint.

    Inherits all HTTP logic from OpenAILLM; only the base URL and
    API-key resolution differ.

    Example:
        >>> settings = load_settings("config/settings.yaml")
        >>> llm = GeminiLLM(settings)
        >>> response = llm.chat([Message(role="user", content="Hello")])
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Resolve API key: explicit > settings > GEMINI_API_KEY env var
        resolved_key = (
            api_key
            or getattr(settings.llm, "api_key", None)
            or os.environ.get("GEMINI_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "Gemini API key not provided. Set llm.api_key in settings.yaml "
                "or the GEMINI_API_KEY environment variable."
            )

        # Determine base URL: explicit arg > settings > default
        resolved_base_url = (
            base_url
            or getattr(settings.llm, "base_url", None)
            or self.DEFAULT_BASE_URL
        )

        super().__init__(
            settings=settings,
            api_key=resolved_key,
            base_url=resolved_base_url,
            **kwargs,
        )
