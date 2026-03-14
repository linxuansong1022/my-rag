"""Gemini Embedding implementation via OpenAI-compatible endpoint.

Google Gemini exposes an OpenAI-compatible embeddings endpoint, so this
provider subclasses OpenAIEmbedding and overrides the base URL and env-var.

Base URL:  https://generativelanguage.googleapis.com/v1beta/openai/
Model:     text-embedding-004  (768 dimensions)
Auth:      Bearer <GEMINI_API_KEY>
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.libs.embedding.openai_embedding import OpenAIEmbedding


class GeminiEmbeddingError(RuntimeError):
    """Raised when Gemini Embeddings API call fails."""


class GeminiEmbedding(OpenAIEmbedding):
    """Gemini Embedding provider via OpenAI-compatible endpoint.

    Inherits all HTTP / parsing logic from OpenAIEmbedding.

    Supported models and dimensions:
        text-embedding-004          → 768
        text-multilingual-embedding-002 → 768

    Example:
        >>> settings = load_settings("config/settings.yaml")
        >>> embedding = GeminiEmbedding(settings)
        >>> vectors = embedding.embed(["hello world"])
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Known Gemini embedding model dimensions
    _MODEL_DIMENSIONS: dict[str, int] = {
        "text-embedding-004": 768,
        "text-multilingual-embedding-002": 768,
    }

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
            or getattr(settings.embedding, "api_key", None)
            or os.environ.get("GEMINI_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "Gemini API key not provided. Set embedding.api_key in settings.yaml "
                "or the GEMINI_API_KEY environment variable."
            )

        # Determine base URL: explicit arg > settings > default
        resolved_base_url = (
            base_url
            or getattr(settings.embedding, "base_url", None)
            or self.DEFAULT_BASE_URL
        )

        super().__init__(
            settings=settings,
            api_key=resolved_key,
            base_url=resolved_base_url,
            **kwargs,
        )

    def get_dimension(self) -> Optional[int]:
        """Return embedding dimension for the configured Gemini model."""
        if self.dimensions is not None:
            return self.dimensions
        return self._MODEL_DIMENSIONS.get(self.model)
