"""Gemini Vision LLM implementation via OpenAI-compatible endpoint.

Gemini models (gemini-2.0-flash, gemini-1.5-pro, etc.) support multimodal
input through the OpenAI-compatible chat/completions endpoint with
image_url content blocks — identical to GPT-4o Vision.

Base URL: https://generativelanguage.googleapis.com/v1beta/openai/
Auth:     Bearer <GEMINI_API_KEY>
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.libs.llm.openai_vision_llm import OpenAIVisionLLM


class GeminiVisionLLMError(RuntimeError):
    """Raised when Gemini Vision API call fails."""


class GeminiVisionLLM(OpenAIVisionLLM):
    """Gemini Vision LLM provider via OpenAI-compatible endpoint.

    Inherits all image preprocessing and HTTP logic from OpenAIVisionLLM.

    Example:
        >>> settings = load_settings("config/settings.yaml")
        >>> vision_llm = GeminiVisionLLM(settings)
        >>> image = ImageInput(path="diagram.png")
        >>> response = vision_llm.chat_with_image("Describe this", image)
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_image_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        vision_settings = getattr(settings, "vision_llm", None)

        # Resolve API key: explicit > vision_llm section > llm section > env var
        resolved_key = api_key
        if not resolved_key and vision_settings:
            resolved_key = getattr(vision_settings, "api_key", None)
        if not resolved_key:
            resolved_key = getattr(settings.llm, "api_key", None)
        if not resolved_key:
            resolved_key = os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Gemini API key not provided. Set vision_llm.api_key in settings.yaml "
                "or the GEMINI_API_KEY environment variable."
            )

        # Determine base URL: explicit arg > vision_settings > default
        resolved_base_url = base_url
        if not resolved_base_url and vision_settings:
            resolved_base_url = getattr(vision_settings, "base_url", None)
        if not resolved_base_url:
            resolved_base_url = self.DEFAULT_BASE_URL

        super().__init__(
            settings=settings,
            api_key=resolved_key,
            base_url=resolved_base_url,
            max_image_size=max_image_size,
            **kwargs,
        )
