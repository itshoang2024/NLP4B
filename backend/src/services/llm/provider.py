"""
provider.py - Abstract base class for LLM providers.

All LLM interactions in the backend (translation, intent extraction)
go through this interface.  Concrete implementations live in sibling
modules (gemini_provider, openai_compat_provider).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Thin abstraction over an LLM backend.

    Every provider must implement ``generate()``, which accepts a user
    prompt, a system instruction, generation parameters, and an optional
    ``json_mode`` flag.  The method returns the model's raw text output.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str = "",
        temperature: float = 0.1,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a completion and return the raw text.

        Parameters
        ----------
        prompt : str
            User-facing prompt / instruction.
        system_instruction : str
            System-level instruction prepended to the conversation.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Maximum output tokens.
        json_mode : bool
            If True, the provider should attempt to return valid JSON.
            The exact mechanism is provider-specific (Gemini uses
            ``response_mime_type``; OpenAI-compat uses
            ``response_format``).

        Returns
        -------
        str
            Raw model output text.

        Raises
        ------
        Exception
            Implementation-specific errors (network, auth, rate-limit).
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier used by this provider."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"
