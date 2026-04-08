"""
llm - Pluggable LLM provider layer.

Public API:
    LLMProvider      - abstract base class
    get_llm_provider - cached factory (reads LLM_BACKEND env var)
"""

from .provider import LLMProvider
from .factory import get_llm_provider

__all__ = ["LLMProvider", "get_llm_provider"]
