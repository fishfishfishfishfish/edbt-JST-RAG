"""Base LLM abstraction.

Every LLM backend (remote OpenAI-compatible, local Ollama, ...) implements
:class:`BaseLLM` so the rest of the framework only depends on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """Structured result of an LLM generation call."""

    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: object = field(default=None, repr=False)


class BaseLLM(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion for ``prompt``."""

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Synchronous wrapper around :meth:`agenerate`."""
        import asyncio

        return asyncio.run(
            self.agenerate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        )

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the human-readable model identifier used by this backend."""
