"""OpenAI-compatible remote cloud LLM backend.

Supports any OpenAI-compatible endpoint (OpenAI proper, Azure OpenAI,
siliconflow, vLLM OpenAI server, etc.) via ``base_url``.
"""

from __future__ import annotations

from typing import Any

from tcrag.llm.base import BaseLLM, LLMResponse


class OpenAILLM(BaseLLM):
    """LLM backend backed by the OpenAI REST API (or compatible)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 120,
        **kwargs,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAILLM requires a non-empty api_key")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        # Import lazily so the rest of the package works without openai installed.
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(**client_kwargs)

    @property
    def model_name(self) -> str:
        return self._model

    async def agenerate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature if temperature is None else temperature,
            max_tokens=self._max_tokens if max_tokens is None else max_tokens,
            **kwargs,
        )
        content = ""
        if resp.choices:
            content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=content,
            model=getattr(resp, "model", self._model),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            raw=resp,
        )
