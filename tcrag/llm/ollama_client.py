"""Local Ollama LLM backend."""

from __future__ import annotations

from tcrag.llm.base import BaseLLM, LLMResponse


class OllamaLLM(BaseLLM):
    """LLM backend backed by a locally running Ollama server."""

    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 120,
        **kwargs,
    ) -> None:
        self._model = model
        self._host = host
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        import ollama

        self._client = ollama.AsyncClient(host=host, timeout=timeout)

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
        options: dict[str, object] = {
            "temperature": self._temperature if temperature is None else temperature,
            "num_predict": self._max_tokens if max_tokens is None else max_tokens,
        }
        resp = await self._client.chat(
            model=self._model,
            messages=[
                *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                {"role": "user", "content": prompt},
            ],
            options=options,
        )
        # ollama python client: newer versions return a pydantic ChatResponse,
        # older ones a plain dict. Normalize so both shapes work.
        if isinstance(resp, dict):
            raw = resp
        else:
            raw = resp.model_dump() if hasattr(resp, "model_dump") else {}
        message = raw.get("message") or {}
        if isinstance(message, dict):
            content = message.get("content", "") or ""
        else:
            content = getattr(message, "content", "") or ""
        # Ollama does not always return token counts; normalize defensively.
        prompt_eval = raw.get("prompt_eval_count", 0) or 0
        eval_count = raw.get("eval_count", 0) or 0
        return LLMResponse(
            content=content,
            model=self._model,
            prompt_tokens=prompt_eval,
            completion_tokens=eval_count,
            total_tokens=prompt_eval + eval_count,
            raw=resp,
        )
