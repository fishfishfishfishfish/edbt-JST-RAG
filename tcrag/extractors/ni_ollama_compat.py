"""Ollama fault-tolerant structured-output client, conforming to the nuggetindex ``LLMClient`` protocol.

nuggetindex's native :class:`OllamaClient` performs strict validation via instructor; small models
(e.g. llama3.2:3b) commonly violate the ``ExtractionPayload`` schema in two ways:

1. Emitting a single fact object instead of a ``{"facts": [...]}`` array;
2. Writing ``confidence`` as ``null`` (the schema requires a float, and an explicit null is invalid).

instructor's repair loop feeds the ValidationError back for retries, but small models often cannot fix it cleanly,
and after the retries are exhausted the whole document becomes ingest failed. This client bypasses instructor and directly requests
Ollama's OpenAI-compatible endpoint, normalizing the output before pydantic validation to cover the patterns above;
when it still fails, it raises ``ValueError``, which the upper ingest layer's try/except degrades into skipping that single document.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

__all__ = ["OllamaCompatClient"]

_DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


class OllamaCompatClient:
    """Duck-types the nuggetindex ``LLMClient`` (only ``achat_structured`` is required)."""

    def __init__(self, cfg: Any, *, max_retries: int = 1) -> None:
        from openai import AsyncOpenAI

        self._cfg = cfg
        self._max_retries = max_retries
        self._client = AsyncOpenAI(
            api_key=cfg.api_key or "ollama",
            base_url=cfg.base_url or _DEFAULT_OLLAMA_URL,
            timeout=cfg.timeout_seconds,
        )

    async def achat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> BaseModel:
        last_exc: Exception | None = None
        for _ in range(self._max_retries + 1):
            # Ollama structured output: constrained decoding enforces schema validity at the token level
            # (JSON object/array, confidence not null), so even small models (llama3.2) produce stable output;
            # _normalize serves only as a fallback safety net.
            # Add maxItems on the schema copy: a small model may keep generating facts without bound until
            # max_tokens truncation produces invalid JSON; this matches the semantics of the framework's max_facts.
            schema = response_model.model_json_schema()
            facts_prop = schema.get("properties", {}).get("facts")
            if isinstance(facts_prop, dict):
                facts_prop.setdefault("maxItems", 20)
            resp = await self._client.chat.completions.create(
                model=self._cfg.model,
                messages=messages,
                temperature=self._cfg.temperature,
                max_tokens=self._cfg.max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                    },
                },
            )
            content = resp.choices[0].message.content or ""
            try:
                return self._parse(content, response_model)
            except Exception as exc:  # noqa: BLE001 - parse boundary, retried below
                last_exc = exc
        raise ValueError(
            f"structured output failed after {self._max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    @classmethod
    def _parse(cls, content: str, response_model: type[BaseModel]) -> BaseModel:
        if not content:
            raise ValueError("LLM returned empty content")
        return response_model.model_validate(cls._normalize(cls._decode_json(content)))

    @staticmethod
    def _decode_json(content: str) -> Any:
        """Tolerate markdown fences and surrounding explanatory text; decode the first JSON value.

        llama3.2 often emits ``Here are ...:\\n```json [...]`` (a top-level array) or an object with
        explanatory text, so from the first ``{``/``[`` it uses ``raw_decode`` to extract the first complete
        JSON value; top-level objects, arrays, and concatenated bare objects are all covered.
        """
        starts = [i for i in (content.find("{"), content.find("[")) if i >= 0]
        if starts:
            try:
                data, _ = json.JSONDecoder().raw_decode(content[min(starts) :].lstrip())
                return data
            except json.JSONDecodeError:
                pass
        m = re.search(r"\{.*\}", content, re.DOTALL)
        raw = m.group(0) if m else content
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON from LLM: {exc}") from exc

    @staticmethod
    def _normalize(data: Any) -> Any:
        """Repair the typical violations of ``ExtractionPayload`` made by small models."""
        if isinstance(data, list):
            data = {"facts": data}
        if not isinstance(data, dict):
            return data
        if "facts" not in data and "subject" in data and "predicate" in data:
            data = {"facts": [data]}
        facts = data.get("facts")
        if isinstance(facts, list):
            cleaned = []
            for f in facts:
                if not isinstance(f, dict):
                    continue
                # Drop entries with empty subject/predicate/object (FactTriple requires non-empty strings)
                if not all(
                    isinstance(f.get(k), str) and f.get(k, "").strip()
                    for k in ("subject", "predicate", "object")
                ):
                    continue
                if f.get("confidence") is None:
                    # Remove the null so that the pydantic default (0.8) takes effect
                    f = {k: v for k, v in f.items() if k != "confidence"}
                cleaned.append(f)
            data["facts"] = cleaned
        return data
