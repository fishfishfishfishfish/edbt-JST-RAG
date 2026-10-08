"""LLM-based atomic fact extractor.

Delegates triple extraction to :class:`nuggetindex.extractors.llm.LLMExtractor`
so the framework reuses nuggetindex's curated ``extraction.md`` prompt and
Pydantic structured-output schema (``ExtractionPayload``). A small adapter
bridges the framework's :class:`BaseLLM` into nuggetindex's ``LLMClient``
protocol, and the nugget-side ``ExtractionResult`` list is converted back into
the framework's :class:`AtomicFact` list.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from tcrag.data.models import AtomicFact
from tcrag.extractors.base import BaseExtractor
from tcrag.llm.base import BaseLLM


class _BaseLLMClientAdapter:
    """Adapt a framework :class:`BaseLLM` into nuggetindex's ``LLMClient``.

    nuggetindex's :class:`LLMExtractor` expects a client that returns a
    parsed Pydantic model from ``achat_structured``. The framework's
    ``BaseLLM`` only returns a raw completion string, so this adapter calls
    ``agenerate`` and then validates the JSON payload against the requested
    response model. Markdown fences are tolerated, mirroring nuggetindex's
    own JSON-mode clients.
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self._llm = llm
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def achat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> BaseModel:
        system_prompt = next(
            (m["content"] for m in messages if m.get("role") == "system"), None
        )
        user_prompt = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "user"
        )
        print(f"system_prompt: {system_prompt}")
        print(f"user_prompt: {user_prompt}")
        resp = await self._llm.agenerate(
            user_prompt,
            system_prompt=system_prompt,
            temperature=self._temperature,
            # 长段落会产出多条 pretty-printed JSON 事实;本地带 thinking 的模型
            # (如 qwen3)思维链也占用生成预算,1024 容易截断 JSON 导致解析为空。
            max_tokens=self._max_tokens,
        )
        print(f"resp.content: {resp.content}")
        return self._parse(resp.content, response_model)

    @staticmethod
    def _parse(content: str, response_model: type[BaseModel]) -> BaseModel:
        if not content:
            raise ValueError("LLM returned empty content; cannot parse structured output")
        # extraction.md asks the model for a JSON object; tolerate markdown fences.
        m = re.search(r"\{.*\}", content, re.DOTALL)
        payload = m.group(0) if m else content
        try:
            return response_model.model_validate_json(payload)
        except Exception as exc:  # noqa: BLE001 - structured-output parse boundary
            raise ValueError(f"failed to parse structured LLM output: {exc}") from exc


def _to_atomic_facts(results: list[Any], source_text: str) -> list[AtomicFact]:
    """Convert nuggetindex ``ExtractionResult`` list into ``AtomicFact`` list.

    Field mapping mirrors ``ExtractorWrapper`` in the reverse direction
    so a fact round-trips losslessly between the two systems.
    """
    facts: list[AtomicFact] = []
    for r in results:
        nugget = r.nugget
        fact = nugget.fact
        provenance = nugget.provenance[0] if nugget.provenance else None
        evidence = (
            (provenance.evidence_span if provenance else None)
            or fact.text
            or source_text
        )
        meta: dict[str, Any] = {
            "confidence": float(r.confidence),
            "extraction_confidence": float(
                getattr(nugget, "extraction_confidence", None) or r.confidence
            ),
        }
        if provenance is not None:
            meta["evidence_span"] = provenance.evidence_span
            meta["source_id"] = provenance.source_id
            meta["char_start"] = getattr(provenance, "char_start", 0)
            meta["char_end"] = getattr(provenance, "char_end", 0)
        validity = getattr(nugget, "validity", None)
        if validity is not None:
            meta["validity_known"] = getattr(validity, "validity_known", True)
        facts.append(
            AtomicFact(
                subject=fact.subject,
                predicate=fact.predicate,
                object=fact.object,
                text=evidence,
                subject_type=fact.subject_type,
                object_type=fact.object_type,
                validity_start=getattr(validity, "start", None) if validity else None,
                validity_end=getattr(validity, "end", None) if validity else None,
                metadata=meta,
            )
        )
    return facts


class LLMExtractor(BaseExtractor):
    """Extract atomic facts by delegating to nuggetindex's ``LLMExtractor``."""

    def __init__(self, llm: BaseLLM, *, max_facts: int = 20) -> None:
        self._llm = llm
        self._max_facts = max_facts
        self._inner = self._build_inner(llm)

    @staticmethod
    def _build_inner(llm: BaseLLM) -> Any:
        from nuggetindex.extractors import LLMConfig, LLMExtractor as NiLLMExtractor

        client = _BaseLLMClientAdapter(llm)
        # cfg is unused once a client is supplied; provider/model are placeholders
        # kept only to satisfy LLMConfig's required fields.
        # cfg = LLMConfig(provider="openai_compat", model=llm.model_name)
        # return NiLLMExtractor(cfg, client=client)
        cfg = LLMConfig(provider="ollama", model=llm.model_name)
        return NiLLMExtractor(cfg)

    async def aextract(
        self,
        text: str,
        *,
        context: str = "",
        source_id: str | None = None,
    ) -> list[AtomicFact]:
        if not text:
            return []
        try:
            results = await self._inner.aextract(
                text, context=context, source_id=source_id
            )
        except ValueError as exc:
            # JSON parse / structured-output validation failure: degrade
            # gracefully to no facts rather than aborting the ingest batch.
            return _log_and_empty(exc)
        facts = _to_atomic_facts(results, text)
        if self._max_facts > 0:
            facts = facts[: self._max_facts]
        return facts


def _log_and_empty(exc: Exception) -> list[AtomicFact]:
    from tcrag.logging_config import get_logger

    get_logger("extractors.llm").warning(
        "nuggetindex LLMExtractor returned unparseable output: %s", exc
    )
    return []
