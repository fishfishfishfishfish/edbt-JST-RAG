"""End-to-end RAG QA pipeline.

Combines a :class:`BaseRAGSystem` (retrieval) with a :class:`BaseLLM`
(generation). The :class:`ContextBuilder` assembles retrieved hits into a
prompt bounded by a token budget, and the pipeline returns the generated
answer plus the retrieved context for evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from tcrag.data.models import QAResult, Query, RetrievalHit
from tcrag.llm.base import BaseLLM
from tcrag.rag_systems.base import BaseRAGSystem


_SYSTEM_PROMPT = (
    "You are a retrieval-augmented question answering assistant. "
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information to answer, "
    "respond with exactly: I cannot answer this from the provided context. "
    "Keep your answer concise and factual. Cite the source ids in parentheses."
)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


@dataclass
class ContextBuildResult:
    text: str
    used_doc_ids: list[str]
    truncated: bool


class ContextBuilder:
    """Assemble retrieval hits into a token-bounded context string."""

    def __init__(self, token_budget: int = 3000) -> None:
        self._budget = token_budget

    def build(self, hits: list[RetrievalHit]) -> ContextBuildResult:
        blocks: list[str] = []
        used: list[str] = []
        total = 0
        truncated = False
        for hit in hits:
            header = f"[Source: {hit.doc_id}]"
            block = f"{header}\n{hit.text}"
            cost = _estimate_tokens(block)
            if total + cost > self._budget:
                truncated = True
                break
            blocks.append(block)
            used.append(hit.doc_id)
            total += cost
        return ContextBuildResult(text="\n\n".join(blocks), used_doc_ids=used, truncated=truncated)


class QAPipeline:
    """Orchestrates retrieval -> context building -> LLM generation."""

    def __init__(
        self,
        rag_system: BaseRAGSystem,
        llm: BaseLLM,
        *,
        top_k: int = 10,
        ctx_top_k: int | None = None,
        context_token_budget: int = 3000,
    ) -> None:
        """``top_k`` 为检索深度(rag top_k,返回给评估的 hits 数);
        ``ctx_top_k`` 为实际用于构造 LLM 上下文的段落数,为 None 时等于 top_k。
        检索指标(QAResult.retrieved)始终基于全量 top_k hits 计算。
        """
        self._rag = rag_system
        self._llm = llm
        self._top_k = top_k
        self._ctx_top_k = top_k if ctx_top_k is None else ctx_top_k
        self._ctx = ContextBuilder(token_budget=context_token_budget)

    async def aanswer(self, query: Query) -> QAResult:
        start = time.perf_counter()
        t_ret = time.perf_counter()
        hits = await self._rag.aretrieve(
            query.text,
            top_k=self._top_k,
            reference_time=query.reference_time,
        )
        retrieval_latency = time.perf_counter() - t_ret
        # ctx top_k:仅取检索结果前 N 条构造上下文;检索指标仍用全量 hits
        ctx = self._ctx.build(hits[: self._ctx_top_k])
        prompt = (
            f"Context:\n{ctx.text}\n\n"
            f"Question: {query.text}\n\n"
            "Answer:"
        )
        t_llm = time.perf_counter()
        resp = await self._llm.agenerate(
            prompt,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.0,
        )
        llm_latency = time.perf_counter() - t_llm
        elapsed = time.perf_counter() - start
        return QAResult(
            query_id=query.id,
            answer=resp.content.strip(),
            retrieved=hits,
            context=ctx.text,
            latency_seconds=elapsed,
            retrieval_latency_seconds=retrieval_latency,
            llm_latency_seconds=llm_latency,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            metadata={"context_truncated": ctx.truncated, "context_doc_ids": ctx.used_doc_ids},
        )
