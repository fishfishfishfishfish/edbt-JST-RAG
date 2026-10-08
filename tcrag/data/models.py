"""Data models for documents, queries, and atomic facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Document:
    """A single document to be ingested into a RAG system.

    ``source_id`` uniquely identifies the document and is used as the
    retrieval-relevance key during evaluation.
    """

    source_id: str
    text: str
    uri: str | None = None
    reference_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    """A QA query with its gold answers and relevant document ids.

    ``relevant_doc_ids`` holds the source ids of documents that support the
    answer; used to compute retrieval Precision@k / Recall@k.
    ``answers`` holds one or more acceptable answer strings.
    """

    id: str
    text: str
    reference_time: datetime | None = None
    relevant_doc_ids: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomicFact:
    """A single atomic fact extracted from text.

    Mirrors the common subject/predicate/object triple shape used by both
    nuggetindex and graphiti so extractors can feed either system.
    """

    subject: str
    predicate: str
    object: str
    text: str = ""
    subject_type: str | None = None
    object_type: str | None = None
    validity_start: datetime | None = None
    validity_end: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    """A single retrieval result returned by a RAG system."""

    doc_id: str
    text: str
    score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResult:
    """Result of an end-to-end RAG question-answering call."""

    query_id: str
    answer: str
    retrieved: list[RetrievalHit] = field(default_factory=list)
    context: str = ""
    latency_seconds: float = 0.0
    # 分段耗时:retrieval_latency_seconds 仅检索,llm_latency_seconds 仅
    # LLM 生成(含 context 拼装,该部分开销很小)。两者之和 ≤ latency_seconds
    # (总 wall-clock)。
    retrieval_latency_seconds: float = 0.0
    llm_latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
