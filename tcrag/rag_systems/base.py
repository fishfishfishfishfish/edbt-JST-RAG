"""Standardized RAG system template.

Every RAG backend implements :class:`BaseRAGSystem`. The framework calls
:meth:`aingest_documents` once to build the index, then :meth:`aretrieve`
for each query. This decouples the benchmark harness from the specifics of
nuggetindex / graphiti.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from tcrag.data.models import Document, RetrievalHit


class BaseRAGSystem(ABC):
    """Abstract base class for atomic-fact-based RAG systems."""

    name: str = "base"

    @abstractmethod
    async def aingest_documents(self, documents: list[Document]) -> dict:
        """Build the retrieval index from ``documents``.

        Returns a summary dict (e.g. number of facts ingested, latency).
        """

    @abstractmethod
    async def aretrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        reference_time: datetime | None = None,
        **kwargs,
    ) -> list[RetrievalHit]:
        """Return the top-k relevant :class:`RetrievalHit`s for ``query``."""

    async def aclose(self) -> None:
        """Release resources (connections, file handles). Default: no-op."""
        return None
