"""Abstract base class for a pluggable ``DocumentConstructor``.

nuggetindex's ``NuggetStore.aingest`` lazily builds a native
``DocumentConstructor`` on first ingestion and caches it on ``store._constructor``; that attribute can be
pre-assigned externally (the same pattern as the custom-retriever injection via ``store._retriever``), thereby
replacing, **without modifying the nuggetindex repository**,

    extract -> canonicalize -> temporal -> alias -> entity validation -> dedup -> conflict

the entire document construction pipeline.

This base class specifies a method signature exactly matching nuggetindex's native
``DocumentConstructor.aprocess``, so any subclass instance
can be used directly as ``store._constructor``. Implementations are uniformly placed under
:mod:`tcrag.constructors` and registered in the registry inside its ``__init__``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from nuggetindex.core.models import Nugget
    from nuggetindex.pipeline.aliases import AliasResolver
    from nuggetindex.pipeline.constructor import Document

    # Aligned with nuggetindex.pipeline.constructor.FetchExistingByKey:
    # (subject, predicate, object) -> already-ingested nuggets with the same key.
    FetchExistingByKey = Callable[[tuple[str, str, str]], Awaitable[list[Nugget]]]


class BaseDocumentConstructor(ABC):
    """Base class for custom document constructors (duck-type compatible with the native nuggetindex implementation)."""

    @abstractmethod
    async def aprocess(
        self,
        doc: Document,
        *,
        existing: list[Nugget] | None = None,
        fetch_existing_by_key: FetchExistingByKey | None = None,
        alias_resolver: AliasResolver | None = None,
    ) -> list[Nugget]:
        """Run the construction pipeline on a single document, returning a list of nuggets ready to be persisted.

        Args:
            doc: A nuggetindex ``Document`` (``source_id`` / ``text`` /
                ``uri`` / ``source_date``).
            existing: Existing nuggets of the same document explicitly supplied by the caller (the direct-call scenario);
                when entering via ``NuggetStore.aingest`` it is ``None``, and cross-document peers
                are fetched via ``fetch_existing_by_key``.
            fetch_existing_by_key: A callback that asynchronously retrieves peers already persisted in the backend
                by ``(subject, predicate, object)``, used for dedup / conflict detection.
            alias_resolver: A store-level, cross-document accumulating alias resolver; when None,
                the implementation may degrade to within-document resolution.
        """
        # pragma: no cover - the abstract signature only declares the contract
        raise NotImplementedError
