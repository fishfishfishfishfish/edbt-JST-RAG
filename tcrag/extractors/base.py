"""Base extractor interface.

All extractors return a list of :class:`AtomicFact`. The signature mirrors
nuggetindex's ``BaseExtractor.aextract`` so a single extractor instance can
be adapted into nuggetindex's pipeline (see
``tcrag.extractors.extractor_wrapper.ExtractorWrapper``) as well as
injected into graphiti's extraction path via a stub LLM client.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from tcrag.data.models import AtomicFact


class BaseExtractor(ABC):
    """Extract atomic facts from natural-language text."""

    @abstractmethod
    async def aextract(
        self,
        text: str,
        *,
        context: str = "",
        source_id: str | None = None,
    ) -> list[AtomicFact]:
        """Return the list of atomic facts found in ``text``."""

    def extract(
        self,
        text: str,
        *,
        context: str = "",
        source_id: str | None = None,
    ) -> list[AtomicFact]:
        """Synchronous wrapper around :meth:`aextract`."""
        import asyncio

        return asyncio.run(self.aextract(text, context=context, source_id=source_id))
