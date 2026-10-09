"""nuggetindex-side LLMExtractor subclass: replace now() with placeholder validity.

nuggetindex's native :class:`LLMExtractor` uses
``ValidityInterval(start=datetime.now(UTC))`` inside :meth:`aextract` as the nugget's initial validity
(see ``nuggetindex/extractors/llm.py`` L144). This leads to:

1. ``is_placeholder()`` returns False (``source_type="document"`` rather than ``"placeholder"``)
2. The temporal stage of :class:`DocumentConstructor` treats that validity as the ``prior``,
   and does not re-infer it using ``source_date``
3. When the document has no explicit ``reference_time``, validity_start falls on the ingestion time,
   and queries earlier than that time (e.g. historical times in TimeQA) filter out every nugget

This subclass overrides :meth:`aextract`, replacing each nugget's validity with
``ValidityInterval.unknown()`` (``start=0001-01-01, source_type="placeholder"``),
so the temporal stage re-infers using ``source_date``. When the text contains real temporal clues, the real time is used;
when there are no clues, ``source_date`` is used as the fallback.

Usage (replaces ``nuggetindex.extractors.LLMExtractor``)::

    from tcrag.extractors.ni_llm_placeholder import PlaceholderValidityLLMExtractor
    extractor = PlaceholderValidityLLMExtractor(ni_cfg, client=build_client(ni_cfg))
"""

from __future__ import annotations

from nuggetindex.core.models import Nugget, ValidityInterval
from nuggetindex.extractors.base import ExtractionResult
from nuggetindex.extractors.llm import LLMExtractor as NiLLMExtractor

__all__ = ["PlaceholderValidityLLMExtractor"]


class PlaceholderValidityLLMExtractor(NiLLMExtractor):
    """LLMExtractor subclass: replace nugget validity with ``unknown()``.

    Inherits from nuggetindex's native :class:`LLMExtractor`; just before :meth:`aextract`
    returns, each ``ExtractionResult.nugget`` is rebuilt with
    ``validity=ValidityInterval.unknown()``, triggering the temporal stage to re-infer using
    ``source_date``.
    """

    async def aextract(
        self,
        text: str,
        *,
        context: str = "",
        source_id: str | None = None,
    ) -> list[ExtractionResult]:
        results = await super().aextract(text, context=context, source_id=source_id)
        out: list[ExtractionResult] = []
        for r in results:
            old = r.nugget
            # Replace now() with unknown(): source_type="placeholder" makes
            # the ``is_placeholder()`` in constructor.py return True,
            # so that ``prior = None`` and the temporal stage re-infers using source_date.
            new_nugget = Nugget.new(
                kind=old.kind,
                fact=old.fact,
                validity=ValidityInterval.unknown(),
                epistemic=old.epistemic,
                provenance=old.provenance,
                parent_id=old.parent_id,
                extraction_confidence=old.extraction_confidence,
                created_at=old.created_at,
                updated_at=old.updated_at,
            )
            out.append(
                ExtractionResult(
                    nugget=new_nugget,
                    confidence=r.confidence,
                    rationale=r.rationale,
                )
            )
        return out
