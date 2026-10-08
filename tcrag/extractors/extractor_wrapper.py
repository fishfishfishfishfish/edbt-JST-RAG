"""framework → nuggetindex 的 extractor 包装器。

把 framework 侧 :class:`tcrag.extractors.base.BaseExtractor`
(产 :class:`tcrag.data.models.AtomicFact` 列表)包装成 nuggetindex
duck-type extractor:``aextract`` 返回
``nuggetindex.extractors.base.ExtractionResult`` 列表,可直接作为
``NuggetStore(extractor=...)`` / DocumentConstructor 的 extractor 注入。

用法::

    from tcrag.extractors.extractor_wrapper import ExtractorWrapper
    from tcrag.extractors.spacy_extractor import SpacyFactExtractor

    extractor = ExtractorWrapper(SpacyFactExtractor(model="en_core_web_sm"))
"""

from __future__ import annotations

from datetime import UTC

from tcrag.extractors.base import BaseExtractor

__all__ = ["ExtractorWrapper"]


class ExtractorWrapper:
    """Adapt a framework extractor into nuggetindex's pipeline."""

    emits_placeholder_validity: bool = True

    def __init__(self, extractor: BaseExtractor) -> None:
        self._extractor = extractor

    async def aextract(self, text, *, context="", source_id=None):
        from nuggetindex.core.enums import NuggetKind
        from nuggetindex.core.models import (
            EpistemicState, FactTriple, Nugget, ProvenanceRecord, ValidityInterval,
        )
        from nuggetindex.extractors.base import ExtractionResult

        facts = await self._extractor.aextract(text, context=context, source_id=source_id)
        out = []
        sid = source_id or "tcrag"
        for f in facts:
            if not f.subject or not f.predicate or not f.object:
                continue
            meta = f.metadata or {}
            validity = ValidityInterval.unknown()
            if f.validity_start is not None:
                start = f.validity_start if f.validity_start.tzinfo else f.validity_start.replace(tzinfo=UTC)
                # Forward validity_known so extractors that emit a concrete
                # but partially-unknown interval (e.g. TimeQAExtractor's
                # ``until 2000`` case) are not reparsed by the temporal stage.
                validity = ValidityInterval(
                    start=start,
                    end=f.validity_end,
                    validity_known=bool(meta.get("validity_known", True)),
                )
            confidence = float(meta.get("confidence", 1.0))
            evidence_span = meta.get("evidence_span") or f.text or text
            provenance = ProvenanceRecord(
                source_id=meta.get("source_id", sid),
                evidence_span=evidence_span,
                char_start=int(meta.get("char_start", 0)),
                char_end=int(meta.get("char_end", 0)),
            )
            nugget = Nugget.new(
                kind=NuggetKind.SEMANTIC_FACT,
                fact=FactTriple(subject=f.subject, predicate=f.predicate, object=f.object,
                                text=f.text or text, subject_type=f.subject_type, object_type=f.object_type),
                validity=validity,
                epistemic=EpistemicState(confidence=confidence),
                provenance=(provenance,),
                extraction_confidence=confidence,
            )
            out.append(ExtractionResult(nugget=nugget, confidence=confidence))
        return out
