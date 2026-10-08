"""nuggetindex 侧 LLMExtractor 子类:用 placeholder validity 替换 now()。

nuggetindex 原生 :class:`LLMExtractor` 在 :meth:`aextract` 里用
``ValidityInterval(start=datetime.now(UTC))`` 作为 nugget 的初始 validity
(见 ``nuggetindex/extractors/llm.py`` L144)。这会导致:

1. ``is_placeholder()`` 返回 False(``source_type="document"`` 而非 ``"placeholder"``)
2. :class:`DocumentConstructor` 的 temporal stage 把该 validity 作为 ``prior``,
   不会用 ``source_date`` 重新推理
3. 当 document 没有显式 ``reference_time`` 时,validity_start 落到 ingestion 时刻,
   早于该时刻的查询(如 TimeQA 的历史时间)会把 nugget 全部过滤掉

本子类 override :meth:`aextract`,把每个 nugget 的 validity 替换为
``ValidityInterval.unknown()``(``start=0001-01-01, source_type="placeholder"``),
让 temporal stage 用 ``source_date`` 重新推理。文本含真实时间线索时用真实时间,
无线索时用 ``source_date`` 兜底。

用法(替代 ``nuggetindex.extractors.LLMExtractor``)::

    from tcrag.extractors.ni_llm_placeholder import PlaceholderValidityLLMExtractor
    extractor = PlaceholderValidityLLMExtractor(ni_cfg, client=build_client(ni_cfg))
"""

from __future__ import annotations

from nuggetindex.core.models import Nugget, ValidityInterval
from nuggetindex.extractors.base import ExtractionResult
from nuggetindex.extractors.llm import LLMExtractor as NiLLMExtractor

__all__ = ["PlaceholderValidityLLMExtractor"]


class PlaceholderValidityLLMExtractor(NiLLMExtractor):
    """LLMExtractor 子类:把 nugget validity 替换为 ``unknown()``。

    继承 nuggetindex 原生 :class:`LLMExtractor`,仅在 :meth:`aextract`
    返回前把每个 ``ExtractionResult.nugget`` 重建为
    ``validity=ValidityInterval.unknown()``,触发 temporal stage 用
    ``source_date`` 重新推理。
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
            # 用 unknown() 替换 now():source_type="placeholder" 让
            # constructor.py 的 ``is_placeholder()`` 返回 True,
            # 从而 ``prior = None``,temporal stage 用 source_date 重新推理。
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
