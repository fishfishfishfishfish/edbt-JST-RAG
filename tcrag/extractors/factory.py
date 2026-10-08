"""Factory for building the configured extractor."""

from __future__ import annotations

from tcrag.config import AppConfig
from tcrag.extractors.base import BaseExtractor
from tcrag.extractors.llm_extractor import LLMExtractor
from tcrag.extractors.rule_based import RuleBasedExtractor
from tcrag.llm.factory import create_llm


def create_extractor(cfg: AppConfig, *, external: BaseExtractor | None = None) -> BaseExtractor:
    """Build the extractor described by ``cfg.extractor``.

    When ``external`` is provided it is returned as-is (the "test-time
    external extractor" path required by the spec).
    """
    if external is not None:
        return external
    kind = (cfg.extractor.type or "").lower()
    if kind == "rule_based":
        return RuleBasedExtractor()
    if kind in ("spacy", "spacy_sm"):
        # 延迟 import:spacy 是可选重依赖,不影响其他 extractor 的加载。
        from tcrag.extractors.spacy_extractor import SpacyFactExtractor

        return SpacyFactExtractor(
            model=cfg.extractor.spacy_model,
            max_facts=cfg.extractor.spacy_max_facts,
        )
    if kind == "llm":
        # The LLM extractor can reuse the main LLM config or a dedicated one.
        llm_cfg = cfg.llm
        if cfg.extractor.llm_provider and cfg.extractor.llm_provider != cfg.llm.provider:
            from tcrag.config import LLMConfig

            llm_cfg = LLMConfig(provider=cfg.extractor.llm_provider)
        return LLMExtractor(create_llm(llm_cfg))
    if kind == "external":
        raise ValueError(
            "extractor.type is 'external' but no external extractor was passed. "
            "Pass an extractor instance to create_extractor(external=...)."
        )
    raise ValueError(f"Unsupported extractor type: {kind!r}")
