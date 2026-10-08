"""Unified atomic-fact extractors for TCRag."""

from tcrag.extractors.base import BaseExtractor
from tcrag.extractors.rule_based import RuleBasedExtractor
from tcrag.extractors.llm_extractor import LLMExtractor
from tcrag.extractors.spacy_extractor import SpacyFactExtractor
from tcrag.extractors.factory import create_extractor

__all__ = [
    "BaseExtractor",
    "RuleBasedExtractor",
    "LLMExtractor",
    "SpacyFactExtractor",
    "create_extractor",
]
