"""Rule-based trigger-verb extractor.

A lightweight, dependency-free extractor that scans text for a small set of
relation trigger patterns and emits atomic facts. This gives the framework a
working baseline that needs neither an LLM nor spaCy.

For higher-quality rule-based extraction, pass nuggetindex's
``TriggerExtractor`` via the ``external`` extractor option.
"""

from __future__ import annotations

import re
from typing import Iterable

from tcrag.data.models import AtomicFact
from tcrag.extractors.base import BaseExtractor


# (predicate, compiled regex) pairs. The regex captures a subject phrase
# before the trigger and an object phrase after it.
_TRIGGERS: list[tuple[str, re.Pattern[str]]] = [
    ("isA", re.compile(r"\b([A-Z][\w\s'-]{1,40}?)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*([A-Z][\w\s'-]{1,40}?)\b", re.IGNORECASE)),
    ("bornIn", re.compile(r"\b([A-Z][\w\s'-]{1,40}?)\s+(?:was\s+)?born\s+(?:in|on)\s+([\w\s,'-]{1,40}?)\b", re.IGNORECASE)),
    ("locatedIn", re.compile(r"\b([A-Z][\w\s'-]{1,40}?)\s+(?:is|was|are|were)\s+located\s+in\s+([\w\s,'-]{1,40}?)\b", re.IGNORECASE)),
    ("workedAt", re.compile(r"\b([A-Z][\w\s'-]{1,40}?)\s+(?:works?|worked|working)\s+(?:at|for)\s+([\w\s,'-]{1,40}?)\b", re.IGNORECASE)),
    ("founded", re.compile(r"\b([A-Z][\w\s'-]{1,40}?)\s+(?:founded|established|created)\s+([\w\s,'-]{1,40}?)\b", re.IGNORECASE)),
    ("marriedTo", re.compile(r"\b([A-Z][\w\s'-]{1,40}?)\s+(?:married|wed)\s+([A-Z][\w\s'-]{1,40}?)\b", re.IGNORECASE)),
    ("succeededBy", re.compile(r"\b([A-Z][\w\s'-]{1,40}?)\s+(?:was\s+)?succeeded\s+by\s+([A-Z][\w\s'-]{1,40}?)\b", re.IGNORECASE)),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _clean(phrase: str) -> str:
    return phrase.strip(" ,.;:!?").strip()


class RuleBasedExtractor(BaseExtractor):
    """Extract facts using trigger-verb regular expressions."""

    def __init__(self, triggers: Iterable[tuple[str, re.Pattern[str]]] | None = None) -> None:
        self._triggers = list(triggers) if triggers is not None else list(_TRIGGERS)

    async def aextract(
        self,
        text: str,
        *,
        context: str = "",
        source_id: str | None = None,
    ) -> list[AtomicFact]:
        if not text:
            return []
        facts: list[AtomicFact] = []
        seen: set[tuple[str, str, str]] = set()
        for sentence in _SENTENCE_SPLIT.split(text):
            for predicate, pattern in self._triggers:
                for match in pattern.finditer(sentence):
                    subject = _clean(match.group(1))
                    obj = _clean(match.group(2))
                    if not subject or not obj:
                        continue
                    if len(subject) > 60 or len(obj) > 60:
                        continue
                    key = (subject, predicate, obj)
                    if key in seen:
                        continue
                    seen.add(key)
                    facts.append(
                        AtomicFact(
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            text=sentence.strip(),
                        )
                    )
        return facts
