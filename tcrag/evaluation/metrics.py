"""Retrieval and answer-quality metrics.

Retrieval metrics operate at the document level: a retrieved hit is relevant
if its ``doc_id`` appears in the query's ``relevant_doc_ids``.

Answer-quality metrics are reference-based string similarity measures that
work without an LLM judge: exact match, token-level F1, and answer recall
(fraction of gold answer tokens covered by the prediction).
"""

from __future__ import annotations

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for d in top if d in relevant_ids)
    return hits / len(top)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    hits = sum(1 for d in relevant_ids if d in top)
    return hits / len(relevant_ids)


def f1_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Answer-quality metrics
# ---------------------------------------------------------------------------

# \w 等价于 [a-zA-Z0-9_](字母、数字、下划线)
# 开启 Unicode 模式后,\w 的语义扩展为 [a-zA-Z0-9_] + 所有 Unicode 字母/数字(包括中文、日文、西里尔字母等)。
# 标点/空格作为天然分隔符:被 \w+ 自然跳过,不需要显式分词
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    pred = (prediction or "").strip().lower()
    for gold in gold_answers:
        if gold.strip().lower() == pred:
            return 1.0
    return 0.0


def answer_f1(prediction: str, gold_answers: list[str]) -> float:
    """Max token-F1 over all gold answers."""
    pred_tokens = _tokens(prediction)
    if not pred_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    best = 0.0
    for gold in gold_answers:
        gold_tokens = _tokens(gold)
        if not gold_tokens:
            continue
        gold_counter = Counter(gold_tokens)
        common = pred_counter & gold_counter
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def answer_recall(prediction: str, gold_answers: list[str]) -> float:
    """Fraction of gold-answer tokens present in the prediction (max over golds)."""
    pred_tokens = set(_tokens(prediction))
    best = 0.0
    for gold in gold_answers:
        gold_tokens = _tokens(gold)
        if not gold_tokens:
            continue
        covered = sum(1 for t in gold_tokens if t in pred_tokens)
        best = max(best, covered / len(gold_tokens))
    return best
