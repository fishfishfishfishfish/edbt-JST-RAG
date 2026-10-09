"""Train a binary classifier that decides whether a document is suitable for
fact extraction directly with ``SpacyFactExtractor``.

Weakly supervised labels (based on extracted fact counts)
=========================================================

For each loaded passage, run both:

- :class:`tcrag.extractors.spacy_extractor.SpacyFactExtractor` (free, local);
- the native nuggetindex LLMExtractor pipeline (``llama3.2`` +
  ``OllamaCompatClient`` json_schema constrained decoding + the
  PlaceholderValidity wrapper, ``--max-tokens 4096``).

Labeling rule (``--label-gap-threshold``, default 0)::

    y = 1 (spacy-suitable)  when n_spacy >= n_llm
                             or n_llm - n_spacy <= label_gap_threshold
    y = 0 (negative)        otherwise

That is, spacy may extract up to ``label_gap_threshold`` fewer facts than the
LLM and the passage still counts as a positive; with threshold=0 the rule
degrades to the original ``n_spacy >= n_llm``. The cache stores only the
counts ``n_spacy`` / ``n_llm``, and labels are recomputed at training time with
the current threshold, so retraining with an adjusted threshold requires no
new LLM labeling calls.

Note: when the LLM extractor fails to parse, it returns an empty list in line
with production behavior (``n_llm=0``), and the passage is then labeled
positive — this follows the literal meaning of the rule and reflects the weakly
supervised interpretation that "spacy is not much worse than the LLM".

Two stages
==========

1. **label**: load passages with ``load_timeqa`` / ``load_tempevalrag``, run
   the two extractors concurrently, and **incrementally append** the results
   to the JSONL cache (including the raw text and counts); rerunning after an
   interruption skips passages already labeled, allowing long-term incremental
   accumulation.
2. **train**: read the cache → deduplicate by text_hash → stratified
   train/test split → TF-IDF (word 1-2gram + char 3-5gram) plus document
   statistics features → :class:`sklearn.linear_model.LogisticRegression` →
   print precision/recall/F1/confusion matrix → save with joblib into
   ``models/``.

Running (inside the tcrag conda environment, with ollama started)::

    conda activate tcrag

    # Small trial run (take 100 passages per dataset)
    python scripts/utils/train_spacy_suitability_classifier.py \
        --limit-per-dataset 100 --workers 3

    # Full incremental labeling (can be interrupted and resumed repeatedly; no training)
    python scripts/utils/train_spacy_suitability_classifier.py --label-only

    # Train from the cache only (no further LLM calls)
    python scripts/utils/train_spacy_suitability_classifier.py --no-label

    # Relax the positive rule (spacy extracting <=2 fewer facts still counts as positive), recompute labels directly from cached counts and retrain
    python scripts/utils/train_spacy_suitability_classifier.py \
        --no-label --label-gap-threshold 2

Loading at inference time (via ``load_classifier`` so the custom feature class is deserialized correctly)::

    import sys
    sys.path.insert(0, "scripts/utils")
    from train_spacy_suitability_classifier import load_classifier

    clf = load_classifier("models/spacy_suitability_logreg.joblib")
    clf.predict(["Gary Griffith is a citizen of Trinidad and Tobago."])
    clf.predict_proba([...])
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# When run directly as ``python scripts/utils/xxx.py``, the repository root is not on sys.path.
# scripts/utils/<file>.py: parents[0]=utils, parents[1]=scripts, parents[2]=repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The local Ollama service must bypass the system proxy (otherwise VPN/proxy setups intercept localhost requests).
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from tcrag.data.loader import load_tempevalrag, load_timeqa
from tcrag.data.models import Document
from tcrag.logging_config import get_logger, setup_logging

logger = get_logger("train_spacy_suitability")

_DEFAULT_TIMEQA = _REPO_ROOT / "data" / "TimeQA" / "annotated_dev.json"
_DEFAULT_TEMPEVALRAG = _REPO_ROOT / "data" / "TempEvalRAG"
_DEFAULT_CACHE = _REPO_ROOT / "data" / "spacy_suitability_labels.jsonl"
_DEFAULT_MODEL_OUT = _REPO_ROOT / "models" / "spacy_suitability_logreg.joblib"


# ── Features: document statistics ───────────────────────────────────────

class TextStats:
    """Shallow stylistic features extracted from plain text (spacy excels at simple
    SVO / nominal facts and produces markedly fewer facts on text with many
    pronouns, long sentences and low numeric density).

    Outputs 8 features: log character count, log word count, average word length,
    average sentence length, pronoun ratio, numeric-word ratio, capitalized-word
    ratio and comma density.
    """

    _PRONOUNS = {
        "he", "she", "they", "him", "her", "his", "their", "theirs",
        "it", "its", "them", "i", "we", "us", "our",
    }

    def fit(self, X: Any, y: Any = None) -> "TextStats":
        return self

    def transform(self, X: Any) -> Any:
        import numpy as np

        rows: list[list[float]] = []
        for text in X:
            text = str(text)
            words = text.lower().split()
            tokenish = [w.strip(".,;:!?\"'()[]") for w in words]
            tokenish = [w for w in tokenish if w]
            n_words = max(len(tokenish), 1)
            n_chars = max(len(text), 1)
            n_sents = max(text.count(".") + text.count("?") + text.count("!"), 1)
            n_pron = sum(1 for w in tokenish if w in self._PRONOUNS)
            n_digit = sum(1 for w in tokenish if any(c.isdigit() for c in w))
            n_capital = sum(1 for w in text.split() if len(w) > 1 and w[0].isupper())
            rows.append(
                [
                    float(np.log(n_chars)),
                    float(np.log(n_words)),
                    sum(len(w) for w in tokenish) / n_words,
                    n_words / n_sents,
                    n_pron / n_words,
                    n_digit / n_words,
                    n_capital / n_words,
                    text.count(",") / n_words,
                ]
            )
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        import numpy as np

        return np.asarray(
            [
                "log_chars",
                "log_words",
                "avg_word_len",
                "words_per_sentence",
                "pronoun_ratio",
                "number_ratio",
                "capitalized_ratio",
                "comma_density",
            ]
        )


# joblib deserialization compatibility: when the script runs directly,
# ``__name__ == "__main__"``, pickle records TextStats as ``__main__.TextStats``,
# which external code cannot resolve when loading. Pin its owning module name and
# register the current module in sys.modules so joblib.load works after ``import train_spacy_suitability_classifier``.
_STATS_MODULE = "train_spacy_suitability_classifier"
TextStats.__module__ = _STATS_MODULE
sys.modules.setdefault(_STATS_MODULE, sys.modules[__name__])


def load_classifier(model_path: str | Path = _DEFAULT_MODEL_OUT) -> Any:
    """Load a classifier trained and saved by this script (joblib).

    Usage::

        import sys
        sys.path.insert(0, "scripts/utils")
        from train_spacy_suitability_classifier import load_classifier

        clf = load_classifier()
        clf.predict([...])           # 1 = suitable for direct spacy extraction
        clf.predict_proba([...])
    """
    import joblib

    return joblib.load(model_path)


# ── Data loading ────────────────────────────────────────────────────────

def load_passages(
    *,
    datasets: list[str],
    timeqa_path: Path,
    tempevalrag_path: Path,
    limit_per_dataset: int,
) -> list[tuple[str, Document]]:
    """Load passages from the specified datasets, returning [(dataset_tag, Document), ...]."""
    out: list[tuple[str, Document]] = []
    if "timeqa" in datasets:
        logger.info("loading TimeQA passages from %s", timeqa_path)
        docs, _ = load_timeqa(timeqa_path)
        if limit_per_dataset > 0:
            docs = docs[:limit_per_dataset]
        logger.info("TimeQA passages: %d", len(docs))
        out.extend(("timeqa", d) for d in docs)
    if "tempevalrag" in datasets:
        logger.info("loading TempEvalRAG passages from %s", tempevalrag_path)
        docs, _ = load_tempevalrag(
            tempevalrag_path,
            passage_limit=limit_per_dataset if limit_per_dataset > 0 else None,
        )
        logger.info("TempEvalRAG passages: %d", len(docs))
        out.extend(("tempevalrag", d) for d in docs)
    return out


# ── Label cache ─────────────────────────────────────────────────────────

def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _cache_key(dataset: str, source_id: str) -> str:
    return f"{dataset}:{source_id}"


def load_label_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Read back the cache ``{cache_key: row}``; stale rows whose hash mismatches are discarded automatically."""
    rows: dict[str, dict[str, Any]] = {}
    if not cache_path.is_file():
        return rows
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key")
            if not key:
                continue
            # Hash differs from the current raw text → stale; leave it for relabeling in this run.
            if row.get("text_hash") != _text_hash(row.get("text", "")):
                continue
            rows[key] = row
    return rows


# ── Stage 1: labeling ───────────────────────────────────────────────────

def is_spacy_suitable(n_spacy: int, n_llm: int, gap_threshold: int = 0) -> int:
    """Weakly supervised positive rule: spacy extracts no fewer facts, or no more
    than ``gap_threshold`` fewer facts.

    Returns 1 (spacy-suitable) / 0. With threshold=0 it is equivalent to
    ``n_spacy >= n_llm``.
    """
    return int(n_spacy >= n_llm or (n_llm - n_spacy) <= gap_threshold)


def _build_nugget_llm_extractor(
    *,
    llm_model: str,
    llm_host: str,
    llm_max_tokens: int,
    llm_timeout: float,
    structured_compat: bool,
) -> Any:
    """Construct a nuggetindex LLM extractor with the same settings as the benchmark script.

    - ``structured_compat=True`` (default for llama3.2): ``OllamaCompatClient``,
      Ollama json_schema constrained decoding + output normalization;
    - False: the native nuggetindex instructor client (strong models such as qwen3);
    - the outer layer is always ``PlaceholderValidityLLMExtractor`` (affects only
      validity, not the fact count).
    """
    from nuggetindex.extractors import LLMConfig, build_client
    from tcrag.extractors.ni_llm_placeholder import (
        PlaceholderValidityLLMExtractor,
    )

    ni_cfg = LLMConfig(
        provider="ollama",
        model=llm_model,
        base_url=f"{llm_host.rstrip('/')}/v1",
        max_tokens=llm_max_tokens,
        timeout_seconds=llm_timeout,
    )
    if structured_compat:
        from tcrag.extractors.ni_ollama_compat import OllamaCompatClient

        client = OllamaCompatClient(ni_cfg)
    else:
        client = build_client(ni_cfg)
    return PlaceholderValidityLLMExtractor(ni_cfg, client=client)


async def label_passages(
    passages: list[tuple[str, Document]],
    *,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    spacy_model: str,
    llm_model: str,
    llm_host: str,
    llm_max_tokens: int,
    llm_timeout: float,
    structured_compat: bool,
    max_facts: int,
    workers: int,
    label_gap_threshold: int = 0,
) -> None:
    """Run the two extractors on cache-missing passages and incrementally write them into the cache."""
    from tcrag.extractors.spacy_extractor import SpacyFactExtractor

    todo: list[tuple[str, Document]] = []
    for dataset, doc in passages:
        key = _cache_key(dataset, doc.source_id)
        cached = cache.get(key)
        if cached is not None and cached.get("text_hash") == _text_hash(doc.text):
            continue
        todo.append((dataset, doc))

    if not todo:
        logger.info("所有 passage 均已在标注缓存中,跳过 LLM 调用")
        return

    logger.info(
        "待标注 passage: %d(缓存已有 %d);spacy=%s, llm=%s@%s, workers=%d, "
        "正样本规则: n_spacy>=n_llm 或 gap<=%d",
        len(todo),
        len(cache),
        spacy_model,
        llm_model,
        llm_host,
        workers,
        label_gap_threshold,
    )

    spacy_ext = SpacyFactExtractor(model=spacy_model, max_facts=max_facts)
    # Load the model up front to avoid lazy-loading races across multiple threads.
    spacy_ext._ensure_nlp()
    # Production pipeline: small models such as llama3.2 go through
    # OllamaCompatClient (json_schema constrained decoding) wrapped by the
    # PlaceholderValidity wrapper; the fact-count semantics are identical to the benchmark run.
    llm_ext = _build_nugget_llm_extractor(
        llm_model=llm_model,
        llm_host=llm_host,
        llm_max_tokens=llm_max_tokens,
        llm_timeout=llm_timeout,
        structured_compat=structured_compat,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()
    spacy_lock = asyncio.Lock()
    sem = asyncio.Semaphore(workers)

    state = {"done": 0, "pos": 0, "neg": 0, "errors": 0}
    start = time.monotonic()
    stop_progress = asyncio.Event()

    async def _progress_loop() -> None:
        total = len(todo)
        while not stop_progress.is_set():
            await asyncio.sleep(30)
            done = state["done"]
            elapsed = time.monotonic() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            remain = total - done
            eta = remain / rate if rate > 0 else 0.0
            logger.info(
                "labeling progress: %d/%d (%.1f%%), pos=%d neg=%d errors=%d, "
                "rate %.2f docs/s, elapsed %.0fs, eta %.0fs",
                done,
                total,
                100.0 * done / total,
                state["pos"],
                state["neg"],
                state["errors"],
                rate,
                elapsed,
                eta,
            )

    async def _process_one(dataset: str, doc: Document) -> None:
        from tcrag.extractors.llm_extractor import _to_atomic_facts

        async with sem:
            key = _cache_key(dataset, doc.source_id)
            try:
                # spaCy parsing shares one model, so conservatively serialize it (LLM waiting dominates the runtime).
                async with spacy_lock:
                    spacy_facts = await spacy_ext.aextract(
                        doc.text, source_id=doc.source_id
                    )
            except Exception as exc:  # noqa: BLE001 - a single-document failure must not affect the whole batch
                state["done"] += 1
                state["errors"] += 1
                logger.warning("spacy 标注失败(不写入缓存,下次续跑重试) %s: %r", key, exc)
                return

            try:
                results = await llm_ext.aextract(
                    doc.text, source_id=doc.source_id
                )
            except ValueError as exc:
                # Consistent with tcrag LLMExtractor: failure to parse structured output → degrade to 0 facts.
                logger.warning("LLM 输出无法解析,按 0 事实计 %s: %s", key, exc)
                results = []
            except Exception as exc:  # noqa: BLE001 - truncation/connection errors etc. are left for retry on resume
                state["done"] += 1
                state["errors"] += 1
                logger.warning("LLM 标注失败(不写入缓存,下次续跑重试) %s: %r", key, exc)
                return
            llm_facts = _to_atomic_facts(results, doc.text)
            if max_facts > 0:
                llm_facts = llm_facts[:max_facts]

            n_spacy = len(spacy_facts)
            n_llm = len(llm_facts)
            label = is_spacy_suitable(n_spacy, n_llm, label_gap_threshold)
            row = {
                "key": key,
                "dataset": dataset,
                "source_id": doc.source_id,
                "text_hash": _text_hash(doc.text),
                "n_spacy": n_spacy,
                "n_llm": n_llm,
                "label": label,
                "label_gap_threshold": label_gap_threshold,
                "text": doc.text,
            }
            async with write_lock:
                with cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                cache[key] = row
            state["done"] += 1
            if label:
                state["pos"] += 1
            else:
                state["neg"] += 1

    progress_task = asyncio.create_task(_progress_loop())
    try:
        await asyncio.gather(
            *(_process_one(ds, doc) for ds, doc in todo),
            return_exceptions=False,
        )
    finally:
        stop_progress.set()
        await progress_task

    logger.info(
        "标注完成:新增 %d(pos=%d neg=%d),失败 %d,缓存总计 %d",
        state["done"],
        state["pos"],
        state["neg"],
        state["errors"],
        len(cache),
    )


# ── Stage 2: training ───────────────────────────────────────────────────

def train_from_cache(
    cache: dict[str, dict[str, Any]],
    *,
    model_out: Path,
    test_size: float,
    random_state: int,
    label_gap_threshold: int = 0,
) -> None:
    """Read the cache, deduplicate → split → train LogisticRegression → evaluate and save.

    Labels are recomputed from ``n_spacy`` / ``n_llm`` with the current
    ``label_gap_threshold`` rather than using the ``label`` field written into
    each cache row, so retraining with a different threshold requires no
    relabeling.
    """
    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import StandardScaler

    # Deduplicate by text_hash (the same passage may repeat across datasets; avoid train/test leakage).
    unique: dict[str, dict[str, Any]] = {}
    for row in cache.values():
        unique.setdefault(row["text_hash"], row)

    rows = list(unique.values())
    texts = [r["text"] for r in rows]
    y = np.asarray(
        [
            is_spacy_suitable(
                int(r.get("n_spacy", 0)),
                int(r.get("n_llm", 0)),
                label_gap_threshold,
            )
            for r in rows
        ],
        dtype=int,
    )

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    logger.info(
        "训练语料(按文本去重后):%d passages,正样本(spacy,gap<=%d)=%d,负样本=%d",
        len(y),
        label_gap_threshold,
        n_pos,
        n_neg,
    )
    if len(y) < 10:
        raise RuntimeError(f"标注样本过少({len(y)}),无法训练;请先积累标注缓存")
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"只有单一类别(pos={n_pos}, neg={n_neg}),无法训练二分类器"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "split: train=%d test=%d (test_size=%.2f, random_state=%d)",
        len(X_train),
        len(X_test),
        test_size,
        random_state,
    )

    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=50000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=5,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            (
                "stats",
                Pipeline(
                    [
                        ("extract", TextStats()),
                        ("scale", StandardScaler()),
                    ]
                ),
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    logger.info("fitting LogisticRegression ...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["0 (use LLM)", "1 (use spacy)"],
        digits=4,
    )
    cm = confusion_matrix(y_test, y_pred)
    logger.info("test set 分类报告:\n%s", report)
    logger.info("混淆矩阵 (rows=true, cols=pred) [LLM, spacy]:\n%s", cm)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)
    logger.info("模型已保存: %s", model_out)

    meta = {
        "model": str(model_out),
        "label_rule": (
            "1 (spacy) if n_spacy_facts >= n_llm_facts "
            "or n_llm_facts - n_spacy_facts <= label_gap_threshold else 0"
        ),
        "label_gap_threshold": label_gap_threshold,
        "n_unique_passages": len(y),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
        "classifier": "sklearn.linear_model.LogisticRegression",
        "classifier_params": {
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
        },
        "features": ["word_tfidf(1,2)", "char_tfidf(3,5)", "doc_stats(8)"],
    }
    meta_path = model_out.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("元信息已保存: %s", meta_path)


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="训练 LogisticRegression 判断 passage 是否适合直接用 spacy 提取事实",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--timeqa", type=Path, default=_DEFAULT_TIMEQA,
                   help="TimeQA annotated_dev.json 路径")
    p.add_argument("--tempevalrag", type=Path, default=_DEFAULT_TEMPEVALRAG,
                   help="TempEvalRAG 目录(含 docs.jsonl)")
    p.add_argument("--datasets", type=str, default="timeqa,tempevalrag",
                   help="逗号分隔,可选 timeqa / tempevalrag")
    p.add_argument("--limit-per-dataset", type=int, default=0,
                   help="每个数据集最多加载的 passage 数,0 表示全部")
    p.add_argument("--workers", type=int, default=3,
                   help="并发标注协程数(ollama 并发请求数)")
    p.add_argument("--llm-model", type=str, default="llama3.2",
                   help="用于标注的 Ollama 模型名")
    p.add_argument("--llm-host", type=str, default="http://localhost:11434")
    p.add_argument("--llm-max-tokens", type=int, default=4096,
                   help="LLM 生成上限(与基准脚本一致,默认 4096)")
    p.add_argument("--llm-timeout", type=float, default=180.0,
                   help="单次 LLM 请求超时秒数")
    p.add_argument("--structured-compat", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="llama3.2 等小模型用 OllamaCompatClient 约束解码;"
                        "强模型可 --no-structured-compat")
    p.add_argument("--spacy-model", type=str, default="en_core_web_sm")
    p.add_argument("--max-facts", type=int, default=20,
                   help="两个提取器共同使用的事实上限,0 表示不限")
    p.add_argument("--label-gap-threshold", type=int, default=0,
                   help="正样本容差:n_spacy >= n_llm 或 n_llm - n_spacy "
                        "<= 该值即判为适合 spacy;0 表示严格 n_spacy >= n_llm。"
                        "训练时按此值由缓存计数重算标签,无需重新标注")
    p.add_argument("--label-cache", type=Path, default=_DEFAULT_CACHE,
                   help="增量标注缓存 JSONL 路径")
    p.add_argument("--model-out", type=Path, default=_DEFAULT_MODEL_OUT,
                   help="joblib 模型输出路径")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--label-only", action="store_true",
                   help="只做标注,不训练")
    p.add_argument("--no-label", action="store_true",
                   help="跳过标注,直接用现有缓存训练")
    p.add_argument("--log-file", type=Path, default=None,
                   help="可选:日志同时写入该文件(默认只输出到 stderr)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(
        level="INFO",
        log_file=args.log_file,
        console=True,
    )
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    for s in datasets:
        if s not in {"timeqa", "tempevalrag"}:
            raise SystemExit(f"未知数据集: {s}(可选 timeqa / tempevalrag)")
    if args.label_gap_threshold < 0:
        raise SystemExit(
            f"--label-gap-threshold 必须 >= 0(当前 {args.label_gap_threshold})"
        )

    cache = load_label_cache(args.label_cache)
    logger.info("标注缓存: %s(已有 %d 条)", args.label_cache, len(cache))

    if not args.no_label:
        passages = load_passages(
            datasets=datasets,
            timeqa_path=args.timeqa,
            tempevalrag_path=args.tempevalrag,
            limit_per_dataset=args.limit_per_dataset,
        )
        if not passages:
            raise SystemExit("没有加载到任何 passage")
        asyncio.run(
            label_passages(
                passages,
                cache=cache,
                cache_path=args.label_cache,
                spacy_model=args.spacy_model,
                llm_model=args.llm_model,
                llm_host=args.llm_host,
                llm_max_tokens=args.llm_max_tokens,
                llm_timeout=args.llm_timeout,
                structured_compat=args.structured_compat,
                max_facts=args.max_facts,
                workers=args.workers,
                label_gap_threshold=args.label_gap_threshold,
            )
        )

    if args.label_only:
        logger.info("--label-only: 跳过训练")
        return

    train_from_cache(
        cache,
        model_out=args.model_out,
        test_size=args.test_size,
        random_state=args.random_state,
        label_gap_threshold=args.label_gap_threshold,
    )


if __name__ == "__main__":
    main()
