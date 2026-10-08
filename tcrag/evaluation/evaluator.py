"""Benchmark evaluator.

Runs a set of queries against one or more RAG systems, collects retrieval
and answer-quality metrics plus system performance (latency, throughput),
and returns a structured report.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from tcrag.config import LLMConfig, RagasConfig
from tcrag.data.models import Query
from tcrag.evaluation.metrics import (
    answer_f1,
    answer_recall,
    exact_match,
    f1_at_k,
    precision_at_k,
    recall_at_k,
)
from tcrag.llm.base import BaseLLM
from tcrag.logging_config import get_logger
from tcrag.pipeline import QAPipeline
from tcrag.rag_systems.base import BaseRAGSystem

logger = get_logger("evaluation.evaluator")


@dataclass
class QueryResult:
    query_id: str
    retrieved_doc_ids: list[str]
    answer: str
    latency_seconds: float
    # 分段耗时:retrieval 为纯检索,llm 为纯生成(--no-qa 模式下为 0.0)。
    retrieval_latency_seconds: float = 0.0
    llm_latency_seconds: float = 0.0
    retrieval_metrics: dict[str, float] = field(default_factory=dict)
    answer_metrics: dict[str, float] = field(default_factory=dict)
    ragas_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class SystemReport:
    system_name: str
    ingest_summary: dict[str, Any]
    retrieval: dict[str, float] = field(default_factory=dict)
    answer: dict[str, float] = field(default_factory=dict)
    performance: dict[str, float] = field(default_factory=dict)
    per_query: list[QueryResult] = field(default_factory=list)
    ragas: dict[str, float] = field(default_factory=dict)
    # 存储占用(字节):输入数据文件大小与落盘索引大小,由基准入口脚本填充。
    # 约定键:dataset_file/dataset_bytes、queries_file/queries_bytes、
    # index_path/index_bytes。
    storage: dict[str, Any] = field(default_factory=dict)
    # 运行参数与关键配置快照(CLI args + cfg 摘要),由基准入口脚本填充,
    # reporter 写入 md 报告顶部 "Run Configuration" 小节,便于复现。
    run_meta: dict[str, Any] = field(default_factory=dict)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100
    lo, hi = int(idx), min(int(idx) + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


class _RagasScorerSet:
    """懒构建并缓存 ragas 评判 scorers,实现三级降级。

    - ragas 未安装 → available=False,跳过全部 ragas 指标
    - 已安装但无 OPENAI_API_KEY → llm_ready=False,仅运行 RougeScore/BleuScore
    - 已安装 + 有 key → 全部运行
    """

    def __init__(self, ragas_cfg: RagasConfig, llm_cfg: LLMConfig) -> None:
        self._ragas_cfg = ragas_cfg
        self._llm_cfg = llm_cfg
        self.available = False
        self.llm_ready = False
        self._llm = None
        self._embeddings = None
        self._client = None
        self._scorers: dict[str, Any] = {}

    def _try_import(self) -> bool:
        try:
            from ragas.metrics.collections import (  # noqa: F401
                AnswerCorrectness,
                BleuScore,
                ContextPrecision,
                ContextRecall,
                RougeScore,
            )
            return True
        except ImportError:
            return False

    async def _ensure_llm(self) -> None:
        """懒构建 AsyncOpenAI client + llm_factory + embedding_factory。"""
        if self._llm is not None:
            return
        from openai import AsyncOpenAI

        api_key = self._llm_cfg.api_key
        if not api_key:
            return
        model = self._ragas_cfg.judge_llm_model or self._llm_cfg.model
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self._llm_cfg.timeout,
}
        if self._llm_cfg.base_url:
            client_kwargs["base_url"] = self._llm_cfg.base_url
        self._client = AsyncOpenAI(**client_kwargs)

        from ragas.embeddings import embedding_factory
        from ragas.llms import llm_factory

        self._llm = llm_factory(model, client=self._client)
        self._embeddings = embedding_factory(
            self._ragas_cfg.embedding_provider,
            model=self._ragas_cfg.embedding_model,
            client=self._client,
        )
        self.llm_ready = True

    async def get_scorers(self) -> dict[str, Any]:
        """返回 {metric_name: scorer} 字典,按 config 开关过滤。"""
        if not self.available:
            return {}
        if self._scorers:
            return self._scorers
        from ragas.metrics.collections import (
            AnswerCorrectness,
            BleuScore,
            ContextPrecision,
            ContextRecall,
            RougeScore,
        )

        cfg = self._ragas_cfg
        if cfg.rouge:
            self._scorers["rouge"] = RougeScore()
        if cfg.bleu:
            self._scorers["bleu"] = BleuScore()
        if cfg.context_recall or cfg.context_precision or cfg.answer_correctness:
            await self._ensure_llm()
            if self.llm_ready:
                if cfg.context_recall:
                    self._scorers["context_recall"] = ContextRecall(llm=self._llm)
                if cfg.context_precision:
                    self._scorers["context_precision"] = ContextPrecision(llm=self._llm)
                if cfg.answer_correctness:
                    self._scorers["answer_correctness"] = AnswerCorrectness(
                        llm=self._llm, embeddings=self._embeddings,
                    )
        return self._scorers

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


class Evaluator:
    """Run the benchmark and aggregate metrics per system."""

    def __init__(
        self,
        *,
        retrieval_k: list[int] | None = None,
        evaluate_qa: bool = True,
        evaluate_perf: bool = True,
        ragas_cfg: RagasConfig | None = None,
        llm_cfg: LLMConfig | None = None,
    ) -> None:
        self._k_values = retrieval_k or [5, 10]
        self._eval_qa = evaluate_qa
        self._eval_perf = evaluate_perf
        self._ragas_set: _RagasScorerSet | None = None
        if ragas_cfg and ragas_cfg.enabled and llm_cfg:
            self._ragas_set = _RagasScorerSet(ragas_cfg, llm_cfg)
            self._ragas_set.available = self._ragas_set._try_import()
            if not self._ragas_set.available:
                logger.warning("ragas 未安装,跳过 ragas 指标;pip install ragas>=0.4.0 启用")
                self._ragas_set = None

    async def aclose(self) -> None:
        if self._ragas_set is not None:
            await self._ragas_set.aclose()

    async def evaluate_system(
        self,
        system: BaseRAGSystem,
        queries: list[Query],
        *,
        llm: BaseLLM | None = None,
        top_k: int = 10,
        ctx_top_k: int | None = None,
        context_token_budget: int = 3000,
    ) -> SystemReport:
        ingest_summary: dict[str, Any] = {}
        # Ingestion is performed by the caller before evaluation; the system
        # may carry a precomputed summary via an attribute.
        ingest_summary = getattr(system, "_last_ingest_summary", {}) or {}

        pipeline = (
            QAPipeline(
                system, llm,
                top_k=top_k, ctx_top_k=ctx_top_k,
                context_token_budget=context_token_budget,
            )
            if llm else None
        )

        per_query: list[QueryResult] = []
        latencies: list[float] = []
        retrieval_latencies: list[float] = []
        llm_latencies: list[float] = []
        retrieval_scores: dict[int, list[tuple[float, float, float]]] = {k: [] for k in self._k_values}
        em_scores: list[float] = []
        f1_scores: list[float] = []
        recall_scores: list[float] = []

        # 评估进度日志:每 eval_log_interval_queries 个 query 或至少间隔
        # eval_log_interval_seconds 秒打一次,避免长批次(含 LLM 问答,单 query
        # 可达数秒)运行时无任何输出。
        eval_log_interval_queries = 20
        eval_log_interval_seconds = 120.0
        eval_start = time.perf_counter()
        last_log = eval_start
        n_queries = len(queries)

        for i, q in enumerate(queries, 1):
            t0 = time.perf_counter()
            if pipeline is not None:
                qa = await pipeline.aanswer(q)
                retrieved_ids = [h.doc_id for h in qa.retrieved]
                answer = qa.answer
                retrieval_lat = qa.retrieval_latency_seconds
                llm_lat = qa.llm_latency_seconds
            else:
                hits = await system.aretrieve(q.text, top_k=top_k, reference_time=q.reference_time)
                retrieved_ids = [h.doc_id for h in hits]
                answer = ""
                # --no-qa:无 LLM 阶段,总耗时即检索耗时(下方统一赋值)
                retrieval_lat = 0.0
                llm_lat = 0.0
            latency = time.perf_counter() - t0
            latencies.append(latency)
            # retrieval-only 模式下总耗时即检索耗时;pipeline 模式用分段计时。
            retrieval_latencies.append(
                retrieval_lat if pipeline is not None else latency
            )
            if pipeline is not None:
                llm_latencies.append(llm_lat)

            relevant = set(q.relevant_doc_ids)
            ret_metrics: dict[str, float] = {}
            for k in self._k_values:
                p = precision_at_k(retrieved_ids, relevant, k)
                r = recall_at_k(retrieved_ids, relevant, k)
                f = f1_at_k(retrieved_ids, relevant, k)
                ret_metrics[f"precision@{k}"] = p
                ret_metrics[f"recall@{k}"] = r
                ret_metrics[f"f1@{k}"] = f
                retrieval_scores[k].append((p, r, f))

            ans_metrics: dict[str, float] = {}
            if self._eval_qa and q.answers:
                em = exact_match(answer, q.answers)
                f1 = answer_f1(answer, q.answers)
                ar = answer_recall(answer, q.answers)
                ans_metrics = {"exact_match": em, "answer_f1": f1, "answer_recall": ar}
                em_scores.append(em)
                f1_scores.append(f1)
                recall_scores.append(ar)

            # --- ragas 指标(context recall/precision, rouge/bleu, answer correctness)---
            ragas_metrics: dict[str, float] = {}
            if self._ragas_set is not None and q.answers:
                reference = q.answers[0]
                retrieved_hits = qa.retrieved if pipeline is not None else hits
                retrieved_contexts = [h.text for h in retrieved_hits]
                scorers = await self._ragas_set.get_scorers()
                for name, scorer in scorers.items():
                    try:
                        if name in ("rouge", "bleu"):
                            if not answer:
                                continue
                            res = await scorer.ascore(response=answer, reference=reference)
                        elif name in ("context_recall", "context_precision"):
                            res = await scorer.ascore(
                                user_input=q.text,
                                retrieved_contexts=retrieved_contexts,
                                reference=reference,
                            )
                        elif name == "answer_correctness":
                            if not answer:
                                continue
                            res = await scorer.ascore(
                                user_input=q.text,
                                response=answer,
                                reference=reference,
                            )
                        else:
                            continue
                        ragas_metrics[name] = float(res.value)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("ragas 指标 %s 对 query %s 失败: %s", name, q.id, exc)
                        ragas_metrics[name] = 0.0

            per_query.append(QueryResult(
                query_id=q.id,
                retrieved_doc_ids=retrieved_ids,
                answer=answer,
                latency_seconds=latency,
                retrieval_latency_seconds=retrieval_latencies[-1],
                llm_latency_seconds=llm_lat if pipeline is not None else 0.0,
                retrieval_metrics=ret_metrics,
                answer_metrics=ans_metrics,
                ragas_metrics=ragas_metrics,
            ))

            now = time.perf_counter()
            if i % eval_log_interval_queries == 0 or now - last_log >= eval_log_interval_seconds:
                elapsed = now - eval_start
                avg_latency = statistics.fmean(latencies)
                # 用配置中的首个 k(如 5)展示当前滚动 recall
                k0 = self._k_values[0]
                running_recall = (
                    statistics.fmean(t[1] for t in retrieval_scores[k0])
                    if retrieval_scores[k0] else 0.0
                )
                eta = elapsed / i * (n_queries - i) if i else 0.0
                logger.info(
                    "evaluation progress: %d/%d queries (%.0f%%), elapsed %.0fs, "
                    "avg latency %.0fms, running recall@%d %.4f, eta %.0fs",
                    i, n_queries, 100.0 * i / n_queries, elapsed,
                    avg_latency * 1000, k0, running_recall, eta,
                )
                last_log = now

        # Aggregate retrieval metrics
        retrieval_agg: dict[str, float] = {}
        for k in self._k_values:
            triples = retrieval_scores[k]
            if triples:
                retrieval_agg[f"precision@{k}"] = statistics.fmean(t[0] for t in triples)
                retrieval_agg[f"recall@{k}"] = statistics.fmean(t[1] for t in triples)
                retrieval_agg[f"f1@{k}"] = statistics.fmean(t[2] for t in triples)

        answer_agg: dict[str, float] = {}
        if self._eval_qa:
            if em_scores:
                answer_agg["exact_match"] = statistics.fmean(em_scores)
            if f1_scores:
                answer_agg["answer_f1"] = statistics.fmean(f1_scores)
            if recall_scores:
                answer_agg["answer_recall"] = statistics.fmean(recall_scores)

        perf_agg: dict[str, float] = {}
        if self._eval_perf and latencies:
            total = sum(latencies)
            perf_agg = {
                "n_queries": len(latencies),
                "mean_latency_ms": statistics.fmean(latencies) * 1000,
                "p50_latency_ms": _percentile(latencies, 50) * 1000,
                "p95_latency_ms": _percentile(latencies, 95) * 1000,
                "throughput_qps": len(latencies) / total if total > 0 else 0.0,
                "total_seconds": total,
            }
            # 分段耗时:检索段(两种模式都有)与 LLM 生成段(仅 QA 模式)。
            if retrieval_latencies:
                perf_agg.update({
                    "retrieval_mean_latency_ms": statistics.fmean(retrieval_latencies) * 1000,
                    "retrieval_p50_latency_ms": _percentile(retrieval_latencies, 50) * 1000,
                    "retrieval_p95_latency_ms": _percentile(retrieval_latencies, 95) * 1000,
                })
            if llm_latencies:
                perf_agg.update({
                    "llm_mean_latency_ms": statistics.fmean(llm_latencies) * 1000,
                    "llm_p50_latency_ms": _percentile(llm_latencies, 50) * 1000,
                    "llm_p95_latency_ms": _percentile(llm_latencies, 95) * 1000,
                })

        ragas_agg: dict[str, float] = {}
        if self._ragas_set is not None:
            for metric_name in (await self._ragas_set.get_scorers()):
                vals = [
                    pq.ragas_metrics.get(metric_name)
                    for pq in per_query
                    if pq.ragas_metrics.get(metric_name) is not None
                ]
                if vals:
                    ragas_agg[metric_name] = statistics.fmean(vals)

        return SystemReport(
            system_name=system.name,
            ingest_summary=ingest_summary,
            retrieval=retrieval_agg,
            answer=answer_agg,
            performance=perf_agg,
            per_query=per_query,
            ragas=ragas_agg,
        )
