"""Benchmark shared utilities for the per-system entry-point scripts.

拆分自原 ``scripts/run_benchmark.py``:每个入口脚本只服务一个 RAG 系统,
共享的数据加载、evaluator 构造、单系统评估循环、报告写入等逻辑集中在此,
避免重复。
"""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from tcrag.config import AppConfig
from tcrag.data.loader import (
    load_dataset,
    load_documents,
    load_queries,
    load_timeqa,
    load_ravine, 
    load_tempevalrag, 
    load_situatedqa
)
from tcrag.data.models import Document, Query
from tcrag.evaluation.evaluator import Evaluator, SystemReport
from tcrag.evaluation.reporter import write_json_report, write_markdown_report
from tcrag.llm.base import BaseLLM
from tcrag.rag_systems.base import BaseRAGSystem

__all__ = [
    "setup_path",
    "load_data",
    "resolve_retriever_factory",
    "build_evaluator",
    "run_single_eval",
    "write_reports",
    "write_per_query_details",
]


def setup_path() -> None:
    """把项目根目录注入 ``sys.path``,让脚本可裸跑(未安装包)。"""
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))


def load_data(args: argparse.Namespace) -> tuple[list[Document], list[Query]]:
    """按 ``--format`` 加载数据集为 (documents, queries)。"""
    fmt = (args.format or "auto").lower()
    if fmt == "timeqa":
        return load_timeqa(
            args.dataset,
            passage_limit=args.passage_limit,
            query_limit=args.query_limit,
        )
    if fmt == "ravine":
        return load_ravine(
            Path(args.dataset),
            passage_limit=args.passage_limit,
            query_limit=args.query_limit
        )
    if fmt == "tempevalrag":
        return load_tempevalrag(
            Path(args.dataset),
            passage_limit=args.passage_limit,
            query_limit=args.query_limit
        )
    if fmt == "situatedqa":
        return load_situatedqa(
            Path(args.dataset),
            passage_limit=args.passage_limit,
            query_limit=args.query_limit
        )
    if args.queries:
        docs = load_documents(args.dataset)
        queries = load_queries(args.queries)
    else:
        # 自动检测:把单个文件拆成 documents 与 queries。
        docs, queries = load_dataset(args.dataset)
    if args.passage_limit:
        docs = docs[: args.passage_limit]
    if args.query_limit:
        queries = queries[: args.query_limit]
    return docs, queries


def resolve_retriever_factory(spec: str | None) -> Callable[[Any], Any] | None:
    """解析 ``module:attr`` / ``module.attr`` 字符串为 callable。

    为空时返回 None(使用 nuggetindex 原生 Retriever);无法导入或不可调用
    时抛 ValueError。
    """
    if not spec:
        return None
    module_name, _, attr = spec.partition(":")
    if not attr:
        # "a.b.c:fn" 优先;兼容 "a.b.c.fn",以最后一个 "." 作分隔。
        module_name, _, attr = spec.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"retriever_factory must be 'module:attr', got: {spec!r}")
    mod = importlib.import_module(module_name)
    obj = getattr(mod, attr)
    if not callable(obj):
        raise ValueError(f"retriever_factory target {spec!r} is not callable")
    return obj


def build_evaluator(cfg: AppConfig, llm: BaseLLM | None) -> Evaluator:
    """按 cfg 构造 Evaluator(QA 评估仅在 llm 可用时启用)。"""
    return Evaluator(
        retrieval_k=cfg.evaluation.retrieval_k,
        evaluate_qa=cfg.evaluation.qa and llm is not None,
        evaluate_perf=cfg.evaluation.system_perf,
        ragas_cfg=cfg.evaluation.ragas,
        llm_cfg=cfg.llm,
    )


async def run_single_eval(
    system: BaseRAGSystem,
    docs: list[Document],
    queries: list[Query],
    evaluator: Evaluator,
    llm: BaseLLM | None,
    cfg: AppConfig,
    skip_ingest: bool = False,
) -> SystemReport | None:
    """对单个系统执行 (ingest) → evaluate → close,带异常兜底。

    ``skip_ingest=True`` 时跳过 document ingest(用于复用已构建好的索引库,
    如 nuggetindex 的 ``--reuse-db``),``docs`` 可为空;system 应已在初始化
    时打开已有库。返回 SystemReport;失败时返回 None(已打印错误)。
    """
    name = system.name
    if not skip_ingest:
        try:
            ingest_summary = await system.aingest_documents(docs)
            system._last_ingest_summary = ingest_summary  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR ingesting {name}: {exc}", file=sys.stderr)
            return None

    try:
        report = await evaluator.evaluate_system(
            system,
            queries,
            llm=llm,
            top_k=cfg.rag.top_k,
            ctx_top_k=cfg.rag.ctx_top_k,
            context_token_budget=cfg.rag.context_token_budget,
        )
        print(f"  retrieval: {report.retrieval}")
        if report.answer:
            print(f"  answer:    {report.answer}")
        if report.performance:
            print(f"  perf:      {report.performance}")
        return report
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR evaluating {name}: {exc}", file=sys.stderr)
        return None
    finally:
        await system.aclose()


def write_reports(
    reports: list[SystemReport], cfg: AppConfig, run_id: str | None = None,
) -> tuple[Path, Path]:
    """把报告写入 cfg.evaluation.output_dir,返回 (json_path, md_path)。

    ``run_id`` 通常为基准入口启动时捕获的 UTC 时间戳,用于统一本次运行
    产出的所有报告文件(json / md / per_query CSV)的文件名前缀。
    传 None 时回退到当前时间(向后兼容)。
    """
    out_dir = Path(cfg.evaluation.output_dir)
    ts = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = write_json_report(reports, out_dir / f"report_{ts}.json")
    md_path = write_markdown_report(reports, out_dir / f"report_{ts}.md")
    print(f"\nReports written: {json_path} , {md_path}")
    return json_path, md_path


def write_per_query_details(
    report: SystemReport,
    queries: list[Query],
    cfg: AppConfig,
    run_id: str | None = None,
) -> Path | None:
    """写入 per-query 明细 CSV,记录每个 query 的:query_id、query 原文 text、
    reference_time、检索状态、系统答案、正确答案、retrieved_doc_ids、
    relevant_doc_ids。

    检索状态(retrieval_status)取值:
      - 0: pq.retrieved_doc_ids 为空(完全未召回)
      - 1: q.relevant_doc_ids 与 pq.retrieved_doc_ids 无交集(相关 chunk 未进 top_k)
      - 2: pq.retrieved_doc_ids 包含至少一个 q.relevant_doc_ids(命中相关 chunk)

    list 字段以 ``"; "`` 连接,便于在表格中查看。按 query_id 把 report.per_query
    与 gold(queries)对齐。无 per_query 结果时返回 None。
    """
    if not report.per_query:
        return None
    out_dir = Path(cfg.evaluation.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"report_{ts}_per_query.csv"
    qmap = {q.id: q for q in queries}
    fieldnames = [
        "query_id", "query_text", "reference_time", "retrieval_status",
        "answer", "correct_answers",
        "retrieved_doc_ids", "relevant_doc_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pq in report.per_query:
            q = qmap.get(pq.query_id)
            retrieved_set = set(pq.retrieved_doc_ids)
            if not pq.retrieved_doc_ids:
                retrieval_status = 0
            elif q is None:
                retrieval_status = ""
            else:
                relevant_set = set(q.relevant_doc_ids)
                retrieval_status = 2 if relevant_set & retrieved_set else 1
            ref_time = q.reference_time if q else None
            writer.writerow({
                "query_id": pq.query_id,
                "query_text": q.text if q else "",
                "reference_time": ref_time.isoformat() if ref_time else "",
                "retrieval_status": retrieval_status,
                "answer": pq.answer,
                "correct_answers": "; ".join(q.answers) if q else "",
                "retrieved_doc_ids": "; ".join(pq.retrieved_doc_ids),
                "relevant_doc_ids": "; ".join(q.relevant_doc_ids) if q else "",
            })
    print(f"Per-query details written: {path}")
    return path
