"""Benchmark shared utilities for the per-system entry-point scripts.

Split out of the original ``scripts/run_benchmark.py``: each entry-point script
serves a single RAG system, while the shared logic for data loading, evaluator
construction, the single-system evaluation loop and report writing is
centralized here to avoid duplication.
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
    """Inject the project root into ``sys.path`` so scripts can run standalone (without installing the package)."""
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))


def load_data(args: argparse.Namespace) -> tuple[list[Document], list[Query]]:
    """Load the dataset as (documents, queries) according to ``--format``."""
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
        # Auto-detect: split a single file into documents and queries.
        docs, queries = load_dataset(args.dataset)
    if args.passage_limit:
        docs = docs[: args.passage_limit]
    if args.query_limit:
        queries = queries[: args.query_limit]
    return docs, queries


def resolve_retriever_factory(spec: str | None) -> Callable[[Any], Any] | None:
    """Resolve a ``module:attr`` / ``module.attr`` string to a callable.

    Returns None when the spec is empty (the nuggetindex native Retriever is
    used); raises ValueError if it cannot be imported or the target is not
    callable.
    """
    if not spec:
        return None
    module_name, _, attr = spec.partition(":")
    if not attr:
        # Prefer "a.b.c:fn"; also accept "a.b.c.fn", splitting at the last ".".
        module_name, _, attr = spec.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"retriever_factory must be 'module:attr', got: {spec!r}")
    mod = importlib.import_module(module_name)
    obj = getattr(mod, attr)
    if not callable(obj):
        raise ValueError(f"retriever_factory target {spec!r} is not callable")
    return obj


def build_evaluator(cfg: AppConfig, llm: BaseLLM | None) -> Evaluator:
    """Construct the Evaluator from cfg (QA evaluation is enabled only when llm is available)."""
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
    """Run (ingest) → evaluate → close for a single system, with exception safety.

    With ``skip_ingest=True`` document ingest is skipped (used to reuse an already
    built index, e.g. nuggetindex's ``--reuse-db``) and ``docs`` may be empty; the
    system should already have opened the existing database during initialization.
    Returns the SystemReport; returns None on failure (the error has been printed).
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
    """Write reports into cfg.evaluation.output_dir and return (json_path, md_path).

    ``run_id`` is typically the UTC timestamp captured at benchmark entry startup,
    used to unify the filename prefix of all report files produced by this run
    (json / md / per_query CSV). When None, it falls back to the current time
    (backward compatibility).
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
    """Write a per-query detail CSV recording, for each query: query_id, the raw
    query text, reference_time, retrieval status, the system answer, the correct
    answers, retrieved_doc_ids and relevant_doc_ids.

    Retrieval status (retrieval_status) values:
      - 0: pq.retrieved_doc_ids is empty (nothing recalled at all)
      - 1: q.relevant_doc_ids and pq.retrieved_doc_ids have no intersection (relevant chunks did not make it into top_k)
      - 2: pq.retrieved_doc_ids contains at least one q.relevant_doc_ids (a relevant chunk was hit)

    list fields are joined with ``"; "`` for easy viewing in a spreadsheet. The
    report.per_query entries are aligned with the gold data (queries) by query_id.
    Returns None when there are no per_query results.
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
