# -*- coding: utf-8 -*-
"""Entry point for the TCRag nuggetindex × SpacyLLMHybridConstructor benchmark.

Single-system entry point: the document constructor registered under the name
``"spacy_llm_hybrid"`` is hard-wired (the LR classifier routes documents to
the two extractor branches, spaCy / LLM), while the retriever / db_path /
extractor are still determined by the YAML configuration and the CLI. This
entry point ships with its own complete argparse and run logic; the shared
evaluation orchestration is reused from ``_benchmark_common``.

Example:
  python scripts/run_benchmark_jstrag.py \
      --dataset data/timeqa_annotated_dev.json \
      --format timeqa --passage-limit 50 --query-limit 20
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import os
import sys
from pathlib import Path
from typing import Any

# Inject the project root + scripts directory so _benchmark_common and tcrag can be imported directly (the script needs no installed package).
_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _benchmark_common import (
    build_evaluator, load_data, resolve_retriever_factory,
    run_single_eval, write_per_query_details, write_reports,
)
from tcrag.config import load_config
from tcrag.llm.factory import create_llm
from tcrag.logging_config import configure_from_dict
from tcrag.rag_systems.jst_system import (
    JSTRAGSystem,
)

DEFAULT_CONFIG = (
    "configs/jstrag_default.yaml"
)
#: Registration name of the document constructor hard-wired in this entry point.
CONSTRUCTOR_NAME = "spacy_llm_hybrid"
#: Report system name used when systems.jstrag.report_name is not set in YAML.
DEFAULT_SYSTEM_NAME = "nuggetindex_duopart_spacy_llm_hybrid"


def _file_size_bytes(path) -> int:
    """Return the file size in bytes; returns 0 when the file does not exist or is unreadable."""
    try:
        p = Path(path)
        return p.stat().st_size if p.is_file() else 0
    except OSError:
        return 0


def _sqlite_storage_bytes(db_path) -> int:
    """Measure on-disk SQLite usage: the main database plus the ``-wal`` / ``-shm`` sidecars of WAL mode."""
    base = Path(db_path)
    total = _file_size_bytes(base)
    for suffix in ("-wal", "-shm"):
        total += _file_size_bytes(base.with_name(base.name + suffix))
    return total


def _build_llm_extractor(cfg):
    """Constructs a nuggetindex LLMExtractor wrapped with placeholder validity checks.

    This extractor acts as the 0th branch (documents routed to LLM) of the hybrid
    constructor's LR router. When the configuration ``llm.structured_compat=true``
    is set (for small models like Llama 3.2 with weak structured output
    capabilities), switch to ``OllamaCompatClient`` (constrained decoding +
    normalization); otherwise, use nuggetindex's native instructor client.
    """
    from nuggetindex.extractors import LLMConfig, build_client
    from tcrag.extractors.ni_llm_placeholder import (
        PlaceholderValidityLLMExtractor,
    )

    ni_cfg = LLMConfig(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        api_key=cfg.llm.api_key or None,
        # ollama uses the default host; otherwise use cfg.llm.base_url
        base_url=cfg.llm.base_url or None,
        temperature=cfg.llm.temperature,
        # Long passages produce multiple pretty-printed JSON facts; for local
        # thinking-enabled models the chain-of-thought also consumes the generation budget, so 1024 easily truncates the JSON into an empty parse.
        max_tokens=4096,
        timeout_seconds=float(cfg.llm.timeout),
    )
    if ni_cfg.provider == "ollama" and cfg.llm.structured_compat:
        from tcrag.extractors.ni_ollama_compat import OllamaCompatClient
        client = OllamaCompatClient(ni_cfg)
    else:
        client = build_client(ni_cfg)
    return PlaceholderValidityLLMExtractor(ni_cfg, client=client)


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean MRAG environment variable, following the same semantics as the factory function (``_env_flag``)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no")


def _mrag_env_snapshot() -> dict[str, Any]:
    """Collect the **effective values** of environment variables related to semantic
    reranking / cascade early stopping.

    Unset variables fall back to the same defaults as the ``create_jst_retriever``
    factory (see jstretriever.py), with identical type-conversion semantics so the
    report is self-contained and reproducible. ``bm25_index_path`` is None when unset
    (the factory then falls back to the store-backend BM25).
    """
    return {
        "reranker_model": os.getenv("RERANKER_MODEL", "nvidia/NV-Embed-v2"),
        "reranker_type": os.getenv("RERANKER_TYPE", "nv_embed"),
        "bm25_index_path": os.getenv("BM25_INDEX_PATH"),
        "hybrid_base": float(os.getenv("HYBRID_BASE", "0.0")),
        "snt_with_title": _env_flag("SNT_WITH_TITLE", True),
        "cascade_early_stop": _env_flag("CASCADE_EARLY_STOP", True),
        "cascade_t_high": float(os.getenv("CASCADE_T_HIGH", "0.9")),
        "cascade_t_low": float(os.getenv("CASCADE_T_LOW", "0.6")),
        "cascade_t_bins": int(os.getenv("CASCADE_T_BINS", "1")),
        "cascade_c_bins": int(os.getenv("CASCADE_C_BINS", "3")),
        "cascade_gate_topk": int(os.getenv("CASCADE_GATE_TOPK", "10")),
        "cascade_patience": int(os.getenv("CASCADE_PATIENCE", "2")),
    }


def _build_run_meta(args: argparse.Namespace, cfg, skip_ingest: bool | None = None) -> dict:
    """Assemble run parameters plus a snapshot of key configuration, written into the
    md/json reports for reproducibility.

    ``skip_ingest`` is the effective value (--skip-ingest only takes effect together
    with --reuse-db); when omitted it falls back to the raw args value.
    """
    run_meta: dict[str, Any] = {
        "cli_args": {
            "config": args.config or "configs/default.yaml",
            "dataset": args.dataset,
            "queries": args.queries or "",
            "format": args.format,
            "extractor": cfg.extractor.type,
            "passage_limit": args.passage_limit,
            "query_limit": args.query_limit,
            "no_qa": args.no_qa,
            "reuse_db": args.reuse_db,
            "skip_ingest": args.skip_ingest if skip_ingest is None else skip_ingest,
            "per_query": args.per_query,
        },
        "llm": {
            "provider": cfg.llm.provider,
            "model": cfg.llm.model,
            "ollama_model": cfg.llm.ollama_model,
            "structured_compat": cfg.llm.structured_compat,
            "temperature": cfg.llm.temperature,
            "max_tokens": cfg.llm.max_tokens,
            "timeout": cfg.llm.timeout,
        },
        "extractor": {
            "type": cfg.extractor.type,
            "llm_provider": getattr(cfg.extractor, "llm_provider", ""),
        },
        "rag": {
            "top_k": cfg.rag.top_k,
            "ctx_top_k": cfg.rag.ctx_top_k or cfg.rag.top_k,
            "context_token_budget": cfg.rag.context_token_budget,
            "fusion": cfg.rag.fusion,
        },
        "jstrag": {
            "db_path": cfg.jstrag.db_path,
            "retriever_factory": cfg.jstrag.retriever_factory or "native",
        },
        "evaluation": {
            "retrieval_k": cfg.evaluation.retrieval_k,
            "qa": cfg.evaluation.qa,
            "system_perf": cfg.evaluation.system_perf,
        },
    }
    # Cascade/reranking environment variables only take effect for custom retrievers; native
    # nuggetindex and similar runs do not append this section, avoiding report parameters unrelated to the actual pipeline.
    if "metriever" in (cfg.jstrag.retriever_factory or ""):
        run_meta["mrag_retriever"] = _mrag_env_snapshot()
    return run_meta


async def _run(args: argparse.Namespace) -> int:
    # Unique run ID (UTC timestamp) used to unify json / md / per_query CSV filenames,
    # captured at entry startup to ensure all report files share the same prefix.
    from datetime import UTC, datetime
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"=== Run ID: {run_id} ===")

    cfg = load_config(args.config)
    configure_from_dict(cfg.logging)

    if args.db_path:
        cfg.jstrag.db_path = args.db_path

    # skip_ingest only takes effect when --reuse-db is enabled: tolerate cases where
    # --skip-ingest is mistakenly passed without --reuse-db (full ingest proceeds in
    # that case). Defaults to True, so --reuse-db skips ingest by default; to do
    # incremental ingest on a reused database, explicitly pass --reuse-db --no-skip-ingest.
    skip_ingest = bool(args.reuse_db and args.skip_ingest)

    docs, queries = load_data(args)
    print(f"Loaded {len(docs)} documents and {len(queries)} queries")
    if not queries or (not skip_ingest and not docs):
        print(
            "ERROR: need at least one query"
            + ("" if skip_ingest else " and one document"),
            file=sys.stderr,
        )
        return 1

    # Input data file sizes (bytes), recorded in the report.
    storage = {
        "dataset_file": str(args.dataset),
        "dataset_bytes": _file_size_bytes(args.dataset),
    }
    if args.queries:
        storage["queries_file"] = str(args.queries)
        storage["queries_bytes"] = _file_size_bytes(args.queries)

    # Whether to construct the extractor / skip ingest is determined by effective skip_ingest
    # (True only when reuse_db and skip_ingest are both set). When using --reuse-db --no-skip-ingest,
    # the extractor is still constructed for incremental ingestion, and aingest_documents will
    # skip source_ids that already exist in the passages table.
    llm_extractor = None if skip_ingest else _build_llm_extractor(cfg)
    retriever_factory = resolve_retriever_factory(cfg.jstrag.retriever_factory)

    # The LLM is created up front: besides QA answer generation it is also injected into custom
    # retrievers (e.g., MRAG keyword extraction and QFS summarization). It is None with --no-qa or when unavailable, and retrievers skip it automatically.
    llm = None
    if not args.no_qa:
        try:
            llm = create_llm(cfg.llm)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: LLM unavailable, skipping answer generation: {exc}", file=sys.stderr)
            llm = None

    if retriever_factory is not None:
        # create_mrag_retriever(store, *, llm=None): inject llm via partial while
        # keeping the retriever_factory(store) calling convention unchanged.
        retriever_factory = functools.partial(retriever_factory, llm=llm)

    # Assemble JSTRAGSystem directly and force the spacy_llm_hybrid constructor.
    system = JSTRAGSystem(
        constructor_factory=CONSTRUCTOR_NAME,
        db_path=cfg.jstrag.db_path,
        extractor=llm_extractor,                   # None under effective skip_ingest
        fusion=cfg.rag.fusion,
        retriever_factory=retriever_factory,
        reuse_existing=args.reuse_db,
    )
    # Report system name: the YAML report_name takes precedence; otherwise use the entry-point default name.
    system.name = cfg.jstrag.report_name or DEFAULT_SYSTEM_NAME

    evaluator = build_evaluator(cfg, llm)
    print(f"\n=== Evaluating system: {system.name} ===")
    report = await run_single_eval(system, docs, queries, evaluator, llm, cfg,
                                   skip_ingest=skip_ingest)
    if report is not None:
        # When run_single_eval returns, the system has already been aclose'd (SQLite
        # connections closed, WAL checkpointed); this is the most stable point to measure on-disk index usage.
        storage["index_path"] = str(cfg.jstrag.db_path)
        storage["index_bytes"] = _sqlite_storage_bytes(cfg.jstrag.db_path)
        report.storage = storage
        # Run parameters and a key-configuration snapshot, written at the top of the md report for reproducibility.
        report.run_meta = _build_run_meta(args, cfg, skip_ingest)
    write_reports([report] if report else [], cfg, run_id=run_id)
    if args.per_query and report is not None:
        write_per_query_details(report, queries, cfg, run_id=run_id)
    await evaluator.aclose()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TCRag nuggetindex × SpacyLLMHybridConstructor benchmark"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Path to YAML config (default: {DEFAULT_CONFIG})")
    parser.add_argument("--dataset", required=True, help="Path to the dataset file")
    parser.add_argument("--queries", default=None, help="Optional separate queries file")
    parser.add_argument("--format", default="auto", choices=["auto", "timeqa", "ravine", "tempevalrag", "situatedqa", "json", "jsonl", "csv"],
                        help="Dataset format (default: auto-detect)")
    parser.add_argument("--passage-limit", type=int, default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--no-qa", action="store_true", help="Skip LLM answer generation (retrieval only)")
    parser.add_argument("--reuse-db", action="store_true",
                        help="Open the existing index db without backup/rebuild "
                             "(requires a prebuilt db); ingest is skipped by default, "
                             "pass --reuse-db --no-skip-ingest for incremental ingest")
    parser.add_argument("--skip-ingest", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip document ingest (no extractor built); only effective "
                             "together with --reuse-db, ignored otherwise (default: true)")
    parser.add_argument("--db-path", default=None,
                        help="Override nuggetindex db path (default: cfg systems.jstrag.db_path)")
    parser.add_argument("--per-query", action="store_true", help="Write per-query detail CSV (query_id, answer, correct answers, retrieved/relevant doc ids)")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
