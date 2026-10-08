# -*- coding: utf-8 -*-
"""TCRag nuggetindex × SpacyLLMHybridConstructor benchmark 入口。

单系统入口:固定装配注册名 ``"spacy_llm_hybrid"`` 的文档构造器(LR
分类器路由 spaCy / LLM 两支 extractor),检索器 / db_path / extractor
等仍由 YAML 配置与 CLI 决定。本入口自带完整 argparse 与运行逻辑,
公共评测编排复用 ``_benchmark_common``。

示例:
  python scripts/run_benchmark_jstrag.py \\
      --dataset data/timeqa_annotated_dev.json \\
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

# 注入项目根 + scripts 目录,使 _benchmark_common 与 tcrag 可裸导入(脚本无需安装包)
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
#: 本入口固定装配的文档构造器注册名。
CONSTRUCTOR_NAME = "spacy_llm_hybrid"
#: YAML 未设置 systems.jstrag.report_name 时的报告系统名。
DEFAULT_SYSTEM_NAME = "nuggetindex_duopart_spacy_llm_hybrid"


def _file_size_bytes(path) -> int:
    """返回文件字节数;文件不存在或不可读时返回 0。"""
    try:
        p = Path(path)
        return p.stat().st_size if p.is_file() else 0
    except OSError:
        return 0


def _sqlite_storage_bytes(db_path) -> int:
    """统计 SQLite 落盘占用:主库 + WAL 模式的 ``-wal`` / ``-shm`` sidecar。"""
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
        # ollama 走默认 host;否则用 cfg.llm.base_url
        base_url=cfg.llm.base_url or None,
        temperature=cfg.llm.temperature,
        # 长段落会产出多条 pretty-printed JSON 事实;本地带 thinking 的模型
        # 思维链也占用生成预算,1024 容易截断 JSON 导致解析为空。
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
    """布尔型 MRAG 环境变量解析,口径与工厂函数(``_env_flag``)一致。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no")


def _mrag_env_snapshot() -> dict[str, Any]:
    """采集语义重排/级联早停相关 MRAG_* 环境变量的**实际生效值**。

    未设置环境变量时回落到 ``create_mrag_*`` 工厂函数的同款默认值
    (见 metriever_semantic_temporal*.py),类型转换口径也保持一致,
    使报告自包含、可复现。``bm25_index_path`` 未设置时为 None
    (工厂据此回退 store 后端 BM25)。
    """
    return {
        "reranker_model": os.getenv("MRAG_RERANKER_MODEL", "nvidia/NV-Embed-v2"),
        "reranker_type": os.getenv("MRAG_RERANKER_TYPE", "nv_embed"),
        "bm25_index_path": os.getenv("MRAG_BM25_INDEX_PATH"),
        "hybrid_base": float(os.getenv("MRAG_HYBRID_BASE", "0.0")),
        "snt_with_title": _env_flag("MRAG_SNT_WITH_TITLE", True),
        "cascade_early_stop": _env_flag("MRAG_CASCADE_EARLY_STOP", True),
        "cascade_t_high": float(os.getenv("MRAG_CASCADE_T_HIGH", "0.9")),
        "cascade_t_low": float(os.getenv("MRAG_CASCADE_T_LOW", "0.6")),
        "cascade_t_bins": int(os.getenv("MRAG_CASCADE_T_BINS", "1")),
        "cascade_c_bins": int(os.getenv("MRAG_CASCADE_C_BINS", "3")),
        "cascade_gate_topk": int(os.getenv("MRAG_CASCADE_GATE_TOPK", "10")),
        "cascade_patience": int(os.getenv("MRAG_CASCADE_PATIENCE", "2")),
    }


def _build_run_meta(args: argparse.Namespace, cfg, skip_ingest: bool | None = None) -> dict:
    """组装运行参数 + 关键配置快照,写入 md/json 报告便于复现。

    ``skip_ingest`` 为实际生效值(仅 --reuse-db 时 --skip-ingest 才生效);
    缺省回退到原始 args 值。
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
    # MRAG_* 环境变量仅对 metriever* 自定义检索器生效;原生 nuggetindex
    # 等运行不附加该节,避免报告出现与实际管道无关的参数。
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

    # LLM 提前创建:除 QA 答案生成外,还注入自定义 retriever(如 MRAG 的
    # 关键词抽取与 QFS 摘要)。--no-qa 或 LLM 不可用时为 None,检索器自动跳过。
    llm = None
    if not args.no_qa:
        try:
            llm = create_llm(cfg.llm)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: LLM unavailable, skipping answer generation: {exc}", file=sys.stderr)
            llm = None

    if retriever_factory is not None:
        # create_mrag_retriever(store, *, llm=None):通过 partial 注入 llm,
        # 保持 retriever_factory(store) 的调用约定不变。
        retriever_factory = functools.partial(retriever_factory, llm=llm)

    # 直接装配 JSTRAGSystem 并强制 spacy_llm_hybrid 构造器。
    system = JSTRAGSystem(
        constructor_factory=CONSTRUCTOR_NAME,
        db_path=cfg.jstrag.db_path,
        extractor=llm_extractor,                   # effective skip_ingest 时为 None
        fusion=cfg.rag.fusion,
        retriever_factory=retriever_factory,
        reuse_existing=args.reuse_db,
    )
    # 报告系统名:YAML report_name 优先,否则用入口默认名。
    system.name = cfg.jstrag.report_name or DEFAULT_SYSTEM_NAME

    evaluator = build_evaluator(cfg, llm)
    print(f"\n=== Evaluating system: {system.name} ===")
    report = await run_single_eval(system, docs, queries, evaluator, llm, cfg,
                                   skip_ingest=skip_ingest)
    if report is not None:
        # run_single_eval 返回时系统已 aclose(SQLite 连接关闭、WAL 已
        # checkpoint),此时统计落盘索引占用最稳定。
        storage["index_path"] = str(cfg.jstrag.db_path)
        storage["index_bytes"] = _sqlite_storage_bytes(cfg.jstrag.db_path)
        report.storage = storage
        # 运行参数与关键配置快照,写入 md 报告顶部便于复现。
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
