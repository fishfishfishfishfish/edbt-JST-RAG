"""``JSTRAGSystem``(独立实现):``DocumentConstructor`` 可替换的 RAG 系统。

直接继承 :class:`tcrag.rag_systems.base.BaseRAGSystem`,
建库备份 / 增量复用 / FTS5 sanitize /
按文档聚合检索 / 自定义 retriever 等逻辑均在本模块内实现。

系统唯一的开放点是入库管线的构造器可替换——在首次 ``aingest`` 前,
把所选实现安装到 ``NuggetStore._constructor``(nuggetindex 只在该属性
为 None 时才懒建原生实现,因此预设值优先生效),无需修改 nuggetindex
仓库。可选实现统一放在 :mod:`tcrag.constructors`。

示例::

    # 1) 按注册名选用构造器(默认即 spacy_llm_hybrid)
    system = JSTRAGSystem(
        db_path="data/nuggetindex.db",
        extractor=extractor,
    )

    # 2) 传入自定义工厂(签名 factory(store, **kwargs))
    system = JSTRAGSystem(
        db_path="data/nuggetindex.db",
        extractor=extractor,
        constructor_factory=lambda store: MyConstructor(extractor=...),
    )

    # 3) constructor_factory=None:完全不注入,退回 NuggetStore 自身懒建
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tcrag.constructors import build_constructor
from tcrag.data.models import Document, RetrievalHit
from tcrag.logging_config import get_logger
from tcrag.rag_systems.base import BaseRAGSystem
from tcrag.utils import sanitize_fts_query

logger = get_logger("rag.jst_system")

#: Type alias documentation: ConstructorSpec specifies the constructor configuration, 
# supporting a registered name string, 
# factory function (signature ``factory(store, **kwargs)``), or ``None`` (skip injection)
ConstructorSpec = str | Callable[..., Any] | None


class JSTRAGSystem(BaseRAGSystem):
    """``JSTRAGSystem``:开放 ``DocumentConstructor`` 为可替换实现。

    Args:
        db_path: SQLite 索引路径。
        extractor: nuggetindex duck-type extractor(产 ExtractionResult)。
        fusion: 原生 retriever 的融合方式(如 ``"rrf"``)。
        retriever_factory: 自定义检索器工厂 ``factory(store)``;为 None
            时使用 nuggetindex 原生 Retriever。
        reuse_existing: 直接打开已有索引(不备份/不重建),并以 passages
            表为准跳过已入库文档。
        constructor_factory: 构造器规格。默认 ``"spacy_llm_hybrid"``——
            LR 分类器路由 spaCy / LLM 两支 extractor;传其他注册名或
            ``factory(store)`` 即可替换整条入库管线;传 ``None`` 则不
            注入,沿用 NuggetStore 的隐式懒建。
        constructor_kwargs: 透传给构造器工厂的额外关键字参数。
    """

    name = "jstrag"

    def __init__(
        self,
        *,
        db_path: str | Path = "data/jstrag.db",
        extractor: Any = None,
        fusion: str = "rrf",
        retriever_factory: Callable[[Any], Any] | None = None,
        reuse_existing: bool = False,
        constructor_factory: ConstructorSpec = "spacy_llm_hybrid",
        constructor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from nuggetindex import NuggetStore

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._reuse_existing = reuse_existing
        if reuse_existing:
            # 复用已有索引:不备份/不重建,直接打开;此时无需 extractor。
            if not self._db_path.exists():
                raise FileNotFoundError(
                    f"reuse_existing=True but db not found: {self._db_path}"
                )
            logger.info(
                "jstrag: reusing existing db at %s (skip ingest)",
                self._db_path,
            )
        elif self._db_path.exists():
            # 不直接删除已有索引文件,而是重命名为 .bak.{timestamp} 备份,
            # 便于回溯或对比上一次运行的落盘数据。
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup = self._db_path.with_suffix(
                f".bak.{ts}" + self._db_path.suffix
            )
            self._db_path.rename(backup)
            logger.info("jstrag: existing db backed up to %s", backup)

        self._store = NuggetStore(
            db_path=str(self._db_path), extractor=extractor
        )
        self._fusion = fusion
        # Custom retriever (metriever) needs raw query text (for temporal trigger detection);
        # the OR-join sanitization in aretrieve only applies to the native retriever.
        self._custom_retriever = retriever_factory is not None
        self._doc_text_cache: dict[str, str] = {}
        if reuse_existing:
            # Reconstruct source_id → raw text cache from the passages table so that
            # RetrievalHit.text / QA context matches the normal ingest path in reuse mode.
            self._doc_text_cache = self._load_passage_texts()
            n_nuggets = self._count_table("nuggets")
            self._last_ingest_summary = {
                "reused_db": True,
                "documents": len(self._doc_text_cache),
                "nuggets": n_nuggets,
            }
        if retriever_factory is not None:
            self._store._retriever = retriever_factory(self._store)
            logger.info(
                "jstrag: installed custom retriever via factory=%r",
                retriever_factory,
            )

        self._constructor_spec: ConstructorSpec = constructor_factory
        self._constructor_kwargs: dict[str, Any] = dict(
            constructor_kwargs or {}
        )
        self._constructor_installed = False

    def _open_readonly_conn(self) -> sqlite3.Connection:
        """以只读方式打开 SQLite(WAL 下与 store 的写连接互不干扰)。"""
        uri = f"file:{self._db_path}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    def _load_passage_texts(self) -> dict[str, str]:
        """从 passages 表读取 {source_id: 原文};表缺失时回退空缓存。"""
        try:
            con = self._open_readonly_conn()
            try:
                rows = con.execute(
                    "SELECT source_id, text FROM passages"
                ).fetchall()
            finally:
                con.close()
        except sqlite3.Error as exc:
            logger.warning(
                "jstrag: failed to load passages for reuse: %s", exc
            )
            return {}
        return {sid: text for sid, text in rows}

    def _count_table(self, table: str) -> int:
        """只读统计表行数;表缺失或查询失败返回 0。"""
        try:
            con = self._open_readonly_conn()
            try:
                return int(
                    con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
            finally:
                con.close()
        except sqlite3.Error:
            return 0

    def _load_existing_source_ids(self) -> set[str]:
        """读取 passages 表中已入库的 source_id 集合;失败返回空集。"""
        try:
            con = self._open_readonly_conn()
            try:
                rows = con.execute(
                    "SELECT source_id FROM passages"
                ).fetchall()
            finally:
                con.close()
        except sqlite3.Error as exc:
            logger.warning(
                "jstrag: failed to load existing passage source_ids: %s",
                exc,
            )
            return set()
        return {sid for (sid,) in rows}

    def _ensure_constructor(self) -> None:
        """Install the selected constructor into ``store._constructor`` once, before the first ingest.
        """
        if self._constructor_installed:
            return
        self._constructor_installed = True

        spec = self._constructor_spec
        if spec is None:
            logger.info(
                "constructor_factory=None, using NuggetStore's native lazy construction"
            )
            return

        try:
            constructor = build_constructor(
                spec, self._store, **self._constructor_kwargs
            )
        except RuntimeError as exc:
            if getattr(self._store, "_extractor", None) is None:
                logger.info(
                    "constructor assembly requires extractor, but none is available (pure reuse / skip-ingest mode),"
                    "skipping injection(spec=%r): %s",
                    spec, exc,
                )
                return
            raise

        self._store._constructor = constructor
        logger.info(
            "installed DocumentConstructor via spec=%r -> %s",
            spec,
            type(constructor).__name__,
        )

    async def aingest_documents(self, documents: list[Document]) -> dict:
        from nuggetindex.pipeline.constructor import Document as NiDocument

        self._ensure_constructor()

        start = time.perf_counter()
        total = 0
        skipped = 0
        # reuse_existing 增量入库:以 passages 表已有 source_id 为准跳过,
        # To avoid re-running the (idempotent but computationally expensive) LLM extractor on already-indexed documents.
        existing_ids: set[str] = (
            self._load_existing_source_ids() if self._reuse_existing else set()
        )
        if existing_ids:
            logger.info(
                "jstrag: reuse mode, %d passages already ingested, will skip",
                len(existing_ids),
            )
        log_interval_seconds = 120.0  # log progress every 120s
        log_interval_docs = 5  # log progress every 5 docs
        last_log = start
        for i, doc in enumerate(documents, 1):
            if doc.source_id in existing_ids:
                skipped += 1
                # 复用模式下缓存已从 passages 重建;setdefault 兜底重建失败的情况。
                self._doc_text_cache.setdefault(doc.source_id, doc.text)
                continue
            self._doc_text_cache[doc.source_id] = doc.text
            ni_doc = NiDocument(
                source_id=doc.source_id, text=doc.text, uri=doc.uri,
                source_date=doc.reference_time,
            )
            try:
                r = await self._store.aingest(ni_doc)
                total += r.nuggets_added + r.nuggets_merged
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ingest failed for %s: %s", doc.source_id, exc
                )
            else:
                # 成功(或幂等合并)后写回集合,同批次内重复 source_id 不重复抽取。
                existing_ids.add(doc.source_id)
            if i % log_interval_docs == 0:
                now = time.perf_counter()
                if now - last_log >= log_interval_seconds:
                    logger.info(
                        "jstrag ingest progress: %d/%d docs, %d nuggets, "
                        "%d skipped, %.1fs elapsed",
                        i, len(documents), total, skipped, now - start,
                    )
                    last_log = now
        elapsed = time.perf_counter() - start
        logger.info(
            "jstrag ingested %d docs (%d skipped), %d nuggets in %.2fs",
            len(documents), skipped, total, elapsed,
        )
        return {"documents": len(documents), "nuggets": total,
                "skipped": skipped, "elapsed_seconds": elapsed}

    async def aretrieve(
        self, query, *, top_k=10, reference_time=None, **kwargs
    ):
        if self._custom_retriever:
            # 自定义 retriever(metriever)拿到原始 query:OR join 会把 "as of"
            # 拆成 "as OR of",破坏时间触发词检测;FTS5 sanitize 由检索器
            # 在调用 abm25_search 前自行完成。
            safe_query = query
            if not (safe_query or "").strip():
                return []
        else:
            # 原生 retriever 直接查 FTS5:去特殊符号 + OR join 保召回。
            safe_query = sanitize_fts_query(query)
            if not safe_query:
                return []
        results = await self._store.aretrieve(
            safe_query, top_k=top_k * 3,
            query_time=reference_time or datetime.now(UTC),
            fusion=self._fusion,
        )
        best: dict[str, tuple[float, str]] = {}
        for r in results:
            if not r.nugget.provenance:
                continue
            doc_id = r.nugget.provenance[0].source_id
            text = self._doc_text_cache.get(doc_id, r.nugget.fact.text)
            if doc_id not in best or r.score > best[doc_id][0]:
                best[doc_id] = (r.score, text)
        ranked = sorted(
            best.items(), key=lambda kv: kv[1][0], reverse=True
        )[:top_k]
        return [RetrievalHit(doc_id=d, text=t, score=s, rank=i + 1)
                for i, (d, (s, t)) in enumerate(ranked)]

    async def aclose(self) -> None:
        backend = getattr(self._store, "_backend_impl", None)
        close = getattr(backend, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001
                pass
