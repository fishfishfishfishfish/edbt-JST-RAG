"""``JSTRAGSystem`` (standalone implementation): a RAG system with a replaceable ``DocumentConstructor``.

It directly subclasses :class:`tcrag.rag_systems.base.BaseRAGSystem`;
index backup / incremental reuse / FTS5 sanitization /
per-document aggregated retrieval / custom retriever support are all
implemented within this module.

The system's only extension point is the replaceable constructor of the
ingestion pipeline: before the first ``aingest`` call, install the chosen
implementation into ``NuggetStore._constructor`` (nuggetindex lazily builds
its native implementation only when this attribute is None, so a preset value
takes precedence), without modifying the nuggetindex repository. The available
implementations live in :mod:`tcrag.constructors`.

Example::

    # 1) Select a constructor by registered name (the default is spacy_llm_hybrid)
    system = JSTRAGSystem(
        db_path="data/nuggetindex.db",
        extractor=extractor,
    )

    # 2) Pass a custom factory (signature factory(store, **kwargs))
    system = JSTRAGSystem(
        db_path="data/nuggetindex.db",
        extractor=extractor,
        constructor_factory=lambda store: MyConstructor(extractor=...),
    )

    # 3) constructor_factory=None: skip injection entirely and fall back to NuggetStore's own lazy construction
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
    """``JSTRAGSystem``: exposes ``DocumentConstructor`` as a replaceable implementation.

    Args:
        db_path: Path to the SQLite index.
        extractor: A nuggetindex duck-typed extractor (producing ExtractionResult).
        fusion: Fusion strategy for the native retriever (e.g. ``"rrf"``).
        retriever_factory: Custom retriever factory ``factory(store)``; when
            None, the nuggetindex native Retriever is used.
        reuse_existing: Open an existing index directly (no backup/rebuild) and
            skip already-ingested documents based on the passages table.
        constructor_factory: Constructor specification. Defaults to
            ``"spacy_llm_hybrid"`` — an LR classifier routes between the spaCy
            and LLM extractor branches; pass another registered name or a
            ``factory(store)`` to replace the entire ingestion pipeline; pass
            ``None`` to skip injection and keep NuggetStore's implicit lazy
            construction.
        constructor_kwargs: Additional keyword arguments forwarded to the
            constructor factory.
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
            # Reuse an existing index: open it directly without backup/rebuild; no extractor is needed in this case.
            if not self._db_path.exists():
                raise FileNotFoundError(
                    f"reuse_existing=True but db not found: {self._db_path}"
                )
            logger.info(
                "jstrag: reusing existing db at %s (skip ingest)",
                self._db_path,
            )
        elif self._db_path.exists():
            # Do not delete the existing index file directly; rename it to a .bak.{timestamp} backup
            # so that the persisted data from the previous run can be inspected or compared.
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
        """Open SQLite in read-only mode (under WAL this does not interfere with the store's write connection)."""
        uri = f"file:{self._db_path}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    def _load_passage_texts(self) -> dict[str, str]:
        """Load a {source_id: raw text} mapping from the passages table; fall back to an empty cache if the table is missing."""
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
        """Count rows in a table using a read-only connection; return 0 if the table is missing or the query fails."""
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
        """Read the set of already-ingested source_ids from the passages table; return an empty set on failure."""
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
        # Incremental ingest with reuse_existing: skip documents whose source_id already exists in the passages table,
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
                # In reuse mode the cache is rebuilt from the passages table; setdefault covers the case where rebuilding fails.
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
                # Add back to the set after success (or idempotent merge) so duplicate source_ids within the same batch are not re-extracted.
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
            # The custom retriever (metriever) receives the raw query: an OR join would split
            # "as of" into "as OR of", breaking temporal trigger-word detection; FTS5 sanitization is
            # performed by the retriever itself before calling abm25_search.
            safe_query = query
            if not (safe_query or "").strip():
                return []
        else:
            # The native retriever queries FTS5 directly: strip special characters and OR-join the terms to preserve recall.
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
