"""Configuration loader for TCRag.

Loads YAML config files and merges in environment-variable overrides for
sensitive values (API keys, passwords) so secrets never need to live in
the config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: int = 120
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    # Enable only when provider=ollama and a small model's structured output is
    # unstable (e.g. llama3.2): uses OllamaCompatClient (json_schema constrained
    # decoding + output normalization). Strong models like qwen3 stay False and
    # use nuggetindex's native instructor client.
    structured_compat: bool = False


@dataclass
class ExtractorConfig:
    type: str = "rule_based"
    llm_provider: str = "openai"
    # for spacy extractor (type: spacy / spacy_sm)
    spacy_model: str = "en_core_web_sm"
    spacy_max_facts: int = 20


@dataclass
class RAGConfig:
    top_k: int = 10
    # Number of passages actually used to build the LLM context; equals top_k when None.
    ctx_top_k: int | None = None
    context_token_budget: int = 3000
    fusion: str = "rrf"


@dataclass
class NuggetIndexConfig:
    enabled: bool = True
    db_path: str = "data/nuggetindex.db"
    # Importable path "module:attr" or "module.attr"; when empty, nuggetindex's
    # native Retriever is used. Callable signature: (store: NuggetStore) -> Any;
    # the returned object must implement
    # async aretrieve(query, *, query_time, view, top_k, fusion, filters).
    retriever_factory: str | None = None
    # Registered name of the DocumentConstructor builder (see the tcrag.constructors
    # registry): "native" (native four stages) / "extract_only" (extraction only) /
    # "sentence" (one line per sentence); only JSTRAGSystem consumes this option.
    # When left empty, the entry-point default applies.
    constructor_factory: str | None = None
    # System name used in reports (BaseRAGSystem.name): when the same system code
    # runs different constructor/retriever variants, this distinguishes them in
    # reports (e.g. nuggetindex_extract_only). When empty, each entry point falls
    # back to its own default name.
    report_name: str | None = None


@dataclass
class GraphitiConfig:
    enabled: bool = True
    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    group_id: str | None = None
    max_coroutines: int | None = None
    # Embedding model used by graphiti's native LLM pipeline; when empty it
    # defaults according to llm.provider: openai -> text-embedding-3-small;
    # ollama -> bge-m3 (requires `ollama pull bge-m3` first)
    embedding_model: str = ""
    # Dedicated model-name override for graphiti on the ollama pipeline; when
    # empty it inherits llm.ollama_model. Rationale: ollama's default
    # num_ctx=4096, while graphiti structured-extraction requests use
    # max_tokens=16384 and the /v1 endpoint does not accept per-request
    # options, so long JSON gets truncated at the 4096-context wall. It must
    # point to a derived model with a large num_ctx (auto-created by the scripts
    # preflight check), e.g. qwen3.8-ctx32k.
    ollama_model: str = ""
    # max_tokens sent to /v1 chat.completions on the ollama pipeline (overrides
    # graphiti's default of 16384). The long JSON from graphiti entity/edge
    # extraction hard-truncated at 16384 tokens causes JSONDecodeError; the
    # value must be below the derived model's num_ctx (default 32768) while
    # leaving headroom for the prompt.
    ollama_max_tokens: int = 28000
    # Small models such as llama3.2 deterministically produce invalid JSON
    # under graphiti's complex schema (all 4 retries for the same episode fail
    # at the same char position), so retries are pure waste (~6 min each x 4
    # plus backoff ~= 25 min/episode). When enabled, monkey-patches graphiti's
    # is_server_or_retry_error so JSONDecodeError is not retried: failures are
    # raised immediately, cutting the worst case from ~10h to ~2.5h. Other
    # recoverable errors (RateLimit / 5xx) are still retried.
    ollama_no_retry_json_error: bool = False
    # Switches aingest_documents to the add_episode_bulk path: batch parallel
    # extraction + cross-episode in-memory deduplication + batched Neo4j
    # writes. For Ollama single-slot models the real parallelism is limited by
    # OLLAMA_NUM_PARALLEL (default 1), but it still helps with cross-entity
    # deduplication across multiple episodes and DB round-trips. On failure
    # the whole batch falls back to one-by-one add_episode.
    use_bulk_ingest: bool = False
    # Number of episodes per bulk batch. bulk_utils uses an internal
    # CHUNK_SIZE=10, so a value <= 10 is recommended. When the LLM is unstable
    # (llama3.2 has a high JSON failure rate), reduce it to 5 so a failure
    # costs less.
    bulk_chunk_size: int = 10
    # Monkey-patches graphiti_core.graphiti.extract_nodes_and_edges_bulk to
    # force use_combined_extraction=True: nodes and edges are merged into a
    # single LLM call, halving the number of LLM calls during extraction. In
    # stub mode make_stub_openai already supports the CombinedExtraction
    # model. Works best together with use_bulk_ingest, but can also be enabled
    # independently (it only affects the node/edge extraction chain inside
    # bulk).
    use_combined_extraction: bool = False


@dataclass
class SQLiteFaissConfig:
    """Configuration for the SQLite (FTS5) + FAISS temporal-retrieval comparison systems.

    The two comparison systems (``timefilter``: hard time filtering plus
    BM25/vector fusion; ``distproduct``: product of vector distance and
    temporal distance) share the same storage backend and differ only in
    retrieval strategy. Separate ``<name>.db`` files are created per system
    name under ``db_dir``.
    """

    enabled: bool = True
    db_dir: str = "data"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    time_scale_days: float = 365.0
    fusion: str = "rrf"
    candidate_factor: int = 3


@dataclass
class BM25DocConfig:
    """Configuration for the document-level-only BM25 baseline (SQLite FTS5, no nugget extraction)."""

    enabled: bool = True
    db_path: str = "data/bm25_doc.db"


@dataclass
class ContrieverConfig:
    """Configuration for the Contriever dense-retrieval baseline (on-disk FAISS ``IndexFlatIP`` vector index).

    Mirrors :class:`BM25DocConfig` one-to-one; both doc and sentence
    granularities share this configuration. The default index path for the
    sentence-level granularity is derived from ``db_path`` by the benchmark
    entry point (``_doc`` in the file name is replaced with ``_sentence``),
    and can also be overridden explicitly via the command-line ``--db-path``.
    ``db_path`` points to the ``.faiss`` vector index; the same-stem
    ``.meta.jsonl`` is the metadata sidecar. The encoder is
    ``facebook/contriever`` (transformers + torch, mean pooling plus L2
    normalization, 768 dimensions).
    """

    enabled: bool = True
    db_path: str = "data/contriever_doc.faiss"
    model_name: str = "facebook/contriever"
    # Encoding device: auto (use GPU if it actually works, otherwise fall back
    # to CPU) / cuda / cuda:0 / cpu
    device: str = "auto"
    # Whether to use fp16 half precision on GPU (significantly speeds up V100
    # etc.); ignored on CPU
    fp16: bool = True
    # Encoding batch size for ingestion (can be increased when GPU/memory allow)
    encode_batch_size: int = 32


@dataclass
class RagasConfig:
    """Configuration for ragas evaluation metrics (judge LLM + embeddings).

    Falls back to the main LLMConfig.model when judge_llm_model is empty.
    embedding_provider is reserved for future extensions (e.g. "huggingface").
    """

    enabled: bool = False
    judge_llm_model: str = ""
    judge_llm_provider: str = "openai"
    judge_llm_temperature: float = 0.0
    judge_llm_max_tokens: int = 1024
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "openai"
    context_recall: bool = True
    context_precision: bool = True
    answer_correctness: bool = True
    rouge: bool = True
    bleu: bool = True


@dataclass
class EvaluationConfig:
    retrieval_k: list[int] = field(default_factory=lambda: [5, 10])
    qa: bool = True
    system_perf: bool = True
    output_dir: str = "reports"
    ragas: RagasConfig = field(default_factory=RagasConfig)


@dataclass
class AppConfig:
    logging: dict[str, Any] = field(default_factory=dict)
    llm: LLMConfig = field(default_factory=LLMConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    jstrag: NuggetIndexConfig = field(default_factory=NuggetIndexConfig)
    graphiti: GraphitiConfig = field(default_factory=GraphitiConfig)
    sqlite_faiss: SQLiteFaissConfig = field(default_factory=SQLiteFaissConfig)
    bm25_doc: BM25DocConfig = field(default_factory=BM25DocConfig)
    contriever: ContrieverConfig = field(default_factory=ContrieverConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _section(raw: dict[str, Any], *keys: str) -> dict[str, Any]:
    node: Any = raw
    for k in keys:
        if not isinstance(node, dict):
            return {}
        node = node.get(k, {})
    return node if isinstance(node, dict) else {}


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate the application configuration.

    Environment overrides applied:
      - OPENAI_API_KEY -> llm.api_key
      - OPENAI_BASE_URL -> llm.base_url
      - NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD -> graphiti.*
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    llm_raw = _section(raw, "llm")
    extractor_raw = _section(raw, "extractor")
    rag_raw = _section(raw, "rag")
    jstrag_raw = _section(raw, "systems", "jstrag")
    gr_raw = _section(raw, "systems", "graphiti")
    sf_raw = _section(raw, "systems", "sqlite_faiss")
    bd_raw = _section(raw, "systems", "bm25_doc")
    ct_raw = _section(raw, "systems", "contriever")
    eval_raw = _section(raw, "evaluation")

    llm = LLMConfig(
        provider=llm_raw.get("provider", "openai"),
        model=llm_raw.get("model", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY", llm_raw.get("api_key", "")),
        base_url=os.getenv("OPENAI_BASE_URL", llm_raw.get("base_url", "")),
        temperature=float(llm_raw.get("temperature", 0.0)),
        max_tokens=int(llm_raw.get("max_tokens", 512)),
        timeout=int(llm_raw.get("timeout", 120)),
        ollama_host=llm_raw.get("ollama_host", "http://localhost:11434"),
        ollama_model=llm_raw.get("ollama_model", "llama3.1"),
        structured_compat=bool(llm_raw.get("structured_compat", False)),
    )

    extractor = ExtractorConfig(
        type=extractor_raw.get("type", "rule_based"),
        llm_provider=extractor_raw.get("llm_provider", llm.provider),
        spacy_model=extractor_raw.get("spacy_model", "en_core_web_sm"),
        spacy_max_facts=int(extractor_raw.get("spacy_max_facts", 20)),
    )

    rag = RAGConfig(
        top_k=int(rag_raw.get("top_k", 10)),
        ctx_top_k=(int(rag_raw["ctx_top_k"]) if rag_raw.get("ctx_top_k") is not None else None),
        context_token_budget=int(rag_raw.get("context_token_budget", 3000)),
        fusion=rag_raw.get("fusion", "rrf"),
    )

    jstrag = NuggetIndexConfig(
        enabled=bool(jstrag_raw.get("enabled", True)),
        db_path=jstrag_raw.get("db_path", "data/nuggetindex.db"),
        retriever_factory=jstrag_raw.get("retriever_factory"),
        constructor_factory=jstrag_raw.get("constructor_factory"),
        report_name=jstrag_raw.get("report_name"),
    )

    graphiti = GraphitiConfig(
        enabled=bool(gr_raw.get("enabled", True)),
        neo4j_uri=os.getenv("NEO4J_URI", gr_raw.get("neo4j_uri", "neo4j://127.0.0.1:7687")),
        neo4j_user=os.getenv("NEO4J_USER", gr_raw.get("neo4j_user", "neo4j")),
        neo4j_password=os.getenv("NEO4J_PASSWORD", gr_raw.get("neo4j_password", "password")),
        group_id=gr_raw.get("group_id"),
        max_coroutines=gr_raw.get("max_coroutines"),
        embedding_model=gr_raw.get("embedding_model", "") or "",
        ollama_model=gr_raw.get("ollama_model", "") or "",
        ollama_max_tokens=int(gr_raw.get("ollama_max_tokens", 28000)),
        ollama_no_retry_json_error=bool(gr_raw.get("ollama_no_retry_json_error", False)),
        use_bulk_ingest=bool(gr_raw.get("use_bulk_ingest", False)),
        bulk_chunk_size=int(gr_raw.get("bulk_chunk_size", 10)),
        use_combined_extraction=bool(gr_raw.get("use_combined_extraction", False)),
    )

    sqlite_faiss = SQLiteFaissConfig(
        enabled=bool(sf_raw.get("enabled", True)),
        db_dir=sf_raw.get("db_dir", "data"),
        embedding_model=sf_raw.get("embedding_model", "BAAI/bge-small-en-v1.5"),
        time_scale_days=float(sf_raw.get("time_scale_days", 365.0)),
        fusion=sf_raw.get("fusion", "rrf"),
        candidate_factor=int(sf_raw.get("candidate_factor", 3)),
    )

    bm25_doc = BM25DocConfig(
        enabled=bool(bd_raw.get("enabled", True)),
        db_path=bd_raw.get("db_path", "data/bm25_doc.db"),
    )

    contriever = ContrieverConfig(
        enabled=bool(ct_raw.get("enabled", True)),
        db_path=ct_raw.get("db_path", "data/contriever_doc.faiss"),
        model_name=ct_raw.get("model_name", "facebook/contriever"),
        device=str(ct_raw.get("device", "auto")),
        fp16=bool(ct_raw.get("fp16", True)),
        encode_batch_size=int(ct_raw.get("encode_batch_size", 32)),
    )

    ragas_raw = eval_raw.get("ragas", {})
    ragas = RagasConfig(
        enabled=bool(ragas_raw.get("enabled", False)),
        judge_llm_model=ragas_raw.get("judge_llm_model", ""),
        judge_llm_provider=ragas_raw.get("judge_llm_provider", "openai"),
        judge_llm_temperature=float(ragas_raw.get("judge_llm_temperature", 0.0)),
        judge_llm_max_tokens=int(ragas_raw.get("judge_llm_max_tokens", 1024)),
        embedding_model=ragas_raw.get("embedding_model", "text-embedding-3-small"),
        embedding_provider=ragas_raw.get("embedding_provider", "openai"),
        context_recall=bool(ragas_raw.get("context_recall", True)),
        context_precision=bool(ragas_raw.get("context_precision", True)),
        answer_correctness=bool(ragas_raw.get("answer_correctness", True)),
        rouge=bool(ragas_raw.get("rouge", True)),
        bleu=bool(ragas_raw.get("bleu", True)),
    )

    evaluation = EvaluationConfig(
        retrieval_k=list(eval_raw.get("metrics", {}).get("retrieval_k", [5, 10])),
        qa=bool(eval_raw.get("metrics", {}).get("qa", True)),
        system_perf=bool(eval_raw.get("metrics", {}).get("system_perf", True)),
        output_dir=eval_raw.get("output_dir", "reports"),
        ragas=ragas,
    )

    return AppConfig(
        logging=_section(raw, "logging") or {"level": "INFO", "console": True},
        llm=llm,
        extractor=extractor,
        rag=rag,
        jstrag=jstrag,
        graphiti=graphiti,
        sqlite_faiss=sqlite_faiss,
        bm25_doc=bm25_doc,
        contriever=contriever,
        evaluation=evaluation,
    )
