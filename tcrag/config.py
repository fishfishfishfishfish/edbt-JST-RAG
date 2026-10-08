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
    # 仅 provider=ollama 且小模型结构化输出不稳定时开启(如 llama3.2):
    # 使用 OllamaCompatClient(json_schema constrained decoding + 输出规范化)。
    # qwen3 等强模型保持 False,走 nuggetindex 原生 instructor client。
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
    # 实际用于构造 LLM 上下文的段落数;None 时等于 top_k
    ctx_top_k: int | None = None
    context_token_budget: int = 3000
    fusion: str = "rrf"


@dataclass
class NuggetIndexConfig:
    enabled: bool = True
    db_path: str = "data/nuggetindex.db"
    # 可导入路径 "module:attr" 或 "module.attr";为空时使用 nuggetindex 原生
    # Retriever。callable 签名: (store: NuggetStore) -> Any,返回的对象必须实现
    # async aretrieve(query, *, query_time, view, top_k, fusion, filters)。
    retriever_factory: str | None = None
    # DocumentConstructor 构造器注册名(见 tcrag.constructors 注册表):
    # "native"(原生四阶段)/ "extract_only"(仅抽取)/ "sentence"(按句成行);
    # 仅 JSTRAGSystem 消费此项。
    # 留空时按入口默认处理。
    constructor_factory: str | None = None
    # 报告中使用的系统名(BaseRAGSystem.name):同一份系统代码跑不同构造器 /
    # 检索器变体时,靠它在报告里区分(如 nuggetindex_extract_only)。
    # 留空时各入口回退到自身默认名。
    report_name: str | None = None


@dataclass
class GraphitiConfig:
    enabled: bool = True
    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    group_id: str | None = None
    max_coroutines: int | None = None
    # graphiti 原生 LLM 链路使用的 embedding 模型;留空则按 llm.provider 默认:
    # openai -> text-embedding-3-small;ollama -> bge-m3(需先 ollama pull bge-m3)
    embedding_model: str = ""
    # ollama 链路 graphiti 专用的模型名覆盖;留空则沿用 llm.ollama_model。
    # 用途:ollama 默认 num_ctx=4096,而 graphiti 结构化抽取请求 max_tokens=16384,
    # /v1 端点又不接受 per-request options,长 JSON 会在 4096 上下文墙处被截断。
    # 需指向一个带大 num_ctx 的派生模型(由 scripts 预检自动创建),如 qwen3.8-ctx32k。
    ollama_model: str = ""
    # ollama 链路下发给 /v1 chat.completions 的 max_tokens(覆盖 graphiti 默认 16384)。
    # graphiti 实体/边抽取的长 JSON 在 16384 tokens 处被硬截断会导致 JSONDecodeError;
    # 值需 < 派生模型的 num_ctx(默认 32768)并给 prompt 留余量。
    ollama_max_tokens: int = 28000
    # llama3.2 等小模型在 graphiti 复杂 schema 下会确定性地产出非法 JSON
    # (同一 episode 4 次重试报同一 char 位置),重试纯属浪费时间(每次 ~6min
    # × 4 + 退避 ≈ 25min/episode)。开启后 monkey-patch graphiti 的
    # is_server_or_retry_error,对 JSONDecodeError 不重试,失败直接抛出,
    # 将最坏情况从 ~10h 降到 ~2.5h。其他可恢复错误(RateLimit / 5xx)仍重试。
    ollama_no_retry_json_error: bool = False
    # 切换 aingest_documents 到 add_episode_bulk 路径:批量并行抽取 + 跨
    # episode 内存去重 + 批量 Neo4j 写入。对 Ollama 单槽模型真实并行度
    # 受限于 OLLAMA_NUM_PARALLEL(默认 1);但对多 episode 跨实体去重和
    # DB 往返仍有收益。失败时整批回退到单条 add_episode。
    use_bulk_ingest: bool = False
    # bulk 每批 episode 数。bulk_utils 内部 CHUNK_SIZE=10,建议 ≤10。
    # LLM 不稳定(llama3.2 JSON 失败率高)时调小到 5,失败损失更小。
    bulk_chunk_size: int = 10
    # monkey-patch graphiti_core.graphiti.extract_nodes_and_edges_bulk 强制
    # use_combined_extraction=True:节点+边合并为单次 LLM 调用,抽取阶段
    # LLM 调用数减半。stub 模式下 make_stub_openai 已支持 CombinedExtraction
    # 模型。与 use_bulk_ingest 配合使用最佳,但也可独立启用(只影响 bulk
    # 内部节点/边抽取链路)。
    use_combined_extraction: bool = False


@dataclass
class SQLiteFaissConfig:
    """SQLite(FTS5)+ FAISS 时序检索对比系统配置。

    两个对比系统(``timefilter`` 硬时间过滤 + BM25/向量融合;
    ``distproduct`` 向量距离 × 时间距离乘积)共享同一存储底座,仅检索
    策略不同。``db_dir`` 下按系统名生成各自的 ``<name>.db`` 文件。
    """

    enabled: bool = True
    db_dir: str = "data"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    time_scale_days: float = 365.0
    fusion: str = "rrf"
    candidate_factor: int = 3


@dataclass
class BM25DocConfig:
    """纯文档级 BM25 基线配置(SQLite FTS5,不做 nugget 抽取)。"""

    enabled: bool = True
    db_path: str = "data/bm25_doc.db"


@dataclass
class ContrieverConfig:
    """Contriever 稠密检索基线配置(FAISS ``IndexFlatIP`` 落盘向量索引)。

    与 :class:`BM25DocConfig` 一一对应,doc / sentence 两种粒度共用本配置;
    句子级粒度的默认索引路径由基准入口按 ``db_path`` 派生
    (文件名中的 ``_doc`` 替换为 ``_sentence``),也可用命令行 ``--db-path``
    显式覆盖。``db_path`` 指 ``.faiss`` 向量索引,同 stem 的
    ``.meta.jsonl`` 为元数据 sidecar。编码器为 ``facebook/contriever``
    (transformers + torch,mean pooling + L2 normalize,768 维)。
    """

    enabled: bool = True
    db_path: str = "data/contriever_doc.faiss"
    model_name: str = "facebook/contriever"
    # 编码设备:auto(实测 GPU 可算则用,否则回退 CPU)/cuda/cuda:0/cpu
    device: str = "auto"
    # GPU 上是否使用 fp16 半精度(V100 等可显著提速);CPU 时忽略
    fp16: bool = True
    # ingest 编码批大小(GPU/内存允许时可调大)
    encode_batch_size: int = 32


@dataclass
class RagasConfig:
    """ragas 评估指标配置(judge LLM + embeddings)。

    judge_llm_model 为空时回退到主 LLMConfig.model。
    embedding_provider 支持未来扩展(如 "huggingface")。
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
