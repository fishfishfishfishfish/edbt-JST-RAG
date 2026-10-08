"""Generate benchmark reports (JSON + Markdown comparison)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tcrag.evaluation.evaluator import SystemReport
from tcrag.logging_config import get_logger

logger = get_logger("evaluation.reporter")


def _serialisable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [_serialisable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _serialisable(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return _serialisable(asdict(obj))
    return str(obj)


def _human_bytes(num_bytes: float) -> str:
    """把字节数格式化为人类可读字符串(如 ``1.23 MB``)。"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def write_json_report(reports: list[SystemReport], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "systems": [_serialisable(r) for r in reports],
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    logger.info("JSON report written to %s", out)
    return out


def write_markdown_report(reports: list[SystemReport], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# TCRag Benchmark Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(UTC).isoformat()}_")
    lines.append("")

    if not reports:
        lines.append("_No systems evaluated._")
    else:
        # Run Configuration:运行参数与关键配置快照(取首个 report,
        # 单系统入口一次只产生一个 report;多系统对比共享同一组 args/cfg)。
        run_meta = reports[0].run_meta
        if run_meta:
            lines.append("## Run Configuration")
            lines.append("")
            for section, items in run_meta.items():
                lines.append(f"**{section}**")
                lines.append("")
                lines.append("| key | value |")
                lines.append("|---|---|")
                for k, v in items.items():
                    lines.append(f"| {k} | `{v}` |")
                lines.append("")

        lines.append("## Summary Comparison")
        lines.append("")
        # Build a wide comparison table.
        all_retrieval_keys: list[str] = []
        all_answer_keys: list[str] = []
        all_perf_keys: list[str] = []
        all_ragas_keys: list[str] = []
        for r in reports:
            for k in r.retrieval:
                if k not in all_retrieval_keys:
                    all_retrieval_keys.append(k)
            for k in r.answer:
                if k not in all_answer_keys:
                    all_answer_keys.append(k)
            for k in r.performance:
                if k not in all_perf_keys:
                    all_perf_keys.append(k)
            for k in r.ragas:
                if k not in all_ragas_keys:
                    all_ragas_keys.append(k)

        header = ["Metric"] + [r.system_name for r in reports]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        def row(name: str, getter):
            cells = [name]
            for r in reports:
                val = getter(r)
                cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(cells) + " |")

        for k in all_retrieval_keys:
            row(k, lambda r, k=k: r.retrieval.get(k, 0.0))
        for k in all_answer_keys:
            row(k, lambda r, k=k: r.answer.get(k, 0.0))
        for k in all_perf_keys:
            row(k, lambda r, k=k: r.performance.get(k, 0.0))
        for k in all_ragas_keys:
            row(k, lambda r, k=k: r.ragas.get(k, 0.0))
        lines.append("")

        # Per-system detail
        for r in reports:
            lines.append(f"## {r.system_name}")
            lines.append("")
            if r.ingest_summary:
                lines.append("**Ingestion:** " + ", ".join(f"{k}={v}" for k, v in r.ingest_summary.items()))
                lines.append("")
            if r.storage:
                storage_parts: list[str] = []
                if r.storage.get("dataset_bytes") is not None:
                    storage_parts.append(
                        f"dataset `{r.storage.get('dataset_file', '')}` = "
                        f"{_human_bytes(r.storage['dataset_bytes'])}"
                    )
                if r.storage.get("queries_bytes") is not None:
                    storage_parts.append(
                        f"queries `{r.storage.get('queries_file', '')}` = "
                        f"{_human_bytes(r.storage['queries_bytes'])}"
                    )
                # index:SQLite 等本地索引只填 index_bytes;Neo4j 等服务端
                # 索引量不到文件,填 index_nodes/index_edges(字节数尽力获取)。
                if r.storage.get("index_bytes") is not None or r.storage.get("index_nodes") is not None:
                    index_desc: list[str] = []
                    if r.storage.get("index_bytes") is not None:
                        index_desc.append(_human_bytes(r.storage["index_bytes"]))
                    if r.storage.get("index_nodes") is not None:
                        index_desc.append(
                            f"{r.storage['index_nodes']} nodes/"
                            f"{r.storage.get('index_edges', '?')} edges"
                        )
                    storage_parts.append(
                        f"index `{r.storage.get('index_path', '')}` = "
                        + ", ".join(index_desc)
                    )
                if storage_parts:
                    lines.append("**Storage:** " + "; ".join(storage_parts))
                    lines.append("")
            if r.retrieval:
                lines.append("**Retrieval:** " + ", ".join(f"{k}={v:.4f}" for k, v in r.retrieval.items()))
                lines.append("")
            if r.answer:
                lines.append("**Answer quality:** " + ", ".join(f"{k}={v:.4f}" for k, v in r.answer.items()))
                lines.append("")
            if r.performance:
                lines.append("**Performance:** " + ", ".join(f"{k}={v:.4f}" for k, v in r.performance.items()))
                lines.append("")
            if r.ragas:
                lines.append("**Ragas:** " + ", ".join(f"{k}={v:.4f}" for k, v in r.ragas.items()))
                lines.append("")

    with out.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    logger.info("Markdown report written to %s", out)
    return out
