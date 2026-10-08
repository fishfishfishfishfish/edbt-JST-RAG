"""单独生成训练数据文件:逐 passage 标注来源 + spacy/llm 事实数量。

本脚本**只做数据生成、不训练模型**。它复用同目录的
[train_spacy_suitability_classifier.py](file:///home/cxy/TCRag/scripts/utils/train_spacy_suitability_classifier.py)
中已有的加载与标注逻辑,产出与该脚本标注缓存**完全同格式**的 JSONL,
因此生成的文件可被其直接复用(``--no-label --label-cache <输出文件>``)。

输出文件每行一个 JSON::

    {
      "key": "timeqa:timeqa_/wiki/Ian_Gibson_(politician)_1",
      "dataset": "timeqa",                  # 来源数据集
      "source_id": "timeqa_/wiki/...",      # passage 来源标识
      "text_hash": "719690932cafee69",
      "n_spacy": 1,                         # spacy 提取的事实数量
      "n_llm": 6,                           # llm 提取的事实数量
      "label": 0,                           # n_spacy >= n_llm → 1, 否则 0
      "text": "Ian Gibson ( ... ) ..."       # passage 原文
    }

特性
====

- 默认输出到分类器的默认缓存路径 ``data/spacy_suitability_labels.jsonl``,
  生成后直接 ``--no-label`` 即可训练,无需指定路径;
- **增量续跑**:已在输出文件中且 text_hash 一致的 passage 自动跳过,
  中断后可反复运行;标注失败的 passage 不写入,下次自动重试;
- 结束后打印各数据集的 passage 数与 spacy/llm 计数汇总。

运行(需在 tcrag conda 环境内、ollama 已启动)::

    conda activate tcrag

    # 小规模试跑
    python scripts/utils/generate_spacy_suitability_data.py \
        --limit-per-dataset 100 --workers 3

    # 全量生成(TimeQA dev + TempEvalRAG;可多次中断续跑)
    python scripts/utils/generate_spacy_suitability_data.py

    # 生成后直接复用训练
    python scripts/utils/train_spacy_suitability_classifier.py --no-label
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

# 直接以 ``python scripts/utils/xxx.py`` 运行时,仓库根目录不在 sys.path 中。
# scripts/utils/<file>.py: parents[0]=utils, parents[1]=scripts, parents[2]=仓库根。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# 复用同目录分类器脚本中的标注实现(避免重复维护两份逻辑)。
_UTILS_DIR = Path(__file__).resolve().parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

# 本地 Ollama 服务不走系统代理(VPN/代理环境下否则会拦截 localhost 请求)。
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from tcrag.logging_config import get_logger, setup_logging
from train_spacy_suitability_classifier import (
    _DEFAULT_CACHE,
    _DEFAULT_TEMPEVALRAG,
    _DEFAULT_TIMEQA,
    label_passages,
    load_label_cache,
    load_passages,
)

logger = get_logger("generate_spacy_data")


# ── 汇总报告 ────────────────────────────────────────────────────────────

def summarize(cache: dict[str, dict[str, Any]]) -> None:
    """按数据集汇总 passage 数与 spacy/llm 事实计数。"""
    groups: dict[str, dict[str, Any]] = {}
    for row in cache.values():
        dataset = str(row.get("dataset", "unknown"))
        g = groups.setdefault(
            dataset,
            {"n": 0, "spacy": 0, "llm": 0, "pos": 0},
        )
        g["n"] += 1
        g["spacy"] += int(row.get("n_spacy", 0))
        g["llm"] += int(row.get("n_llm", 0))
        g["pos"] += int(row.get("label", 0))

    total_n = sum(g["n"] for g in groups.values())
    total_spacy = sum(g["spacy"] for g in groups.values())
    total_llm = sum(g["llm"] for g in groups.values())
    total_pos = sum(g["pos"] for g in groups.values())

    header = f"{'dataset':<14}{'passages':>9}{'n_spacy':>9}{'n_llm':>9}{'pos(spacy)':>12}{'mean s/p':>10}{'mean l/p':>10}"
    logger.info("生成数据汇总:\n%s", header)
    for dataset in sorted(groups):
        g = groups[dataset]
        logger.info(
            "%-14s%9d%9d%9d%12d%10.2f%10.2f",
            dataset,
            g["n"],
            g["spacy"],
            g["llm"],
            g["pos"],
            g["spacy"] / g["n"],
            g["llm"] / g["n"],
        )
    logger.info(
        "%-14s%9d%9d%9d%12d%10.2f%10.2f",
        "TOTAL",
        total_n,
        total_spacy,
        total_llm,
        total_pos,
        total_spacy / total_n if total_n else 0.0,
        total_llm / total_n if total_n else 0.0,
    )
    logger.info(
        "标签分布:正样本(n_spacy>=n_llm)=%d (%.1f%%),负样本=%d (%.1f%%)",
        total_pos,
        100.0 * total_pos / total_n if total_n else 0.0,
        total_n - total_pos,
        100.0 * (total_n - total_pos) / total_n if total_n else 0.0,
    )


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="生成训练数据文件:标注每个 passage 的来源与 spacy/llm 事实数量",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--timeqa", type=Path, default=_DEFAULT_TIMEQA,
                   help="TimeQA annotated_dev.json 路径")
    p.add_argument("--tempevalrag", type=Path, default=_DEFAULT_TEMPEVALRAG,
                   help="TempEvalRAG 目录(含 docs.jsonl)")
    p.add_argument("--datasets", type=str, default="timeqa,tempevalrag",
                   help="逗号分隔,可选 timeqa / tempevalrag")
    p.add_argument("--limit-per-dataset", type=int, default=0,
                   help="每个数据集最多加载的 passage 数,0 表示全部")
    p.add_argument("--workers", type=int, default=3,
                   help="并发标注协程数(ollama 并发请求数)")
    p.add_argument("--llm-model", type=str, default="llama3.2",
                   help="用于标注的 Ollama 模型名")
    p.add_argument("--llm-host", type=str, default="http://localhost:11434")
    p.add_argument("--llm-max-tokens", type=int, default=4096,
                   help="LLM 生成上限(与基准脚本一致,默认 4096)")
    p.add_argument("--llm-timeout", type=float, default=180.0,
                   help="单次 LLM 请求超时秒数")
    p.add_argument("--structured-compat", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="llama3.2 等小模型用 OllamaCompatClient 约束解码;"
                        "强模型可 --no-structured-compat")
    p.add_argument("--spacy-model", type=str, default="en_core_web_sm")
    p.add_argument("--max-facts", type=int, default=20,
                   help="两个提取器共同使用的事实上限,0 表示不限")
    p.add_argument("--output", type=Path, default=_DEFAULT_CACHE,
                   help="输出 JSONL 文件(默认即分类器的缓存路径)")
    p.add_argument("--log-file", type=Path, default=None,
                   help="可选:日志同时写入该文件(默认只输出到 stderr)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(level="INFO", log_file=args.log_file, console=True)

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    for s in datasets:
        if s not in {"timeqa", "tempevalrag"}:
            raise SystemExit(f"未知数据集: {s}(可选 timeqa / tempevalrag)")

    logger.info("输出文件: %s", args.output)
    cache = load_label_cache(args.output)
    logger.info("已有标注: %d 条(将增量续跑)", len(cache))

    passages = load_passages(
        datasets=datasets,
        timeqa_path=args.timeqa,
        tempevalrag_path=args.tempevalrag,
        limit_per_dataset=args.limit_per_dataset,
    )
    if not passages:
        raise SystemExit("没有加载到任何 passage")

    asyncio.run(
        label_passages(
            passages,
            cache=cache,
            cache_path=args.output,
            spacy_model=args.spacy_model,
            llm_model=args.llm_model,
            llm_host=args.llm_host,
            llm_max_tokens=args.llm_max_tokens,
            llm_timeout=args.llm_timeout,
            structured_compat=args.structured_compat,
            max_facts=args.max_facts,
            workers=args.workers,
        )
    )

    # 重新从文件读回,保证汇总基于实际落盘内容。
    summarize(load_label_cache(args.output))
    logger.info(
        "训练数据已就绪,可直接训练:"
        " python scripts/utils/train_spacy_suitability_classifier.py --no-label"
        " --label-cache %s",
        args.output,
    )


if __name__ == "__main__":
    main()
