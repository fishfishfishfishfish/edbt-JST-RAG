"""LR 路由的 spaCy / LLM 混合抽取构造器(注册名 ``"spacy_llm_hybrid"``)。

每篇文档进入 :meth:`SpacyLLMHybridConstructor.aprocess` 时,先由
``scripts/utils/train_spacy_suitability_classifier.py`` 训练的
LogisticRegression 对原文做一次路由推断:

  - 模型输出 ``1``(该 passage 适合 spaCy)→ 走本地
    :class:`tcrag.extractors.spacy_extractor.SpacyFactExtractor`
    (经 :class:`tcrag.extractors.extractor_wrapper.ExtractorWrapper`
    适配到 nuggetindex 接口),零 LLM 调用;
  - 模型输出 ``0`` → 走 store 已配置的 LLM extractor(要求建库配置
    ``extractor.type: llm``;即 PlaceholderValidity 包装的 nuggetindex
    LLMExtractor / OllamaCompatClient 链路)。

两支 extractor 均暴露 nuggetindex 的
``aextract(text, source_id=...) -> list[ExtractionResult]`` 接口,本构造器
与 :class:`tcrag.constructors.extract_only.ExtractOnlyConstructor` 一样
只取 ``.nugget``,不做 canonicalize / 时间推断 / 去重 / 冲突消解等后处理。

分类器模型用 joblib 序列化,其中自定义特征类 ``TextStats`` 的 pickle 归属
模块固定为 ``train_spacy_suitability_classifier``(见训练脚本),因此加载
前需要把 ``scripts/utils`` 加入 ``sys.path`` 并 import 该模块一次,
:func:`load_suitability_classifier` 已封装此引导。

工厂参数(经 ``JSTRAGSystem(constructor_kwargs=...)`` 透传)::

    build_spacy_llm_hybrid_constructor(
        store,
        classifier_path="models/spacy_suitability_logreg_th6.joblib",  # 默认
        spacy_model="en_core_web_sm",
        spacy_max_facts=20,
    )
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tcrag.constructors import register_constructor
from tcrag.constructors.base import BaseDocumentConstructor
from tcrag.logging_config import get_logger

if TYPE_CHECKING:
    from nuggetindex.core.models import Nugget
    from nuggetindex.pipeline.aliases import AliasResolver
    from nuggetindex.pipeline.constructor import Document

    from tcrag.constructors.base import FetchExistingByKey

logger = get_logger("constructors.spacy_llm_hybrid")

# tcrag/constructors/<file>.py: parents[0]=constructors, parents[1]=tcrag,
# parents[2]=仓库根。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CLASSIFIER = (
    _REPO_ROOT / "models" / "spacy_suitability_logreg_th6.joblib"
)
_TRAIN_UTILS_DIR = _REPO_ROOT / "scripts" / "utils"


def load_suitability_classifier(model_path: str | Path) -> Any:
    """加载 spaCy 适用性 LR 模型(joblib),并完成 pickle 反序列化引导。

    模型 pipeline 内含训练脚本定义的自定义特征类 ``TextStats``,其 pickle
    归属模块为 ``train_spacy_suitability_classifier``;import 该训练模块时
    它会自行把模块对象注册进 ``sys.modules``,joblib 据此还原类。
    """
    import joblib

    model_path = Path(model_path)
    if not model_path.is_file():
        raise RuntimeError(f"spaCy 适用性分类器模型不存在: {model_path}")

    utils_dir = str(_TRAIN_UTILS_DIR)
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    # import 副作用:训练脚本执行 sys.modules.setdefault(
    # "train_spacy_suitability_classifier", ...),使 joblib 能解析 TextStats。
    importlib.import_module("train_spacy_suitability_classifier")

    model = joblib.load(model_path)
    logger.info("loaded spacy suitability classifier: %s", model_path)
    return model


class SpacyLLMHybridConstructor(BaseDocumentConstructor):
    """按 LR 路由在 spaCy / LLM 两个 extractor 间二选一的构造器。"""

    def __init__(
        self,
        *,
        llm_extractor: Any,
        spacy_extractor: Any,
        classifier: Any,
        classifier_path: str | Path | None = None,
        log_every: int = 100,
    ) -> None:
        self._llm_extractor = llm_extractor
        self._spacy_extractor = spacy_extractor
        self._classifier = classifier
        self._classifier_path = str(classifier_path or "")
        self._log_every = max(int(log_every), 1)

        from nuggetindex.extractors.base import accepts_source_id

        self._llm_accepts_source_id = accepts_source_id(llm_extractor)
        self._spacy_accepts_source_id = accepts_source_id(spacy_extractor)

        # 路由计数,便于建库后核对 LLM 调用节省比例。
        self._n_spacy = 0
        self._n_llm = 0
        self._n_total = 0

    def _route(self, text: str) -> int:
        """阻塞式 LR 推断;返回 1=spaCy,0=LLM(仅在线程池中调用)。"""
        pred = self._classifier.predict([text])
        return int(pred[0])

    async def aprocess(
        self,
        doc: Document,
        *,
        existing: list[Nugget] | None = None,
        fetch_existing_by_key: FetchExistingByKey | None = None,
        alias_resolver: AliasResolver | None = None,
    ) -> list[Nugget]:
        # sklearn 推断是 CPU 阻塞调用,丢线程池避免卡住事件循环。
        use_spacy = await asyncio.to_thread(self._route, doc.text)

        if use_spacy == 1:
            extractor = self._spacy_extractor
            accepts_source_id = self._spacy_accepts_source_id
            route_name = "spacy"
            self._n_spacy += 1
        else:
            extractor = self._llm_extractor
            accepts_source_id = self._llm_accepts_source_id
            route_name = "llm"
            self._n_llm += 1
        self._n_total += 1

        kwargs: dict[str, str] = {}
        if accepts_source_id:
            kwargs["source_id"] = doc.source_id

        logger.debug(
            "route -> %s (classifier=%s, source_id=%s)",
            route_name,
            Path(self._classifier_path).name or "<inline>",
            doc.source_id,
        )
        if self._n_total % self._log_every == 0:
            logger.info(
                "hybrid routing after %d docs: spacy=%d (%.1f%%), llm=%d (%.1f%%)",
                self._n_total,
                self._n_spacy,
                100.0 * self._n_spacy / self._n_total,
                self._n_llm,
                100.0 * self._n_llm / self._n_total,
            )

        raw_results = await extractor.aextract(doc.text, **kwargs)
        return [r.nugget for r in raw_results]


@register_constructor("spacy_llm_hybrid")
def build_spacy_llm_hybrid_constructor(
    store: Any,
    *,
    classifier_path: str | Path | None = None,
    spacy_model: str = "en_core_web_sm",
    spacy_max_facts: int = 20,
) -> SpacyLLMHybridConstructor:
    """从 store 装配混合构造器。

    Args:
        store: 已配置 LLM extractor 的 ``NuggetStore``(建库 YAML 需为
            ``extractor.type: llm``);``store._extractor`` 作为 LLM 分支。
        classifier_path: LR 模型 joblib 路径,默认
            ``models/spacy_suitability_logreg_th6.joblib``。
        spacy_model: spaCy 分支使用的模型名。
        spacy_max_facts: spaCy 分支每篇文档的事实上限。

    Raises:
        RuntimeError: store 未配置 extractor,或分类器模型文件不存在。
    """
    llm_extractor = getattr(store, "_extractor", None)
    if llm_extractor is None:
        raise RuntimeError(
            "spacy_llm_hybrid DocumentConstructor 需要 LLM extractor 作为"
            " 0 号分支,但 NuggetStore 未配置 extractor(请确认建库配置 "
            "extractor.type=llm,且不是 --reuse-db/skip-ingest 模式)。"
        )

    from tcrag.extractors.spacy_extractor import SpacyFactExtractor
    from tcrag.extractors.extractor_wrapper import ExtractorWrapper

    spacy_extractor = ExtractorWrapper(
        SpacyFactExtractor(model=spacy_model, max_facts=spacy_max_facts)
    )
    # 提前加载 spaCy 模型,与标注脚本同口径:首篇文档并发时避免懒加载竞争,
    # 也让模型缺失在建库开始时就快速失败。
    spacy_extractor._extractor._ensure_nlp()

    clf_path = Path(classifier_path) if classifier_path else _DEFAULT_CLASSIFIER
    classifier = load_suitability_classifier(clf_path)

    logger.info(
        "building SpacyLLMHybridConstructor: llm=%s, spacy=%s, classifier=%s",
        type(llm_extractor).__name__,
        spacy_model,
        clf_path,
    )
    return SpacyLLMHybridConstructor(
        llm_extractor=llm_extractor,
        spacy_extractor=spacy_extractor,
        classifier=classifier,
        classifier_path=clf_path,
    )
