"""可插拔 ``DocumentConstructor`` 的抽象基类。

nuggetindex 的 ``NuggetStore.aingest`` 在首次入库时懒建一个原生
``DocumentConstructor`` 并缓存到 ``store._constructor``;该属性可以被
外部预先赋值(与 ``store._retriever`` 的自定义检索器注入同型),从而在
**不修改 nuggetindex 仓库** 的前提下替换

    extract -> canonicalize -> temporal -> alias -> 实体校验 -> 去重 -> 冲突

整条文档构造管线。

本基类约定与 nuggetindex 原生
``DocumentConstructor.aprocess`` 完全一致的方法签名,因此任何子类实例
都可以直接作为 ``store._constructor`` 使用。实现类统一放在
:mod:`tcrag.constructors` 下,并在其 ``__init__`` 的注册表中登记。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from nuggetindex.core.models import Nugget
    from nuggetindex.pipeline.aliases import AliasResolver
    from nuggetindex.pipeline.constructor import Document

    # 与 nuggetindex.pipeline.constructor.FetchExistingByKey 对齐:
    # (subject, predicate, object) -> 已入库同 key nuggets。
    FetchExistingByKey = Callable[[tuple[str, str, str]], Awaitable[list[Nugget]]]


class BaseDocumentConstructor(ABC):
    """自定义文档构造器基类(duck-type 兼容 nuggetindex 原生实现)。"""

    @abstractmethod
    async def aprocess(
        self,
        doc: Document,
        *,
        existing: list[Nugget] | None = None,
        fetch_existing_by_key: FetchExistingByKey | None = None,
        alias_resolver: AliasResolver | None = None,
    ) -> list[Nugget]:
        """对单个文档执行构造管线,返回可直接落盘的 nugget 列表。

        Args:
            doc: nuggetindex ``Document``(``source_id`` / ``text`` /
                ``uri`` / ``source_date``)。
            existing: 调用方显式给定的同文档已有 nugget(直接调用场景);
                经 ``NuggetStore.aingest`` 进入时为 ``None``,跨文档 peers
                通过 ``fetch_existing_by_key`` 拉取。
            fetch_existing_by_key: 按 ``(subject, predicate, object)``
                异步取回后端已落库 peers 的回调,用于去重 / 冲突检测。
            alias_resolver: store 级、跨文档累积的别名解析器;为 None 时
                实现可退化为单文档内解析。
        """
        # pragma: no cover - 抽象签名仅用于声明契约
        raise NotImplementedError
