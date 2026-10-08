"""可插拔 ``DocumentConstructor`` 实现注册表。

RAG 系统(见
:class:`tcrag.rag_systems.jst_system.JSTRAGSystem`)
在首次入库前通过 :func:`build_constructor` 把所选实现安装到
``NuggetStore._constructor``。``spec`` 支持三种形式:

  - ``None``:不注入,沿用 nuggetindex 自身的懒建原生构造器;
  - ``str``:查本表内置注册名(``"spacy_llm_hybrid"``);
  - ``callable``:直接作为 ``factory(store, **kwargs)`` 调用,便于在
    benchmark 脚本里用 ``lambda store: ...`` / ``functools.partial``
    传参,风格与 ``retriever_factory(store)`` 保持一致。

新增一种构造器实现只需:

  1. 在本目录新增模块,继承
     :class:`tcrag.constructors.base.BaseDocumentConstructor` 实现
     ``aprocess``,并提供 ``factory(store) -> constructor``;
  2. 用 :func:`register_constructor` 注册(或直接写入
     :data:`CONSTRUCTOR_REGISTRY`),之后即可按名称引用。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tcrag.constructors.base import BaseDocumentConstructor

__all__ = [
    "BaseDocumentConstructor",
    "SpacyLLMHybridConstructor",
    "CONSTRUCTOR_REGISTRY",
    "ConstructorFactory",
    "register_constructor",
    "build_constructor",
    "build_spacy_llm_hybrid_constructor",
]

#: A factory that takes a prepared NuggetStore and returns an instance satisfying the
#: ``DocumentConstructor.aprocess`` contract.
ConstructorFactory = Callable[..., Any]


#: Registry of constructor implementations; modules self-register via
#: :func:`register_constructor` on import.
CONSTRUCTOR_REGISTRY: dict[str, ConstructorFactory] = {}


def register_constructor(name: str) -> Callable[[ConstructorFactory], ConstructorFactory]:
    """装饰器:把 ``factory(store, **kwargs)`` 以 ``name`` 登记进注册表。"""

    def _decorator(factory: ConstructorFactory) -> ConstructorFactory:
        CONSTRUCTOR_REGISTRY[name] = factory
        return factory

    return _decorator


# 必须在注册表与装饰器定义之后再导入实现模块:子模块顶部的
# @register_constructor 在导入时把自己登记进已存在的注册表(顺序反过来会
# 命中半初始化的包,产生循环导入错误)。
from tcrag.constructors.spacy_llm_hybrid import (  # noqa: E402
    SpacyLLMHybridConstructor,
    build_spacy_llm_hybrid_constructor,
)


def build_constructor(
    spec: str | Callable[..., Any] | None,
    store: Any,
    **kwargs: Any,
) -> Any:
    """Constructs a DocumentConstructor instance according to ``spec``.

    Args:
        spec: Registered name / ``factory(store)`` callable / None.
        store: The target ``NuggetStore``, passed to the factory to read the
            extractor, schema, etc.
        **kwargs: Extra parameters forwarded to the factory (built-in factories
            only accept ``store``).

    Raises:
        ValueError: Unknown registered name.
        TypeError: Unsupported ``spec`` type.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        try:
            factory = CONSTRUCTOR_REGISTRY[spec]
        except KeyError:
            known = ", ".join(sorted(CONSTRUCTOR_REGISTRY)) or "(empty)"
            raise ValueError(
                f"unknown constructor {spec!r}; available: {known}"
            ) from None
    elif callable(spec):
        factory = spec
    else:
        raise TypeError(
            "constructor spec must be a registered name (str), a "
            f"factory(store) callable, or None; got {type(spec).__name__}"
        )
    return factory(store, **kwargs) if kwargs else factory(store)
