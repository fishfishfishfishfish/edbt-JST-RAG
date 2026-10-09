"""Registry of pluggable ``DocumentConstructor`` implementations.

Before its first ingestion, the RAG system (see
:class:`tcrag.rag_systems.jst_system.JSTRAGSystem`)
installs the selected implementation onto ``NuggetStore._constructor`` via :func:`build_constructor`.
``spec`` supports three forms:

  - ``None``: do not inject; keep nuggetindex's own lazily-built native constructor;
  - ``str``: look up a built-in registered name in this table (``"spacy_llm_hybrid"``);
  - ``callable``: invoked directly as ``factory(store, **kwargs)``, convenient for passing parameters in
    benchmark scripts with ``lambda store: ...`` / ``functools.partial``,
    keeping a style consistent with ``retriever_factory(store)``.

Adding a new constructor implementation requires only:

  1. Add a module in this directory that inherits from
     :class:`tcrag.constructors.base.BaseDocumentConstructor`, implements
     ``aprocess``, and provides ``factory(store) -> constructor``;
  2. Register it with :func:`register_constructor` (or write it directly into
     :data:`CONSTRUCTOR_REGISTRY`), after which it can be referenced by name.
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
    """Decorator: register ``factory(store, **kwargs)`` into the registry under ``name``."""

    def _decorator(factory: ConstructorFactory) -> ConstructorFactory:
        CONSTRUCTOR_REGISTRY[name] = factory
        return factory

    return _decorator


# Implementation modules must be imported only after the registry and decorator are defined: the
# @register_constructor at the top of a submodule registers itself into the already-existing registry on import (reversing the order would
# hit a half-initialized package and cause a circular-import error).
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
