"""LR-routed spaCy / LLM hybrid extraction constructor (registered name ``"spacy_llm_hybrid"``).

When each document enters :meth:`SpacyLLMHybridConstructor.aprocess`, the
LogisticRegression trained by ``scripts/utils/train_spacy_suitability_classifier.py``
makes one routing inference on the raw text:

  - The model outputs ``1`` (the passage is suitable for spaCy) → it uses the local
    :class:`tcrag.extractors.spacy_extractor.SpacyFactExtractor`
    (adapted to the nuggetindex interface via
    :class:`tcrag.extractors.extractor_wrapper.ExtractorWrapper`), with zero LLM calls;
  - The model outputs ``0`` → it uses the LLM extractor already configured on the store (the index-build config must
    set ``extractor.type: llm``; i.e. the nuggetindex LLMExtractor / OllamaCompatClient chain
    wrapped by PlaceholderValidity).

Both extractor branches expose the nuggetindex
``aextract(text, source_id=...) -> list[ExtractionResult]`` interface; like
:class:`tcrag.constructors.extract_only.ExtractOnlyConstructor`, this constructor
takes only ``.nugget`` and does no post-processing such as canonicalization / temporal inference / dedup / conflict resolution.

The classifier model is serialized with joblib; the pickle owner module of its custom feature class ``TextStats`` is fixed
to ``train_spacy_suitability_classifier`` (see the training script), so before loading,
``scripts/utils`` must be added to ``sys.path`` and that module imported once;
:func:`load_suitability_classifier` already encapsulates this bootstrap.

Factory parameters (passed through via ``JSTRAGSystem(constructor_kwargs=...)``)::

    build_spacy_llm_hybrid_constructor(
        store,
        classifier_path="models/spacy_suitability_logreg_th6.joblib",  # default
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
# parents[2]=repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CLASSIFIER = (
    _REPO_ROOT / "models" / "spacy_suitability_logreg_th6.joblib"
)
_TRAIN_UTILS_DIR = _REPO_ROOT / "scripts" / "utils"


def load_suitability_classifier(model_path: str | Path) -> Any:
    """Load the spaCy suitability LR model (joblib) and complete the pickle deserialization bootstrap.

    The model pipeline contains the custom feature class ``TextStats`` defined by the training script; its pickle
    owner module is ``train_spacy_suitability_classifier``; when that training module is imported,
    it registers its module object into ``sys.modules`` itself, and joblib restores the class based on that.
    """
    import joblib

    model_path = Path(model_path)
    if not model_path.is_file():
        raise RuntimeError(f"spaCy 适用性分类器模型不存在: {model_path}")

    utils_dir = str(_TRAIN_UTILS_DIR)
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    # Import side effect: the training script executes sys.modules.setdefault(
    # "train_spacy_suitability_classifier", ...), enabling joblib to resolve TextStats.
    importlib.import_module("train_spacy_suitability_classifier")

    model = joblib.load(model_path)
    logger.info("loaded spacy suitability classifier: %s", model_path)
    return model


class SpacyLLMHybridConstructor(BaseDocumentConstructor):
    """Constructor that picks one of the two extractors (spaCy / LLM) according to LR routing."""

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

        # Routing counters, to verify the proportion of LLM calls saved after index construction.
        self._n_spacy = 0
        self._n_llm = 0
        self._n_total = 0

    def _route(self, text: str) -> int:
        """Blocking LR inference; returns 1=spaCy, 0=LLM (called only inside the thread pool)."""
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
        # sklearn inference is a blocking CPU call; offload it to the thread pool to avoid blocking the event loop.
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
    """Assemble the hybrid constructor from the store.

    Args:
        store: A ``NuggetStore`` configured with an LLM extractor (the index-build YAML must set
            ``extractor.type: llm``); ``store._extractor`` serves as the LLM branch.
        classifier_path: Path to the LR model joblib, defaulting to
            ``models/spacy_suitability_logreg_th6.joblib``.
        spacy_model: Model name used by the spaCy branch.
        spacy_max_facts: Upper bound of facts per document for the spaCy branch.

    Raises:
        RuntimeError: The store has no extractor configured, or the classifier model file does not exist.
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
    # Eagerly load the spaCy model, consistent with the annotation script: avoid lazy-loading races when the first
    # documents arrive concurrently, and also fail fast at the start of index construction if the model is missing.
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
