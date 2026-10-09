"""spaCy small-model based atomic-fact extractor.

It uses a small spaCy model such as ``en_core_web_sm`` (default) to perform
open-domain triple extraction via **dependency parsing + NER**, without an
LLM or nuggetindex, producing :class:`~tcrag.data.models.AtomicFact` objects
(subject / predicate / object / text + NER types) consistent with the framework's other extractors.

Extraction strategy (per sentence)
===================================

1. Dependency parsing finds the verbal head (POS VERB; copular AUX+attr included):
   - subject: nsubj / nsubjpass;
   - object: dobj / attr / dative / oprd;
   - prepositional object: pobj of the verb's prep child (as in "worked at Norwich" →
     predicate="work at", object="Norwich"); the passive agent ("founded by
     Y") is also covered;
   - coordinated verbs (conj) reuse the same subject ("He served and retired").
2. Subject/object phrases preferentially use the spaCy noun_chunk (stripping leading determiners/possessive pronouns),
   keeping the phrase complete ("the House of Commons" rather than a bare "Commons").
3. NER labels are written into ``subject_type`` / ``object_type`` (PERSON / ORG /
   GPE / DATE ...).
4. When the subject is a pronoun ("He / She") and the TimeQA page entity can be recovered from ``source_id``,
   replace it with the page entity; if unrecoverable, skip the triple (pronoun subjects have no retrieval value).
5. The predicate is the verb lemma plus prepositions/particles ("give up"); passive forms keep the lemma form.

Dependency handling
===================

spaCy and its models are treated within **this module** as soft dependencies with lazy loading: this file does not import
spacy at the top level; ``spacy.load`` runs only on the first call to :meth:`SpacyFactExtractor.aextract`,
and a missing package/model raises a clear :class:`RuntimeError` carrying the install command (note that the package-level
``tcrag/extractors/__init__.py`` still imports the other extractors, which may depend on
nuggetindex — this is existing behavior; to use this module directly, run inside the tcrag conda environment).

    pip install spacy
    python -m spacy download en_core_web_sm

Configuration (configs/default.yaml)::

    extractor:
      type: spacy                # or spacy_sm
      spacy_model: en_core_web_sm
      spacy_max_facts: 20
"""

from __future__ import annotations

import asyncio
import re
from urllib.parse import unquote

from tcrag.data.models import AtomicFact
from tcrag.extractors.base import BaseExtractor

# source_id -> page entity resolution, consistent with timeqa_extractor (a
# local copy, to avoid importing timeqa_extractor which would make nuggetindex a hard dependency).
_TIMEQA_SOURCE_RE = re.compile(r"^(?:timeqa_)?/wiki/(?P<title>.+)_(?P<paragraph>\d+)$")
_SPACE_RE = re.compile(r"\s+")

_SUBJECT_DEPS = {"nsubj", "nsubjpass"}
_OBJECT_DEPS = {"dobj", "attr", "dative", "oprd", "pcomp"}
_PREP_DEPS = {"prep", "prepc"}          # legacy-label compatibility (spaCy 3 unifies them as prep)
_LEADING_DET_DEPS = {"det", "poss"}
# When extending PPs, reject prepositional phrases that contain clauses (which/that clauses, adverbial clauses, etc.).
_CLAUSAL_DEPS = {"relcl", "acl", "advcl", "ccomp", "xcomp", "parataxis"}
# Surface token -> predicate override: spaCy lemmatizes "born" as "bear",
# and "bear in Manchester" in a triple is meaningless for both retrieval and reading.
_VERB_SURFACE_OVERRIDE = {"born": "born", "borne": "born"}
# Prepositions that introduce clauses (whilst Dean ... / when he became ...), not nominal objects.
# Note "as" cannot be added: "worked as a scientist" is a useful nominal PP.
_CLAUSAL_PREPS = {
    "when", "while", "whilst", "although", "though", "if", "because",
    "unless", "whereas",
}

# Phrase length cap, filtering the occasional oversized subtree from the parser (same as rule_based's 60-per-sentence policy).
_MAX_PHRASE_CHARS = 80


def entity_from_source_id(source_id: str | None) -> str:
    """``timeqa_/wiki/Page_Title_3`` → ``"Page Title"``; empty when it cannot be parsed."""
    if not source_id:
        return ""
    m = _TIMEQA_SOURCE_RE.match(source_id.strip())
    if m is None:
        return ""
    title = unquote(m.group("title")).replace("_", " ").replace("&amp;", "&")
    return _SPACE_RE.sub(" ", title).strip()


class SpacyFactExtractor(BaseExtractor):
    """Extract (subject, predicate, object) facts with a small spaCy model (dependency parsing + NER).

    Parameters
    ----------
    model:
        spaCy model name, default ``en_core_web_sm``.
    max_facts:
        Upper bound on the facts returned per extraction (default 20, consistent with :class:`LLMExtractor`);
        ``<=0`` means no limit.
    """

    def __init__(self, model: str = "en_core_web_sm", *, max_facts: int = 20) -> None:
        self._model_name = model
        self._max_facts = max_facts
        self._nlp: object | None = None  # lazy-loaded; the runtime type is spacy.Language

    # ── Model loading (soft dependency) ────────────────────────────────

    def _ensure_nlp(self) -> object:
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - dependency-missing boundary
            raise RuntimeError(
                "SpacyFactExtractor 需要 spaCy,但当前环境未安装:"
                " pip install spacy && python -m spacy download "
                f"{self._model_name}"
            ) from exc
        try:
            # All tagger/parser/ner pipeline components are required (all enabled by default; declared explicitly here).
            self._nlp = spacy.load(self._model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy 模型 {self._model_name!r} 不可用,请先安装:"
                f" python -m spacy download {self._model_name}"
            ) from exc
        return self._nlp

    # ── Public interface ───────────────────────────────────────────────

    async def aextract(
        self,
        text: str,
        *,
        context: str = "",
        source_id: str | None = None,
    ) -> list[AtomicFact]:
        if not text or not text.strip():
            return []
        # spaCy parsing is a blocking CPU call; offload it to a thread pool to avoid blocking the event loop.
        return await asyncio.to_thread(self._extract_sync, text, source_id)

    # ── Core implementation ────────────────────────────────────────────

    def _extract_sync(self, text: str, source_id: str | None) -> list[AtomicFact]:
        nlp = self._ensure_nlp()
        doc = nlp(text)
        page_entity = entity_from_source_id(source_id)

        # token.i -> (chunk_text with determiners stripped, chunk.root)
        chunk_by_root: dict[int, object] = {}
        for chunk in doc.noun_chunks:
            chunk_by_root[chunk.root.i] = chunk

        # token.i -> NER label (every token inside the entity is mapped).
        ent_label: dict[int, str] = {}
        for ent in doc.ents:
            for tok in ent:
                ent_label[tok.i] = ent.label_

        facts: list[AtomicFact] = []
        seen: set[tuple[str, str, str]] = set()

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            for head in self._clause_heads(sent):
                subject_tok = self._subject_token(head)
                if subject_tok is None:
                    continue
                subject = self._phrase_for(subject_tok, chunk_by_root)
                if not subject:
                    continue
                # Pronoun subject: replace it if recoverable, otherwise skip (no value for BM25/indexing).
                if subject_tok.pos_ == "PRON":
                    if page_entity:
                        subject = page_entity
                    else:
                        continue

                for predicate, object_tok in self._objects_of(head):
                    obj = self._phrase_for(object_tok, chunk_by_root)
                    if not obj:
                        continue
                    key = (subject.lower(), predicate, obj.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    facts.append(
                        AtomicFact(
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            text=sent_text,
                            subject_type=ent_label.get(subject_tok.i),
                            object_type=ent_label.get(object_tok.i),
                            metadata={
                                "extractor": "spacy",
                                "spacy_model": self._model_name,
                                "char_start": sent.start_char,
                                "char_end": sent.end_char,
                            },
                        )
                    )

        if self._max_facts > 0:
            facts = facts[: self._max_facts]
        return facts

    # ── Parsing helpers ────────────────────────────────────────────────

    @staticmethod
    def _clause_heads(sent: object) -> list[object]:
        """Predicate heads of the sentence: the main-clause verb plus coordinated verbs reusing its subject.

        In copular constructions ("X was Y") the root may be an AUX; it still counts as long as it carries an attr.
        """
        heads: list[object] = []
        root = sent.root
        if root.pos_ == "VERB" or (
            root.pos_ == "AUX" and any(c.dep_ in _OBJECT_DEPS for c in root.children)
        ):
            heads.append(root)
        for child in root.children:
            if child.pos_ == "VERB" and child.dep_ in {"conj", "parataxis"}:
                heads.append(child)
        return heads

    @staticmethod
    def _subject_token(head: object) -> object | None:
        """Return the verb's subject; fall back to the parent clause's subject when a coordinated verb has no subject of its own."""
        for child in head.children:
            if child.dep_ in _SUBJECT_DEPS:
                return child
        # "He moved to London and worked at a bank" → worked reuses He.
        parent = head.head
        if head.dep_ in {"conj", "parataxis"} and parent is not head:
            for child in parent.children:
                if child.dep_ in _SUBJECT_DEPS:
                    return child
        return None

    @staticmethod
    def _objects_of(head: object) -> list[tuple[str, object]]:
        """Return [(predicate text, object token), ...].

        predicate = verb lemma, optionally with a particle (prt, e.g. "give up") and a preposition
        ("work at"); the passive agent is handled via agent→by→pobj.
        """
        verb_lemma = _VERB_SURFACE_OVERRIDE.get(head.text.lower(), head.lemma_)
        # Particles of phrasal verbs (give up / set up); conservatively take only prt, excluding acomp complements.
        particles = [c.lemma_ for c in head.children if c.dep_ == "prt"]
        base_pred = " ".join([verb_lemma, *particles]).strip()

        out: list[tuple[str, object]] = []

        # Direct objects / predicatives / datives / complements.
        for child in head.children:
            if child.dep_ in _OBJECT_DEPS:
                out.append((base_pred, child))

        # Prepositional objects / passive agent: worked at <X>; was founded by <Y>.
        for child in head.children:
            if child.dep_ in _PREP_DEPS or child.dep_ == "agent":
                prep = child.lemma_.lower()
                if child.dep_ == "agent":
                    prep = "by"
                # Skip clausal prepositions (when/whilst/...), and PPs whose subtree still contains a verb
                # (which indicates an adverbial clause rather than a nominal object).
                if prep in _CLAUSAL_PREPS:
                    continue
                sub_tokens = list(child.subtree)
                if any(t.pos_ in {"VERB", "AUX"} and t is not head for t in sub_tokens):
                    continue
                for grand in child.children:
                    if grand.dep_ in {"pobj", "pcomp"}:
                        out.append((f"{base_pred} {prep}".strip(), grand))

        return out

    @staticmethod
    def _phrase_for(token: object, chunk_by_root: dict[int, object]) -> str:
        """The complete phrase for a subject/object.

        spaCy v3 noun_chunks do **not** include following PPs by default, so taking the chunk directly would truncate
        "the University of East Anglia" into "University". The chunk is used here as the base: strip leading
        determiners/possessives → extend the PP along the noun's prep subtree (including one level of nesting, e.g.
        "Member of Parliament for Norwich North"; PPs containing clauses are rejected)
        → add nummod/npadvmod ("May 1997").
        """
        chunk = chunk_by_root.get(token.i)
        if chunk is not None:
            idxs = {
                t.i
                for t in chunk
                if not (t.dep_ in _LEADING_DET_DEPS and t.i < chunk.root.i)
            }
            # Extend nominal PPs with BFS: root → prep → pobj → prep ...
            queue = [chunk.root]
            while queue:
                noun = queue.pop()
                for child in noun.children:
                    if child.dep_ != "prep":
                        continue
                    sub = list(child.subtree)
                    if any(t.dep_ in _CLAUSAL_DEPS for t in sub):
                        continue
                    idxs.update(t.i for t in sub)
                    pobj = next(
                        (g for g in child.children if g.dep_ in {"pobj", "pcomp"}),
                        None,
                    )
                    if pobj is not None:
                        queue.append(pobj)
            root_token = chunk.root
        else:
            # Dates/numbers etc. may not lie inside a noun_chunk; start from the token itself.
            idxs = {token.i}
            root_token = token

        # Numeric/nominal modifiers (1997, $5M, etc.) are added whether or not they lie inside the chunk.
        for child in root_token.children:
            if child.dep_ in {"nummod", "npadvmod"}:
                idxs.update(t.i for t in child.subtree)

        doc = token.doc
        toks = [doc[i] for i in sorted(idxs)]
        phrase = "".join(t.text_with_ws for t in toks).strip()
        phrase = phrase.strip(" ,.;:!?'\"()[]{}").strip()
        phrase = _SPACE_RE.sub(" ", phrase)
        if len(phrase) > _MAX_PHRASE_CHARS:
            return ""
        return phrase
