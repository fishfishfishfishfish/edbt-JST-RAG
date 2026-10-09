"""MRAG time-aware retriever: a standalone, reusable Retriever component.

Refactored from the original MRAG metriever.py, preserving the core MRAG retrieval pipeline:

1. **Initial retrieval**: BM25 (via the nuggetindex store backend or Pyserini LuceneSearcher)
2. **Keyword ranking**: compute keyword-weighted hit scores for candidate passages, keeping the top ctx_topk
3. **Semantic ranking**: rerank with models such as CrossEncoder / BGE / NV-Embed
4. **QFS summarization**: the LLM generates question-focused summaries for the top-k passages (keeping key dates)
5. **Sentence-level keyword ranking**: sentence-level keyword hits plus summaries participating in ranking
6. **Temporal-semantic hybrid ranking**:
   ``final_score = hybrid_base * semantic + (1-hybrid_base) * semantic * temporal_coeff``
   By default ``hybrid_base=0``, which is equivalent to ``final_score = semantic * temporal_coeff``

Time-awareness core:
- ``year_identifier(text)``: extract years with regex, first expanding "1990-93"/"1990-1995" ranges
- ``remove_implicit_condition(q)``: detect implicit conditions such as latest/last/first/earliest
- ``get_spline_function(...)``: linear interpolation, window [0.6, 1.0], span of 50 years
- ``get_temporal_coeffs(...)``: filter eligible years by temporal relation type (between/before/after)

Compatible with the ``retriever_factory`` configuration of
``scripts/run_benchmark_jstrag.py``:

.. code-block:: yaml

    systems:
      jstrag:
        retriever_factory: "tcrag.retrievers.metriever:create_mrag_retriever"

The MRAGRetriever returned by the factory function
``create_mrag_retriever(store, *, llm=None)`` implements
``async aretrieve(query, *, query_time, view, top_k, fusion, filters) -> list[RetrievalResult]``.

The rerank model is selected through the environment variables ``RERANKER_TYPE``
and ``RERANKER_MODEL``.
RERANKER_TYPE="cross_encoder"
RERANKER_MODEL="/home/xinyuchen/TCRag/models/ms-marco-MiniLM-L6-v2"
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from tcrag.logging_config import get_logger
from tcrag.utils import sanitize_fts_query

logger = get_logger("retrievers.metriever")

#: Paragraph index at the end of source_id (TimeQA: ``.../wiki/Besart_Berisha_18``).
_PARA_IDX_SUFFIX = re.compile(r"_\d+$")


def _title_from_source_id(source_id: str) -> str:
    """TimeQA-style ``source_id`` → human-readable document title.

    ``timeqa_/wiki/Besart_Berisha_18`` → ``Besart Berisha``.
    Returns ``""`` when no paragraph can be extracted (the caller keeps the empty title).
    """
    tail = source_id.rsplit("/", 1)[-1]
    if not tail:
        return ""
    tail = _PARA_IDX_SUFFIX.sub("", tail)
    return tail.replace("_", " ").strip()


# ── Optional dependencies (with graceful degradation when missing) ────────

# NLTK dependencies come in two layers (a common pitfall: the nltk pip package is
# installed but the corpus data is missing, raising LookupError at runtime):
#   1) pip package:
#        pip install nltk
#   2) Corpus data -- nltk>=3.9 requires punkt_tab / averaged_perceptron_tagger_eng;
#      online downloads (`python -m nltk.downloader ...`) go through raw.githubusercontent.com,
#      which on weak networks often hangs on timeout or fails with SSL EOF. It is
#      recommended to manually download the zip and extract it to the target directory:
#
#      Data package download URL
#      git clone https://github.com/nltk/nltk_data.git
#      Rename nltk_data/packages to nltk_data and place it at ~/nltk_data

#      If the nltk_data directory is group-writable it triggers a
#      "non-private download directory" warning:
#        chmod 755 ~/nltk_data
#      The data can also be placed inside the conda environment (to avoid polluting
#      home): ~/miniconda3/envs/<env>/nltk_data/;
#      or set the environment variable NLTK_DATA=/your/dir to customize the search path.
#       Check the nltk_data directory path via
#       python -c "import nltk; print(nltk.data.path)"
#
#      Verification:
#        python -c "from nltk.tokenize import sent_tokenize; \
#           print(sent_tokenize('A test. Another one.'))"
#   Note: the legacy punkt / averaged_perceptron_tagger are no longer used in
#   nltk>=3.9 and need not be downloaded.
# If either is missing, fall back to regex sentence splitting/tokenization,
# all-NN POS tags, and an empty lemmatizer:
#   the MRAG pipeline still runs, but morphological variant expansion of keywords
#   is disabled, which may reduce recall.
_nltk_data_available = False
try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    from nltk.tokenize import word_tokenize as _nltk_word_tokenize
    from nltk.tag import pos_tag as _nltk_pos_tag
    from nltk.stem import WordNetLemmatizer as _NLTKWordNetLemmatizer
    _nltk_sent_tokenize("Test sentence.")  # Trigger the data availability check
    _nltk_data_available = True
except Exception as _nltk_exc:  # ImportError (nltk package missing) or LookupError (corpus data missing)
    logger.debug("NLTK 语料不可用,降级为 regex 分词: %s", _nltk_exc)

if _nltk_data_available:
    sent_tokenize = _nltk_sent_tokenize
    word_tokenize = _nltk_word_tokenize
    pos_tag = _nltk_pos_tag
    WordNetLemmatizer = _NLTKWordNetLemmatizer
else:
    def sent_tokenize(text: str) -> list[str]:
        """Regex sentence splitter (fallback implementation when NLTK data is missing)."""
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sents if s]

    def word_tokenize(text: str) -> list[str]:
        """Regex tokenizer (fallback implementation when NLTK data is missing)."""
        return re.findall(r"\b[\w'-]+\b", text)

    def pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
        """Fallback POS tagger: tag every token as NN."""
        return [(t, "NN") for t in tokens]

    class WordNetLemmatizer:  # type: ignore[no-redef]
        """Fallback lemmatizer: return the original word without lemmatization."""
        def lemmatize(self, word: str, pos: str = "n") -> str:
            return word

# pattern.en is used for morphological variant expansion; falls back to a no-op when missing
try:
    from pattern.en import lexeme as _pattern_lexeme
    _pattern_available = True
except Exception:
    _pattern_available = False
    _pattern_lexeme = None

# torch / transformers (used by semantic models, optional)
try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    _torch_available = True
except ImportError:
    _torch_available = False

# regex library (used by the Unicode-aware tokenizer, optional)
try:
    import regex as _re_mod
    import unicodedata as _ud
    _regex_available = True
except ImportError:
    _regex_available = False
    import re as _re_mod  # noqa: F811


# ═══════════════════════════════════════════════════════════════════════════
# Constants and configuration
# ═══════════════════════════════════════════════════════════════════════════

# Keyword exclusion list of low-information words
EXCL = [
    "time", "years", "for", "new", "recent", "current",
    "whom", "who", "out", "place", "not",
]

# Keyword hit weights
KEYWORD_WEIGHTS = {
    "special": 1.0,
    "superlative": 0.7,
    "numeric": 0.5,
    "general": 0.4,
    "adjective": 0.4,
}

# Number mappings
NUMBER_MAP = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty",
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth",
}
NUMBER_MAP_B = {v: k for k, v in NUMBER_MAP.items()}

# Month mappings
MONTH_TO_NUMBER = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}
SHORT_MONTH_TO_NUMBER = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Temporal coefficient curve parameters
TEMPORAL_LOW = 0.6
TEMPORAL_SPAN = 50
TEMPORAL_FALLBACK = 0.5

# Temporal-relation triggers scanned when parse_temporal_question performs auto-detection.
# The classification matches the original MRAG: before-type / after-type; multi-word triggers take priority.
_TIME_RELATION_TRIGGERS: tuple[str, ...] = (
    "as of", "before", "after", "since", "until", "from", "by",
)


# ═══════════════════════════════════════════════════════════════════════════
# DPR-style SimpleTokenizer & has_answer (ported from contriever/src/evaluation.py)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleTokenizer:
    """DPR-style Unicode-aware tokenizer.

    Splits tokens using Unicode property escapes (``\\p{L}`` etc.) from the
    ``regex`` library; falls back to the ``\\w`` / ``\\S`` patterns of the
    standard ``re`` library when ``regex`` is unavailable.
    """

    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"
    # Fallback patterns for the re library (\p Unicode properties are not supported)
    _ALPHA_NUM_FALLBACK = r"[\w]+"
    _NON_WS_FALLBACK = r"\S"

    def __init__(self):
        if _regex_available:
            pattern_str = f"({self.ALPHA_NUM})|({self.NON_WS})"
            self._regexp = _re_mod.compile(
                pattern_str,
                flags=_re_mod.IGNORECASE + _re_mod.UNICODE + _re_mod.MULTILINE,
            )
        else:
            pattern_str = f"({self._ALPHA_NUM_FALLBACK})|({self._NON_WS_FALLBACK})"
            self._regexp = re.compile(pattern_str, flags=re.IGNORECASE | re.MULTILINE)

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            return [m.group().lower() for m in matches]
        return [m.group() for m in matches]


def _normalize(text: str) -> str:
    """NFD normalization."""
    if _regex_available:
        return _ud.normalize("NFD", text)
    return text


def has_answer(answers, text, tokenizer: SimpleTokenizer) -> bool:
    """Check whether the document contains the answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)
    for answer in answers:
        answer = _normalize(answer)
        answer_tokens = tokenizer.tokenize(answer, uncased=True)
        for i in range(len(text) - len(answer_tokens) + 1):
            if answer_tokens == text[i : i + len(answer_tokens)]:
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Temporal parsing utilities (ported from utils.py)
# ═══════════════════════════════════════════════════════════════════════════

def find_month(w: str) -> int | None:
    """Extract the month number from a string."""
    w = w.lower()
    for m, num in MONTH_TO_NUMBER.items():
        if m in w:
            return num
    for m, num in SHORT_MONTH_TO_NUMBER.items():
        if m in w:
            return num
    return None


def replace_dates(text: str) -> str:
    """Expand year ranges in the "1990-93" format."""
    pattern = r"(\b\d{4})[–-](\d{2}\b)"

    def replace_func(match):
        start_year = match.group(1)
        end_year = start_year[:2] + match.group(2)
        return " ".join(str(i) for i in range(int(start_year), int(end_year) + 1))

    return re.sub(pattern, replace_func, text)


def expand_year_range(text: str) -> str:
    """Expand year ranges in the "1990-1995" format."""
    def replace_range(match):
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        return " ".join(str(year) for year in range(start_year, end_year + 1))

    pattern = r"(\d{4})[–-](\d{4})"
    return re.sub(pattern, replace_range, text)


def year_identifier(timestamp: str) -> list[int]:
    """Extract all four-digit years from the text.

    Year ranges are expanded first (e.g. "1990-93" → "1990 1991 ... 1993") and
    then matched with regex.
    Returns a deduplicated, sorted list of integers; returns an empty list when
    nothing matches.
    """
    timestamp = replace_dates(timestamp)
    timestamp = expand_year_range(timestamp)
    pattern = r"\b(\d{4})(?:s)?\b"
    years = re.findall(pattern, timestamp)
    if years:
        return sorted(set(map(int, years)))
    return []


def remove_implicit_condition(no_time_question: str) -> tuple[str, str | None]:
    """Detect and remove implicit temporal condition words.

    Returns:
        (normalized_question, implicit_condition):
        implicit_condition is 'first', 'last', or None.
    """
    mapping = {
        " latest": "last",
        " last": "last",
        " first": "first",
        " earliest": "first",
        " most recent": "last",
        " recent": "last",
    }
    implicit_condition = None
    for key, val in mapping.items():
        if key in no_time_question:
            no_time_question = no_time_question.replace(key, "")
            implicit_condition = val
            break
    no_time_question = no_time_question.strip()
    if no_time_question and no_time_question[-1] not in ".?":
        no_time_question += "?"
    return no_time_question, implicit_condition


def get_wordnet_pos(treebank_tag: str) -> str:
    """Map a treebank POS tag to a WordNet POS."""
    if treebank_tag.startswith("J"):
        return "a"
    elif treebank_tag.startswith("V"):
        return "v"
    elif treebank_tag.startswith("N"):
        return "n"
    elif treebank_tag.startswith("R"):
        return "r"
    return "n"


def expand_keywords(
    keyword_list: list[str],
    normalized_question: str,
    verbose: bool = False,
) -> tuple[list[list[str]], list[str]]:
    """Expand keyword variants and classify them.

    Returns:
        (expanded_keyword_list, keyword_type_list):
        - expanded_keyword_list: list of variants for each keyword (including the original)
        - keyword_type_list: type of each keyword (special/superlative/general/numeric/adjective)
    """
    if not _pattern_available:
        # No morphological variant expansion when pattern.en is unavailable
        keyword_types = []
        for kw in keyword_list:
            if kw[0].isupper():
                keyword_types.append("special")
            elif kw.lower() in NUMBER_MAP or kw.lower() in NUMBER_MAP_B:
                keyword_types.append("numeric")
            else:
                keyword_types.append("general")
        return [[kw] for kw in keyword_list], keyword_types

    lemmatizer = WordNetLemmatizer()
    q_words, q_tags, q_lemmas = [], [], []

    tokens = word_tokenize(normalized_question)
    tagged_tokens = pos_tag(tokens)
    for word, tag in tagged_tokens:
        q_words.append(word)
        q_tags.append(tag)
        wordnet_pos = get_wordnet_pos(tag)
        q_lemmas.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    expanded_keyword_list = []
    keyword_type_list = []

    for kw in keyword_list:
        new_kw: list[str] = []
        kw_type: str

        if kw[0].isupper():
            kw_type = "special"
        elif kw.lower() in NUMBER_MAP:
            kw_type = "numeric"
            new_kw.append(NUMBER_MAP[kw.lower()])
        elif kw.lower() in NUMBER_MAP_B:
            kw_type = "numeric"
            new_kw.append(NUMBER_MAP_B[kw.lower()])
        else:
            kw_list = kw.split()
            n_words = len(kw_list)
            index = None
            try:
                for i in range(len(q_words)):
                    flgs = [
                        q_words[i + j].lower() == kw_list[j].lower()
                        for j in range(len(kw_list))
                    ]
                    if all(flgs):
                        index = i
                        break
            except Exception:
                pass
            if index is not None:
                last_word = kw_list[-1]
                last_index = index + n_words - 1
                last_tag = q_tags[last_index]
                if last_tag.startswith("J"):
                    kw_type = "superlative" if last_word[-3:] == "est" or last_word.lower() == "most" else "adjective"
                else:
                    kw_type = "general"
                    new_kw += [kw.replace(last_word, x) for x in _pattern_lexeme(last_word)]
            else:
                kw_type = "general"

        tmp = list(set([kw] + new_kw))
        if kw_type == "special" and " and " in kw:
            tmp += [kw.replace(" and ", "&"), kw.replace(" and ", " & "), kw.replace(" and ", " N' ")]
        if "-" in kw:
            tmp.append(kw.replace("-", " "))
        expanded_keyword_list.append(tmp)
        keyword_type_list.append(kw_type)

    return expanded_keyword_list, keyword_type_list


def count_keyword_scores(
    text: str,
    expanded_keyword_list: list[list[str]],
    keyword_type_list: list[str],
) -> float:
    """Compute the weighted keyword-hit score in the text."""
    text = text.lower()
    score = 0.0
    tokenizer = SimpleTokenizer()
    for i in range(len(expanded_keyword_list)):
        keywords = expanded_keyword_list[i]
        kw_type = keyword_type_list[i]
        if kw_type == "general":
            hit = any(kw.lower() in text for kw in keywords)
        else:
            hit = has_answer(keywords, text, tokenizer)
        if hit:
            score += KEYWORD_WEIGHTS[kw_type]
    return score


# ═══════════════════════════════════════════════════════════════════════════
# Temporal coefficient computation (ported from the original metriever.py)
# ═══════════════════════════════════════════════════════════════════════════

def get_spline_function(
    time_relation_type: str,
    implicit_condition: str | None,
    question_years: list[int],
) -> Callable:
    """Build a linear interpolation function for temporal coefficients.

    The coefficient within the window varies linearly between TEMPORAL_LOW
    (0.6) and 1.0:
    - implicit_condition='first': favors earlier years (decreasing from 1.0 to 0.6)
    - Otherwise (including 'last'): favors later years (increasing from 0.6 to 1.0)
    """
    from scipy.interpolate import interp1d
    import numpy as np

    low = TEMPORAL_LOW
    span = TEMPORAL_SPAN

    if len(question_years) == 2:
        start = min(question_years)
        end = max(question_years)
    elif time_relation_type == "before":
        end = question_years[0]
        start = end - span
    else:
        start = question_years[0]
        end = start + span

    x_points = np.array([start, end])
    if implicit_condition == "first":
        y_points = np.array([1, low])
    else:
        y_points = np.array([low, 1])

    return interp1d(x_points, y_points, kind="linear")


def get_temporal_coeffs(
    years: list[int],
    sentence_tuples: list[tuple],
    time_relation_type: str,
    implicit_condition: str | None,
    spline: Callable,
) -> list[float]:
    """Compute the temporal coefficient for each sentence.

    Logic:
    1. Extract years from the sentence text
    2. Filter eligible years according to the temporal relation type
    3. Select the representative year (earliest or latest) based on implicit_condition
    4. Compute the coefficient with the spline function; fall back to 0.5 when
       there is no eligible year or the value is out of range
    """
    temporal_coeffs: list[float] = []
    for _, snt, _ in sentence_tuples:
        snt_years = year_identifier(snt)
        closest_year = None

        if time_relation_type == "between":
            start = min(years)
            end = max(years)
            if snt_years:
                relevant = [y for y in snt_years if start <= y <= end]
                if relevant:
                    relevant = sorted(relevant)
                    if implicit_condition == "first":
                        closest_year = relevant[0]
                    else:
                        closest_year = relevant[-1]
        else:
            question_year = years[0]
            if snt_years:
                if time_relation_type == "before":
                    relevant = [y for y in snt_years if y <= question_year]
                else:
                    relevant = [y for y in snt_years if y >= question_year]
                relevant = sorted(relevant)
                if relevant:
                    if implicit_condition == "first":
                        closest_year = min(relevant)
                    else:
                        closest_year = max(relevant)

        try:
            coeff = float(spline(closest_year))
        except Exception:
            coeff = TEMPORAL_FALLBACK
        temporal_coeffs.append(coeff)
    return temporal_coeffs


# ═══════════════════════════════════════════════════════════════════════════
# LLM prompt generation (ported from prompts.py)
# ═══════════════════════════════════════════════════════════════════════════

def get_keyword_prompt(question: str) -> str:
    """Generate the LLM prompt for keyword extraction."""
    return f"""Your task is to extract keywords from the question. Response by a list of keyword strings. Do not include pronouns, prepositions, articles.

There are some examples for you to refer to:
<Question>
When was the last time the United States hosted the Olympics?
</Question>
<Keywords>
["United States", "hosted", "Olympics"]
</Keywords>

<Question>
Who sang 1 national anthem for Super Bowl last year?
</Question>
<Keywords>
["sang", "1", "national anthem", "Super Bowl"]
</Keywords>

<Question>
Most goals in international football?
</Question>
<Keywords>
["most", "goals", "international", "football"]
</Keywords>

<Question>
How many TV episodes in the series The Crossing?
</Question>
<Keywords>
["TV", "episodes", "series", "The Crossing"]
</Keywords>

<Question>
Who runs the fastest 40-yard dash in the NFL?
</Question>
<Keywords>
["runs", "fastest", "40-yard", "dash", "NFL"]
</Keywords>

<Question>
{question}
</Question>
<Keywords>
"""


def get_qfs_prompt(document: str, question: str) -> str:
    """Generate the LLM prompt for QFS (Query-Focused Summarization).

    The LLM is required to keep key dates and return "None" when the document
    is irrelevant.
    """
    return f"""You are a summarizer summarizing a retrieved document about a user question. Keep the key dates in the summarization. Write "None" if the document has no relevant content about the question.

There are some examples for you to refer to:
<Document>
David Beckham | As the summer 2003 transfer window approached, Manchester United appeared keen to sell Beckham to Barcelona and the two clubs even announced that they reached a deal for Beckham's transfer, but instead he joined reigning Spanish champions Real Madrid for €37 million on a four-year contract. Beckham made his Galaxy debut, coming on for Alan Gordon in the 78th minute of a 0–1 friendly loss to Chelsea as part of the World Series of Soccer on 21 July 2007.
</Document>
<Question>
David Beckham played for which team?
</Question>
<Summarization>
David Beckham played for Real Madrid from 2003 to 2007 and for LA Galaxy from July 21, 2007.
</Summarization>

<Document>
{document}
</Document>
<Question>
{question}
</Question>
<Summarization>
"""


# ═══════════════════════════════════════════════════════════════════════════
# Dataclass definitions
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Passage:
    """A single passage candidate."""
    id: str
    title: str
    text: str
    score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalInfo:
    """Temporal information parsed from the question."""
    time_relation: str = ""
    time_relation_type: str = ""
    years: list[int] = field(default_factory=list)
    months: list[int] = field(default_factory=list)
    implicit_condition: str | None = None
    normalized_question: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# MRAGRetriever main class
# ═══════════════════════════════════════════════════════════════════════════

class MRAGRetriever:
    """Standalone, reusable MRAG time-aware retriever.

    Preserves the core MRAG retrieval pipeline:

    1. **Initial retrieval** (BM25 via Pyserini or the nuggetindex store backend)
    2. **Keyword ranking**: compute keyword-weighted hit scores for candidate passages
    3. **Semantic ranking**: rerank with CrossEncoder / embedding models
    4. **QFS summarization**: the LLM generates question-focused summaries for the top-k passages
    5. **Sentence-level keyword ranking**: sentence-level keyword hits plus summaries participating in ranking
    6. **Temporal-semantic hybrid ranking**:
       ``final_score = hybrid_base * semantic + (1-hybrid_base) * semantic * temporal_coeff``

    Compatible with the ``retriever_factory`` configuration of
    ``scripts/run_benchmark_jstrag.py``.
    The nuggetindex store is injected through the :func:`create_mrag_retriever` factory function.

    Args:
        bm25_index_path: Path to the Pyserini Lucene index (standalone mode).
        reranker_model_name: Name of the semantic reranking model (HF model ID).
        reranker_type: Reranking model type: ``cross_encoder`` / ``bge`` / ``nv_embed`` / ``sfr`` / ``jina``.
        llm: Generative model instance (used for keyword extraction and QFS); these two steps are skipped when it is None.
        ctx_topk: Number of passages retained after keyword ranking (default 100).
        qfs_topk: Number of passages for which QFS summaries are generated (default 5).
        snt_topk: Number of sentences entering hybrid ranking (default 200).
        hybrid_score: Whether to enable the semantic-temporal hybrid score (default True).
        hybrid_base: Minimum retained fraction of the semantic score in the hybrid formula (default 0.0).
        snt_with_title: Whether to prepend the passage title to sentences (default True).
        store: NuggetStore instance (nuggetindex integration mode); standalone mode is used when None.
        device: Model device (e.g. "cuda:0"); selected automatically when None.
    """

    def __init__(
        self,
        *,
        bm25_index_path: str | None = None,
        reranker_model_name: str | None = None,
        reranker_type: str = "cross_encoder",
        llm: Any | None = None,
        ctx_topk: int = 100,
        qfs_topk: int = 5,
        snt_topk: int = 200,
        hybrid_score: bool = True,
        hybrid_base: float = 0.0,
        snt_with_title: bool = True,
        llm_temperature: float = 0.2,
        store: Any | None = None,
        device: str | None = None,
    ):
        if ctx_topk <= 0:
            raise ValueError(f"ctx_topk must be positive, got {ctx_topk}")
        if snt_topk <= 0:
            raise ValueError(f"snt_topk must be positive, got {snt_topk}")
        if not 0.0 <= hybrid_base <= 1.0:
            raise ValueError(f"hybrid_base must be in [0, 1], got {hybrid_base}")
        if not 0.0 <= llm_temperature <= 2.0:
            raise ValueError(
                f"llm_temperature must be in [0, 2], got {llm_temperature}")

        self._bm25_index_path = bm25_index_path
        self._reranker_model_name = reranker_model_name
        self._reranker_type = reranker_type
        self._llm = llm
        self._ctx_topk = ctx_topk
        self._qfs_topk = qfs_topk
        self._snt_topk = snt_topk
        self._hybrid_score = hybrid_score
        self._hybrid_base = hybrid_base
        self._snt_with_title = snt_with_title
        # Sampling temperature for keyword extraction/QFS: 0 means greedy decoding
        # (reproducible evaluation); the default 0.2 preserves the original behavior
        self._llm_temperature = llm_temperature
        self._store = store
        self._device = device

        # Lazily loaded model instances
        self._bm25_searcher = None
        self._reranker_model = None
        self._reranker_tokenizer = None
        self._tokenizer = SimpleTokenizer()

        # Keyword cache (keyed by normalized_question)
        self._keyword_cache: dict[str, tuple[list[list[str]], list[str]]] = {}

    # ───────────────────────────────────────────────────────────────────────
    # Lazy model loading
    # ───────────────────────────────────────────────────────────────────────

    def _load_bm25_searcher(self):
        """Load the Pyserini LuceneSearcher."""
        if self._bm25_searcher is None and self._bm25_index_path:
            from pyserini.search.lucene import LuceneSearcher
            self._bm25_searcher = LuceneSearcher(self._bm25_index_path)
        return self._bm25_searcher

    def _load_reranker(self):
        """Load the semantic reranking model."""
        if self._reranker_model is not None or self._reranker_model_name is None:
            return self._reranker_model

        if not _torch_available:
            logger.warning("torch 不可用,跳过语义重排,使用关键词分数替代")
            return None

        name = self._reranker_model_name
        try:
            if self._reranker_type == "bge" or "bge" in name.lower():
                from FlagEmbedding import FlagReranker
                self._reranker_model = FlagReranker(name, use_fp16=True)
            elif self._reranker_type == "nv_embed" or "nv" in name.lower():
                from transformers import AutoModel
                self._reranker_model = AutoModel.from_pretrained(
                    name, trust_remote_code=True, torch_dtype=torch.float16,
                )
            elif self._reranker_type == "jina" or "jina" in name.lower():
                from transformers import AutoModelForSequenceClassification
                self._reranker_model = AutoModelForSequenceClassification.from_pretrained(
                    name, torch_dtype="auto", trust_remote_code=True,
                )
                self._reranker_model.to("cuda" if self._device is None else self._device)
                self._reranker_model.eval()
            elif self._reranker_type == "sfr" or "sfr" in name.lower():
                from transformers import AutoTokenizer, AutoModel
                self._reranker_model = AutoModel.from_pretrained(
                    name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto",
                ).eval()
                self._reranker_tokenizer = AutoTokenizer.from_pretrained(name)
            else:
                # Default: CrossEncoder (MiniLM, TinyBERT, ELECTRA)
                from sentence_transformers import CrossEncoder
                self._reranker_model = CrossEncoder(name)
        except Exception as exc:
            logger.warning("加载语义重排模型 %s 失败: %s,使用关键词分数替代", name, exc)
            self._reranker_model = None

        return self._reranker_model

    # ───────────────────────────────────────────────────────────────────────
    # Initial retrieval module
    # ───────────────────────────────────────────────────────────────────────

    async def _store_bm25_search(self, query: str, top_k: int = 1000) -> list[Passage]:
        """Perform BM25 search through the nuggetindex store backend (integration mode).

        Obtain BM25 candidate nugget IDs from the store backend, then:
        1. Fetch the full passage text with ``aget_passages``
        2. Aggregate by source_id into passages, using the full text as the retrieval unit
        """
        if self._store is None:
            return []
        backend = getattr(self._store, "_backend_impl", None)
        if backend is None:
            return []

        # BM25 search (consistent with the original MRAG: no upfront temporal
        # filtering; time-awareness is handled at the sentence-level hybrid ranking stage)
        # FTS5 sanitization (stripping special symbols + OR join) happens here:
        # the upper layer passes in the raw query, to preserve temporal trigger
        # phrases such as "as of" for parse_temporal_question to detect.
        fts_query = sanitize_fts_query(query)
        if not fts_query:
            return []
        try:
            bm25_results = await backend.abm25_search(
                fts_query, top_k=top_k * 3,
            )
        except Exception as exc:
            logger.warning("abm25_search 失败: %s", exc)
            return []

        if not bm25_results:
            return []

        # 2. Fetch the nugget objects and extract source_id
        source_ids: set[str] = set()
        nugget_by_source: dict[str, list[tuple[str, float]]] = {}
        for nugget_id, score in bm25_results:
            try:
                nugget = await backend.aget(nugget_id)
                if nugget is None or not nugget.provenance:
                    continue
                sid = nugget.provenance[0].source_id
                source_ids.add(sid)
                nugget_by_source.setdefault(sid, []).append((nugget_id, score))
            except Exception:
                continue

        if not source_ids:
            return []

        # 3. Fetch the full passage text
        try:
            passages_text = await backend.aget_passages(source_ids)
        except Exception as exc:
            logger.warning("aget_passages 失败: %s,回退到 nugget fact text", exc)
            passages_text = {}

        # 4. Build Passage objects (aggregated by source_id)
        passages: list[Passage] = []
        for sid, nugget_hits in nugget_by_source.items():
            # Use the full passage text; fall back to the nugget fact text
            text = passages_text.get(sid, "")
            if not text:
                # Fallback: concatenate the nugget fact texts of this source
                fact_texts = []
                for nid, _ in nugget_hits:
                    try:
                        ng = await backend.aget(nid)
                        if ng and ng.fact.text:
                            fact_texts.append(ng.fact.text)
                    except Exception:
                        continue
                text = " ".join(fact_texts) if fact_texts else ""

            if not text:
                continue

            # Use the score of the highest-BM25-scoring nugget under this source
            # as the passage score
            best_score = max(s for _, s in nugget_hits)
            best_nugget_id = max(nugget_hits, key=lambda x: x[1])[0]
            # In store integration mode passages carry no title by default,
            # while the rerankers (rank_by_semantic / the temporal coefficient
            # convention of snt_with_title) model ``title + " " + text``;
            # pronoun-only passages and list-answer cells missing the owning
            # entity get their semantic score driven down to the order of -9.
            # Setting MRAG_TITLE_FROM_SOURCE_ID=true restores the title from the
            # TimeQA source_id (disabled by default to keep existing benchmarks comparable).
            title = ""
            if os.getenv("MRAG_TITLE_FROM_SOURCE_ID", "").strip().lower() in (
                "1", "true", "yes", "on",
            ):
                title = _title_from_source_id(sid)
            passages.append(Passage(
                id=sid, title=title, text=text,
                score=float(best_score), rank=0,
                metadata={
                    "nugget_ids": [nid for nid, _ in nugget_hits],
                    "best_nugget_id": best_nugget_id,
                    "nugget_scores": {nid: s for nid, s in nugget_hits},
                },
            ))

        passages.sort(key=lambda p: p.score, reverse=True)
        for i, p in enumerate(passages):
            p.rank = i + 1
        return passages[:top_k]

    def _bm25_search(self, query: str, top_k: int = 1000) -> list[Passage]:
        """Perform BM25 search with the Pyserini LuceneSearcher (standalone mode).

        Initialization arguments: only the Lucene index path is passed in.
        Calling method: ``searcher.search(question, k=topk)``.
        Returned results: each hit contains ``docid`` (format "id::title") and
        ``score``; the raw JSON is obtained via ``searcher.doc(id).raw()`` and
        parsed at ``['contents']`` to get the text.
        """
        import json
        searcher = self._load_bm25_searcher()
        if searcher is None:
            return []
        hits = searcher.search(query, k=top_k)
        sep = "::"
        passages = []
        for i in range(min(len(hits), top_k)):
            docid = hits[i].docid
            parts = docid.split(sep)
            short_id = parts[0]
            title = sep.join(parts[1:]) if len(parts) > 1 else ""
            try:
                raw = searcher.doc(docid).raw()
                text = json.loads(raw)["contents"]
            except Exception:
                text = ""
            passages.append(Passage(
                id=short_id, title=title, text=text,
                score=float(hits[i].score), rank=i + 1,
            ))
        return passages

    async def _initial_retrieval(
        self,
        query: str,
        candidates: list[Passage] | None = None,
        top_k: int = 1000,
    ) -> list[Passage]:
        """Initial retrieval: BM25 (integration mode first, standalone mode as fallback).

        - If ``candidates`` is non-empty (preloaded mode), use it directly.
        - Otherwise, prefer the backend BM25 of the nuggetindex store (integration mode).
        - Otherwise, use the Pyserini LuceneSearcher (standalone mode).
        """
        if candidates is not None:
            return candidates[:top_k]

        passages: list[Passage] = []

        # Integration mode: fetch BM25 candidates from the nuggetindex store backend
        if self._store is not None:
            passages = await self._store_bm25_search(query, top_k=top_k)

        # Standalone mode: use the Pyserini LuceneSearcher
        if not passages and self._bm25_index_path:
            passages = self._bm25_search(query, top_k=top_k)

        return passages[:top_k]

    # ───────────────────────────────────────────────────────────────────────
    # Temporal-aware reranking module
    # ───────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_time_relation(
        question: str,
    ) -> tuple[str, str, str] | None:
        """Auto-detect from the question text when the caller does not provide ``time_relation`` explicitly.

        Scanning strategy: multi-word triggers first (in descending length
        order), word-boundary matching, case-insensitive; a trigger is accepted
        only when **the text after it can be parsed into at least one four-digit year**,
        to avoid false hits from non-temporal usages such as "a book by John".

        Returns:
            ``(trigger_lower, date_suffix, prefix)``: the trigger (lowercase,
            for classification), the date text after the trigger, and the
            question text before the trigger; returns None on no match.
        """
        lower = question.lower()
        for trigger in sorted(set(_TIME_RELATION_TRIGGERS), key=len, reverse=True):
            pattern = r"(?<![A-Za-z])" + re.escape(trigger) + r"(?![A-Za-z])"
            for m in re.finditer(pattern, lower):
                date_suffix = question[m.end():]
                if year_identifier(date_suffix):
                    return trigger, date_suffix, question[:m.start()]
        return None

    def parse_temporal_question(
        self, question: str, time_relation: str = "",
    ) -> TemporalInfo:
        """Parse temporal information in the question.

        When ``time_relation`` is empty, trigger words are auto-detected from
        the question (as of / before / after / since / until / from / by),
        consistent with the parsing branch for the ``time_relation`` field
        included in the original MRAG dataset.

        Returns:
            TemporalInfo: containing time_relation_type, years, months,
            implicit_condition, normalized_question.
        """
        time_relation = time_relation.strip()
        time_relation_type = ""
        years: list[int] = []
        months: list[int] = []
        no_time_question = question
        date = ""

        if time_relation and time_relation in question:
            # Explicitly provided (field path of the original MRAG dataset)
            parts = question.split(time_relation)
            no_time_question = time_relation.join(parts[:-1])
            date = parts[-1]
            time_relation = time_relation.lower()
        elif not time_relation:
            # Integration path (nuggetindex aretrieve has only the query text): auto-detect
            detected = self._detect_time_relation(question)
            if detected is not None:
                time_relation, date, no_time_question = detected

        if time_relation and date:
            years = year_identifier(date)
            if len(years) > 2:
                years = [min(years), max(years)]

            if len(years) > 1:
                time_relation_type = "between"
            elif time_relation in ["before", "as of", "by", "until"]:
                time_relation_type = "before"
            elif time_relation in ["from", "since", "after"]:
                time_relation_type = "after"
            else:
                time_relation_type = "other"

            # Month parsing (currently not used in final ranking, but retained for extensibility)
            def _append_month(month_str: str):
                m = find_month(month_str)
                months.append(m if m else 0)

            if time_relation_type == "between":
                delimiters = ["and", "to", "until"]
                d_index = [d in date for d in delimiters]
                if any(d_index):
                    delimiter = delimiters[d_index.index(True)]
                    tmp = date.split(delimiter)
                    for w in tmp:
                        _append_month(w.strip())
                else:
                    months = [0, 0]
            else:
                _append_month(date.strip())

        # Clean up trailing whitespace and dangling opening parentheses left
        # after the trigger is excised (e.g. "Who was CEO? (as of 2018)" →
        # the trailing "(" of the prefix).
        if time_relation:
            no_time_question = re.sub(r"[\s(]+$", "", no_time_question)

        normalized_question, implicit_condition = remove_implicit_condition(no_time_question)
        # A fronted temporal adverbial (e.g. "As of 2018, who was CEO?") leaves
        # an empty string after time removal; in that case fall back to the
        # original question to avoid feeding an empty query to semantic ranking.
        if not normalized_question:
            normalized_question = question.rstrip()
        if normalized_question and normalized_question[-1] in ".?!":
            normalized_question = normalized_question[:-1]

        return TemporalInfo(
            time_relation=time_relation,
            time_relation_type=time_relation_type,
            years=years,
            months=months,
            implicit_condition=implicit_condition,
            normalized_question=normalized_question,
        )

    async def aextract_keywords(self, normalized_question: str) -> tuple[list[list[str]], list[str]]:
        """Extract and expand keywords from the question.

        If the LLM is available, it is used to generate keywords; otherwise
        simple non-stopword extraction is used.
        Results are cached by normalized_question.
        """
        if normalized_question in self._keyword_cache:
            return self._keyword_cache[normalized_question]

        # Normalize special question prefixes
        q = normalized_question
        if q.startswith("How many times"):
            q = q.replace("How many times", "When")
        elif q.startswith("How many"):
            q = q.replace("How many", "What")

        keyword_list = await self._allm_extract_keywords(q)
        if not keyword_list:
            keyword_list = self._simple_extract_keywords(q)

        # Post-processing: filter out low-information words and ensure keywords
        # are substrings of the original question
        revised = []
        for kw in keyword_list:
            if kw in EXCL:
                continue
            while kw and kw.lower() not in q.lower():
                kw = " ".join(kw.split()[:-1])
            if kw and kw.lower() in q.lower():
                revised.append(kw)
        revised = list(set(revised))

        expanded, types = expand_keywords(revised, q, verbose=False)
        self._keyword_cache[normalized_question] = (expanded, types)
        return expanded, types

    async def _allm_extract_keywords(self, question: str) -> list[str]:
        """Extract keywords using the LLM."""
        if self._llm is None:
            return []
        prompt = get_keyword_prompt(question)
        try:
            response = (await self._acall_llm([prompt], max_tokens=100))[0]
            start = response.find("[")
            end = response.find("]")
            if start == -1 or end == -1:
                return []
            tmp = response[start : end + 1]
            return list(eval(tmp))  # noqa: S307 (LLM output is a Python list literal)
        except Exception:
            logger.warning("LLM 关键词抽取失败,回退到简单关键词提取", exc_info=True)
            return []

    def _simple_extract_keywords(self, question: str) -> list[str]:
        """Simple keyword extraction (fallback when no LLM is available)."""
        tokens = word_tokenize(question)
        tagged = pos_tag(tokens)
        keywords = []
        for word, tag in tagged:
            if word.lower() not in EXCL and not tag.startswith("DT") and not tag.startswith("IN") and not tag.startswith("PRP"):
                keywords.append(word)
        return keywords

    async def _acall_llm(self, prompts: list[str], max_tokens: int = 100) -> list[str]:
        """Call the LLM asynchronously to generate text.

        The interface of ``self._llm`` is identified in the following order
        (duck typing):

        1. The framework :class:`~tcrag.llm.base.BaseLLM` (with ``agenerate``,
           e.g. the OpenAI / Ollama clients) -- must be checked first:
           ``BaseLLM`` also exposes a synchronous ``generate()`` wrapper, so
           checking it later would misidentify it as vLLM.
        2. The vLLM offline engine (synchronous
           ``generate(prompts, SamplingParams)``), wrapped with
           ``asyncio.to_thread`` to avoid blocking the event loop.
        3. A bare OpenAI-compatible client (``chat.completions.create``).
        """
        if self._llm is None:
            return [""] * len(prompts)

        if hasattr(self._llm, "agenerate"):
            # Unified framework LLM interface: agenerate(prompt, ...) -> LLMResponse
            async def _one(prompt: str) -> str:
                try:
                    resp = await self._llm.agenerate(
                        prompt, temperature=self._llm_temperature,
                        max_tokens=max_tokens,
                    )
                    return (getattr(resp, "content", "") or "").strip()
                except Exception:
                    logger.warning("MRAG 框架 LLM 单次生成失败,返回空字符串", exc_info=True)
                    return ""

            responses = list(await asyncio.gather(*(_one(p) for p in prompts)))
        elif hasattr(self._llm, "generate"):
            # vLLM offline engine (synchronous and blocking; run in a thread pool)
            try:
                from vllm import SamplingParams
                sampling_params = SamplingParams(
                    temperature=self._llm_temperature, top_p=0.95,
                    max_tokens=max_tokens, seed=0,
                )
                outputs = await asyncio.to_thread(
                    self._llm.generate, prompts, sampling_params,
                )
                responses = [o.outputs[0].text for o in outputs]
            except Exception:
                logger.warning("vLLM generate 失败,全部返回空字符串", exc_info=True)
                responses = [""] * len(prompts)
        else:
            # OpenAI-compatible interface (bare synchronous openai.OpenAI client)
            responses = []
            model = getattr(self._llm, "model", "gpt-4o-mini")
            for prompt in prompts:
                try:
                    completion = self._llm.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=self._llm_temperature,
                    )
                    responses.append(completion.choices[0].message.content.strip())
                except Exception:
                    logger.warning("OpenAI 兼容 LLM 单次调用失败,返回空字符串", exc_info=True)
                    responses.append("")

        # Strip stop markers
        for stopper in ["</Keywords>", "</Summarization>", "</Answer>", "</Info>"]:
            responses = [r.split(stopper)[0] if stopper in r else r for r in responses]
        return responses

    # ───────────────────────────────────────────────────────────────────────
    # Reranking step 1: passage keyword ranking
    # ───────────────────────────────────────────────────────────────────────

    def rank_by_keywords(
        self,
        candidates: list[Passage],
        expanded_keywords: list[list[str]],
        keyword_types: list[str],
    ) -> list[Passage]:
        """Passage keyword ranking.

        For each passage, compute the sum of keyword weights hit in
        ``title + text``, sort by total score in descending order, and keep the
        top ``ctx_topk``.
        """
        scored = []
        for ctx in candidates:
            text = ctx.title + " " + ctx.text
            kw_score = count_keyword_scores(text, expanded_keywords, keyword_types)
            scored.append((ctx, kw_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [tp[0] for tp in scored[: self._ctx_topk]]

    # ───────────────────────────────────────────────────────────────────────
    # Reranking step 2: passage semantic ranking
    # ───────────────────────────────────────────────────────────────────────

    def rank_by_semantic(
        self,
        candidates: list[Passage],
        query: str,
        normalized: bool = False,
    ) -> list[Passage]:
        """Passage semantic ranking.

        Scores ``[query, title+text]`` pairs using a semantic model
        (CrossEncoder / BGE / NV-Embed).
        Uses normalized_question (with temporal phrases removed) when
        ``normalized=True``.
        """
        model = self._load_reranker()
        if model is None:
            return candidates

        search_query = query
        model_inputs = [[search_query, ctx.title + " " + ctx.text] for ctx in candidates]

        scores = self._compute_semantic_scores(model_inputs, search_query, candidates)

        for i, ctx in enumerate(candidates):
            ctx.score = float(scores[i])
        return sorted(candidates, key=lambda x: x.score, reverse=True)

    def _compute_semantic_scores(
        self, model_inputs: list[list[str]], query: str, candidates: list[Passage],
    ) -> list[float]:
        """Compute semantic scores according to the reranker model type."""
        name = self._reranker_model_name or ""
        model = self._reranker_model

        if "nv" in name.lower() and self._reranker_type != "bge":
            return self._nv_embed_scores(query, [x[1] for x in model_inputs])
        elif "sfr" in name.lower():
            return self._sfr_scores(query, [x[1] for x in model_inputs])
        elif "bge" in name.lower() or self._reranker_type == "bge":
            return list(model.compute_score(model_inputs))
        elif "jina" in name.lower():
            return list(model.predict(model_inputs))
        else:
            # CrossEncoder (MiniLM, TinyBERT, ELECTRA)
            return list(model.predict(model_inputs))

    def _nv_embed_scores(self, query: str, passages: list[str]) -> list[float]:
        """NV-Embed semantic scoring."""
        task = "Given a question, retrieve passages that answer the question"
        query_prefix = f"Instruct: {task}\nQuery: "
        max_length = 512
        batch_size = 4

        query_emb = self._reranker_model.encode(
            [query], instruction=query_prefix, max_length=max_length,
        )
        query_emb = torch.tensor(query_emb)
        query_emb = F.normalize(query_emb, p=2, dim=1)

        all_scores = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            passage_emb = self._reranker_model.encode(batch, instruction="", max_length=max_length)
            passage_emb = torch.tensor(passage_emb)
            passage_emb = F.normalize(passage_emb, p=2, dim=1)
            scores = (query_emb @ passage_emb.T).view(-1)
            all_scores.extend(scores.tolist())
        return all_scores

    def _sfr_scores(self, query: str, passages: list[str]) -> list[float]:
        """SFR embedding semantic scoring."""
        task = "Given a web search query, retrieve relevant passages that answer the query"
        queries = [f"Instruct: {task}\nQuery: {query}"]
        max_length = 512
        input_texts = queries + passages

        batch_dict = self._reranker_tokenizer(
            input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt",
        ).to(self._reranker_model.device)

        with torch.no_grad():
            outputs = self._reranker_model(**batch_dict)

        embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        query_emb = embeddings[0]
        passage_emb = embeddings[1:]
        scores = (query_emb @ passage_emb.T).tolist()
        return scores

    @staticmethod
    def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Last-token pooling for SFR."""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

    # ───────────────────────────────────────────────────────────────────────
    # Reranking step 3: QFS summary generation
    # ───────────────────────────────────────────────────────────────────────

    async def generate_qfs_summaries(
        self, candidates: list[Passage], query: str, top_k: int | None = None,
    ) -> list[str | None]:
        """Generate QFS summaries for the top-k passages.

        Uses the LLM to produce question-focused summaries that retain key
        dates; returns None when the document is irrelevant.
        """
        k = top_k or self._qfs_topk
        if self._llm is None or k <= 0:
            return [None] * min(len(candidates), k)

        prompts = []
        for ctx in candidates[:k]:
            doc = ctx.title + " | " + ctx.text
            prompts.append(get_qfs_prompt(doc, query))

        responses = await self._acall_llm(prompts, max_tokens=200)
        summaries = []
        for resp in responses:
            if "None" in resp:
                summaries.append(None)
            else:
                summaries.append(resp.strip() if resp else None)
        return summaries

    # ───────────────────────────────────────────────────────────────────────
    # Reranking step 4: sentence keyword ranking
    # ───────────────────────────────────────────────────────────────────────

    def rank_sentences_by_keywords(
        self,
        candidates: list[Passage],
        summaries: list[str | None],
        expanded_keywords: list[list[str]],
        keyword_types: list[str],
    ) -> tuple[list[tuple[str, str, float]], dict[str, Passage]]:
        """Sentence keyword ranking.

        1. Split each passage into sentences with sent_tokenize()
        2. Optionally prepend the title to each sentence
        3. Treat the QFS summary as an additional sentence when it is not None
        4. Compute keyword scores for all sentences and rank them globally

        Returns:
            (sentence_tuples, get_ctx_by_id):
            - sentence_tuples: [(passage_id, sentence, kw_score), ...] in descending score order
            - get_ctx_by_id: {passage_id: Passage}
        """
        get_ctx_by_id: dict[str, Passage] = {}
        sentence_tuples: list[tuple[str, str, float]] = []

        for idx, ctx in enumerate(candidates):
            get_ctx_by_id[ctx.id] = ctx
            snts = sent_tokenize(ctx.text)
            if self._snt_with_title:
                snts = [ctx.title + " " + s for s in snts]

            summary = summaries[idx] if idx < len(summaries) else None
            if summary:
                snts.append(summary)

            for snt in snts:
                snt = snt.strip()
                text = ctx.title + " " + snt
                kw_score = count_keyword_scores(text, expanded_keywords, keyword_types)
                sentence_tuples.append((ctx.id, snt, kw_score))

        sentence_tuples.sort(key=lambda x: x[2], reverse=True)
        return sentence_tuples, get_ctx_by_id

    # ───────────────────────────────────────────────────────────────────────
    # Reranking step 5: temporal-semantic hybrid ranking
    # ───────────────────────────────────────────────────────────────────────

    def hybrid_rank_sentences(
        self,
        sentence_tuples: list[tuple[str, str, float]],
        query: str,
        normalized_query: str,
        temporal_info: TemporalInfo,
        candidates: list[Passage],
    ) -> list[tuple[str, str, float]]:
        """Sentence semantic-temporal hybrid ranking.

        Combination formula:
        ``final_score = hybrid_base * semantic + (1-hybrid_base) * semantic * temporal_coeff``

        By default ``hybrid_base=0``, equivalent to
        ``final_score = semantic * temporal_coeff``.
        For questions without a year or of the other type, the semantic score
        is used directly.
        """
        # Keep the top snt_topk sentences; the rest retain their original order
        snt_topk = min(len(sentence_tuples), self._snt_topk)
        sentence_tuples_unchange = sentence_tuples[snt_topk:]
        sentence_tuples = sentence_tuples[:snt_topk]

        years = temporal_info.years
        time_relation_type = temporal_info.time_relation_type
        implicit_condition = temporal_info.implicit_condition

        # Decide whether to use normalized_question or the original question
        use_hybrid = (
            len(years) > 0
            and time_relation_type != "other"
            and self._hybrid_score
        )
        search_query = normalized_query if use_hybrid else query

        # Compute semantic scores
        model = self._load_reranker()
        if model is not None and sentence_tuples:
            model_inputs = [[search_query, tp[1]] for tp in sentence_tuples]
            name = self._reranker_model_name or ""
            if "nv" in name.lower() and self._reranker_type != "bge":
                semantic_scores = self._nv_embed_scores(search_query, [x[1] for x in model_inputs])
            elif "bge" in name.lower() or self._reranker_type == "bge":
                semantic_scores = list(model.compute_score(model_inputs))
            else:
                semantic_scores = list(model.predict(model_inputs))
            semantic_scores = [float(s) for s in semantic_scores]
        else:
            # When no model is available, use keyword scores as a substitute for semantic scores
            semantic_scores = [tp[2] for tp in sentence_tuples]

        # Compute temporal coefficients and combine
        if use_hybrid:
            spline = get_spline_function(time_relation_type, implicit_condition, years)
            temporal_coeffs = get_temporal_coeffs(
                years, sentence_tuples, time_relation_type, implicit_condition, spline,
            )
            final_scores = [
                self._hybrid_base * score + (1 - self._hybrid_base) * score * coeff
                for score, coeff in zip(semantic_scores, temporal_coeffs)
            ]
        else:
            final_scores = semantic_scores

        # Assemble the final sentence tuples
        sentence_tuples = [
            (tp[0], tp[1], score) for score, tp in zip(final_scores, sentence_tuples)
        ]
        sentence_tuples.sort(key=lambda x: x[2], reverse=True)
        sentence_tuples += sentence_tuples_unchange
        return sentence_tuples

    # ───────────────────────────────────────────────────────────────────────
    # Main retrieval interface
    # ───────────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        *,
        time_relation: str = "",
        candidates: list[Passage] | None = None,
        top_k: int = 10,
    ) -> list[Passage]:
        """Run the full MRAG retrieval pipeline (standalone mode).

        Args:
            query: Query text.
            time_relation: Temporal relation word (e.g. "after", "before"); auto-detected when empty.
            candidates: Preloaded candidate passages; initial retrieval is performed when None.
            top_k: Number of passages to return.

        Returns:
            A list of passages ordered by the final MRAG ranking.
        """
        # Step 0: initial retrieval
        initial_candidates = await self._initial_retrieval(query, candidates)
        if not initial_candidates:
            return []

        # Step 1: temporal information preprocessing
        temporal_info = self.parse_temporal_question(query, time_relation)
        normalized_question = temporal_info.normalized_question or query

        # Step 2: keyword extraction
        # Extract the meaningful words from the query
        # expanded keywords: based on the original words in the query, associate
        # related words and add them to the keyword set as well
        expanded_keywords, keyword_types = await self.aextract_keywords(normalized_question)

        # Step 3: passage keyword ranking → top ctx_topk
        # Count the query keywords in each candidate and sort by the count
        ctx_kw_ranked = self.rank_by_keywords(
            initial_candidates, expanded_keywords, keyword_types,
        )

        # Step 4: passage semantic ranking → rerank the top ctx_topk
        ctx_semantic_ranked = self.rank_by_semantic(
            ctx_kw_ranked, normalized_question, normalized=True,
        )

        # Step 5: QFS summary generation
        summaries = await self.generate_qfs_summaries(
            ctx_semantic_ranked, normalized_question, top_k=self._qfs_topk,
        )
        for i, s in enumerate(summaries):
            if i < len(ctx_semantic_ranked):
                ctx_semantic_ranked[i].metadata["qfs_summary"] = s

        # Step 6: sentence keyword ranking
        sentence_tuples, get_ctx_by_id = self.rank_sentences_by_keywords(
            ctx_semantic_ranked, summaries, expanded_keywords, keyword_types,
        )

        # Step 7: temporal-semantic hybrid ranking
        final_sentence_tuples = self.hybrid_rank_sentences(
            sentence_tuples, query, normalized_question, temporal_info, ctx_semantic_ranked,
        )

        # Derive passage ranks from sentence ranks
        latest_ctxs: list[Passage] = []
        id_included: set[str] = set()
        for ctx_id, snt, score in final_sentence_tuples:
            if ctx_id not in id_included:
                id_included.add(ctx_id)
                ctx = get_ctx_by_id.get(ctx_id)
                if ctx:
                    ctx.score = score
                    latest_ctxs.append(ctx)

        return latest_ctxs[:top_k]

    async def aretrieve(
        self,
        query: str,
        *,
        query_time: datetime | None = None,
        view: str = "active",
        top_k: int = 10,
        fusion: str = "rrf",
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        """NuggetIndex integration interface (compatible with ``retriever_factory``).

        The signature matches nuggetindex ``Retriever.aretrieve``;
        returns a list of ``RetrievalResult``.
        """
        # Fetch initial candidates from the store backend
        candidates = await self._initial_retrieval(query, top_k=max(top_k * 10, 1000))

        if not candidates:
            return []

        # Run the MRAG pipeline
        ranked_passages = await self.retrieve(
            query, candidates=candidates, top_k=top_k,
        )

        # Convert to RetrievalResult
        return await self._to_retrieval_results(ranked_passages, top_k)

    async def _to_retrieval_results(
        self, passages: list[Passage], top_k: int,
    ) -> list[Any]:
        """Convert a list of Passages into nuggetindex RetrievalResults.

        Prefer fetching the original nuggets from the store backend (preserving
        provenance/validity); construct new Nuggets when they cannot be found.
        """
        try:
            from nuggetindex.retrieve.retriever import RetrievalResult
            from nuggetindex.core.models import (
                Nugget, FactTriple, ProvenanceRecord, EpistemicState,
                ValidityInterval,
            )
            from nuggetindex.core.enums import NuggetKind
        except ImportError:
            logger.warning("nuggetindex 不可用,返回简单 dict")
            return [
                {"doc_id": p.id, "text": p.text, "score": p.score, "rank": i + 1}
                for i, p in enumerate(passages[:top_k])
            ]

        backend = getattr(self._store, "_backend_impl", None) if self._store else None

        results = []
        for i, p in enumerate(passages[:top_k]):
            nugget = None

            # Try to fetch the original nuggets from the store backend (preserving provenance/validity)
            if backend is not None:
                nugget_ids = p.metadata.get("nugget_ids", [])
                best_nid = p.metadata.get("best_nugget_id")
                # Prefer best_nugget_id, then try the first one in the list
                candidates_ids = [best_nid] + [nid for nid in nugget_ids if nid != best_nid]
                for nid in candidates_ids:
                    if not nid:
                        continue
                    try:
                        ng = await backend.aget(nid)
                        if ng is not None:
                            nugget = ng
                            break
                    except Exception:
                        continue

            if nugget is None:
                # Fallback: construct a new Nugget
                nugget = Nugget.new(
                    kind=NuggetKind.SEMANTIC_FACT,
                    fact=FactTriple(
                        subject=p.title or p.id, predicate="retrieved", object=p.text,
                        text=p.text,
                    ),
                    validity=ValidityInterval.unknown(),
                    epistemic=EpistemicState(confidence=1.0),
                    provenance=(ProvenanceRecord(
                        source_id=p.id, evidence_span=p.text, char_start=0, char_end=0,
                    ),),
                    extraction_confidence=1.0,
                )

            results.append(RetrievalResult(
                nugget=nugget, score=p.score, rank=i + 1,
                sparse_score=p.metadata.get("nugget_scores", {}).get(
                    p.metadata.get("best_nugget_id", ""), p.score
                ),
                dense_score=None,
            ))
        return results


# ═══════════════════════════════════════════════════════════════════════════
# NuggetIndex integration factory function
# ═══════════════════════════════════════════════════════════════════════════

def create_mrag_retriever(store: Any, *, llm: Any | None = None) -> MRAGRetriever:
    """NuggetIndex integration factory function.

    Compatible with the ``retriever_factory`` configuration of
    ``scripts/run_benchmark_jstrag.py``:

    .. code-block:: yaml

        systems:
          nuggetindex:
            retriever_factory: "tcrag.retrievers.metriever:create_mrag_retriever"

    Signature: ``(store: NuggetStore, *, llm=None) -> MRAGRetriever``
    The returned MRAGRetriever implements
    ``async aretrieve(query, *, query_time, view, top_k, fusion, filters)``.

    Args:
        store: NuggetStore instance (nuggetindex integration mode).
        llm: Generative model instance, passed in by the external run_benchmark
            script (shared with the QA stage); used for keyword extraction and
            QFS summarization; these two steps are skipped when None.

    Environment variables (optional):
      - ``BM25_INDEX_PATH``: Path to the Pyserini Lucene index (standalone BM25 retrieval)
      - ``RERANKER_MODEL``: HF name of the semantic reranking model (default ``nvidia/NV-Embed-v2``,
        following the default stage2 reranking model metriever_model=nv2 in MRAG metriever.py)
      - ``RERANKER_TYPE``: Reranking model type (cross_encoder/bge/nv_embed/sfr/jina,
        default nv_embed)
      - ``MRAG_CTX_TOPK``: Number of passages retained after keyword ranking (default 100)
      - ``MRAG_SNT_TOPK``: Number of sentences entering hybrid ranking (default 200)
      - ``MRAG_QFS_TOPK``: Number of passages for QFS summarization (default 5)
      - ``HYBRID_BASE``: Minimum retained fraction of the semantic score in the hybrid formula (default 0.0)
      - ``MRAG_LLM_TEMPERATURE``: Sampling temperature for keyword extraction/QFS (default 0.2);
        can be set to 0 (greedy decoding) for reproducible evaluation
    """
    bm25_index = os.getenv("BM25_INDEX_PATH")
    # The default stage2 reranking model follows MRAG metriever.py:
    # metriever_model defaults to nv2 → nvidia/NV-Embed-v2, corresponding to
    # reranker_type=nv_embed.
    reranker_model = os.getenv("RERANKER_MODEL", "nvidia/NV-Embed-v2")
    reranker_type = os.getenv("RERANKER_TYPE", "nv_embed")
    ctx_topk = int(os.getenv("MRAG_CTX_TOPK", "100"))
    snt_topk = int(os.getenv("MRAG_SNT_TOPK", "200"))
    qfs_topk = int(os.getenv("MRAG_QFS_TOPK", "5"))
    hybrid_base = float(os.getenv("HYBRID_BASE", "0.0"))
    llm_temperature = float(os.getenv("MRAG_LLM_TEMPERATURE", "0.2"))

    logger.info(
        "创建 MRAGRetriever: bm25_index=%s, reranker=%s(%s), ctx_topk=%d, snt_topk=%d, qfs_topk=%d, hybrid_base=%.2f, llm_temperature=%.2f",
        bm25_index or "<store-backend>", reranker_model or "<none>", reranker_type,
        ctx_topk, snt_topk, qfs_topk, hybrid_base, llm_temperature,
    )

    return MRAGRetriever(
        bm25_index_path=bm25_index,
        reranker_model_name=reranker_model,
        reranker_type=reranker_type,
        llm=llm,
        ctx_topk=ctx_topk,
        snt_topk=snt_topk,
        qfs_topk=qfs_topk,
        hybrid_base=hybrid_base,
        llm_temperature=llm_temperature,
        store=store,
    )
