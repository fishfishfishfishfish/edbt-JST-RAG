"""MRAG 时间感知检索器:独立、可复用的 Retriever 组件。

重构自原始 MRAG metriever.py,保留 MRAG 的核心检索管道:

1. **初始检索**:BM25(通过 nuggetindex store 后端或 Pyserini LuceneSearcher)
2. **关键词排序**:对候选 passage 计算关键词加权命中分数,保留 top-ctx_topk
3. **语义排序**:使用 CrossEncoder / BGE / NV-Embed 等模型重排
4. **QFS 摘要**:LLM 为 top-k passage 生成问题聚焦摘要(保留关键日期)
5. **句子关键词排序**:句子级关键词命中 + 摘要参与排序
6. **时间-语义混合排序**:
   ``final_score = hybrid_base * semantic + (1-hybrid_base) * semantic * temporal_coeff``
   默认 ``hybrid_base=0``,等价于 ``final_score = semantic * temporal_coeff``

时间感知核心:
- ``year_identifier(text)``:正则提取年份,先展开 "1990-93"/"1990-1995" 范围
- ``remove_implicit_condition(q)``:检测 latest/last/first/earliest 等隐式条件
- ``get_spline_function(...)``:线性插值,窗口 [0.6, 1.0],span=50 年
- ``get_temporal_coeffs(...)``:按时间关系类型(between/before/after)筛选合规年份

兼容 ``scripts/run_benchmark_jstrag.py`` 的 ``retriever_factory`` 配置:

.. code-block:: yaml

    systems:
      jstrag:
        retriever_factory: "tcrag.retrievers.metriever:create_mrag_retriever"

工厂函数 ``create_mrag_retriever(store, *, llm=None)`` 返回的 MRAGRetriever 实现了
``async aretrieve(query, *, query_time, view, top_k, fusion, filters) -> list[RetrievalResult]``。

通过环境变量 ``MRAG_RERANKER_TYPE``, ``MRAG_RERANKER_MODEL`` 选择 rerank模型。
MRAG_RERANKER_TYPE="cross_encoder"
MRAG_RERANKER_MODEL="/home/xinyuchen/TCRag/models/ms-marco-MiniLM-L6-v2"
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

#: source_id 末尾的段落序号(TimeQA:``.../wiki/Besart_Berisha_18``)。
_PARA_IDX_SUFFIX = re.compile(r"_\d+$")


def _title_from_source_id(source_id: str) -> str:
    """TimeQA 口径 ``source_id`` → 文档标题人类形式。

    ``timeqa_/wiki/Besart_Berisha_18`` → ``Besart Berisha``。
    无可取段时返回 ``""``(调用方保持空标题)。
    """
    tail = source_id.rsplit("/", 1)[-1]
    if not tail:
        return ""
    tail = _PARA_IDX_SUFFIX.sub("", tail)
    return tail.replace("_", " ").strip()


# ── 可选依赖(缺失时降级)──────────────────────────────────────────────────

# NLTK 依赖分两层(常见坑:pip 装了 nltk 但语料数据缺失,运行时抛 LookupError):
#   1) pip 包:
#        pip install nltk
#   2) 语料数据 —— nltk>=3.9 需要 punkt_tab / averaged_perceptron_tagger_eng;
#      在线下载(`python -m nltk.downloader ...`)走 raw.githubusercontent.com,
#      弱网常卡超时或报 SSL EOF。推荐手工下载 zip 后解压到指定目录:
#
#      数据包下载地址
#      git clone https://github.com/nltk/nltk_data.git
#      将 nltk_data/packages改为nltk_data, 放在~/nltk_data

#      若 nltk_data 目录 group 可写会触发 "non-private download directory" warning:
#        chmod 755 ~/nltk_data
#      数据也可放在 conda 环境内(免污染 home):~/miniconda3/envs/<env>/nltk_data/;
#      或设置环境变量 NLTK_DATA=/your/dir 自定义搜索路径。
#       通过python -c "import nltk; print(nltk.data.path)" 查看 nltk_data 目录路径
#
#      验证:
#        python -c "from nltk.tokenize import sent_tokenize; \
#           print(sent_tokenize('A test. Another one.'))"
#   说明:旧版 punkt / averaged_perceptron_tagger 在 nltk>=3.9 已不被使用,无需下载。
# 任一缺失时降级为 regex 分句/分词、全 NN 词性、空词形还原:
#   MRAG 管道仍可运行,但关键词词形变体扩展关闭,可能降低召回。
_nltk_data_available = False
try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    from nltk.tokenize import word_tokenize as _nltk_word_tokenize
    from nltk.tag import pos_tag as _nltk_pos_tag
    from nltk.stem import WordNetLemmatizer as _NLTKWordNetLemmatizer
    _nltk_sent_tokenize("Test sentence.")  # 触发数据加载检查
    _nltk_data_available = True
except Exception as _nltk_exc:  # ImportError(nltk 包缺失)或 LookupError(语料缺失)
    logger.debug("NLTK 语料不可用,降级为 regex 分词: %s", _nltk_exc)

if _nltk_data_available:
    sent_tokenize = _nltk_sent_tokenize
    word_tokenize = _nltk_word_tokenize
    pos_tag = _nltk_pos_tag
    WordNetLemmatizer = _NLTKWordNetLemmatizer
else:
    def sent_tokenize(text: str) -> list[str]:
        """Regex 分句(NLTK 数据缺失时的降级实现)。"""
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sents if s]

    def word_tokenize(text: str) -> list[str]:
        """Regex 分词(NLTK 数据缺失时的降级实现)。"""
        return re.findall(r"\b[\w'-]+\b", text)

    def pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
        """降级 POS tagger:全部标记为 NN。"""
        return [(t, "NN") for t in tokens]

    class WordNetLemmatizer:  # type: ignore[no-redef]
        """降级 lemmatizer:返回原词不做词形还原。"""
        def lemmatize(self, word: str, pos: str = "n") -> str:
            return word

# pattern.en 用于词形变体扩展;缺失时降级为空实现
try:
    from pattern.en import lexeme as _pattern_lexeme
    _pattern_available = True
except Exception:
    _pattern_available = False
    _pattern_lexeme = None

# torch / transformers(语义模型用,可选)
try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    _torch_available = True
except ImportError:
    _torch_available = False

# regex 库(Unicode-aware tokenizer 用,可选)
try:
    import regex as _re_mod
    import unicodedata as _ud
    _regex_available = True
except ImportError:
    _regex_available = False
    import re as _re_mod  # noqa: F811


# ═══════════════════════════════════════════════════════════════════════════
# 常量与配置
# ═══════════════════════════════════════════════════════════════════════════

# 关键词低信息词排除列表
EXCL = [
    "time", "years", "for", "new", "recent", "current",
    "whom", "who", "out", "place", "not",
]

# 关键词命中权重
KEYWORD_WEIGHTS = {
    "special": 1.0,
    "superlative": 0.7,
    "numeric": 0.5,
    "general": 0.4,
    "adjective": 0.4,
}

# 数字映射
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

# 月份映射
MONTH_TO_NUMBER = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}
SHORT_MONTH_TO_NUMBER = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# 时间系数曲线参数
TEMPORAL_LOW = 0.6
TEMPORAL_SPAN = 50
TEMPORAL_FALLBACK = 0.5

# parse_temporal_question 自动检测时扫描的时间关系触发词。
# 分类与原始 MRAG 一致:before 类 / after 类;多词触发词优先匹配。
_TIME_RELATION_TRIGGERS: tuple[str, ...] = (
    "as of", "before", "after", "since", "until", "from", "by",
)


# ═══════════════════════════════════════════════════════════════════════════
# DPR-style SimpleTokenizer & has_answer(移植自 contriever/src/evaluation.py)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleTokenizer:
    """DPR-style Unicode-aware tokenizer。

    使用 ``regex`` 库的 Unicode 属性转义(``\\p{L}`` 等)做 token 切分;
    ``regex`` 不可用时回退到标准 ``re`` 的 ``\\w`` / ``\\S`` 模式。
    """

    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"
    # re 库回退模式(不支持 \p Unicode 属性)
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
    """NFD normalization。"""
    if _regex_available:
        return _ud.normalize("NFD", text)
    return text


def has_answer(answers, text, tokenizer: SimpleTokenizer) -> bool:
    """检查文档是否包含答案字符串。"""
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
# 时间解析工具(移植自 utils.py)
# ═══════════════════════════════════════════════════════════════════════════

def find_month(w: str) -> int | None:
    """从字符串中提取月份数字。"""
    w = w.lower()
    for m, num in MONTH_TO_NUMBER.items():
        if m in w:
            return num
    for m, num in SHORT_MONTH_TO_NUMBER.items():
        if m in w:
            return num
    return None


def replace_dates(text: str) -> str:
    """展开 "1990-93" 格式的年份范围。"""
    pattern = r"(\b\d{4})[–-](\d{2}\b)"

    def replace_func(match):
        start_year = match.group(1)
        end_year = start_year[:2] + match.group(2)
        return " ".join(str(i) for i in range(int(start_year), int(end_year) + 1))

    return re.sub(pattern, replace_func, text)


def expand_year_range(text: str) -> str:
    """展开 "1990-1995" 格式的年份范围。"""
    def replace_range(match):
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        return " ".join(str(year) for year in range(start_year, end_year + 1))

    pattern = r"(\d{4})[–-](\d{4})"
    return re.sub(pattern, replace_range, text)


def year_identifier(timestamp: str) -> list[int]:
    """从文本中提取所有四位年份。

    先展开年份范围(如 "1990-93" → "1990 1991 ... 1993"),再用正则匹配。
    返回去重排序后的整数列表;无匹配时返回空列表。
    """
    timestamp = replace_dates(timestamp)
    timestamp = expand_year_range(timestamp)
    pattern = r"\b(\d{4})(?:s)?\b"
    years = re.findall(pattern, timestamp)
    if years:
        return sorted(set(map(int, years)))
    return []


def remove_implicit_condition(no_time_question: str) -> tuple[str, str | None]:
    """检测并移除隐式时间条件词。

    Returns:
        (normalized_question, implicit_condition):
        implicit_condition 为 'first' 或 'last' 或 None。
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
    """将 treebank POS tag 映射到 WordNet POS。"""
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
    """扩展关键词变体并分类。

    Returns:
        (expanded_keyword_list, keyword_type_list):
        - expanded_keyword_list: 每个关键词的变体列表(含原词)
        - keyword_type_list: 每个关键词的类型(special/superlative/general/numeric/adjective)
    """
    if not _pattern_available:
        # pattern.en 不可用时,不做词形变体扩展
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
    """计算文本中关键词命中的加权分数。"""
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
# 时间系数计算(移植自 metriever.py 原始实现)
# ═══════════════════════════════════════════════════════════════════════════

def get_spline_function(
    time_relation_type: str,
    implicit_condition: str | None,
    question_years: list[int],
) -> Callable:
    """构造时间系数的线性插值函数。

    窗口内的系数在 TEMPORAL_LOW(0.6) 到 1.0 之间线性变化:
    - implicit_condition='first': 偏好较早年份(从 1.0 下降到 0.6)
    - 其他(含 'last'):偏好较晚年份(从 0.6 上升到 1.0)
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
    """对每个句子计算时间系数。

    逻辑:
    1. 从句子文本中提取年份
    2. 按时间关系类型筛选合规年份
    3. 根据 implicit_condition 选择代表年份(最早或最晚)
    4. 用 spline 函数计算系数;无合规年份或超出区间时回退为 0.5
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
# LLM Prompt 生成(移植自 prompts.py)
# ═══════════════════════════════════════════════════════════════════════════

def get_keyword_prompt(question: str) -> str:
    """生成关键词抽取的 LLM prompt。"""
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
    """生成 QFS(Query-Focused Summarization)的 LLM prompt。

    要求 LLM 保留关键日期,文档无关时返回 "None"。
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
# 数据类定义
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Passage:
    """单个 passage 候选。"""
    id: str
    title: str
    text: str
    score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalInfo:
    """从问题中解析出的时间信息。"""
    time_relation: str = ""
    time_relation_type: str = ""
    years: list[int] = field(default_factory=list)
    months: list[int] = field(default_factory=list)
    implicit_condition: str | None = None
    normalized_question: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# MRAGRetriever 主类
# ═══════════════════════════════════════════════════════════════════════════

class MRAGRetriever:
    """独立、可复用的 MRAG 时间感知检索器。

    保留 MRAG 的核心检索管道:

    1. **初始检索**(BM25 via Pyserini 或 nuggetindex store 后端)
    2. **关键词排序**:对候选 passage 计算关键词加权命中分数
    3. **语义排序**:使用 CrossEncoder / embedding 模型重排
    4. **QFS 摘要**:LLM 为 top-k passage 生成问题聚焦摘要
    5. **句子关键词排序**:句子级关键词命中 + 摘要参与排序
    6. **时间-语义混合排序**:
       ``final_score = hybrid_base * semantic + (1-hybrid_base) * semantic * temporal_coeff``

    兼容 ``scripts/run_benchmark_jstrag.py`` 的 ``retriever_factory`` 配置。
    通过 :func:`create_mrag_retriever` 工厂函数注入 nuggetindex store。

    Args:
        bm25_index_path: Pyserini Lucene 索引路径(独立模式)。
        reranker_model_name: 语义重排模型名称(HF 模型 ID)。
        reranker_type: 重排模型类型:``cross_encoder`` / ``bge`` / ``nv_embed`` / ``sfr`` / ``jina``。
        llm: 生成模型实例(用于关键词抽取和 QFS);为 None 时跳过这两步。
        ctx_topk: 关键词排序后保留的 passage 数(默认 100)。
        qfs_topk: 生成 QFS 摘要的 passage 数(默认 5)。
        snt_topk: 进入混合排序的句子数(默认 200)。
        hybrid_score: 是否启用语义-时间混合分数(默认 True)。
        hybrid_base: 混合公式中语义分数的最低保留比例(默认 0.0)。
        snt_with_title: 是否在句子前附加 passage 标题(默认 True)。
        store: NuggetStore 实例(nuggetindex 集成模式);为 None 时使用独立模式。
        device: 模型设备(如 "cuda:0");为 None 时自动选择。
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
        # 关键词抽取/QFS 的采样温度:0 为贪婪解码(评估可复现),默认 0.2 保持原行为
        self._llm_temperature = llm_temperature
        self._store = store
        self._device = device

        # 懒加载的模型实例
        self._bm25_searcher = None
        self._reranker_model = None
        self._reranker_tokenizer = None
        self._tokenizer = SimpleTokenizer()

        # 关键词缓存(按 normalized_question)
        self._keyword_cache: dict[str, tuple[list[list[str]], list[str]]] = {}

    # ───────────────────────────────────────────────────────────────────────
    # 模型懒加载
    # ───────────────────────────────────────────────────────────────────────

    def _load_bm25_searcher(self):
        """加载 Pyserini LuceneSearcher。"""
        if self._bm25_searcher is None and self._bm25_index_path:
            from pyserini.search.lucene import LuceneSearcher
            self._bm25_searcher = LuceneSearcher(self._bm25_index_path)
        return self._bm25_searcher

    def _load_reranker(self):
        """加载语义重排模型。"""
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
    # 初始检索模块(Initial Retrieval)
    # ───────────────────────────────────────────────────────────────────────

    async def _store_bm25_search(self, query: str, top_k: int = 1000) -> list[Passage]:
        """通过 nuggetindex store 后端执行 BM25 检索(集成模式)。

        从 store 后端获取 BM25 候选 nugget ID,然后:
        1. 用 ``aget_passages`` 获取完整 passage 文本
        2. 按 source_id 聚合为 passage,用完整文本作为检索单元
        """
        if self._store is None:
            return []
        backend = getattr(self._store, "_backend_impl", None)
        if backend is None:
            return []

        # BM25 检索(与原始 MRAG 一致:不做前置时间过滤,时间感知在句子级混合排序阶段处理)
        # FTS5 sanitize(去特殊符号 + OR join)在此处完成:上层传入的是原始 query,
        # 以保留 "as of" 等时间触发词短语供 parse_temporal_question 检测。
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

        # 2. 获取 nugget 对象,提取 source_id
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

        # 3. 获取完整 passage 文本
        try:
            passages_text = await backend.aget_passages(source_ids)
        except Exception as exc:
            logger.warning("aget_passages 失败: %s,回退到 nugget fact text", exc)
            passages_text = {}

        # 4. 构建 Passage 对象(按 source_id 聚合)
        passages: list[Passage] = []
        for sid, nugget_hits in nugget_by_source.items():
            # 使用完整 passage 文本;回退到 nugget fact text
            text = passages_text.get(sid, "")
            if not text:
                # 回退:拼接该 source 的 nugget fact text
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

            # 取该 source 下 BM25 最高的 nugget 的分数作为 passage 分数
            best_score = max(s for _, s in nugget_hits)
            best_nugget_id = max(nugget_hits, key=lambda x: x[1])[0]
            # store 集成模式下 passage 默认不带标题,而重排器
            # (rank_by_semantic / snt_with_title 的时间系数口径)按
            # ``title + " " + text`` 建模;代词段与列表答案单元格缺了
            # 归属实体后语义分会被打到 −9 量级。设置
            # MRAG_TITLE_FROM_SOURCE_ID=true 可按 TimeQA source_id
            # 还原标题(默认关闭,保持既有 benchmark 可比)。
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
        """使用 Pyserini LuceneSearcher 执行 BM25 检索(独立模式)。

        初始化参数:仅传入 Lucene 索引路径。
        调用方法:``searcher.search(question, k=topk)``。
        返回结果:每个 hit 包含 ``docid``(格式 "id::title")、``score``;
        通过 ``searcher.doc(id).raw()`` 获取原文 JSON,解析 ``['contents']`` 取文本。
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
        """初始检索:BM25(集成模式优先,独立模式兜底)。

        - 若 ``candidates`` 非空(预加载模式),直接使用。
        - 否则优先使用 nuggetindex store 的后端 BM25(集成模式)。
        - 否则使用 Pyserini LuceneSearcher(独立模式)。
        """
        if candidates is not None:
            return candidates[:top_k]

        passages: list[Passage] = []

        # 集成模式:从 nuggetindex store 后端获取 BM25 候选
        if self._store is not None:
            passages = await self._store_bm25_search(query, top_k=top_k)

        # 独立模式:使用 Pyserini LuceneSearcher
        if not passages and self._bm25_index_path:
            passages = self._bm25_search(query, top_k=top_k)

        return passages[:top_k]

    # ───────────────────────────────────────────────────────────────────────
    # 时间感知重排序模块(Temporal-aware Reranking)
    # ───────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_time_relation(
        question: str,
    ) -> tuple[str, str, str] | None:
        """当调用方未显式给出 ``time_relation`` 时,从问题文本中自动检测。

        扫描策略:多词触发词优先(按长度降序)、词边界匹配、大小写不敏感;
        仅当触发词**之后的文本能解析出至少一个四位年份**时才采纳,
        以排除 "a book by John" 这类非时间用法的误命中。

        Returns:
            ``(trigger_lower, date_suffix, prefix)``:触发词(小写,供分类)、
            触发词之后的日期文本、触发词之前的问题文本;未命中返回 None。
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
        """解析问题中的时间信息。

        ``time_relation`` 为空时自动从问题中检测触发词
        (as of / before / after / since / until / from / by),
        与原始 MRAG 数据集自带 ``time_relation`` 字段的解析分支一致。

        Returns:
            TemporalInfo: 包含 time_relation_type, years, months,
            implicit_condition, normalized_question。
        """
        time_relation = time_relation.strip()
        time_relation_type = ""
        years: list[int] = []
        months: list[int] = []
        no_time_question = question
        date = ""

        if time_relation and time_relation in question:
            # 显式传入(原始 MRAG 数据集字段路径)
            parts = question.split(time_relation)
            no_time_question = time_relation.join(parts[:-1])
            date = parts[-1]
            time_relation = time_relation.lower()
        elif not time_relation:
            # 集成路径(nuggetindex aretrieve 只有 query 文本):自动检测
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

            # 月份解析(当前未参与最终排序,但保留以供扩展)
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

        # 清理触发词被切除后残留的尾部空白与悬空开括号
        # (如 "Who was CEO? (as of 2018)" → prefix 末尾的 "(")。
        if time_relation:
            no_time_question = re.sub(r"[\s(]+$", "", no_time_question)

        normalized_question, implicit_condition = remove_implicit_condition(no_time_question)
        # 时间状语前置(如 "As of 2018, who was CEO?")会导致去时间后为空,
        # 此时回退到原始问题,避免语义排序拿到空 query。
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
        """从问题中抽取并扩展关键词。

        若 LLM 可用,使用 LLM 生成关键词;否则使用简单的非停用词提取。
        结果按 normalized_question 缓存。
        """
        if normalized_question in self._keyword_cache:
            return self._keyword_cache[normalized_question]

        # 规范化特殊问题前缀
        q = normalized_question
        if q.startswith("How many times"):
            q = q.replace("How many times", "When")
        elif q.startswith("How many"):
            q = q.replace("How many", "What")

        keyword_list = await self._allm_extract_keywords(q)
        if not keyword_list:
            keyword_list = self._simple_extract_keywords(q)

        # 后处理:过滤低信息词,确保是原问题子串
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
        """使用 LLM 抽取关键词。"""
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
            return list(eval(tmp))  # noqa: S307 (LLM 输出为 Python list 字面量)
        except Exception:
            logger.warning("LLM 关键词抽取失败,回退到简单关键词提取", exc_info=True)
            return []

    def _simple_extract_keywords(self, question: str) -> list[str]:
        """简单关键词提取(无 LLM 时的兜底)。"""
        tokens = word_tokenize(question)
        tagged = pos_tag(tokens)
        keywords = []
        for word, tag in tagged:
            if word.lower() not in EXCL and not tag.startswith("DT") and not tag.startswith("IN") and not tag.startswith("PRP"):
                keywords.append(word)
        return keywords

    async def _acall_llm(self, prompts: list[str], max_tokens: int = 100) -> list[str]:
        """异步调用 LLM 生成文本。

        按以下顺序识别 ``self._llm`` 的接口(鸭子类型):

        1. 框架 :class:`~tcrag.llm.base.BaseLLM`(有 ``agenerate``,
           如 OpenAI / Ollama 客户端)——必须最先检测:``BaseLLM`` 同时
           带有同步 ``generate()`` 包装,放后面会被误判为 vLLM。
        2. vLLM 离线引擎(同步 ``generate(prompts, SamplingParams)``),
           用 ``asyncio.to_thread`` 包裹避免阻塞事件循环。
        3. 裸 OpenAI 兼容客户端(``chat.completions.create``)。
        """
        if self._llm is None:
            return [""] * len(prompts)

        if hasattr(self._llm, "agenerate"):
            # 框架统一 LLM 接口:agenerate(prompt, ...) -> LLMResponse
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
            # vLLM 离线引擎(同步阻塞,放线程池执行)
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
            # OpenAI 兼容接口(裸同步 openai.OpenAI 客户端)
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

        # 去除停止标记
        for stopper in ["</Keywords>", "</Summarization>", "</Answer>", "</Info>"]:
            responses = [r.split(stopper)[0] if stopper in r else r for r in responses]
        return responses

    # ───────────────────────────────────────────────────────────────────────
    # 重排步骤 1:Passage 关键词排序
    # ───────────────────────────────────────────────────────────────────────

    def rank_by_keywords(
        self,
        candidates: list[Passage],
        expanded_keywords: list[list[str]],
        keyword_types: list[str],
    ) -> list[Passage]:
        """Passage 关键词排序。

        对每个 passage 计算 ``title + text`` 中命中的关键词权重之和,
        按总分降序排列,保留前 ``ctx_topk`` 个。
        """
        scored = []
        for ctx in candidates:
            text = ctx.title + " " + ctx.text
            kw_score = count_keyword_scores(text, expanded_keywords, keyword_types)
            scored.append((ctx, kw_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [tp[0] for tp in scored[: self._ctx_topk]]

    # ───────────────────────────────────────────────────────────────────────
    # 重排步骤 2:Passage 语义排序
    # ───────────────────────────────────────────────────────────────────────

    def rank_by_semantic(
        self,
        candidates: list[Passage],
        query: str,
        normalized: bool = False,
    ) -> list[Passage]:
        """Passage 语义排序。

        使用语义模型(CrossEncoder / BGE / NV-Embed)对 ``[query, title+text]`` 对打分。
        ``normalized=True`` 时使用 normalized_question(去除时间短语)。
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
        """根据重排模型类型计算语义分数。"""
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
        """NV-Embed 语义打分。"""
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
        """SFR Embedding 语义打分。"""
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
        """SFR 的 last-token pooling。"""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

    # ───────────────────────────────────────────────────────────────────────
    # 重排步骤 3:QFS 摘要生成
    # ───────────────────────────────────────────────────────────────────────

    async def generate_qfs_summaries(
        self, candidates: list[Passage], query: str, top_k: int | None = None,
    ) -> list[str | None]:
        """为 top-k passage 生成 QFS 摘要。

        使用 LLM 生成保留关键日期的问题聚焦摘要;文档无关时返回 None。
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
    # 重排步骤 4:句子关键词排序
    # ───────────────────────────────────────────────────────────────────────

    def rank_sentences_by_keywords(
        self,
        candidates: list[Passage],
        summaries: list[str | None],
        expanded_keywords: list[list[str]],
        keyword_types: list[str],
    ) -> tuple[list[tuple[str, str, float]], dict[str, Passage]]:
        """句子关键词排序。

        1. 用 sent_tokenize() 切分每个 passage 的句子
        2. 可选地在每个句子前附加标题
        3. 若 QFS 摘要不为 None,把摘要当作额外句子
        4. 对所有句子计算关键词分数并全局排序

        Returns:
            (sentence_tuples, get_ctx_by_id):
            - sentence_tuples: [(passage_id, sentence, kw_score), ...] 按分数降序
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
    # 重排步骤 5:时间-语义混合排序
    # ───────────────────────────────────────────────────────────────────────

    def hybrid_rank_sentences(
        self,
        sentence_tuples: list[tuple[str, str, float]],
        query: str,
        normalized_query: str,
        temporal_info: TemporalInfo,
        candidates: list[Passage],
    ) -> list[tuple[str, str, float]]:
        """句子语义-时间混合排序。

        组合公式:
        ``final_score = hybrid_base * semantic + (1-hybrid_base) * semantic * temporal_coeff``

        默认 ``hybrid_base=0``,等价于 ``final_score = semantic * temporal_coeff``。
        对无年份或 other 类型的问题,直接使用语义分数。
        """
        # 截取 top-snt_topk 句子,其余保持原序
        snt_topk = min(len(sentence_tuples), self._snt_topk)
        sentence_tuples_unchange = sentence_tuples[snt_topk:]
        sentence_tuples = sentence_tuples[:snt_topk]

        years = temporal_info.years
        time_relation_type = temporal_info.time_relation_type
        implicit_condition = temporal_info.implicit_condition

        # 决定使用 normalized_question 还是原始 question
        use_hybrid = (
            len(years) > 0
            and time_relation_type != "other"
            and self._hybrid_score
        )
        search_query = normalized_query if use_hybrid else query

        # 计算语义分数
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
            # 无模型时使用关键词分数作为语义分数的替代
            semantic_scores = [tp[2] for tp in sentence_tuples]

        # 计算时间系数并组合
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

        # 组装最终句子三元组
        sentence_tuples = [
            (tp[0], tp[1], score) for score, tp in zip(final_scores, sentence_tuples)
        ]
        sentence_tuples.sort(key=lambda x: x[2], reverse=True)
        sentence_tuples += sentence_tuples_unchange
        return sentence_tuples

    # ───────────────────────────────────────────────────────────────────────
    # 主检索接口
    # ───────────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        *,
        time_relation: str = "",
        candidates: list[Passage] | None = None,
        top_k: int = 10,
    ) -> list[Passage]:
        """执行完整的 MRAG 检索管道(独立模式)。

        Args:
            query: 查询文本。
            time_relation: 时间关系词(如 "after", "before");为空时自动检测。
            candidates: 预加载的候选 passage;为 None 时执行初始检索。
            top_k: 返回的 passage 数量。

        Returns:
            按 MRAG 最终排名排序的 passage 列表。
        """
        # Step 0: 初始检索
        initial_candidates = await self._initial_retrieval(query, candidates)
        if not initial_candidates:
            return []

        # Step 1: 时间信息预处理
        temporal_info = self.parse_temporal_question(query, time_relation)
        normalized_question = temporal_info.normalized_question or query

        # Step 2: 关键词抽取
        # 提取出query里有意义的词语
        # expended keywords: 基于query内原来的词语，联想出相关的词也加入关键词
        expanded_keywords, keyword_types = await self.aextract_keywords(normalized_question)

        # Step 3: Passage 关键词排序 → top-ctx_topk
        # 计数candidate里的query keyword的数量,并按数量排序
        ctx_kw_ranked = self.rank_by_keywords(
            initial_candidates, expanded_keywords, keyword_types,
        )

        # Step 4: Passage 语义排序 → 重排 top-ctx_topk
        ctx_semantic_ranked = self.rank_by_semantic(
            ctx_kw_ranked, normalized_question, normalized=True,
        )

        # Step 5: QFS 摘要生成
        summaries = await self.generate_qfs_summaries(
            ctx_semantic_ranked, normalized_question, top_k=self._qfs_topk,
        )
        for i, s in enumerate(summaries):
            if i < len(ctx_semantic_ranked):
                ctx_semantic_ranked[i].metadata["qfs_summary"] = s

        # Step 6: 句子关键词排序
        sentence_tuples, get_ctx_by_id = self.rank_sentences_by_keywords(
            ctx_semantic_ranked, summaries, expanded_keywords, keyword_types,
        )

        # Step 7: 时间-语义混合排序
        final_sentence_tuples = self.hybrid_rank_sentences(
            sentence_tuples, query, normalized_question, temporal_info, ctx_semantic_ranked,
        )

        # 根据 sentence rank 反推 passage rank
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
        """NuggetIndex 集成接口(兼容 ``retriever_factory``)。

        签名匹配 nuggetindex ``Retriever.aretrieve``,
        返回 ``RetrievalResult`` 列表。
        """
        # 从 store 后端获取初始候选
        candidates = await self._initial_retrieval(query, top_k=max(top_k * 10, 1000))

        if not candidates:
            return []

        # 执行 MRAG 管道
        ranked_passages = await self.retrieve(
            query, candidates=candidates, top_k=top_k,
        )

        # 转换为 RetrievalResult
        return await self._to_retrieval_results(ranked_passages, top_k)

    async def _to_retrieval_results(
        self, passages: list[Passage], top_k: int,
    ) -> list[Any]:
        """将 Passage 列表转换为 nuggetindex RetrievalResult。

        优先从 store 后端获取原始 nugget(保留 provenance/validity),
        找不到时构造新的 Nugget。
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

            # 尝试从 store 后端获取原始 nugget(保留 provenance/validity)
            if backend is not None:
                nugget_ids = p.metadata.get("nugget_ids", [])
                best_nid = p.metadata.get("best_nugget_id")
                # 优先用 best_nugget_id,其次尝试列表中的第一个
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
                # 回退:构造新的 Nugget
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
# NuggetIndex 集成工厂函数
# ═══════════════════════════════════════════════════════════════════════════

def create_mrag_retriever(store: Any, *, llm: Any | None = None) -> MRAGRetriever:
    """NuggetIndex 集成工厂函数。

    兼容 ``scripts/run_benchmark_jstrag.py`` 的 ``retriever_factory`` 配置:

    .. code-block:: yaml

        systems:
          nuggetindex:
            retriever_factory: "tcrag.retrievers.metriever:create_mrag_retriever"

    签名: ``(store: NuggetStore, *, llm=None) -> MRAGRetriever``
    返回的 MRAGRetriever 实现
    ``async aretrieve(query, *, query_time, view, top_k, fusion, filters)``。

    Args:
        store: NuggetStore 实例(nuggetindex 集成模式)。
        llm: 生成模型实例,由外部 run_benchmark 脚本传入(与 QA 阶段共用);
            用于关键词抽取和 QFS 摘要,为 None 时跳过这两步。

    环境变量(可选):
      - ``MRAG_BM25_INDEX_PATH``: Pyserini Lucene 索引路径(独立 BM25 检索)
      - ``MRAG_RERANKER_MODEL``: 语义重排模型 HF 名称(默认 ``nvidia/NV-Embed-v2``,
        参照 MRAG metriever.py 的默认 stage2 重排模型 metriever_model=nv2)
      - ``MRAG_RERANKER_TYPE``: 重排模型类型(cross_encoder/bge/nv_embed/sfr/jina,
        默认 nv_embed)
      - ``MRAG_CTX_TOPK``: 关键词排序后保留的 passage 数(默认 100)
      - ``MRAG_SNT_TOPK``: 进入混合排序的句子数(默认 200)
      - ``MRAG_QFS_TOPK``: QFS 摘要的 passage 数(默认 5)
      - ``MRAG_HYBRID_BASE``: 混合公式中语义分数的最低保留比例(默认 0.0)
      - ``MRAG_LLM_TEMPERATURE``: 关键词抽取/QFS 的采样温度(默认 0.2);
        评估复现可设 0(贪婪解码)
    """
    bm25_index = os.getenv("MRAG_BM25_INDEX_PATH")
    # 默认 stage2 重排模型参照 MRAG metriever.py:metriever_model 默认 nv2
    # → nvidia/NV-Embed-v2,对应 reranker_type=nv_embed。
    reranker_model = os.getenv("MRAG_RERANKER_MODEL", "nvidia/NV-Embed-v2")
    reranker_type = os.getenv("MRAG_RERANKER_TYPE", "nv_embed")
    ctx_topk = int(os.getenv("MRAG_CTX_TOPK", "100"))
    snt_topk = int(os.getenv("MRAG_SNT_TOPK", "200"))
    qfs_topk = int(os.getenv("MRAG_QFS_TOPK", "5"))
    hybrid_base = float(os.getenv("MRAG_HYBRID_BASE", "0.0"))
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
