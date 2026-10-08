"""spaCy small-model based atomic-fact extractor.

使用 ``en_core_web_sm``(默认)等 spaCy 小模型,通过**依存句法 + NER** 做
开放式三元组抽取,不依赖 LLM、不依赖 nuggetindex,产出与框架其他提取器
一致的 :class:`~tcrag.data.models.AtomicFact`(subject / predicate /
object / text + NER 类型)。

抽取策略(每个句子)
====================

1. 依存分析找动词中心语(pos 为 VERB;含系动词 AUX+attr):
   - 主语:nsubj / nsubjpass;
   - 宾语:dobj / attr / dative / oprd;
   - 介词宾语:动词的 prep 子节点的 pobj(如 "worked at Norwich" →
     predicate="work at",object="Norwich");被动施事 agent("founded by
     Y")同样覆盖;
   - 并列动词(conj)复用同一主语("He served and retired")。
2. 主语/宾语短语优先取 spaCy noun_chunk(剥离前导冠词/代词所有格),
   保证短语完整("the House of Commons" 而不是单个 "Commons")。
3. NER 标签写入 ``subject_type`` / ``object_type``(PERSON / ORG /
   GPE / DATE ...)。
4. 主语是代词("He / She")且能从 ``source_id`` 还原 TimeQA 页面实体时,
   用页面实体替换;无法还原则跳过该三元组(代词主语对检索无价值)。
5. 谓词取动词 lemma + 介词/助词("give up"),被动形态保持 lemma 形式。

依赖处理
========

spaCy 与模型在**本模块内**采用软依赖 + 懒加载:本文件顶层不 import
spacy,首次调用 :meth:`SpacyFactExtractor.aextract` 时才 ``spacy.load``,
缺包/缺模型时抛带安装命令的明确 :class:`RuntimeError`(注意包级
``tcrag/extractors/__init__.py`` 仍会导入其他提取器,它们可能依赖
nuggetindex,这是既有行为;直接使用本模块请在 tcrag conda 环境内)。

    pip install spacy
    python -m spacy download en_core_web_sm

配置(configs/default.yaml)::

    extractor:
      type: spacy                # 或 spacy_sm
      spacy_model: en_core_web_sm
      spacy_max_facts: 20
"""

from __future__ import annotations

import asyncio
import re
from urllib.parse import unquote

from tcrag.data.models import AtomicFact
from tcrag.extractors.base import BaseExtractor

# 与 timeqa_extractor 同口径的 source_id → 页面实体解析(本地复制一份,
# 避免 import timeqa_extractor 把 nuggetindex 拖成硬依赖)。
_TIMEQA_SOURCE_RE = re.compile(r"^(?:timeqa_)?/wiki/(?P<title>.+)_(?P<paragraph>\d+)$")
_SPACE_RE = re.compile(r"\s+")

_SUBJECT_DEPS = {"nsubj", "nsubjpass"}
_OBJECT_DEPS = {"dobj", "attr", "dative", "oprd", "pcomp"}
_PREP_DEPS = {"prep", "prepc"}          # 旧标签兼容(spaCy 3 统一为 prep)
_LEADING_DET_DEPS = {"det", "poss"}
# PP 扩展时拒绝带子句的介词短语(which/that 从句、状语从句等)。
_CLAUSAL_DEPS = {"relcl", "acl", "advcl", "ccomp", "xcomp", "parataxis"}
# 表层分词 → 谓词覆盖:spaCy 对 "born" 的 lemma 是 "bear",
# 三元组里出现 "bear in Manchester" 对检索/阅读都无意义。
_VERB_SURFACE_OVERRIDE = {"born": "born", "borne": "born"}
# 引导从句的介词(whilst Dean ... / when he became ...),不是名词性宾语。
# 注意 "as" 不能加:"worked as a scientist" 是有用的名词性 PP。
_CLAUSAL_PREPS = {
    "when", "while", "whilst", "although", "though", "if", "because",
    "unless", "whereas",
}

# 短语长度上限,过滤解析器偶发的巨型子树(同 rule_based 的 60/句策略)。
_MAX_PHRASE_CHARS = 80


def entity_from_source_id(source_id: str | None) -> str:
    """``timeqa_/wiki/Page_Title_3`` → ``"Page Title"``;不可解析时为空。"""
    if not source_id:
        return ""
    m = _TIMEQA_SOURCE_RE.match(source_id.strip())
    if m is None:
        return ""
    title = unquote(m.group("title")).replace("_", " ").replace("&amp;", "&")
    return _SPACE_RE.sub(" ", title).strip()


class SpacyFactExtractor(BaseExtractor):
    """用 spaCy 小模型(依存 + NER)抽取 (subject, predicate, object) 事实。

    Parameters
    ----------
    model:
        spaCy 模型名,默认 ``en_core_web_sm``。
    max_facts:
        单次抽取返回的事实上限(默认 20,与 :class:`LLMExtractor` 一致);
        ``<=0`` 表示不限。
    """

    def __init__(self, model: str = "en_core_web_sm", *, max_facts: int = 20) -> None:
        self._model_name = model
        self._max_facts = max_facts
        self._nlp: object | None = None  # 懒加载,类型为 spacy.Language

    # ── 模型加载(软依赖)────────────────────────────────────────────────

    def _ensure_nlp(self) -> object:
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - 依赖缺失边界
            raise RuntimeError(
                "SpacyFactExtractor 需要 spaCy,但当前环境未安装:"
                " pip install spacy && python -m spacy download "
                f"{self._model_name}"
            ) from exc
        try:
            # 需要 tagger/parser/ner 全部管道(默认全开,这里显式声明)。
            self._nlp = spacy.load(self._model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy 模型 {self._model_name!r} 不可用,请先安装:"
                f" python -m spacy download {self._model_name}"
            ) from exc
        return self._nlp

    # ── 公共接口 ────────────────────────────────────────────────────────

    async def aextract(
        self,
        text: str,
        *,
        context: str = "",
        source_id: str | None = None,
    ) -> list[AtomicFact]:
        if not text or not text.strip():
            return []
        # spaCy 解析是阻塞 CPU 调用,丢到线程池避免卡住事件循环。
        return await asyncio.to_thread(self._extract_sync, text, source_id)

    # ── 核心实现 ────────────────────────────────────────────────────────

    def _extract_sync(self, text: str, source_id: str | None) -> list[AtomicFact]:
        nlp = self._ensure_nlp()
        doc = nlp(text)
        page_entity = entity_from_source_id(source_id)

        # token.i -> (chunk_text 剥冠词, chunk.root)
        chunk_by_root: dict[int, object] = {}
        for chunk in doc.noun_chunks:
            chunk_by_root[chunk.root.i] = chunk

        # token.i -> NER 标签(实体内每个 token 都映射)。
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
                # 代词主语:能还原则替换,否则跳过(对 BM25/索引无价值)。
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

    # ── 句法辅助 ────────────────────────────────────────────────────────

    @staticmethod
    def _clause_heads(sent: object) -> list[object]:
        """句子的谓词中心语:主句动词 + 复用其主语的并列动词。

        系动词结构("X was Y")中 root 可能是 AUX,只要带 attr 也算。
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
        """取动词的主语;并列动词没带自己的主语时回退到父句主语。"""
        for child in head.children:
            if child.dep_ in _SUBJECT_DEPS:
                return child
        # "He moved to London and worked at a bank" → worked 复用 He。
        parent = head.head
        if head.dep_ in {"conj", "parataxis"} and parent is not head:
            for child in parent.children:
                if child.dep_ in _SUBJECT_DEPS:
                    return child
        return None

    @staticmethod
    def _objects_of(head: object) -> list[tuple[str, object]]:
        """返回 [(predicate 文本, object token), ...]。

        predicate = 动词 lemma,可带助词(prt,如 "give up")和介词
        ("work at");被动施事走 agent→by→pobj。
        """
        verb_lemma = _VERB_SURFACE_OVERRIDE.get(head.text.lower(), head.lemma_)
        # 短语动词助词(give up / set up),保守只取 prt,不把 acomp 补语算进来。
        particles = [c.lemma_ for c in head.children if c.dep_ == "prt"]
        base_pred = " ".join([verb_lemma, *particles]).strip()

        out: list[tuple[str, object]] = []

        # 直接宾语/表语/与格/补语。
        for child in head.children:
            if child.dep_ in _OBJECT_DEPS:
                out.append((base_pred, child))

        # 介词宾语 / 被动施事:worked at <X>;was founded by <Y>。
        for child in head.children:
            if child.dep_ in _PREP_DEPS or child.dep_ == "agent":
                prep = child.lemma_.lower()
                if child.dep_ == "agent":
                    prep = "by"
                # 跳过从句介词(when/whilst/...),以及子树里还套着动词的
                # 介词短语(说明是状语从句而非名词性宾语)。
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
        """主语/宾语的完整短语。

        spaCy v3 的 noun_chunks 默认**不含**后续 PP,直接取 chunk 会把
        "the University of East Anglia" 截成 "University"。这里以 chunk
        为基础:剥前导冠词/所有格 → 沿名词的 prep 子树扩展 PP(含一层
        嵌套,如 "Member of Parliament for Norwich North";拒绝带子句的
        PP)→ 补 nummod/npadvmod("May 1997")。
        """
        chunk = chunk_by_root.get(token.i)
        if chunk is not None:
            idxs = {
                t.i
                for t in chunk
                if not (t.dep_ in _LEADING_DET_DEPS and t.i < chunk.root.i)
            }
            # BFS 扩展名词性 PP:root → prep → pobj → prep ...
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
            # 日期/数字等可能不在 noun_chunk 内,从 token 自身起步。
            idxs = {token.i}
            root_token = token

        # 数字/名词修饰(1997、$5M 之类),无论是否在 chunk 内都补上。
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
