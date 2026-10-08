"""Ollama 容错 structured-output client,对接 nuggetindex ``LLMClient`` 协议。

nuggetindex 原生 :class:`OllamaClient` 基于 instructor 严格校验,小模型
(如 llama3.2:3b)常以两种方式违反 ``ExtractionPayload`` schema:

1. 输出单个 fact 对象而非 ``{"facts": [...]}`` 数组;
2. 把 ``confidence`` 写成 ``null``(schema 要求 float,显式 null 不合法)。

instructor 的修复循环把 ValidationError 回传重试,但小模型往往修不干净,
重试耗尽后整篇文档 ingest failed。本 client 绕过 instructor,直接请求
Ollama 的 OpenAI 兼容接口,在 pydantic 校验前规范化输出,兜住上述模式;
仍失败时抛 ``ValueError``,由上层 ingest 的 try/except 降级为单文档跳过。
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

__all__ = ["OllamaCompatClient"]

_DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


class OllamaCompatClient:
    """Duck-type nuggetindex ``LLMClient``(仅需 ``achat_structured``)。"""

    def __init__(self, cfg: Any, *, max_retries: int = 1) -> None:
        from openai import AsyncOpenAI

        self._cfg = cfg
        self._max_retries = max_retries
        self._client = AsyncOpenAI(
            api_key=cfg.api_key or "ollama",
            base_url=cfg.base_url or _DEFAULT_OLLAMA_URL,
            timeout=cfg.timeout_seconds,
        )

    async def achat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> BaseModel:
        last_exc: Exception | None = None
        for _ in range(self._max_retries + 1):
            # Ollama structured output:constrained decoding 在 token 层
            # 强制 schema 合法(JSON 对象/数组、confidence 不为 null),
            # 小模型(llama3.2)也能稳定产出;_normalize 仅作兜底。
            # schema 副本上加 maxItems:小模型可能无限生成事实直到
            # max_tokens 截断产出非法 JSON;与框架 max_facts 语义一致。
            schema = response_model.model_json_schema()
            facts_prop = schema.get("properties", {}).get("facts")
            if isinstance(facts_prop, dict):
                facts_prop.setdefault("maxItems", 20)
            resp = await self._client.chat.completions.create(
                model=self._cfg.model,
                messages=messages,
                temperature=self._cfg.temperature,
                max_tokens=self._cfg.max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                    },
                },
            )
            content = resp.choices[0].message.content or ""
            try:
                return self._parse(content, response_model)
            except Exception as exc:  # noqa: BLE001 - parse boundary, retried below
                last_exc = exc
        raise ValueError(
            f"structured output failed after {self._max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    @classmethod
    def _parse(cls, content: str, response_model: type[BaseModel]) -> BaseModel:
        if not content:
            raise ValueError("LLM returned empty content")
        return response_model.model_validate(cls._normalize(cls._decode_json(content)))

    @staticmethod
    def _decode_json(content: str) -> Any:
        """容忍 markdown fence 与前后缀说明文本,解码首个 JSON 值。

        llama3.2 常输出 ``Here are ...:\\n```json [...]``(顶层数组)或带
        说明文字的对象,故从首个 ``{``/``[`` 用 ``raw_decode`` 取首个完整
        JSON 值;顶层对象/数组/裸对象拼接均能覆盖。
        """
        starts = [i for i in (content.find("{"), content.find("[")) if i >= 0]
        if starts:
            try:
                data, _ = json.JSONDecoder().raw_decode(content[min(starts) :].lstrip())
                return data
            except json.JSONDecodeError:
                pass
        m = re.search(r"\{.*\}", content, re.DOTALL)
        raw = m.group(0) if m else content
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON from LLM: {exc}") from exc

    @staticmethod
    def _normalize(data: Any) -> Any:
        """修复小模型对 ``ExtractionPayload`` 的典型违规。"""
        if isinstance(data, list):
            data = {"facts": data}
        if not isinstance(data, dict):
            return data
        if "facts" not in data and "subject" in data and "predicate" in data:
            data = {"facts": [data]}
        facts = data.get("facts")
        if isinstance(facts, list):
            cleaned = []
            for f in facts:
                if not isinstance(f, dict):
                    continue
                # 剔除空 subject/predicate/object(FactTriple 要求非空字符串)
                if not all(
                    isinstance(f.get(k), str) and f.get(k, "").strip()
                    for k in ("subject", "predicate", "object")
                ):
                    continue
                if f.get("confidence") is None:
                    # 剔除 null 让 pydantic default(0.8) 生效
                    f = {k: v for k, v in f.items() if k != "confidence"}
                cleaned.append(f)
            data["facts"] = cleaned
        return data
