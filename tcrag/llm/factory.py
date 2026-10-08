"""Factory that builds the configured LLM backend."""

from __future__ import annotations

from tcrag.config import LLMConfig
from tcrag.llm.base import BaseLLM
from tcrag.llm.ollama_client import OllamaLLM
from tcrag.llm.openai_client import OpenAILLM


def create_llm(cfg: LLMConfig) -> BaseLLM:
    """Instantiate the LLM backend described by ``cfg``."""
    provider = (cfg.provider or "").lower()
    if provider == "openai":
        return OpenAILLM(
            api_key=cfg.api_key,
            model=cfg.model,
            base_url=cfg.base_url or None,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
        )
    if provider == "ollama":
        return OllamaLLM(
            model=cfg.ollama_model,
            host=cfg.ollama_host,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
        )
    raise ValueError(f"Unsupported LLM provider: {provider!r} (expected 'openai' or 'ollama')")
