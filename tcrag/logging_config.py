"""Centralized logging configuration for TCRag."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any


_LOGGER_NAME = "tcrag"
_configured = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger of the TCRag root logger.

    传入的 ``name`` 若已带 ``tcrag.`` 前缀(如 ``__name__`` 在以包形式
    导入时为 ``tcrag.retrievers.xxx``)会先归一化去掉,避免拼出
    ``tcrag.tcrag.retrievers.xxx`` 这种双前缀 logger,导致按
    ``tcrag.retrievers.xxx`` 调 setLevel / 过滤时匹配不到真实 logger。
    """
    root = logging.getLogger(_LOGGER_NAME)
    if name:
        if name == _LOGGER_NAME:
            return root
        prefix = _LOGGER_NAME + "."
        if name.startswith(prefix):
            name = name[len(prefix):]
        return root.getChild(name)
    return root


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    console: bool = True,
) -> logging.Logger:
    """Configure the TCRag logger. Idempotent: re-configures on each call."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Clear existing handlers so re-configuration does not duplicate output.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    _configured = True
    return logger


def configure_from_dict(cfg: dict[str, Any]) -> logging.Logger:
    """Apply logging config from the ``logging`` section of the YAML config.

    支持两个环境变量临时覆盖,均无需改配置文件:
      - ``TCRAG_LOG_LEVEL``:整体级别(如 ``DEBUG``),优先级高于 YAML 的
        ``logging.level``;
      - ``TCRAG_DEBUG_LOGGERS``:逗号分隔的模块名,仅把这些 logger 提到
        DEBUG,其他模块保持原级别。模块名写全名或省略 ``tcrag.`` 前缀
        均可,例如 ``TCRAG_DEBUG_LOGGERS=retrievers.jstretriever``。
    """
    level = os.getenv("TCRAG_LOG_LEVEL", cfg.get("level", "INFO"))
    root = setup_logging(
        level=level,
        log_file=cfg.get("file"),
        console=cfg.get("console", True),
    )
    raw = os.getenv("TCRAG_DEBUG_LOGGERS", "").strip()
    if raw:
        for part in raw.split(","):
            name = part.strip()
            if not name:
                continue
            if not name.startswith(_LOGGER_NAME):
                name = f"{_LOGGER_NAME}.{name}"
            logging.getLogger(name).setLevel(logging.DEBUG)
    return root
