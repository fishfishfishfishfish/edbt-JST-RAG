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

    If the given ``name`` already carries the ``tcrag.`` prefix (e.g.
    ``__name__`` is ``tcrag.retrievers.xxx`` when imported as a package), it
    is first normalized by stripping the prefix, avoiding a double-prefixed
    logger such as ``tcrag.tcrag.retrievers.xxx``, which would not match the
    real logger when calling setLevel / applying filters under
    ``tcrag.retrievers.xxx``.
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

    Two environment variables are supported for temporary overrides, neither
    requiring a config-file change:
      - ``TCRAG_LOG_LEVEL``: global level (e.g. ``DEBUG``), taking precedence
        over the YAML ``logging.level``;
      - ``TCRAG_DEBUG_LOGGERS``: comma-separated module names; only these
        loggers are raised to DEBUG while other modules keep their original
        level. Module names may be given in full or without the ``tcrag.``
        prefix, e.g. ``TCRAG_DEBUG_LOGGERS=retrievers.jstretriever``.
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
