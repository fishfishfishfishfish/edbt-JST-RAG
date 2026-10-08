"""跨模块共享的文本工具函数。"""

from __future__ import annotations

import re

# Non-word characters (including FTS5 operators like . : ( ) " ' * - etc.)
# are uniformly replaced with spaces, then tokens are joined with OR.
# Manually enumerating operators is error-prone (e.g., a period in "P."
# triggers "fts5: syntax error near ."), so we simply strip everything
# that isn't \w or \s.
_FTS5_SPECIAL = re.compile(r"[^\w\s]", re.UNICODE)

__all__ = ["sanitize_fts_query"]


def sanitize_fts_query(query: str) -> str:
    """Strip FTS5 operators and join tokens with OR for recall-friendly BM25.

    FTS5 treats whitespace as implicit AND, which would require every query
    token to appear in a fact (too strict for QA). Joining tokens with ``OR``
    gives the intended any-token-matches behaviour.
    """
    cleaned = _FTS5_SPECIAL.sub(" ", query or "")
    tokens = cleaned.split()
    if not tokens:
        return ""
    return " OR ".join(tokens)
