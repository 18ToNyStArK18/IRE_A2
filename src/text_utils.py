"""Minimal, language-agnostic tokenization shared by indexing and querying."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]
