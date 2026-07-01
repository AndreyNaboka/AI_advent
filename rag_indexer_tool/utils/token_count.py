from __future__ import annotations

import re
from functools import lru_cache


@lru_cache(maxsize=1)
def _encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    enc = _encoding()
    if enc is not None:
        return len(enc.encode(text))
    return len(re.findall(r"\S+", text))


def token_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)
