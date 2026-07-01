from __future__ import annotations

from pathlib import Path

from chunkers.base import ChunkingConfig, RawBlock, finalize_blocks, normalize_text


def chunk_txt(path: Path, config: ChunkingConfig) -> list[RawBlock]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [RawBlock(text=p, metadata={"content_type": "text"}) for p in normalize_text(text).split("\n\n") if p.strip()]
    return finalize_blocks(path, blocks, config)
