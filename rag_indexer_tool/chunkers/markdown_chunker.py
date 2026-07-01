from __future__ import annotations

import re
from pathlib import Path

from chunkers.base import ChunkingConfig, RawBlock, finalize_blocks, normalize_text

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def chunk_markdown(path: Path, config: ChunkingConfig) -> list[RawBlock]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    heading_stack: list[str] = []
    current: list[str] = []
    current_heading: list[str] = []
    blocks: list[RawBlock] = []

    def flush():
        text = normalize_text("\n".join(current))
        if text:
            blocks.append(RawBlock(text=text, metadata={"content_type": "markdown", "heading_path": list(current_heading)}))

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            flush()
            current = []
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(title)
            current_heading = list(heading_stack)
            current.append(line)
        else:
            current.append(line)
    flush()
    return finalize_blocks(path, blocks, config)
