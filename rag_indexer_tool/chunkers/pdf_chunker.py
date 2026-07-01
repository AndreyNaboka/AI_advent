from __future__ import annotations

from pathlib import Path

from chunkers.base import ChunkingConfig, RawBlock, finalize_blocks, normalize_text


def chunk_pdf(path: Path, config: ChunkingConfig) -> list[RawBlock]:
    import fitz

    blocks: list[RawBlock] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = normalize_text(page.get_text("text"))
            if text:
                blocks.append(
                    RawBlock(
                        text=text,
                        metadata={"content_type": "pdf", "page_start": page_index, "page_end": page_index},
                    )
                )
    return finalize_blocks(path, blocks, config)
