from __future__ import annotations

from pathlib import Path

from chunkers.base import ChunkingConfig, RawBlock, finalize_blocks, normalize_text


def chunk_docx(path: Path, config: ChunkingConfig) -> list[RawBlock]:
    from docx import Document

    doc = Document(path)
    blocks: list[RawBlock] = []
    current: list[str] = []
    heading_stack: list[str] = []
    current_heading: list[str] = []

    def flush():
        text = normalize_text("\n\n".join(current))
        if text:
            blocks.append(RawBlock(text=text, metadata={"content_type": "docx", "heading_path": list(current_heading)}))

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower()
        if style.startswith("heading"):
            flush()
            current = []
            try:
                level = int(style.split()[-1])
            except Exception:
                level = 1
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(text)
            current_heading = list(heading_stack)
        current.append(text)
    flush()
    return finalize_blocks(path, blocks, config)
