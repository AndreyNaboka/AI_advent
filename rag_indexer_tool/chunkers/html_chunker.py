from __future__ import annotations

from pathlib import Path

from chunkers.base import ChunkingConfig, RawBlock, finalize_blocks, normalize_text


def chunk_html(path: Path, config: ChunkingConfig) -> list[RawBlock]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    blocks: list[RawBlock] = []
    heading_stack: list[str] = []
    current: list[str] = []
    current_heading: list[str] = []

    def flush():
        text = normalize_text("\n\n".join(current))
        if text:
            blocks.append(RawBlock(text=text, metadata={"content_type": "html", "heading_path": list(current_heading)}))

    body = soup.body or soup
    for element in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "td", "th"], recursive=True):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name and element.name.startswith("h"):
            flush()
            current = []
            level = int(element.name[1])
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(text)
            current_heading = list(heading_stack)
        else:
            current.append(text)
    flush()
    return finalize_blocks(path, blocks, config)
