from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.hashing import sha256_text
from utils.token_count import count_tokens, token_words


@dataclass(frozen=True)
class ChunkingConfig:
    target_chunk_tokens: int = 500
    min_chunk_tokens: int = 150
    max_chunk_tokens: int = 900
    overlap_tokens: int = 80
    code_overlap_tokens: int = 30


@dataclass
class RawBlock:
    text: str
    metadata: dict[str, Any]


SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_recursive(text: str, config: ChunkingConfig, overlap_tokens: int | None = None) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if count_tokens(text) <= config.max_chunk_tokens:
        return [text]

    overlap = config.overlap_tokens if overlap_tokens is None else overlap_tokens

    def hard_word_split(value: str) -> list[str]:
        words = token_words(value)
        if not words:
            return []
        step = max(1, config.target_chunk_tokens - overlap)
        chunks = []
        for start in range(0, len(words), step):
            part = " ".join(words[start : start + config.target_chunk_tokens])
            if part:
                chunks.append(part)
        return chunks

    def split_by_separator(value: str, sep_index: int) -> list[str]:
        if count_tokens(value) <= config.max_chunk_tokens:
            return [value.strip()]
        if sep_index >= len(SEPARATORS):
            return hard_word_split(value)

        sep = SEPARATORS[sep_index]
        parts = [p.strip() for p in value.split(sep) if p.strip()]
        if len(parts) <= 1:
            return split_by_separator(value, sep_index + 1)

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        joiner = sep if sep != " " else " "

        for part in parts:
            part_tokens = count_tokens(part)
            if part_tokens > config.max_chunk_tokens:
                if current:
                    chunks.append(joiner.join(current).strip())
                    current = []
                    current_tokens = 0
                chunks.extend(split_by_separator(part, sep_index + 1))
                continue
            if current and current_tokens + part_tokens > config.target_chunk_tokens:
                chunks.append(joiner.join(current).strip())
                current = []
                current_tokens = 0
            current.append(part)
            current_tokens += part_tokens

        if current:
            chunks.append(joiner.join(current).strip())
        return chunks

    return [c for c in split_by_separator(text, 0) if c]


def merge_small_blocks(blocks: list[RawBlock], config: ChunkingConfig) -> list[RawBlock]:
    merged: list[RawBlock] = []
    for block in blocks:
        if not block.text:
            continue
        if (
            merged
            and count_tokens(block.text) < config.min_chunk_tokens
            and _can_merge(merged[-1].metadata, block.metadata)
            and count_tokens(merged[-1].text) + count_tokens(block.text) <= config.max_chunk_tokens
        ):
            merged[-1].text = f"{merged[-1].text}\n\n{block.text}".strip()
            merged[-1].metadata = _merge_metadata(merged[-1].metadata, block.metadata)
        else:
            merged.append(block)
    return merged


def _can_merge(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = ["source_path", "content_type", "language", "symbol_name", "json_path", "yaml_path", "table_name"]
    if any(a.get(k) != b.get(k) for k in keys):
        return False
    if a.get("content_type") == "pdf":
        end = a.get("page_end")
        start = b.get("page_start")
        return end is None or start is None or start - end <= 1
    if a.get("heading_path") != b.get("heading_path") and a.get("content_type") in {"markdown", "docx", "html"}:
        return False
    return True


def _merge_metadata(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    result = dict(a)
    if a.get("page_start") is not None or b.get("page_start") is not None:
        starts = [v for v in [a.get("page_start"), b.get("page_start")] if v is not None]
        ends = [v for v in [a.get("page_end"), b.get("page_end")] if v is not None]
        result["page_start"] = min(starts) if starts else None
        result["page_end"] = max(ends) if ends else None
    return result


def add_context(text: str, path: Path, metadata: dict[str, Any]) -> str:
    prefix = [f"Document: {path.name}"]
    heading_path = metadata.get("heading_path") or []
    if heading_path:
        prefix.append("Section: " + " > ".join(heading_path))
    if metadata.get("content_type") == "table":
        columns = metadata.get("columns") or []
        if columns:
            prefix.append("Columns: " + ", ".join(columns))
    return "\n".join(prefix) + "\n\n" + text.strip()


def finalize_blocks(path: Path, blocks: list[RawBlock], config: ChunkingConfig, context: bool = True) -> list[RawBlock]:
    expanded: list[RawBlock] = []
    for block in blocks:
        overlap = config.code_overlap_tokens if block.metadata.get("content_type") == "code" else config.overlap_tokens
        for part in split_text_recursive(block.text, config, overlap):
            meta = dict(block.metadata)
            text = add_context(part, path, meta) if context else part
            meta["chunk_hash"] = sha256_text(text)
            expanded.append(RawBlock(text=text, metadata=meta))
    return merge_small_blocks(expanded, config)
