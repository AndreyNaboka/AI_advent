from __future__ import annotations

from pathlib import Path

from chunkers.base import ChunkingConfig, RawBlock, finalize_blocks
from utils.token_count import count_tokens


def chunk_table(path: Path, config: ChunkingConfig) -> list[RawBlock]:
    import pandas as pd

    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    columns = [str(c) for c in df.columns]
    blocks: list[RawBlock] = []
    current_rows: list[str] = []
    current_tokens = 0

    for _, row in df.iterrows():
        row_text = " | ".join(f"{col}: {row[col]}" for col in columns)
        row_tokens = count_tokens(row_text)
        if current_rows and current_tokens + row_tokens > config.target_chunk_tokens:
            blocks.append(_table_block(path, columns, current_rows))
            current_rows = []
            current_tokens = 0
        current_rows.append(row_text)
        current_tokens += row_tokens

    if current_rows:
        blocks.append(_table_block(path, columns, current_rows))
    return finalize_blocks(path, blocks, config)


def _table_block(path: Path, columns: list[str], rows: list[str]) -> RawBlock:
    return RawBlock(
        text="Table: " + path.name + "\n\nRows:\n" + "\n".join(rows),
        metadata={"content_type": "table", "table_name": path.name, "columns": columns},
    )
