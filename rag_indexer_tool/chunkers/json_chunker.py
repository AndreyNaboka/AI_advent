from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from chunkers.base import ChunkingConfig, RawBlock, finalize_blocks


def chunk_json_or_yaml(path: Path, config: ChunkingConfig) -> list[RawBlock]:
    is_yaml = path.suffix.lower() in {".yaml", ".yml"}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        data = yaml.safe_load(fh) if is_yaml else json.load(fh)
    content_type = "yaml" if is_yaml else "json"
    blocks: list[RawBlock] = []

    if isinstance(data, dict):
        for key, value in data.items():
            blocks.append(_block(content_type, [str(key)], value))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            blocks.append(_block(content_type, [str(index)], value))
    else:
        blocks.append(_block(content_type, [], data))
    return finalize_blocks(path, blocks, config)


def _block(content_type: str, path_parts: list[str], value: Any) -> RawBlock:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    key = "yaml_path" if content_type == "yaml" else "json_path"
    return RawBlock(text=text, metadata={"content_type": content_type, key: path_parts})
