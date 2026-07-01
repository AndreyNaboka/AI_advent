from __future__ import annotations

import re
from pathlib import Path

from chunkers.base import ChunkingConfig, RawBlock, finalize_blocks, normalize_text

LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".cs": "csharp",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c_header",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
}

SYMBOL_RE = re.compile(
    r"^\s*(?:(class|def)\s+([A-Za-z_][\w]*)|(?:function)\s+([A-Za-z_][\w]*)|(?:public|private|protected)?\s*(?:static\s+)?(?:[\w:<>,\[\]\*&]+\s+)+([A-Za-z_][\w]*)\s*\()"
)


def chunk_code(path: Path, config: ChunkingConfig) -> list[RawBlock]:
    language = LANG_BY_EXT.get(path.suffix.lower(), "unknown")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks: list[RawBlock] = []
    current: list[str] = []
    symbol_type = None
    symbol_name = None

    def flush():
        text = normalize_text("\n".join(current))
        if text:
            blocks.append(
                RawBlock(
                    text=text,
                    metadata={
                        "content_type": "code",
                        "language": language,
                        "symbol_type": symbol_type,
                        "symbol_name": symbol_name,
                    },
                )
            )

    for line in lines:
        match = SYMBOL_RE.match(line)
        if match and current:
            flush()
            current = []
        if match:
            if match.group(1):
                symbol_type = "class" if match.group(1) == "class" else "function"
                symbol_name = match.group(2)
            elif match.group(3):
                symbol_type = "function"
                symbol_name = match.group(3)
            elif match.group(4):
                symbol_type = "function"
                symbol_name = match.group(4)
        current.append(line)
    flush()
    return finalize_blocks(path, blocks, config)
