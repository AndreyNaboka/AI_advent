from __future__ import annotations

from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".pdf",
    ".docx",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".py",
    ".js",
    ".ts",
    ".cs",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".go",
    ".rs",
    ".php",
    ".rb",
}


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path, path.suffix.lower() in SUPPORTED_EXTENSIONS
