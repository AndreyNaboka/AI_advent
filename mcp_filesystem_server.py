#!/usr/bin/env python3
"""Constrained MCP filesystem tools for autonomous project file tasks."""

from __future__ import annotations

import difflib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_VERSION = "2024-11-05"
DEFAULT_ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv",
    "venv312", "node_modules", "dist", "build", "target", "vendor",
    "AI_advent_export", "qdrant_storage", "assistant_outputs",
}
TEXT_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt",
    ".cs", ".php", ".rb", ".swift", ".c", ".cpp", ".h", ".hpp", ".sh",
    ".sql", ".html", ".css", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
}
WRITABLE_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml"}
WRITABLE_DIRS = {"docs", "assistant_outputs", "adrs"}

TOOLS = [
    {
        "name": "list_project_files",
        "description": "Перечисляет текстовые файлы проекта с фильтром расширений.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "extensions": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 300},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "Читает UTF-8 текстовый файл внутри проекта.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "max_chars": {"type": "integer", "minimum": 100, "maximum": 500000, "default": 100000},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_text",
        "description": "Ищет буквальную строку сразу по текстовым файлам проекта.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "case_sensitive": {"type": "boolean", "default": True},
                "extensions": {"type": "array", "items": {"type": "string"}},
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 200},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": "Создаёт или обновляет разрешённый артефакт и возвращает unified diff.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "content": {"type": "string"},
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
]


def project_root() -> Path:
    configured = os.environ.get("AI_ADVENT_FILES_ROOT")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_ROOT


def safe_path(relative_value: str, must_exist: bool = True) -> Path:
    if not isinstance(relative_value, str) or not relative_value.strip():
        raise ValueError("path должен быть непустой строкой")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Разрешены только относительные пути без '..'")
    root = project_root()
    candidate = root.joinpath(relative)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("Путь выходит за пределы проекта") from error
    if must_exist and (not resolved.exists() or not resolved.is_file()):
        raise ValueError(f"Файл не найден: {relative_value}")
    return resolved


def normalize_extensions(raw: Any) -> set[str] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("extensions должен быть массивом строк")
    normalized = {item.casefold() if item.startswith(".") else f".{item.casefold()}" for item in raw}
    if not normalized <= TEXT_SUFFIXES:
        raise ValueError("Запрошено неподдерживаемое расширение")
    return normalized


def iter_text_files(extensions: set[str] | None = None) -> Iterable[Path]:
    root = project_root()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        if extensions is not None and path.suffix.casefold() not in extensions:
            continue
        yield path


def validate_arguments(arguments: dict[str, Any], allowed: set[str], required: set[str]) -> None:
    unknown = set(arguments) - allowed
    if unknown:
        raise ValueError(f"Неизвестные аргументы: {', '.join(sorted(unknown))}")
    missing = required - set(arguments)
    if missing:
        raise ValueError(f"Отсутствуют аргументы: {', '.join(sorted(missing))}")


def list_project_files(arguments: dict[str, Any]) -> dict[str, Any]:
    validate_arguments(arguments, {"extensions", "limit"}, set())
    extensions = normalize_extensions(arguments.get("extensions"))
    limit = arguments.get("limit", 300)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit должен быть целым числом от 1 до 1000")
    root = project_root()
    files = [path.relative_to(root).as_posix() for path in iter_text_files(extensions)][:limit]
    return {"files": files, "count": len(files), "limit": limit}


def read_file(arguments: dict[str, Any]) -> dict[str, Any]:
    validate_arguments(arguments, {"path", "max_chars"}, {"path"})
    max_chars = arguments.get("max_chars", 100000)
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 100 <= max_chars <= 500000:
        raise ValueError("max_chars должен быть целым числом от 100 до 500000")
    path = safe_path(arguments["path"])
    if path.suffix.casefold() not in TEXT_SUFFIXES:
        raise ValueError("Неподдерживаемый тип файла")
    content = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(content) > max_chars
    return {
        "path": path.relative_to(project_root()).as_posix(),
        "content": content[:max_chars],
        "truncated": truncated,
        "size_chars": len(content),
    }


def search_text(arguments: dict[str, Any]) -> dict[str, Any]:
    validate_arguments(
        arguments,
        {"query", "case_sensitive", "extensions", "max_matches"},
        {"query"},
    )
    query = arguments["query"]
    if not isinstance(query, str) or not query or len(query) > 500:
        raise ValueError("query должен быть строкой длиной от 1 до 500")
    case_sensitive = arguments.get("case_sensitive", True)
    if not isinstance(case_sensitive, bool):
        raise ValueError("case_sensitive должен быть boolean")
    max_matches = arguments.get("max_matches", 200)
    if isinstance(max_matches, bool) or not isinstance(max_matches, int) or not 1 <= max_matches <= 1000:
        raise ValueError("max_matches должен быть целым числом от 1 до 1000")
    extensions = normalize_extensions(arguments.get("extensions"))
    needle = query if case_sensitive else query.casefold()
    root = project_root()
    matches: list[dict[str, Any]] = []
    files_scanned = 0
    matching_files: set[str] = set()
    for path in iter_text_files(extensions):
        files_scanned += 1
        relative = path.relative_to(root).as_posix()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            comparable = line if case_sensitive else line.casefold()
            if needle in comparable:
                matches.append({"path": relative, "line": line_number, "text": line.strip()[:500]})
                matching_files.add(relative)
                if len(matches) >= max_matches:
                    return {
                        "query": query,
                        "matches": matches,
                        "matching_files": sorted(matching_files),
                        "files_scanned": files_scanned,
                        "truncated": True,
                    }
    return {
        "query": query,
        "matches": matches,
        "matching_files": sorted(matching_files),
        "files_scanned": files_scanned,
        "truncated": False,
    }


def write_file(arguments: dict[str, Any]) -> dict[str, Any]:
    validate_arguments(arguments, {"path", "content", "dry_run"}, {"path", "content"})
    relative = Path(arguments["path"])
    content = arguments["content"]
    dry_run = arguments.get("dry_run", False)
    if not isinstance(content, str) or len(content) > 1_000_000:
        raise ValueError("content должен быть строкой не длиннее 1000000 символов")
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run должен быть boolean")
    if relative.suffix.casefold() not in WRITABLE_SUFFIXES:
        raise ValueError("Этот тип файла запрещён для записи")
    if not relative.parts or (
        relative.parts[0] not in WRITABLE_DIRS
        and not (len(relative.parts) == 1 and relative.suffix.casefold() == ".md")
    ):
        raise ValueError("Запись разрешена только в docs/, assistant_outputs/, adrs/ и корневые Markdown")
    path = safe_path(relative.as_posix(), must_exist=False)
    old_content = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    changed = old_content != content
    diff = "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"a/{relative.as_posix()}",
            tofile=f"b/{relative.as_posix()}",
        )
    ) if changed else ""
    if changed and not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    return {
        "path": relative.as_posix(),
        "changed": changed,
        "dry_run": dry_run,
        "diff": diff,
        "size_chars": len(content),
    }


def tool_result(structured: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(structured, ensure_ascii=False, indent=2)}],
        "structuredContent": structured,
    }


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    functions = {
        "list_project_files": list_project_files,
        "read_file": read_file,
        "search_text": search_text,
        "write_file": write_file,
    }
    function = functions.get(name)
    if function is None:
        raise ValueError(f"Неизвестный инструмент: {name!r}")
    return tool_result(function(arguments))


def make_response(message: dict[str, Any]) -> dict[str, Any] | None:
    request_id = message.get("id")
    if request_id is None:
        return None
    method = message.get("method")
    try:
        if method == "initialize":
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "project-files", "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            params = message.get("params") or {}
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ValueError("arguments должен быть JSON-объектом")
            result = call_tool(str(params.get("name", "")), arguments)
        else:
            return {
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except (ValueError, TypeError, OSError) as error:
        return {
            "jsonrpc": "2.0", "id": request_id,
            "error": {"code": -32602, "message": str(error)},
        }


def main() -> None:
    for line in sys.stdin:
        try:
            message = json.loads(line)
            response = make_response(message)
            if response is not None:
                print(json.dumps(response, ensure_ascii=False), flush=True)
        except (json.JSONDecodeError, TypeError) as error:
            print(
                json.dumps(
                    {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(error)}},
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
