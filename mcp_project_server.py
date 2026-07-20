#!/usr/bin/env python3
"""Read-only MCP server exposing useful context from this Git project."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "2024-11-05"
PROJECT_ROOT = Path(__file__).resolve().parent

TOOLS = [
    {
        "name": "git_branch",
        "description": "Возвращает текущую git-ветку проекта.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_files",
        "description": "Возвращает список отслеживаемых Git файлов проекта.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100}
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "git_diff",
        "description": "Возвращает сокращённый diff незакоммиченных изменений.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_chars": {"type": "integer", "minimum": 100, "maximum": 50000, "default": 12000}
            },
            "additionalProperties": False,
        },
    },
]


def run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    return completed.stdout.strip()


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "git_branch":
        if arguments:
            raise ValueError("git_branch не принимает аргументы")
        branch = run_git("branch", "--show-current") or "HEAD (detached)"
        data: Any = {"branch": branch}
        text = branch
    elif name == "list_files":
        unknown = set(arguments) - {"limit"}
        if unknown:
            raise ValueError(f"Неизвестные аргументы: {', '.join(sorted(unknown))}")
        limit = arguments.get("limit", 100)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit должен быть целым числом от 1 до 500")
        files = run_git("ls-files").splitlines()[:limit]
        data = {"files": files, "count": len(files), "limit": limit}
        text = "\n".join(files)
    elif name == "git_diff":
        unknown = set(arguments) - {"max_chars"}
        if unknown:
            raise ValueError(f"Неизвестные аргументы: {', '.join(sorted(unknown))}")
        max_chars = arguments.get("max_chars", 12000)
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 100 <= max_chars <= 50000:
            raise ValueError("max_chars должен быть целым числом от 100 до 50000")
        diff = run_git("diff", "--no-ext-diff")
        truncated = len(diff) > max_chars
        diff = diff[:max_chars]
        data = {"diff": diff, "truncated": truncated}
        text = diff or "Нет незакоммиченных изменений в отслеживаемых файлах."
    else:
        raise ValueError(f"Неизвестный инструмент: {name!r}")
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": data,
    }


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
                "serverInfo": {"name": "ai-advent-project", "version": "1.0.0"},
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
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except (ValueError, TypeError, subprocess.SubprocessError) as error:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
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
