#!/usr/bin/env python3
"""Read-only MCP server backed by a local JSON CRM fixture."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "2024-11-05"
ROOT = Path(__file__).resolve().parent
DEFAULT_CRM_FILE = ROOT / "support_data" / "crm.json"

TOOLS = [
    {
        "name": "get_user",
        "description": "Возвращает профиль пользователя поддержки по user_id.",
        "inputSchema": {
            "type": "object",
            "properties": {"user_id": {"type": "string", "minLength": 1}},
            "required": ["user_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_ticket_context",
        "description": "Возвращает тикет вместе с профилем связанного пользователя.",
        "inputSchema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string", "minLength": 1}},
            "required": ["ticket_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_tickets",
        "description": "Возвращает краткий список тестовых тикетов.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
            },
            "additionalProperties": False,
        },
    },
]


def crm_file() -> Path:
    configured = os.environ.get("AI_ADVENT_CRM_FILE")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_CRM_FILE


def load_crm() -> dict[str, Any]:
    path = crm_file()
    if not path.is_file():
        raise ValueError(f"CRM JSON не найден: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"CRM JSON повреждён: {error}") from error
    if not isinstance(data.get("users"), list) or not isinstance(data.get("tickets"), list):
        raise ValueError("CRM JSON должен содержать массивы users и tickets")
    return data


def require_arguments(arguments: dict[str, Any], allowed: set[str], required: set[str]) -> None:
    unknown = set(arguments) - allowed
    if unknown:
        raise ValueError(f"Неизвестные аргументы: {', '.join(sorted(unknown))}")
    missing = required - set(arguments)
    if missing:
        raise ValueError(f"Отсутствуют аргументы: {', '.join(sorted(missing))}")
    for key in required:
        if not isinstance(arguments[key], str) or not arguments[key].strip():
            raise ValueError(f"{key} должен быть непустой строкой")


def find_by_id(items: list[dict[str, Any]], key: str, value: str, entity: str) -> dict[str, Any]:
    normalized = value.strip().casefold()
    for item in items:
        if str(item.get(key, "")).casefold() == normalized:
            return item
    raise ValueError(f"{entity} {value!r} не найден")


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    data = load_crm()
    if name == "get_user":
        require_arguments(arguments, {"user_id"}, {"user_id"})
        structured: Any = find_by_id(data["users"], "id", arguments["user_id"], "Пользователь")
        text = json.dumps(structured, ensure_ascii=False, indent=2)
    elif name == "get_ticket_context":
        require_arguments(arguments, {"ticket_id"}, {"ticket_id"})
        ticket = find_by_id(data["tickets"], "id", arguments["ticket_id"], "Тикет")
        user = find_by_id(data["users"], "id", str(ticket.get("user_id", "")), "Пользователь")
        structured = {"ticket": ticket, "user": user}
        text = json.dumps(structured, ensure_ascii=False, indent=2)
    elif name == "list_tickets":
        require_arguments(arguments, {"status", "limit"}, set())
        limit = arguments.get("limit", 20)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit должен быть целым числом от 1 до 100")
        status = arguments.get("status")
        if status is not None and (not isinstance(status, str) or not status.strip()):
            raise ValueError("status должен быть непустой строкой")
        tickets = [
            {
                "id": item.get("id"),
                "user_id": item.get("user_id"),
                "status": item.get("status"),
                "category": item.get("category"),
                "subject": item.get("subject"),
            }
            for item in data["tickets"]
            if status is None or str(item.get("status", "")).casefold() == status.casefold()
        ][:limit]
        structured = {"tickets": tickets, "count": len(tickets)}
        text = json.dumps(structured, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Неизвестный инструмент: {name!r}")
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
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
                "serverInfo": {"name": "json-crm", "version": "1.0.0"},
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
    except (ValueError, TypeError, OSError) as error:
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
