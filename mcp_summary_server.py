#!/usr/bin/env python3
"""MCP server that periodically summarizes the local chat into a file."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests


PROTOCOL_VERSION = "2024-11-05"

SUMMARY_TOOL = {
    "name": "summarize_dialog",
    "description": "Делает краткое summary диалога через LLM и сохраняет его в отдельный JSON-файл.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "api_url": {
                "type": "string",
                "description": "URL /v1/chat/completions локального LLM сервера.",
            },
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["role", "content"],
                    "additionalProperties": True,
                },
                "description": "Сообщения диалога для учета в summary.",
            },
            "previous_summary": {
                "type": "string",
                "default": "",
                "description": "Предыдущее summary, если оно уже есть.",
            },
            "output_file": {
                "type": "string",
                "description": "Файл, куда сохранить summary.",
            },
            "current_user": {
                "type": "string",
                "default": "default",
                "description": "Активный пользователь диалога.",
            },
            "context_strategy": {
                "type": "string",
                "default": "recent",
                "description": "Активная стратегия контекста клиента.",
            },
        },
        "required": ["api_url", "messages", "output_file"],
        "additionalProperties": False,
    },
}


def validate_messages(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("messages должен быть массивом")
    messages: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if isinstance(role, str) and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("нет сообщений для summary")
    return messages


def format_messages(messages: List[Dict[str, str]]) -> str:
    return "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )


def load_existing_summary(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if isinstance(value, dict) and isinstance(value.get("summary"), str):
        return value["summary"]
    return ""


def summarize_dialog(arguments: Dict[str, Any]) -> Dict[str, Any]:
    api_url = arguments.get("api_url")
    output_file = arguments.get("output_file")
    if not isinstance(api_url, str) or not api_url.strip():
        raise ValueError("api_url должен быть строкой")
    if not isinstance(output_file, str) or not output_file.strip():
        raise ValueError("output_file должен быть строкой")

    output_path = Path(output_file).expanduser().resolve()
    messages = validate_messages(arguments.get("messages"))
    previous_summary = arguments.get("previous_summary")
    if not isinstance(previous_summary, str) or not previous_summary.strip():
        previous_summary = load_existing_summary(output_path)

    system_prompt = (
        "Ты поддерживаешь компактное summary диалога пользователя с локальным ассистентом. "
        "Обнови summary так, чтобы оно сохраняло: текущие цели, важные решения, ограничения, "
        "открытые вопросы, технические детали и последние существенные шаги. "
        "Не добавляй выдуманных фактов. Пиши кратко, на русском языке."
    )
    user_prompt = (
        "Предыдущее summary:\n"
        f"{previous_summary or '(пусто)'}\n\n"
        "Новые/актуальные сообщения:\n"
        f"{format_messages(messages)}\n\n"
        "Верни только обновленное summary обычным текстом."
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 1200,
        "temperature": 0.1,
    }

    response = requests.post(api_url, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    summary = result["choices"][0]["message"]["content"].strip()
    if not summary:
        raise ValueError("LLM вернула пустое summary")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "current_user": str(arguments.get("current_user", "default")),
        "context_strategy": str(arguments.get("context_strategy", "recent")),
        "message_count": len(messages),
        "summary": summary,
    }
    output_path.write_text(
        json.dumps(saved, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "content": [
            {
                "type": "text",
                "text": f"Summary сохранено: {output_path}",
            }
        ],
        "structuredContent": {
            "output_file": str(output_path),
            "message_count": len(messages),
            "updated_at": saved["updated_at"],
        },
    }


def make_response(message: Dict[str, Any]) -> Dict[str, Any] | None:
    request_id = message.get("id")
    method = message.get("method")
    if request_id is None:
        return None

    try:
        if method == "initialize":
            result: Dict[str, Any] = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "periodic-dialog-summary", "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {"tools": [SUMMARY_TOOL]}
        elif method == "tools/call":
            params = message.get("params") or {}
            if params.get("name") != SUMMARY_TOOL["name"]:
                raise ValueError(f"Неизвестный инструмент: {params.get('name')!r}")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ValueError("arguments должен быть JSON-объектом")
            result = summarize_dialog(arguments)
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except (ValueError, TypeError, KeyError, IndexError) as error:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": str(error)},
        }
    except requests.RequestException as error:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": f"Не удалось сделать summary: {error}"}],
                "isError": True,
            },
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
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {error}"},
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
