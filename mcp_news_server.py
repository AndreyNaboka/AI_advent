#!/usr/bin/env python3
"""Minimal MCP server that exposes current world news over stdio."""

import html
import json
import sys
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List

import requests


PROTOCOL_VERSION = "2024-11-05"
NEWS_URLS = {
    "ru": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=ru&gl=RU&ceid=RU:ru",
    "en": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en",
}

NEWS_TOOL = {
    "name": "get_world_news",
    "description": "Возвращает свежие заголовки мировых новостей из Google News RSS.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
                "description": "Количество новостей.",
            },
            "language": {
                "type": "string",
                "enum": ["ru", "en"],
                "default": "ru",
                "description": "Язык ленты новостей.",
            },
        },
        "additionalProperties": False,
    },
}


def fetch_world_news(limit: int = 5, language: str = "ru") -> List[Dict[str, str]]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("limit должен быть целым числом от 1 до 20")
    if language not in NEWS_URLS:
        raise ValueError("language должен быть 'ru' или 'en'")

    response = requests.get(
        NEWS_URLS[language],
        headers={"User-Agent": "AI-Advent-MCP-News/1.0"},
        timeout=15,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)

    articles: List[Dict[str, str]] = []
    for item in root.findall("./channel/item")[:limit]:
        published = item.findtext("pubDate", default="").strip()
        if published:
            try:
                published = parsedate_to_datetime(published).isoformat()
            except (TypeError, ValueError):
                pass
        source_node = item.find("source")
        articles.append(
            {
                "title": html.unescape(item.findtext("title", default="").strip()),
                "url": item.findtext("link", default="").strip(),
                "source": (
                    html.unescape((source_node.text or "").strip())
                    if source_node is not None
                    else ""
                ),
                "published_at": published,
            }
        )
    return articles


def tool_result(articles: List[Dict[str, str]]) -> Dict[str, Any]:
    lines = []
    for number, article in enumerate(articles, 1):
        details = " — ".join(
            value for value in (article["source"], article["published_at"]) if value
        )
        lines.append(
            f"{number}. {article['title']}"
            + (f" ({details})" if details else "")
            + f"\n   {article['url']}"
        )
    return {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "structuredContent": {"articles": articles},
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
                "serverInfo": {"name": "world-news", "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {"tools": [NEWS_TOOL]}
        elif method == "tools/call":
            params = message.get("params") or {}
            if params.get("name") != NEWS_TOOL["name"]:
                raise ValueError(f"Неизвестный инструмент: {params.get('name')!r}")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ValueError("arguments должен быть JSON-объектом")
            unknown = set(arguments) - {"limit", "language"}
            if unknown:
                raise ValueError(f"Неизвестные аргументы: {', '.join(sorted(unknown))}")
            result = tool_result(fetch_world_news(**arguments))
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except (ValueError, TypeError) as error:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": str(error)},
        }
    except (requests.RequestException, ET.ParseError) as error:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": f"Не удалось получить новости: {error}"}],
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
