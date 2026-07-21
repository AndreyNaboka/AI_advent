#!/usr/bin/env python3
"""RAG support assistant enriched with user/ticket context from a JSON MCP CRM."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from mcp_client import MCPClientError, MCPCRMClient


ROOT = Path(__file__).resolve().parent
DEFAULT_KB_DIR = ROOT / "support_kb"
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_.]+")
SAFE_SIGNAL_KEYS = {
    "failed_attempts", "account_locked", "locked_until", "email_verified",
    "verification_sent", "last_verification_sent_at", "requested_domain",
    "configured_domain", "idp_status",
}


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text) if len(token) > 2]


@dataclass(frozen=True)
class SupportChunk:
    source: str
    section: str
    text: str


@dataclass(frozen=True)
class SupportHit:
    chunk: SupportChunk
    score: float


class SupportRagIndex:
    def __init__(self, kb_dir: Path = DEFAULT_KB_DIR):
        self.kb_dir = kb_dir.resolve()
        self.chunks = self._load_chunks()
        self.counts = [Counter(tokenize(f"{chunk.section} {chunk.text}")) for chunk in self.chunks]
        self.document_frequency = Counter(term for counts in self.counts for term in counts)

    def _load_chunks(self) -> list[SupportChunk]:
        if not self.kb_dir.is_dir():
            return []
        chunks: list[SupportChunk] = []
        for path in sorted(self.kb_dir.rglob("*.md")):
            section = "Документ"
            buffer: list[str] = []

            def flush() -> None:
                text = "\n".join(buffer).strip()
                if text:
                    chunks.append(
                        SupportChunk(path.relative_to(self.kb_dir).as_posix(), section, text)
                    )
                buffer.clear()

            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
                if heading:
                    flush()
                    section = heading.group(1)
                else:
                    buffer.append(line)
            flush()
        return chunks

    def search(self, query: str, top_k: int = 3) -> list[SupportHit]:
        terms = set(tokenize(query))
        total = len(self.chunks)
        if not terms or not total:
            return []
        hits: list[SupportHit] = []
        for chunk, counts in zip(self.chunks, self.counts):
            score = 0.0
            section_terms = set(tokenize(chunk.section))
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency[term]
                inverse_frequency = math.log(1 + (total - df + 0.5) / (df + 0.5))
                score += inverse_frequency * (1 + frequency / (frequency + 1))
                if term in section_terms:
                    score += 2.0
            if score:
                hits.append(SupportHit(chunk, score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]


@dataclass(frozen=True)
class SupportResponse:
    ticket_id: str
    user_id: str
    answer: str
    crm_context: dict[str, Any]
    sources: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_text(self) -> str:
        sources = "\n".join(
            f"- support_kb/{item['source']} — {item['section']} (score={item['score']:.2f})"
            for item in self.sources
        ) or "- нет релевантных источников"
        return (
            f"Тикет: {self.ticket_id}; пользователь: {self.user_id}\n\n"
            f"Контекст CRM через MCP:\n{json.dumps(self.crm_context, ensure_ascii=False, indent=2)}\n\n"
            f"Ответ поддержки:\n{self.answer}\n\n"
            f"Источники RAG:\n{sources}"
        )


class SupportError(RuntimeError):
    pass


class SupportAssistant:
    def __init__(
        self,
        kb_dir: Path = DEFAULT_KB_DIR,
        crm_client: MCPCRMClient | None = None,
    ):
        self.rag = SupportRagIndex(kb_dir)
        self.crm = crm_client or MCPCRMClient()

    def _ticket_context(self, ticket_id: str) -> dict[str, Any]:
        try:
            if not self.crm.is_running:
                self.crm.start()
            result = self.crm.call_tool("get_ticket_context", {"ticket_id": ticket_id})
        except (MCPClientError, OSError) as error:
            raise SupportError(f"Не удалось получить контекст тикета через MCP: {error}") from error
        context = result.get("structuredContent")
        if not isinstance(context, dict):
            raise SupportError("MCP CRM вернул некорректный контекст")
        return context

    @staticmethod
    def _safe_context(ticket: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
        signals = ticket.get("signals") if isinstance(ticket.get("signals"), dict) else {}
        return {
            "user": {
                "id": user.get("id"),
                "plan": user.get("plan"),
                "account_status": user.get("account_status"),
                "locale": user.get("locale"),
            },
            "ticket": {
                "id": ticket.get("id"),
                "status": ticket.get("status"),
                "priority": ticket.get("priority"),
                "category": ticket.get("category"),
                "diagnostic_code": ticket.get("diagnostic_code"),
                "signals": {key: value for key, value in signals.items() if key in SAFE_SIGNAL_KEYS},
            },
        }

    def ask(self, ticket_id: str, question: str) -> SupportResponse:
        ticket_id = ticket_id.strip()
        question = question.strip()
        if not ticket_id or not question:
            raise SupportError("Нужны непустые ticket_id и question")
        context = self._ticket_context(ticket_id)
        ticket = context.get("ticket")
        user = context.get("user")
        if not isinstance(ticket, dict) or not isinstance(user, dict):
            raise SupportError("В MCP-контексте отсутствует ticket или user")
        safe_context = self._safe_context(ticket, user)
        query = " ".join(
            [
                question,
                str(ticket.get("subject", "")),
                str(ticket.get("description", "")),
                str(ticket.get("diagnostic_code", "")),
                " ".join(str(tag) for tag in ticket.get("tags", [])),
                json.dumps(safe_context, ensure_ascii=False),
            ]
        )
        hits = self.rag.search(query)
        diagnostic = ticket.get("diagnostic_code") or "не определён"
        if hits:
            primary = hits[0].chunk
            knowledge = " ".join(primary.text.split())
            answer = (
                f"По данным тикета зафиксирована диагностика {diagnostic}. "
                f"Для этого случая база знаний рекомендует: {knowledge}"
            )
        else:
            answer = (
                f"В тикете указан код {diagnostic}, но в FAQ и документации нет "
                "достаточного ответа. Передайте тикет специалисту второй линии."
            )
        sources = [
            {"source": hit.chunk.source, "section": hit.chunk.section, "score": hit.score}
            for hit in hits
        ]
        return SupportResponse(
            ticket_id=str(ticket.get("id", ticket_id)),
            user_id=str(user.get("id", "")),
            answer=answer,
            crm_context=safe_context,
            sources=sources,
        )

    def list_tickets(self) -> list[dict[str, Any]]:
        try:
            if not self.crm.is_running:
                self.crm.start()
            result = self.crm.call_tool("list_tickets", {"limit": 100})
            tickets = result.get("structuredContent", {}).get("tickets", [])
            return tickets if isinstance(tickets, list) else []
        except (MCPClientError, OSError) as error:
            raise SupportError(f"Не удалось получить тикеты через MCP: {error}") from error

    def close(self) -> None:
        self.crm.stop()


def make_http_handler(assistant: SupportAssistant) -> type[BaseHTTPRequestHandler]:
    class SupportHttpHandler(BaseHTTPRequestHandler):
        def _json_response(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json_response(200, {"status": "ok", "rag_chunks": len(assistant.rag.chunks)})
            else:
                self._json_response(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/support":
                self._json_response(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 65536:
                    raise SupportError("Размер JSON должен быть от 1 до 65536 байт")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise SupportError("Тело запроса должно быть JSON-объектом")
                response = assistant.ask(
                    str(payload.get("ticket_id", "")),
                    str(payload.get("question", "")),
                )
                self._json_response(200, response.to_dict())
            except (SupportError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                self._json_response(400, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return SupportHttpHandler


def run_http(assistant: SupportAssistant, host: str, port: int) -> None:
    server = HTTPServer((host, port), make_http_handler(assistant))
    print(f"Support service: http://{host}:{port}")
    print("GET /health; POST /support")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def run_interactive(assistant: SupportAssistant) -> None:
    print("Тестовые тикеты:")
    for ticket in assistant.list_tickets():
        print(f"- {ticket.get('id')}: {ticket.get('subject')}")
    ticket_id = input("Ticket ID: ").strip()
    print("Задавайте вопросы; /ticket <ID> переключает тикет, /exit завершает работу.")
    while True:
        value = input("Вы: ").strip()
        if value in {"/exit", "/quit"}:
            return
        if value.startswith("/ticket "):
            ticket_id = value.split(maxsplit=1)[1].strip()
            print(f"Активный тикет: {ticket_id}")
        elif value:
            try:
                print("\n" + assistant.ask(ticket_id, value).to_text() + "\n")
            except SupportError as error:
                print(f"Ошибка: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG-ассистент службы поддержки + MCP CRM")
    parser.add_argument("--ticket", help="ID тикета для однократного запроса")
    parser.add_argument("--question", help="Вопрос пользователя")
    parser.add_argument("--json", action="store_true", help="Вывести структурированный JSON")
    parser.add_argument("--serve", action="store_true", help="Запустить локальный HTTP-сервис")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assistant = SupportAssistant(args.kb_dir)
    try:
        if args.serve:
            run_http(assistant, args.host, args.port)
            return 0
        if args.ticket or args.question:
            if not args.ticket or not args.question:
                raise SupportError("Для однократного запроса нужны --ticket и --question")
            response = assistant.ask(args.ticket, args.question)
            print(
                json.dumps(response.to_dict(), ensure_ascii=False, indent=2)
                if args.json
                else response.to_text()
            )
            return 0
        run_interactive(assistant)
        return 0
    except SupportError as error:
        print(f"Ошибка поддержки: {error}")
        return 2
    finally:
        assistant.close()


if __name__ == "__main__":
    raise SystemExit(main())
