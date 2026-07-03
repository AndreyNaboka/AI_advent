#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_KB_DIR = ROOT / "mini_chat_kb"
DEFAULT_STATE_FILE = ROOT / "mini_chat_state.json"
DEFAULT_SCENARIOS_FILE = ROOT / "mini_chat_scenarios.json"
DEFAULT_REPORT_FILE = ROOT / "mini_chat_scenario_report.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokenize(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9_.:-]+", text)
        if len(token) > 2
    }


@dataclass
class SourceChunk:
    source_path: str
    section: str
    chunk_id: int
    text: str


@dataclass
class RagHit:
    chunk: SourceChunk
    score: float
    overlap_terms: list[str]


@dataclass
class TaskState:
    goal: str = ""
    clarified: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    terms: dict[str, str] = field(default_factory=dict)
    updated_at: str = field(default_factory=now_iso)

    def update_from_user(self, message: str) -> None:
        normalized = message.casefold()
        goal_match = re.search(r"(?:цель|goal)\s*[:=-]\s*(.+)", message, re.IGNORECASE)
        if goal_match:
            self.goal = goal_match.group(1).strip()
        elif not self.goal and any(word in normalized for word in ("хочу", "нужно", "надо", "задача")):
            self.goal = message.strip()

        term_match = re.search(
            r"(?:термин|term)\s+([^:=]+)\s*[:=]\s*(.+)", message, re.IGNORECASE
        )
        if term_match:
            key = term_match.group(1).strip()
            value = term_match.group(2).strip()
            if key and value:
                self.terms[key] = value

        if any(word in normalized for word in ("уточняю", "важно", "запомни", "добавь")):
            self._append_unique(self.clarified, message.strip())

        if any(
            word in normalized
            for word in ("огранич", "только", "без ", "нельзя", "не используй", "коротко", "источник")
        ):
            self._append_unique(self.constraints, message.strip())

        for code in re.findall(r"\b[A-ZА-ЯЁ]{2,}[A-ZА-ЯЁ0-9_.:-]*\b", message):
            self.terms.setdefault(code, "упомянутый пользователем термин")

        self.updated_at = now_iso()

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        if value and value not in items:
            items.append(value)


class MarkdownRagStore:
    def __init__(self, kb_dir: Path) -> None:
        self.kb_dir = kb_dir
        self.chunks = self._load_chunks()

    def _load_chunks(self) -> list[SourceChunk]:
        chunks: list[SourceChunk] = []
        chunk_id = 0
        for path in sorted(self.kb_dir.rglob("*.md")):
            current_section = "root"
            buffer: list[str] = []

            def flush() -> None:
                nonlocal chunk_id, buffer
                text = "\n".join(buffer).strip()
                if text:
                    chunks.append(
                        SourceChunk(
                            source_path=str(path.relative_to(ROOT)),
                            section=current_section,
                            chunk_id=chunk_id,
                            text=text,
                        )
                    )
                    chunk_id += 1
                buffer = []

            for line in path.read_text(encoding="utf-8").splitlines():
                heading = re.match(r"^(#{1,6})\s+(.+)$", line)
                if heading:
                    flush()
                    current_section = heading.group(2).strip()
                    continue
                buffer.append(line)
            flush()
        return chunks

    def search(self, query: str, top_k: int = 4, min_score: float = 0.03) -> list[RagHit]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        hits: list[RagHit] = []
        for chunk in self.chunks:
            chunk_terms = tokenize(f"{chunk.source_path} {chunk.section} {chunk.text}")
            overlap = sorted(query_terms & chunk_terms)
            if not overlap:
                continue
            coverage = len(overlap) / min(len(query_terms), 8)
            density = len(overlap) / max(1, len(chunk_terms))
            exact_code_boost = 0.15 if any("-" in term for term in overlap) else 0.0
            score = min(1.0, (0.80 * coverage) + (0.15 * density) + exact_code_boost)
            if score >= min_score:
                hits.append(RagHit(chunk=chunk, score=score, overlap_terms=overlap))

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


class MiniRagChat:
    def __init__(
        self,
        kb_dir: Path = DEFAULT_KB_DIR,
        state_file: Path = DEFAULT_STATE_FILE,
        top_k: int = 4,
        min_score: float = 0.03,
    ) -> None:
        self.state_file = state_file
        self.top_k = top_k
        self.min_score = min_score
        self.rag = MarkdownRagStore(kb_dir)
        self.history: list[dict[str, str]] = []
        self.task_state = TaskState()
        self.load_state()

    def load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.history = [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in data.get("history", [])
            if isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
        ]
        task = data.get("task_state", {})
        if isinstance(task, dict):
            self.task_state = TaskState(
                goal=str(task.get("goal", "")),
                clarified=[str(item) for item in task.get("clarified", []) if str(item).strip()],
                constraints=[str(item) for item in task.get("constraints", []) if str(item).strip()],
                terms={
                    str(key): str(value)
                    for key, value in task.get("terms", {}).items()
                }
                if isinstance(task.get("terms"), dict)
                else {},
                updated_at=str(task.get("updated_at") or now_iso()),
            )

    def save_state(self) -> None:
        payload = {
            "history": self.history,
            "task_state": asdict(self.task_state),
        }
        self.state_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def ask(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        self.task_state.update_from_user(message)
        rag_query = self.build_rag_query(message)
        hits = self.rag.search(rag_query, top_k=self.top_k, min_score=self.min_score)
        answer = self.build_answer(message, hits)
        self.history.append({"role": "assistant", "content": answer})
        self.save_state()
        return answer

    def build_rag_query(self, message: str) -> str:
        state_parts = [
            self.task_state.goal,
            " ".join(self.task_state.constraints[-4:]),
            " ".join(self.task_state.terms.keys()),
        ]
        return "\n".join(part for part in [message, *state_parts] if part)

    def build_answer(self, message: str, hits: list[RagHit]) -> str:
        if not hits:
            return self.format_response(
                "Не знаю: в локальной базе не найден подходящий контекст. "
                "Уточните термин, проект или добавьте документ в базу.",
                [],
                [],
            )

        selected_sentences = self.extract_supporting_sentences(message, hits)
        answer = " ".join(selected_sentences[:4]).strip()
        if not answer:
            answer = "Найден релевантный контекст, но в нем нет короткого фрагмента для прямого ответа."

        if self.task_state.goal:
            answer += f"\n\nПамять задачи: цель диалога - {self.task_state.goal}"
        if self.task_state.constraints:
            answer += "\nОграничения: " + "; ".join(self.task_state.constraints[-3:])
        if self.task_state.terms:
            answer += "\nТермины: " + ", ".join(sorted(self.task_state.terms.keys())[:8])

        quotes = [self.short_quote(hit.chunk.text) for hit in hits]
        return self.format_response(answer, hits, quotes)

    def extract_supporting_sentences(self, message: str, hits: list[RagHit]) -> list[str]:
        query_terms = tokenize(self.build_rag_query(message))
        scored: list[tuple[float, str]] = []
        for hit in hits:
            for sentence in re.split(r"(?<=[.!?])\s+", " ".join(hit.chunk.text.split())):
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_terms = tokenize(sentence)
                overlap = query_terms & sentence_terms
                score = len(overlap) / max(1, len(query_terms))
                if score > 0:
                    scored.append((score, sentence))
        scored.sort(key=lambda item: item[0], reverse=True)

        result: list[str] = []
        for _, sentence in scored:
            if sentence not in result:
                result.append(sentence)
        return result

    @staticmethod
    def short_quote(text: str, limit: int = 220) -> str:
        quote = " ".join(text.split())
        if len(quote) <= limit:
            return quote
        return quote[:limit].rstrip() + "..."

    def format_response(
        self,
        answer: str,
        hits: list[RagHit],
        quotes: list[str],
    ) -> str:
        if hits:
            sources = [
                (
                    f"{index}. source={hit.chunk.source_path}; "
                    f"section={hit.chunk.section}; chunk_id={hit.chunk.chunk_id}; "
                    f"score={hit.score:.3f}"
                )
                for index, hit in enumerate(hits, 1)
            ]
            quote_lines = [f"{index}. \"{quote}\"" for index, quote in enumerate(quotes, 1)]
        else:
            sources = ["нет источников выше порога релевантности"]
            quote_lines = ["нет цитат выше порога релевантности"]

        return (
            "Ответ:\n"
            f"{answer.strip()}\n\n"
            "Источники:\n"
            + "\n".join(sources)
            + "\n\nЦитаты:\n"
            + "\n".join(quote_lines)
        )

    def task_state_view(self) -> str:
        return json.dumps(asdict(self.task_state), ensure_ascii=False, indent=2)


def run_cli(args: argparse.Namespace) -> None:
    chat = MiniRagChat(
        kb_dir=Path(args.kb_dir),
        state_file=Path(args.state_file),
        top_k=args.top_k,
        min_score=args.min_score,
    )
    print("Мини-чат RAG. Команды: /state, /history, /exit")
    while True:
        try:
            message = input("Вы: ").strip()
        except EOFError:
            break
        if not message:
            continue
        if message in {"/exit", "/quit"}:
            break
        if message == "/state":
            print(chat.task_state_view())
            continue
        if message == "/history":
            print(json.dumps(chat.history, ensure_ascii=False, indent=2))
            continue
        print("\nАссистент:")
        print(chat.ask(message))
        print()


def run_scenarios(args: argparse.Namespace) -> None:
    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "created_at": now_iso(),
        "scenario_count": len(scenarios),
        "scenarios": [],
    }
    all_passed = True
    for scenario in scenarios:
        state_path = Path(args.state_file).with_name(
            f"{Path(args.state_file).stem}_{scenario['id']}.json"
        )
        if state_path.exists():
            state_path.unlink()
        chat = MiniRagChat(
            kb_dir=Path(args.kb_dir),
            state_file=state_path,
            top_k=args.top_k,
            min_score=args.min_score,
        )
        turns = []
        for message in scenario["messages"]:
            answer = chat.ask(message)
            checks = {
                "has_sources_section": "Источники:" in answer,
                "has_quotes_section": "Цитаты:" in answer,
                "kept_goal": bool(chat.task_state.goal),
                "has_source_when_expected": (
                    "нет источников выше порога" not in answer.casefold()
                ),
            }
            checks["passed"] = all(checks.values())
            all_passed = all_passed and checks["passed"]
            turns.append(
                {
                    "user": message,
                    "assistant": answer,
                    "task_state": asdict(chat.task_state),
                    "checks": checks,
                }
            )
        report["scenarios"].append(
            {
                "id": scenario["id"],
                "message_count": len(scenario["messages"]),
                "passed": all(turn["checks"]["passed"] for turn in turns),
                "final_task_state": asdict(chat.task_state),
                "turns": turns,
            }
        )

    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"passed": all_passed, "report": args.report}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Мини-чат с RAG, источниками и памятью задачи."
    )
    parser.add_argument("--kb-dir", default=str(DEFAULT_KB_DIR))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.03)
    parser.add_argument("--run-scenarios", action="store_true")
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS_FILE))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_FILE))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.run_scenarios:
        run_scenarios(args)
    else:
        run_cli(args)


if __name__ == "__main__":
    main()
