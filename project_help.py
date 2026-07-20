#!/usr/bin/env python3
"""Project documentation assistant: README/docs RAG plus live MCP context."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mcp_client import MCPClientError, MCPProjectClient


ROOT = Path(__file__).resolve().parent
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".json", ".yaml", ".yml"}
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_.:/]+")


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in WORD_RE.findall(text) if len(token) > 2]


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    section: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: DocumentChunk
    score: float


class ProjectRagIndex:
    """Small in-memory BM25-like index built from README and docs files."""

    def __init__(self, project_root: Path = ROOT):
        self.project_root = project_root.resolve()
        self.chunks = self._load_chunks()
        self.term_counts = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self.document_frequency = Counter(
            term for counts in self.term_counts for term in counts
        )

    def source_files(self) -> list[Path]:
        files = [self.project_root / "README.md"]
        docs = self.project_root / "docs"
        if docs.is_dir():
            files.extend(
                path for path in docs.rglob("*")
                if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
            )
        return sorted({path for path in files if path.is_file()})

    def _load_chunks(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for path in self.source_files():
            relative = path.relative_to(self.project_root).as_posix()
            section = "Документ"
            buffer: list[str] = []

            def flush() -> None:
                text = "\n".join(buffer).strip()
                if text:
                    chunks.append(DocumentChunk(relative, section, text))
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

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        query_terms = set(tokenize(query))
        if not query_terms or not self.chunks:
            return []
        total = len(self.chunks)
        hits: list[SearchHit] = []
        for chunk, counts in zip(self.chunks, self.term_counts):
            score = 0.0
            searchable = tokenize(f"{chunk.source} {chunk.section}")
            metadata_terms = set(searchable)
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency and term not in metadata_terms:
                    continue
                document_frequency = self.document_frequency[term]
                inverse_frequency = math.log(
                    1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                # Saturation prevents a frequently repeated generic word from
                # outranking a rarer exact term such as "структура".
                score += inverse_frequency * (1 + frequency / (frequency + 1))
                if term in metadata_terms:
                    score += 0.5
            if score:
                hits.append(SearchHit(chunk, score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]


class ProjectHelpAssistant:
    def __init__(self, project_root: Path = ROOT, mcp_client: MCPProjectClient | None = None):
        self.index = ProjectRagIndex(project_root)
        self.mcp = mcp_client or MCPProjectClient()

    def current_branch(self) -> str:
        try:
            if not self.mcp.is_running:
                self.mcp.start()
            result = self.mcp.call_tool("git_branch", {})
            branch = result.get("structuredContent", {}).get("branch")
            return str(branch or "не определена")
        except (MCPClientError, OSError) as error:
            return f"недоступна ({error})"

    @staticmethod
    def _summary(hit: SearchHit, query_terms: set[str]) -> str:
        compact = " ".join(hit.chunk.text.split())
        sentences = re.split(r"(?<=[.!?])\s+|(?<=:)\s+", compact)
        ranked = sorted(
            sentences,
            key=lambda sentence: len(query_terms & set(tokenize(sentence))),
            reverse=True,
        )
        selected = [sentence for sentence in ranked if sentence][:2]
        text = " ".join(selected) if selected else compact
        return text[:500].rstrip() + ("…" if len(text) > 500 else "")

    def ask(self, question: str) -> str:
        question = question.strip()
        if not question:
            return "Формат: /help <вопрос о проекте>"
        hits = self.index.search(question)
        branch = self.current_branch()
        if not hits:
            return (
                "Не нашёл ответа в README.md и docs/. Уточните вопрос или добавьте "
                "нужное описание в документацию.\n\n"
                f"Контекст проекта (MCP): git branch = {branch}\n"
                "Источники RAG: нет релевантных фрагментов"
            )
        query_terms = set(tokenize(question))
        summaries = [self._summary(hit, query_terms) for hit in hits[:2]]
        sources = [
            f"- {hit.chunk.source} — {hit.chunk.section} (score={hit.score:.2f})"
            for hit in hits
        ]
        return (
            "Ответ по документации:\n"
            + "\n".join(f"{index}. {summary}" for index, summary in enumerate(summaries, 1))
            + "\n\n"
            + f"Контекст проекта (MCP): git branch = {branch}\n\n"
            + "Источники RAG:\n"
            + "\n".join(sources)
        )

    def close(self) -> None:
        self.mcp.stop()


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="RAG-помощник по документации проекта")
    parser.add_argument("--question", help="Задать один вопрос и завершить работу")
    args = parser.parse_args()
    assistant = ProjectHelpAssistant()
    try:
        if args.question:
            print(assistant.ask(args.question))
            return
        print("Помощник проекта. Используйте /help <вопрос> или /exit.")
        while True:
            value = input("Вы: ").strip()
            if value in {"/exit", "/quit"}:
                break
            if value == "/help":
                print("Формат: /help <вопрос о проекте>")
            elif value.startswith("/help "):
                print(assistant.ask(value[6:]))
            elif value:
                print("Вопросы о проекте задаются командой /help <вопрос>.")
    finally:
        assistant.close()


if __name__ == "__main__":
    run_cli()
