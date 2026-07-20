#!/usr/bin/env python3
"""Review a Git diff using local documentation/code RAG and optional local LLM."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
SOURCE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt",
    ".cs", ".php", ".rb", ".swift", ".c", ".cpp", ".h", ".hpp", ".sh",
    ".sql", ".html", ".css", ".json", ".yaml", ".yml", ".toml",
}
DOC_SUFFIXES = {".md", ".markdown", ".txt", ".json", ".yaml", ".yml"}
SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules",
    "venv", "venv312", ".venv", "dist", "build", "target", "vendor",
    "AI_advent_export", "qdrant_storage",
}
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_.]+")


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text) if len(token) > 2]


@dataclass
class DiffLine:
    kind: str
    content: str
    old_line: int | None
    new_line: int | None


@dataclass
class ChangedFile:
    path: str
    old_path: str = ""
    lines: list[DiffLine] = field(default_factory=list)

    @property
    def added(self) -> list[DiffLine]:
        return [line for line in self.lines if line.kind == "added"]

    @property
    def removed(self) -> list[DiffLine]:
        return [line for line in self.lines if line.kind == "removed"]


class DiffError(RuntimeError):
    pass


def parse_unified_diff(diff_text: str) -> list[ChangedFile]:
    files: list[ChangedFile] = []
    current: ChangedFile | None = None
    old_line = new_line = 0
    in_hunk = False

    for raw_line in diff_text.splitlines():
        match = re.match(r"^diff --git a/(.+) b/(.+)$", raw_line)
        if match:
            current = ChangedFile(path=match.group(2), old_path=match.group(1))
            files.append(current)
            in_hunk = False
            continue
        if current is None:
            continue
        if raw_line.startswith("+++ "):
            value = raw_line[4:]
            if value == "/dev/null":
                current.path = current.old_path
            else:
                current.path = value[2:] if value.startswith("b/") else value
            continue
        hunk = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw_line)
        if hunk:
            old_line, new_line = int(hunk.group(1)), int(hunk.group(2))
            in_hunk = True
            continue
        if not in_hunk or raw_line.startswith("\\ No newline"):
            continue
        if raw_line.startswith("+"):
            current.lines.append(DiffLine("added", raw_line[1:], None, new_line))
            new_line += 1
        elif raw_line.startswith("-"):
            current.lines.append(DiffLine("removed", raw_line[1:], old_line, None))
            old_line += 1
        else:
            content = raw_line[1:] if raw_line.startswith(" ") else raw_line
            current.lines.append(DiffLine("context", content, old_line, new_line))
            old_line += 1
            new_line += 1
    return files


class GitDiffProvider:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()

    def get_diff(
        self,
        diff_file: Path | None = None,
        base_ref: str | None = None,
        head_ref: str | None = None,
    ) -> tuple[str, str]:
        if diff_file:
            path = diff_file.expanduser().resolve()
            if not path.is_file():
                raise DiffError(f"Diff-файл не найден: {path}")
            return path.read_text(encoding="utf-8", errors="replace"), str(path)
        if bool(base_ref) != bool(head_ref):
            raise DiffError("Для сравнения refs укажите одновременно --base-ref и --head-ref")
        args = ["git", "-C", str(self.project_root), "diff", "--no-ext-diff"]
        label = "working tree относительно HEAD"
        if base_ref and head_ref:
            args.append(f"{base_ref}...{head_ref}")
            label = f"{base_ref}...{head_ref}"
        else:
            args.append("HEAD")
        try:
            completed = subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
        except (subprocess.SubprocessError, OSError) as error:
            raise DiffError(f"Не удалось получить git diff: {error}") from error
        return completed.stdout, label


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    source_type: str
    text: str


@dataclass(frozen=True)
class RagHit:
    chunk: KnowledgeChunk
    score: float


class ReviewKnowledgeIndex:
    """In-memory RAG index that deliberately covers both docs and source code."""

    def __init__(self, project_root: Path, max_files: int = 400):
        self.project_root = project_root.resolve()
        self.chunks = self._load_chunks(max_files)
        self.counts = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self.document_frequency = Counter(term for counts in self.counts for term in counts)

    def _candidate_files(self) -> Iterable[tuple[Path, str]]:
        readme = self.project_root / "README.md"
        if readme.is_file():
            yield readme, "documentation"
        docs_dir = self.project_root / "docs"
        if docs_dir.is_dir():
            for path in sorted(docs_dir.rglob("*")):
                if path.is_file() and path.suffix.casefold() in DOC_SUFFIXES:
                    yield path, "documentation"
        for path in sorted(self.project_root.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in SOURCE_SUFFIXES:
                continue
            relative = path.relative_to(self.project_root)
            if any(part in SKIP_DIRS for part in relative.parts) or "docs" in relative.parts:
                continue
            yield path, "code"

    def _load_chunks(self, max_files: int) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        seen: set[Path] = set()
        for path, source_type in self._candidate_files():
            if path in seen:
                continue
            seen.add(path)
            if len(seen) > max_files:
                break
            relative = path.relative_to(self.project_root).as_posix()
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for start in range(0, len(lines), 80):
                text = "\n".join(lines[start:start + 100]).strip()
                if text:
                    chunks.append(KnowledgeChunk(relative, source_type, text))
        return chunks

    def search(self, query: str, per_type: int = 2) -> list[RagHit]:
        terms = set(tokenize(query))
        total = len(self.chunks)
        scored: list[RagHit] = []
        for chunk, counts in zip(self.chunks, self.counts):
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency[term]
                inverse_frequency = math.log(1 + (total - df + 0.5) / (df + 0.5))
                score += inverse_frequency * (1 + frequency / (frequency + 1))
            if score:
                scored.append(RagHit(chunk, score))
        result: list[RagHit] = []
        for source_type in ("documentation", "code"):
            typed = sorted(
                (hit for hit in scored if hit.chunk.source_type == source_type),
                key=lambda hit: hit.score,
                reverse=True,
            )
            result.extend(typed[:per_type])
        return result


@dataclass(frozen=True)
class Finding:
    category: str
    severity: str
    title: str
    file: str
    line: int | None
    evidence: str
    recommendation: str
    rag_sources: tuple[str, ...] = ()


class ReviewAnalyzer:
    def __init__(self, index: ReviewKnowledgeIndex):
        self.index = index

    def _finding(
        self,
        category: str,
        severity: str,
        title: str,
        changed_file: ChangedFile,
        line: DiffLine | None,
        evidence: str,
        recommendation: str,
    ) -> Finding:
        query = f"{changed_file.path} {title} {evidence} {recommendation}"
        sources = tuple(
            dict.fromkeys(hit.chunk.source for hit in self.index.search(query))
        )
        return Finding(
            category, severity, title, changed_file.path,
            (line.new_line if line.new_line is not None else line.old_line) if line else None,
            evidence.strip(), recommendation, sources,
        )

    def analyze(self, changed_files: list[ChangedFile]) -> list[Finding]:
        findings: list[Finding] = []
        production_changed = False
        tests_changed = False
        for changed_file in changed_files:
            normalized_path = changed_file.path.casefold()
            is_test = any(part in normalized_path for part in ("test", "spec", "fixture"))
            tests_changed = tests_changed or is_test
            production_changed = production_changed or not is_test
            added_text = "\n".join(line.content for line in changed_file.added)
            removed_text = "\n".join(line.content for line in changed_file.removed)

            for line in changed_file.added:
                code = line.content.strip()
                if re.search(
                    r"\b[A-Za-z_]*(?:api_?key|secret|password|token)[A-Za-z_]*\s*=\s*['\"][^'\"]+['\"]",
                    code,
                    re.I,
                ):
                    findings.append(self._finding(
                        "bug", "critical", "Секрет добавлен в исходный код", changed_file, line,
                        code, "Удалите секрет из Git и загружайте его из secret storage или переменной окружения.",
                    ))
                if "verify=False" in code:
                    findings.append(self._finding(
                        "bug", "high", "Отключена проверка TLS-сертификата", changed_file, line,
                        code, "Не отключайте TLS verification; настройте доверенный CA bundle.",
                    ))
                if re.search(r"\brequests\.(?:get|post|put|patch|delete)\s*\(", code) and "timeout=" not in code:
                    findings.append(self._finding(
                        "bug", "medium", "HTTP-запрос не ограничен timeout", changed_file, line,
                        code, "Добавьте конечный connect/read timeout и обработайте requests.Timeout.",
                    ))
                if re.match(r"except\s*:", code):
                    findings.append(self._finding(
                        "bug", "medium", "Bare except скрывает системные ошибки", changed_file, line,
                        code, "Перехватывайте конкретные исключения и логируйте диагностический контекст.",
                    ))
                if re.search(r"\b(?:eval|exec)\s*\(", code):
                    findings.append(self._finding(
                        "bug", "high", "Динамическое выполнение входных данных", changed_file, line,
                        code, "Замените eval/exec безопасным парсером с явной схемой.",
                    ))
                if re.match(r"(?:from\s+\S+\s+import\s+\*|import\s+\*)", code):
                    findings.append(self._finding(
                        "architecture", "medium", "Wildcard import размывает зависимости модуля",
                        changed_file, line, code, "Импортируйте только необходимые публичные символы.",
                    ))
                if any(marker in code for marker in ("TODO", "FIXME")):
                    findings.append(self._finding(
                        "recommendation", "low", "В изменении оставлен незавершённый участок",
                        changed_file, line, code, "Закройте TODO до merge или свяжите его с задачей в tracker.",
                    ))

            validation_terms = ("validate", "validation", "провер", "permission", "authorize")
            removed_guard = any(
                re.match(r"\s*if\s+.+(?:<=|>=|==|\bnot\b|\bis\s+none\b)", line.content, re.I)
                for line in changed_file.removed
            ) and any("raise " in line.content for line in changed_file.removed)
            removed_validation = (
                any(term in removed_text.casefold() for term in validation_terms)
                or removed_guard
            )
            added_validation = any(term in added_text.casefold() for term in validation_terms)
            if removed_validation and not added_validation:
                first_removed = next(
                    (
                        line for line in changed_file.removed
                        if any(term in line.content.casefold() for term in validation_terms)
                        or re.match(
                            r"\s*if\s+.+(?:<=|>=|==|\bnot\b|\bis\s+none\b)",
                            line.content,
                            re.I,
                        )
                    ),
                    None,
                )
                findings.append(self._finding(
                    "architecture", "high", "Удалена проверка входных данных или прав",
                    changed_file, first_removed, first_removed.content if first_removed else removed_text[:200],
                    "Верните проверку на границе системы или документируйте новый доверенный контракт.",
                ))
            if len(changed_file.added) > 250:
                findings.append(self._finding(
                    "architecture", "medium", "Слишком крупное изменение в одном файле",
                    changed_file, None, f"Добавлено строк: {len(changed_file.added)}",
                    "Разделите изменение на небольшие компоненты и независимые PR.",
                ))

        if production_changed and not tests_changed and changed_files:
            findings.append(self._finding(
                "recommendation", "medium", "Нет изменений в автоматических тестах",
                changed_files[0], None, "Изменены production-файлы, но diff не содержит test/spec-файлов.",
                "Добавьте тесты для успешного сценария и найденных ошибочных веток.",
            ))
        return self._deduplicate(findings)

    @staticmethod
    def _deduplicate(findings: list[Finding]) -> list[Finding]:
        result: list[Finding] = []
        seen: set[tuple[str, str, int | None]] = set()
        for finding in findings:
            key = (finding.title, finding.file, finding.line)
            if key not in seen:
                seen.add(key)
                result.append(finding)
        return result


SECTION_NAMES = {
    "bug": "Потенциальные баги",
    "architecture": "Архитектурные проблемы",
    "recommendation": "Рекомендации",
}
SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def format_review(
    source: str,
    changed_files: list[ChangedFile],
    findings: list[Finding],
    index: ReviewKnowledgeIndex,
) -> str:
    added = sum(len(item.added) for item in changed_files)
    removed = sum(len(item.removed) for item in changed_files)
    lines = [
        "# Автоматическое AI-ревью",
        "",
        f"- Diff: `{source}`",
        f"- Изменено файлов: {len(changed_files)}",
        f"- Строк: +{added} / -{removed}",
        f"- RAG-индекс: {sum(c.source_type == 'documentation' for c in index.chunks)} чанков документации, "
        f"{sum(c.source_type == 'code' for c in index.chunks)} чанков кода",
        "- Файлы: " + (", ".join(f"`{item.path}`" for item in changed_files) or "нет"),
    ]
    for category in ("bug", "architecture", "recommendation"):
        lines.extend(["", f"## {SECTION_NAMES[category]}", ""])
        category_findings = sorted(
            (item for item in findings if item.category == category),
            key=lambda item: SEVERITY_RANK[item.severity],
            reverse=True,
        )
        if not category_findings:
            lines.append("Не обнаружены.")
            continue
        for number, finding in enumerate(category_findings, 1):
            location = finding.file + (f":{finding.line}" if finding.line else "")
            rag = ", ".join(f"`{source}`" for source in finding.rag_sources) or "нет совпадений"
            evidence = " ".join(finding.evidence.split())[:240]
            lines.extend([
                f"{number}. **[{finding.severity.upper()}] {finding.title}** — `{location}`",
                f"   - Основание: `{evidence}`",
                f"   - Рекомендация: {finding.recommendation}",
                f"   - RAG-контекст (документация + код): {rag}",
            ])
    return "\n".join(lines) + "\n"


def enhance_with_llm(report: str, diff_text: str, api_url: str) -> str:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты senior code reviewer. Улучши черновик ревью только на основании diff и "
                    "RAG-находок. Сохрани разделы: Потенциальные баги, Архитектурные проблемы, "
                    "Рекомендации. Не выдумывай отсутствующий код. Верни Markdown."
                ),
            },
            {"role": "user", "content": f"DIFF:\n{diff_text[:30000]}\n\nЧЕРНОВИК:\n{report}"},
        ],
        "temperature": 0.1,
        "max_tokens": 3000,
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"]).strip() + "\n"


def should_fail(findings: list[Finding], fail_on: str) -> bool:
    if fail_on == "none":
        return False
    threshold = SEVERITY_RANK[fail_on]
    return any(SEVERITY_RANK[item.severity] >= threshold for item in findings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Автоматическое RAG-ревью Git diff")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--diff-file", type=Path, help="Готовый unified diff для демо или CI")
    parser.add_argument("--base-ref", help="Базовая Git-ссылка, например origin/main")
    parser.add_argument("--head-ref", help="Проверяемая Git-ссылка, например HEAD")
    parser.add_argument("--output", type=Path, help="Дополнительно сохранить Markdown-отчёт")
    parser.add_argument("--llm", action="store_true", help="Улучшить черновик локальной LLM")
    parser.add_argument(
        "--api-url", default="http://localhost:8080/v1/chat/completions",
        help="OpenAI-compatible endpoint для --llm",
    )
    parser.add_argument(
        "--fail-on", choices=["none", "critical", "high", "medium", "low"], default="none",
        help="Вернуть exit code 1 при замечании заданной или более высокой важности",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    try:
        diff_text, source = GitDiffProvider(root).get_diff(
            args.diff_file, args.base_ref, args.head_ref
        )
        if not diff_text.strip():
            raise DiffError("Diff пуст: нечего проверять")
        changed_files = parse_unified_diff(diff_text)
        if not changed_files:
            raise DiffError("В diff не найдены изменённые файлы")
        index = ReviewKnowledgeIndex(root)
        findings = ReviewAnalyzer(index).analyze(changed_files)
        report = format_review(source, changed_files, findings, index)
        if args.llm:
            try:
                report = enhance_with_llm(report, diff_text, args.api_url)
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as error:
                report += f"\n> LLM недоступна, оставлен локальный review: {error}\n"
        print(report, end="")
        if args.output:
            args.output.expanduser().write_text(report, encoding="utf-8")
        return 1 if should_fail(findings, args.fail_on) else 0
    except (DiffError, OSError) as error:
        print(f"Ошибка review: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
