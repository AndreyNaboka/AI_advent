#!/usr/bin/env python3
"""Goal-driven assistant that reads, searches, analyzes and writes project files."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_client import MCPClientError, MCPFileToolsClient


ROOT = Path(__file__).resolve().parent


class FileAssistantError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileTaskResult:
    scenario: str
    goal: str
    files_read: list[str]
    files_written: list[str]
    changed: bool
    dry_run: bool
    diff: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "goal": self.goal,
            "files_read": self.files_read,
            "files_written": self.files_written,
            "changed": self.changed,
            "dry_run": self.dry_run,
            "diff": self.diff,
            "summary": self.summary,
        }

    def to_text(self) -> str:
        lines = [
            f"Сценарий: {self.scenario}",
            f"Результат: {self.summary}",
            f"Прочитано файлов: {len(self.files_read)}",
            f"Файлы результата: {', '.join(self.files_written) or 'нет'}",
            f"Изменения: {'да' if self.changed else 'нет'}; dry-run: {'да' if self.dry_run else 'нет'}",
        ]
        if self.diff:
            lines.extend(["", "Diff:", self.diff.rstrip()])
        else:
            lines.extend(["", "Diff пуст: результат уже актуален."])
        return "\n".join(lines)


class FileAssistant:
    def __init__(self, client: MCPFileToolsClient | None = None):
        self.client = client or MCPFileToolsClient()

    def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if not self.client.is_running:
                self.client.start()
            result = self.client.call_tool(name, arguments)
        except (MCPClientError, OSError) as error:
            raise FileAssistantError(f"Ошибка MCP file tools: {error}") from error
        structured = result.get("structuredContent")
        if not isinstance(structured, dict):
            raise FileAssistantError(f"Инструмент {name} вернул некорректный результат")
        return structured

    def execute(self, goal: str, dry_run: bool = False) -> FileTaskResult:
        normalized = goal.casefold().strip()
        if not normalized:
            raise FileAssistantError("Цель не должна быть пустой")
        if any(word in normalized for word in ("найди", "найти", "find", "использован")) and any(
            word in normalized for word in ("использ", "компонент", "api", "usage")
        ):
            return self._find_component_usage(goal, dry_run)
        if any(word in normalized for word in ("обнови", "обновить", "синхрониз", "update")) and any(
            word in normalized for word in ("документ", "модул", "структур", "inventory")
        ):
            return self._update_code_inventory(goal, dry_run)
        raise FileAssistantError(
            "Не удалось выбрать сценарий. Поддерживаются цели поиска использований "
            "компонента и обновления документации по Python-модулям."
        )

    @staticmethod
    def _extract_component(goal: str) -> str:
        quoted = re.search(r"[`\"«](.+?)[`\"»]", goal)
        if quoted and quoted.group(1).strip():
            return quoted.group(1).strip()
        explicit = re.search(
            r"(?:компонент(?:а|у|ом)?|api|класс(?:а|у|ом)?)\s+([A-Za-z_][A-Za-z0-9_.]*)",
            goal,
            re.IGNORECASE,
        )
        if explicit:
            return explicit.group(1)
        candidates = re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", goal)
        if candidates:
            return candidates[-1]
        raise FileAssistantError("Укажите компонент в кавычках, например \"MCPCRMClient\"")

    @staticmethod
    def _usage_kind(line: str, component: str) -> str:
        stripped = line.strip()
        if re.search(rf"\bclass\s+{re.escape(component)}\b", stripped):
            return "определение"
        if stripped.startswith(("import ", "from ")):
            return "импорт"
        if re.search(rf"\b{re.escape(component)}\s*\(", stripped):
            return "создание/вызов"
        return "ссылка"

    def _find_component_usage(self, goal: str, dry_run: bool) -> FileTaskResult:
        component = self._extract_component(goal)
        search = self._call(
            "search_text",
            {
                "query": component,
                "case_sensitive": True,
                "extensions": [".py", ".js", ".ts", ".md"],
                "max_matches": 500,
            },
        )
        matching_files = [str(path) for path in search.get("matching_files", [])]
        files_read: list[str] = []
        file_line_counts: dict[str, int] = {}
        for path in matching_files[:50]:
            content = self._call("read_file", {"path": path, "max_chars": 200000})
            files_read.append(path)
            file_line_counts[path] = len(str(content.get("content", "")).splitlines())

        matches = search.get("matches", [])
        groups: dict[str, list[dict[str, Any]]] = {}
        for match in matches:
            if not isinstance(match, dict):
                continue
            groups.setdefault(str(match.get("path", "unknown")), []).append(match)
        report_lines = [
            f"# Использования `{component}`",
            "",
            "Автоматически создано файловым ассистентом через MCP search/read/write tools.",
            "",
            "## Сводка",
            "",
            f"- Просканировано файлов: {search.get('files_scanned', 0)}",
            f"- Файлов с совпадениями: {len(groups)}",
            f"- Всего совпадений: {len(matches)}",
            f"- Результат поиска обрезан: {'да' if search.get('truncated') else 'нет'}",
            "",
            "## Найденные места",
            "",
        ]
        if not groups:
            report_lines.append("Совпадений не найдено.")
        for path in sorted(groups):
            category = "тест" if "test" in path.casefold() else (
                "документация" if path.endswith(".md") else "исходный код"
            )
            report_lines.extend([
                f"### `{path}`",
                "",
                f"Категория: {category}; строк в файле: {file_line_counts.get(path, 'неизвестно')}.",
                "",
            ])
            for match in groups[path]:
                text = str(match.get("text", ""))
                kind = self._usage_kind(text, component)
                report_lines.append(
                    f"- строка {match.get('line')}: **{kind}** — `{text.replace('`', "'")}`"
                )
            report_lines.append("")

        slug = re.sub(r"[^a-z0-9]+", "_", component.casefold()).strip("_") or "component"
        output_path = f"assistant_outputs/usage_{slug}.md"
        write = self._call(
            "write_file",
            {"path": output_path, "content": "\n".join(report_lines).rstrip() + "\n", "dry_run": dry_run},
        )
        return FileTaskResult(
            scenario="component_usage",
            goal=goal,
            files_read=files_read,
            files_written=[output_path],
            changed=bool(write.get("changed")),
            dry_run=dry_run,
            diff=str(write.get("diff", "")),
            summary=f"Найдено {len(matches)} использований {component} в {len(groups)} файлах.",
        )

    @staticmethod
    def _module_summary(path: str, content: str) -> dict[str, Any]:
        try:
            tree = ast.parse(content, filename=path)
        except SyntaxError as error:
            return {"path": path, "error": f"SyntaxError: {error.msg} (line {error.lineno})"}
        docstring = ast.get_docstring(tree) or ""
        classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        functions = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        return {
            "path": path,
            "description": " ".join(docstring.split())[:180] or "Нет module docstring.",
            "classes": classes,
            "functions": functions,
            "imports": sorted(set(imports)),
        }

    def _update_code_inventory(self, goal: str, dry_run: bool) -> FileTaskResult:
        listed = self._call(
            "list_project_files", {"extensions": [".py"], "limit": 500}
        )
        all_files = [str(path) for path in listed.get("files", [])]
        selected = [
            path for path in all_files
            if not any(part in {"tests", "test", "fixtures"} for part in Path(path).parts)
        ][:60]
        modules: list[dict[str, Any]] = []
        for path in selected:
            read = self._call("read_file", {"path": path, "max_chars": 300000})
            modules.append(self._module_summary(path, str(read.get("content", ""))))

        lines = [
            "# Инвентаризация Python-модулей",
            "",
            "Этот файл автоматически генерируется `file_assistant.py` из текущего исходного кода.",
            "Ручные изменения будут заменены при следующем запуске.",
            "",
            "## Сводка",
            "",
            f"- Найдено Python-файлов: {len(all_files)}",
            f"- Проанализировано production-модулей: {len(modules)}",
            f"- Пропущено test/fixture-файлов: {len(all_files) - len(selected)}",
            "",
            "## Модули",
            "",
        ]
        for module in modules:
            lines.extend([f"### `{module['path']}`", ""])
            if "error" in module:
                lines.extend([f"Ошибка анализа: {module['error']}", ""])
                continue
            lines.extend([
                module["description"],
                "",
                "- Классы: " + (", ".join(f"`{name}`" for name in module["classes"]) or "нет"),
                "- Функции: " + (", ".join(f"`{name}`" for name in module["functions"]) or "нет"),
                "- Импорты: " + (", ".join(f"`{name}`" for name in module["imports"]) or "нет"),
                "",
            ])
        output_path = "docs/generated_code_inventory.md"
        write = self._call(
            "write_file",
            {"path": output_path, "content": "\n".join(lines).rstrip() + "\n", "dry_run": dry_run},
        )
        return FileTaskResult(
            scenario="update_code_inventory",
            goal=goal,
            files_read=selected,
            files_written=[output_path],
            changed=bool(write.get("changed")),
            dry_run=dry_run,
            diff=str(write.get("diff", "")),
            summary=f"Документация построена по {len(modules)} Python-модулям.",
        )

    def close(self) -> None:
        self.client.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Goal-driven файловый ассистент через MCP")
    parser.add_argument("--goal", required=True, help="Цель на естественном языке")
    parser.add_argument("--dry-run", action="store_true", help="Только показать diff")
    parser.add_argument("--json", action="store_true", help="Вывести JSON результата")
    parser.add_argument("--root", type=Path, help="Корень проекта для изолированной демонстрации")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.root:
        os.environ["AI_ADVENT_FILES_ROOT"] = str(args.root.expanduser().resolve())
    assistant = FileAssistant()
    try:
        result = assistant.execute(args.goal, dry_run=args.dry_run)
        print(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
            if args.json
            else result.to_text()
        )
        return 0
    except FileAssistantError as error:
        print(f"Ошибка файлового ассистента: {error}")
        return 2
    finally:
        assistant.close()


if __name__ == "__main__":
    raise SystemExit(main())
