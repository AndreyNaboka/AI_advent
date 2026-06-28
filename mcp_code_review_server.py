#!/usr/bin/env python3
"""MCP server for reviewing source folders and writing bug reports."""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


PROTOCOL_VERSION = "2024-11-05"
SOURCE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".go", ".rs", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".php", ".rb", ".swift", ".sh", ".bash", ".zsh",
    ".html", ".css", ".scss", ".json", ".yaml", ".yml", ".toml",
    ".md", ".sql",
}
SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", "venv", ".venv", "venv312", "dist", "build",
    ".next", ".nuxt", "coverage", "target", "vendor",
}

REVIEW_TOOL = {
    "name": "review_code_folder",
    "description": "Читает исходный код из указанной папки, делает ревью через LLM и возвращает список проблем.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "api_url": {"type": "string"},
            "folder_path": {"type": "string"},
            "max_files": {"type": "integer", "minimum": 1, "maximum": 500, "default": 120},
            "max_chars_per_file": {"type": "integer", "minimum": 1000, "maximum": 50000, "default": 6000},
            "batch_chars": {"type": "integer", "minimum": 5000, "maximum": 120000, "default": 30000},
        },
        "required": ["api_url", "folder_path"],
        "additionalProperties": False,
    },
}

BUG_REPORT_TOOL = {
    "name": "write_bug_reports",
    "description": "Создает в папке проекта подпапку bugs и пишет отдельный txt-файл на каждую проблему.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "project_dir": {"type": "string"},
            "problems": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["project_dir", "problems"],
        "additionalProperties": False,
    },
}

BUG_FIX_TOOL = {
    "name": "fix_bugs_from_folder",
    "description": "Читает txt-отчеты из папки bugs и применяет точечные исправления к исходникам проекта.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "api_url": {"type": "string"},
            "project_dir": {"type": "string"},
            "max_bugs": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            "max_file_chars": {"type": "integer", "minimum": 1000, "maximum": 100000, "default": 30000},
        },
        "required": ["api_url", "project_dir"],
        "additionalProperties": False,
    },
}


def read_text_file(path: Path, max_chars: int) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return None
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... <truncated> ..."
    return text


def collect_source_files(
    folder: Path,
    max_files: int,
    max_chars_per_file: int,
) -> List[Dict[str, str]]:
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Папка не найдена: {folder}")

    files: List[Dict[str, str]] = []
    for path in sorted(folder.rglob("*")):
        if len(files) >= max_files:
            break
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(folder).parts):
            continue
        if path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        text = read_text_file(path, max_chars_per_file)
        if text is None or not text.strip():
            continue
        files.append(
            {
                "path": str(path.relative_to(folder)),
                "content": text,
            }
        )
    if not files:
        raise ValueError("В папке не найдено читаемых файлов исходного кода")
    return files


def strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```json"):
        value = value[7:]
    if value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    return value.strip()


def extract_json_object(text: str) -> str:
    value = strip_code_fence(text)
    start = value.find("{")
    end = value.rfind("}")
    if start != -1 and end != -1 and end > start:
        return value[start:end + 1]
    return value


def extract_problem_objects(raw: str) -> List[Dict[str, Any]]:
    value = strip_code_fence(raw)
    problems_pos = value.find('"problems"')
    if problems_pos == -1:
        problems_pos = value.find("'problems'")
    if problems_pos == -1:
        return []

    array_start = value.find("[", problems_pos)
    if array_start == -1:
        return []

    objects: List[Dict[str, Any]] = []
    depth = 0
    start: Optional[int] = None
    in_string = False
    quote_char = ""
    escaped = False

    for index in range(array_start + 1, len(value)):
        char = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                in_string = False
            continue

        if char in {'"', "'"}:
            in_string = True
            quote_char = char
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                chunk = value[start:index + 1]
                try:
                    parsed = json.loads(chunk)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, dict):
                    objects.append(parsed)
                start = None

    return objects


def parse_review_response(raw: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(extract_json_object(raw))
        raw_problems = parsed.get("problems", [])
        if not isinstance(raw_problems, list):
            raise ValueError("LLM вернула problems не в виде массива")
    except (json.JSONDecodeError, ValueError, TypeError):
        raw_problems = extract_problem_objects(raw)
        if not raw_problems:
            raise

    return [
        normalize_problem(problem, index)
        for index, problem in enumerate(raw_problems, 1)
        if isinstance(problem, dict)
    ]


def fallback_problem_from_raw_response(raw: str, error: Exception) -> Dict[str, Any]:
    clipped = raw.strip()
    if len(clipped) > 4000:
        clipped = clipped[:4000] + "\n\n... <truncated raw model response> ..."
    return normalize_problem(
        {
            "id": "BUG-REVIEW-OUTPUT",
            "title": "Модель вернула невалидный JSON review",
            "severity": "medium",
            "file": "",
            "line": None,
            "description": (
                "Code review был выполнен, но ответ модели не удалось разобрать как JSON. "
                "Это обычно происходит, когда модель вставляет переносы строк или кавычки "
                "внутрь JSON-строк без экранирования."
            ),
            "evidence": f"Ошибка парсинга: {error}\n\nСырой ответ модели:\n{clipped}",
            "suggestion": (
                "Повторите review на меньшей папке или уменьшите объем файлов. "
                "Также можно открыть этот файл и перенести найденные моделью замечания вручную."
            ),
        },
        1,
    )


def normalize_problem(problem: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "id": str(problem.get("id") or f"BUG-{index:03d}"),
        "title": str(problem.get("title") or "Проблема в коде"),
        "severity": str(problem.get("severity") or "medium"),
        "file": str(problem.get("file") or ""),
        "line": problem.get("line") if isinstance(problem.get("line"), int) else None,
        "description": str(problem.get("description") or ""),
        "evidence": str(problem.get("evidence") or ""),
        "suggestion": str(problem.get("suggestion") or ""),
    }


def build_review_batches(files: List[Dict[str, str]], batch_chars: int) -> List[List[Dict[str, str]]]:
    batches: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    current_chars = 0
    for item in files:
        size = len(item["content"])
        if current and current_chars + size > batch_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def review_file_batch(
    api_url: str,
    files: List[Dict[str, str]],
    batch_number: int,
    total_batches: int,
) -> List[Dict[str, Any]]:
    file_blocks = []
    for item in files:
        file_blocks.append(
            f"FILE: {item['path']}\n```text\n{item['content']}\n```"
        )

    system_prompt = (
        "Ты строгий senior code reviewer. Найди реальные проблемы, которые нужно исправить: "
        "баги, race conditions, ошибки безопасности, неправильную обработку ошибок, "
        "битые контракты, риск потери данных, проблемы запуска и тестируемости. "
        "Не включай вкусовые замечания. Верни только валидный JSON. "
        "Все строковые значения делай короткими и однострочными, без markdown."
    )
    schema_hint = {
        "problems": [
            {
                "id": "BUG-001",
                "title": "Краткий заголовок",
                "severity": "critical|high|medium|low",
                "file": "relative/path.py",
                "line": 123,
                "description": "Что сломано и когда проявится",
                "evidence": "Фрагмент или причина",
                "suggestion": "Как исправить",
            }
        ]
    }
    user_prompt = (
        f"Сделай ревью batch {batch_number}/{total_batches}. "
        "Верни JSON строго по схеме:\n"
        f"{json.dumps(schema_hint, ensure_ascii=False, indent=2)}\n\n"
        "Если проблем нет, верни {\"problems\": []}.\n\n"
        + "\n\n".join(file_blocks)
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 4000,
        "temperature": 0.1,
    }
    response = requests.post(api_url, json=payload, timeout=180)
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    try:
        return parse_review_response(raw)
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        return [fallback_problem_from_raw_response(raw, error)]


def review_code_folder(arguments: Dict[str, Any]) -> Dict[str, Any]:
    api_url = arguments.get("api_url")
    folder_path = arguments.get("folder_path")
    if not isinstance(api_url, str) or not api_url.strip():
        raise ValueError("api_url должен быть строкой")
    if not isinstance(folder_path, str) or not folder_path.strip():
        raise ValueError("folder_path должен быть строкой")

    max_files = int(arguments.get("max_files", 120))
    max_chars_per_file = int(arguments.get("max_chars_per_file", 6000))
    batch_chars = int(arguments.get("batch_chars", 30000))
    folder = Path(folder_path).expanduser().resolve()
    files = collect_source_files(
        folder,
        max_files,
        max_chars_per_file,
    )
    batches = build_review_batches(files, batch_chars)
    problems: List[Dict[str, Any]] = []
    for batch_index, batch in enumerate(batches, 1):
        batch_problems = review_file_batch(api_url, batch, batch_index, len(batches))
        problems.extend(batch_problems)
    for index, problem in enumerate(problems, 1):
        if problem["id"].startswith("BUG-"):
            problem["id"] = f"BUG-{index:03d}"

    if problems:
        lines = [
            f"Проверено файлов: {len(files)}; batch-запросов: {len(batches)}",
            "Найдены проблемы:",
        ]
        for problem in problems:
            location = problem["file"]
            if problem["line"]:
                location += f":{problem['line']}"
            lines.append(
                f"- [{problem['severity']}] {problem['id']} {problem['title']} ({location})"
            )
    else:
        lines = [
            f"Проверено файлов: {len(files)}; batch-запросов: {len(batches)}",
            "Проблемы не найдены.",
        ]

    return {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "structuredContent": {
            "folder": str(folder),
            "files_reviewed": [item["path"] for item in files],
            "batches": len(batches),
            "problems": problems,
        },
    }


def safe_filename(value: str, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9А-Яа-яЁё._-]+", "_", value).strip("._-")
    return (name or fallback)[:80]


def problem_text(problem: Dict[str, Any]) -> str:
    line = problem.get("line")
    location = str(problem.get("file") or "")
    if isinstance(line, int):
        location += f":{line}"
    return (
        f"ID: {problem.get('id', '')}\n"
        f"Заголовок: {problem.get('title', '')}\n"
        f"Severity: {problem.get('severity', '')}\n"
        f"Место: {location}\n"
        f"Создано: {datetime.now(timezone.utc).isoformat()}\n\n"
        "Описание проблемы:\n"
        f"{problem.get('description', '')}\n\n"
        "Доказательство / контекст:\n"
        f"{problem.get('evidence', '')}\n\n"
        "Предложение по исправлению:\n"
        f"{problem.get('suggestion', '')}\n"
    )


def write_bug_reports(arguments: Dict[str, Any]) -> Dict[str, Any]:
    project_dir = arguments.get("project_dir")
    problems = arguments.get("problems")
    if not isinstance(project_dir, str) or not project_dir.strip():
        raise ValueError("project_dir должен быть строкой")
    if not isinstance(problems, list):
        raise ValueError("problems должен быть массивом")

    root = Path(project_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Папка проекта не найдена: {root}")

    bugs_dir = root / "bugs"
    bugs_dir.mkdir(exist_ok=True)
    written: List[str] = []
    for index, raw_problem in enumerate(problems, 1):
        if not isinstance(raw_problem, dict):
            continue
        problem = normalize_problem(raw_problem, index)
        base = safe_filename(
            f"{problem['id']}_{problem['title']}",
            f"bug_{index:03d}",
        )
        path = bugs_dir / f"{index:03d}_{base}.txt"
        duplicate = 2
        while path.exists():
            path = bugs_dir / f"{index:03d}_{base}_{duplicate}.txt"
            duplicate += 1
        path.write_text(problem_text(problem), encoding="utf-8")
        written.append(str(path))

    return {
        "content": [
            {
                "type": "text",
                "text": f"Создано файлов: {len(written)}\n" + "\n".join(written),
            }
        ],
        "structuredContent": {"bugs_dir": str(bugs_dir), "files": written},
    }


def parse_bug_file(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {"bug_file": str(path), "raw": text}
    for key, label in (
        ("id", "ID:"),
        ("title", "Заголовок:"),
        ("severity", "Severity:"),
        ("location", "Место:"),
    ):
        match = re.search(rf"^{re.escape(label)}\s*(.*)$", text, flags=re.MULTILINE)
        result[key] = match.group(1).strip() if match else ""
    return result


def resolve_bug_source_path(project_dir: Path, bug: Dict[str, str]) -> Optional[Path]:
    location = bug.get("location", "").strip()
    if not location:
        return None
    if ":" in location:
        location = location.rsplit(":", 1)[0]
    candidate = (project_dir / location).resolve()
    try:
        candidate.relative_to(project_dir)
    except ValueError:
        return None
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def parse_fix_response(raw: str) -> Dict[str, Any]:
    parsed = json.loads(extract_json_object(raw))
    edits = parsed.get("edits", [])
    if not isinstance(edits, list):
        raise ValueError("LLM вернула edits не в виде массива")
    normalized = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        file_path = edit.get("file")
        original = edit.get("original")
        replacement = edit.get("replacement")
        if all(isinstance(value, str) for value in (file_path, original, replacement)):
            normalized.append(
                {
                    "file": file_path,
                    "original": original,
                    "replacement": replacement,
                }
            )
    return {"edits": normalized}


def ask_llm_for_bug_fix(
    api_url: str,
    project_dir: Path,
    bug: Dict[str, str],
    source_path: Path,
    source_text: str,
) -> Dict[str, Any]:
    relative_path = str(source_path.relative_to(project_dir))
    schema_hint = {
        "edits": [
            {
                "file": relative_path,
                "original": "точный фрагмент из файла",
                "replacement": "новый фрагмент",
            }
        ]
    }
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты исправляешь баги в коде. Верни только валидный JSON. "
                    "Используй только точечные exact-match замены: original должен "
                    "полностью совпадать с фрагментом исходного файла. Не используй markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Баг-репорт:\n"
                    f"{bug['raw']}\n\n"
                    f"Файл: {relative_path}\n"
                    "Исходный код:\n"
                    f"```text\n{source_text}\n```\n\n"
                    "Верни JSON строго по схеме:\n"
                    f"{json.dumps(schema_hint, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        "max_tokens": 3000,
        "temperature": 0.1,
    }
    response = requests.post(api_url, json=payload, timeout=180)
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    return parse_fix_response(raw)


def backup_source_file(project_dir: Path, path: Path, text: str) -> str:
    relative = path.relative_to(project_dir)
    backup_dir = project_dir / "bugs" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = safe_filename(str(relative).replace("/", "__"), "source")
    backup_path = backup_dir / f"{safe_name}.{stamp}.bak"
    duplicate = 2
    while backup_path.exists():
        backup_path = backup_dir / f"{safe_name}.{stamp}_{duplicate}.bak"
        duplicate += 1
    backup_path.write_text(text, encoding="utf-8")
    return str(backup_path)


def apply_exact_edits(project_dir: Path, edits: List[Dict[str, str]]) -> List[Dict[str, str]]:
    results = []
    for edit in edits:
        path = (project_dir / edit["file"]).resolve()
        try:
            path.relative_to(project_dir)
        except ValueError:
            results.append({"file": edit["file"], "status": "skipped", "reason": "path outside project"})
            continue
        if not path.exists() or not path.is_file():
            results.append({"file": edit["file"], "status": "skipped", "reason": "file not found"})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        original = edit["original"]
        if not original:
            results.append({"file": edit["file"], "status": "skipped", "reason": "empty original"})
            continue
        count = text.count(original)
        if count != 1:
            results.append(
                {
                    "file": edit["file"],
                    "status": "skipped",
                    "reason": f"original match count is {count}",
                }
            )
            continue
        backup_path = backup_source_file(project_dir, path, text)
        path.write_text(text.replace(original, edit["replacement"], 1), encoding="utf-8")
        results.append(
            {
                "file": edit["file"],
                "status": "applied",
                "reason": "",
                "backup": backup_path,
            }
        )
    return results


def fix_bugs_from_folder(arguments: Dict[str, Any]) -> Dict[str, Any]:
    api_url = arguments.get("api_url")
    project_dir_value = arguments.get("project_dir")
    if not isinstance(api_url, str) or not api_url.strip():
        raise ValueError("api_url должен быть строкой")
    if not isinstance(project_dir_value, str) or not project_dir_value.strip():
        raise ValueError("project_dir должен быть строкой")

    project_dir = Path(project_dir_value).expanduser().resolve()
    if not project_dir.exists() or not project_dir.is_dir():
        raise ValueError(f"Папка проекта не найдена: {project_dir}")
    bugs_dir = project_dir / "bugs"
    if not bugs_dir.exists() or not bugs_dir.is_dir():
        raise ValueError(f"Папка bugs не найдена: {bugs_dir}")

    max_bugs = int(arguments.get("max_bugs", 10))
    max_file_chars = int(arguments.get("max_file_chars", 30000))
    bug_files = sorted(bugs_dir.glob("*.txt"))[:max_bugs]
    if not bug_files:
        raise ValueError(f"В {bugs_dir} нет txt-файлов с багами")

    all_results = []
    for bug_file in bug_files:
        bug = parse_bug_file(bug_file)
        source_path = resolve_bug_source_path(project_dir, bug)
        if source_path is None:
            all_results.append(
                {
                    "bug_file": str(bug_file),
                    "status": "skipped",
                    "details": "не удалось определить исходный файл из поля Место",
                }
            )
            continue
        source_text = read_text_file(source_path, max_file_chars)
        if source_text is None:
            all_results.append(
                {
                    "bug_file": str(bug_file),
                    "status": "skipped",
                    "details": "не удалось прочитать исходный файл",
                }
            )
            continue
        try:
            fix = ask_llm_for_bug_fix(api_url, project_dir, bug, source_path, source_text)
            edit_results = apply_exact_edits(project_dir, fix["edits"])
        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as error:
            all_results.append(
                {"bug_file": str(bug_file), "status": "failed", "details": str(error)}
            )
            continue

        applied = sum(1 for item in edit_results if item["status"] == "applied")
        all_results.append(
            {
                "bug_file": str(bug_file),
                "status": "applied" if applied else "skipped",
                "details": edit_results,
            }
        )

    lines = ["Результат исправления багов:"]
    for result in all_results:
        lines.append(f"- {result['status']}: {result['bug_file']}")
        if isinstance(result.get("details"), str):
            lines.append(f"  {result['details']}")

    return {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "structuredContent": {
            "project_dir": str(project_dir),
            "bugs_dir": str(bugs_dir),
            "results": all_results,
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
                "serverInfo": {"name": "code-review", "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {"tools": [REVIEW_TOOL, BUG_REPORT_TOOL, BUG_FIX_TOOL]}
        elif method == "tools/call":
            params = message.get("params") or {}
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ValueError("arguments должен быть JSON-объектом")
            name = params.get("name")
            if name == REVIEW_TOOL["name"]:
                result = review_code_folder(arguments)
            elif name == BUG_REPORT_TOOL["name"]:
                result = write_bug_reports(arguments)
            elif name == BUG_FIX_TOOL["name"]:
                result = fix_bugs_from_folder(arguments)
            else:
                raise ValueError(f"Неизвестный инструмент: {name!r}")
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as error:
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
                "content": [{"type": "text", "text": f"Ошибка LLM review: {error}"}],
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
