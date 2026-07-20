# AI Advent

Учебный Python-проект с локальным OpenAI-compatible чатом, несколькими стратегиями
памяти, RAG и MCP-инструментами. Основной интерактивный клиент находится в
`main.py`, а автономный помощник по документации — в `project_help.py`.

## Структура проекта

- `main.py` — основной CLI локального LLM-чата и маршрутизация команд.
- `project_help.py` — команда `/help <вопрос>`, локальный RAG по документации и
  добавление живого MCP-контекста.
- `mcp_project_server.py` — read-only MCP-сервер проекта с инструментами
  `git_branch`, `list_files` и `git_diff`.
- `mcp_client.py` — stdio JSON-RPC/MCP-клиенты.
- `rag_indexer_tool/` — расширенный индексатор с Ollama embeddings и Qdrant.
- `docs/` — документация, схемы и тестовый контент для RAG.
- `mcp_news_server.py`, `mcp_summary_server.py`, `mcp_code_review_server.py` —
  дополнительные MCP-серверы учебного чата.

## Помощник по проекту

Минимальное демо не требует Qdrant, Ollama или запущенной LLM:

```bash
venv312/bin/python project_help.py
```

В интерактивном режиме задайте вопрос:

```text
/help Как устроен проект?
```

Для однократного запроса:

```bash
venv312/bin/python project_help.py --question "Какие MCP-инструменты доступны?"
```

Помощник при старте строит in-memory индекс из корневого `README.md` и всех
поддерживаемых файлов внутри `docs/`. Поиск выбирает релевантные фрагменты,
ответ содержит использованные источники. Текущая Git-ветка получается не
напрямую, а вызовом инструмента `git_branch` локального MCP-сервера.

Та же команда встроена в основной чат: `/help` показывает список команд, а
`/help <вопрос>` отвечает по документации проекта.

## Расширенный RAG

Для семантического поиска через embeddings используется `rag_indexer_tool/`.
Он разбивает документы на чанки, получает embeddings из Ollama и сохраняет их
в Qdrant. Подробные команды приведены в `rag_indexer_tool/README.md`.

## Автоматическое ревью diff

`pr_review.py` анализирует рабочий Git diff, сравнение двух refs или сохранённый
unified diff. Пайплайн определяет изменённые файлы, ищет связанный контекст в
README/docs и исходном коде, затем выводит потенциальные баги, архитектурные
проблемы и рекомендации.

Воспроизводимое демо на намеренно проблемном тестовом PR:

```bash
venv312/bin/python pr_review.py \
  --project-root tests/fixtures/pr_review/project \
  --diff-file tests/fixtures/pr_review/problematic_pr.diff
```

Проверка реальной ветки относительно `main`:

```bash
venv312/bin/python pr_review.py --base-ref main --head-ref HEAD
```

Для генеративного review через локальную OpenAI-compatible LLM добавьте `--llm`.
Без этого флага работает полностью локальный детерминированный анализ, удобный
для тестов и CI. Полный сценарий приведён в `docs/pr_review.md`.

## Проверка

```bash
venv312/bin/python -m unittest discover -s tests -v
venv312/bin/python project_help.py --question "Как устроен проект?"
```
