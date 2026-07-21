# Инвентаризация Python-модулей

Этот файл автоматически генерируется `file_assistant.py` из текущего исходного кода.
Ручные изменения будут заменены при следующем запуске.

## Сводка

- Найдено Python-файлов: 39
- Проанализировано production-модулей: 30
- Пропущено test/fixture-файлов: 9

## Модули

### `file_assistant.py`

Goal-driven assistant that reads, searches, analyzes and writes project files.

- Классы: `FileAssistantError`, `FileTaskResult`, `FileAssistant`
- Функции: `build_parser`, `main`
- Импорты: `__future__`, `argparse`, `ast`, `dataclasses`, `json`, `mcp_client`, `os`, `pathlib`, `re`, `typing`

### `main.py`

Нет module docstring.

- Классы: `LLMResponse`, `LLMChat`
- Функции: `normalize_api_url`, `api_base_from_url`, `main`
- Импорты: `argparse`, `json`, `jsonschema`, `mcp_client`, `os`, `pathlib`, `project_help`, `re`, `requests`, `socket`, `sys`, `threading`, `time`, `typing`, `urllib.parse`, `yaml`

### `mcp_client.py`

Small stdio MCP client used by the interactive application.

- Классы: `MCPClientError`, `MCPNewsClient`, `MCPPeriodicSummaryClient`, `MCPCodeReviewClient`, `MCPProjectClient`, `MCPCRMClient`, `MCPFileToolsClient`
- Функции: нет
- Импорты: `json`, `pathlib`, `subprocess`, `sys`, `typing`

### `mcp_code_review_server.py`

MCP server for reviewing source folders and writing bug reports.

- Классы: нет
- Функции: `read_text_file`, `collect_source_files`, `strip_code_fence`, `extract_json_object`, `extract_problem_objects`, `parse_review_response`, `fallback_problem_from_raw_response`, `normalize_problem`, `build_review_batches`, `review_file_batch`, `review_code_folder`, `safe_filename`, `problem_text`, `write_bug_reports`, `parse_bug_file`, `resolve_bug_source_path`, `parse_fix_response`, `ask_llm_for_bug_fix`, `backup_source_file`, `apply_exact_edits`, `fix_bugs_from_folder`, `make_response`, `main`
- Импорты: `datetime`, `json`, `pathlib`, `re`, `requests`, `sys`, `typing`

### `mcp_crm_server.py`

Read-only MCP server backed by a local JSON CRM fixture.

- Классы: нет
- Функции: `crm_file`, `load_crm`, `require_arguments`, `find_by_id`, `call_tool`, `make_response`, `main`
- Импорты: `__future__`, `json`, `os`, `pathlib`, `sys`, `typing`

### `mcp_filesystem_server.py`

Constrained MCP filesystem tools for autonomous project file tasks.

- Классы: нет
- Функции: `project_root`, `safe_path`, `normalize_extensions`, `iter_text_files`, `validate_arguments`, `list_project_files`, `read_file`, `search_text`, `write_file`, `tool_result`, `call_tool`, `make_response`, `main`
- Импорты: `__future__`, `difflib`, `json`, `os`, `pathlib`, `sys`, `tempfile`, `typing`

### `mcp_news_server.py`

Minimal MCP server that exposes current world news over stdio.

- Классы: нет
- Функции: `fetch_world_news`, `tool_result`, `make_response`, `main`
- Импорты: `email.utils`, `html`, `json`, `requests`, `sys`, `typing`, `xml.etree.ElementTree`

### `mcp_project_server.py`

Read-only MCP server exposing useful context from this Git project.

- Классы: нет
- Функции: `run_git`, `call_tool`, `make_response`, `main`
- Импорты: `__future__`, `json`, `pathlib`, `subprocess`, `sys`, `typing`

### `mcp_summary_server.py`

MCP server that periodically summarizes the local chat into a file.

- Классы: нет
- Функции: `validate_messages`, `format_messages`, `load_existing_summary`, `summarize_dialog`, `make_response`, `main`
- Импорты: `datetime`, `json`, `pathlib`, `requests`, `sys`, `typing`

### `mini_rag_chat.py`

Нет module docstring.

- Классы: `SourceChunk`, `RagHit`, `TaskState`, `MarkdownRagStore`, `MiniRagChat`
- Функции: `now_iso`, `tokenize`, `run_cli`, `run_scenarios`, `build_parser`, `main`
- Импорты: `__future__`, `argparse`, `dataclasses`, `datetime`, `json`, `pathlib`, `re`, `typing`

### `pr_review.py`

Review a Git diff using local documentation/code RAG and optional local LLM.

- Классы: `DiffLine`, `ChangedFile`, `DiffError`, `GitDiffProvider`, `KnowledgeChunk`, `RagHit`, `ReviewKnowledgeIndex`, `Finding`, `ReviewAnalyzer`
- Функции: `tokenize`, `parse_unified_diff`, `format_review`, `enhance_with_llm`, `should_fail`, `build_parser`, `main`
- Импорты: `__future__`, `argparse`, `collections`, `dataclasses`, `json`, `math`, `pathlib`, `re`, `subprocess`, `sys`, `typing`, `urllib.error`, `urllib.request`

### `project_help.py`

Project documentation assistant: README/docs RAG plus live MCP context.

- Классы: `DocumentChunk`, `SearchHit`, `ProjectRagIndex`, `ProjectHelpAssistant`
- Функции: `tokenize`, `run_cli`
- Импорты: `__future__`, `argparse`, `collections`, `dataclasses`, `math`, `mcp_client`, `pathlib`, `re`

### `rag_compare_eval.py`

Нет module docstring.

- Классы: нет
- Функции: `normalize_api_url`, `load_config`, `load_questions`, `embed_query`, `search_chunks`, `format_sources`, `build_rag_context`, `ask_llm`, `rewrite_query`, `tokenize_for_rerank`, `filter_and_rerank`, `build_answer_from_context`, `contains_all`, `sources_match`, `section_after`, `has_required_rag_sections`, `has_nonempty_sources_and_quotes`, `quotes_support_expected`, `build_unknown_context_test`, `evaluate`, `print_summary`, `print_answers`, `build_parser`, `main`
- Импорты: `__future__`, `argparse`, `datetime`, `json`, `os`, `pathlib`, `qdrant_client`, `requests`, `sys`, `typing`, `yaml`

### `rag_indexer_tool/chunkers/base.py`

Нет module docstring.

- Классы: `ChunkingConfig`, `RawBlock`
- Функции: `normalize_text`, `split_text_recursive`, `merge_small_blocks`, `_can_merge`, `_merge_metadata`, `add_context`, `finalize_blocks`
- Импорты: `__future__`, `dataclasses`, `pathlib`, `re`, `typing`, `utils.hashing`, `utils.token_count`

### `rag_indexer_tool/chunkers/code_chunker.py`

Нет module docstring.

- Классы: нет
- Функции: `chunk_code`
- Импорты: `__future__`, `chunkers.base`, `pathlib`, `re`

### `rag_indexer_tool/chunkers/docx_chunker.py`

Нет module docstring.

- Классы: нет
- Функции: `chunk_docx`
- Импорты: `__future__`, `chunkers.base`, `pathlib`

### `rag_indexer_tool/chunkers/html_chunker.py`

Нет module docstring.

- Классы: нет
- Функции: `chunk_html`
- Импорты: `__future__`, `chunkers.base`, `pathlib`

### `rag_indexer_tool/chunkers/json_chunker.py`

Нет module docstring.

- Классы: нет
- Функции: `chunk_json_or_yaml`, `_block`
- Импорты: `__future__`, `chunkers.base`, `json`, `pathlib`, `typing`, `yaml`

### `rag_indexer_tool/chunkers/markdown_chunker.py`

Нет module docstring.

- Классы: нет
- Функции: `chunk_markdown`
- Импорты: `__future__`, `chunkers.base`, `pathlib`, `re`

### `rag_indexer_tool/chunkers/pdf_chunker.py`

Нет module docstring.

- Классы: нет
- Функции: `chunk_pdf`
- Импорты: `__future__`, `chunkers.base`, `pathlib`

### `rag_indexer_tool/chunkers/table_chunker.py`

Нет module docstring.

- Классы: нет
- Функции: `chunk_table`, `_table_block`
- Импорты: `__future__`, `chunkers.base`, `pathlib`, `utils.token_count`

### `rag_indexer_tool/chunkers/text_chunker.py`

Нет module docstring.

- Классы: нет
- Функции: `chunk_txt`
- Импорты: `__future__`, `chunkers.base`, `pathlib`

### `rag_indexer_tool/embeddings/ollama_embedder.py`

Нет module docstring.

- Классы: `OllamaEmbedder`
- Функции: нет
- Импорты: `__future__`, `requests`, `typing`

### `rag_indexer_tool/main.py`

Нет module docstring.

- Классы: нет
- Функции: `load_config`, `build_chunk_config`, `chunk_file`, `run_chunk`, `load_chunks`, `run_embed`, `interactive_args`, `build_parser`, `main`
- Импорты: `__future__`, `argparse`, `chunkers.base`, `chunkers.code_chunker`, `chunkers.docx_chunker`, `chunkers.html_chunker`, `chunkers.json_chunker`, `chunkers.markdown_chunker`, `chunkers.pdf_chunker`, `chunkers.table_chunker`, `chunkers.text_chunker`, `datetime`, `json`, `pathlib`, `sys`, `typing`, `utils.file_walker`, `utils.hashing`, `utils.logging`, `yaml`

### `rag_indexer_tool/utils/file_walker.py`

Нет module docstring.

- Классы: нет
- Функции: `iter_files`
- Импорты: `__future__`, `pathlib`

### `rag_indexer_tool/utils/hashing.py`

Нет module docstring.

- Классы: нет
- Функции: `sha256_text`, `sha256_file`
- Импорты: `__future__`, `hashlib`, `pathlib`

### `rag_indexer_tool/utils/logging.py`

Нет module docstring.

- Классы: нет
- Функции: `setup_logging`
- Импорты: `__future__`, `logging`, `pathlib`

### `rag_indexer_tool/utils/token_count.py`

Нет module docstring.

- Классы: нет
- Функции: `_encoding`, `count_tokens`, `token_words`
- Импорты: `__future__`, `functools`, `re`

### `rag_indexer_tool/vector_db/qdrant_store.py`

Нет module docstring.

- Классы: `QdrantStore`
- Функции: нет
- Импорты: `__future__`, `qdrant_client`, `qdrant_client.http`, `typing`, `uuid`

### `support_service.py`

RAG support assistant enriched with user/ticket context from a JSON MCP CRM.

- Классы: `SupportChunk`, `SupportHit`, `SupportRagIndex`, `SupportResponse`, `SupportError`, `SupportAssistant`
- Функции: `tokenize`, `make_http_handler`, `run_http`, `run_interactive`, `build_parser`, `main`
- Импорты: `__future__`, `argparse`, `collections`, `dataclasses`, `http.server`, `json`, `math`, `mcp_client`, `pathlib`, `re`, `typing`
