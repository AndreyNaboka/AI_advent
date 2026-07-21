# Использования `MCPCRMClient`

Автоматически создано файловым ассистентом через MCP search/read/write tools.

## Сводка

- Просканировано файлов: 56
- Файлов с совпадениями: 8
- Всего совпадений: 12
- Результат поиска обрезан: нет

## Найденные места

### `README.md`

Категория: документация; строк в файле: 118.

- строка 112: **ссылка** — `--goal 'Найди все использования компонента "MCPCRMClient" и подготовь отчёт'`

### `docs/file_assistant.md`

Категория: документация; строк в файле: 75.

- строка 25: **ссылка** — `--goal 'Найди все использования компонента "MCPCRMClient" и подготовь отчёт'`
- строка 30: **ссылка** — `1. ищет 'MCPCRMClient' в Python, JavaScript, TypeScript и Markdown;`

### `docs/generated_code_inventory.md`

Категория: документация; строк в файле: 252.

- строка 34: **ссылка** — `- Классы: 'MCPClientError', 'MCPNewsClient', 'MCPPeriodicSummaryClient', 'MCPCodeReviewClient', 'MCPProjectClient', 'MCPCRMClient', 'MCPFileToolsClient'`

### `docs/support_assistant.md`

Категория: документация; строк в файле: 96.

- строка 8: **ссылка** — `2. 'MCPCRMClient' получает профиль пользователя и тикет из read-only JSON CRM`

### `file_assistant.py`

Категория: исходный код; строк в файле: 319.

- строка 111: **ссылка** — `raise FileAssistantError("Укажите компонент в кавычках, например \"MCPCRMClient\"")`

### `mcp_client.py`

Категория: исходный код; строк в файле: 166.

- строка 151: **определение** — `class MCPCRMClient(MCPNewsClient):`

### `support_service.py`

Категория: исходный код; строк в файле: 345.

- строка 16: **импорт** — `from mcp_client import MCPClientError, MCPCRMClient`
- строка 135: **ссылка** — `crm_client: MCPCRMClient | None = None,`
- строка 138: **создание/вызов** — `self.crm = crm_client or MCPCRMClient()`

### `tests/test_support_service.py`

Категория: тест; строк в файле: 110.

- строка 7: **импорт** — `from mcp_client import MCPCRMClient`
- строка 21: **создание/вызов** — `client = MCPCRMClient()`
