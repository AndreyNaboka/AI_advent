# Мини-чат с RAG, источниками и памятью задачи

Код: `mini_rag_chat.py`

Интерактивный запуск:

```bash
venv312/bin/python mini_rag_chat.py
```

Проверка двух длинных сценариев по 13 сообщений:

```bash
venv312/bin/python mini_rag_chat.py --run-scenarios
```

Отчёт создаётся в `mini_chat_scenario_report.json`. Он проверяет, что на каждом
ходе есть блоки `Источники` и `Цитаты`, а цель диалога остаётся в task state.

Запись видео демо:

```bash
asciinema rec mini_rag_chat.cast --command "venv312/bin/python mini_rag_chat.py --run-scenarios"
```

Если нужен обычный mp4/webm, можно записать этот же запуск любым screen recorder.
