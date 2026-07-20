# Автоматическое AI-ревью Git diff

## Назначение

`pr_review.py` реализует локальный pipeline проверки нового PR:

1. получает unified diff из файла, рабочего дерева или пары Git refs;
2. извлекает изменённые файлы и номера добавленных/удалённых строк;
3. строит RAG-индекс из `README.md`, `docs/` и исходного кода проекта;
4. сопоставляет каждую находку с контекстом документации и кода;
5. возвращает Markdown с потенциальными багами, архитектурными проблемами и
   рекомендациями.

## Воспроизводимый пользовательский сценарий

Из корня репозитория запустите подготовленный проблемный PR:

```bash
venv312/bin/python pr_review.py \
  --project-root tests/fixtures/pr_review/project \
  --diff-file tests/fixtures/pr_review/problematic_pr.diff \
  --output /tmp/ai-review.md
```

В начале отчёта должны быть указаны `app/payment_service.py`, число добавленных
и удалённых строк, а также количество чанков документации и кода.

В разделе «Потенциальные баги» ожидаются как минимум:

- секрет `PAYMENT_API_KEY` в исходном коде;
- `verify=False`, отключающий проверку TLS;
- HTTP-запрос без timeout;
- bare `except`.

В разделе «Архитектурные проблемы» ожидаются удалённая проверка положительной
суммы и wildcard import. В рекомендациях должно быть требование добавить тесты.
У находок выводится `RAG-контекст`, например `docs/architecture.md` и
`app/payment_service.py`.

Сохранённый отчёт можно открыть командой:

```bash
sed -n '1,240p' /tmp/ai-review.md
```

## Проверка настоящего diff

Незакоммиченные и staged-изменения относительно `HEAD`:

```bash
venv312/bin/python pr_review.py
```

Сравнение ветки PR с основной веткой:

```bash
venv312/bin/python pr_review.py --base-ref main --head-ref feature/my-change
```

Для CI можно завершать команду с кодом `1`, если найдена проблема заданной или
более высокой важности:

```bash
venv312/bin/python pr_review.py \
  --base-ref origin/main \
  --head-ref HEAD \
  --fail-on high \
  --output ai-review.md
```

## Режим локальной LLM

После запуска локального OpenAI-compatible сервера черновик можно дополнительно
передать модели:

```bash
venv312/bin/python pr_review.py \
  --diff-file tests/fixtures/pr_review/problematic_pr.diff \
  --project-root tests/fixtures/pr_review/project \
  --llm \
  --api-url http://localhost:8080/v1/chat/completions
```

Если модель недоступна, pipeline не теряет результат: он оставляет локальное
RAG-review и добавляет диагностическое сообщение.

## Автоматические тесты

```bash
venv312/bin/python -m unittest tests.test_pr_review -v
```

Тесты проверяют parser diff, наличие документации и кода в RAG, все три раздела
review и запись итогового Markdown-файла.
