# RAG Indexer Tool

Локальный индексатор для построения chunks и embeddings из произвольной папки.

## Установка

```bash
cd rag_indexer_tool
chmod +x install.sh
./install.sh
```

Нужны внешние приложения:

- Docker для Qdrant.
- Ollama для локальных embeddings.

Best-effort установка Docker/Ollama:

```bash
./install_system_tools.sh
```

Запуск Qdrant:

```bash
docker run -p 6333:6333 -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Установка embedding-модели:

```bash
ollama pull nomic-embed-text
```

## Использование

Интерактивно:

```bash
./.venv/bin/python main.py
```

Создать chunks:

```bash
./.venv/bin/python main.py chunk --input ./docs --output ./chunks.jsonl
```

Построить embeddings и загрузить их в Qdrant:

```bash
./.venv/bin/python main.py embed --chunks ./chunks.jsonl
```

Настройки находятся в `config.yaml`.
