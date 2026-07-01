#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

find_python() {
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
      then
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON_BIN="$(find_python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "Ошибка: нужен Python 3.10 или новее."
  echo "macOS: brew install python@3.12"
  echo "Linux: установите python3 и python3-venv через пакетный менеджер."
  exit 1
fi

cd "$SCRIPT_DIR"

if [ ! -d "$VENV_DIR" ]; then
  echo "Создаю виртуальное окружение: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "Python-зависимости установлены."
echo ""
echo "Дополнительные приложения:"
echo "1. Docker нужен для локального Qdrant: https://docs.docker.com/get-docker/"
echo "   Запуск Qdrant из этой папки:"
echo "   docker run -p 6333:6333 -v \"\$PWD/qdrant_storage:/qdrant/storage\" qdrant/qdrant"
echo ""
echo "2. Ollama нужен для локальных embeddings: https://ollama.com/download"
echo "   Установка embedding-модели:"
echo "   ollama pull nomic-embed-text"
echo ""
echo "Best-effort автоустановка Docker/Ollama:"
echo "  ./install_system_tools.sh"
echo ""
echo "Проверка скрипта:"
echo "  $VENV_DIR/bin/python main.py --help"
