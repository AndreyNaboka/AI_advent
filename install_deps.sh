#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv312"

cd "$PROJECT_DIR"

find_python() {
    for candidate in python3.12 python3.11 python3.10; do
        if command -v "$candidate" &> /dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

PYTHON_BIN=$(find_python || true)

if [ -z "$PYTHON_BIN" ]; then
    echo "Ошибка: нужен Python 3.10 или новее."
    echo "Сейчас найден только: $(python3 --version 2>/dev/null || echo 'python3 не найден')"
    echo ""
    echo "Самый простой вариант на macOS:"
    echo "  brew install python@3.12"
    echo "  rm -rf venv312"
    echo "  ./install_deps.sh"
    echo ""
    echo "Если Homebrew не установлен: https://brew.sh"
    exit 1
fi

if [ -x "$VENV_DIR/bin/python" ]; then
    VENV_VERSION=$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
        :
    else
        echo "Найден старый venv на Python $VENV_VERSION, пересоздаю его."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Создаю виртуальное окружение: $VENV_DIR ($($PYTHON_BIN --version))"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Обновляю pip..."
python -m pip install --upgrade pip

echo "Устанавливаю зависимости из requirements.txt..."
python -m pip install -r requirements.txt

echo ""
echo "Готово."
echo "Проверка:"
echo "  ./venv312/bin/python -c 'import requests, jsonschema, llama_cpp; print(\"ok\")'"
echo ""
echo "Запуск клиента:"
echo "  python3 main.py --server <IP_СЕРВЕРА>:8080"
