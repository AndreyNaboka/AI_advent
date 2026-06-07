#!/bin/bash

# Переменные для удобства (измените пути, если нужно)
PROJECT_DIR=~/code/AI_advent
VENV_DIR="$PROJECT_DIR/venv"
MODELS_DIR=~/code/models

# --- НАСТРОЙКИ МОДЕЛЕЙ ---
declare -A MODEL_REPOS
declare -A MODEL_FILES

MODEL_REPOS["strong"]="bartowski/Qwen2.5-14B-Instruct-GGUF"
MODEL_FILES["strong"]="Qwen2.5-14B-Instruct-Q4_K_M.gguf"

MODEL_REPOS["medium"]="bartowski/Qwen2.5-7B-Instruct-1M-GGUF"
MODEL_FILES["medium"]="Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"

MODEL_REPOS["weak"]="Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILES["weak"]="qwen2.5-1.5b-instruct-q8_0.gguf"

# --- ФУНКЦИЯ ПОМОЩИ ---
show_help() {
    echo "Использование: $0 {strong|medium|weak}"
    echo ""
    echo "  strong   - Запустить сильную модель (14B, высокое качество)"
    echo "  medium   - Запустить среднюю модель (7B, баланс скорости и качества)"
    echo "  weak     - Запустить слабую модель (1.5B, максимальная скорость)"
    echo ""
    echo "Пример: $0 medium"
    exit 1
}

# --- ФУНКЦИЯ ПРОВЕРКИ И УСТАНОВКИ huggingface_hub ---
ensure_hf_installed() {
    if ! command -v hf &> /dev/null; then
        echo "Утилита hf не найдена. Устанавливаю huggingface_hub..."
        pip install --upgrade huggingface_hub
    fi
}

# --- ФУНКЦИЯ ПОКАЗА ДОСТУПНЫХ GGUF ФАЙЛОВ ---
show_available_files() {
    local repo=$1
    echo "Доступные GGUF файлы в репозитории $repo:"
    # Используем hf models ls --format json и извлекаем имена файлов .gguf
    hf models ls "$repo" --format json 2>/dev/null | grep -o '"path":"[^"]*\.gguf"' | sed 's/"path":"//;s/"//' | sed 's/^/  - /'
    if [ $? -ne 0 ]; then
        echo "  (не удалось получить список файлов, попробуйте позже)"
    fi
}

# --- ПРОВЕРКА АРГУМЕНТОВ ---
if [ $# -ne 1 ]; then
    echo "Ошибка: необходимо указать тип модели (strong, medium, weak)."
    show_help
fi

MODEL_TYPE=$1

if [[ -z "${MODEL_REPOS[$MODEL_TYPE]}" ]]; then
    echo "Ошибка: неизвестный тип модели '$MODEL_TYPE'."
    show_help
fi

# --- АКТИВАЦИЯ ОКРУЖЕНИЯ И ПОДГОТОВКА ---
cd "$PROJECT_DIR" || { echo "Ошибка: папка проекта $PROJECT_DIR не найдена"; exit 1; }
source "$VENV_DIR/bin/activate"

# Убеждаемся, что huggingface_hub установлен
ensure_hf_installed

REPO="${MODEL_REPOS[$MODEL_TYPE]}"
FILE="${MODEL_FILES[$MODEL_TYPE]}"
MODEL_PATH="$MODELS_DIR/$MODEL_TYPE/$FILE"

echo "=== Запуск LLM сервера (Тип: $MODEL_TYPE) ==="
echo "Репозиторий: $REPO"
echo "Модель: $FILE"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Модель не найдена локально."
    
    # Показываем доступные файлы
    show_available_files "$REPO"
    
    echo ""
    read -p "Продолжить скачивание '$FILE'? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Отменено."
        exit 0
    fi
    
    echo "Скачиваю модель: $FILE"
    mkdir -p "$MODELS_DIR/$MODEL_TYPE"
    
    hf download "$REPO" "$FILE" --local-dir "$MODELS_DIR/$MODEL_TYPE"
    
    if [ $? -ne 0 ]; then
        echo "Ошибка при скачивании модели."
        exit 1
    fi
else
    echo "Модель найдена в кэше."
fi

# Запуск сервера с флагом verbose
python3 -m llama_cpp.server \
    --model "$MODEL_PATH" \
    --n_gpu_layers -1 \
    --port 8080 \
    --n_ctx 8192 \
    --verbose true