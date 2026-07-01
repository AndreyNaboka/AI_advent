#!/usr/bin/env bash
set -euo pipefail

OS="$(uname -s)"

install_macos() {
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew не найден. Установите его с https://brew.sh и запустите скрипт снова."
    exit 1
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "Устанавливаю Docker Desktop через Homebrew Cask..."
    brew install --cask docker
  else
    echo "Docker уже установлен."
  fi

  if ! command -v ollama >/dev/null 2>&1; then
    echo "Устанавливаю Ollama через Homebrew..."
    brew install ollama
  else
    echo "Ollama уже установлен."
  fi
}

install_linux() {
  if command -v apt-get >/dev/null 2>&1; then
    echo "Устанавливаю Docker и curl через apt..."
    sudo apt-get update
    sudo apt-get install -y docker.io curl
    sudo systemctl enable --now docker || true
  else
    echo "Автоустановка Docker поддержана только для apt-based Linux."
    echo "Установите Docker вручную: https://docs.docker.com/engine/install/"
  fi

  if ! command -v ollama >/dev/null 2>&1; then
    echo "Устанавливаю Ollama официальным install script..."
    curl -fsSL https://ollama.com/install.sh | sh
  else
    echo "Ollama уже установлен."
  fi
}

case "$OS" in
  Darwin)
    install_macos
    ;;
  Linux)
    install_linux
    ;;
  *)
    echo "Неподдерживаемая ОС: $OS"
    exit 1
    ;;
esac

echo ""
echo "Готово. Следующие шаги:"
echo "1. Запустите Docker Desktop на macOS, если он установлен впервые."
echo "2. Запустите Qdrant:"
echo "   docker run -p 6333:6333 -v \"\$PWD/qdrant_storage:/qdrant/storage\" qdrant/qdrant"
echo "3. Установите embedding-модель:"
echo "   ollama pull nomic-embed-text"
