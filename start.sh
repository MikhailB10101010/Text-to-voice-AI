#!/bin/bash
set -e  # Прерывать выполнение при ошибке

# sudo apt update
sudo apt install python3.12-venv

# Проверка наличия Python
if ! command -v python3 &> /dev/null; then
    echo "Ошибка: Python3 не установлен"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Создание виртуального окружения
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Активация в зависимости от ОС
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    ACTIVATE_PATH=".venv/Scripts/activate"
else
    ACTIVATE_PATH=".venv/bin/activate"
fi

if [ -f "$ACTIVATE_PATH" ]; then
    source "$ACTIVATE_PATH"
else
    echo "Ошибка: Не найден файл активации: $ACTIVATE_PATH"
    exit 1
fi

# Установка зависимостей
if [ -f "requirements.txt" ]; then
    # echo "Устанавливаю зависимости..."
    pip install --upgrade pip
    pip install -r requirements.txt
    # echo "Зависимости установлены"
else
    echo "Файл requirements.txt не найден"
fi
