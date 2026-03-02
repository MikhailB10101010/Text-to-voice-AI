#!/bin/bash

# rand id
random_id=${1:-123}

# нагромождение
echo 'Проверка библиотек'
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
cd ..
source start.sh > /dev/null 2>&1
echo 'Ок!'
echo "Запуск $0"
echo "rand_id = $random_id"

#
source .venv/bin/activate

cd lab1
python data_creation.py "$random_id"
python data_preprocessing.py
python model_preparation.py "$random_id"
python model_testing.py
