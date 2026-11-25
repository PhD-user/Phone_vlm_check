#!/usr/bin/env bash
# Запускайте через bash; архив не обязан сохранять исполняемый бит файла.
set -euo pipefail
cd "$(dirname "$0")/.."
vlm_device="${1:-cpu}"
case "$vlm_device" in
  cpu) vlm_index="cpu" ;;
  cuda) vlm_index="cu126" ;;
  *) echo "Usage: bash scripts/setup_linux.sh cpu|cuda" >&2; exit 2 ;;
esac
# Можно задать VLM_PYTHON=python3.12. Переменные окружения ОС не переопределяем.
vlm_python="${VLM_PYTHON:-python3}"
if [[ ! -x .venv/bin/python ]]; then
  "$vlm_python" -m venv .venv
fi
# Суффикс версии принудительно выбирает нужную сборку, даже если другая уже есть.
.venv/bin/python -m pip install "torch==2.8.0+$vlm_index" "torchvision==0.23.0+$vlm_index" --index-url "https://download.pytorch.org/whl/$vlm_index"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m phone_vlm doctor
echo "Готово. Перед обучением прочитайте docs/TRAINING.md."
