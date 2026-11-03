#!/bin/sh
# Запуск UI лог-анализатора под macOS/Linux.
cd "$(dirname "$0")" || exit 1

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
    echo "Нет .venv — создаю и ставлю зависимости..."
    python3 -m venv .venv || exit 1
    "$PY" -m pip install -r requirements.txt || exit 1
fi
exec "$PY" -m log_analyser.ui
