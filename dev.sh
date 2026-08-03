#!/usr/bin/env bash
# Удобный запуск для разработки: backend (FastAPI, порт 8000, --reload) и
# frontend (Vite, порт 5173, hot reload) одновременно, из одной команды.
# Для production/демо без hot reload -- docker compose up app (см. README.md).
set -e
set -m   # job control on -> каждый фоновый `&`-джоб получает свою process group,
         # что даёт возможность убить его целиком (включая внуков вроде vite,
         # которых npm run dev не убивает сам) через `kill -- -PID`
cd "$(dirname "$0")"

if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "Не установлены зависимости API. Выполните: pip install -e \".[dev,api]\"" >&2
    exit 1
fi

if [ ! -d frontend/node_modules ]; then
    echo "=== frontend/node_modules не найден, ставлю зависимости (один раз) ==="
    (cd frontend && npm install)
fi

cleaned_up=0
cleanup() {
    [ "$cleaned_up" = 1 ] && return
    cleaned_up=1
    echo
    echo "=== Останавливаю backend и frontend ==="
    # Убиваем всю process group каждого джоба (-PID), а не только сам PID --
    # `npm run dev` не передаёт сигнал дочернему vite, точечный kill его не трогает.
    kill -- -"$BACKEND_PID" -"$FRONTEND_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

python3 -m uvicorn service.app:app --reload --port 8000 &
BACKEND_PID=$!
(cd frontend && npm run dev -- --port 5173) &
FRONTEND_PID=$!

echo
echo "=== Backend:  http://localhost:8000/docs ==="
echo "=== Frontend: http://localhost:5173 ==="
echo "=== Ctrl+C останавливает оба процесса ==="
echo

wait
