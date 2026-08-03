# Фронт DS (React + TypeScript + Vite)

Лёгкий SPA поверх `service/` (FastAPI, L3-L5 + каталог датасетов + статистика
обучения) -- три вкладки, без клиентского роутера (переключениеState в
`src/App.tsx`, отдельных URL не заводили -- страниц немного, и глубокие
ссылки не нужны):

- **Обучение** (`src/pages/TrainingPage.tsx`) -- графики loss/accuracy/F1 по
  эпохам для NER и классификатора поражений, читает
  `GET /training/history` (сервис читает `build/*/history.json`, которые
  `training/finetune_*.py` пишут по завершении обучения).
- **Тест на точке** (`src/pages/PredictPage.tsx`) -- форма из 10 признаков
  (§2.3.2), вызывает `POST /prediction/predict`, показывает прогноз и вклад
  модальностей по SHAP (§2.3.4) в виде столбчатой диаграммы.
- **Датасеты** (`src/pages/DatasetsPage.tsx`) -- список карточек датасетов
  (`docs/dataset_cards/`) сопоставленный с тем, что реально лежит в `data/`
  (`GET /datasets`), плюс форма загрузки нового датасета архивом
  (`POST /datasets/upload`).

Графики -- `recharts`, больше сторонних UI-библиотек нет намеренно: страниц
и состояний немного, велосипед своего дизайн-системы не нужен.

## Разработка

Нужны два процесса одновременно -- бэкенд (FastAPI, порт 8000) и фронт (Vite
dev-сервер, порт 5173, с hot reload):

```bash
# в корне репозитория
pip install -e ".[dev,api]"
python3 -m uvicorn service.app:app --reload --port 8000

# в отдельном терминале
cd frontend
npm install
npm run dev    # http://localhost:5173
```

CORS для `localhost:5173` уже разрешён в `service/app.py` -- обращаться к API
можно сразу, без прокси. `src/api.ts` сам выбирает базовый URL: абсолютный
`http://localhost:8000` в dev (`import.meta.env.DEV`), пустая строка (тот же
origin) в production-сборке.

## Production-сборка

```bash
npm run build    # -> frontend/dist/
```

`service/app.py` при старте проверяет `frontend/dist/` и, если она собрана,
монтирует её статикой на `/` -- то есть после `npm run build` тот же
`uvicorn service.app:app` отдаёт и API, и фронт с одного порта. Именно так
собран `Dockerfile` (двухстадийная сборка: `node:20-slim` собирает
`frontend/dist`, затем копируется в финальный Python-образ -- Node в
финальном образе нет). Без сборки фронта сервис остаётся чистым API, как и
раньше -- это не ломает `docker compose run --rm test`.

## Структура

```
src/
  api.ts               -- типизированные обёртки над эндпоинтами service/
  App.tsx              -- вкладки (без роутера)
  pages/
    TrainingPage.tsx
    PredictPage.tsx
    DatasetsPage.tsx
  index.css             -- общие стили (без CSS-фреймворка)
```
