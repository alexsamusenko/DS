# Фронт DS (React + TypeScript + Vite)

Лёгкий SPA поверх `service/` (FastAPI, L3-L5, L7 + каталог датасетов +
статистика обучения) -- четыре вкладки, без клиентского роутера (переключение
state в `src/App.tsx`, отдельных URL не заводили -- страниц немного, и
глубокие ссылки не нужны):

- **Обучение** (`src/pages/TrainingPage.tsx`) -- графики loss/accuracy/F1 по
  эпохам для NER и классификатора поражений, читает
  `GET /training/history` (сервис читает `build/*/history.json`, которые
  `training/finetune_*.py` пишут по завершении обучения).
- **Тест на точке** (`src/pages/PredictPage.tsx`) -- форма из 10 признаков
  (§2.3.2), вызывает `POST /prediction/predict`, показывает прогноз и вклад
  модальностей по SHAP (§2.3.4) в виде столбчатой диаграммы.
- **Внесение удобрений** (`src/pages/OptimizePage.tsx`) -- редактируемая
  таблица участков, вызывает `POST /optimization/optimize` (§2.4, L5) и
  показывает дозы + прирост прибыли против равномерного сценария, плюс кнопка
  скачивания карты-задания в ISO 11783-10 через `POST /integration/export-isoxml`
  (§2.6, L7) -- реальный `<a download>` по Blob из ответа, не заглушка.
- **Датасеты** (`src/pages/DatasetsPage.tsx`) -- список карточек датасетов
  (`docs/dataset_cards/`) сопоставленный с тем, что реально лежит в `data/`
  (`GET /datasets`), плюс форма загрузки нового датасета архивом
  (`POST /datasets/upload`).

Графики -- `recharts`, больше сторонних UI-библиотек нет намеренно: страниц
и состояний немного, велосипед своего дизайн-системы не нужен.

Каждая вкладка -- отдельный чанк (`React.lazy` + `Suspense` в `App.tsx`), а не
общий бандл: `recharts` весит больше всего остального фронта вместе взятого,
и без code-splitting он тянулся бы даже для вкладки «Датасеты», которая
графиков не показывает.

## Разработка

Нужны два процесса одновременно -- бэкенд (FastAPI, порт 8000) и фронт (Vite
dev-сервер, порт 5173, с hot reload). Самый простой способ -- один скрипт из
корня репозитория:

```bash
./dev.sh
```

Ставит `frontend/node_modules` при первом запуске (если их ещё нет), поднимает
оба процесса, печатает оба URL, по Ctrl+C останавливает оба (включая дочерний
`vite`, которого `npm run dev` сам не убивает -- `dev.sh` убивает всю process
group). Требует, чтобы `pip install -e ".[dev,api]"` уже был выполнен --
скрипт проверяет и подсказывает, если нет.

Вручную (два терминала), если нужен больший контроль:

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
    OptimizePage.tsx
    DatasetsPage.tsx
  index.css             -- общие стили (без CSS-фреймворка)
```
