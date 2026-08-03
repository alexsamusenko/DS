# DS — практическая часть диссертации

Прототип программного комплекса для комплексного анализа слабоструктурированных агрономических данных. Архитектура и диаграммы: см. опубликованный артефакт «Архитектура DS» (ссылка — в переписке). Подробное описание каждого модуля — `docs/project_overview.md`. Лицензия кода — MIT (`LICENSE`); лицензии сторонних данных/моделей/библиотек — `docs/governance/licenses.md`.

## Статус

| Уровень | Описание | Статус |
|---|---|---|
| L2 | Семантическая интеграция: динамическая онтология, граф знаний | формализовано + прототип (`docs/chapter2/ontology_model.md`, `src/ds_ontology/`) |
| L3 | Комбинированная предобработка: детекция аномалий + кригинг + временной тренд | формализовано + прототип (`docs/chapter2/preprocessing_model.md`, `src/ds_preprocessing/`) |
| L4 | Мультимодальный прогноз урожайности (num/geo/img/text) + SHAP | формализовано + прототип (`docs/chapter2/prediction_model.md`, `src/ds_prediction/`) |
| L5 | Оптимизация дифференцированного внесения удобрений | формализовано + прототип (`docs/chapter2/optimization_model.md`, `src/ds_optimization/`) |
| L1 (модели) | Стратегия моделей: NER, детекция по фото, LLM только для открытых задач | формализовано (`docs/chapter2/model_training.md`), скрипты — `training/` |
| L6 | Единый API поверх L3-L5 | реализовано (`service/`) |
| L7 | Интеграция с внешними ИС: REST/OpenAPI + экспорт в ISO 11783-10 (ISOXML) | формализовано + прототип (`docs/chapter2/integration_model.md`, `src/ds_integration/`) |
| Фронт | Веб-интерфейс: статистика обучения, тест прогноза на точке, каталог датасетов | реализовано (`frontend/`) |
| L0 | Источники (реальные открытые данные вместо синтетики) | не начаты (задача 6) |

## Структура

```
docs/
  project_overview.md          -- подробное описание назначения каждого модуля (начните отсюда)
  chapter2/                    -- формальные постановки §2.1-2.6
    ontology_model.md            -- O_t = <C, R_O, R_D, Ax, I, tau> (§2.1)
    preprocessing_model.md       -- комбинированное восстановление пропусков (§2.2)
    prediction_model.md          -- мультимодальный прогноз + SHAP (§2.3)
    optimization_model.md        -- дифференцированное внесение удобрений (§2.4)
    model_training.md            -- стратегия моделей: что дообучаем, что промптим (§2.5)
    integration_model.md         -- интеграция с внешними ИС: REST + ISO 11783-10 (§2.6)
  governance/                  -- документация, обязательная для каждого добавляемого актива
    licenses.md                  -- реестр лицензий (код, зависимости, данные, модели)
    best_practices.md            -- принятые отраслевые стандарты (FAIR, Model Cards, ISOBUS)
  model_cards/, dataset_cards/  -- карточки моделей/датасетов
  datasets.md                  -- открытые источники данных по уровням архитектуры

src/
  ds_ontology/       -- L2: schema.py (TBox), integration.py (mu = phi union lambda), build.py
  ds_preprocessing/  -- L3: anomaly.py, spatial.py (кригинг), temporal.py, combine.py, validation.py
  ds_prediction/     -- L4: features.py, model.py (Gradient Boosting + GroupKFold), explain.py (SHAP)
  ds_optimization/   -- L5: response.py (Митчерлих), optimize.py (аналитика + Лагранж), validation.py
  ds_integration/    -- L7: isoxml_export.py (ISO 11783-10 Task Data), geometry.py

training/  -- скрипты дообучения (NER, классификатор поражений), см. docs/chapter2/model_training.md
service/   -- единый FastAPI-сервис поверх L3-L5, L7-экспорт, каталог датасетов + статистика обучения
frontend/  -- React+TS SPA поверх service/ (обучение / тест на точке / внесение удобрений + экспорт / датасеты), см. frontend/README.md
tests/     -- test_schema.py, test_preprocessing.py, test_prediction.py, test_optimization.py, test_integration.py, test_service.py, test_frontend_api.py
```

Фото и голосовые записи не образуют отдельных модальностей: фото проходит детекцию признаков поражения (`training/finetune_leaf_classifier.py`) и становится экземпляром `ВредительБолезнь` (модальность `img`), голос распознаётся речь-в-текст (ASR) и обрабатывается тем же NER-конвейером, что и письменный текст (модальность `text`) — детали в `docs/chapter2/model_training.md`.

## Запуск

### Локально (Python)

```bash
pip install -e ".[dev]"                                   # ставит пакет в editable-режиме + pytest
python3 -m ds_ontology.build                               # соберёт build/agro_demo.owl
python3 -m ds_preprocessing.build_demo                     # сравнение RMSE методов восстановления
python3 -m ds_prediction.build_demo                        # сравнение по модальностям + вклад SHAP
python3 -m ds_optimization.build_demo                      # дифференцированное внесение vs равномерное
python3 -m ds_integration.build_demo                        # экспорт карты-задания в ISO 11783-10 -> build/integration/TASKDATA.xml
python3 -m pytest tests/ -v
```

Без установки пакета (без `pip install -e`) команды тоже работают с префиксом `PYTHONPATH=src`.

### Единый API

```bash
pip install -e ".[api]"
python3 -m uvicorn service.app:app --reload
```

Документация (Swagger UI, автогенерируется из Pydantic-схем) — `http://localhost:8000/docs`. Маршруты: `POST /preprocessing/fill-gaps`, `POST /prediction/predict`, `GET /prediction/training-summary`, `POST /optimization/optimize`, `POST /integration/export-isoxml`, `GET /training/history`, `GET /datasets`, `POST /datasets/upload`. Подробности — `docs/project_overview.md`, раздел `service/`.

### Интеграция с внешними ИС (L7)

Результат L5 (карта-задание) доступен не только через собственный `frontend/`, но и напрямую внешним системам агрохолдинга — два канала (`docs/chapter2/integration_model.md`, §2.6):

- **REST/OpenAPI** — тот же `/openapi.json`, что и у остальных маршрутов: любая внешняя ИС, умеющая HTTP/JSON, читает машиночитаемый контракт без обращения к исходному коду.
- **ISO 11783-10 (ISOBUS Task Data)** — `POST /integration/export-isoxml` принимает те же данные, что и `/optimization/optimize`, и возвращает готовый `TASKDATA.XML` — формат, который понимают бортовые терминалы сельхозтехники и FMIS независимо от производителя, без частных коннекторов под конкретную систему.

```bash
curl -X POST http://localhost:8000/integration/export-isoxml \
  -H "Content-Type: application/json" \
  -d '{"plots": [{"plot_id": 0, "baseline": 40, "R": 8, "s": 60, "area": 1.2}],
       "price_yield": 1300, "price_fert": 50}' \
  -o TASKDATA.xml
```

**Доступ.** По умолчанию API открыт (как и раньше) — для локального прототипа этого достаточно. Если сервис реально открывается внешней системе (не localhost), задайте `DS_API_KEY` в окружении перед запуском — все маршруты, кроме `/health`, потребуют заголовок `X-API-Key` с этим значением (`service/auth.py`, §2.6.4). Без переменной поведение не меняется. Для `docker compose`: `DS_API_KEY=<значение> docker compose up app`. Если фронт должен продолжать работать при включённом ключе, соберите его с тем же значением в `VITE_API_KEY` (`frontend/.env` или переменная окружения при `npm run build`).

Геометрия участков в экспорте — синтетическая (в системе нет реальных границ полей хозяйства, см. `docs/governance/licenses.md`); код показателя DDI для дозы внесения требует сверки с официальным реестром AEF ISOBUS перед использованием с реальной техникой — оба ограничения задокументированы явно, не молча.

### Веб-интерфейс (frontend/)

Четыре вкладки поверх API выше — статистика обучения (графики loss/accuracy/F1 по эпохам и гиперпараметры из `build/*/history.json`), ручной тест прогноза на точке (форма с готовыми пресетами сценариев + вклад модальностей по SHAP + сравнение прогнозов), внесение удобрений (редактируемая таблица участков → оптимальные дозы → скачивание карты-задания в ISO 11783-10, L5+L7 живьём), каталог датасетов (разворачиваемая карточка — источник/состав/лицензия/ограничения — + наличие на диске + загрузка нового датасета архивом).

Удобный запуск для разработки — один скрипт из корня репозитория:

```bash
./dev.sh   # backend :8000 (--reload) + frontend :5173 (hot reload), Ctrl+C останавливает оба
```

Ставит `frontend/node_modules` при первом запуске, если их ещё нет. Требует, чтобы `pip install -e ".[dev,api]"` уже был выполнен. Подробности — `frontend/README.md`. Для production сборка (`npm run build`) отдаётся тем же `uvicorn service.app:app` с того же порта — отдельный контейнер/порт не нужен, `Dockerfile` уже собирает фронт двухстадийно (см. ниже).

### В Docker

Не требует локального Python/Java — только Docker. Четыре независимых сценария, каждый — отдельный сервис в `docker-compose.yml` (сборка описания: `docker compose config`):

| Сервис | Назначение | Образ |
|---|---|---|
| `app` | запуск единого API (L3-L5, L7) + веб-интерфейс (frontend/) на том же порту | `Dockerfile` (лёгкий: без torch/transformers; фронт собирается двухстадийно, Node только на стадии сборки) |
| `test` | полный набор тестов (`pytest tests/ -v`) | `Dockerfile` |
| `validate` | формальная проверка каждого модуля L2-L5 — сборка всех демо-примеров подряд | `Dockerfile` |
| `train` | дообучение моделей (`training/`) | `Dockerfile.training` (отдельный, тяжёлый: torch, torchvision, transformers, datasets, accelerate) |

Разделение на два образа осознанное: тяжёлые ML-зависимости нужны только для обучения, и не должны раздувать образ API/тестов.

```bash
# API -- демон, поднимается и слушает порт
docker compose up app                                        # http://localhost:8000/docs

# Тесты и валидация -- разовый прогон, контейнер завершается сам
docker compose run --rm test
docker compose run --rm validate                             # -> ./build/*.owl и сравнения RMSE/SHAP в stdout

# Обучение -- разовый прогон, тяжёлый образ (torch и т.п.)
docker compose run --rm train                                # по умолчанию: NER smoke-test на синтетическом корпусе
```

`./build/` (для `validate` и `train`) и `./data/` (для `train`) на хосте примонтированы в контейнер как `/app/build` и `/app/data`. Оба скрипта дообучения по умолчанию пишут чекпоинты через `--output-dir build/...` (относительный путь, разрешается в `/app/build/...` внутри контейнера) — то есть веса **не теряются с удалением контейнера**, а сразу оказываются на хосте: `./build/ner_model/` и `./build/leaf_classifier/best_model.pt`. Контейнер `train` можно свободно удалять после каждого запуска (`docker compose run --rm ...` уже это делает) — результат остаётся в проекте.

Для реального (не smoke-test) обучения на своих данных ничего переопределять не нужно — оба скрипта работают по принципу «положил файлы в стандартное место — запустил без флагов»:

- NER: положите размеченные `ner_train.jsonl` и `ner_eval.jsonl` в `./data/` на хосте.
- Классификатор поражений: распакуйте датасет (например, PlantDoc — `docs/dataset_cards/plantdoc.md`) в `./data/plantdoc/` в формате `train/<class>/*.jpg` + `test/<class>/*.jpg` (или `val/`, определяется автоматически).

Затем:

```bash
docker compose run --rm train python3 training/finetune_ner.py
docker compose run --rm train python3 training/finetune_leaf_classifier.py --epochs-head 10 --epochs-full 20
```

Оба скрипта по умолчанию скачивают предобученные веса (`DeepPavlov/rubert-base-cased` для NER, `EfficientNet_B0_Weights.IMAGENET1K_V1` для классификатора — нужен доступ в сеть при первом запуске; источники и лицензии — `docs/governance/licenses.md`, раздел «Предобученные веса моделей»). Другие пути к данным — через `--train`/`--eval` (NER) или `--data-dir`/`--val-subdir` (классификатор). Флаги `--no-pretrained` (классификатор) / `--smoke-test` (оба скрипта) — офлайн-резерв на случай, если download.pytorch.org/huggingface.co недоступны из текущей сети (как в песочнице разработки этого проекта).

**GPU**: если на хосте установлен `nvidia-container-toolkit`, раскомментируйте блок `deploy.resources.reservations` для сервиса `train` в `docker-compose.yml` — официальные wheel'ы `torch` с PyPI уже содержат поддержку CUDA, отдельный образ на базе `nvidia/cuda` не требуется.

### Дообучение моделей (без Docker)

```bash
pip install -e ".[training]"    # torch, torchvision, transformers -- тяжёлые зависимости, отдельная группа
python3 training/finetune_ner.py --smoke-test --epochs 30 --batch-size 16 --lr 3e-4
python3 training/finetune_leaf_classifier.py --smoke-test --epochs-head 8 --epochs-full 12
```

`--smoke-test` в обоих скриптах работает офлайн (huggingface.co и download.pytorch.org недоступны из текущей среды разработки) на программно сгенерированных данных, со случайной инициализацией модели — подтверждённая сходимость (NER: F1 0→1.0 за ~20 эпох; классификатор: точность 0.25→0.55 за 20 эпох). Продовый режим (`--train/--eval` для NER, `--data-dir` для классификатора) использует предобученные веса **по умолчанию** (`DeepPavlov/rubert-base-cased` для NER, `EfficientNet_B0_Weights.IMAGENET1K_V1` для классификатора — источники и лицензии в `docs/governance/licenses.md`) и требует реальных данных — гиперпараметры и источники данных (PlantDoc, CC BY 4.0) — `docs/chapter2/model_training.md`. Реальный прогон на PlantDoc (`--data-dir data/plantdoc --val-subdir test`) выполняется локально пользователем с доступом к download.pytorch.org — код и датасет готовы, точное обучение вне песочницы разработки.

## Дальше по ритму

Формальные постановки L2-L5, стратегия моделей (L1) и единый API (L6) готовы. Следующий шаг — задача 6: апробация на данных конкретного хозяйства вместо контролируемых синтетических наборов, использованных для валидации алгоритмов на этапе разработки (см. `docs/datasets.md`).
