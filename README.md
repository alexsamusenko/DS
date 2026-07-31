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
| L0 | Источники (реальные открытые данные вместо синтетики) | не начаты (задача 6) |

## Структура

```
docs/
  project_overview.md          -- подробное описание назначения каждого модуля (начните отсюда)
  chapter2/                    -- формальные постановки §2.1-2.5
    ontology_model.md            -- O_t = <C, R_O, R_D, Ax, I, tau> (§2.1)
    preprocessing_model.md       -- комбинированное восстановление пропусков (§2.2)
    prediction_model.md          -- мультимодальный прогноз + SHAP (§2.3)
    optimization_model.md        -- дифференцированное внесение удобрений (§2.4)
    model_training.md            -- стратегия моделей: что дообучаем, что промптим (§2.5)
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

training/  -- скрипты дообучения (NER, классификатор поражений), см. docs/chapter2/model_training.md
service/   -- единый FastAPI-сервис поверх L3-L5
tests/     -- test_schema.py, test_preprocessing.py, test_prediction.py, test_optimization.py, test_service.py
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
python3 -m pytest tests/ -v
```

Без установки пакета (без `pip install -e`) команды тоже работают с префиксом `PYTHONPATH=src`.

### Единый API

```bash
pip install -e ".[api]"
python3 -m uvicorn service.app:app --reload
```

Документация (Swagger UI, автогенерируется из Pydantic-схем) — `http://localhost:8000/docs`. Маршруты: `POST /preprocessing/fill-gaps`, `POST /prediction/predict`, `GET /prediction/training-summary`, `POST /optimization/optimize`. Подробности — `docs/project_overview.md`, раздел `service/`.

### В Docker

Не требует локального Python/Java — только Docker.

```bash
docker compose build
docker compose run --rm ds                                 # демо-сборка -> ./build/agro_demo.owl
docker compose run --rm ds python3 -m pytest tests/ -v      # весь набор тестов, включая ризонер HermiT
docker compose up api                                       # API на http://localhost:8000
```

`./build/` на хосте примонтирован в контейнер, поэтому `agro_demo.owl` появится локально после запуска.

### Дообучение моделей

```bash
pip install -e ".[training]"    # torch, torchvision, transformers -- тяжёлые зависимости, отдельная группа
python3 training/finetune_ner.py --smoke-test --epochs 30 --batch-size 16 --lr 3e-4
python3 training/finetune_leaf_classifier.py --smoke-test --epochs-head 8 --epochs-full 12
```

`--smoke-test` в обоих скриптах работает офлайн (huggingface.co и веса ImageNet недоступны из текущей среды разработки) на программно сгенерированных данных — подтверждённая сходимость (NER: F1 0→1.0 за ~20 эпох; классификатор: точность 0.25→0.55 за 20 эпох). Продовый режим (`--train/--eval` для NER, `--data-dir` для классификатора) требует сети и реальных данных — гиперпараметры и источники данных (PlantDoc, CC BY 4.0) — `docs/chapter2/model_training.md`.

## Дальше по ритму

Формальные постановки L2-L5, стратегия моделей (L1) и единый API (L6) готовы. Следующий шаг — задача 6: апробация на данных конкретного хозяйства вместо контролируемых синтетических наборов, использованных для валидации алгоритмов на этапе разработки (см. `docs/datasets.md`).
