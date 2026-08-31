"""Централизованная конфигурация сервиса из переменных окружения.

Раньше настройки (`DS_API_KEY` и т.п.) читались через разрозненные
`os.environ.get(...)` по разным модулям -- список того, что вообще можно
настроить, существовал только в README-прозе. Этот модуль -- единая точка
входа: что читается, какое значение по умолчанию, что оно значит.
См. также `.env.example` -- перечень переменных с комментариями для оператора.
"""

import os


# "dev" (по умолчанию, локальная разработка/тесты/docker compose run --rm test)
# или "production". В production ужесточаются проверки при старте (см.
# service/app.py: lifespan) -- сервис не поднимется молча в небезопасной или
# вводящей в заблуждение конфигурации (без ключа доступа; без предобученной
# модели L4 на диске -- см. README "Развёртывание").
def is_production() -> bool:
    return os.environ.get("DS_ENV", "dev").strip().lower() == "production"


# Снимки на момент импорта -- см. предупреждение у API_KEY ниже: для кода,
# которому важна возможность подменить окружение после импорта (тесты,
# lifespan-проверка в app.py), используйте is_production()/get_api_key().
ENV = os.environ.get("DS_ENV", "dev").strip().lower()
IS_PRODUCTION = is_production()


# См. service/auth.py -- если не задан, API открыт (поведение по умолчанию для
# локальной разработки). В production обязателен -- проверяется при старте.
#
# Сознательно функция, а не замороженная на импорте константа: require_api_key
# (auth.py) вызывает get_api_key() на каждый запрос, поэтому переменную
# окружения можно менять после старта процесса -- это и тестовое удобство
# (monkeypatch.setenv в tests/test_auth.py), и корректное поведение как
# таковое (значение end-to-end читается из текущего окружения, а не кэшируется
# на весь срок жизни процесса).
def get_api_key() -> str | None:
    return os.environ.get("DS_API_KEY")


# Снимок на момент импорта -- ТОЛЬКО для одноразовой проверки при старте
# приложения (service/app.py, lifespan); в require_api_key используйте
# get_api_key(), а не эту константу.
API_KEY = get_api_key()

# Список origin, которым разрешён CORS (через запятую). По умолчанию -- порт
# дев-сервера Vite. В production фронт собирается статикой и отдаётся тем же
# приложением (см. service/app.py) -- CORS там не участвует, но переменная
# остаётся настраиваемой на случай отдельного фронтенд-деплоя (CDN и т.п.).
_cors_env = os.environ.get("DS_CORS_ORIGINS")
CORS_ORIGINS = (
    [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
    if _cors_env
    else ["http://localhost:5173", "http://127.0.0.1:5173"]
)

# Путь к обученной модели L4 (см. service/routers/prediction.py). В production
# при её отсутствии сервис отказывается стартовать, а не тихо обучается на
# синтетике -- это сознательное отличие от dev/smoke-режима.
PREDICTION_MODEL_PATH = os.environ.get("DS_MODEL_PATH", "build/models/l4_gradient_boosting.joblib")

# Уровень логирования (см. service/logging_config.py).
LOG_LEVEL = os.environ.get("DS_LOG_LEVEL", "INFO").strip().upper()
