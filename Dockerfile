# Стадия 1: сборка фронта (service/ отдаёт его статикой, см. service/app.py) --
# Node нужен только здесь, в финальном образе его нет.
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim

# default-jre-headless -- нужен ризонеру HermiT (owlready2 sync_reasoner) для
# проверки консистентности Ax, см. docs/chapter2/ontology_model.md §2.1.1
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml LICENSE README.md ./
COPY src/ src/
COPY tests/ tests/
COPY docs/ docs/
COPY service/ service/
COPY --from=frontend-build /frontend/dist frontend/dist

RUN pip install --no-cache-dir -e ".[dev,api]"

# Непривилегированный пользователь -- контейнер (в т.ч. JRE-ризонер HermiT,
# запускаемый owlready2) не должен работать от root без необходимости.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/build /app/data \
    && chown -R appuser:appuser /app
USER appuser

# Проверяет именно готовность обслуживать запросы (см. service/app.py, /health
# и /ready), а не только "процесс жив" -- используется docker-compose.yml
# и любым внешним оркестратором, который умеет читать статус контейнера.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["python3", "-m", "ds_ontology.build"]
