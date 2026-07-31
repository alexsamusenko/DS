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

RUN pip install --no-cache-dir -e ".[dev,api]"

CMD ["python3", "-m", "ds_ontology.build"]
