FROM python:3.11-slim

# default-jre-headless -- нужен ризонеру HermiT (owlready2 sync_reasoner) для
# проверки консистентности Ax, см. docs/chapter2_ontology_model.md §2.1.1
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY docs/ docs/
COPY README.md .

ENV PYTHONPATH=/app/src

CMD ["python3", "-m", "ds_ontology.build"]
