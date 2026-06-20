FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HQCA_LOG_FORMAT=json \
    HQCA_LOG_LEVEL=INFO \
    HQCA_OUTPUT_CSV=/app/output/hqca_synthetic_data.csv \
    HQCA_OUTPUT_JSON=/app/output/hqca_synthetic_data.json \
    HQCA_METRICS_FILE=/app/output/hqca_metrics.json \
    HQCA_REPORT_FILE=/app/output/data_report.txt

WORKDIR /app
ARG INSTALL_QUANTUM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md data.py ./
COPY scripts ./scripts

RUN python -m pip install --no-cache-dir --upgrade pip \
    && if [ "$INSTALL_QUANTUM" = "true" ]; then \
        python -m pip install --no-cache-dir ".[quantum]"; \
    else \
        python -m pip install --no-cache-dir "."; \
    fi

COPY . .

RUN mkdir -p /app/output

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python scripts/healthcheck.py || exit 1

CMD ["python", "data.py"]
