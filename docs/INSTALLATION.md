# HQCA Installation and Operations Guide

This repository currently ships the HQCA synthetic data pipeline as a Python batch
runtime. The infrastructure files in this guide provide a repeatable local,
Docker, CI, and monitoring baseline.

## Requirements

- Python 3.10 or newer
- `pip`
- Docker 24+ for containerized execution

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the optional quantum simulator dependency when PennyLane execution is
needed:

```bash
python -m pip install -e ".[quantum,dev]"
```

## Environment configuration

Copy the sample file and adjust values as needed:

```bash
cp .env.example .env
```

Supported environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `HQCA_LOG_LEVEL` | `INFO` | Python logging level. |
| `HQCA_LOG_FORMAT` | `json` | Use `json` for structured logs or `text` for local readable logs. |
| `HQCA_RDKIT_LOGS` | `disabled` | Keep RDKit parser output quiet for structured logging; set another value to allow RDKit logs. |
| `HQCA_RANDOM_SEED` | `2025` | Random seed for reproducible synthetic data generation. |
| `HQCA_NUM_PAIRS` | `500` | Number of drug-protein pairs to generate. |
| `HQCA_OUTPUT_CSV` | `hqca_synthetic_data_500.csv` | CSV output path. |
| `HQCA_OUTPUT_JSON` | `hqca_synthetic_data_500.json` | JSON output path. |
| `HQCA_METRICS_FILE` | `hqca_metrics.json` | Runtime metrics JSON output path. |
| `HQCA_REPORT_FILE` | `data_report.txt` | Text report output path. |

## Run locally

```bash
HQCA_NUM_PAIRS=10 \
HQCA_OUTPUT_CSV=output/sample.csv \
HQCA_OUTPUT_JSON=output/sample.json \
HQCA_METRICS_FILE=output/metrics.json \
HQCA_REPORT_FILE=output/report.txt \
python data.py
```

## Run tests

```bash
python -m pytest
```

The CI workflow runs the healthcheck and the pytest suite on Python 3.11 and
3.12.

## Docker

Build:

```bash
docker build -t hqca-data-pipeline .
```

Build with PennyLane support:

```bash
docker build --build-arg INSTALL_QUANTUM=true -t hqca-data-pipeline:quantum .
```

Run a small dataset generation job:

```bash
docker run --rm \
  -e HQCA_NUM_PAIRS=10 \
  -v "$PWD/output:/app/output" \
  hqca-data-pipeline
```

Run the container healthcheck manually:

```bash
docker run --rm hqca-data-pipeline python scripts/healthcheck.py
```

## Structured logging

By default, runtime logs are emitted as JSON to stdout. Each log record includes
standard fields such as `timestamp`, `level`, `logger`, and `message`, plus
event-specific fields such as `event`, `num_pairs`, `records_generated`, and
`duration_seconds`.

For local text logs:

```bash
HQCA_LOG_FORMAT=text python data.py
```

## Monitoring baseline

The current product is a batch pipeline, so monitoring is file and process
oriented:

1. `scripts/healthcheck.py` validates the runtime and RDKit descriptor path.
2. `HQCA_METRICS_FILE` writes a JSON metrics file per run.
3. Docker `HEALTHCHECK` executes the healthcheck script.
4. CI fails if the healthcheck or tests fail.

Example metrics output:

```json
{
  "duration_seconds": 1.234,
  "num_pairs_requested": 10,
  "records_generated": 10,
  "records_failed": 0,
  "rdkit_available": true,
  "pennylane_available": false,
  "use_quantum": false
}
```

When an API server is added, this baseline should be extended with `/healthz`
and `/metrics` endpoints for uptime checks and Prometheus scraping.
