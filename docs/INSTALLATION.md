# HQCA Installation and Operations Guide

This repository ships an HQCA MVP with a FastAPI backend, a React UI, and the
existing synthetic data pipeline as an internal service.

## Requirements

- Python 3.10 or newer
- `pip`
- Node.js 22+ and npm for the React UI
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
| `HQCA_API_OUTPUT_DIR` | `output/api` | Directory for API synthetic generation job outputs. |
| `HQCA_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated origins allowed to call the API. |

## Run the FastAPI backend

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Open the interactive API docs:

- Swagger UI: <http://localhost:8000/docs>
- Healthcheck: <http://localhost:8000/healthz>

Core endpoints:

- `POST /predict`
- `POST /generate_synthetic`
- `GET /status/{task_id}`
- `GET /history`
- `GET /reports/{request_id}/csv`
- `GET /reports/{request_id}/pdf`

Example prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles":"CCO","fasta":">target\nACDEFGHIKLMNPQRSTVWY"}'
```

## Run the React UI

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

Open <http://localhost:5173>.

The UI includes:

- a prediction input form
- a results page with Binding Score chart
- a Three.js pocket-center viewer
- CSV/PDF report download links
- recent user request history stored in the browser and refreshed from the API

Build the UI:

```bash
cd frontend
npm run build
```

## Run the batch pipeline locally

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

The CI workflow runs the healthcheck and pytest suite on Python 3.11 and 3.12,
trains a smoke model on the fixture dataset, builds the React UI, and builds the
Docker image.

## Train on BindingDB/PDBbind data

See [`docs/DATASETS_AND_TRAINING.md`](DATASETS_AND_TRAINING.md) for supported
CSV columns, the nM/pKd to energy conversion, the documented binding score
formula, RandomForest baseline training, VQC parameter artifact saving, and MAE
benchmark outputs.

Example:

```bash
python training.py \
  --dataset data/raw/bindingdb_subset.csv \
  --source bindingdb \
  --artifact-dir models
```

Use the trained artifact in the API:

```bash
HQCA_MODEL_ARTIFACT=models/hqca_model.joblib uvicorn api:app --reload
```

## Docker

Build:

```bash
docker build -t hqca-data-pipeline .
```

Build with PennyLane support:

```bash
docker build --build-arg INSTALL_QUANTUM=true -t hqca-data-pipeline:quantum .
```

Run the API server:

```bash
docker run --rm \
  -p 8000:8000 \
  -v "$PWD/output:/app/output" \
  hqca-data-pipeline
```

Run the batch pipeline from the same image:

```bash
docker run --rm \
  -e HQCA_NUM_PAIRS=10 \
  -v "$PWD/output:/app/output" \
  hqca-data-pipeline python data.py
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

The current monitoring baseline covers both the API runtime and batch pipeline:

1. `scripts/healthcheck.py` validates the runtime and RDKit descriptor path.
2. `HQCA_METRICS_FILE` writes a JSON metrics file per run.
3. Docker `HEALTHCHECK` executes the healthcheck script.
4. CI fails if the healthcheck or tests fail.
5. The API exposes `/healthz` for liveness checks.

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
