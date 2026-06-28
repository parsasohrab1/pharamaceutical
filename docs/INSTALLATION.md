# HQCA Installation

## Requirements
- Python 3.10+
- Node.js optional (static frontend)

## Local development

```bash
pip install -r requirements.txt
uvicorn api:app --reload --host 127.0.0.1 --port 18080
```

Open API docs: http://localhost:8000/docs

Serve frontend (separate terminal):

```bash
cd frontend
python -m http.server 5173
```

Default admin: `admin` / `admin12345`

## Docker

```bash
docker compose up --build
```

- API: http://localhost:8000
- Frontend: http://localhost:5173
- MinIO console: http://localhost:9001

## Environment

Copy `.env.example` to `.env` and set:
- `HQCA_SECRET_KEY`
- `HQCA_ENCRYPTION_KEY`
- `HQCA_DATABASE_URL` (PostgreSQL in production)
- MinIO settings when `HQCA_USE_MINIO=true`

## Acceptance tests

```bash
python evaluation.py
pytest tests/
```
