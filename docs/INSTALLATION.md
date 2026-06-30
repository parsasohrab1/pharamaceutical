# HQCA Installation

## Requirements
- Python 3.10+
- Optional: Docker, PostgreSQL, MinIO

## Quick start (auto-connect dashboard)

```powershell
pip install -r requirements.txt
python run_dashboard.py
```

Opens **http://127.0.0.1:18080/** — dashboard and API on the same server (connection automatic).

| Service | URL |
|---------|-----|
| **Dashboard** | http://127.0.0.1:18080/ |
| **API Swagger** | http://127.0.0.1:18080/docs |
| **Health** | http://127.0.0.1:18080/health |

Default admin: `admin` / `admin12345`

## Manual start

```powershell
$env:HQCA_USE_MINIO = "false"
$env:HQCA_DATABASE_URL = "sqlite:///output/hqca.db"
python run_api.py
```

Legacy separate frontend (port 5173) still works with API retry logic.

## Docker

```powershell
docker compose up --build
```

## Environment

Copy `.env.example` to `.env` — set `HQCA_PORT`, secrets, database URL.

## Tests

```powershell
python -m pytest tests/ -q
python evaluation.py
```
