# HQCA Installation

## Requirements
- Python 3.10+
- Optional: Docker, PostgreSQL, MinIO

## Local development

```powershell
pip install -r requirements.txt
$env:HQCA_USE_MINIO = "false"
$env:HQCA_DATABASE_URL = "sqlite:///output/hqca.db"
$env:HQCA_PORT = "18080"
python run_api.py
```

| Service | URL |
|---------|-----|
| Dashboard | http://127.0.0.1:5173 |
| API Swagger | http://127.0.0.1:18080/docs |
| Health | http://127.0.0.1:18080/health |

Frontend (second terminal):

```powershell
cd frontend
python -m http.server 5173 --bind 127.0.0.1
```

Default admin: `admin` / `admin12345`

## Docker

```powershell
docker compose up --build
```

- API: http://localhost:8000 (inside compose)
- Frontend: http://localhost:5173
- MinIO console: http://localhost:9001

## Environment

Copy `.env.example` to `.env`:

- `HQCA_PORT` — API port (default `18080` on Windows)
- `HQCA_SECRET_KEY` / `HQCA_ENCRYPTION_KEY`
- `HQCA_DATABASE_URL` — PostgreSQL in production
- `HQCA_USE_MINIO=true` + MinIO vars for object storage

## Acceptance tests

```powershell
python -m pytest tests/ -q
python evaluation.py
```
