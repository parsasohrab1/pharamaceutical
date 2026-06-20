# Enterprise deployment baseline

HQCA includes an enterprise-ready baseline that can run locally with fallback
storage or connect to PostgreSQL, MinIO, and Redis/RQ.

## Services

- **PostgreSQL** stores users, projects, prediction results, and processing task
  metadata.
- **MinIO** stores generated reports, PDB files, and synthetic dataset artifacts.
- **Redis/RQ** queues long-running synthetic data generation jobs.
- **JWT auth + RBAC** protects user/project/admin APIs.
- **Fernet encryption** protects sensitive SMILES/FASTA values before they are
  stored in the database.
- **JSON logs** are emitted by default through `HQCA_LOG_FORMAT=json`.

## Run the enterprise stack

```bash
docker compose up --build
```

Services:

- API: <http://localhost:8000>
- Swagger: <http://localhost:8000/docs>
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`
- MinIO API: <http://localhost:9000>
- MinIO console: <http://localhost:9001>

## Key environment variables

| Variable | Purpose |
| --- | --- |
| `HQCA_DATABASE_URL` | SQLAlchemy URL, e.g. `postgresql+psycopg://hqca:hqca@postgres:5432/hqca`. |
| `HQCA_SECRET_KEY` | JWT signing secret and fallback encryption secret. Change in production. |
| `HQCA_ENCRYPTION_KEY` | Optional Fernet key for sensitive fields. |
| `HQCA_MINIO_ENDPOINT` | MinIO/S3-compatible endpoint. Leave unset for local filesystem storage. |
| `HQCA_MINIO_ACCESS_KEY` | MinIO access key. |
| `HQCA_MINIO_SECRET_KEY` | MinIO secret key. |
| `HQCA_MINIO_BUCKET` | Bucket name for reports and PDB files. |
| `HQCA_QUEUE_BACKEND` | `background` for local/CI or `rq` for Redis/RQ. |
| `HQCA_REDIS_URL` | Redis URL for RQ. |

## Auth and RBAC

Register and log in:

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"strong-password"}'

TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"strong-password"}' | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
```

Use the token:

```bash
curl http://localhost:8000/users/me -H "Authorization: Bearer $TOKEN"
```

Roles:

- `researcher`: own projects and own prediction history.
- `admin`: user listing and cross-project visibility.

The first registered user may request `role=admin`; later public registrations
are forced to `researcher`.

## Projects and encrypted results

```bash
PROJECT_ID=$(curl -s -X POST http://localhost:8000/projects \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"Kinase program"}' | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')

curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"smiles\":\"CCO\",\"fasta\":\">target\nACDEFGHIKLMNPQRSTVWY\",\"project_id\":$PROJECT_ID}"
```

The API stores SMILES and FASTA encrypted in PostgreSQL and stores generated
CSV/PDF/PDB artifacts in MinIO or the local object store fallback.

## Queue workers

With `HQCA_QUEUE_BACKEND=rq`, `/generate_synthetic` enqueues jobs into Redis:

```bash
rq worker hqca --url redis://localhost:6379/0
```

`docker-compose.yml` starts a worker service automatically.

## Integration tests

`tests/test_fr_integration.py` maps FR-01 through FR-19 to an executable
integration contract covering validation, synthetic generation, descriptors,
backend selection contract, MAE training smoke, score/confidence/pocket output,
and CSV/PDF/PDB exports.
