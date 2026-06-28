# HQCA API Reference

Base URL: `http://localhost:8000`

Swagger UI: `/docs`

## Authentication
- `POST /auth/register` — create user (researcher role)
- `POST /auth/login` — JWT bearer token

## Prediction (FR-01, FR-02, FR-17, FR-18, FR-19)
- `POST /predict`
  - Body: `{ "smiles": "CCO", "fasta": ">t\nACDEF...", "backend": "auto" }`
  - Response: binding score 0–100, confidence, PDB/CSV/PDF/3D viewer URLs

## Synthetic data (FR-04, FR-05, FR-06)
- `POST /generate_synthetic` — `{ "num_samples": 500, "smiles_seed": ["CCO"] }`
- `GET /status/{task_id}` — poll async job

## Admin / RBAC (NFR-06)
- `GET /predictions/history` — researcher sees own results; admin sees all
- `GET /admin/logs` — admin only

## Backends (FR-13)
- `GET /health` — lists available quantum backends

## Files
- `GET /files/{path}` — download stored artifacts (MinIO or local)
