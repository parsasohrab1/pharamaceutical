# Monitoring

HQCA currently includes a FastAPI MVP backend plus the batch data-generation
pipeline. Monitoring is implemented as a lightweight baseline:

- Container healthcheck: `python scripts/healthcheck.py`
- API liveness endpoint: `GET /healthz`
- Structured JSON logs on stdout
- Per-run metrics JSON via `HQCA_METRICS_FILE`
- CI validation of healthcheck and tests

Recommended production extensions for the future API service:

- `/readyz` for readiness checks
- `/metrics` for Prometheus counters and histograms
- Alerts for high `records_failed`, long `duration_seconds`, and failed jobs
