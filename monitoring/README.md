# Monitoring

HQCA is currently a batch data-generation pipeline. Monitoring is implemented as
a lightweight baseline:

- Container healthcheck: `python scripts/healthcheck.py`
- Structured JSON logs on stdout
- Per-run metrics JSON via `HQCA_METRICS_FILE`
- CI validation of healthcheck and tests

Recommended production extensions for the future API service:

- `/healthz` for liveness and dependency checks
- `/readyz` for readiness checks
- `/metrics` for Prometheus counters and histograms
- Alerts for high `records_failed`, long `duration_seconds`, and failed jobs
