"""Queue adapter for synthetic generation jobs."""

from __future__ import annotations

import os
from typing import Any

from fastapi import BackgroundTasks

from data import LOGGER


QUEUE_BACKEND = os.getenv("HQCA_QUEUE_BACKEND", "background").lower()


def enqueue_synthetic_job(background_tasks: BackgroundTasks, task_id: str, payload: dict[str, Any]) -> str:
    """Queue a synthetic generation job.

    In production, set `HQCA_QUEUE_BACKEND=rq` and `HQCA_REDIS_URL` to enqueue
    into Redis/RQ. In local development and CI, the job runs through FastAPI
    BackgroundTasks.
    """
    if QUEUE_BACKEND == "rq":
        try:
            from redis import Redis
            from rq import Queue
        except ImportError as exc:
            raise RuntimeError("RQ backend requires redis and rq packages.") from exc

        redis_url = os.getenv("HQCA_REDIS_URL", "redis://localhost:6379/0")
        queue_name = os.getenv("HQCA_RQ_QUEUE", "hqca")
        queue = Queue(queue_name, connection=Redis.from_url(redis_url))
        queue.enqueue("api.run_synthetic_generation_payload", task_id, payload)
        LOGGER.info("Synthetic generation enqueued in RQ.", extra={"event": "rq_job_enqueued"})
        return "rq"

    from api import run_synthetic_generation_payload

    background_tasks.add_task(run_synthetic_generation_payload, task_id, payload)
    return "background"
