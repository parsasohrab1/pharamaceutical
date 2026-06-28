"""Background synthetic dataset jobs."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from data import SyntheticDataPipeline
from database import ProcessingTask, SessionLocal, utc_now
from logging_config import LOGGER
from storage import object_storage


def _update_task(db: Session, task_id: str, **fields) -> None:
    task = db.get(ProcessingTask, task_id)
    if task is None:
        return
    for key, value in fields.items():
        setattr(task, key, value)
    task.updated_at = utc_now()
    db.commit()


def run_synthetic_job(
    task_id: str,
    num_samples: int,
    smiles_seed: List[str],
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
    fasta: Optional[str] = None,
) -> None:
    db = SessionLocal()
    output_dir = Path("output/tasks") / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dataset.csv"
    json_path = output_dir / "dataset.json"
    pdf_path = output_dir / "report.pdf"

    try:
        _update_task(db, task_id, status="running")
        pipeline = SyntheticDataPipeline(smiles_seed, random_state=42)
        df = pipeline.generate_dataset(
            num_pairs=num_samples,
            output_csv=str(csv_path),
            output_json=str(json_path),
            output_pdf=str(pdf_path),
            fasta=fasta,
        )
        csv_obj = object_storage.put_file(str(csv_path), f"tasks/{task_id}/dataset.csv")
        json_obj = object_storage.put_file(str(json_path), f"tasks/{task_id}/dataset.json")
        pdf_obj = object_storage.put_file(str(pdf_path), f"tasks/{task_id}/report.pdf")
        _update_task(
            db,
            task_id,
            status="completed",
            records_generated=len(df),
            records_failed=max(0, num_samples - len(df)),
            output_csv_object=csv_obj,
            output_json_object=json_obj,
            output_pdf_object=pdf_obj,
        )
        LOGGER.info(
            "Synthetic job completed",
            extra={"hqca_event": "synthetic_completed", "hqca_task_id": task_id},
        )
    except Exception as exc:
        LOGGER.exception(
            "Synthetic job failed",
            extra={"hqca_event": "synthetic_failed", "hqca_task_id": task_id},
        )
        _update_task(db, task_id, status="failed", error=str(exc))
    finally:
        db.close()


def enqueue_synthetic_job(
    db: Session,
    num_samples: int,
    smiles_seed: List[str],
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
    fasta: Optional[str] = None,
) -> str:
    task_id = uuid.uuid4().hex
    task = ProcessingTask(
        task_id=task_id,
        user_id=user_id,
        project_id=project_id,
        status="pending",
        num_samples=num_samples,
    )
    db.add(task)
    db.commit()
    thread = threading.Thread(
        target=run_synthetic_job,
        args=(task_id, num_samples, smiles_seed, user_id, project_id, fasta),
        daemon=True,
    )
    thread.start()
    return task_id
