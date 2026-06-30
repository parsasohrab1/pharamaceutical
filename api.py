"""FastAPI backend for HQCA MVP (NFR-03, NFR-07)."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from data import QuantumVQESimulator, predict_binding
from database import PredictionResult, ProcessingTask, Project, User, get_db, init_db, utc_now
from logging_config import LOGGER
from queueing import enqueue_synthetic_job
from reporting import generate_pdf_report
from security import (
    create_access_token,
    decrypt_sensitive,
    encrypt_sensitive,
    get_current_user_payload,
    hash_password,
    require_role,
    verify_password,
)
from seed_data import seed_demo_data
from storage import object_storage
from validation import normalize_fasta, validate_smiles

DEFAULT_OUTPUT_DIR = Path(os.getenv("HQCA_API_OUTPUT_DIR", "output/api"))
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="HQCA API",
    description="Hybrid Quantum-Classical Assistant for Drug Discovery",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("HQCA_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    LOGGER.info(
        "HTTP request",
        extra={"hqca_event": "http_request", "hqca_method": request.method, "hqca_path": request.url.path},
    )
    response = await call_next(request)
    LOGGER.info(
        "HTTP response",
        extra={"hqca_event": "http_response", "hqca_status": response.status_code, "hqca_path": request.url.path},
    )
    return response


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    from database import SessionLocal
    from seed_data import seed_demo_data

    db = SessionLocal()
    try:
        if not db.query(User).filter(User.username == "admin").first():
            admin = User(
                username="admin",
                password_hash=hash_password(os.getenv("HQCA_ADMIN_PASSWORD", "admin12345")),
                role="admin",
            )
            db.add(admin)
            db.commit()
            LOGGER.info("Default admin user created", extra={"hqca_event": "admin_seed"})
        if os.getenv("HQCA_SEED_DEMO", "true").lower() == "true":
            seeded = seed_demo_data(db)
            if seeded["predictions"] or seeded["datasets"]:
                LOGGER.info("Demo data ready", extra={"hqca_event": "demo_seeded", **{f"hqca_{k}": v for k, v in seeded.items()}})
    finally:
        db.close()


# --- Schemas ---
class PredictRequest(BaseModel):
    smiles: str = Field(..., max_length=200)
    fasta: str = Field(..., min_length=1, max_length=5000)
    project_id: Optional[int] = None
    backend: Literal["auto", "pennylane_default_qubit", "qiskit_aer_simulator", "classical_fallback"] = "auto"

    @field_validator("smiles")
    @classmethod
    def valid_smiles(cls, v: str) -> str:
        return validate_smiles(v)

    @field_validator("fasta")
    @classmethod
    def valid_fasta(cls, v: str) -> str:
        return normalize_fasta(v)


class PredictResponse(BaseModel):
    request_id: str
    created_at: str
    binding_score: float
    binding_energy_kcal_mol: float
    confidence: float
    backend: str
    gate_depth: int
    pocket_center: dict
    report_csv_url: str
    report_pdf_url: str
    pocket_pdb_url: str
    viewer_html_url: str


class GenerateSyntheticRequest(BaseModel):
    num_samples: int = Field(500, ge=1, le=5000)
    smiles_seed: List[str] = Field(..., min_length=1)
    fasta: Optional[str] = None
    project_id: Optional[int] = None

    @field_validator("smiles_seed")
    @classmethod
    def valid_seeds(cls, values: List[str]) -> List[str]:
        return [validate_smiles(v) for v in values]

    @field_validator("fasta")
    @classmethod
    def valid_optional_fasta(cls, v: Optional[str]) -> Optional[str]:
        return normalize_fasta(v) if v else None


class GenerateSyntheticResponse(BaseModel):
    task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    created_at: str
    updated_at: str
    num_samples: int
    records_generated: int
    records_failed: int
    output_csv: Optional[str] = None
    output_json: Optional[str] = None
    output_pdf: Optional[str] = None
    error: Optional[str] = None


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=80)
    password: str = Field(..., min_length=8, max_length=128)
    role: Literal["researcher", "admin"] = "researcher"


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class ProjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=160)


# --- Routes ---
@app.get("/health")
def health():
    return {"status": "ok", "backends": QuantumVQESimulator.available_backends()}


@app.post("/auth/register", response_model=TokenResponse)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=409, detail="Username already exists")
    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        role=body.role if body.role == "researcher" else "researcher",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.username, user.role, user.id)
    return TokenResponse(access_token=token, role=user.role)


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == body.username).first()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(user.username, user.role, user.id), role=user.role)


@app.post("/projects")
def create_project(
    body: ProjectCreateRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("researcher", "admin")),
):
    project = Project(owner_id=user["uid"], name=body.name)
    db.add(project)
    db.commit()
    db.refresh(project)
    return {"id": project.id, "name": project.name, "created_at": project.created_at}


@app.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    db: Session = Depends(get_db),
    user_payload: Optional[dict] = Depends(get_current_user_payload),
):
    request_id = uuid.uuid4().hex
    out_dir = DEFAULT_OUTPUT_DIR / request_id
    result = predict_binding(
        body.smiles,
        fasta=body.fasta,
        output_dir=str(out_dir),
        backend=body.backend,
    )

    csv_path = out_dir / "prediction.csv"
    pd.DataFrame([result.to_dict()]).to_csv(csv_path, index=False)
    pdf_path = out_dir / "report.pdf"
    generate_pdf_report(result.to_dict(), str(pdf_path))

    csv_url = object_storage.put_file(str(csv_path), f"predictions/{request_id}/prediction.csv")
    pdf_url = object_storage.put_file(str(pdf_path), f"predictions/{request_id}/report.pdf")
    pdb_url = object_storage.put_file(result.pdb_path, f"predictions/{request_id}/pocket.pdb")
    viewer_url = object_storage.put_file(result.viewer_html_path, f"predictions/{request_id}/viewer.html")

    center = result.pocket.get("center", (0, 0, 0))
    record = PredictionResult(
        request_id=request_id,
        user_id=user_payload["uid"] if user_payload else None,
        project_id=body.project_id,
        encrypted_smiles=encrypt_sensitive(body.smiles),
        encrypted_fasta=encrypt_sensitive(body.fasta),
        binding_score=result.binding_score,
        binding_energy_kcal_mol=result.binding_energy_kcal_mol,
        confidence=result.confidence_pct,
        pocket_center_json=json.dumps({"x": center[0], "y": center[1], "z": center[2]}),
        report_csv_object=csv_url,
        report_pdf_object=pdf_url,
        pocket_pdb_object=pdb_url,
        viewer_html_object=viewer_url,
        backend=result.backend,
    )
    db.add(record)
    db.commit()

    return PredictResponse(
        request_id=request_id,
        created_at=utc_now(),
        binding_score=result.binding_score,
        binding_energy_kcal_mol=result.binding_energy_kcal_mol,
        confidence=result.confidence_pct,
        backend=result.backend,
        gate_depth=result.gate_depth,
        pocket_center={"x": center[0], "y": center[1], "z": center[2]},
        report_csv_url=csv_url,
        report_pdf_url=pdf_url,
        pocket_pdb_url=pdb_url,
        viewer_html_url=viewer_url,
    )


@app.post("/generate_synthetic", response_model=GenerateSyntheticResponse)
def generate_synthetic(
    body: GenerateSyntheticRequest,
    db: Session = Depends(get_db),
    user_payload: Optional[dict] = Depends(get_current_user_payload),
):
    task_id = enqueue_synthetic_job(
        db,
        num_samples=body.num_samples,
        smiles_seed=body.smiles_seed,
        user_id=user_payload["uid"] if user_payload else None,
        project_id=body.project_id,
        fasta=body.fasta,
    )
    return GenerateSyntheticResponse(task_id=task_id, status="pending")


@app.get("/status/{task_id}", response_model=TaskStatusResponse)
def task_status(task_id: str, db: Session = Depends(get_db)):
    task = db.get(ProcessingTask, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        created_at=task.created_at,
        updated_at=task.updated_at,
        num_samples=task.num_samples,
        records_generated=task.records_generated,
        records_failed=task.records_failed,
        output_csv=task.output_csv_object,
        output_json=task.output_json_object,
        output_pdf=task.output_pdf_object,
        error=task.error,
    )


@app.get("/predictions/history")
def prediction_history(
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("researcher", "admin")),
):
    query = db.query(PredictionResult)
    if user["role"] != "admin":
        query = query.filter(PredictionResult.user_id == user["uid"])
    rows = query.order_by(PredictionResult.created_at.desc()).limit(50).all()
    return [
        {
            "request_id": r.request_id,
            "created_at": r.created_at,
            "binding_score": r.binding_score,
            "binding_energy_kcal_mol": r.binding_energy_kcal_mol,
            "confidence": r.confidence,
            "backend": r.backend,
            "smiles_preview": decrypt_sensitive(r.encrypted_smiles)[:24],
            "viewer_html_url": r.viewer_html_object,
            "report_pdf_url": r.report_pdf_object,
            "report_csv_url": r.report_csv_object,
            "pocket_pdb_url": r.pocket_pdb_object,
        }
        for r in rows
    ]


@app.get("/predictions/{request_id}")
def prediction_detail(
    request_id: str,
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("researcher", "admin")),
):
    row = db.get(PredictionResult, request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if user["role"] != "admin" and row.user_id != user["uid"]:
        raise HTTPException(status_code=403, detail="Access denied")
    center = json.loads(row.pocket_center_json)
    return {
        "request_id": row.request_id,
        "created_at": row.created_at,
        "smiles": decrypt_sensitive(row.encrypted_smiles),
        "fasta": decrypt_sensitive(row.encrypted_fasta),
        "binding_score": row.binding_score,
        "binding_energy_kcal_mol": row.binding_energy_kcal_mol,
        "confidence": row.confidence,
        "backend": row.backend,
        "pocket_center": center,
        "viewer_html_url": row.viewer_html_object,
        "report_pdf_url": row.report_pdf_object,
        "report_csv_url": row.report_csv_object,
        "pocket_pdb_url": row.pocket_pdb_object,
    }


@app.get("/dashboard")
def dashboard_summary(
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("researcher", "admin")),
):
    pred_query = db.query(PredictionResult)
    task_query = db.query(ProcessingTask).filter(ProcessingTask.status == "completed")
    if user["role"] != "admin":
        pred_query = pred_query.filter(PredictionResult.user_id == user["uid"])
        task_query = task_query.filter(ProcessingTask.user_id == user["uid"])

    predictions = pred_query.order_by(PredictionResult.created_at.desc()).limit(10).all()
    tasks = task_query.order_by(ProcessingTask.updated_at.desc()).limit(5).all()
    latest = predictions[0] if predictions else None

    return {
        "stats": {
            "total_predictions": pred_query.count(),
            "total_synthetic_jobs": task_query.count(),
            "avg_binding_score": round(
                sum(p.binding_score for p in predictions) / max(len(predictions), 1), 2
            ),
        },
        "latest_prediction": None
        if latest is None
        else {
            "request_id": latest.request_id,
            "created_at": latest.created_at,
            "binding_score": latest.binding_score,
            "binding_energy_kcal_mol": latest.binding_energy_kcal_mol,
            "confidence": latest.confidence,
            "backend": latest.backend,
            "smiles_preview": decrypt_sensitive(latest.encrypted_smiles)[:40],
            "viewer_html_url": latest.viewer_html_object,
            "report_pdf_url": latest.report_pdf_object,
            "report_csv_url": latest.report_csv_object,
            "pocket_pdb_url": latest.pocket_pdb_object,
        },
        "predictions": [
            {
                "request_id": p.request_id,
                "created_at": p.created_at,
                "binding_score": p.binding_score,
                "confidence": p.confidence,
                "smiles_preview": decrypt_sensitive(p.encrypted_smiles)[:24],
                "viewer_html_url": p.viewer_html_object,
            }
            for p in predictions
        ],
        "synthetic_datasets": [
            {
                "task_id": t.task_id,
                "num_samples": t.num_samples,
                "records_generated": t.records_generated,
                "output_csv": t.output_csv_object,
                "output_json": t.output_json_object,
                "output_pdf": t.output_pdf_object,
                "updated_at": t.updated_at,
            }
            for t in tasks
        ],
    }


@app.get("/dashboard/datasets")
def dashboard_datasets(
    db: Session = Depends(get_db),
    user: dict = Depends(require_role("researcher", "admin")),
):
    query = db.query(ProcessingTask).filter(ProcessingTask.status == "completed")
    if user["role"] != "admin":
        query = query.filter(ProcessingTask.user_id == user["uid"])
    tasks = query.order_by(ProcessingTask.updated_at.desc()).all()
    return [
        {
            "task_id": t.task_id,
            "num_samples": t.num_samples,
            "records_generated": t.records_generated,
            "output_csv": t.output_csv_object,
            "output_json": t.output_json_object,
            "output_pdf": t.output_pdf_object,
        }
        for t in tasks
    ]


@app.get("/admin/logs")
def admin_logs(user: dict = Depends(require_role("admin"))):
    return {"message": "Structured JSON logs are emitted to stdout (NFR-08)."}


@app.get("/files/{file_path:path}")
def get_file(file_path: str):
    local = object_storage.resolve_local(file_path)
    if not local.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(local)
