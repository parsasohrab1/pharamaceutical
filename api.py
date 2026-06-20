"""FastAPI application for the HQCA MVP backend."""

from __future__ import annotations

import os
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from rdkit import Chem
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy.orm import Session

from data import (
    LOGGER,
    MolecularDescriptors,
    QuantumVQESimulator,
    SyntheticDataPipeline,
    SyntheticPocketGenerator,
    normalize_descriptors,
)
from database import (
    PredictionResult,
    ProcessingTask,
    Project,
    SessionLocal,
    User,
    get_db,
    init_db,
)
from queueing import enqueue_synthetic_job
from scoring import binding_energy_to_score
from security import (
    create_access_token,
    decode_access_token,
    encrypt_sensitive,
    hash_password,
    require_role,
    verify_password,
)
from storage import object_storage
from training import load_model_artifact, predict_energy_from_artifact


VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_OUTPUT_DIR = Path(os.getenv("HQCA_API_OUTPUT_DIR", "output/api"))
REPORTS_DIR = DEFAULT_OUTPUT_DIR / "reports"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_fasta(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    sequence = "".join(line for line in lines if not line.startswith(">")).upper()
    if not sequence:
        raise ValueError("FASTA sequence is required.")
    invalid = sorted(set(sequence) - VALID_AMINO_ACIDS)
    if invalid:
        raise ValueError(f"FASTA contains invalid amino acid codes: {', '.join(invalid)}")
    return sequence


def validate_smiles(value: str) -> str:
    smiles = value.strip()
    if len(smiles) > 200:
        raise ValueError("SMILES must be 200 characters or fewer.")
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError("SMILES is not valid.")
    return smiles


def estimate_confidence(features_normalized: np.ndarray, fasta_length: int) -> float:
    unclipped_ratio = float(np.mean((features_normalized > 0.0) & (features_normalized < 1.0)))
    length_factor = min(fasta_length, 60) / 60.0
    confidence = 55.0 + 30.0 * unclipped_ratio + 10.0 * length_factor
    return round(float(np.clip(confidence, 50.0, 95.0)), 2)


class PredictRequest(BaseModel):
    smiles: str = Field(..., max_length=200, examples=["CCO"])
    fasta: str = Field(..., min_length=1, max_length=5000, examples=[">target\nACDEFGHIKLMNPQRSTVWY"])
    project_id: Optional[int] = None
    backend: Literal["auto", "pennylane_default_qubit", "classical_fallback"] = "auto"

    @field_validator("smiles")
    @classmethod
    def smiles_must_be_valid(cls, value: str) -> str:
        return validate_smiles(value)

    @field_validator("fasta")
    @classmethod
    def fasta_must_be_valid(cls, value: str) -> str:
        return normalize_fasta(value)


class PocketCenter(BaseModel):
    x: float
    y: float
    z: float


class PredictResponse(BaseModel):
    request_id: str
    created_at: str
    binding_score: float
    binding_energy_kcal_mol: float
    confidence: float
    pocket_center: PocketCenter
    report_csv_url: str
    report_pdf_url: str
    pocket_pdb_url: str


class GenerateSyntheticRequest(BaseModel):
    num_samples: int = Field(500, ge=1, le=5000)
    smiles_seed: List[str] = Field(..., min_length=1)
    project_id: Optional[int] = None

    @field_validator("smiles_seed")
    @classmethod
    def seeds_must_be_valid(cls, values: List[str]) -> List[str]:
        return [validate_smiles(value) for value in values]


class GenerateSyntheticResponse(BaseModel):
    task_id: str
    status: Literal["pending", "running", "completed", "failed"]


class TaskStatus(BaseModel):
    task_id: str
    status: Literal["pending", "running", "completed", "failed"]
    created_at: str
    updated_at: str
    num_samples: int
    records_generated: int = 0
    records_failed: int = 0
    output_csv: Optional[str] = None
    output_json: Optional[str] = None
    output_metrics: Optional[str] = None
    error: Optional[str] = None


class PredictionHistoryItem(BaseModel):
    request_id: str
    created_at: str
    smiles: str
    fasta_preview: str
    binding_score: float
    binding_energy_kcal_mol: float
    confidence: float
    report_csv_url: str
    report_pdf_url: str
    pocket_pdb_url: str


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


class UserResponse(BaseModel):
    id: int
    username: str
    role: str


class ProjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=160)


class ProjectResponse(BaseModel):
    id: int
    name: str
    owner_id: int
    created_at: str


class PredictionService:
    def __init__(self) -> None:
        self.pocket_generator = SyntheticPocketGenerator(random_seed=42)
        self.quantum_simulator = QuantumVQESimulator(n_qubits=7)
        self.model_artifact_path = os.getenv("HQCA_MODEL_ARTIFACT")
        self.model_artifact = None
        if self.model_artifact_path and Path(self.model_artifact_path).exists():
            self.model_artifact = load_model_artifact(self.model_artifact_path)
            LOGGER.info(
                "Prediction model artifact loaded.",
                extra={"event": "model_artifact_loaded", "output_metrics": self.model_artifact_path},
            )

    def predict(
        self,
        request: PredictRequest,
        db: Optional[Session] = None,
        user: Optional[User] = None,
    ) -> PredictResponse:
        request_id = uuid.uuid4().hex
        created_at = utc_now()
        descriptors = MolecularDescriptors.compute(request.smiles)
        features = normalize_descriptors(descriptors.to_array())
        pocket = self.pocket_generator.generate_pocket(
            sequence=request.fasta,
            length=min(max(len(request.fasta), 1), 60),
        )
        if self.model_artifact is not None:
            energy = predict_energy_from_artifact(self.model_artifact, features)
        else:
            energy = float(self.quantum_simulator.predict_affinity(features, optimize=False))
        center = pocket["center"]
        response = PredictResponse(
            request_id=request_id,
            created_at=created_at,
            binding_score=binding_energy_to_score(energy),
            binding_energy_kcal_mol=round(energy, 4),
            confidence=estimate_confidence(features, len(request.fasta)),
            pocket_center=PocketCenter(
                x=round(float(center[0]), 4),
                y=round(float(center[1]), 4),
                z=round(float(center[2]), 4),
            ),
            report_csv_url=f"/reports/{request_id}/csv",
            report_pdf_url=f"/reports/{request_id}/pdf",
            pocket_pdb_url=f"/reports/{request_id}/pdb",
        )
        object_keys = write_prediction_artifacts(request, response)
        prediction_history[request_id] = PredictionHistoryItem(
            request_id=request_id,
            created_at=created_at,
            smiles=request.smiles,
            fasta_preview=request.fasta[:80],
            binding_score=response.binding_score,
            binding_energy_kcal_mol=response.binding_energy_kcal_mol,
            confidence=response.confidence,
            report_csv_url=response.report_csv_url,
            report_pdf_url=response.report_pdf_url,
            pocket_pdb_url=response.pocket_pdb_url,
        )
        if db is not None:
            project_id = request.project_id
            if project_id is not None:
                project = db.get(Project, project_id)
                if project is None:
                    raise HTTPException(status_code=404, detail="Project not found.")
                if user is not None and user.role != "admin" and project.owner_id != user.id:
                    raise HTTPException(status_code=403, detail="Project access denied.")
            db.add(
                PredictionResult(
                    request_id=request_id,
                    user_id=user.id if user else None,
                    project_id=project_id,
                    created_at=created_at,
                    encrypted_smiles=encrypt_sensitive(request.smiles),
                    encrypted_fasta=encrypt_sensitive(request.fasta),
                    binding_score=response.binding_score,
                    binding_energy_kcal_mol=response.binding_energy_kcal_mol,
                    confidence=response.confidence,
                    pocket_center_json=json.dumps(response.pocket_center.model_dump()),
                    report_csv_object=object_keys["csv"],
                    report_pdf_object=object_keys["pdf"],
                    pocket_pdb_object=object_keys["pdb"],
                )
            )
            db.commit()
        return response


prediction_service = PredictionService()
task_lock = threading.Lock()
tasks: Dict[str, TaskStatus] = {}
prediction_history: Dict[str, PredictionHistoryItem] = {}
auth_scheme = HTTPBearer(auto_error=False)


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    if credentials is None:
        return None
    try:
        payload = decode_access_token(credentials.credentials)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid authentication token.") from exc
    user = db.query(User).filter(User.username == payload.get("sub")).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


def get_current_user(user: Optional[User] = Depends(get_optional_user)) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def get_admin_user(user: User = Depends(get_current_user)) -> User:
    try:
        require_role(user.role, {"admin"})
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return user


def report_paths(request_id: str) -> tuple[Path, Path, Path]:
    return (
        REPORTS_DIR / f"{request_id}.csv",
        REPORTS_DIR / f"{request_id}.pdf",
        REPORTS_DIR / f"{request_id}.pdb",
    )


def pdb_from_pocket_center(center: PocketCenter) -> str:
    return (
        "HEADER    HQCA PREDICTED POCKET CENTER\n"
        f"ATOM      1  CEN POC A   1    {center.x:8.3f}{center.y:8.3f}{center.z:8.3f}"
        "  1.00  0.00           C\n"
        "END\n"
    )


def write_prediction_artifacts(request: PredictRequest, response: PredictResponse) -> dict[str, str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path, pdf_path, pdb_path = report_paths(response.request_id)
    row = {
        "request_id": response.request_id,
        "created_at": response.created_at,
        "smiles": request.smiles,
        "fasta": request.fasta,
        "binding_score": response.binding_score,
        "binding_energy_kcal_mol": response.binding_energy_kcal_mol,
        "confidence": response.confidence,
        "pocket_center_x": response.pocket_center.x,
        "pocket_center_y": response.pocket_center.y,
        "pocket_center_z": response.pocket_center.z,
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    pdb_path.write_text(pdb_from_pocket_center(response.pocket_center), encoding="utf-8")

    pdf = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    y = height - 72
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(72, y, "HQCA Binding Prediction Report")
    y -= 34
    pdf.setFont("Helvetica", 10)
    for label, value in row.items():
        if label == "fasta":
            value = f"{str(value)[:120]}..."
        pdf.drawString(72, y, f"{label}: {value}")
        y -= 18
        if y < 72:
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y = height - 72
    pdf.save()

    keys = {
        "csv": f"reports/{response.request_id}.csv",
        "pdf": f"reports/{response.request_id}.pdf",
        "pdb": f"pdb/{response.request_id}.pdb",
    }
    object_storage.put_file(keys["csv"], csv_path, content_type="text/csv")
    object_storage.put_file(keys["pdf"], pdf_path, content_type="application/pdf")
    object_storage.put_file(keys["pdb"], pdb_path, content_type="chemical/x-pdb")
    return keys


def update_task(task_id: str, **updates: object) -> None:
    with task_lock:
        task = tasks[task_id]
        task_data = task.model_dump()
        task_data.update(updates)
        task_data["updated_at"] = utc_now()
        tasks[task_id] = TaskStatus(**task_data)


def run_synthetic_generation(task_id: str, request: GenerateSyntheticRequest) -> None:
    update_task(task_id, status="running")
    db = SessionLocal()
    db_task = db.get(ProcessingTask, task_id)
    if db_task is not None:
        db_task.status = "running"
        db_task.updated_at = utc_now()
        db.commit()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = DEFAULT_OUTPUT_DIR / f"{task_id}.csv"
    output_json = DEFAULT_OUTPUT_DIR / f"{task_id}.json"
    output_metrics = DEFAULT_OUTPUT_DIR / f"{task_id}_metrics.json"

    try:
        pipeline = SyntheticDataPipeline(request.smiles_seed, random_state=42)
        df = pipeline.generate_dataset(
            num_pairs=request.num_samples,
            output_csv=str(output_csv),
            output_json=str(output_json),
            output_metrics=str(output_metrics),
        )
        csv_key = f"synthetic/{task_id}.csv"
        json_key = f"synthetic/{task_id}.json"
        metrics_key = f"synthetic/{task_id}_metrics.json"
        object_storage.put_file(csv_key, output_csv, content_type="text/csv")
        object_storage.put_file(json_key, output_json, content_type="application/json")
        object_storage.put_file(metrics_key, output_metrics, content_type="application/json")
        update_task(
            task_id,
            status="completed",
            records_generated=int(len(df)),
            output_csv=csv_key,
            output_json=json_key,
            output_metrics=metrics_key,
        )
        if db_task is not None:
            db_task.status = "completed"
            db_task.records_generated = int(len(df))
            db_task.output_csv_object = csv_key
            db_task.output_json_object = json_key
            db_task.output_metrics_object = metrics_key
            db_task.updated_at = utc_now()
            db.commit()
    except Exception as exc:
        LOGGER.exception("Synthetic generation task failed.", extra={"event": "api_task_failed"})
        update_task(task_id, status="failed", error=str(exc))
        if db_task is not None:
            db_task.status = "failed"
            db_task.error = str(exc)
            db_task.updated_at = utc_now()
            db.commit()
    finally:
        db.close()


def run_synthetic_generation_payload(task_id: str, payload: dict) -> None:
    request = GenerateSyntheticRequest(**payload)
    run_synthetic_generation(task_id, request)


app = FastAPI(
    title="HQCA API",
    version="0.1.0",
    description="MVP API for HQCA drug-protein binding prediction and synthetic data generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("HQCA_CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/healthz")
def healthz() -> Dict[str, object]:
    return {
        "status": "ok",
        "timestamp": utc_now(),
        "queue_backend": os.getenv("HQCA_QUEUE_BACKEND", "background"),
        "object_storage_backend": object_storage.backend,
    }


@app.post("/auth/register", response_model=UserResponse, status_code=201)
def register(request: RegisterRequest, db: Session = Depends(get_db)) -> UserResponse:
    init_db()
    if db.query(User).filter(User.username == request.username).first() is not None:
        raise HTTPException(status_code=409, detail="Username already exists.")
    role = request.role if db.query(User).count() == 0 else "researcher"
    user = User(username=request.username, password_hash=hash_password(request.password), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserResponse(id=user.id, username=user.username, role=user.role)


@app.post("/auth/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(User).filter(User.username == request.username).first()
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return TokenResponse(access_token=create_access_token(user.username, user.role), role=user.role)


@app.get("/users/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse(id=user.id, username=user.username, role=user.role)


@app.get("/admin/users", response_model=List[UserResponse])
def list_users(_: User = Depends(get_admin_user), db: Session = Depends(get_db)) -> List[UserResponse]:
    return [UserResponse(id=user.id, username=user.username, role=user.role) for user in db.query(User).all()]


@app.post("/projects", response_model=ProjectResponse, status_code=201)
def create_project(
    request: ProjectCreateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ProjectResponse:
    project = Project(owner_id=user.id, name=request.name)
    db.add(project)
    db.commit()
    db.refresh(project)
    return ProjectResponse(id=project.id, name=project.name, owner_id=project.owner_id, created_at=project.created_at)


@app.get("/projects", response_model=List[ProjectResponse])
def list_projects(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[ProjectResponse]:
    query = db.query(Project)
    if user.role != "admin":
        query = query.filter(Project.owner_id == user.id)
    return [
        ProjectResponse(id=project.id, name=project.name, owner_id=project.owner_id, created_at=project.created_at)
        for project in query.all()
    ]


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db),
) -> PredictResponse:
    init_db()
    return prediction_service.predict(request, db=db, user=user)


@app.get("/history", response_model=List[PredictionHistoryItem])
def history(
    user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db),
) -> List[PredictionHistoryItem]:
    if user is not None:
        query = db.query(PredictionResult)
        if user.role != "admin":
            query = query.filter(PredictionResult.user_id == user.id)
        return [
            PredictionHistoryItem(
                request_id=result.request_id,
                created_at=result.created_at,
                smiles="[encrypted]",
                fasta_preview="[encrypted]",
                binding_score=result.binding_score,
                binding_energy_kcal_mol=result.binding_energy_kcal_mol,
                confidence=result.confidence,
                report_csv_url=f"/reports/{result.request_id}/csv",
                report_pdf_url=f"/reports/{result.request_id}/pdf",
                pocket_pdb_url=f"/reports/{result.request_id}/pdb",
            )
            for result in query.order_by(PredictionResult.created_at.desc()).limit(50).all()
        ]
    return sorted(prediction_history.values(), key=lambda item: item.created_at, reverse=True)


@app.get("/reports/{request_id}/csv")
def download_prediction_csv(request_id: str):
    return object_storage.response(
        f"reports/{request_id}.csv",
        media_type="text/csv",
        filename=f"hqca_prediction_{request_id}.csv",
    )


@app.get("/reports/{request_id}/pdf")
def download_prediction_pdf(request_id: str):
    return object_storage.response(
        f"reports/{request_id}.pdf",
        media_type="application/pdf",
        filename=f"hqca_prediction_{request_id}.pdf",
    )


@app.get("/reports/{request_id}/pdb")
def download_prediction_pdb(request_id: str):
    return object_storage.response(
        f"pdb/{request_id}.pdb",
        media_type="chemical/x-pdb",
        filename=f"hqca_pocket_{request_id}.pdb",
    )


@app.post("/generate_synthetic", response_model=GenerateSyntheticResponse, status_code=202)
def generate_synthetic(
    request: GenerateSyntheticRequest,
    background_tasks: BackgroundTasks,
    user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db),
) -> GenerateSyntheticResponse:
    init_db()
    task_id = uuid.uuid4().hex
    now = utc_now()
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        created_at=now,
        updated_at=now,
        num_samples=request.num_samples,
    )
    db.add(
        ProcessingTask(
            task_id=task_id,
            user_id=user.id if user else None,
            project_id=request.project_id,
            status="pending",
            created_at=now,
            updated_at=now,
            num_samples=request.num_samples,
        )
    )
    db.commit()
    enqueue_synthetic_job(background_tasks, task_id, request.model_dump())
    return GenerateSyntheticResponse(task_id=task_id, status="pending")


@app.get("/status/{task_id}", response_model=TaskStatus)
def status(task_id: str, db: Session = Depends(get_db)) -> TaskStatus:
    task = tasks.get(task_id)
    if task is not None:
        return task
    db_task = db.get(ProcessingTask, task_id)
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskStatus(
        task_id=db_task.task_id,
        status=db_task.status,
        created_at=db_task.created_at,
        updated_at=db_task.updated_at,
        num_samples=db_task.num_samples,
        records_generated=db_task.records_generated,
        records_failed=db_task.records_failed,
        output_csv=db_task.output_csv_object,
        output_json=db_task.output_json_object,
        output_metrics=db_task.output_metrics_object,
        error=db_task.error,
    )
