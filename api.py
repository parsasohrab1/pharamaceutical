"""FastAPI application for the HQCA MVP backend."""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from rdkit import Chem
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from data import (
    LOGGER,
    MolecularDescriptors,
    QuantumVQESimulator,
    SyntheticDataPipeline,
    SyntheticPocketGenerator,
    normalize_descriptors,
)
from scoring import binding_energy_to_score
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


class GenerateSyntheticRequest(BaseModel):
    num_samples: int = Field(500, ge=1, le=5000)
    smiles_seed: List[str] = Field(..., min_length=1)

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

    def predict(self, request: PredictRequest) -> PredictResponse:
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
        )
        write_prediction_reports(request, response)
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
        )
        return response


prediction_service = PredictionService()
task_lock = threading.Lock()
tasks: Dict[str, TaskStatus] = {}
prediction_history: Dict[str, PredictionHistoryItem] = {}


def report_paths(request_id: str) -> tuple[Path, Path]:
    return REPORTS_DIR / f"{request_id}.csv", REPORTS_DIR / f"{request_id}.pdf"


def write_prediction_reports(request: PredictRequest, response: PredictResponse) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path, pdf_path = report_paths(response.request_id)
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


def update_task(task_id: str, **updates: object) -> None:
    with task_lock:
        task = tasks[task_id]
        task_data = task.model_dump()
        task_data.update(updates)
        task_data["updated_at"] = utc_now()
        tasks[task_id] = TaskStatus(**task_data)


def run_synthetic_generation(task_id: str, request: GenerateSyntheticRequest) -> None:
    update_task(task_id, status="running")
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
        update_task(
            task_id,
            status="completed",
            records_generated=int(len(df)),
            output_csv=str(output_csv),
            output_json=str(output_json),
            output_metrics=str(output_metrics),
        )
    except Exception as exc:
        LOGGER.exception("Synthetic generation task failed.", extra={"event": "api_task_failed"})
        update_task(task_id, status="failed", error=str(exc))


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


@app.get("/healthz")
def healthz() -> Dict[str, object]:
    return {"status": "ok", "timestamp": utc_now()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    return prediction_service.predict(request)


@app.get("/history", response_model=List[PredictionHistoryItem])
def history() -> List[PredictionHistoryItem]:
    return sorted(prediction_history.values(), key=lambda item: item.created_at, reverse=True)


@app.get("/reports/{request_id}/csv")
def download_prediction_csv(request_id: str) -> FileResponse:
    csv_path, _ = report_paths(request_id)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV report not found.")
    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=f"hqca_prediction_{request_id}.csv",
    )


@app.get("/reports/{request_id}/pdf")
def download_prediction_pdf(request_id: str) -> FileResponse:
    _, pdf_path = report_paths(request_id)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF report not found.")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"hqca_prediction_{request_id}.pdf",
    )


@app.post("/generate_synthetic", response_model=GenerateSyntheticResponse, status_code=202)
def generate_synthetic(
    request: GenerateSyntheticRequest,
    background_tasks: BackgroundTasks,
) -> GenerateSyntheticResponse:
    task_id = uuid.uuid4().hex
    now = utc_now()
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        created_at=now,
        updated_at=now,
        num_samples=request.num_samples,
    )
    background_tasks.add_task(run_synthetic_generation, task_id, request)
    return GenerateSyntheticResponse(task_id=task_id, status="pending")


@app.get("/status/{task_id}", response_model=TaskStatus)
def status(task_id: str) -> TaskStatus:
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task
