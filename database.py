"""Database models — PostgreSQL or SQLite (F7)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

DEFAULT_SQLITE_PATH = Path(os.getenv("HQCA_SQLITE_PATH", "output/hqca.db"))
DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
DATABASE_URL = os.getenv(
    "HQCA_DATABASE_URL",
    os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"),
)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="researcher", index=True)
    created_at: Mapped[str] = mapped_column(String(32), default=utc_now)

    projects: Mapped[list["Project"]] = relationship(back_populates="owner")
    predictions: Mapped[list["PredictionResult"]] = relationship(back_populates="user")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    created_at: Mapped[str] = mapped_column(String(32), default=utc_now)

    owner: Mapped[User] = relationship(back_populates="projects")
    predictions: Mapped[list["PredictionResult"]] = relationship(back_populates="project")


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    created_at: Mapped[str] = mapped_column(String(32), default=utc_now)
    encrypted_smiles: Mapped[str] = mapped_column(Text)
    encrypted_fasta: Mapped[str] = mapped_column(Text)
    binding_score: Mapped[float] = mapped_column(Float)
    binding_energy_kcal_mol: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    pocket_center_json: Mapped[str] = mapped_column(Text)
    report_csv_object: Mapped[str] = mapped_column(String(255))
    report_pdf_object: Mapped[str] = mapped_column(String(255))
    pocket_pdb_object: Mapped[str] = mapped_column(String(255))
    viewer_html_object: Mapped[str] = mapped_column(String(255), default="")
    backend: Mapped[str] = mapped_column(String(64), default="auto")

    user: Mapped[User | None] = relationship(back_populates="predictions")
    project: Mapped[Project | None] = relationship(back_populates="predictions")


class ProcessingTask(Base):
    __tablename__ = "processing_tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    created_at: Mapped[str] = mapped_column(String(32), default=utc_now)
    updated_at: Mapped[str] = mapped_column(String(32), default=utc_now)
    num_samples: Mapped[int] = mapped_column(Integer)
    records_generated: Mapped[int] = mapped_column(Integer, default=0)
    records_failed: Mapped[int] = mapped_column(Integer, default=0)
    output_csv_object: Mapped[str | None] = mapped_column(String(255), nullable=True)
    output_json_object: Mapped[str | None] = mapped_column(String(255), nullable=True)
    output_pdf_object: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
