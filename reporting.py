"""Patent and dataset reports: TXT + PDF (F8, FR-19)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_text_report(df: pd.DataFrame, output_file: str = "data_report.txt") -> str:
    """Statistical text report for patent appendix."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("گزارش تولید داده سنتتیک برای سامانه HQCA\n")
        f.write("========================================\n")
        f.write(f"تاریخ تولید: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"تعداد نمونه‌ها: {len(df)}\n\n")
        f.write("محدوده توصیفگرها:\n")
        cols = [
            "MW", "LogP", "HBD", "HBA", "RotatableBonds",
            "AromaticRings", "TPSA", "binding_energy_kcal_mol", "binding_score",
        ]
        for col in cols:
            if col in df.columns:
                f.write(
                    f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
                    f"mean={df[col].mean():.2f}\n"
                )
        if "binding_energy_kcal_mol" in df.columns:
            f.write("\nتوزیع انرژی اتصال:\n")
            f.write(df["binding_energy_kcal_mol"].value_counts(bins=10).to_string())
    return str(path)


def generate_pdf_report(
    prediction: Dict[str, Any],
    output_file: str = "hqca_report.pdf",
    dataset_summary: Optional[pd.DataFrame] = None,
) -> str:
    """Patent-ready PDF report (FR-19, F8)."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not REPORTLAB_AVAILABLE:
        _write_minimal_pdf(path, prediction)
        return str(path)

    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 50
    lines = [
        "HQCA - Hybrid Quantum-Classical Drug Binding Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"SMILES: {prediction.get('smiles', 'N/A')}",
        f"Protein length: {prediction.get('protein_length', 'N/A')}",
        f"Binding energy (kcal/mol): {prediction.get('binding_energy_kcal_mol', 0):.3f}",
        f"Binding score (0-100): {prediction.get('binding_score', 0):.2f}",
        f"Confidence: {prediction.get('confidence_pct', 0):.1f}%",
        f"Pockets generated: {prediction.get('pocket_count', 1)}",
        f"PDB file: {prediction.get('pdb_path', 'N/A')}",
        f"3D viewer: {prediction.get('viewer_html_path', 'N/A')}",
        "",
        "Method: VQE variational quantum circuit (FR-11) + RDKit descriptors",
        "Synthetic generation: ChemBFN/PocketGen-compatible RDKit pipeline (F1)",
    ]
    if dataset_summary is not None and len(dataset_summary) > 0:
        lines.extend(["", f"Dataset samples: {len(dataset_summary)}"])
        if "binding_score" in dataset_summary.columns:
            lines.append(
                f"Mean binding score: {dataset_summary['binding_score'].mean():.2f}"
            )

    for line in lines:
        c.drawString(50, y, line[:95])
        y -= 18
        if y < 60:
            c.showPage()
            y = height - 50
    c.save()
    return str(path)


def _write_minimal_pdf(path: Path, prediction: Dict[str, Any]) -> None:
    """Fallback one-page PDF without reportlab."""
    score = prediction.get("binding_score", 0)
    energy = prediction.get("binding_energy_kcal_mol", 0)
    content = (
        f"HQCA Report | score={score} | energy={energy:.3f} kcal/mol | "
        f"confidence={prediction.get('confidence_pct', 0)}%"
    )
    # Minimal valid PDF structure
    objects = []
    objects.append("1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n")
    objects.append("2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n")
    objects.append(
        "3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj\n"
    )
    stream = f"BT /F1 12 Tf 50 750 Td ({content[:120]}) Tj ET"
    objects.append(f"4 0 obj<< /Length {len(stream)} >>stream\n{stream}\nendstream endobj\n")
    objects.append("5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n")
    xref_offset = 0
    body = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(body))
        body += obj
    xref_start = len(body)
    body += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n"
    for off in offsets[1:]:
        body += f"{off:010d} 00000 n \n"
    body += f"trailer<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF"
    path.write_text(body, encoding="latin-1")
