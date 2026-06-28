"""SMILES and FASTA validation (FR-01, FR-02, FR-03)."""

from __future__ import annotations

from rdkit import Chem

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def validate_smiles(value: str) -> str:
    smiles = value.strip()
    if len(smiles) > 200:
        raise ValueError("SMILES must be 200 characters or fewer.")
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError("SMILES is not valid.")
    return smiles


def normalize_fasta(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    sequence = "".join(line for line in lines if not line.startswith(">")).upper()
    if not sequence:
        raise ValueError("FASTA sequence is required.")
    invalid = sorted(set(sequence) - VALID_AMINO_ACIDS)
    if invalid:
        raise ValueError(f"FASTA contains invalid amino acid codes: {', '.join(invalid)}")
    return sequence
