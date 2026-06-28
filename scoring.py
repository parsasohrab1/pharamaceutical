"""Binding score and confidence utilities for HQCA."""

from __future__ import annotations

import numpy as np

ENERGY_MIN_KCAL_MOL = -15.0
ENERGY_MAX_KCAL_MOL = -0.1


def binding_energy_to_score(energy_kcal_mol: float) -> float:
    """Map binding energy (kcal/mol) to a 0–100 score per FR-17."""
    score = 100.0 * (ENERGY_MAX_KCAL_MOL - energy_kcal_mol) / (
        ENERGY_MAX_KCAL_MOL - ENERGY_MIN_KCAL_MOL
    )
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def estimate_confidence(
    features_normalized: np.ndarray,
    fasta_length: int,
    pocket_count: int = 1,
) -> float:
    """Heuristic confidence (%) for FR-18."""
    in_range = float(np.mean((features_normalized > 0.0) & (features_normalized < 1.0)))
    length_factor = min(fasta_length, 60) / 60.0
    pocket_factor = min(pocket_count, 5) / 5.0
    confidence = 50.0 + 25.0 * in_range + 10.0 * length_factor + 10.0 * pocket_factor
    return round(float(np.clip(confidence, 50.0, 95.0)), 2)
