"""Binding score conversion utilities for HQCA."""

from __future__ import annotations

import numpy as np


ENERGY_MIN_KCAL_MOL = -15.0
ENERGY_MAX_KCAL_MOL = -0.1


def binding_energy_to_score(energy_kcal_mol: float) -> float:
    """Convert binding energy to a 0-100 score.

    HQCA uses a documented linear mapping where stronger, more negative binding
    energies receive higher scores:

        score = 100 * (E_max - E) / (E_max - E_min)

    with E_min = -15.0 kcal/mol and E_max = -0.1 kcal/mol. Values outside this
    calibrated range are clipped to [0, 100].
    """
    score = 100.0 * (ENERGY_MAX_KCAL_MOL - energy_kcal_mol) / (
        ENERGY_MAX_KCAL_MOL - ENERGY_MIN_KCAL_MOL
    )
    return round(float(np.clip(score, 0.0, 100.0)), 2)
