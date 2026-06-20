"""Training pipeline for real BindingDB/PDBbind style HQCA datasets."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from data import LOGGER, MolecularDescriptors, normalize_descriptors
from scoring import ENERGY_MAX_KCAL_MOL, ENERGY_MIN_KCAL_MOL, binding_energy_to_score


R_KCAL_MOL_K = 0.0019872041
DEFAULT_TEMPERATURE_K = 298.15
FEATURE_COLUMNS = ["MW", "LogP", "HBD", "HBA", "RotatableBonds", "AromaticRings", "TPSA"]

SMILES_COLUMNS = [
    "smiles",
    "SMILES",
    "Ligand SMILES",
    "ligand_smiles",
    "canonical_smiles",
]
FASTA_COLUMNS = ["fasta", "FASTA", "protein_sequence", "target_sequence", "sequence"]
ENERGY_COLUMNS = ["binding_energy_kcal_mol", "delta_g_kcal_mol", "DeltaG", "dG"]
PKD_COLUMNS = ["pKd", "pki", "pKi", "affinity_pKd", "-logKd/Ki"]
AFFINITY_NM_COLUMNS = [
    "affinity_nM",
    "Kd (nM)",
    "Ki (nM)",
    "IC50 (nM)",
    "EC50 (nM)",
    "kd_nm",
    "ki_nm",
    "ic50_nm",
]


@dataclass
class TrainingMetrics:
    source: str
    records_loaded: int
    records_used: int
    model_type: str
    train_mae_kcal_mol: float
    test_mae_kcal_mol: float
    vqc_surrogate_test_mae_kcal_mol: float
    baseline_score_mae: float
    energy_min_kcal_mol: float
    energy_max_kcal_mol: float
    created_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def first_existing_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def parse_numeric(value: object) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if match is None:
        return None
    return float(match.group(0))


def affinity_nm_to_energy(affinity_nm: float, temperature_k: float = DEFAULT_TEMPERATURE_K) -> float:
    """Convert Kd/Ki/IC50-like nM affinity values to DeltaG in kcal/mol."""
    if affinity_nm <= 0:
        raise ValueError("Affinity must be positive to convert to binding energy.")
    affinity_molar = affinity_nm * 1e-9
    return float(R_KCAL_MOL_K * temperature_k * np.log(affinity_molar))


def pkd_to_energy(pkd: float, temperature_k: float = DEFAULT_TEMPERATURE_K) -> float:
    kd_molar = 10 ** (-pkd)
    return float(R_KCAL_MOL_K * temperature_k * np.log(kd_molar))


def extract_binding_energy(row: pd.Series) -> Optional[float]:
    energy_col = first_existing_column(row.index, ENERGY_COLUMNS)
    if energy_col:
        value = parse_numeric(row[energy_col])
        if value is not None:
            return value

    pkd_col = first_existing_column(row.index, PKD_COLUMNS)
    if pkd_col:
        value = parse_numeric(row[pkd_col])
        if value is not None:
            return pkd_to_energy(value)

    for affinity_col in AFFINITY_NM_COLUMNS:
        if affinity_col in row.index:
            value = parse_numeric(row[affinity_col])
            if value is not None and value > 0:
                return affinity_nm_to_energy(value)
    return None


def load_binding_dataset(dataset_path: str | Path, source: str = "bindingdb") -> pd.DataFrame:
    """Load BindingDB/PDBbind-style CSV data into normalized HQCA records.

    Supported inputs may provide either `binding_energy_kcal_mol`, pKd-style
    columns, or nM affinity columns such as `Kd (nM)`, `Ki (nM)`, or `IC50 (nM)`.
    """
    raw = pd.read_csv(dataset_path)
    smiles_col = first_existing_column(raw.columns, SMILES_COLUMNS)
    if smiles_col is None:
        raise ValueError(f"Dataset must contain one SMILES column from: {SMILES_COLUMNS}")

    fasta_col = first_existing_column(raw.columns, FASTA_COLUMNS)
    records: List[Dict[str, object]] = []
    for _, row in raw.iterrows():
        smiles = str(row[smiles_col]).strip()
        if not smiles:
            continue
        energy = extract_binding_energy(row)
        if energy is None:
            continue
        records.append(
            {
                "source": source,
                "smiles": smiles,
                "fasta": str(row[fasta_col]).strip() if fasta_col else "",
                "binding_energy_kcal_mol": float(energy),
            }
        )

    normalized = pd.DataFrame(records)
    if normalized.empty:
        raise ValueError("No usable binding records were loaded from the dataset.")
    return normalized


def featurize_records(records: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_rows: List[np.ndarray] = []
    target_rows: List[float] = []
    kept_records: List[Dict[str, object]] = []

    for record in records.to_dict(orient="records"):
        try:
            descriptors = MolecularDescriptors.compute(str(record["smiles"]))
        except ValueError:
            continue
        raw_features = descriptors.to_array()
        feature_rows.append(normalize_descriptors(raw_features))
        target_rows.append(float(record["binding_energy_kcal_mol"]))
        kept_records.append({**record, **dict(zip(FEATURE_COLUMNS, raw_features))})

    if not feature_rows:
        raise ValueError("No records could be featurized with RDKit descriptors.")
    return np.vstack(feature_rows), np.asarray(target_rows, dtype=float), pd.DataFrame(kept_records)


def fit_vqc_surrogate_params(features: np.ndarray, targets: np.ndarray) -> Dict[str, object]:
    """Fit lightweight VQC surrogate parameters for persisted calibration.

    The actual quantum backend remains optional in this repository. These
    persisted parameters provide a supervised calibration vector over the same
    seven normalized descriptor inputs and can be replaced by PennyLane/Qiskit
    optimized VQC parameters when that training path is enabled.
    """
    regressor = Ridge(alpha=1e-3)
    regressor.fit(features, targets)
    return {
        "type": "ridge_vqc_surrogate",
        "n_qubits": int(features.shape[1]),
        "weights": regressor.coef_.astype(float).round(8).tolist(),
        "intercept": float(regressor.intercept_),
        "feature_columns": FEATURE_COLUMNS,
    }


def predict_vqc_surrogate(vqc_params: Dict[str, object], features: np.ndarray) -> np.ndarray:
    weights = np.asarray(vqc_params["weights"], dtype=float)
    intercept = float(vqc_params["intercept"])
    return np.dot(features, weights) + intercept


def build_baseline_model(model_type: str, n_estimators: int, random_state: int):
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            min_samples_leaf=1,
        )
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost baseline requested but xgboost is not installed. "
                "Install with: python -m pip install -e '.[xgboost]'"
            ) from exc
        return XGBRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            objective="reg:squarederror",
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def train_binding_model(
    dataset_path: str | Path,
    source: str,
    artifact_dir: str | Path,
    model_type: str = "random_forest",
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 200,
) -> TrainingMetrics:
    records = load_binding_dataset(dataset_path, source=source)
    features, targets, featurized = featurize_records(records)
    if len(targets) < 4:
        raise ValueError("At least 4 usable records are required for train/test evaluation.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
    )
    model = build_baseline_model(model_type, n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    vqc_params = fit_vqc_surrogate_params(x_train, y_train)
    vqc_test_predictions = predict_vqc_surrogate(vqc_params, x_test)

    test_score_predictions = [binding_energy_to_score(value) for value in test_predictions]
    test_score_targets = [binding_energy_to_score(value) for value in y_test]
    metrics = TrainingMetrics(
        source=source,
        records_loaded=int(len(records)),
        records_used=int(len(featurized)),
        model_type=model_type,
        train_mae_kcal_mol=round(float(mean_absolute_error(y_train, train_predictions)), 4),
        test_mae_kcal_mol=round(float(mean_absolute_error(y_test, test_predictions)), 4),
        vqc_surrogate_test_mae_kcal_mol=round(float(mean_absolute_error(y_test, vqc_test_predictions)), 4),
        baseline_score_mae=round(float(mean_absolute_error(test_score_targets, test_score_predictions)), 4),
        energy_min_kcal_mol=ENERGY_MIN_KCAL_MOL,
        energy_max_kcal_mol=ENERGY_MAX_KCAL_MOL,
        created_at=utc_now(),
    )

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "model_type": metrics.model_type,
        "feature_columns": FEATURE_COLUMNS,
        "vqc_params": vqc_params,
        "metrics": asdict(metrics),
        "score_formula": {
            "description": "score = 100 * (E_max - E) / (E_max - E_min), clipped to [0, 100]",
            "energy_min_kcal_mol": ENERGY_MIN_KCAL_MOL,
            "energy_max_kcal_mol": ENERGY_MAX_KCAL_MOL,
        },
    }
    joblib.dump(artifact, artifact_dir / "hqca_model.joblib")
    (artifact_dir / "vqc_params.json").write_text(json.dumps(vqc_params, indent=2), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    featurized.to_csv(artifact_dir / "training_records.csv", index=False)

    LOGGER.info(
        "Training pipeline completed.",
        extra={
            "event": "training_completed",
            "records_generated": metrics.records_used,
            "duration_seconds": metrics.test_mae_kcal_mol,
        },
    )
    return metrics


def load_model_artifact(artifact_path: str | Path) -> Dict[str, object]:
    return joblib.load(artifact_path)


def predict_energy_from_artifact(artifact: Dict[str, object], features_normalized: np.ndarray) -> float:
    model = artifact["model"]
    prediction = model.predict(np.asarray([features_normalized], dtype=float))[0]
    return float(prediction)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train HQCA on BindingDB/PDBbind-style CSV data.")
    parser.add_argument("--dataset", required=True, help="Path to a BindingDB/PDBbind-style CSV file.")
    parser.add_argument("--source", default="bindingdb", choices=["bindingdb", "pdbbind", "custom"])
    parser.add_argument("--artifact-dir", default="models", help="Directory for model and metrics artifacts.")
    parser.add_argument("--model-type", default="random_forest", choices=["random_forest", "xgboost"])
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = train_binding_model(
        dataset_path=args.dataset,
        source=args.source,
        artifact_dir=args.artifact_dir,
        model_type=args.model_type,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(asdict(metrics), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
