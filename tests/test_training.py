import json
import math

import numpy as np

import training
from data import MolecularDescriptors, normalize_descriptors
from scoring import binding_energy_to_score


FIXTURE_DATASET = "tests/fixtures/bindingdb_sample.csv"


def test_affinity_nm_conversion_matches_thermodynamic_formula():
    energy = training.affinity_nm_to_energy(10.0)

    assert energy == np.float64(energy)
    assert energy == training.pkd_to_energy(8.0)
    assert math.isclose(energy, -10.9228, rel_tol=1e-3)


def test_binding_score_formula_is_monotonic_and_clipped():
    assert binding_energy_to_score(-15.0) == 100.0
    assert binding_energy_to_score(-0.1) == 0.0
    assert binding_energy_to_score(-30.0) == 100.0
    assert binding_energy_to_score(1.0) == 0.0
    assert binding_energy_to_score(-8.0) > binding_energy_to_score(-4.0)


def test_training_pipeline_saves_model_metrics_and_vqc_params(tmp_path):
    metrics = training.train_binding_model(
        dataset_path=FIXTURE_DATASET,
        source="bindingdb",
        artifact_dir=tmp_path,
        n_estimators=50,
        random_state=11,
    )

    assert metrics.records_used == 12
    assert metrics.test_mae_kcal_mol < 3.0
    assert metrics.vqc_surrogate_test_mae_kcal_mol < 3.0
    assert (tmp_path / "hqca_model.joblib").exists()
    assert (tmp_path / "vqc_params.json").exists()
    assert (tmp_path / "metrics.json").exists()

    vqc_params = json.loads((tmp_path / "vqc_params.json").read_text())
    assert vqc_params["n_qubits"] == 7
    assert len(vqc_params["weights"]) == 7


def test_saved_artifact_predicts_binding_energy(tmp_path):
    training.train_binding_model(
        dataset_path=FIXTURE_DATASET,
        source="bindingdb",
        artifact_dir=tmp_path,
        n_estimators=50,
        random_state=11,
    )
    artifact = training.load_model_artifact(tmp_path / "hqca_model.joblib")
    descriptors = MolecularDescriptors.compute("CCO")
    features = normalize_descriptors(descriptors.to_array())

    energy = training.predict_energy_from_artifact(artifact, features)

    assert -15.0 <= energy <= -0.1
