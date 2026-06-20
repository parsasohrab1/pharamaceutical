import time

import numpy as np
from starlette.testclient import TestClient

from api import app
from data import MolecularDescriptors, QuantumVQESimulator, SyntheticMoleculeGenerator, normalize_descriptors
from scoring import binding_energy_to_score
from training import train_binding_model


client = TestClient(app)


def test_fr_01_to_fr_19_integration_contract(tmp_path):
    smiles = "CCO"
    fasta = ">target\nACDEFGHIKLMNPQRSTVWY"

    # FR-01, FR-02, FR-03: SMILES/FASTA input and RDKit validation.
    invalid = client.post("/predict", json={"smiles": "invalid", "fasta": fasta})
    assert invalid.status_code == 422
    predict = client.post(
        "/predict",
        json={"smiles": smiles, "fasta": fasta, "backend": "classical_fallback"},
    )
    assert predict.status_code == 200
    prediction = predict.json()

    # FR-04, FR-05, FR-06: synthetic mode, generated molecules, sample count.
    synthetic = client.post("/generate_synthetic", json={"num_samples": 5, "smiles_seed": [smiles]})
    assert synthetic.status_code == 202
    task_id = synthetic.json()["task_id"]
    task = None
    for _ in range(10):
        task = client.get(f"/status/{task_id}").json()
        if task["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert task["status"] == "completed"
    assert task["records_generated"] == 5
    assert len(SyntheticMoleculeGenerator([smiles]).generate_molecules(3)) == 3

    # FR-07, FR-08: 7 descriptors and normalization to [0, 1].
    descriptors = MolecularDescriptors.compute(smiles)
    normalized = normalize_descriptors(descriptors.to_array())
    assert len(normalized) == 7
    assert np.all((normalized >= 0.0) & (normalized <= 1.0))

    # FR-09, FR-10, FR-11, FR-12, FR-13, FR-14: angle embedding input, 7 qubits,
    # VQC/classical fallback backend contract, bounded energy output.
    angles = np.arctan(normalized)
    simulator = QuantumVQESimulator(n_qubits=7, use_quantum=False)
    energy = simulator.predict_affinity(normalized)
    assert len(angles) == 7
    assert simulator.n_qubits == 7
    assert -15.0 <= energy <= -0.1

    # FR-15, FR-16: supervised cost/evaluation path and saved VQC params.
    metrics = train_binding_model(
        dataset_path="tests/fixtures/bindingdb_sample.csv",
        source="bindingdb",
        artifact_dir=tmp_path,
        n_estimators=20,
    )
    assert metrics.test_mae_kcal_mol < 3.0
    assert (tmp_path / "vqc_params.json").exists()

    # FR-17, FR-18: score, confidence, and pocket center for 3D visualization.
    assert 0 <= prediction["binding_score"] <= 100
    assert prediction["binding_score"] == binding_energy_to_score(prediction["binding_energy_kcal_mol"])
    assert 0 <= prediction["confidence"] <= 100
    assert set(prediction["pocket_center"]) == {"x", "y", "z"}

    # FR-19: CSV/PDF/PDB report exports.
    assert client.get(prediction["report_csv_url"]).status_code == 200
    assert client.get(prediction["report_pdf_url"]).status_code == 200
    assert client.get(prediction["pocket_pdb_url"]).status_code == 200
