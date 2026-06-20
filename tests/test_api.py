import time

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_predict_returns_standard_output():
    response = client.post(
        "/predict",
        json={"smiles": "CCO", "fasta": ">target\nACDEFGHIKLMNPQRSTVWY"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "binding_score",
        "binding_energy_kcal_mol",
        "confidence",
        "pocket_center",
    }
    assert 0 <= payload["binding_score"] <= 100
    assert -15 <= payload["binding_energy_kcal_mol"] <= -0.1
    assert 0 <= payload["confidence"] <= 100
    assert set(payload["pocket_center"]) == {"x", "y", "z"}


def test_predict_rejects_invalid_smiles_and_fasta():
    invalid_smiles = client.post("/predict", json={"smiles": "not-valid", "fasta": "ACDE"})
    invalid_fasta = client.post("/predict", json={"smiles": "CCO", "fasta": "ACDZ"})

    assert invalid_smiles.status_code == 422
    assert invalid_fasta.status_code == 422


def test_generate_synthetic_creates_status_task():
    response = client.post(
        "/generate_synthetic",
        json={"num_samples": 1, "smiles_seed": ["CCO"]},
    )

    assert response.status_code == 202
    task_id = response.json()["task_id"]

    status = None
    for _ in range(10):
        status_response = client.get(f"/status/{task_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        if status["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert status is not None
    assert status["status"] == "completed"
    assert status["records_generated"] == 1
    assert status["output_csv"]
    assert status["output_json"]
    assert status["output_metrics"]


def test_status_returns_404_for_unknown_task():
    response = client.get("/status/missing")

    assert response.status_code == 404
