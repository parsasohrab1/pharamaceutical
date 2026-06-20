import time
import uuid
import warnings

from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
    category=StarletteDeprecationWarning,
)
from starlette.testclient import TestClient

from api import app
from database import PredictionResult, SessionLocal
from security import decrypt_sensitive


client = TestClient(app)


def test_predict_returns_standard_output():
    response = client.post(
        "/predict",
        json={"smiles": "CCO", "fasta": ">target\nACDEFGHIKLMNPQRSTVWY"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "request_id",
        "created_at",
        "binding_score",
        "binding_energy_kcal_mol",
        "confidence",
        "pocket_center",
        "report_csv_url",
        "report_pdf_url",
        "pocket_pdb_url",
    }
    assert 0 <= payload["binding_score"] <= 100
    assert -15 <= payload["binding_energy_kcal_mol"] <= -0.1
    assert 0 <= payload["confidence"] <= 100
    assert set(payload["pocket_center"]) == {"x", "y", "z"}
    assert payload["report_csv_url"].endswith("/csv")
    assert payload["report_pdf_url"].endswith("/pdf")
    assert payload["pocket_pdb_url"].endswith("/pdb")


def test_prediction_reports_and_history_are_downloadable():
    response = client.post(
        "/predict",
        json={"smiles": "CCO", "fasta": ">target\nACDEFGHIKLMNPQRSTVWY"},
    )
    payload = response.json()

    csv_response = client.get(payload["report_csv_url"])
    pdf_response = client.get(payload["report_pdf_url"])
    pdb_response = client.get(payload["pocket_pdb_url"])
    history_response = client.get("/history")

    assert csv_response.status_code == 200
    assert "binding_score" in csv_response.text
    assert pdf_response.status_code == 200
    assert pdf_response.headers["content-type"].startswith("application/pdf")
    assert pdb_response.status_code == 200
    assert "HQCA PREDICTED POCKET CENTER" in pdb_response.text
    assert history_response.status_code == 200
    assert any(item["request_id"] == payload["request_id"] for item in history_response.json())


def test_auth_project_prediction_persists_encrypted_result_and_rbac():
    username = f"researcher-{uuid.uuid4().hex[:8]}"
    password = "strong-password"

    register_response = client.post(
        "/auth/register",
        json={"username": username, "password": password},
    )
    login_response = client.post("/auth/login", json={"username": username, "password": password})
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    project_response = client.post("/projects", json={"name": "Kinase screen"}, headers=headers)
    me_response = client.get("/users/me", headers=headers)
    admin_response = client.get("/admin/users", headers=headers)
    predict_response = client.post(
        "/predict",
        json={
            "smiles": "CCO",
            "fasta": ">target\nACDEFGHIKLMNPQRSTVWY",
            "project_id": project_response.json()["id"],
        },
        headers=headers,
    )

    assert register_response.status_code == 201
    assert login_response.status_code == 200
    assert project_response.status_code == 201
    assert me_response.status_code == 200
    assert admin_response.status_code == 403
    assert predict_response.status_code == 200

    request_id = predict_response.json()["request_id"]
    db = SessionLocal()
    try:
        result = db.get(PredictionResult, request_id)
        assert result is not None
        assert result.encrypted_smiles != "CCO"
        assert decrypt_sensitive(result.encrypted_smiles) == "CCO"
        assert result.project_id == project_response.json()["id"]
    finally:
        db.close()


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
