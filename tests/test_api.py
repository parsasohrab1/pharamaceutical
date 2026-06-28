"""API integration tests."""


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert "backends" in res.json()


def test_login_and_predict(client):
    login = client.post("/auth/login", json={"username": "admin", "password": "admin12345"})
    assert login.status_code == 200
    token = login.json()["access_token"]
    res = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {token}"},
        json={"smiles": "CCO", "fasta": "ACDEFGHIKLMNPQRSTVWY", "backend": "auto"},
    )
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["binding_score"] <= 100
    assert body["confidence"] >= 50
    assert body["viewer_html_url"].startswith("/files/")


def test_generate_synthetic(client):
    res = client.post(
        "/generate_synthetic",
        json={"num_samples": 3, "smiles_seed": ["CCO", "CC(C)O"]},
    )
    assert res.status_code == 200
    task_id = res.json()["task_id"]
    status = client.get(f"/status/{task_id}")
    assert status.status_code == 200


def test_cobyla_optimize():
    import numpy as np
    from data import QuantumVQESimulator

    sim = QuantumVQESimulator(backend="auto")
    energy, depth = sim.predict_affinity(np.ones(7) * 0.5, optimize=True)
    assert -15.0 <= energy <= -0.1
    assert depth >= 0
