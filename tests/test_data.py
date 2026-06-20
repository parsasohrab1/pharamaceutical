import json

import numpy as np
import pytest


pytest.importorskip("rdkit")

import data


def test_normalize_descriptors_clips_values():
    raw = np.array([900.0, -10.0, 3.0, 5.0, 25.0, 2.0, 100.0])

    normalized = data.normalize_descriptors(raw)

    assert normalized.shape == (7,)
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)
    assert normalized[0] == 1.0
    assert normalized[1] == 0.0


def test_molecular_descriptors_for_ethanol():
    descriptors = data.MolecularDescriptors.compute("CCO")

    assert descriptors.MW == pytest.approx(46.069, rel=1e-3)
    assert descriptors.HBD == 1
    assert descriptors.HBA == 1
    assert len(descriptors.to_array()) == 7


def test_classical_affinity_prediction_stays_in_expected_range():
    simulator = data.QuantumVQESimulator(use_quantum=False)

    energy = simulator.predict_affinity(np.full(7, 0.5))

    assert -15.0 <= energy <= -0.1


def test_pipeline_writes_outputs_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PENNYLANE_AVAILABLE", False)
    pipeline = data.SyntheticDataPipeline(["CCO"], random_state=7)

    output_csv = tmp_path / "dataset.csv"
    output_json = tmp_path / "dataset.json"
    output_metrics = tmp_path / "metrics.json"

    df = pipeline.generate_dataset(
        num_pairs=3,
        output_csv=str(output_csv),
        output_json=str(output_json),
        output_metrics=str(output_metrics),
    )

    assert len(df) == 3
    assert output_csv.exists()
    assert output_json.exists()
    assert output_metrics.exists()

    metrics = json.loads(output_metrics.read_text())
    assert metrics["num_pairs_requested"] == 3
    assert metrics["records_generated"] == 3
    assert metrics["records_failed"] == 0
    assert metrics["use_quantum"] is False
