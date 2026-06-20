import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function ResultCard({ result }) {
  if (!result) return null;

  return (
    <section className="card result-card">
      <h2>Prediction result</h2>
      <div className="metrics">
        <div>
          <span>Binding score</span>
          <strong>{result.binding_score}</strong>
        </div>
        <div>
          <span>Energy</span>
          <strong>{result.binding_energy_kcal_mol} kcal/mol</strong>
        </div>
        <div>
          <span>Confidence</span>
          <strong>{result.confidence}%</strong>
        </div>
      </div>
      <p className="pocket">
        Pocket center: x={result.pocket_center.x}, y={result.pocket_center.y}, z=
        {result.pocket_center.z}
      </p>
    </section>
  );
}

function StatusCard({ task }) {
  if (!task) return null;

  return (
    <section className="card">
      <h2>Synthetic generation status</h2>
      <dl>
        <dt>Task ID</dt>
        <dd>{task.task_id}</dd>
        <dt>Status</dt>
        <dd>{task.status}</dd>
        <dt>Records generated</dt>
        <dd>{task.records_generated}</dd>
        {task.output_csv && (
          <>
            <dt>CSV</dt>
            <dd>{task.output_csv}</dd>
          </>
        )}
        {task.error && (
          <>
            <dt>Error</dt>
            <dd className="error-text">{task.error}</dd>
          </>
        )}
      </dl>
    </section>
  );
}

function App() {
  const [smiles, setSmiles] = useState("CCO");
  const [fasta, setFasta] = useState(">target\nACDEFGHIKLMNPQRSTVWY");
  const [numSamples, setNumSamples] = useState(10);
  const [result, setResult] = useState(null);
  const [task, setTask] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function requestJson(path, options) {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(payload.detail)
        ? payload.detail.map((item) => item.msg).join("; ")
        : payload.detail || "Request failed";
      throw new Error(detail);
    }
    return payload;
  }

  async function handlePredict(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const payload = await requestJson("/predict", {
        method: "POST",
        body: JSON.stringify({ smiles, fasta }),
      });
      setResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleGenerateSynthetic() {
    setLoading(true);
    setError("");
    setTask(null);
    try {
      const started = await requestJson("/generate_synthetic", {
        method: "POST",
        body: JSON.stringify({ num_samples: Number(numSamples), smiles_seed: [smiles] }),
      });
      let current = started;
      for (let attempt = 0; attempt < 20; attempt += 1) {
        current = await requestJson(`/status/${started.task_id}`);
        setTask(current);
        if (current.status === "completed" || current.status === "failed") break;
        await new Promise((resolve) => setTimeout(resolve, 750));
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <header>
        <p className="eyebrow">Hybrid Quantum-Classical Assistant</p>
        <h1>HQCA MVP</h1>
        <p>
          Enter a drug SMILES string and target FASTA sequence to run a binding
          prediction through the FastAPI backend.
        </p>
      </header>

      <form className="card form" onSubmit={handlePredict}>
        <label>
          SMILES
          <input value={smiles} maxLength={200} onChange={(event) => setSmiles(event.target.value)} />
        </label>
        <label>
          FASTA
          <textarea rows={8} value={fasta} onChange={(event) => setFasta(event.target.value)} />
        </label>
        <div className="actions">
          <button type="submit" disabled={loading}>
            {loading ? "Working..." : "Predict"}
          </button>
        </div>
      </form>

      <section className="card synthetic">
        <h2>Synthetic data</h2>
        <label>
          Samples
          <input
            type="number"
            min="1"
            max="5000"
            value={numSamples}
            onChange={(event) => setNumSamples(event.target.value)}
          />
        </label>
        <button type="button" disabled={loading} onClick={handleGenerateSynthetic}>
          Generate synthetic dataset
        </button>
      </section>

      {error && <p className="error">{error}</p>}
      <ResultCard result={result} />
      <StatusCard task={task} />
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
