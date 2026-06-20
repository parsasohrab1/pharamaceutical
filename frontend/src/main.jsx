import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const HISTORY_KEY = "hqca_prediction_history";

function absoluteUrl(path) {
  if (!path) return "#";
  if (path.startsWith("http")) return path;
  return `${API_BASE_URL}${path}`;
}

function loadLocalHistory() {
  try {
    return JSON.parse(window.localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function BindingScoreChart({ score }) {
  const safeScore = Math.max(0, Math.min(100, Number(score) || 0));
  return (
    <div className="score-chart" aria-label={`Binding score ${safeScore}`}>
      <div className="score-ring" style={{ "--score": `${safeScore * 3.6}deg` }}>
        <span>{safeScore}</span>
      </div>
      <div className="score-bars">
        <div>
          <span>Low</span>
          <span>Medium</span>
          <span>High</span>
        </div>
        <div className="bar-track">
          <div className="bar-fill" style={{ width: `${safeScore}%` }} />
        </div>
      </div>
    </div>
  );
}

function PocketViewer({ center }) {
  const mountRef = useRef(null);

  useEffect(() => {
    if (!center || !mountRef.current) return undefined;

    const mount = mountRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf3f8fb);
    const camera = new THREE.PerspectiveCamera(45, mount.clientWidth / 320, 0.1, 1000);
    camera.position.set(35, 28, 45);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mount.clientWidth, 320);
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const light = new THREE.DirectionalLight(0xffffff, 0.8);
    light.position.set(20, 30, 40);
    scene.add(light);

    const grid = new THREE.GridHelper(60, 12, 0x9fb3c8, 0xd7e0ea);
    scene.add(grid);
    scene.add(new THREE.AxesHelper(18));

    const geometry = new THREE.SphereGeometry(3.2, 32, 32);
    const material = new THREE.MeshStandardMaterial({
      color: 0x2b7a78,
      metalness: 0.1,
      roughness: 0.35,
    });
    const pocket = new THREE.Mesh(geometry, material);
    pocket.position.set(center.x, center.y, center.z);
    scene.add(pocket);

    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(5.2, 32, 32),
      new THREE.MeshBasicMaterial({ color: 0x3aafa9, transparent: true, opacity: 0.18 }),
    );
    halo.position.copy(pocket.position);
    scene.add(halo);

    let animationId;
    function animate() {
      pocket.rotation.y += 0.012;
      halo.rotation.y -= 0.006;
      renderer.render(scene, camera);
      animationId = window.requestAnimationFrame(animate);
    }
    animate();

    function resize() {
      camera.aspect = mount.clientWidth / 320;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, 320);
    }
    window.addEventListener("resize", resize);

    return () => {
      window.cancelAnimationFrame(animationId);
      window.removeEventListener("resize", resize);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [center]);

  return (
    <div>
      <div className="viewer" ref={mountRef} />
      <p className="viewer-caption">
        Interactive pocket center marker rendered with Three.js. Coordinates are
        returned by the backend prediction service.
      </p>
    </div>
  );
}

function ResultPage({ result }) {
  if (!result) return null;

  return (
    <section className="card result-page">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Results page</p>
          <h2>Prediction result</h2>
        </div>
        <div className="download-actions">
          <a href={absoluteUrl(result.report_csv_url)} download>
            Download CSV
          </a>
          <a href={absoluteUrl(result.report_pdf_url)} download>
            Download PDF
          </a>
        </div>
      </div>
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
      <div className="result-grid">
        <BindingScoreChart score={result.binding_score} />
        <PocketViewer center={result.pocket_center} />
      </div>
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
  const [history, setHistory] = useState(loadLocalHistory);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    requestJson("/history")
      .then((remoteHistory) => {
        if (remoteHistory.length > 0) {
          setHistory(remoteHistory.slice(0, 10));
        }
      })
      .catch(() => {
        // Local history still provides value when the API has no server-side state.
      });
  }, []);

  useEffect(() => {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 10)));
  }, [history]);

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
      setHistory((current) => [
        {
          request_id: payload.request_id,
          created_at: payload.created_at,
          smiles,
          fasta_preview: fasta.replace(/\s+/g, "").slice(0, 80),
          binding_score: payload.binding_score,
          binding_energy_kcal_mol: payload.binding_energy_kcal_mol,
          confidence: payload.confidence,
          report_csv_url: payload.report_csv_url,
          report_pdf_url: payload.report_pdf_url,
        },
        ...current.filter((item) => item.request_id !== payload.request_id),
      ].slice(0, 10));
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
        <h1>HQCA Product Preview</h1>
        <p>
          Run binding predictions, inspect a 3D pocket marker, export reports,
          and keep a lightweight request history.
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
      <ResultPage result={result} />
      <StatusCard task={task} />

      <section className="card">
        <div className="section-heading">
          <div>
            <p className="eyebrow">User history</p>
            <h2>Recent prediction requests</h2>
          </div>
          <button type="button" className="secondary" onClick={() => setHistory([])}>
            Clear history
          </button>
        </div>
        {history.length === 0 ? (
          <p className="muted">No prediction history yet.</p>
        ) : (
          <div className="history-list">
            {history.map((item) => (
              <article key={item.request_id} className="history-item">
                <div>
                  <strong>{item.smiles}</strong>
                  <p>
                    Score {item.binding_score} · Energy {item.binding_energy_kcal_mol} kcal/mol ·{" "}
                    {item.created_at}
                  </p>
                </div>
                <div className="download-actions compact">
                  <a href={absoluteUrl(item.report_csv_url)} download>
                    CSV
                  </a>
                  <a href={absoluteUrl(item.report_pdf_url)} download>
                    PDF
                  </a>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
