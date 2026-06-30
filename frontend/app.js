/** HQCA Dashboard — auto-connect to API on load */

let API = "";
let token = localStorage.getItem("hqca_token") || "";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function apiCandidates() {
  const origin = window.location.origin;
  const list = [
    "",
    localStorage.getItem("hqca_api"),
    origin.includes("5173") ? "http://127.0.0.1:18080" : null,
    "http://127.0.0.1:18080",
    "http://localhost:18080",
  ].filter((v) => v !== null);
  const seen = new Set();
  return list.filter((v) => {
    const key = v || "__same_origin__";
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function apiUrl(path) {
  if (!path.startsWith("/")) path = `/${path}`;
  return API ? `${API}${path}` : path;
}

function setConnectionStatus(state, message) {
  const el = document.getElementById("connection-status");
  el.className = `connection ${state}`;
  el.textContent = message;
}

async function probeApi(base) {
  const url = base ? `${base}/health` : "/health";
  const res = await fetch(url, { signal: AbortSignal.timeout(4000) });
  if (!res.ok) return null;
  return base || window.location.origin;
}

async function ensureConnection() {
  setConnectionStatus("connecting", "در حال اتصال به API...");
  for (let attempt = 1; attempt <= 40; attempt++) {
    for (const base of apiCandidates()) {
      try {
        const resolved = await probeApi(base);
        if (resolved) {
          API = base;
          localStorage.setItem("hqca_api", resolved);
          setConnectionStatus("connected", `✓ متصل — ${resolved}`);
          return;
        }
      } catch {
        /* try next candidate */
      }
    }
    setConnectionStatus("connecting", `در حال اتصال... (${attempt}/40)`);
    await sleep(1000);
  }
  setConnectionStatus("error", "✗ اتصال برقرار نشد — API را اجرا کنید: python run_dashboard.py");
  throw new Error("API unavailable");
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(apiUrl(path), { ...options, headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail));
  }
  return res.json();
}

function showPrediction(item) {
  if (!item) return;
  document.getElementById("result-summary").innerHTML = `
    <strong>SMILES:</strong> ${item.smiles_preview || item.smiles || "—"}<br/>
    <strong>نمره اتصال:</strong> ${item.binding_score} / 100<br/>
    <strong>انرژی:</strong> ${Number(item.binding_energy_kcal_mol ?? 0).toFixed(3)} kcal/mol<br/>
    <strong>اطمینان:</strong> ${item.confidence}%<br/>
    <strong>Backend:</strong> ${item.backend || "auto"}
  `;
  const frame = document.getElementById("viewer-frame");
  if (item.viewer_html_url) {
    frame.src = apiUrl(item.viewer_html_url);
  }
  document.getElementById("download-links").innerHTML = `
    <p><a href="${apiUrl(item.report_pdf_url)}" target="_blank">PDF</a></p>
    <p><a href="${apiUrl(item.report_csv_url)}" target="_blank">CSV</a></p>
    <p><a href="${apiUrl(item.pocket_pdb_url)}" target="_blank">PDB</a></p>
  `;
}

function renderChart(predictions) {
  const el = document.getElementById("score-chart");
  if (!predictions.length) {
    el.innerHTML = "<p>داده‌ای برای نمودار نیست.</p>";
    return;
  }
  const max = Math.max(...predictions.map((p) => p.binding_score), 1);
  el.innerHTML = predictions
    .map(
      (p) => `
    <div class="bar-row">
      <span class="bar-label">${(p.smiles_preview || p.request_id).slice(0, 12)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(p.binding_score / max) * 100}%"></div></div>
      <span class="bar-val">${p.binding_score}</span>
    </div>`
    )
    .join("");
}

function renderHistory(predictions) {
  const tbody = document.getElementById("history-body");
  tbody.innerHTML = predictions
    .map(
      (p) => `
    <tr data-id="${p.request_id}" class="history-row">
      <td>${p.smiles_preview}</td>
      <td>${p.binding_score}</td>
      <td>${p.confidence}%</td>
      <td>${(p.created_at || "").slice(0, 19) || "—"}</td>
    </tr>`
    )
    .join("");
  tbody.querySelectorAll(".history-row").forEach((row) => {
    row.onclick = async () => {
      const detail = await api(`/predictions/${row.dataset.id}`);
      showPrediction(detail);
    };
  });
}

function renderDataset(datasets) {
  const panel = document.getElementById("dataset-panel");
  if (!datasets.length) {
    panel.textContent = "دیتاستی موجود نیست.";
    return;
  }
  const d = datasets[0];
  panel.innerHTML = `
    <p><strong>Task:</strong> ${d.task_id}</p>
    <p><strong>نمونه‌ها:</strong> ${d.records_generated} / ${d.num_samples}</p>
    <p><a href="${apiUrl(d.output_csv)}" target="_blank">دانلود CSV</a></p>
    <p><a href="${apiUrl(d.output_json)}" target="_blank">دانلود JSON</a></p>
    <p><a href="${apiUrl(d.output_pdf)}" target="_blank">دانلود PDF</a></p>
  `;
}

async function loadDashboard() {
  const data = await api("/dashboard");
  document.getElementById("stat-predictions").textContent = data.stats.total_predictions;
  document.getElementById("stat-datasets").textContent = data.stats.total_synthetic_jobs;
  document.getElementById("stat-avg-score").textContent = data.stats.avg_binding_score;
  if (data.latest_prediction) showPrediction(data.latest_prediction);
  renderChart(data.predictions);
  renderHistory(data.predictions);
  renderDataset(data.synthetic_datasets);
}

async function autoLogin() {
  const username = document.getElementById("username").value;
  const password = document.getElementById("password").value;
  const data = await api("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  token = data.access_token;
  localStorage.setItem("hqca_token", token);
  document.getElementById("auth-status").textContent = `✓ ${data.role}`;
  await loadDashboard();
}

document.getElementById("login-btn").onclick = async () => {
  try {
    await ensureConnection();
    await autoLogin();
  } catch (e) {
    document.getElementById("auth-status").textContent = e.message;
  }
};

document.getElementById("predict-btn").onclick = async () => {
  const body = {
    smiles: document.getElementById("smiles").value,
    fasta: document.getElementById("fasta").value,
    backend: document.getElementById("backend").value,
  };
  const data = await api("/predict", { method: "POST", body: JSON.stringify(body) });
  showPrediction(data);
  await loadDashboard();
};

document.getElementById("generate-btn").onclick = async () => {
  const num_samples = Number(document.getElementById("num-samples").value);
  const smiles_seed = document.getElementById("seed-smiles").value.split(",").map((s) => s.trim());
  const data = await api("/generate_synthetic", {
    method: "POST",
    body: JSON.stringify({ num_samples, smiles_seed }),
  });
  const el = document.getElementById("task-status");
  el.textContent = `Task ${data.task_id}: ${data.status}`;
  const poll = setInterval(async () => {
    const st = await api(`/status/${data.task_id}`);
    el.textContent = JSON.stringify(st, null, 2);
    if (st.status === "completed" || st.status === "failed") {
      clearInterval(poll);
      await loadDashboard();
    }
  }, 2000);
};

async function bootstrap() {
  await ensureConnection();
  await autoLogin();
}

bootstrap().catch((e) => {
  document.getElementById("auth-status").textContent = "ورود دستی پس از اتصال API";
  document.getElementById("result-summary").textContent = e.message;
});
