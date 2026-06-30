const API = window.location.origin.includes("5173")
  ? "http://127.0.0.1:18080"
  : window.location.origin;

let token = localStorage.getItem("hqca_token") || "";

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`${API}${path}`, { ...options, headers });
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
    <strong>انرژی:</strong> ${(item.binding_energy_kcal_mol ?? 0).toFixed?.(3) ?? item.binding_energy_kcal_mol} kcal/mol<br/>
    <strong>اطمینان:</strong> ${item.confidence}%<br/>
    <strong>Backend:</strong> ${item.backend || "auto"}
  `;
  const frame = document.getElementById("viewer-frame");
  if (item.viewer_html_url) {
    frame.src = `${API}${item.viewer_html_url}`;
  }
  document.getElementById("download-links").innerHTML = `
    <p><a href="${API}${item.report_pdf_url}" target="_blank">📄 PDF</a></p>
    <p><a href="${API}${item.report_csv_url}" target="_blank">📊 CSV</a></p>
    <p><a href="${API}${item.pocket_pdb_url}" target="_blank">🧬 PDB</a></p>
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
      <span class="bar-label">${p.smiles_preview?.slice(0, 12) || p.request_id.slice(0, 8)}</span>
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
      <td>${p.created_at?.slice(0, 19) || "—"}</td>
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
    <p><a href="${API}${d.output_csv}" target="_blank">دانلود CSV</a></p>
    <p><a href="${API}${d.output_json}" target="_blank">دانلود JSON</a></p>
    <p><a href="${API}${d.output_pdf}" target="_blank">دانلود PDF</a></p>
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

document.getElementById("login-btn").onclick = autoLogin;

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

autoLogin().catch((e) => {
  document.getElementById("auth-status").textContent = "ورود دستی لازم است";
  document.getElementById("result-summary").textContent = e.message;
});
