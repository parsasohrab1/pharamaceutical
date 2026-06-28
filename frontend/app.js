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
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

document.getElementById("login-btn").onclick = async () => {
  const username = document.getElementById("username").value;
  const password = document.getElementById("password").value;
  const data = await api("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  token = data.access_token;
  localStorage.setItem("hqca_token", token);
  document.getElementById("auth-status").textContent = `Logged in (${data.role})`;
};

document.getElementById("predict-btn").onclick = async () => {
  const body = {
    smiles: document.getElementById("smiles").value,
    fasta: document.getElementById("fasta").value,
    backend: document.getElementById("backend").value,
  };
  const data = await api("/predict", { method: "POST", body: JSON.stringify(body) });
  document.getElementById("result-summary").innerHTML = `
    <strong>نمره اتصال:</strong> ${data.binding_score} / 100<br/>
    <strong>انرژی:</strong> ${data.binding_energy_kcal_mol.toFixed(3)} kcal/mol<br/>
    <strong>اطمینان:</strong> ${data.confidence}%<br/>
    <strong>Backend:</strong> ${data.backend} (depth ${data.gate_depth})
  `;
  const frame = document.getElementById("viewer-frame");
  frame.style.display = "block";
  frame.src = `${API}${data.viewer_html_url}`;
  document.getElementById("download-links").innerHTML = `
    <p><a href="${API}${data.report_pdf_url}" target="_blank">PDF report</a></p>
    <p><a href="${API}${data.report_csv_url}" target="_blank">CSV data</a></p>
    <p><a href="${API}${data.pocket_pdb_url}" target="_blank">PDB pocket</a></p>
  `;
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
    if (st.status === "completed" || st.status === "failed") clearInterval(poll);
  }, 2000);
};
