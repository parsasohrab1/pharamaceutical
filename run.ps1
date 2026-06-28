# HQCA quick start (Windows)
pip install -r requirements.txt
$env:HQCA_USE_MINIO = "false"
$env:HQCA_DATABASE_URL = "sqlite:///output/hqca.db"
Write-Host "Starting API on http://localhost:8000"
Start-Process python -ArgumentList "-m","uvicorn","api:app","--host","0.0.0.0","--port","8000"
Write-Host "Frontend: cd frontend; python -m http.server 5173"
