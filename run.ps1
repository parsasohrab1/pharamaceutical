# HQCA quick start (Windows)
pip install -r requirements.txt
$env:HQCA_USE_MINIO = "false"
$env:HQCA_DATABASE_URL = "sqlite:///output/hqca.db"
$env:HQCA_PORT = "18080"
Write-Host "API:    http://127.0.0.1:18080/docs"
Write-Host "Dashboard: http://127.0.0.1:5173"
Start-Process python -ArgumentList "run_api.py"
Start-Sleep -Seconds 2
Start-Process python -ArgumentList "-m","http.server","5173","--bind","127.0.0.1" -WorkingDirectory "frontend"
