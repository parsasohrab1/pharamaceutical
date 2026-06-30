# HQCA — one command: API + Dashboard (auto-connect)
pip install -r requirements.txt -q
$env:HQCA_USE_MINIO = "false"
$env:HQCA_DATABASE_URL = "sqlite:///output/hqca.db"
$env:HQCA_SEED_DEMO = "true"
$env:HQCA_PORT = "18080"
python run_dashboard.py
