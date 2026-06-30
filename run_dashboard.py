#!/usr/bin/env python3
"""Start API + open dashboard with guaranteed connection."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser


def wait_for_api(port: int, timeout: int = 60) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    for i in range(timeout):
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
        print(f"Waiting for API... ({i + 1}/{timeout})")
    return False


def main() -> int:
    os.environ.setdefault("HQCA_USE_MINIO", "false")
    os.environ.setdefault("HQCA_DATABASE_URL", "sqlite:///output/hqca.db")
    os.environ.setdefault("HQCA_SEED_DEMO", "true")
    port = int(os.environ.get("HQCA_PORT", "18080"))

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api:app",
        "--host",
        "127.0.0.1",
        f"--port",
        str(port),
    ]
    print(f"Starting HQCA API on port {port}...")
    proc = subprocess.Popen(cmd)

    if not wait_for_api(port):
        print("ERROR: API did not start in time.")
        proc.terminate()
        return 1

    dashboard_url = f"http://127.0.0.1:{port}/"
    docs_url = f"http://127.0.0.1:{port}/docs"
    print(f"\nDashboard: {dashboard_url}")
    print(f"API Docs:  {docs_url}")
    print("Connection: dashboard and API on same origin (auto-connect)\n")

    try:
        webbrowser.open(dashboard_url)
    except Exception:
        pass

    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
