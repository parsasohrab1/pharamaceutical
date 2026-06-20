#!/usr/bin/env python3
"""Lightweight healthcheck for the HQCA batch pipeline runtime."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HQCA_LOG_LEVEL", "ERROR")


def main() -> int:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "service": "hqca-data-pipeline",
        "status": "ok",
        "checks": {},
    }

    try:
        import data

        descriptor = data.MolecularDescriptors.compute("CCO")
        normalized = data.normalize_descriptors(descriptor.to_array())
        payload["checks"] = {
            "rdkit_available": data.RDKIT_AVAILABLE,
            "pennylane_available": data.PENNYLANE_AVAILABLE,
            "descriptor_count": int(len(normalized)),
        }
        if not data.RDKIT_AVAILABLE or len(normalized) != 7:
            payload["status"] = "unhealthy"
    except Exception as exc:
        payload["status"] = "unhealthy"
        payload["error"] = str(exc)

    print(json.dumps(payload))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
