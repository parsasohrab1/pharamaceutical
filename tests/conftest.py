import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("HQCA_DATABASE_URL", "sqlite:///output/test_api.db")
os.environ.setdefault("HQCA_USE_MINIO", "false")

from api import app  # noqa: E402
from database import init_db  # noqa: E402


@pytest.fixture()
def client():
    init_db()
    with TestClient(app) as c:
        yield c
