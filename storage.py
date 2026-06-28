"""Object storage: MinIO or local filesystem fallback."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import quote

try:
    from minio import Minio

    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False


class ObjectStorage:
    def __init__(self) -> None:
        self.endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "hqca")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "hqca-secret")
        self.bucket = os.getenv("MINIO_BUCKET", "hqca")
        self.secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        self.local_root = Path(os.getenv("HQCA_LOCAL_STORAGE", "output/storage"))
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.use_minio = MINIO_AVAILABLE and os.getenv("HQCA_USE_MINIO", "false").lower() == "true"
        self._client: Optional[Minio] = None
        if self.use_minio:
            self._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
            if not self._client.bucket_exists(self.bucket):
                self._client.make_bucket(self.bucket)

    def put_file(self, local_path: str, object_name: str) -> str:
        src = Path(local_path)
        if self.use_minio and self._client:
            self._client.fput_object(self.bucket, object_name, str(src))
            return f"/files/{quote(object_name)}"
        dest = self.local_root / object_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return f"/files/{quote(object_name.replace(chr(92), '/'))}"

    def resolve_local(self, object_name: str) -> Path:
        if self.use_minio and self._client:
            target = self.local_root / object_name
            target.parent.mkdir(parents=True, exist_ok=True)
            self._client.fget_object(self.bucket, object_name, str(target))
            return target
        return self.local_root / object_name


object_storage = ObjectStorage()
