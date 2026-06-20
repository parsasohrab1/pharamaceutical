"""Object storage adapter with MinIO and local filesystem backends."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response


class ObjectStorage:
    def __init__(self) -> None:
        self.bucket = os.getenv("HQCA_MINIO_BUCKET", "hqca")
        self.local_root = Path(os.getenv("HQCA_OBJECT_STORE_DIR", "output/objects"))
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.backend = "local"
        self.client = None

        endpoint = os.getenv("HQCA_MINIO_ENDPOINT")
        access_key = os.getenv("HQCA_MINIO_ACCESS_KEY")
        secret_key = os.getenv("HQCA_MINIO_SECRET_KEY")
        if endpoint and access_key and secret_key:
            from minio import Minio

            secure = os.getenv("HQCA_MINIO_SECURE", "false").lower() == "true"
            self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
            self.backend = "minio"

    def put_bytes(self, key: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        if self.backend == "minio" and self.client is not None:
            from io import BytesIO

            self.client.put_object(
                self.bucket,
                key,
                BytesIO(content),
                length=len(content),
                content_type=content_type,
            )
            return key

        path = self.local_root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return key

    def put_file(self, key: str, file_path: str | Path, content_type: str = "application/octet-stream") -> str:
        return self.put_bytes(key, Path(file_path).read_bytes(), content_type=content_type)

    def response(self, key: str, media_type: str, filename: str) -> Response:
        if self.backend == "minio" and self.client is not None:
            try:
                obj = self.client.get_object(self.bucket, key)
                content = obj.read()
            except Exception as exc:  # pragma: no cover - depends on external MinIO
                raise HTTPException(status_code=404, detail="Object not found.") from exc
            return Response(
                content=content,
                media_type=media_type,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        path = self.local_root / key
        if not path.exists():
            raise HTTPException(status_code=404, detail="Object not found.")
        return FileResponse(path, media_type=media_type, filename=filename)


object_storage = ObjectStorage()
