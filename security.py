"""Authentication, RBAC, and encryption helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Iterable

import jwt
from cryptography.fernet import Fernet


SECRET_KEY = os.getenv("HQCA_SECRET_KEY", "change-me-in-production")
JWT_ALGORITHM = "HS256"
TOKEN_TTL_MINUTES = int(os.getenv("HQCA_TOKEN_TTL_MINUTES", "120"))


def _derive_fernet_key(secret: str) -> bytes:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


fernet = Fernet(os.getenv("HQCA_ENCRYPTION_KEY", "").encode() or _derive_fernet_key(SECRET_KEY))


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return f"pbkdf2_sha256${salt}${base64.b64encode(digest).decode()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, salt, stored = password_hash.split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return hmac.compare_digest(base64.b64encode(digest).decode(), stored)


def create_access_token(username: str, role: str) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_TTL_MINUTES)
    payload = {"sub": username, "role": role, "exp": expires_at}
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])


def encrypt_sensitive(value: str) -> str:
    return fernet.encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_sensitive(value: str) -> str:
    return fernet.decrypt(value.encode("utf-8")).decode("utf-8")


def require_role(actual_role: str, allowed_roles: Iterable[str]) -> None:
    if actual_role not in set(allowed_roles):
        raise PermissionError("Insufficient role for this operation.")
