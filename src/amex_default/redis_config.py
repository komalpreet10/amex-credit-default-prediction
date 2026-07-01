from __future__ import annotations

import os
from pathlib import Path

from gcp.config import REDIS_SSL_CA_CERT_CONTENT, REDIS_SSL_CA_CERTS

DEFAULT_CA_CERT_PATH = "/tmp/redis-ca.pem"


def redis_ssl_ca_certs() -> str | None:
    if REDIS_SSL_CA_CERTS:
        return REDIS_SSL_CA_CERTS

    if not REDIS_SSL_CA_CERT_CONTENT:
        return None

    cert_path = Path(os.environ.get("REDIS_SSL_CA_CERT_PATH", DEFAULT_CA_CERT_PATH))
    cert_path.write_text(REDIS_SSL_CA_CERT_CONTENT, encoding="utf-8")
    return str(cert_path)
