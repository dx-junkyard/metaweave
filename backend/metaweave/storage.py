"""MinIO storage manager for MetaWeave."""

from __future__ import annotations

import io
import os
from datetime import timedelta

from minio import Minio

BUCKETS = ("raw-papers", "raw-texts", "extracted-structures")


class StorageManager:
    """Thin wrapper around the MinIO Python client.

    Environment variables
    ---------------------
    MINIO_ENDPOINT   Endpoint for all MinIO operations (default: localhost:9000).
                     With ``extra_hosts: localhost:host-gateway`` in docker-compose,
                     this resolves to the Docker host inside the container, which is
                     the same address the browser uses — so pre-signed URL signatures
                     are always valid without any hostname rewriting.
    MINIO_ACCESS_KEY MinIO access key (default: minioadmin).
    MINIO_SECRET_KEY MinIO secret key (default: minioadmin).
    """

    def __init__(self) -> None:
        endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")

        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
        self._ensure_buckets()

    # ------------------------------------------------------------------
    # Bucket helpers
    # ------------------------------------------------------------------

    def _ensure_buckets(self) -> None:
        """Create the required buckets if they do not exist."""
        for bucket in BUCKETS:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_pdf(self, bucket: str, object_name: str, data: bytes) -> str:
        """Upload a PDF (as *bytes*) to MinIO and return the object name."""
        self.client.put_object(
            bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type="application/pdf",
        )
        return object_name

    # ------------------------------------------------------------------
    # Pre-signed URL
    # ------------------------------------------------------------------

    def presigned_url(
        self,
        bucket: str,
        object_name: str,
        expires: timedelta = timedelta(hours=1),
    ) -> str:
        """Return a pre-signed GET URL.

        Because the client is initialised with ``localhost:9000`` (the same
        host the browser uses), the generated URL is immediately valid in the
        browser — no hostname rewriting required.
        """
        return self.client.presigned_get_object(bucket, object_name, expires=expires)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """Return a list of object names under *prefix* in *bucket*."""
        return [
            obj.object_name
            for obj in self.client.list_objects(bucket, prefix=prefix, recursive=True)
        ]
