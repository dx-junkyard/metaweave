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
    MINIO_ENDPOINT        Internal endpoint used for API calls (default: localhost:9000).
    MINIO_ACCESS_KEY      MinIO access key (default: minioadmin).
    MINIO_SECRET_KEY      MinIO secret key (default: minioadmin).
    MINIO_PUBLIC_ENDPOINT Public-facing endpoint injected into pre-signed URLs so that
                          browsers (outside Docker) can resolve the URL.
                          Example: localhost:9000
                          If unset, the internal endpoint is used as-is.
    """

    def __init__(self) -> None:
        endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        self._public_endpoint: str = os.environ.get("MINIO_PUBLIC_ENDPOINT", endpoint)

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
    # Pre-signed URL (with public-endpoint rewrite)
    # ------------------------------------------------------------------

    def presigned_url(
        self,
        bucket: str,
        object_name: str,
        expires: timedelta = timedelta(hours=1),
    ) -> str:
        """Return a pre-signed GET URL, rewriting the host to the public endpoint.

        MinIO generates URLs whose host matches the endpoint used to initialise
        the client (e.g. ``minio:9000``).  Browsers running on the developer's
        machine cannot resolve that hostname.  We therefore string-replace the
        internal host with the value of ``MINIO_PUBLIC_ENDPOINT``.
        """
        url: str = self.client.presigned_get_object(bucket, object_name, expires=expires)
        # Replace internal hostname with the public one if they differ.
        internal = self.client._base_url._url.netloc  # type: ignore[attr-defined]
        if internal != self._public_endpoint:
            url = url.replace(f"//{internal}/", f"//{self._public_endpoint}/", 1)
        return url

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """Return a list of object names under *prefix* in *bucket*."""
        return [
            obj.object_name
            for obj in self.client.list_objects(bucket, prefix=prefix, recursive=True)
        ]
