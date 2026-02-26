"""MinIO storage manager for MetaWeave."""

from __future__ import annotations

import io
from datetime import timedelta

from minio import Minio
from minio.error import S3Error

BUCKETS = ("raw-papers", "raw-texts", "extracted-structures")

DEFAULT_ENDPOINT = "localhost:9000"
DEFAULT_ACCESS_KEY = "minioadmin"
DEFAULT_SECRET_KEY = "minioadmin"


class StorageManager:
    """Thin wrapper around the MinIO Python client."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        access_key: str = DEFAULT_ACCESS_KEY,
        secret_key: str = DEFAULT_SECRET_KEY,
        secure: bool = False,
    ) -> None:
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
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

    def presigned_url(self, bucket: str, object_name: str, expires: timedelta = timedelta(hours=1)) -> str:
        """Return a pre-signed GET URL for the given object."""
        return self.client.presigned_get_object(bucket, object_name, expires=expires)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """Return a list of object names under *prefix* in *bucket*."""
        return [obj.object_name for obj in self.client.list_objects(bucket, prefix=prefix, recursive=True)]
