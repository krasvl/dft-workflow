"""S3-compatible object storage client (MinIO)."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from common.logging import setup_logging
from common.settings import Settings, get_settings

logger = setup_logging("storage")


class ObjectStorage:
    """Thin wrapper around boto3 S3 API for MinIO."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: BaseClient | None = None

    @property
    def client(self) -> BaseClient:
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url(),
                aws_access_key_id=self.settings.minio_access_key,
                aws_secret_access_key=self.settings.minio_secret_key,
                region_name=self.settings.minio_region,
            )
        return self._client

    def _endpoint_url(self) -> str:
        scheme = "https" if self.settings.minio_secure else "http"
        return f"{scheme}://{self.settings.minio_endpoint}"

    def ensure_bucket(self, bucket: str | None = None) -> None:
        """Create bucket if it does not exist."""
        name = bucket or self.settings.minio_bucket
        try:
            self.client.head_bucket(Bucket=name)
            logger.debug("bucket_exists bucket=%s", name)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in ("404", "NoSuchBucket", "NotFound"):
                raise
            logger.info("creating_bucket bucket=%s", name)
            self.client.create_bucket(Bucket=name)

    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        bucket: str | None = None,
    ) -> str:
        name = bucket or self.settings.minio_bucket
        self.ensure_bucket(name)
        self.client.put_object(
            Bucket=name,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        logger.info("put_object bucket=%s key=%s bytes=%d", name, key, len(data))
        return key

    def get_bytes(self, key: str, *, bucket: str | None = None) -> bytes:
        name = bucket or self.settings.minio_bucket
        response = self.client.get_object(Bucket=name, Key=key)
        body: bytes = response["Body"].read()
        logger.debug("get_object bucket=%s key=%s bytes=%d", name, key, len(body))
        return body

    def put_json(self, key: str, payload: dict[str, Any], *, bucket: str | None = None) -> str:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        return self.put_bytes(key, data, content_type="application/json", bucket=bucket)

    def get_json(self, key: str, *, bucket: str | None = None) -> dict[str, Any]:
        raw = self.get_bytes(key, bucket=bucket)
        result: dict[str, Any] = json.loads(raw.decode("utf-8"))
        return result

    def object_exists(self, key: str, *, bucket: str | None = None) -> bool:
        name = bucket or self.settings.minio_bucket
        try:
            self.client.head_object(Bucket=name, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list_keys(self, prefix: str, *, bucket: str | None = None) -> list[str]:
        """List object keys under a prefix (non-recursive pagination handled)."""
        name = bucket or self.settings.minio_bucket
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=name, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item.get("Key")
                if key:
                    keys.append(str(key))
        return keys


@lru_cache
def get_storage() -> ObjectStorage:
    return ObjectStorage()
