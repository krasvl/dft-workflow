"""Unit tests for ObjectStorage with mocked boto3 client."""

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from common.storage import ObjectStorage


@pytest.fixture
def storage() -> ObjectStorage:
    s = ObjectStorage()
    s._client = MagicMock()
    return s


def test_put_and_get_bytes(storage: ObjectStorage) -> None:
    mock_body = MagicMock()
    mock_body.read.return_value = b"hello"
    storage.client.get_object.return_value = {"Body": mock_body}

    storage.put_bytes("test/key.bin", b"hello", content_type="text/plain")
    data = storage.get_bytes("test/key.bin")

    assert data == b"hello"
    storage.client.put_object.assert_called_once()
    storage.client.get_object.assert_called_once()


def test_put_and_get_json(storage: ObjectStorage) -> None:
    captured: dict = {}

    def put_object(**kwargs):  # type: ignore[no-untyped-def]
        captured["body"] = kwargs["Body"]

    mock_body = MagicMock()
    mock_body.read.return_value = b'{"a": 1}'
    storage.client.put_object.side_effect = put_object
    storage.client.get_object.return_value = {"Body": mock_body}

    storage.put_json("test/data.json", {"a": 1})
    result = storage.get_json("test/data.json")

    assert result == {"a": 1}
    assert b'"a": 1' in captured["body"]


def test_object_exists_true_and_false(storage: ObjectStorage) -> None:
    def head_object(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["Key"] == "missing":
            raise ClientError(
                {"Error": {"Code": "404"}},
                "HeadObject",
            )

    storage.client.head_object.side_effect = head_object

    assert storage.object_exists("present")
    assert not storage.object_exists("missing")


def test_ensure_bucket_creates_when_missing(storage: ObjectStorage) -> None:
    def head_bucket(**kwargs):  # type: ignore[no-untyped-def]
        raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")

    storage.client.head_bucket.side_effect = head_bucket

    storage.ensure_bucket("dft-workflow")

    storage.client.create_bucket.assert_called_once_with(Bucket="dft-workflow")
