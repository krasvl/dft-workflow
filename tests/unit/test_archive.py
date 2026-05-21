"""Unit tests for archive ingestion in data-service."""

from __future__ import annotations

import io
import tarfile
import zipfile

import pytest

from common.xyz import WATER
from services.data_service.app.archive import (
    ArchiveError,
    MAX_ENTRIES,
    MAX_FILE_BYTES,
    iter_archive_xyz,
)

H2 = b"2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n"


def _make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _make_tar_gz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_zip_extracts_only_xyz() -> None:
    payload = _make_zip({
        "water.xyz": WATER,
        "nested/dir/h2.xyz": H2,
        "readme.txt": b"ignore me",
        "__MACOSX/water.xyz": b"junk",
        ".DS_Store": b"junk",
        "._water.xyz": b"junk",
    })
    items = list(iter_archive_xyz(payload, "mols.zip"))
    names = sorted(n for n, _ in items)
    assert names == ["nested/dir/h2.xyz", "water.xyz"]
    bodies = {n: b for n, b in items}
    assert bodies["water.xyz"] == WATER
    assert bodies["nested/dir/h2.xyz"] == H2


def test_tar_gz_works() -> None:
    payload = _make_tar_gz({"a.xyz": WATER, "b.xyz": H2, "skip.md": b"x"})
    items = list(iter_archive_xyz(payload, "mols.tar.gz"))
    assert sorted(n for n, _ in items) == ["a.xyz", "b.xyz"]


def test_detects_by_magic_when_filename_missing() -> None:
    payload = _make_zip({"x.xyz": WATER})
    items = list(iter_archive_xyz(payload, None))
    assert items[0][0] == "x.xyz"


def test_unsupported_format_raises() -> None:
    with pytest.raises(ArchiveError, match="unsupported archive format"):
        list(iter_archive_xyz(b"not an archive", "mols.txt"))


def test_empty_payload_raises() -> None:
    with pytest.raises(ArchiveError, match="empty"):
        list(iter_archive_xyz(b"", "mols.zip"))


def test_corrupted_zip_raises() -> None:
    with pytest.raises(ArchiveError, match="failed to read zip"):
        list(iter_archive_xyz(b"PK\x03\x04garbage", "mols.zip"))


def test_too_many_entries_raises() -> None:
    files = {f"m{i}.xyz": WATER for i in range(MAX_ENTRIES + 1)}
    payload = _make_zip(files)
    with pytest.raises(ArchiveError, match="more than"):
        list(iter_archive_xyz(payload, "many.zip"))


def test_oversized_entry_raises() -> None:
    big = b"H 0 0 0\n" * (MAX_FILE_BYTES // 8 + 16)
    payload = _make_zip({"huge.xyz": b"1\nbig\n" + big})
    with pytest.raises(ArchiveError, match="too large"):
        list(iter_archive_xyz(payload, "huge.zip"))


def test_archive_with_only_skipped_entries_yields_nothing() -> None:
    payload = _make_zip({"readme.md": b"x", "__MACOSX/foo": b"y"})
    assert list(iter_archive_xyz(payload, "junk.zip")) == []
