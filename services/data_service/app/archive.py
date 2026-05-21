"""Read .xyz entries out of a zip/tar archive (used by batch upload).

Hard limits guard against zip-bombs and pathologically large payloads:

* ``MAX_ENTRIES``        — max usable .xyz files per archive
* ``MAX_FILE_BYTES``     — uncompressed size cap per .xyz entry
* ``MAX_TOTAL_BYTES``    — uncompressed size cap across the whole archive

Directories, dotfiles, ``__MACOSX/*`` and non-``.xyz`` files are silently
skipped. Bad archive containers raise :class:`ArchiveError`.
"""

from __future__ import annotations

import io
import logging
import tarfile
import zipfile
from typing import Iterator

logger = logging.getLogger("data_service.archive")

MAX_ENTRIES = 1000
MAX_FILE_BYTES = 5_000_000          # 5 MB per .xyz (generous; real XYZ are tiny)
MAX_TOTAL_BYTES = 200_000_000       # 200 MB total uncompressed

ZIP_MAGIC = b"PK\x03\x04"
ZIP_EMPTY_MAGIC = b"PK\x05\x06"
ZIP_SPANNED_MAGIC = b"PK\x07\x08"
GZIP_MAGIC = b"\x1f\x8b"
BZ2_MAGIC = b"BZh"
XZ_MAGIC = b"\xfd7zXZ\x00"
TAR_USTAR_OFFSET = 257
TAR_USTAR_MAGIC = b"ustar"


class ArchiveError(ValueError):
    """Raised when the upload is not a usable archive."""


def _is_zip(head: bytes) -> bool:
    return (
        head.startswith(ZIP_MAGIC)
        or head.startswith(ZIP_EMPTY_MAGIC)
        or head.startswith(ZIP_SPANNED_MAGIC)
    )


def _is_tarball(head: bytes, content: bytes) -> bool:
    if head.startswith(GZIP_MAGIC) or head.startswith(BZ2_MAGIC) or head.startswith(XZ_MAGIC):
        return True
    end = TAR_USTAR_OFFSET + len(TAR_USTAR_MAGIC)
    if len(content) >= end and content[TAR_USTAR_OFFSET:end] == TAR_USTAR_MAGIC:
        return True
    return False


def _detect_kind(filename: str | None, content: bytes) -> str | None:
    name = (filename or "").lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz")):
        return "tar"
    head = content[:8]
    if _is_zip(head):
        return "zip"
    if _is_tarball(head, content):
        return "tar"
    return None


def _skip_entry(name: str) -> bool:
    parts = name.split("/")
    if "__MACOSX" in parts:
        return True
    base = parts[-1]
    if not base:
        return True  # directory entry
    if base.startswith(".") or base.startswith("._"):
        return True
    if not base.lower().endswith(".xyz"):
        return True
    return False


def _check_size(name: str, declared_size: int, total_so_far: int) -> None:
    if declared_size > MAX_FILE_BYTES:
        raise ArchiveError(
            f"{name}: entry too large ({declared_size} bytes > {MAX_FILE_BYTES} limit)"
        )
    if total_so_far > MAX_TOTAL_BYTES:
        raise ArchiveError(
            f"archive total uncompressed size exceeds {MAX_TOTAL_BYTES} bytes"
        )


def _iter_zip(content: bytes) -> Iterator[tuple[str, bytes]]:
    total = 0
    yielded = 0
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for info in zf.infolist():
            if info.is_dir() or _skip_entry(info.filename):
                continue
            total += info.file_size
            _check_size(info.filename, info.file_size, total)
            yielded += 1
            if yielded > MAX_ENTRIES:
                raise ArchiveError(
                    f"archive has more than {MAX_ENTRIES} usable .xyz entries"
                )
            with zf.open(info, "r") as fh:
                # Defensive: read at most MAX_FILE_BYTES+1 to catch malformed metadata.
                data = fh.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise ArchiveError(
                    f"{info.filename}: entry too large after decompression"
                )
            yield info.filename, data


def _iter_tar(content: bytes) -> Iterator[tuple[str, bytes]]:
    total = 0
    yielded = 0
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as tf:
        for member in tf:
            if not member.isfile() or _skip_entry(member.name):
                continue
            total += member.size
            _check_size(member.name, member.size, total)
            yielded += 1
            if yielded > MAX_ENTRIES:
                raise ArchiveError(
                    f"archive has more than {MAX_ENTRIES} usable .xyz entries"
                )
            fh = tf.extractfile(member)
            if fh is None:
                continue
            data = fh.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise ArchiveError(
                    f"{member.name}: entry too large after decompression"
                )
            yield member.name, data


def iter_archive_xyz(
    content: bytes, filename: str | None = None
) -> Iterator[tuple[str, bytes]]:
    """Yield ``(entry_name, xyz_bytes)`` for every ``.xyz`` in zip/tar(.gz|.bz2|.xz)."""
    if not content:
        raise ArchiveError("empty upload")
    kind = _detect_kind(filename, content)
    if kind is None:
        raise ArchiveError(
            f"unsupported archive format (filename={filename!r}); "
            "expected .zip or .tar(.gz|.bz2|.xz)"
        )
    iterator = _iter_zip if kind == "zip" else _iter_tar
    try:
        yield from iterator(content)
    except ArchiveError:
        raise
    except (zipfile.BadZipFile, tarfile.TarError) as exc:
        raise ArchiveError(f"failed to read {kind} archive: {exc}") from exc
