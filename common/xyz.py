"""Validation and small helpers for XYZ molecular structure files."""

# Reusable water molecule for tests and examples.
WATER = b"""3
Water
O 0.0 0.0 0.0
H 0.0 0.7 0.0
H 0.0 -0.7 0.0
"""


class XyzValidationError(ValueError):
    pass


def parse_atom_count(content: bytes) -> int:
    lines = content.decode("utf-8").strip().splitlines()
    if not lines:
        raise XyzValidationError("Empty XYZ file")
    return int(lines[0].split()[0])


def validate_xyz(content: bytes, filename: str | None = None) -> None:
    if filename and not filename.lower().endswith(".xyz"):
        raise XyzValidationError("File must have .xyz extension")

    if not content or not content.strip():
        raise XyzValidationError("XYZ file is empty")

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise XyzValidationError("XYZ file must be UTF-8 text") from exc

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 3:
        raise XyzValidationError("XYZ must contain atom count, comment, and coordinates")

    try:
        atom_count = int(lines[0].split()[0])
    except (ValueError, IndexError) as exc:
        raise XyzValidationError("First line must be integer atom count") from exc

    if atom_count < 1:
        raise XyzValidationError("Atom count must be positive")

    coord_lines = lines[2:]
    if len(coord_lines) < atom_count:
        raise XyzValidationError(
            f"Expected at least {atom_count} coordinate lines, got {len(coord_lines)}"
        )

    for i, line in enumerate(coord_lines[:atom_count], start=1):
        parts = line.split()
        if len(parts) < 4:
            raise XyzValidationError(f"Line {i + 2}: expected symbol and 3 coordinates")
        try:
            float(parts[1])
            float(parts[2])
            float(parts[3])
        except ValueError as exc:
            raise XyzValidationError(f"Line {i + 2}: invalid coordinates") from exc
