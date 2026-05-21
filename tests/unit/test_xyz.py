import pytest

from services.data_service.app.xyz import XyzValidationError, validate_xyz

WATER = b"""3
Water
O 0.0 0.0 0.0
H 0.0 0.7 0.0
H 0.0 -0.7 0.0
"""


def test_valid_water_xyz() -> None:
    validate_xyz(WATER, "water.xyz")


def test_rejects_empty() -> None:
    with pytest.raises(XyzValidationError):
        validate_xyz(b"", "water.xyz")


def test_rejects_bad_extension() -> None:
    with pytest.raises(XyzValidationError):
        validate_xyz(WATER, "water.txt")


def test_rejects_missing_coordinates() -> None:
    bad = b"3\ncomment\nO 0 0 0\n"
    with pytest.raises(XyzValidationError):
        validate_xyz(bad, "bad.xyz")
