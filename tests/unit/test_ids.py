from common.ids import (
    new_calculation_id,
    new_job_id,
    new_molecule_id,
    new_model_version,
)


def test_molecule_id_prefix() -> None:
    mid = new_molecule_id()
    assert mid.startswith("mol_")
    assert len(mid) > len("mol_")


def test_calculation_id_prefix() -> None:
    cid = new_calculation_id()
    assert cid.startswith("dft_")


def test_job_id_prefix() -> None:
    jid = new_job_id()
    assert jid.startswith("job_")


def test_model_version_prefix() -> None:
    version = new_model_version()
    assert version.startswith("v")
    assert version[1:].isdigit()
