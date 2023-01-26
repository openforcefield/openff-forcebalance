import numpy
from openff.utilities import get_data_file_path

from openff.forcebalance.molecule.qc import read_qdata


def test_read_qdata_coords():
    qdata = get_data_file_path("tests/files/qdata0.txt", "openff.forcebalance")
    result = read_qdata(qdata)

    assert len(result) == 2
    assert "qm_energies" in result.keys()
    assert "xyzs" in result.keys()
    assert len(result["xyzs"]) == 24
    assert len(result["qm_energies"]) == 24

    assert result["xyzs"][-1].shape == (9, 3)
    assert isinstance(result["qm_energies"][-1], float)
