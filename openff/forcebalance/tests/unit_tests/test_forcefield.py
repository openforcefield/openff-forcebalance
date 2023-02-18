import pytest

from openff.forcebalance.forcefield import determine_fftype


def test_determine_fftype():
    assert determine_fftype("openff_unconstrained-1.0.0.offxml") == "smirnoff"
    assert determine_fftype("openff_2.0.0.offxml") == "smirnoff"


def test_determine_fftype_unsupported():
    with pytest.raises(Exception, match="Only SMIRNOFF"):
        determine_fftype("ff.forcefield")
