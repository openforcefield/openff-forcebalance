from openff.forcebalance.tests.fitting_tests import fit, fitting_cd


@fitting_cd("optgeo")
def test_opgeo():
    fit()
