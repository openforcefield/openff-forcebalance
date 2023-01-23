from openff.forcebalance.tests.fitting_tests import fitting_cd, fit


@fitting_cd("optgeo")
def test_opgeo():
    fit()