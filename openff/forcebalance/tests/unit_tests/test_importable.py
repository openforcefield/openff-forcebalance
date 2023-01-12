def test_importable():
    from openff.forcebalance import __file__, __version__

    assert __version__ is not None
    assert __file__ is not None
