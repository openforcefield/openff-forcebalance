def test_importable():
    from openff.forcebalance import __version__, __file__

    assert __version__ is not None
    assert __file__ is not None
