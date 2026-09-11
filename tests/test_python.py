import ai5d


def test_import():
    assert hasattr(ai5d, "Brain")
    assert hasattr(ai5d, "__version__")


def test_version():
    assert isinstance(ai5d.__version__, str)
    assert ai5d.__version__


def test_brain_creation():
    brain = ai5d.Brain()
    assert brain is not None


def test_brain_pipeline():
    brain = ai5d.Brain()

    result = (
        brain
        .in_1d([1.0, 2.0, 3.0, 4.0])
        .to_2d()
        .talk_3d()
        .filter_4d()
        .get_5d()
        .result()
    )

    assert isinstance(result, list)
    assert len(result) == 4


def test_brain_reset():
    brain = ai5d.Brain()

    brain.in_1d([1.0, 2.0, 3.0])
    brain.reset()

    assert brain.result() == []