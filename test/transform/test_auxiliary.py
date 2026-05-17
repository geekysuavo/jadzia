"""Tests of auxiliary transformations."""

# Third party
import numpy as np
import pytest

# Dax
from jadzia.data.transform.auxiliary import (
    Insert,
    InsertCopy,
    InsertFromSequence,
    InsertRandomChoice,
    InsertRandomInteger,
    InsertRandomUniform,
)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng()


@pytest.mark.parametrize("value", [42, 3.5, True, "hello"])
def test_insert(value):
    """Fixed values may be inserted into examples."""
    transform = Insert(input_names=(), output_names=(), value=value)
    assert transform() == value


def test_insert_copy():
    """Values may be copied within examples."""
    transform = InsertCopy(input_names=(), output_names=())

    for value in (55, 7.3, "x", False):
        assert transform(value) == value

    value = [2, 3, 5]
    output = transform(value)
    assert output == value
    assert output is not value

    value = np.arange(10)
    output = transform(value)
    assert np.array_equal(output, value)
    assert output is not value


def test_insert_from_seq():
    """Values may be inserted into examples in sequence."""
    values = (2, 3, 5.5, "hello", True)
    transform = InsertFromSequence(input_names=(), output_names=(), values=values)

    for index in range(len(values)):
        assert transform(index=index) == values[index]

    for index in range(len(values)):
        assert transform(index=index + len(values)) == values[index]


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("value_type", ["bool", "float", "int", "str"])
def test_insert_rand_choice(rng, value_type, weighted):
    """Values may be inserted by sampling from a discrete set."""
    values = {
        "bool": (True, True, False),
        "float": (1.2, 3.4, 5.6),
        "int": (12, 34, 56),
        "str": ("x", "y", "z"),
    }
    transform = InsertRandomChoice(
        input_names=(),
        output_names=(),
        values=values[value_type],
        weights=(0.25, 0.25, 0.5) if weighted else None,
    )
    assert transform(rng=rng) in values[value_type]


def test_insert_rand_int(rng):
    """Values may be inserted by sampling from an integer range."""
    transform = InsertRandomInteger(input_names=(), output_names=())
    assert isinstance(transform(rng=rng), int)


def test_insert_rand_unif(rng):
    """Values may be inserted by sampling from a continuous range."""
    transform = InsertRandomUniform(input_names=(), output_names=())
    assert isinstance(transform(rng=rng), float)
