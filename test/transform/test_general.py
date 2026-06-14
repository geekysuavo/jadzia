"""Tests of general transformations."""

# Third party
import numpy as np
import pytest

# Dax
from jadzia.data.transform.general import ArraySize, Unary


@pytest.mark.parametrize(("axis", "size"), [(0, 2), (-1, 3), (None, 6)])
def test_array_size(axis, size):
    """Per-axis size or number of elements may be obtained from arrays."""
    transform = ArraySize(
        input_names=("x",),
        output_names=("y",),
        axis=axis,
    )
    x = np.random.default_rng().normal(size=(2, 3))
    assert transform(x) == size


@pytest.mark.parametrize("deg", [False, True])
def test_unary_numpy_angle(deg):
    """Unary functions may be applied to values, accepting keyword arguments."""
    transform = Unary(
        input_names=("x",),
        output_names=("y",),
        function=np.angle,
        args=(),
        kwargs={"deg": deg},
    )
    x = np.random.default_rng().normal(size=2)
    assert transform(x).shape == x.shape


@pytest.mark.parametrize("exponent", [2, 3.0])
def test_unary_numpy_pow(exponent):
    """Unary functions may be applied to values, accepting positional arguments."""
    transform = Unary(
        input_names=("x",),
        output_names=("y",),
        function=np.pow,
        args=(exponent,),
        kwargs={},
    )
    x = np.random.default_rng().normal(size=2)
    assert transform(x).shape == x.shape
