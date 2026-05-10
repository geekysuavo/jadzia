"""Tests of fission modules."""

# Third party
import jax
import jax.numpy as jnp
import pytest

# Dax
from jadzia.nn.fission import SplitFission, UnstackFission


@pytest.fixture
def input_array():
    return jnp.linspace(0, 1, 30).reshape(2, 3, 5)


@pytest.mark.parametrize(
    ("axis", "sizes"),
    [(0, (1, 1)), (1, (1, 2)), (1, (2, 1)), (2, (1, 4)), (-1, (3, 2))],
)
def test_split_fission(input_array, axis, sizes):
    """Split fission modules return the expected shapes."""
    module = SplitFission(axis=axis, sizes=sizes)

    ys = module(input_array)

    assert isinstance(ys, tuple)
    assert len(ys) == len(sizes)
    assert all(isinstance(y, jax.Array) for y in ys)

    for y, size in zip(ys, sizes, strict=True):
        expected_shape = list(input_array.shape)
        expected_shape[axis] = size
        assert y.shape == tuple(expected_shape)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_unbind_fission(input_array, axis):
    """Unstack fission modules return the expected shapes."""
    module = UnstackFission(axis=axis)

    ys = module(input_array)

    assert isinstance(ys, tuple)
    assert len(ys) == input_array.shape[axis]
    assert all(isinstance(y, jax.Array) for y in ys)

    expected_shape = list(input_array.shape)
    expected_shape.pop(axis)
    assert all(y.shape == tuple(expected_shape) for y in ys)
