"""Tests of fusion modules."""

# Third party
import jax
import jax.numpy as jnp
import pytest

# Dax
from jadzia.nn.fusion import MaxFusion, MeanFusion, SumFusion


@pytest.fixture
def input_arrays():
    return (
        jnp.linspace(0, 1, 10).reshape(2, 5),
        jnp.linspace(1, 2, 10).reshape(2, 5),
        jnp.linspace(2, 3, 10).reshape(2, 5),
    )


def test_max_fusion(input_arrays):
    """Max fusion modules apply a max-reduction."""
    module = MaxFusion()
    y = module(*input_arrays)
    assert isinstance(y, jax.Array)
    assert y.shape == input_arrays[0].shape
    assert (y >= 2).all()


def test_mean_fusion(input_arrays):
    """Mean fusion modules apply a mean-reduction."""
    module = MeanFusion()
    y = module(*input_arrays)
    assert isinstance(y, jax.Array)
    assert y.shape == input_arrays[0].shape


def test_sum_fusion(input_arrays):
    """Sum fusion modules apply sum-reduction."""
    module = SumFusion()
    y = module(*input_arrays)
    assert isinstance(y, jax.Array)
    assert y.shape == input_arrays[0].shape
