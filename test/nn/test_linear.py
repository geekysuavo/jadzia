"""Tests of linear modules."""

# Third party
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

# Dax
from jadzia.nn.linear import Linear


@pytest.mark.parametrize("bias", [True, False])
def test_linear(bias):
    """Linear modules work as expected."""
    (key, x_key) = jr.split(jr.key(42))
    module = Linear(8, 4, bias=bias, key=key)
    x = jr.normal(x_key, shape=(10, 8))
    y = module(x)
    assert isinstance(y, jax.Array)
    assert jnp.isfinite(y).all()
    assert y.shape == (10, 4)
