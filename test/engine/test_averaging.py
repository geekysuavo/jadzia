"""Tests of averaging."""

# Standard
import random

# Third party
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

# Dax
from jadzia.engine.averaging import Average


@pytest.mark.parametrize("num", [10, 100])
def test_float(num):
    """Simple scalars may be averaged."""
    values = [random.normalvariate() for _ in range(num)]
    expected = sum(values) / num

    (first, *rest) = values
    average = Average[float](first)
    for value in rest:
        average = average.update(value)

    assert average.value == pytest.approx(expected)
    assert average.num == num


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("num", [10, 100])
def test_numpy_array(num, shape):
    """Numpy arrays may be averaged."""
    rng = np.random.default_rng()
    values = [rng.normal(size=shape) for _ in range(num)]
    expected = np.mean(np.stack(values, axis=0), axis=0)

    (first, *rest) = values
    average = Average[np.ndarray](first)
    for value in rest:
        average = average.update(value)

    np.testing.assert_allclose(average.value, expected)
    assert average.num == num


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("num", [10, 100])
def test_jax_array(num, shape):
    """Jax arrays may be averaged."""
    keys = jr.split(jr.key(42), num)
    values = [jr.normal(key, shape=shape) for key in keys]
    expected = jnp.mean(jnp.stack(values, axis=0), axis=0)

    (first, *rest) = values
    average = Average[jax.Array](first)
    for value in rest:
        average = average.update(value)

    assert jnp.allclose(average.value, expected)
    assert average.num == num


@pytest.mark.parametrize("shape", [(4,), (2, 3)])
@pytest.mark.parametrize("num", [10, 100])
def test_pytree(num, shape):
    """Arbitrary pytrees may be averaged."""
    rng = np.random.default_rng()
    keys = jr.split(jr.key(42), num)

    values = [
        {
            "flt": random.normalvariate(),
            "arr": (rng.normal(size=shape), jr.normal(key, shape=shape)),
        }
        for key in keys
    ]

    expected = {
        "flt": sum(v["flt"] for v in values) / num,
        "arr": (
            np.mean(np.stack([v["arr"][0] for v in values], axis=0), axis=0),
            jnp.mean(jnp.stack([v["arr"][1] for v in values], axis=0), axis=0),
        ),
    }

    (first, *rest) = values
    average = Average(first)
    for value in rest:
        average = average.update(value)

    assert jax.tree.map(jnp.allclose, average.value, expected)
    assert average.num == num
