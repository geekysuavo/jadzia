"""Tests of activation function configurations."""

# Third party
import jax
import jax.numpy as jnp
import pytest
from pydantic import TypeAdapter

# Dax
from jadzia.config.activation import ActivationStr

config_adapter = TypeAdapter(ActivationStr)
config_data = [
    "ReLU",
    "LeakyReLU",
    {"type": "ReLU"},
    {"type": "LeakyReLU"},
    {"type": "LeakyReLU", "negative_slope": 0.5},
]


@pytest.fixture(scope="session")
def inputs():
    return jnp.linspace(0, 1, 10)


@pytest.mark.parametrize("data", config_data, ids=range(len(config_data)))
def test_activation_configs(data, inputs):
    """Activation function configurations validate and build callables."""
    config = config_adapter.validate_python(data)
    func = config.build_instance()
    assert callable(func)

    outputs = jax.jit(func)(inputs)
    assert jnp.isfinite(outputs).all()
