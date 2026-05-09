"""Tests of validated loss function classes."""

# Standard
from typing import NamedTuple

# Third party
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

# Dax
from jadzia.core.validation.loss import GeneralLoss, SimpleLoss


class General(eqx.Module):
    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        /,
        z: jax.Array,
        w: jax.Array,
    ) -> jax.Array:
        return (x - z)**2 + jnp.abs(y - w)


class Simple(eqx.Module):
    def __call__(self, x: jax.Array, /, y: jax.Array) -> jax.Array:
        return (x - y)**2


class InvalidOutput(eqx.Module):
    class ReturnType(NamedTuple):
        ell2: jax.Array
        ell1: jax.Array

    def forward(self, x: jax.Array, /, y: jax.Array) -> ReturnType:
        return self.ReturnType(x + y, x - y)


def test_general_loss():
    """Signature discovery of general losses produces the expected results."""
    config_cls = GeneralLoss.auto(General)
    assert issubclass(config_cls, GeneralLoss)

    config = config_cls(inputs={"x": "X", "y": "Y", "z": "Z", "w": "W"})
    assert config.prediction_names == ("X", "Y")
    assert config.reference_names == ("Z", "W")

    obj = config.build_instance()
    assert isinstance(obj, General)

    with pytest.raises(ValueError, match="Field required"):
        config_cls()

    with pytest.raises(ValueError, match="invalid module call inputs"):
        config_cls(inputs={})

    with pytest.raises(ValueError, match="invalid module call inputs"):
        config_cls(inputs={"x": "X", "y": "Y", "z": "Z", "w": "W", "extra": "E"})


def test_simple_loss():
    """Signature discovery of simple losses produces the expected results."""
    config_cls = SimpleLoss.auto(Simple)
    assert issubclass(config_cls, SimpleLoss)

    config = config_cls(input="foobar")
    assert config.prediction_names == ("foobar",)
    assert config.reference_names == ("foobar",)

    obj = config.build_instance()
    assert isinstance(obj, Simple)

    with pytest.raises(ValueError, match="Field required"):
        config_cls()


def test_incompatible_signatures_fail():
    """Signatures are checked for compatibility with loss config classes."""
    with pytest.raises(ValueError, match="expected one module call output"):
        GeneralLoss.auto(InvalidOutput)(inputs={})

    with pytest.raises(ValueError, match="expected one module call output"):
        SimpleLoss.auto(InvalidOutput)(input="foo")

    with pytest.raises(ValueError, match="expected one module call input pair"):
        SimpleLoss.auto(General)(input="bar")
