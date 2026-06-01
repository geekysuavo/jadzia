"""Tests of validated loss function classes."""

# Standard
from typing import NamedTuple

# Third party
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

# Dax
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.validation.loss import LossConfig


class General(eqx.Module):
    def __call__(
        self,
        x: jax.Array,
        y: jax.Array,
        /,
        z: jax.Array,
        w: jax.Array,
    ) -> jax.Array:
        return jnp.square(x - z) + jnp.abs(y - w)


class Simple(eqx.Module):
    def __call__(self, x: jax.Array, /, y: jax.Array) -> jax.Array:
        return jnp.square(x - y)


class Markerless(eqx.Module):
    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.square(x - y)


class InvalidOutput(eqx.Module):
    class ReturnType(NamedTuple):
        ell2: jax.Array
        ell1: jax.Array

    def __call__(self, x: jax.Array, /, y: jax.Array) -> ReturnType:
        return self.ReturnType(x + y, x - y)


def test_simple_loss():
    """Signature discovery of simple losses produces the expected results."""
    config_cls = LossConfig.auto(Simple)
    assert issubclass(config_cls, LossConfig)

    config = config_cls(inputs={"x": "foo.f", "y": "bar.b"})
    assert config.prediction_names == (Identifier("foo", "f"),)
    assert config.reference_names == (Identifier("bar", "b"),)

    obj = config.build_instance()
    assert isinstance(obj, Simple)

    with pytest.raises(ValueError, match="Field required"):
        config_cls()


def test_general_loss():
    """Signature discovery of general losses produces the expected results."""
    config_cls = LossConfig.auto(General)
    assert issubclass(config_cls, LossConfig)

    config = config_cls(inputs={"x": "X.x", "y": "Y.y", "z": "Z.z", "w": "W.w"})
    assert config.prediction_names == (Identifier("X", "x"), Identifier("Y", "y"))
    assert config.reference_names == (Identifier("Z", "z"), Identifier("W", "w"))

    obj = config.build_instance()
    assert isinstance(obj, General)

    with pytest.raises(ValueError, match="Field required"):
        config_cls()

    with pytest.raises(ValueError, match="invalid module call inputs"):
        config_cls(inputs={})

    with pytest.raises(ValueError, match="invalid module call inputs"):
        config_cls(inputs={"x": "X.x", "y": "Y.y", "z": "Z.z", "w": "W.w", "e": "E.e"})


def test_markerless_loss():
    """Signature discovery of markerless losses produces the expected results."""
    config_cls = LossConfig.auto(Markerless)
    assert issubclass(config_cls, LossConfig)

    config = config_cls(inputs={"x": "x.X", "y": "y.Y"})
    assert config.prediction_names == (Identifier("x", "X"), Identifier("y", "Y"))
    assert config.reference_names == ()

    obj = config.build_instance()
    assert isinstance(obj, Markerless)


def test_incompatible_signatures_fail():
    """Signatures are checked for compatibility with loss config classes."""
    with pytest.raises(ValueError, match="expected one module call output"):
        LossConfig.auto(InvalidOutput)(inputs={})
