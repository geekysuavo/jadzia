"""Tests of validated module classes."""

# Standard
from typing import NamedTuple, TypedDict

# Third party
import equinox as eqx
import jax
import pytest
from pydantic import PositiveInt

# Dax
from jadzia.core.validation.module import ModuleConfig


class InputTypes(eqx.Module):
    weight: jax.Array

    def __init__(self, n: PositiveInt, key: jax.Array):
        self.weight = jax.random.normal(key, shape=(n,))

    def __call__(self, x: jax.Array, /, y: jax.Array, *, z: jax.Array) -> jax.Array:
        return self.weight * x * (y + z)


class SingleOutput(eqx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.numpy.abs(x)


class DictOutput(eqx.Module):
    class ReturnType(TypedDict):
        hello: jax.Array
        world: jax.Array

    def __call__(self, x: jax.Array, y: jax.Array) -> ReturnType:
        return self.ReturnType(hello=x + y, world=x - y)


class TupleOutput(eqx.Module):
    class ReturnType(NamedTuple):
        foo: jax.Array
        bar: jax.Array
        state: eqx.nn.State

    def __call__(self, x: jax.Array, y: jax.Array, state: eqx.nn.State) -> ReturnType:
        return self.ReturnType(x + y, x - y, state)


def test_input_types():
    """Inspection of call methods produces the expected inputs."""
    config_cls = ModuleConfig.auto(InputTypes)

    assert config_cls.instance_type is InputTypes
    assert config_cls.instance_call.inputs_pos_only == ("x",)
    assert config_cls.instance_call.inputs_pos_or_kw == ("y",)
    assert config_cls.instance_call.inputs_kw_only == ("z",)
    assert config_cls.instance_call.outputs == ("output",)
    assert config_cls.instance_call.requires_state is False
    assert config_cls.requires_key is True

    config = config_cls(n=8)

    with pytest.raises(ValueError, match="requires a key"):
        config.build_instance()

    key = jax.random.key(42)
    module = config.build_instance(key)
    assert isinstance(module, InputTypes)

    x = jax.numpy.linspace(0, 1, 8)
    y = jax.numpy.linspace(0, 1, 8)
    z = jax.numpy.linspace(0, 1, 8)
    out = module(x, y, z=z)
    assert out.shape == (8,)


def test_single_output():
    """Inspection of single-output call methods produces the expected result."""
    config_cls = ModuleConfig.auto(SingleOutput)

    assert config_cls.instance_type is SingleOutput
    assert config_cls.instance_call.inputs_pos_only == ()
    assert config_cls.instance_call.inputs_pos_or_kw == ("x",)
    assert config_cls.instance_call.inputs_kw_only == ()
    assert config_cls.instance_call.outputs == ("output",)
    assert config_cls.instance_call.requires_state is False
    assert config_cls.requires_key is False

    config = config_cls()
    module = config.build_instance()
    assert isinstance(module, SingleOutput)

    key = jax.random.key(42)
    with pytest.raises(ValueError, match="does not require a key"):
        config.build_instance(key)

    x = jax.numpy.linspace(0, 1, 8)
    out = module(x)
    assert out.shape == (8,)


def test_dict_output():
    """Inspection of dict-output call methods produces the expected result."""
    config_cls = ModuleConfig.auto(DictOutput)

    assert config_cls.instance_type is DictOutput
    assert config_cls.instance_call.inputs_pos_only == ()
    assert config_cls.instance_call.inputs_pos_or_kw == ("x", "y")
    assert config_cls.instance_call.inputs_kw_only == ()
    assert config_cls.instance_call.outputs == ("hello", "world")
    assert config_cls.instance_call.requires_state is False
    assert config_cls.requires_key is False

    config = config_cls()
    module = config.build_instance()
    assert isinstance(module, DictOutput)

    x = jax.numpy.linspace(0, 1, 8)
    y = jax.numpy.linspace(1, 0, 8)
    out = module(x, y)
    assert isinstance(out, dict)
    assert out["hello"].shape == (8,)
    assert out["world"].shape == (8,)


def test_tuple_output():
    """Inspection of tuple-output call methods produces the expected result."""
    config_cls = ModuleConfig.auto(TupleOutput)
    assert config_cls.instance_type is TupleOutput
    assert config_cls.instance_call.inputs_pos_only == ()
    assert config_cls.instance_call.inputs_pos_or_kw == ("x", "y")
    assert config_cls.instance_call.inputs_kw_only == ()
    assert config_cls.instance_call.outputs == ("foo", "bar")
    assert config_cls.instance_call.requires_state is True
    assert config_cls.requires_key is False

    config = config_cls()
    module = config.build_instance()
    assert isinstance(module, TupleOutput)

    x = jax.numpy.linspace(0, 1, 8)
    y = jax.numpy.linspace(1, 0, 8)
    out = module(x, y, None)
    assert isinstance(out, tuple)
    assert out.foo.shape == (8,)
    assert out.bar.shape == (8,)
    assert out.state is None


@pytest.mark.parametrize("with_sharding", [False, True])
def test_build_with_state(with_sharding):
    """Modules may optionally built with initial state."""
    (device, *_) = jax.devices()
    sharding = jax.sharding.SingleDeviceSharding(device) if with_sharding else None

    config_cls = ModuleConfig.auto(eqx.nn.BatchNorm)
    config = config_cls(input_size=3, axis_name="batch", mode="batch")

    module = config.build_instance(sharding=sharding)
    assert isinstance(module, eqx.nn.BatchNorm)

    (module, state) = config.build_instance(sharding=sharding, with_state=True)
    assert isinstance(module, eqx.nn.BatchNorm)
    assert isinstance(state, eqx.nn.State)
