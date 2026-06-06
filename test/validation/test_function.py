"""Tests of validated function classes."""

# Standard
from collections.abc import Callable
from typing import ClassVar

# Third party
import numpy as np
import pytest
from pydantic import TypeAdapter

# Dax
from jadzia.core.validation.function import FunctionConfig, FunctionStr


def add(x: int, value: int) -> int:
    return x + value


class AddConfig(FunctionConfig):
    instance_type: ClassVar[Callable[..., int]] = add
    value: int = 1


@pytest.mark.parametrize("value", [1, 8, -3])
def test_build(value):
    """Functions may be bound to their parameters by configuration classes."""
    config = AddConfig(value=value)
    func = config.build_instance()
    assert callable(func)
    assert all(func(x) == x + value for x in range(-10, 11))


@pytest.mark.parametrize("name", ["jax.nn.elu", "numpy.cos", "np.fft.fft", "math.exp"])
def test_function_str(name):
    """Functions may be resolved by name during validation."""
    adapter = TypeAdapter(FunctionStr)
    func = adapter.validate_python(name)
    assert callable(func)

    assert isinstance(adapter.dump_python(func), str)
    assert adapter.validate_python(adapter.dump_python(func)) is func

    if name.startswith("math"):
        assert isinstance(func(3.14), float)
    else:
        x = np.random.default_rng().normal(size=(4,))
        y = func(x)
        assert y.shape == x.shape


def test_invalid_function_str():
    """Function name validation only supports math, jax, and numpy functions."""
    adapter = TypeAdapter(FunctionStr)

    with pytest.raises(TypeError, match="expected str"):
        adapter.validate_python(5)

    with pytest.raises(ValueError, match="not supported"):
        adapter.validate_python("random.uniform")

    with pytest.raises(ValueError, match="not found"):
        adapter.validate_python("math.awesome.serendipity")

    with pytest.raises(ValueError, match="not callable"):
        adapter.validate_python("np.nan")

    with pytest.raises(ValueError, match="not supported"):
        adapter.dump_python(pytest.param)
