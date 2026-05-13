"""Tests of validated function classes."""

# Standard
from collections.abc import Callable
from typing import ClassVar

# Third party
import pytest

# Dax
from jadzia.core.validation.function import FunctionConfig


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
