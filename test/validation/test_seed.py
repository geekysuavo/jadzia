"""Tests of random seed validation."""

# Third party
import pytest

# Dax
from jadzia.core.validation.base import Config
from jadzia.core.validation.seed import RandomSeed


class Example(Config):
    foo: RandomSeed


def test_default_factory():
    """Random seeds are imputed when default values are not set."""
    config = Example()
    assert isinstance(config.foo, int)
    assert config.foo > 0


def test_with_value():
    """Random seeds respect the value provided from configuration data."""
    config = Example(foo=57)
    assert config.foo == 57


def test_invalid_values():
    """Random seeds must be positive integers."""
    with pytest.raises(ValueError, match="should be a valid integer"):
        Example(foo=1.875)

    with pytest.raises(ValueError, match="should be greater than 0"):
        Example(foo=0)
