"""Tests of the validation mapping class."""

# Third party
import pytest

# Dax
from jadzia.core.validation.base import Config
from jadzia.core.validation.mapping import MappingConfig


class Example(Config):
    foo: int = 2
    bar: float = 3.5


class Examples(MappingConfig[Example]): ...


def test_mapping():
    """Configuration mappings provide the expected interface and behavior."""
    config_data = {"xy": {}, "ab": {"foo": 7}}
    config = Examples.model_validate(config_data)
    assert set(config) == {"xy", "ab"}
    assert len(config) == 2

    assert set(config.keys()) == {"xy", "ab"}
    assert set(config.values()) == {config["xy"], config["ab"]}
    assert set(config.items()) == {("xy", config["xy"]), ("ab", config["ab"])}

    assert "xy" in config
    assert "ab" in config
    assert "zz" not in config

    with pytest.raises(KeyError, match="zz"):
        config["zz"]

    with pytest.raises(ValueError, match="should be a valid integer"):
        Examples.model_validate({"oops": {"foo": 3.3}})
