"""Tests of the validation mapping class."""

# Third party
import pytest

# Dax
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier, IdentifierStr
from jadzia.core.validation.base import Config
from jadzia.core.validation.mapping import NodeMappingConfig


class Example(Config):
    foo: int = 2
    bar: float = 3.5
    output_names: tuple[str, ...] = ()
    input_names: tuple[IdentifierStr, ...] = ()


class Examples(NodeMappingConfig[Example]): ...


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

    with pytest.raises(ValueError, match="name is reserved"):
        Examples.model_validate({"loss": {}})

    with pytest.raises(ValueError, match="should be a valid integer"):
        Examples.model_validate({"oops": {"foo": 3.3}})


def test_node_mapping():
    """Configuration mappings may also contain graph nodes."""
    config_data = {
        "foo": {"output_names": ("x", "y")},
        "baz": {"input_names": ("foo.y", "bar.z")},
        "bar": {"input_names": ("foo.x",), "output_names": ("z",)},
    }
    config = Examples.model_validate(config_data)
    graph = config.build_subgraph()

    expected = {
        "foo.*": (),
        "foo.x": ("foo.*",),
        "foo.y": ("foo.*",),
        "bar.*": ("foo.x",),
        "bar.z": ("bar.*",),
        "baz.*": ("foo.y", "bar.z"),
    }
    expected = {
        Identifier.from_str(key): set(map(Identifier.from_str, values))
        for key, values in expected.items()
    }
    assert graph == Graph(expected)

    assert config.node_order() == ("foo", "bar", "baz")

    subset = (Identifier.from_str("bar.*"),)
    assert config.node_order(subset) == ("foo", "bar")
