"""Tests of tagged union functions."""

# Standard
from typing import Annotated

# Third party
import pytest
from pydantic import RootModel, Tag

# Dax
from jadzia.core.validation.base import Config
from jadzia.core.validation.union import UnionDiscriminator, UnionSerializer


class Foo(Config):
    hello: int = 42


class Bar(Config):
    hello: str = "world"


class Baz(Config):
    world: float = 3.14


type FooBar = Annotated[
    Annotated[Foo, Tag("Foo")] | Annotated[Bar, Tag("Bar")],
    UnionDiscriminator(),
    UnionSerializer(),
]


def test_tagged_unions():
    """Tagged unions are validated and serialized correctly."""
    config_cls = RootModel[dict[str, FooBar]]

    config_data = {"foo": {"type": "Foo"}, "bar": {"type": "Bar"}}
    config = config_cls.model_validate(config_data)
    assert config.root["foo"].hello == 42
    assert config.root["bar"].hello == "world"

    model_data = {"foo": config.root["foo"], "bar": config.root["bar"]}
    assert config_cls.model_validate(model_data) == config

    dumped_data = config.model_dump()
    assert dumped_data["foo"]["type"] == "Foo"
    assert dumped_data["bar"]["type"] == "Bar"

    invalid_type = {"zoo": 5}
    with pytest.raises(ValueError, match="Unable to extract tag using discriminator"):
        config_cls.model_validate(invalid_type)

    invalid_tag = {"baz": {"type": "Baz"}}
    with pytest.raises(ValueError, match="does not match any of the expected tags"):
        config_cls.model_validate(invalid_tag)
