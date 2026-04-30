"""Tests of the validation base class."""

# Standard
from enum import StrEnum

# Dax
from jadzia.core.validation.base import Config


def test_yaml_round_trip(tmp_path):
    """Configuration objects round-trip through yaml successfully."""

    class Foo(StrEnum):
        HELLO = "hello"
        WORLD = "world"

    class Example(Config):
        foo: Foo
        bar: int

    original = Example(foo=Foo.WORLD, bar=57)

    yaml_file = tmp_path / "example.yaml"
    assert not yaml_file.exists()

    original.to_yaml(yaml_file)
    assert yaml_file.exists()

    reloaded = Example.from_yaml(yaml_file)
    assert reloaded == original
