"""Tests of identifier classes."""

# Third party
import pytest
from pydantic import BaseModel

# Dax
from jadzia.core.dependency.identifier import Identifier, IdentifierStr, Reserved


class ConfigCls(BaseModel):
    ident: IdentifierStr

    @property
    def output_names(self) -> tuple[str, ...]:
        return (self.ident.output,)


@pytest.mark.parametrize("output", ["foo", "bar"])
@pytest.mark.parametrize("reserved", Reserved)
def test_reserved(reserved, output):
    """Identifiers may be built from reserved node names."""
    example = reserved(output)
    assert example == Identifier.from_str(f"{reserved}.{output}")


@pytest.mark.parametrize("node", ["hello", "world"])
def test_is_gather(node):
    """Identifiers may be configured as "gather" nodes."""
    example = Identifier.gather(node)
    assert example == Identifier(node, Reserved.ALL)
    assert example.is_gather is True

    example = Identifier(node, "other")
    assert example.is_gather is False


def test_string_conversion():
    """Identifiers may be converted from strings."""
    example = Identifier.from_str("foo.bar")
    assert example == Identifier("foo", "bar")
    assert example.node == "foo"
    assert example.output == "bar"

    assert Identifier.from_str(example) is example

    with pytest.raises(TypeError, match="expected str"):
        Identifier.from_str(5)

    with pytest.raises(ValueError, match=r"expected '.*' format"):
        Identifier.from_str("too_few_parts")

    with pytest.raises(ValueError, match=r"expected '.*' format"):
        Identifier.from_str("too.many.parts")


def test_conversion_to_string():
    """Identifiers may be converted to strings."""
    ident = Identifier("hello", "world")
    assert str(ident) == "hello.world"


def test_identifier_annotation():
    """Identifiers inside configuration classes are converted to and from strings."""
    config = ConfigCls(ident="ermagherd.out")
    assert config.ident == Identifier("ermagherd", "out")

    new_idents = Identifier.from_outputs("blerk", config)
    assert new_idents == (Identifier("blerk", "out"),)

    with pytest.raises(ValueError, match=r"expected '.*' format"):
        ConfigCls(ident="berks")
