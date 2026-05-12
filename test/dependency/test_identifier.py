"""Tests of identifier classes."""

# Third party
import pytest
from pydantic import BaseModel

# Dax
from jadzia.core.dependency.identifier import Identifier, IdentifierStr, SpecialNodes


class ConfigCls(BaseModel):
    ident: IdentifierStr

    @property
    def output_names(self) -> tuple[str, ...]:
        return (self.ident.output,)


def test_string_conversion():
    """Identifiers may be converted from strings."""
    example = Identifier.from_str("foo.bar")
    assert example == Identifier("foo", "bar")
    assert example.node == "foo"
    assert example.output == "bar"

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


def test_special_nodes():
    """Identifier attributes are parsed correctly."""
    ident = Identifier("foo", "bar")
    assert ident.is_special is False
    assert ident.is_batch is False

    ident = Identifier(SpecialNodes.BATCH, "out")
    assert ident.is_special is True
    assert ident.is_batch is True


def test_identifier_embedded():
    """Identifiers inside configuration classes are converted to and from strings."""
    config = ConfigCls(ident="ermagherd.out")
    assert config.ident == Identifier("ermagherd", "out")

    new_idents = Identifier.from_outputs("blerk", config)
    assert new_idents == (Identifier("blerk", "out"),)

    with pytest.raises(ValueError, match=r"expected '.*' format"):
        ConfigCls(ident="berks")
