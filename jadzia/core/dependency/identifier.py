"""Identifier classes for dependency management."""

# Standard
from enum import StrEnum
from typing import Annotated, NamedTuple, Protocol, Self

# Third party
from pydantic import AfterValidator, BeforeValidator, PlainSerializer


class Reserved(StrEnum):
    """
    Enumeration of reserved node names.

    - `id`: Identifying data inserted by the data source.
    - `source`: Value provided by the data source manifest or archives.
    """

    ID = "id"
    SOURCE = "source"

    def __call__(self, output: str) -> "Identifier":
        """
        Build an identifier by combining this node name with an output name.

        Args:
            output: Output name.

        Returns:
            Identifier with this node name and the specified output name.
        """
        return Identifier(self.value, output)


class Node(Protocol):
    """Node that indicates its inputs and outputs."""

    @property
    def input_names(self) -> tuple["Identifier", ...]: ...

    @property
    def output_names(self) -> tuple[str, ...]: ...


class Identifier(NamedTuple):
    """
    A node name and the name of its output, separated by a period.

    For example, a node named `blk` that returns two outputs `foo` and `bar`
    will have output names `blk.foo` and `blk.bar`.

    Attributes:
        node: Name of the node that produced the output.
        output: Name of the output produced by the node.
        is_batch: Whether the output is dataset-derived.
    """

    node: str
    output: str
    is_batch: bool = False

    def __str__(self) -> str:
        """Return a string representation of this instance."""
        return f"{self.node}.{self.output}"

    @classmethod
    def from_outputs(cls, name: str, node: Node) -> tuple[Self, ...]:
        """
        Build a tuple of identifiers from a named node.

        Args:
            name: Name of the node, used as the prefix of identifiers.
            node: Node that returns outputs, used as the suffixes of identifiers.

        Returns:
            Tuple of identifiers.
        """
        return tuple(cls(name, output) for output in node.output_names)

    @classmethod
    def from_str(cls, value: str | Self, *, is_batch: bool = False) -> Self:
        """
        Convert a string into an identifier.

        Args:
            value: String value to convert.
            is_batch: Whether to create a batch identifier.

        Returns:
            Parsed identifier.

        Raises:
            TypeError: If the value is neither a string nor an identifier.
            ValueError: If the value is not a proper identifier string.
        """
        if isinstance(value, cls):
            return value
        elif not isinstance(value, str):
            raise TypeError(f"expected str, got {type(value).__name__}")

        parts = value.split(".")
        if len(parts) != 2:
            raise ValueError(f"expected 'node.output' format, got {value!r}")

        (node, output) = parts
        return cls(node, output, is_batch)


def _is_batch_identifier(value: Identifier) -> Identifier:
    """Validator that ensures an identifier has `is_batch` set to true."""
    if value.is_batch:
        return value
    else:
        raise ValueError(f"expected a batch identifier, but '{value}' is not")


type IdentifierStr = Annotated[
    Identifier,
    BeforeValidator(Identifier.from_str),
    PlainSerializer(Identifier.__str__),
]
"""Type annotation for identifiers that may be coerced from strings."""

type BatchIdentifierStr = Annotated[
    Identifier,
    BeforeValidator(Identifier.from_str),
    AfterValidator(_is_batch_identifier),
    PlainSerializer(Identifier.__str__),
]
"""Type annotation for batch identifiers that may be coerced from strings."""

__all__ = ["BatchIdentifierStr", "Identifier", "IdentifierStr", "Reserved"]
