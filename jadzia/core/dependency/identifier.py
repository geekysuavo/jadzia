"""Identifier classes for dependency management."""

# Standard
from enum import StrEnum
from typing import Annotated, NamedTuple, Protocol, Self

# Third party
from pydantic import BeforeValidator, PlainSerializer


class _Node(Protocol):
    @property
    def output_names(self) -> tuple[str, ...]: ...


class SpecialNodes(StrEnum):
    """
    Enumeration of specially reserved node names.

    - `batch`: Data taken from the input minibatch.
    """

    BATCH = "batch"


class Identifier(NamedTuple):
    """
    A node name and the name of its output, separated by a period.

    For example, a node named `blk` that returns two outputs `foo` and `bar`
    will have output names `blk.foo` and `blk.bar`.

    Attributes:
        node: Name of the node that produced the output.
        output: Name of the output produced by the node.
    """

    node: str
    output: str

    def __str__(self) -> str:
        """Return a string representation of this instance."""
        return f"{self.node}.{self.output}"

    @property
    def is_special(self) -> bool:
        """Whether the identifier is from any special node."""
        return self.node in SpecialNodes

    @property
    def is_batch(self) -> bool:
        """Whether the identifier is from the special `batch` node."""
        return self.node == SpecialNodes.BATCH

    @classmethod
    def from_outputs(cls, name: str, node: _Node) -> tuple[Self, ...]:
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
    def from_str(cls, value: str) -> Self:
        """
        Convert a string into an identifier.

        Args:
            value: String value to convert.

        Returns:
            Parsed identifier.

        Raises:
            TypeError: If the value is not a string.
            ValueError: If the value is not a proper identifier string.
        """
        if not isinstance(value, str):
            raise TypeError(f"expected str, got {type(value).__name__}")

        parts = value.split(".")
        if len(parts) != 2:
            raise ValueError(f"expected 'node.output' format, got {value!r}")

        return cls(*parts)


type IdentifierStr = Annotated[
    Identifier,
    BeforeValidator(Identifier.from_str),
    PlainSerializer(Identifier.__str__),
]
"""Type annotation for identifiers that may be coerced from strings."""

__all__ = ["Identifier", "IdentifierStr", "SpecialNodes"]
