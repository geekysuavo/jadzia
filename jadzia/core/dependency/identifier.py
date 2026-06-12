"""Identifier classes for dependency management."""

# Standard
from enum import StrEnum
from typing import Annotated, NamedTuple, Protocol, Self

# Third party
from pydantic import BeforeValidator, PlainSerializer


class Reserved(StrEnum):
    """
    Enumeration of reserved node names.

    - `*`: Wildcard used to gather node outputs together.
    - `id`: Identifying data inserted by the data source.
    - `eval`: Evaluator result.
    - `loss`: Composite loss function value.
    - `source`: Value provided by the data source manifest or archives.
    """

    ALL = "*"
    ID = "id"
    EVAL = "eval"
    LOSS = "loss"
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
    """

    node: str
    output: str

    def __str__(self) -> str:
        """Return a string representation of this instance."""
        return f"{self.node}.{self.output}"

    @property
    def is_gather(self) -> bool:
        """Whether this identifier refers to a "gather" node of type `name.*`."""
        return self.output == Reserved.ALL

    @classmethod
    def gather(cls, name: str) -> Self:
        """
        Build a gather identifier from a node name.

        Args:
            name: Name of the node.

        Returns:
            Gather identifier.
        """
        return cls(name, Reserved.ALL)

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
    def from_str(cls, value: str | Self) -> Self:
        """
        Convert a string into an identifier.

        Args:
            value: String value to convert.

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
        return cls(node, output)


type IdentifierStr = Annotated[
    Identifier,
    BeforeValidator(Identifier.from_str),
    PlainSerializer(Identifier.__str__),
]
"""Type annotation for identifiers that may be coerced from strings."""

__all__ = ["Identifier", "IdentifierStr", "Reserved"]
