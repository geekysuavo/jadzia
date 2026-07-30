"""Condition string validation."""

# Standard
import contextlib
import json
import operator
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Self

# Third party
from pydantic import (
    BeforeValidator,
    PlainSerializer,
    StrictBool,
    StrictFloat,
    StrictInt,
)

# Dax
from jadzia.core.validation.base import Config


class Relation(StrEnum):
    """
    Enumeration of supported operator strings.

    - `==`: [operator.eq][], Equal.
    - `!=`: [operator.ne][], Not equal.
    - `<=`: [operator.le][], Less than or equal.
    - `>=`: [operator.ge][], Greater than or equal.
    - `<`: [operator.lt][], Less than.
    - `>`: [operator.gt][], Greater than.
    """

    EQ = "=="
    NE = "!="
    LE = "<="
    GE = ">="
    LT = "<"
    GT = ">"

    def __call__(self, a: Any, b: Any) -> bool:
        """
        Evaluate the operator on two values.

        Args:
            a: First value.
            b: Second value.

        Returns:
            Comparison result.
        """
        op = getattr(operator, self.name.lower())
        return op(a, b)


class Condition(Config):
    """
    Condition expression.

    Attributes:
        name: Name to read values from.
        relation: Comparison operator.
        value: Value to compare against.
    """

    name: str
    relation: Relation
    value: StrictInt | StrictFloat | StrictBool | str

    PATTERN: ClassVar[re.Pattern] = re.compile(r"(\w+)\s+([!<>=]+)\s+(.*)")

    def __str__(self) -> str:
        """Return a string representation of this instance."""
        return f"{self.name} {self.relation} {json.dumps(self.value)}"

    def __call__(self, obj: Mapping[str, Any]) -> bool:
        """
        Evaluate the condition on an input mapping.

        Args:
            obj: Mapping from names to values.

        Returns:
            Condition evaluation result.
        """
        return self.relation(obj[self.name], self.value)

    @classmethod
    def from_str(cls, value: str) -> Self:
        """
        Convert a string into a condition.

        Args:
            value: String value to convert.

        Returns:
            Parsed condition.

        Raises:
            TypeError: If the value is not a string.
            ValueError: If the value is not a proper condition string.
        """
        if not isinstance(value, str):
            raise TypeError(f"expected str, got {type(value).__name__}")

        matched = cls.PATTERN.match(value)
        if not matched:
            raise ValueError(f"invalid condition expression: '{value}'")

        (name, relation, value_str) = matched.groups()
        value = json.loads(value_str)

        return cls(name=name, relation=relation, value=value)


class ForEach(Config):
    """
    For-each expression.

    Attributes:
        name: Name to enumerate values from.
    """

    name: str

    KEYWORD: ClassVar[str] = "for-each"
    PATTERN: ClassVar[re.Pattern] = re.compile(rf"{KEYWORD}\s+(\w+)")

    def __str__(self) -> str:
        """Return a string representation of this instance."""
        return f"{type(self).KEYWORD} {self.name}"

    @classmethod
    def from_str(cls, value: str) -> Self:
        """
        Convert a string into a for-each.

        Args:
            value: String value to convert.

        Returns:
            Parsed for-each.

        Raises:
            TypeError: If the value is not a string.
            ValueError: If the value is not a proper for-each string.
        """
        if not isinstance(value, str):
            raise TypeError(f"expected str, got {type(value).__name__}")

        matched = cls.PATTERN.match(value)
        if not matched:
            raise ValueError(f"invalid for-each expression: '{value}'")

        (name,) = matched.groups()
        return cls(name=name)


def _validate_union(value: Any) -> Condition | ForEach:
    """Convert a string into a condition or for-each expression."""
    with contextlib.suppress(TypeError, ValueError):
        return Condition.from_str(value)

    with contextlib.suppress(TypeError, ValueError):
        return ForEach.from_str(value)

    raise ValueError("invalid condition or for-each expression")


def _serialize_union(value: Condition | ForEach) -> str:
    """Convert a condition or for-each expressions into a string."""
    return str(value)


type ConditionStr = Annotated[
    Condition,
    BeforeValidator(Condition.from_str),
    PlainSerializer(Condition.__str__),
]
"""Type annotation for conditions that may be coerced from strings."""

type ConditionOrForEachStr = Annotated[
    Condition | ForEach,
    BeforeValidator(_validate_union),
    PlainSerializer(_serialize_union),
]
"""Type annotation for conditions or for-eaches that may be coerced from strings."""

__all__ = [
    "Condition",
    "ConditionOrForEachStr",
    "ConditionStr",
    "ForEach",
    "Relation",
]
