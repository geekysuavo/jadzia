"""Transformations that create auxiliary data."""

# Standard
import sys
from typing import Any

# Third party
import numpy as np
from numpy.random import Generator

# Dax
from jadzia.data.transform.base import Map, MapWithIndex, RandomMap

type Value = bool | float | int | str
"""Expected type of inserted values."""


class Insert(Map):
    """
    Insert a new value into the example.

    Attributes:
        value: Value to insert.
    """

    value: Value

    def __call__(self) -> Value:
        """Apply the transformation."""
        return self.value


class InsertCopy(Map):
    """Insert a copy of a value in an example."""

    def __call__(self, source: Any) -> Any:
        """
        Apply the tranformation.

        Args:
            source: Value to copy.

        Returns:
            Copy of the source value.
        """
        if isinstance(source, dict | list | set | np.ndarray):
            return source.copy()
        else:
            return source


class InsertFromSequence(MapWithIndex):
    """
    Insert a new value into the example from a sequence of values.

    Values are inserted such that iterating over a dataset sequentially
    will result in round-robin insertion from the sequence.

    Attributes:
        values: Values to insert in sequence.
    """

    values: tuple[Value, ...]

    def __call__(self, *, index: int) -> Value:
        """
        Apply the transformation.

        Args:
            index: Example index.

        Returns:
            Value to insert.
        """
        values_index = index % len(self.values)
        return self.values[values_index]


class InsertRandomChoice(RandomMap):
    """
    Insert a new value into the example, randomly sampled from a sequence of values.

    Attributes:
        values: Values to insert randomly.
        weights: Optional weights of each value.
    """

    values: tuple[Value, ...]
    weights: tuple[float, ...] | None

    def __call__(self, *, rng: Generator) -> Value:
        """
        Apply the transformation.

        Args:
            rng: Random number generator.

        Returns:
            Sampled value to insert.
        """
        return rng.choice(self.values, p=self.weights).item()


class InsertRandomInteger(RandomMap):
    """
    Insert a random integer value into the example.

    Attributes:
        lower: Lower bound, inclusive.
        upper: Upper bound, exclusive.
    """

    lower: int = 0
    upper: int = sys.maxsize

    def __call__(self, *, rng: Generator) -> int:
        """
        Apply the transformation.

        Args:
            rng: Random number generator.

        Returns:
            Sampled value to insert.
        """
        return rng.integers(self.lower, self.upper).item()


class InsertRandomUniform(RandomMap):
    """
    Insert a random uniform value into the example.

    Attributes:
        lower: Lower bound.
        upper: Upper bound.
    """

    lower: float = 0.0
    upper: float = 1.0

    def __call__(self, *, rng: Generator) -> float:
        """
        Apply the transformation.

        Args:
            rng: Random number generator.

        Returns:
            Sampled value to insert.
        """
        return rng.uniform(self.lower, self.upper)


__all__ = [
    "Insert",
    "InsertCopy",
    "InsertFromSequence",
    "InsertRandomChoice",
    "InsertRandomInteger",
    "InsertRandomUniform",
]
