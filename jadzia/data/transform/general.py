"""Transformations that perform general computations."""

# Standard
from collections.abc import Callable, Mapping, Sequence

# Third party
import jax
import jax.typing as jt
import numpy as np

# Dax
from jadzia.data.transform.base import Map

type Arg = bool | float | int | str
"""Expected type of function arguments."""


class ArraySize(Map):
    """
    Return the size of an array, either along a single axis or by element count.

    Attributes:
        axis: Optional axis to return the size of, or `None` to return the
            total number of array elements.
    """

    axis: int | None

    def __call__(self, x: jax.Array | np.ndarray) -> int:
        """
        Apply the transformation.

        Args:
            x: Input array.

        Returns:
            Array axis size or number of elements.
        """
        if self.axis is not None:
            return x.shape[self.axis]
        else:
            return x.size


class Unary(Map):
    """
    Apply a user-specified unary function to a value.

    Attributes:
        function: Function to call.
        args: Positional function arguments.
        kwargs: Keyword function arguments.
    """

    function: Callable
    args: Sequence[Arg]
    kwargs: Mapping[str, Arg]

    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        """
        Apply the transformation.

        Args:
            x: Function input value.

        Returns:
            Function output value.
        """
        return self.function(x, *self.args, **self.kwargs)


__all__ = ["ArraySize", "Unary"]
