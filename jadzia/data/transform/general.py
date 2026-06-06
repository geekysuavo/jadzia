"""Transformations that perform general computations."""

# Standard
from collections.abc import Callable, Mapping, Sequence

# Third party
import jax.typing as jt

# Dax
from jadzia.data.transform.base import Map

type Arg = bool | float | int | str
"""Expected type of function arguments."""


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


__all__ = ["Unary"]
