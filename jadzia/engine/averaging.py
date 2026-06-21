"""Loss averaging functionality."""

# Standard
from typing import Any, NamedTuple, Self

# Third party
import jax


class Average[T: Any](NamedTuple):
    """Class representing a running average of a given type."""

    value: T
    num: int = 1

    def update(self, sample: T) -> Self:
        """
        Update the average with a new sample.

        Args:
            sample: New value to include in the average.

        Returns:
            Updated average.
        """

        def update(prev: jax.Array, new: jax.Array) -> jax.Array:
            return prev + (new - prev) / (self.num + 1)

        new_value = jax.tree.map(update, self.value, sample)
        return type(self)(new_value, self.num + 1)


__all__ = ["Average"]
