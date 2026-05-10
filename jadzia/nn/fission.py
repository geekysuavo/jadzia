"""Modules that break arrays apart."""

# Standard
from itertools import accumulate

# Third party
import equinox as eqx
import jax
import jax.numpy as jnp


class SplitFission(eqx.Module):
    """Split an array by sizes along a specified axis."""

    axis: int
    indices: tuple[int, ...]

    def __init__(self, axis: int, sizes: tuple[int, ...]):
        """
        Initialize the module.

        Args:
            axis: Split axis.
            sizes: Sizes to split the array into.
        """
        self.axis = axis
        self.indices = tuple(accumulate(sizes))[:-1]

    def __call__(self, x: jax.Array) -> tuple[jax.Array, ...]:
        r"""
        Split an input array.

        Args:
            x: $(\dots_1, n, \dots_2)$, Input array.

        Returns:
            Each $(\dots_1, n_i \dots_2)$ where $n = \sum_i n_i$, Output arrays.
        """
        return tuple(jnp.split(x, self.indices, axis=self.axis))


class UnstackFission(eqx.Module):
    """Unstack an array along a specified axis."""

    axis: int

    def __call__(self, x: jax.Array) -> tuple[jax.Array, ...]:
        r"""
        Unstack an input array.

        Args:
            x: $(\dots_1, n, \dots_2)$, Input array.

        Returns:
            Each $(\dots_1, \dots_2)$, Output arrays.
        """
        return jnp.unstack(x, axis=self.axis)


__all__ = ["SplitFission", "UnstackFission"]
