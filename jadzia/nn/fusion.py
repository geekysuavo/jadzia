"""Modules that combine arrays."""

# Third party
import equinox as eqx
import jax
import jax.numpy as jnp


class SumFusion(eqx.Module):
    """Sum multiple arrays together."""

    def __call__(self, *xs: jax.Array) -> jax.Array:
        r"""
        Compute the sum of multiple identically shaped arrays.

        Args:
            xs: Each $(\dots)$, Input arrays.

        Returns:
            $(\dots)$, Output sum.
        """
        return jnp.stack(xs).sum(axis=0)


class MeanFusion(eqx.Module):
    """Average multiple arrays together."""

    def __call__(self, *xs: jax.Array) -> jax.Array:
        r"""
        Compute the mean of multiple identically shaped arrays.

        Args:
            xs: Each $(\dots)$, Input arrays.

        Returns:
            $(\dots)$, Output mean.
        """
        return jnp.stack(xs).mean(axis=0)


class MaxFusion(eqx.Module):
    """Combine multiple arrays together via max-reduction."""

    def __call__(self, *xs: jax.Array) -> jax.Array:
        r"""
        Compute the maximum of multiple identically shaped arrays.

        Args:
            xs: Each $(\dots)$, Input arrays.

        Returns:
            $(\dots)$, Output maximum.
        """
        return jnp.stack(xs).max(axis=0)


__all__ = ["MaxFusion", "MeanFusion", "SumFusion"]
