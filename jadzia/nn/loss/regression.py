"""Regression loss functions."""

# Third party
import equinox as eqx
import jax
import jax.numpy as jnp


class MAE(eqx.Module):
    """Mean absolute error loss."""

    def __call__(self, yhat: jax.Array, /, y: jax.Array) -> jax.Array:
        r"""
        Compute the loss function values.

        Args:
            yhat: $(\dots)$, Predicted values.
            y: $(\dots)$, Reference values.

        Returns:
            $(\dots)$, Unreduced loss function value.
        """
        return jnp.abs(yhat - y)


class MSE(eqx.Module):
    """Mean squared error loss."""

    def __call__(self, yhat: jax.Array, /, y: jax.Array) -> jax.Array:
        r"""
        Compute the loss function values.

        Args:
            yhat: $(\dots)$, Predicted values.
            y: $(\dots)$, Reference values.

        Returns:
            $(\dots)$, Unreduced loss function value.
        """
        return jnp.square(yhat - y)


class Huber(eqx.Module):
    """Huber loss."""

    delta: float

    def __init__(self, delta: float = 1.0):
        """
        Initialize the module.

        Args:
            delta: Threshold between the quadratic and linear loss regions.
        """
        self.delta = delta

    def __call__(self, yhat: jax.Array, /, y: jax.Array) -> jax.Array:
        r"""
        Compute the loss function values.

        Args:
            yhat: $(\dots)$, Predicted values.
            y: $(\dots)$, Reference values.

        Returns:
            $(\dots)$, Unreduced loss function value.
        """
        error = yhat - y
        abs_error = jnp.abs(error)
        quadratic = jnp.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        return 0.5 * quadratic**2 + self.delta * linear


__all__ = ["MAE", "MSE", "Huber"]
