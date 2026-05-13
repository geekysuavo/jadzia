"""Direct regularization (prediction-only) loss functions."""

# Third party
import equinox as eqx
import jax


class Direct(eqx.Module):
    """Direct loss."""

    def __call__(self, prediction: jax.Array) -> jax.Array:
        r"""
        Compute the loss function values.

        Args:
            prediction: $(\dots)$, Predicted values.

        Returns:
            $(\dots)$, Unreduced loss function value.
        """
        return prediction


__all__ = ["Direct"]
