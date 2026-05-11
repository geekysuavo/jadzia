"""Linear layers."""

# Standard
import math

# Third party
import equinox as eqx
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
import jax.random as jr


class Linear(eqx.Module):
    """Linear layer with an optional bias."""

    weight: jax.Array
    bias: jax.Array | None

    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        bias: bool,
        key: jax.Array,
    ):
        r"""
        Initialize the module.

        Args:
            num_input_features: $n_{\mathrm{in}}$, Number of input features.
            num_output_features: $n_{\mathrm{out}}$, Number of output features.
            bias: Whether to include a bias term.
            key: Pseudorandom number generator key.
        """
        if bias:
            (weight_key, bias_key) = jr.split(key)
        else:
            (weight_key, bias_key) = (key, None)

        weight_shape = (num_input_features, num_output_features)
        self.weight = init.kaiming_uniform()(weight_key, weight_shape)

        if bias:
            assert bias_key is not None
            bias_shape = (num_output_features,)
            scale = 1 / math.sqrt(num_input_features)
            self.bias = init.uniform(2 * scale)(bias_key, bias_shape) - scale
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        r"""
        Apply the linear layer.

        Args:
            x: $(\dots, n_{\mathrm{in}})$, Input array.

        Returns:
            $(\dots, n_{\mathrm{out}})$, Output array.
        """
        if self.bias is not None:
            return jnp.matmul(x, self.weight) + self.bias
        else:
            return jnp.matmul(x, self.weight)


__all__ = ["Linear"]
