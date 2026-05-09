"""Finiteness masking for loss functions."""

# Third party
import jax
import jax.numpy as jnp


def build_mask(array: jax.Array) -> jax.Array:
    r"""
    Return a finiteness mask for a single reference array of any shape.

    Args:
        array: $(n, \dots)$, Input array potentially containing nonfinite entries.

    Returns:
        $(n)$, Mask indicating which batch indices are completely finite.
    """
    return jnp.isfinite(jax.lax.collapse(array, start_dimension=1)).all(axis=1)


def apply_mask(array: jax.Array, mask: jax.Array) -> jax.Array:
    r"""
    Zero out array elements corresponding to incompletely finite batch indices.

    Args:
        array: $(n, \dots)$, Input array to apply the mask to.
        mask: $(n)$, Mask returned by `build_mask`.

    Returns:
        $(n, \dots)$, Tensor containing zeros where `mask` is false.
    """
    mask_shape = (mask.size,) + (1,) * (array.ndim - 1)
    return jnp.where(mask.reshape(mask_shape), array, 0)


__all__ = ["apply_mask", "build_mask"]
