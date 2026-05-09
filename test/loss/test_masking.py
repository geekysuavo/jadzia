"""Tests of finiteness masking functions."""

# Third party
import jax.numpy as jnp

# Dax
from jadzia.nn.loss.masking import apply_mask, build_mask


def test_build_and_apply():
    """Masks have the correct shape and content."""
    x = jnp.array([[-1.1, 2.3], [0.5, jnp.nan], [jnp.nan, jnp.nan]])
    xm_true = jnp.array([[-1.1, 2.3], [0.0, 0.0], [0.0, 0.0]])
    mask_true = jnp.array([True, False, False])

    mask = build_mask(x)
    assert jnp.array_equal(mask, mask_true)

    xm = apply_mask(x, mask)
    assert jnp.array_equal(xm, xm_true)
