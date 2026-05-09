"""Tests of direct regularization losses."""

# Third party
import jax
import jax.numpy as jnp
import jax.random as jr

# Dax
from jadzia.nn.loss.direct import Direct


def test_direct():
    """The mean absolute error loss produces reasonable values and gradients."""
    yhat = jr.normal(jr.key(42), shape=(8,))

    loss_fn = Direct()
    grad_fn = jax.grad(lambda yhat: loss_fn(yhat).mean())

    loss = loss_fn(yhat)
    assert loss.shape == yhat.shape
    assert jnp.isfinite(loss).all()

    grad = grad_fn(yhat)
    assert grad.shape == yhat.shape
    assert jnp.isfinite(grad).all()
