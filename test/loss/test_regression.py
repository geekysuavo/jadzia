"""Tests of regression losses."""

# Third party
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

# Dax
from jadzia.nn.loss.regression import MAE, MSE, Huber


@pytest.fixture(scope="session")
def scalar_inputs():
    """Example predictions and references for scalar regression losses."""
    (yhat_key, y_key) = jr.split(jr.key(42))
    yhat = jr.normal(yhat_key, shape=(8,))
    y = jr.normal(y_key, shape=(8,))
    return (yhat, y)


def test_mae(scalar_inputs):
    """The mean absolute error loss produces reasonable values and gradients."""
    loss_fn = MAE()
    grad_fn = jax.grad(loss_fn)

    (yhat, y) = scalar_inputs
    loss = loss_fn(yhat, y)
    assert loss.shape == ()
    assert jnp.isfinite(loss)

    grad = grad_fn(yhat, y)
    assert grad.shape == yhat.shape
    assert jnp.isfinite(grad).all()


def test_mse(scalar_inputs):
    """The mean squared error loss products reasonable values and gradients."""
    loss_fn = MSE()
    grad_fn = jax.grad(loss_fn)

    (yhat, y) = scalar_inputs
    loss = loss_fn(yhat, y)
    assert loss.shape == ()
    assert jnp.isfinite(loss)

    grad = grad_fn(yhat, y)
    assert grad.shape == yhat.shape
    assert jnp.isfinite(grad).all()


def test_huber(scalar_inputs):
    """The huber loss products reasonable values and gradients."""
    loss_fn = Huber()
    grad_fn = jax.grad(loss_fn)

    (yhat, y) = scalar_inputs
    loss = loss_fn(yhat, y)
    assert loss.shape == ()
    assert jnp.isfinite(loss)

    grad = grad_fn(yhat, y)
    assert grad.shape == yhat.shape
    assert jnp.isfinite(grad).all()
