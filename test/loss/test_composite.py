"""Tests of composite losses."""

# Third party
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

# Dax
from jadzia.nn.loss.composite import CompositeLoss, CompositeLossTerm
from jadzia.nn.loss.direct import Direct
from jadzia.nn.loss.regression import MSE, Huber


@pytest.mark.parametrize("missing", [False, True])
def test_composite_loss(missing):
    """Composite loss functions return reasonable values and gradients."""
    terms = (
        CompositeLossTerm(
            module=MSE(),
            weight=1.5,
            prediction_names=("mse",),
            reference_names=("mse",),
        ),
        CompositeLossTerm(
            module=Huber(),
            weight=2.2,
            prediction_names=("huber",),
            reference_names=("huber",),
        ),
        CompositeLossTerm(
            module=Direct(),
            weight=0.7,
            prediction_names=("direct",),
            reference_names=(),
        ),
    )

    loss_fn = jax.jit(CompositeLoss(terms))
    grad_fn = jax.grad(loss_fn)

    key = jr.key(42)
    (key, mse_pred_key, mse_ref_key) = jr.split(key, num=3)
    mse_pred = jr.normal(mse_pred_key, shape=(8,))
    mse_ref = jr.normal(mse_ref_key, shape=(8,))

    (key, huber_pred_key, huber_ref_key) = jr.split(key, num=3)
    huber_pred = jr.normal(huber_pred_key, shape=(8, 5))
    huber_ref = jr.normal(huber_ref_key, shape=(8, 5))

    direct_pred = jr.normal(key, shape=(8, 3))

    if missing:
        mse_ref = mse_ref * jnp.nan  # all missing, grad must be zero
        huber_ref = huber_ref.at[0].set(jnp.nan)

    predictions = {"mse": mse_pred, "huber": huber_pred, "direct": direct_pred}
    references = {"mse": mse_ref, "huber": huber_ref}

    loss = loss_fn(predictions, references)
    assert loss.shape == ()
    assert jnp.isfinite(loss)

    grads = grad_fn(predictions, references)
    assert isinstance(grads, dict)
    for name, grad in grads.items():
        assert grad.shape == predictions[name].shape
        assert jnp.isfinite(grad).all()

    if missing:
        assert (grads["mse"] == 0).all()
