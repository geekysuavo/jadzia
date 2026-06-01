"""Tests of composite models."""

# Third party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

# Dax
from jadzia.core.dependency.identifier import Identifier
from jadzia.nn.model import CompositeModel, CompositeModelBlock


def test_composite_model():
    """Composite model return reasonable values and gradients."""
    (proj_key, x_key, y_key) = jr.split(jr.key(42), num=3)
    x = jr.normal(x_key, shape=(4, 8))
    y = jr.normal(y_key, shape=(4, 3))

    proj = eqx.nn.Linear(8, 1, key=proj_key)
    norm = eqx.nn.BatchNorm(3, axis_name="batch", mode="batch")
    modules = {"proj": proj, "norm": norm}

    blocks = (
        CompositeModelBlock(
            name="proj",
            input_names=(Identifier("batch", "x"),),
            output_names=("output",),
        ),
        CompositeModelBlock(
            name="norm",
            input_names=(Identifier("batch", "y"),),
            output_names=("output",),
        ),
    )

    outputs = (
        ("x_proj", Identifier("proj", "output")),
        ("y_norm", Identifier("norm", "output")),
    )

    model = CompositeModel(modules, blocks, outputs)
    (model, state) = eqx.nn.make_with_state(lambda: model)()
    batch_model = jax.jit(jax.vmap(model, axis_name="batch", in_axes=(0, None)))

    inputs = {"x": x, "y": y}
    (outputs, new_state) = batch_model(inputs, state)
    assert outputs["x_proj"].shape == (4, 1)
    assert outputs["y_norm"].shape == (4, 3)
    assert isinstance(new_state, eqx.nn.State)

    def loss_fn(model, inputs, state):
        batch_model = jax.vmap(model, axis_name="batch", in_axes=(0, None))
        (outputs, new_state) = batch_model(inputs, state)
        loss = sum(value.sum() for value in outputs.values())
        return (loss, new_state)

    grad_fn = eqx.filter_grad(loss_fn, has_aux=True)
    grads, grad_state = grad_fn(model, inputs, state)
    assert isinstance(grads, CompositeModel)
    assert isinstance(grad_state, eqx.nn.State)
    for name in modules:
        assert grads.modules[name].weight.shape == modules[name].weight.shape
        assert grads.modules[name].bias.shape == modules[name].bias.shape
        assert jnp.isfinite(grads.modules[name].weight).all()
        assert jnp.isfinite(grads.modules[name].bias).all()
