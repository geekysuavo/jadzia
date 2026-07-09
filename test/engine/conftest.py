"""Engine test fixtures."""

# Standard
import random

# Third party
import equinox as eqx
import grain
import jax
import jax.random as jr
import numpy as np
import optax
import pytest
from orbax.checkpoint import v1 as ocp

# Dax
from jadzia.nn.linear import Linear
from jadzia.nn.loss.composite import CompositeLoss, CompositeLossTerm
from jadzia.nn.loss.regression import MSE
from jadzia.nn.model import CompositeModel, CompositeModelBlock


@pytest.fixture(scope="session")
def model():
    """Example model."""
    block = CompositeModelBlock(
        name="linear",
        input_names=("source.value",),
        output_names=("linear.output",),
    )
    return CompositeModel(
        modules={"linear": Linear(16, 1, bias=False, key=jr.key(42))},
        outputs=("linear.output",),
        blocks=(block,),
    )


@pytest.fixture(scope="session")
def model_state(model):
    """Example model state."""
    (_, state) = eqx.nn.make_with_state(lambda: model)()
    return state


@pytest.fixture(scope="session")
def loss_fn():
    """Example loss function."""
    term = CompositeLossTerm(
        module=MSE(),
        name="mse",
        weight=1.0,
        prediction_names=("linear.output",),
        reference_names=("source.label",),
    )
    return CompositeLoss(terms=(term,))


@pytest.fixture(scope="session")
def optimizer():
    """Example optimizer."""
    return optax.adam(learning_rate=0.001)


@pytest.fixture(scope="session")
def optimizer_state(optimizer, model):
    """Example optimizer state."""
    return optimizer.init(eqx.filter(model, eqx.is_inexact_array))


@pytest.fixture(scope="session")
def dataloader():
    """Example data loader."""
    rng = np.random.default_rng()
    num_records = 100
    source = [
        {
            "source.value": rng.normal(size=(16,)),
            "source.label": random.normalvariate(),
        }
        for _ in range(num_records)
    ]
    return grain.DataLoader(
        data_source=source,
        sampler=grain.samplers.SequentialSampler(num_records),
        operations=(grain.transforms.Batch(batch_size=8),),
    )


@pytest.fixture(scope="session")
def sharding():
    """Example data sharding."""
    mesh = jax.make_mesh((jax.device_count(),), ("device",))
    return jax.sharding.NamedSharding(mesh, jax.P())


@pytest.fixture(params=(True, False))
def checkpointer(tmp_path, request):
    """Example checkpointer."""
    if request.param:
        return ocp.training.Checkpointer(str(tmp_path.resolve()))
    else:
        return None
