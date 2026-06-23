"""Tests of checkpointing."""

# Standard
from pathlib import Path

# Third party
import equinox as eqx
import grain
import jax.random as jr
import pytest
from optax import adam
from orbax.checkpoint import v1 as ocp

# Dax
from jadzia.engine.checkpointing import (
    Checkpoint,
    DataLoaderCheckpoint,
    ModelCheckpoint,
)


def test_round_trip(tmp_path):
    """Checkpointable data round-trip successfully."""
    num_records = 100
    source = tuple(range(num_records))

    def make(*, seed: int, steps: int) -> Checkpoint:
        (model, model_state) = eqx.nn.make_with_state(eqx.nn.MultiheadAttention)(
            num_heads=8,
            query_size=32,
            key=jr.key(seed),
        )

        optimizer = adam(learning_rate=0.001)
        optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        dataloader = grain.DataLoader(
            data_source=source,
            sampler=grain.samplers.SequentialSampler(num_records),
        )

        dataiter = iter(dataloader)
        for _ in range(steps):
            next(dataiter)

        return Checkpoint(
            step=steps,
            model=model,
            model_state=model_state,
            dataloader_iter=dataiter,
            optimizer_state=optimizer_state,
        )

    data = make(seed=42, steps=2)
    saved = data.save(checkpointer=None)
    assert saved is False

    loaded = data.load(checkpointer=None)
    assert loaded is None

    checkpointer = ocp.training.Checkpointer(str(tmp_path))

    loaded = data.load(checkpointer)
    assert loaded is None

    saved = data.save(checkpointer)
    assert saved is True

    new_data = make(seed=35, steps=0)
    loaded = new_data.load(checkpointer)

    assert loaded.step == data.step
    assert loaded.model == data.model
    assert loaded.model_state.tree_flatten() == data.model_state.tree_flatten()
    assert loaded.optimizer_state == data.optimizer_state
    for _ in range(3):
        assert next(data.dataloader_iter) == next(new_data.dataloader_iter)

    loader_data = DataLoaderCheckpoint(new_data.step, new_data.dataloader_iter)
    loaded = loader_data.load(checkpointer)
    assert loaded.step == data.step

    model_data = ModelCheckpoint(new_data.model, new_data.model_state)
    checkpoint_dir = Path(checkpointer.latest.path)
    loaded = model_data.load(checkpoint_dir)
    assert loaded.model == data.model
    assert loaded.model_state.tree_flatten() == data.model_state.tree_flatten()

    with pytest.raises(NotADirectoryError):
        model_data.load(tmp_path / "does-not-exist")
