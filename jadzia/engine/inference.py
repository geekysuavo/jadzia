"""Inference functionality."""

# Standard
import itertools
import warnings
from pathlib import Path
from typing import NamedTuple

# Third party
import equinox as eqx
import jax
import numpy as np
import polars as pl
from grain import DataLoader
from orbax.checkpoint import v1 as ocp

# Dax
import jadzia.engine.logging as log
from jadzia.engine.checkpointing import DataLoaderCheckpoint, ModelCheckpoint
from jadzia.engine.utility import make_send_to_device
from jadzia.nn.model import CompositeModel


@eqx.filter_jit
def evaluate(
    model: CompositeModel,
    model_state: eqx.nn.State,
    xy: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    """
    Perform a single model evaluation.

    Args:
        model: Model to evaluate.
        model_state: Model state.
        xy: Model input values.

    Returns:
        Model output values.
    """
    (outputs, _) = model(xy, state=model_state)
    return outputs


def infer(
    model: CompositeModel,
    model_state: eqx.nn.State,
    dataloader: DataLoader,
    sharding: jax.sharding.Sharding,
    capture: tuple[str, ...],
    checkpointer: ocp.training.Checkpointer | None,
    checkpoint_dir: Path,
    output_file: Path | None,
) -> pl.DataFrame:
    """
    Evaluate a model.

    Args:
        model: Model to evaluate.
        model_state: Model state.
        dataloader: Data loader.
        sharding: Batch data sharding.
        capture: Keys (data frame columns) to capture in the output.
        checkpointer: Optional checkpointer used to save inference checkpoints.
        checkpoint_dir: Directory to load the model and its state from.
        output_file: File to write inference outputs into.

    Returns:
        Output data frame.
    """
    send_to_device = make_send_to_device(sharding)
    dataiter = iter(dataloader)
    begin_step = 0
    buffer = []

    loaded = ModelCheckpoint(model, model_state).load(checkpoint_dir)
    model = loaded.model
    model_state = loaded.model_state
    del loaded

    reloaded = DataLoaderCheckpoint(begin_step, dataiter, None).load(checkpointer)
    if reloaded:
        begin_step = reloaded.step + 1
        dataiter = reloaded.dataloader_iter
        del reloaded

    def combine(
        xy: dict[str, jax.Array],
        yhat: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Combine batch data and model outputs, and copy to the host."""
        data = {**xy, **yhat}
        return jax.copy_to_host_async({key: data[key] for key in capture})

    def reload(cp: ocp.training.CheckpointMetadata) -> list[dict[str, jax.Array]]:
        """Reload batch data and model outputs from a checkpoint."""
        # orbax emits a UserWarning when it must first read sharding
        # information from stored metadata, instead of having it
        # provided. this suppresses that warning.
        warnings.simplefilter("ignore", category=UserWarning)
        template = DataLoaderCheckpoint(0, dataiter, None)
        batches = template.load(Path(cp.path)).batches
        assert batches is not None
        return batches

    for step in itertools.count(start=begin_step):
        try:
            batch = send_to_device(next(dataiter))
        except StopIteration:
            break

        outputs = evaluate(model, model_state, batch)
        buffer.append(combine(batch, outputs))

        saved = DataLoaderCheckpoint(step, dataiter, buffer).save(checkpointer)
        if saved:
            assert checkpointer is not None
            log.info(f"[{step=}] Saved checkpoint to {checkpointer.directory}")
            buffer.clear()

    if checkpointer is not None:
        reloaded_buffers = map(reload, checkpointer.checkpoints)
        buffer = [*itertools.chain.from_iterable(reloaded_buffers), *buffer]

    data = jax.tree.map(lambda *xs: np.concat(xs), *buffer)
    df = pl.DataFrame(data)
    if output_file is not None:
        df.write_parquet(output_file)

    return df


class InferArgs(NamedTuple):
    """Encapsulation of arguments to [jadzia.engine.inference.infer][]."""

    model: CompositeModel
    model_state: eqx.nn.State
    dataloader: DataLoader
    sharding: jax.sharding.Sharding
    capture: tuple[str, ...]
    checkpointer: ocp.training.Checkpointer | None
    checkpoint_dir: Path
    output_file: Path | None

    def run(self) -> pl.DataFrame:
        """Run inference with the specified arguments."""
        return infer(*self)


__all__ = ["InferArgs", "infer"]
