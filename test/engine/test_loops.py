"""Tests of the testing loop."""

# Third party
import jax
import polars as pl
import pytest
from orbax.checkpoint import v1 as ocp

# Dax
from jadzia.engine.checkpointing import ModelCheckpoint
from jadzia.engine.inference import InferArgs
from jadzia.engine.testing import TestArgs


@pytest.mark.parametrize("num_steps", [None, 3])
def test_testing(
    model,
    model_state,
    loss_fn,
    dataloader,
    num_steps,
):
    """Testing loops return averaged loss values."""
    args = TestArgs(model, model_state, loss_fn, dataloader, num_steps)

    new_args = args.update(model, model_state)
    assert isinstance(new_args, TestArgs)

    out = args.run()
    assert isinstance(out, tuple)

    (loss, terms) = out
    assert isinstance(loss, jax.Array)
    assert terms["mse"] == loss


def test_inference(
    tmp_path_factory,
    model,
    model_state,
    optimizer_state,
    dataloader,
    checkpointer,
):
    """Inference loops return data frames."""
    model_dir = tmp_path_factory.mktemp("model")
    output_dir = tmp_path_factory.mktemp("output")
    output_file = output_dir / "out.parquet"

    model_checkpointer = ocp.training.Checkpointer(str(model_dir.resolve()))
    saved = ModelCheckpoint(model, model_state).save(model_checkpointer)
    assert saved is True

    capture = ("source.label", "linear.output")
    args = InferArgs(
        model,
        model_state,
        dataloader,
        capture,
        checkpointer,
        checkpoint_dir=model_dir / "0",
        output_file=output_file,
    )

    df = args.run()
    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == set(capture)
    assert len(df) == 100

    # restarts from checkpoint.
    new_df = args.run()
    assert set(df.columns) == set(capture)
    assert len(new_df) == len(df)
