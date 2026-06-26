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
from jadzia.engine.training import TrainArgs


@pytest.mark.parametrize("with_testing", [True, False])
def test_training(
    mocker,
    model,
    model_state,
    loss_fn,
    optimizer,
    optimizer_state,
    dataloader,
    checkpointer,
    with_testing,
):
    """Training loops return updated models."""
    mocker.patch("jadzia.engine.training.log.info")

    test_args = TestArgs(model, model_state, loss_fn, dataloader, num_steps=None)
    args = TrainArgs(
        model,
        model_state,
        loss_fn,
        optimizer,
        optimizer_state,
        dataloader,
        num_steps=4,
        checkpointer=checkpointer,
        test_args=test_args if with_testing else None,
        test_every=2,
        log_every=2,
    )

    out = args.run()
    assert isinstance(out, TrainArgs)
    assert out != args

    # restarts from checkpoint.
    new_out = args.run()
    assert new_out.model == out.model
    assert new_out.optimizer_state == out.optimizer_state

    # exhausts data loader.
    new_args = args._asdict()
    new_args["num_steps"] = 20
    new_args = TrainArgs(**new_args)
    new_args.run()


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
