"""Tests of the command line interface."""

# Third party
import pytest
from click.testing import CliRunner

# Dax
from jadzia.__main__ import main


@pytest.mark.parametrize("with_seed", [True, False])
def test_workflow(
    tmp_path_factory,
    write_train_config_file,
    write_infer_config_file,
    with_seed,
):
    """Training and inference produce the expected results from the CLI."""
    train_config_file = write_train_config_file()
    train_output_dir = tmp_path_factory.mktemp("train")

    checkpoint_dir = train_output_dir / "checkpoints" / "8"
    infer_config_file = write_infer_config_file(checkpoint_dir)
    infer_output_dir = tmp_path_factory.mktemp("infer")

    # without arguments
    runner = CliRunner()
    result = runner.invoke(main)
    assert result.exit_code == 2
    assert "Missing argument" in result.stderr

    # training.
    args = [
        "train",
        "--config",
        str(train_config_file),
        "--output-dir",
        str(train_output_dir),
    ]
    if with_seed:
        args.extend(["--seed", "42"])

    result = runner.invoke(main, args=args)
    assert result.exit_code == 0

    # inference.
    args = [
        "infer",
        "--config",
        str(infer_config_file),
        "--output-dir",
        str(infer_output_dir),
    ]

    result = runner.invoke(main, args=args)
    assert result.exit_code == 0


def test_validation_failure(tmp_path_factory, write_train_config_file):
    """Passing an invalid configuration file will abort execution with an error."""
    runner = CliRunner()
    config_file = write_train_config_file()
    output_dir = tmp_path_factory.mktemp("train")
    args = ["infer", "--config", str(config_file), "--output-dir", str(output_dir)]
    result = runner.invoke(main, args=args)
    assert result.exit_code == 1
