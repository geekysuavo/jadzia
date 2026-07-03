"""Jadzia command line interface."""

# Standard
from pathlib import Path
from shutil import copyfile

# Third party
import click
import jax
from pydantic import ValidationError


@click.command()
@click.option(
    "-c",
    "--config",
    "config_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the config file.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to the output directory.",
)
@click.option(
    "-s",
    "--seed",
    type=int,
    required=False,
    default=None,
    help="Random seed.",
)
@click.argument("run_mode", type=click.Choice(("train", "infer")))
@click.pass_context
def main(
    ctx: click.Context,
    run_mode: str,
    config_file: Path,
    output_dir: Path,
    seed: int | None,
):
    """
    Run model training or inference.

    Use `mpirun -n [num] jadzia ...` to run using multiple processes on a single
    host. Otherwise, use `jadzia ...` to run using a single process.
    """
    # Dax
    from jadzia.config.run.inference import InferenceRunConfig
    from jadzia.config.run.training import TrainingRunConfig
    from jadzia.core.validation.context import RuntimeContext

    config_cls = TrainingRunConfig if run_mode == "train" else InferenceRunConfig

    output_dir.mkdir(parents=True, exist_ok=True)
    config_copy = output_dir / "config.yaml"
    if not config_copy.exists():
        copyfile(config_file, config_copy)

    try:
        context = RuntimeContext.from_dir(output_dir, seed=seed)
        config = config_cls.from_yaml(config_file, context=context)
    except ValidationError as err:
        click.echo(f"Failed to validate configuration:\n{err}")
        ctx.abort()

    config.build_instance().run()


if __name__ == "__main__":  # pragma: no cover
    jax.distributed.initialize(cluster_detection_method="mpi4py")
    main()
    jax.distributed.shutdown()
