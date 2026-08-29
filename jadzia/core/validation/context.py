"""Runtime validation context."""

# Standard
from pathlib import Path
from typing import Self

# Third party
import jax
from pydantic import DirectoryPath, PrivateAttr, ValidationInfo, model_validator

# Dax
from jadzia.core.validation.base import Config
from jadzia.core.validation.seed import RandomSeed


class RuntimeContext(Config):
    """
    Information about the current execution.

    Attributes:
        name: Execution name.
        path: Working directory.
        seed: Random seed. Generated if unset.
    """

    name: str
    path: DirectoryPath
    seed: RandomSeed

    @property
    def checkpoint_dir(self) -> Path:
        """Directory where checkpoints are stored."""
        return self.path / "checkpoints"

    @property
    def device_mesh(self) -> jax.sharding.Mesh:
        """
        Mesh declaring the available hardware resources.

        Important:
            This should never be called before [jax.distributed.initialize][].
        """
        return jax.make_mesh(axis_sizes=(jax.device_count(),), axis_names=("device",))

    @property
    def data_sharding(self) -> jax.sharding.Sharding:
        """
        Sharding used to fan out batch data onto devices.

        Important:
            This should never be called before [jax.distributed.initialize][].
        """
        return jax.sharding.NamedSharding(self.device_mesh, jax.P("device"))

    @property
    def model_sharding(self) -> jax.sharding.Sharding:
        """
        Sharding used to replicate model weights across all devices.

        Important:
            This should never be called before [jax.distributed.initialize][].
        """
        return jax.sharding.NamedSharding(self.device_mesh, jax.P())

    @classmethod
    def from_dir(cls, directory: Path, seed: int | None = None) -> Self:
        """
        Load context from a possibly uninitialized directory.

        Args:
            directory: Directory to initialize or load context from.
            seed: Optional random seed.

        Returns:
            Runtime context.
        """
        file = directory / "context.yaml"
        if file.exists():
            context = cls.from_yaml(file)
            if seed and seed != context.seed:
                raise ValueError("random seed does not match context")
            else:
                return context

        data: dict[str, int | str | Path] = {
            "path": directory.resolve(),
            "name": directory.name,
        }
        if seed:
            data["seed"] = seed

        context = cls.model_validate(data)
        context.to_yaml(file)
        return context


class WithRuntimeContext:
    """Mixin class that adds runtime context as a private attribute."""

    _context: RuntimeContext = PrivateAttr()

    @property
    def context(self) -> RuntimeContext:
        """Attached runtime context."""
        return self._context

    @model_validator(mode="after")
    def _get_runtime_context(self, info: ValidationInfo) -> Self:
        """Get the runtime context and place it in the private attribute."""
        self._context = get_runtime_context(info)
        return self


def get_runtime_context(info: ValidationInfo) -> RuntimeContext:
    """
    Get the passed runtime context.

    Args:
        info: Validation information provided by `pydantic`.

    Returns:
        Runtime context retrieved from the validation information.

    Raises:
        ValueError: If no runtime context was provided during validation.
    """
    if isinstance(info.context, RuntimeContext):
        return info.context
    else:
        # called within pydantic validator; must raise ValueError
        raise ValueError("runtime context not available")  # ruff: ignore[TRY004]


__all__ = ["RuntimeContext", "WithRuntimeContext", "get_runtime_context"]
