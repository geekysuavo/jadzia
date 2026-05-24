"""Runtime validation context."""

# Standard
from typing import Self

# Third party
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


class WithRuntimeContext:
    """Mixin class that adds runtime context as a private attribute."""

    _context: RuntimeContext = PrivateAttr()

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
        raise ValueError("runtime context not available")


__all__ = ["RuntimeContext", "WithRuntimeContext", "get_runtime_context"]
