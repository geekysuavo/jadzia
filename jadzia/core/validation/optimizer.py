"""Validation base class for buildable optimizers."""

# Standard
from collections.abc import Callable
from typing import ClassVar

# Third party
from optax import GradientTransformation

# Dax
from jadzia.core.validation.buildable import BuildableConfig

type OptimizerInstanceType = Callable[..., GradientTransformation]


class OptimizerConfig(BuildableConfig[GradientTransformation]):
    """Base class for all validated optimizer configurations."""

    instance_type: ClassVar[OptimizerInstanceType]


__all__ = ["OptimizerConfig", "OptimizerInstanceType"]
