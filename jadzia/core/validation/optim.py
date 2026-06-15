"""Validation base class for buildable optimizers and schedules."""

# Standard
from collections.abc import Callable
from typing import ClassVar

# Third party
import equinox as eqx
from optax import GradientTransformation, OptState, Schedule

# Dax
from jadzia.core.validation.buildable import BuildableConfig

type OptimizerInstanceType = Callable[..., GradientTransformation]
"""Function that returns an optimizer."""

type ScheduleInstanceType = Callable[..., Schedule]
"""Function that returns a schedule."""


class OptimizerConfig(BuildableConfig[GradientTransformation]):
    """Base class for all validated optimizer configurations."""

    instance_type: ClassVar[OptimizerInstanceType]

    def build_instance(self) -> GradientTransformation:
        """Build an instance based on this configuration."""
        cls = type(self)
        kwargs = self.model_dump_for_instance(build_subclasses=(ScheduleConfig,))
        return cls.instance_type(**kwargs)

    @staticmethod
    def init_state(optimizer: GradientTransformation, model: eqx.Module) -> OptState:
        """
        Initialize optimizer state for a given model.

        Args:
            optimizer: Optimizer used to built initial state.
            model: Module to build optimizer state for.

        Returns:
            New optimizer state.
        """
        return optimizer.init(eqx.filter(model, eqx.is_inexact_array))


class ScheduleConfig(BuildableConfig[Schedule]):
    """Base class for all validated schedule configurations."""

    instance_type: ClassVar[ScheduleInstanceType]


__all__ = [
    "OptimizerConfig",
    "OptimizerInstanceType",
    "ScheduleConfig",
    "ScheduleInstanceType",
]
