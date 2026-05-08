"""Validation base class for buildable optimizers and schedules."""

# Standard
from collections.abc import Callable
from typing import ClassVar

# Third party
from optax import GradientTransformation, Schedule

# Dax
from jadzia.core.validation.buildable import BuildableConfig

type OptimizerInstanceType = Callable[..., GradientTransformation]
type ScheduleInstanceType = Callable[..., Schedule]


class OptimizerConfig(BuildableConfig[GradientTransformation]):
    """Base class for all validated optimizer configurations."""

    instance_type: ClassVar[OptimizerInstanceType]

    def build_instance(self) -> GradientTransformation:
        """Build an instance based on this configuration."""
        cls = type(self)
        kwargs = self.model_dump_for_instance(build_subclasses=(ScheduleConfig,))
        return cls.instance_type(**kwargs)


class ScheduleConfig(BuildableConfig[Schedule]):
    """Base class for all validated schedule configurations."""

    instance_type: ClassVar[ScheduleInstanceType]


__all__ = [
    "OptimizerConfig",
    "OptimizerInstanceType",
    "ScheduleConfig",
    "ScheduleInstanceType",
]
