"""Tests of validated optimization classes."""

# Standard
from typing import ClassVar

# Third party
import optax

# Dax
from jadzia.core.validation.optim import (
    OptimizerConfig,
    OptimizerInstanceType,
    ScheduleConfig,
    ScheduleInstanceType,
)


class OptimizerCls(OptimizerConfig):
    instance_type: ClassVar[OptimizerInstanceType] = optax.identity


class ScheduleCls(ScheduleConfig):
    instance_type: ClassVar[ScheduleInstanceType] = optax.constant_schedule
    value: int = 42


def test_optimizer():
    """Optimizers may be built from their configurations."""
    config = OptimizerCls()
    obj = config.build_instance()
    assert isinstance(obj, optax.GradientTransformation)


def test_schedule():
    """Schedules may be built from their configurations."""
    config = ScheduleCls()
    obj = config.build_instance()
    assert obj(0) == 42
